"""Dataset and batching utilities for training."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler
import torchaudio
from datasets import Dataset as HFDatasetType
from datasets import Dataset as InMemoryDataset
from datasets import load_dataset as load_hf_dataset
from datasets import load_from_disk
from tqdm import tqdm

from f5_tts.core.modules import MelSpec
from f5_tts.core.utils import default


class ManifestDataset(Dataset):
    """Simple dataset backed by a manifest file.

    The manifest is expected to contain at least ``audio_path`` and ``text`` columns,
    with an optional ``duration`` column in seconds. Delimiter is auto-detected.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        target_sample_rate: int = 24_000,
        hop_length: int = 256,
        n_mel_channels: int = 100,
        n_fft: int = 1024,
        win_length: int = 1024,
        mel_spec_type: str = "vocos",
        max_audio_length_seconds: float | None = None,
        manifest_delimiter: str = "|",
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.max_audio_length_seconds = max_audio_length_seconds

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

        self.data = self._load_manifest(delimiter=manifest_delimiter)
        self._frame_lengths = [self._compute_frame_length(row) for row in self.data]

    def _detect_delimiter(self) -> str:
        sample = self.manifest_path.read_text(encoding="utf-8", errors="ignore")[:1024]
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return "|"

    def _load_manifest(self, delimiter: str) -> list[dict]:
        delimiter = delimiter or self._detect_delimiter()
        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return [row for row in reader]

    def _compute_frame_length(self, row: dict) -> int:
        if row.get("duration"):
            return int(float(row["duration"]) * self.target_sample_rate / self.hop_length)

        audio_path = Path(row["audio_path"]).expanduser()
        try:
            info = torchaudio.info(str(audio_path))
            num_frames = info.num_frames
            sample_rate = info.sample_rate
        except Exception:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            num_frames = waveform.size(-1)

        return int(num_frames * self.target_sample_rate / sample_rate / self.hop_length)

    def get_frame_len(self, index: int) -> int:
        return self._frame_lengths[index]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]
        audio_path = Path(row["audio_path"]).expanduser()
        text = row["text"]

        audio, source_sample_rate = torchaudio.load(str(audio_path))
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if source_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
            audio = resampler(audio)

        duration_seconds = audio.size(-1) / self.target_sample_rate
        if self.max_audio_length_seconds and duration_seconds > self.max_audio_length_seconds:
            raise ValueError(
                f"Audio at {audio_path} is longer than allowed max_audio_length_seconds={self.max_audio_length_seconds}"
            )

        mel_spec = self.mel_spectrogram(audio)
        mel_spec = mel_spec.squeeze(0)

        return {"mel_spec": mel_spec, "text": text}


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: HFDatasetType,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]

        audio_tensor = torch.from_numpy(audio).float()
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)
        mel_spec = self.mel_spectrogram(audio_tensor)
        mel_spec = mel_spec.squeeze(0)

        text = row["text"]
        return {"mel_spec": mel_spec, "text": text}


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: InMemoryDataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if self.durations is not None:
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        text = row["text"]
        duration = row.get("duration")

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)

        return {"mel_spec": mel_spec, "text": text, "duration": duration}


class DynamicBatchSampler(Sampler[list[int]]):
    """Dynamic sampler that forms batches based on frame budget."""

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) if item["mel_spec"].ndim == 3 else item["mel_spec"] for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)
    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(mel=mel_specs, mel_lengths=mel_lengths, text=text, text_lengths=text_lengths)


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except Exception:
                train_dataset = InMemoryDataset.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = InMemoryDataset.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except Exception:
            train_dataset = InMemoryDataset.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_hf_dataset(
                f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))
            ),
        )

    return train_dataset


@dataclass
class DynamicBatchConfig:
    max_tokens_per_batch: int
    max_samples_per_batch: int
    manifest_delimiter: str = "|"


def create_dataloader(
    dataset: Dataset,
    *,
    batch_config: DynamicBatchConfig,
    num_workers: int,
    seed: Optional[int] = None,
) -> DataLoader:
    sampler = SequentialSampler(dataset)
    batch_sampler = DynamicBatchSampler(
        sampler,
        batch_config.max_tokens_per_batch,
        max_samples=batch_config.max_samples_per_batch,
        random_seed=seed,
        drop_residual=False,
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        batch_sampler=batch_sampler,
    )

    return dataloader


__all__ = [
    "ManifestDataset",
    "DynamicBatchSampler",
    "collate_fn",
    "create_dataloader",
    "DynamicBatchConfig",
]
