"""Training utilities for F5-TTS."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from f5_tts import F5TTS
from f5_tts.config import SamplingConfig, TrainingConfig
from f5_tts.core.cfm import CFM
from f5_tts.model.dataset import DynamicBatchConfig, create_dataloader
from f5_tts.utils.logger import BaseLogger
from f5_tts.utils_infer import cfg_strength as default_cfg_strength
from f5_tts.utils_infer import load_vocoder
from f5_tts.utils_infer import nfe_step as default_nfe_step
from f5_tts.utils_infer import sway_sampling_coef as default_sway_sampling


class Trainer:
    def __init__(
        self,
        model: CFM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        training_cfg: TrainingConfig,
        *,
        logger: Optional[BaseLogger] = None,
        sampling_cfg: Optional[SamplingConfig] = None,
        save_dir: str,
        resume_state: Optional[dict] = None,
        use_fp16: bool = True,
        use_distributed: bool = True,
        ema: Optional[EMA] = None,
        vocoder_path: Optional[str] = None,
        full_config: Optional[dict] = None,
    ):
        self.training_cfg = training_cfg
        self.logger = logger
        self.sampling_cfg = sampling_cfg
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vocoder_path = vocoder_path
        self.full_config = full_config or {}
        self.use_distributed = use_distributed

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.training_cfg.grad_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision="fp16" if use_fp16 else "no",
        )

        self.model = model
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_cfg.learning_rate,
            weight_decay=self.training_cfg.weight_decay,
        )
        if val_dataloader is not None:
            self.model, self.optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, train_dataloader, val_dataloader
            )
        else:
            self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, train_dataloader
            )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.ema = ema
        if self.ema is not None:
            self.ema.to(self.accelerator.device)

        self.global_step = 0
        self.current_epoch = 0
        self.scheduler = self._build_scheduler()
        if resume_state:
            self._restore_state(resume_state)

        self.vocoder = None
        self.ref_mel = None
        self.ref_len = None
        self.ref_text = None
        self._inference_helper = None
        if self.sampling_cfg:
            self._load_sampling_assets()

    @property
    def is_main(self) -> bool:
        return self.accelerator.is_main_process

    def _build_scheduler(self):
        warmup_updates = self.training_cfg.num_warmup_updates
        steps_per_epoch = math.ceil(len(self.train_dataloader) / self.training_cfg.grad_accumulation_steps)
        total_steps = steps_per_epoch * self.training_cfg.epochs
        decay_steps = max(total_steps - warmup_updates, 1)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates or 1
        )
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=decay_steps)
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_updates],
        )
        return scheduler

    def _checkpoint_dir(self, step: int) -> Path:
        ckpt_dir = self.save_dir / f"checkpoint_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def save_checkpoint(self, step: int):
        self.accelerator.wait_for_everyone()
        if not self.is_main:
            return

        ckpt_dir = self._checkpoint_dir(step)
        model_state = self.accelerator.unwrap_model(self.model).state_dict()
        torch.save(model_state, ckpt_dir / "model.pth")

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": step,
            "epoch": self.current_epoch,
        }
        if self.ema is not None:
            state["ema_model_state_dict"] = self.ema.state_dict()

        torch.save(state, ckpt_dir / "trainer_state.pth")

    def _restore_state(self, state: dict):
        model_state = state.get("model_state_dict")
        if model_state:
            self.accelerator.unwrap_model(self.model).load_state_dict(model_state)
        elif "ema_model_state_dict" in state:
            model_state = {
                k.replace("ema_model.", ""): v
                for k, v in state["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(model_state)

        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if self.ema is not None and "ema_model_state_dict" in state:
            self.ema.load_state_dict(state["ema_model_state_dict"])

        self.global_step = state.get("global_step", 0)
        self.current_epoch = state.get("epoch", 0)

    def _load_sampling_assets(self):
        if not self.sampling_cfg:
            return None

        if self.ref_mel is not None and self.ref_len is not None:
            return self.ref_mel, self.ref_len

        if self.vocoder is None and self.vocoder_path:
            self.vocoder = load_vocoder(Path(self.vocoder_path), self.accelerator.device)
        elif self.vocoder is None:
            raise RuntimeError("Vocoder not loaded; please provide paths.vocoder_path in config.")

        ref_audio_path = Path(self.sampling_cfg.ref_audio_path)
        ref_audio, ref_sr = torchaudio.load(str(ref_audio_path))
        if ref_audio.size(0) > 1:
            ref_audio = ref_audio.mean(dim=0, keepdim=True)

        mel_spec = self.model.mel_spec(ref_audio.to(self.accelerator.device))
        mel_spec = mel_spec.squeeze(0)
        ref_mel = mel_spec.permute(1, 0).unsqueeze(0)
        ref_len = ref_mel.shape[1]
        self.ref_mel = ref_mel
        self.ref_len = ref_len
        self.ref_text = self.sampling_cfg.ref_text
        self._build_inference_helper()
        return self.ref_mel, self.ref_len

    def _build_inference_helper(self):
        if self._inference_helper is None and self.sampling_cfg:
            helper = F5TTS.__new__(F5TTS)
            helper.target_sample_rate = self.model.mel_spec.target_sample_rate
            helper.device = self.accelerator.device
            helper.ode_method = "euler"
            helper.use_ema = False
            helper.vocoder = self.vocoder
            helper.ema_model = self.accelerator.unwrap_model(self.model)
            self._inference_helper = helper

        return self._inference_helper

    def _log_hyperparams_once(self):
        if self.logger:
            self.logger.log_hyperparameters(self.full_config)

    def _run_sampling(self, checkpoint_dir: Path, step: int):
        if not self.sampling_cfg or not self.is_main:
            return

        previous_mode = self.model.training
        self.model.eval()

        ref_mel, ref_len = self._load_sampling_assets()
        if ref_mel is None or ref_len is None:
            return

        inference_helper = self._build_inference_helper()
        if inference_helper is None:
            raise RuntimeError("Inference helper could not be initialized for sampling.")

        sampling_steps = self.sampling_cfg.nfe_step or default_nfe_step
        cfg_strength = self.sampling_cfg.cfg_strength or default_cfg_strength
        sway_coef = self.sampling_cfg.sway_sampling_coef or default_sway_sampling
        ref_duration_seconds = (
            ref_len
            * self.model.mel_spec.hop_length
            / self.model.mel_spec.target_sample_rate
            * self.sampling_cfg.duration_multiplier
        )

        samples_dir = checkpoint_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        metadata_rows = []
        with open(self.sampling_cfg.sample_texts_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            rows = list(reader)

        if rows and rows[0][:2] == ["id", "text"]:
            rows = rows[1:]

        for idx, row in enumerate(rows):
            if len(row) < 2:
                continue
            sample_id, text = row[0], row[1]
            gen_audio, sr, _ = inference_helper.infer_from_mel(
                ref_mel,
                self.ref_text,
                text,
                show_info=lambda *_: None,
                progress=None,
                nfe_step=sampling_steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_coef,
                fix_duration=ref_duration_seconds,
                device=self.accelerator.device,
            )

            filename = f"sample_{sample_id}.wav"
            torchaudio.save(
                str(samples_dir / filename),
                torch.tensor(gen_audio).unsqueeze(0),
                sr,
            )
            metadata_rows.append({"id": sample_id, "text": text, "filename": filename})

        if previous_mode:
            self.model.train()

        metadata_path = samples_dir / "samples_metadata.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "filename"], delimiter="|")
            writer.writeheader()
            writer.writerows(metadata_rows)

        if self.logger:
            rel_path = os.path.relpath(samples_dir, start=self.save_dir)
            self.logger.save_metrics("samples", "path", rel_path, step=step)

    def train(self):
        self._log_hyperparams_once()

        self.scheduler = self.accelerator.prepare(self.scheduler)
        progress_bar_total = math.ceil(len(self.train_dataloader) / self.training_cfg.grad_accumulation_steps)

        for epoch in range(self.current_epoch, self.training_cfg.epochs):
            self.model.train()
            self.accelerator.print(f"Starting epoch {epoch + 1}/{self.training_cfg.epochs}")
            progress_bar = tqdm(
                range(progress_bar_total),
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch + 1}",
            )

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    loss, cond, pred = self.model(mel_spec, text=text_inputs, lens=mel_lengths)
                    self.accelerator.backward(loss)

                    if (
                        self.training_cfg.grad_clip_norm
                        and self.training_cfg.grad_clip_norm > 0
                        and self.accelerator.sync_gradients
                    ):
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_cfg.grad_clip_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.ema is not None and self.is_main:
                        self.ema.update()
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), step=self.global_step)

                if self.logger:
                    self.logger.save_metrics(
                        "train",
                        ["loss", "lr"],
                        [loss.item(), self.scheduler.get_last_lr()[0]],
                        step=self.global_step,
                    )

                if self.global_step % self.training_cfg.save_every_steps == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(self.global_step)
                    self._run_sampling(self._checkpoint_dir(self.global_step), self.global_step)

            self.current_epoch = epoch + 1

            if self.val_dataloader is not None:
                self.model.eval()
                val_losses = []
                for batch in self.val_dataloader:
                    with torch.no_grad():
                        text_inputs = batch["text"]
                        mel_spec = batch["mel"].permute(0, 2, 1)
                        mel_lengths = batch["mel_lengths"]
                        val_loss, _, _ = self.model(mel_spec, text=text_inputs, lens=mel_lengths)
                        gathered = self.accelerator.gather(val_loss.detach())
                        val_losses.append(gathered.mean().item())

                if val_losses and self.logger:
                    mean_val_loss = sum(val_losses) / len(val_losses)
                    self.logger.save_metrics("val", "loss", mean_val_loss, step=self.global_step)
                self.model.train()

        self.save_checkpoint(self.global_step)
        if self.logger:
            self.logger.stop()
        self.accelerator.end_training()


def build_dataloaders(
    train_dataset,
    val_dataset,
    data_cfg,
    batch_seed: Optional[int] = None,
):
    batch_config = DynamicBatchConfig(
        max_tokens_per_batch=data_cfg.max_tokens_per_batch,
        max_samples_per_batch=data_cfg.max_samples_per_batch,
        manifest_delimiter=data_cfg.manifest_delimiter,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_config=batch_config,
        num_workers=data_cfg.num_workers,
        seed=batch_seed,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            batch_config=batch_config,
            num_workers=data_cfg.num_workers,
            seed=batch_seed,
        )

    return train_loader, val_loader


__all__ = ["Trainer", "build_dataloaders"]
