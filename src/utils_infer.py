# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import re
import tempfile
from pathlib import Path
from typing import Literal 
from collections import defaultdict
from importlib.resources import files
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import matplotlib.pylab as plt
matplotlib.use("Agg")

import torch
from torch import nn

from transformers import pipeline
from huggingface_hub import hf_hub_download

import tqdm
import torchaudio
import numpy as np
from pydub import AudioSegment, silence

from vocos import Vocos
from vocos.feature_extractors import EncodecFeatures

from core.cfm import CFM
from core.dit import DiT
from core.utils import get_tokenizer


device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------

def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


################
# LOAD VOCODER #
################


def load_vocoder(
        local_path: Path, 
        device=device, 
    ) -> Vocos:

    config_path = local_path / "config.yaml"
    model_path = local_path/ "pytorch_model.bin"

    vocoder = Vocos.from_hparams(str(config_path))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }

        state_dict.update(encodec_parameters)

    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)
    
    return vocoder


#######################################
# LOAD MODEL CHECKPOINT FOR INFERENCE #
#######################################


def load_checkpoint(
        model: nn.Module, 
        ckpt_path, 
        device: str, 
        use_ema=True
    ):

    dtype = (
        torch.float16
        if "cuda" in device
        and torch.cuda.get_device_properties(device).major >= 7
        and not torch.cuda.get_device_name().endswith("[ZLUDA]")
        else torch.float32
    )

    model = model.to(dtype)

    ckpt_type = str(ckpt_path).split(".")[-1]

    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device=device)

    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}

        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}

        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cfg: dict,
    ckpt_path: Path,
    vocab_file: Path,
    ode_method: Literal["euler", "midpoint"],
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mel_channels: int,
    target_sample_rate: int,
    *,
    use_ema=True,
    device=device,
):

    vocab_char_map, vocab_size = get_tokenizer(vocab_file)

    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),

        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
        ),

        odeint_kwargs=dict(
            method=ode_method,
        ),

        vocab_char_map=vocab_char_map,

    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(
        ref_audio_orig: Path, 
        ref_text: str,
        show_info=print
    ) -> tuple[str, str]:

    with tempfile.NamedTemporaryFile(suffix=".wav", delete_on_close=False) as f: 
        temp_path = f.name

    aseg = AudioSegment.from_file(ref_audio_orig)

    # 1. try to find long silence for clipping
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)

    for non_silent_seg in non_silent_segs:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
            show_info("Audio is over 12s, clipping short. (1)")
            break

        non_silent_wave += non_silent_seg

    # 2. try to find short silence for clipping if 1. failed
    if len(non_silent_wave) > 12000:
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
        )

        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                show_info("Audio is over 12s, clipping short. (2)")
                break

            non_silent_wave += non_silent_seg

    aseg = non_silent_wave

    # 3. if no proper silence found for clipping
    if len(aseg) > 12000:
        aseg = aseg[:12000]
        show_info("Audio is over 12s, clipping short. (3)")

    aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    aseg.export(temp_path, format="wav")
    ref_audio = temp_path

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]

def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    batch_process_type="brand_new", # "classic"
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")

    batch_process = infer_batch_process if batch_process_type == "classic" else brand_new_infer_batch_process

    return next(
        batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )


# infer batches
def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
    
    audio, sr = ref_audio

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))

    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text]

        ref_audio_len = audio.shape[-1] // hop_length

        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)

        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            
            generated_wave = vocoder.decode(generated)

            if streaming:
                del generated

            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate

            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk

    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)

            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


def brand_new_infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
    """
    Микро-батчевый инференс для F5-TTS:
      - батчим тексты схожей длины (уменьшаем паддинг);
      - один вызов sample на микро-батч;
      - вырезаем continuation (после промпта) и декодим вокодером;
      - оффлайн: склейка с кросс-фейдом; стриминг: почанковая отдача после каждого микро-батча.

    Требуются глобали: target_sample_rate, hop_length.
    """

    # ---------- настройки микро-батча (подправь под свою VRAM) ----------
    MAX_BS = 4                 # максимум элементов в одном микро-батче
    BUCKET_STEP_FRAMES = 256   # шаг корзины по мел-кадрам (чем меньше — тем равнее длины в батче)

    # ---------- helpers ----------
    def _tqdm(it):
        if progress is None:
            return it
        
        return progress.tqdm(it) if hasattr(progress, "tqdm") else it

    # Если последний символ ASCII-однобайтный — добавим пробел (стабильность стыка ref+gen)
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    # ---------- 0) препроцессинг референса ----------
    audio, sr = ref_audio  # audio: torch.Tensor [C, T] или [1, T]
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)  # -> [1, T]

    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)  # -> [1, T]

    # RMS нормализация (только подкачиваем тихие референсы)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    boosted = False

    if rms < target_rms and rms > 0:
        audio = audio * (target_rms / rms)
        boosted = True

    # ресемплинг к таргет SR
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    audio = audio.to(device)
    ref_frames = audio.shape[-1] // hop_length  # длина промпта в мел-кадрах

    # попробуем один раз посчитать мел, чтобы не гонять self.mel_spec внутри sample по B раз
    # только если у модели есть публичный mel_spec (иначе передадим сырое wav и пусть сделает сама)
    precomputed_mel = None
    if hasattr(model_obj, "mel_spec"):
        try:
            mel = model_obj.mel_spec(audio)          # [1, d, n]
            precomputed_mel = mel.permute(0, 2, 1)   # -> [1, n, d]
            precomputed_mel = precomputed_mel.to(next(model_obj.parameters()).dtype)

        except Exception:
            precomputed_mel = None  # не страшно, перейдём на аудио-ветку внутри sample

    # ---------- 1) оценим длительности по каждой реплике ----------
    def estimate_frames(gen_text: str) -> int:
        # локальная «подтормозка» для очень коротких реплик (чтобы не получалось «лепета»)
        local_speed = 0.3 if len(gen_text.encode("utf-8")) < 10 else speed

        if fix_duration is not None:
            dur = int(fix_duration * target_sample_rate / hop_length)

        else:
            rtl = max(len(ref_text.encode("utf-8")), 1)
            gtl = len(gen_text.encode("utf-8"))
            dur = ref_frames + int(ref_frames / rtl * gtl / max(local_speed, 1e-3))

        # минимально нужно хотя бы покрыть промпт + 1 кадр
        return max(dur, ref_frames + 1)

    items = []  # (idx, gen_text, est_dur_frames)
    for i, t in enumerate(gen_text_batches):
        items.append((i, t, estimate_frames(t)))

    if not items:
        # пустой случай — как и раньше, вернём "ничего"
        yield None, target_sample_rate, None
        return

    # ---------- 2) бакетизация по длине ----------
    buckets = defaultdict(list)
    for idx, t, est in items:
        key = int(round(est / BUCKET_STEP_FRAMES))
        buckets[key].append((idx, t, est))

    # упорядочим обход корзин по минимальному исходному индексу внутри корзины
    bucket_order = sorted(buckets.keys(), key=lambda k: min(i for i, _, _ in buckets[k]))

    # подготовим накопители результата для оффлайн и план для стриминга
    total = len(items)
    waves_by_index = [None] * total
    mels_by_index = [None] * total

    # Для стриминга: будем по возможности отдавать в исходном порядке
    next_to_yield = 0

    # ---------- 3) прогон по корзинам с микро-батчами ----------
    for bkey in bucket_order:
        bucket = buckets[bkey]
        # нарежем на куски по MAX_BS
        chunks = [bucket[i : i + MAX_BS] for i in range(0, len(bucket), MAX_BS)]

        for chunk in _tqdm(chunks):
            # --- 3.1 подготовка батча ---
            idxs, texts, ests = zip(*chunk)
            B = len(chunk)

            # финальные целевые длительности именно для sample (она всё равно проверит min_need/lens)
            durations = []
            speeds_local = []

            for t in texts:
                speeds_local.append(0.3 if len(t.encode("utf-8")) < 10 else speed)
                durations.append(estimate_frames(t))

            duration_tensor = torch.tensor(durations, device=device, dtype=torch.long)
            lens_tensor = torch.full((B,), ref_frames, device=device, dtype=torch.long)
            text_list = [ref_text + t for t in texts]

            # cond: либо один раз посчитанный мел расширяем по B, либо дублируем аудио
            if precomputed_mel is not None:
                cond_batch = precomputed_mel.expand(B, -1, -1)  # [B, n, d]

            else:
                cond_batch = audio.expand(B, -1)                 # [B, T] сырое аудио (ветка в sample сама сделает мел)

            # --- 3.2 вызов модели (mel без вокодера) ---
            with torch.no_grad():
                out, _traj = model_obj.sample(
                    cond=cond_batch,
                    text=text_list,
                    duration=duration_tensor,
                    lens=lens_tensor,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    vocoder=None,           # получим мел; декодим отдельно
                )
                # out: [B, n_max, d] с пришитым слева промптом точно как во входе
                del _traj

            # --- 3.3 вырезаем continuation для каждого элемента и декодим ---
            for k in range(B):
                dur_k = int(duration_tensor[k].item())
                # мел-кусок только после промпта: [n_mels, T_cont]
                mel_k = out[k, lens_tensor[k] : dur_k, :].permute(1, 0).contiguous()  # -> [d, T]

                p = next(vocoder.parameters())
                mel_k = mel_k.to(device=p.device, dtype=p.dtype)

                # декодер ожидает [B, n_mels, T]
                wav_k = vocoder.decode(mel_k.unsqueeze(0))  # -> [1, n_samples] (torch)

                # вернуть громкость к RMS исходника, если мы подкачивали
                if boosted and target_rms > 0:
                    wav_k = wav_k * (rms / target_rms)

                wav_np = wav_k.squeeze(0).detach().cpu().numpy()
                mel_np = mel_k.detach().cpu().numpy()  # [n_mels, T_cont]

                waves_by_index[idxs[k]] = wav_np
                mels_by_index[idxs[k]] = mel_np

            # --- 3.4 стриминг: после микро-батча отдаём готовые по порядку индексов ---
            if streaming:
                while next_to_yield < total and waves_by_index[next_to_yield] is not None:
                    w = waves_by_index[next_to_yield]

                    # отдаём чанками
                    for j in range(0, len(w), chunk_size):
                        yield w[j : j + chunk_size], target_sample_rate

                    next_to_yield += 1

    # ---------- 4) оффлайн-склейка / комбинированный мел ----------
    if not streaming:
        generated_waves = [w for w in waves_by_index if w is not None]
        spectrograms = [m for m in mels_by_index if m is not None]

        if not generated_waves:
            yield None, target_sample_rate, None
            return

        if cross_fade_duration <= 0:
            final_wave = np.concatenate(generated_waves)
        else:
            final_wave = generated_waves[0]
            for i in range(1, len(generated_waves)):
                prev_wave = final_wave
                next_wave = generated_waves[i]

                cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
                if cross_fade_samples <= 0:
                    final_wave = np.concatenate([prev_wave, next_wave])
                    continue

                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]

                # линейный кросс-фейд (можно заменить на equal-power при желании)
                fade_out = np.linspace(1.0, 0.0, cross_fade_samples, dtype=prev_wave.dtype)
                fade_in  = np.linspace(0.0, 1.0, cross_fade_samples, dtype=next_wave.dtype)

                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                final_wave = np.concatenate(
                    [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                )

        combined_spectrogram = np.concatenate(spectrograms, axis=1)  # по времени

        yield final_wave, target_sample_rate, combined_spectrogram



# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
