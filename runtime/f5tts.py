# -*- coding: utf-8 -*-
# Minimal, integration-friendly TensorRT runtime for F5-TTS.
# Loads:
#   - TensorRT-LLM engine dir (config.json + rank{rank}.engine)
#   - Vocos vocoder TensorRT plan (optional)
# Exposes:
#   - class F5TTS_TRT with .infer(...) API mirroring your src/f5tts.py

import math
import os
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
from tensorrt_llm.runtime.session import Session, TensorInfo
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch

from ..src.utils_infer import (
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    load_vocoder as load_vocoder_fallback,  # fallback if no TRT vocoder
)

from ..src.core.dit import TextEmbedding

# ----------------- small local helpers (strictly what's needed) -----------------

def _remove_tensor_padding(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Pack variable-length sequences from [B, T, ...] -> [sum(T_i), ...]."""
    assert x.dim() >= 2 and lengths is not None
    out = []
    for i in range(x.shape[0]):
        t = int(lengths[i])
        out.append(x[i, :t])
    return torch.cat(out, dim=0).contiguous()



def _load_text_embed_weights_from_ckpt(ckpt_path: Union[str, Path], use_ema: bool = True):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt["ema_model_state_dict"] if use_ema else ckpt["model_state_dict"]
    # map "transformer.text_embed.*" -> "text_embed.*"
    text_state = {}
    for k, v in state.items():
        if "text_embed" in k:
            text_state[k.replace("transformer.text_embed.", "")] = v
    return text_state


# ----------------- TRT Vocoder (plan) -----------------

class VocosTensorRT:
    """TensorRT wrapper for Vocos (mel->waveform)."""
    def __init__(self, engine_path: Union[str, Path], stream: Optional[torch.cuda.Stream] = None):
        self.engine_path = str(engine_path)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

        with open(self.engine_path, "rb") as f:
            engine_buf = f.read()

        self.session = Session.from_serialized_engine(engine_buf)
        self.stream = stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream

    def decode(self, mels_chw: torch.Tensor) -> torch.Tensor:
        # expects float32 CHW (B, C=100, T)
        assert mels_chw.dtype in (torch.float16, torch.float32)
        mels_chw = mels_chw.to(dtype=torch.float32).contiguous()
        outs_info = self.session.infer_shapes([TensorInfo("mel", trt.DataType.FLOAT, tuple(mels_chw.shape))])
        outputs = {t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda") for t in outs_info}
        ok = self.session.run({"mel": mels_chw}, outputs, self.stream)

        if not ok:
            raise RuntimeError("VocosTensorRT: runtime execution failed")
        
        return outputs["waveform"]  # (B, N)


# ----------------- TRT-LLM F5-TTS core -----------------

class _F5TRTCore:
    """
    Low-level TensorRT-LLM runner for the DiT UNet.
    Inputs expected by the engine: noise, cond, time, rope_cos, rope_sin, input_lengths
    Output: denoised
    """

    def __init__(
        self,
        tllm_model_dir: Union[str, Path],
        text_vocab_size: int,
        *,
        model_ckpt_for_text_embed: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        max_mel_len: int = 4096,
        text_dim: int = 512,
        text_conv_layers: int = 4,
        n_mel_channels: int = 100,
        nfe_steps_default: int = 16,
        target_sample_rate: int = 24000,
        dtype: str = "float16",
    ):
        tllm_model_dir = Path(tllm_model_dir)
        config_path = tllm_model_dir / "config.json"

        with open(config_path) as f:
            config = json.load(f)

        # device/stream
        self.device = torch.device(device) if device is not None else torch.device("cuda")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        # engine
        rank = 0
        engine_file = tllm_model_dir / f"rank{rank}.engine"

        with open(engine_file, "rb") as f:
            engine_buf = f.read()

        self.session = Session.from_serialized_engine(engine_buf)

        # precision & model dims
        self.dtype = config["pretrained_config"]["dtype"] if "pretrained_config" in config else dtype
        self.n_mel_channels = n_mel_channels
        self.nfe_steps_default = nfe_steps_default

        # text embedding kept in PyTorch (small & cheap)
        self.max_mel_len = max_mel_len

        self.text_embedding = TextEmbedding(
            text_num_embeds=text_vocab_size,
            text_dim=text_dim,
            conv_layers=text_conv_layers,
            precompute_max_pos=self.max_mel_len,
        ).to(self.device)

        self.text_embedding.load_state_dict(_load_text_embed_weights_from_ckpt(model_ckpt_for_text_embed), strict=True)

        # rotary cache
        head_dim = 64
        base = 10000.0 * (1.0 ** (head_dim / (head_dim - 2)))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(self.max_mel_len, dtype=torch.float32), inv_freq)
        self.rope_cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(0).half()
        self.rope_sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(0).half()

        # default time embedding cache for nfe_steps_default
        self._cache_time(self.nfe_steps_default)

        self.target_sample_rate = target_sample_rate

        # pre-allocate outputs dict each call
        self._outputs = {}

    def _cache_time(self, nfe_steps: int):
        self.nfe_steps = nfe_steps
        t = torch.linspace(0, 1, nfe_steps + 1, dtype=torch.float32)
        time_step = t + (-1.0) * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
        delta_t = torch.diff(time_step)
        tmp_dim = 256
        time_expand = torch.zeros((1, nfe_steps, tmp_dim), dtype=torch.float32)
        half_dim = tmp_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)

        for i in range(nfe_steps):
            emb = time_step[i] * emb_factor
            time_expand[:, i, :] = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_expand = time_expand.to(self.device)  # (1, nfe_steps, 256)
        self.delta_t = torch.cat((delta_t, delta_t), dim=0).contiguous().to(self.device)  # (2 * nfe_steps,)

    # ---- PyTorch-compatible API (to match your utils_infer expectations) ----
    @torch.inference_mode()
    def sample(
        self,
        *,
        cond: torch.Tensor,               # (B, T, 100) mels (ref)
        text: torch.Tensor,               # (B, T) indices with -1 as padding
        duration: Union[List[int], torch.Tensor],   # target mel lengths per sample
        lens: Optional[torch.Tensor] = None,        # ref lens (unused here but kept for sig-compat)
        steps: Optional[int] = None,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,          # unused (kept for compatibility)
        remove_input_padding: bool = False,
        use_perf: bool = False,
    ) -> Tuple[List[torch.Tensor], None]:
        """
        Returns list of B tensors [T_i, 100] (to match your downstream post-processing).
        """
        if steps is None:
            steps = self.nfe_steps_default
        if steps != self.nfe_steps:
            self._cache_time(steps)

        device = self.device
        B, T, C = cond.shape
        assert C == self.n_mel_channels == 100

        # build paired batch (cond + drop-cond)
        # text indices: -1 padding -> +1 shift so padding becomes 0
        text_plus1 = (text.clone() + 1).clamp_min(0).to(device)  # [-1, V-1] -> [0..V]
        text_emb = self.text_embedding(text_plus1)               # (B, T, 512)
        text_drop = text_emb[-1:].repeat(B, 1, 1) if text_emb.shape[0] > 0 else text_emb

        cat_cond = torch.cat([cond.to(device), torch.zeros_like(cond)], dim=0)                    # (2B, T, 100)
        cat_text = torch.cat([text_emb, text_drop], dim=0)                                        # (2B, T, 512)
        cat_mel_text = torch.cat([cat_cond, cat_text], dim=-1).contiguous()                       # (2B, T, 612)

        noise = torch.randn_like(cond, device=device)
        noise_pair = torch.cat([noise, noise], dim=0).contiguous()                                # (2B, T, 100)

        rope_cos = self.rope_cos[:, :T, :].float().repeat(B * 2, 1, 1)
        rope_sin = self.rope_sin[:, :T, :].float().repeat(B * 2, 1, 1)

        if isinstance(duration, list):
            duration = torch.tensor(duration, dtype=torch.int32, device=device)
        else:
            duration = duration.to(torch.int32).to(device)

        input_lengths = torch.cat([duration, duration], dim=0).contiguous()

        # perf ranges
        if use_perf:
            torch.cuda.nvtx.range_push("F5TRTCore.sample")

        # iterative flow-matching steps, but each step is one stateless TRT pass
        noise_half = noise
        torch_dtype = str_dtype_to_torch(self.dtype)

        for i in range(self.nfe_steps):
            # time emb for step i  -> (2B, 256)
            cur_time = self.time_expand[:, i].to(torch_dtype).repeat(2 * B, 1).contiguous()
            inputs = {
                "noise": torch.cat([noise_half, noise_half], dim=0).to(torch_dtype),
                "cond": cat_mel_text.to(torch_dtype),
                "time": cur_time,
                "rope_cos": rope_cos.to(torch_dtype),
                "rope_sin": rope_sin.to(torch_dtype),
                "input_lengths": input_lengths.to(torch.int32),
            }

            # set output buffer per shape
            out_infos = self.session.infer_shapes([
                TensorInfo("noise", trt_dtype=trt.DataType.__dict__[self.dtype.upper()] if self.dtype != "bfloat16" else trt.DataType.BF16, shape=tuple(inputs["noise"].shape)),
                TensorInfo("cond", trt.DataType.FLOAT, tuple(inputs["cond"].shape)),
                TensorInfo("time", trt.DataType.FLOAT, tuple(inputs["time"].shape)),
                TensorInfo("rope_cos", trt.DataType.FLOAT, tuple(inputs["rope_cos"].shape)),
                TensorInfo("rope_sin", trt.DataType.FLOAT, tuple(inputs["rope_sin"].shape)),
                TensorInfo("input_lengths", trt.DataType.INT32, tuple(inputs["input_lengths"].shape)),
            ])
            # allocate outputs lazily (once)
            out_name = "denoised"
            if out_name not in self._outputs or tuple(self._outputs[out_name].shape) != tuple(out_infos[0].shape):
                self._outputs[out_name] = torch.empty(
                    tuple(out_infos[0].shape), dtype=trt_dtype_to_torch(out_infos[0].dtype), device=device
                )

            ok = self.session.run(inputs, self._outputs, torch.cuda.current_stream().cuda_stream)
            if not ok:
                raise RuntimeError("F5TRTCore: TRT-LLM execution failed")

            # classifier-free guidance
            half = B
            pred_cond = self._outputs[out_name][:half]
            pred_uncond = self._outputs[out_name][half:]
            guidance = pred_cond + (pred_cond - pred_uncond) * cfg_strength

            t_scale = self.delta_t[i].to(torch_dtype)
            noise_half = noise_half + guidance * t_scale

        if use_perf:
            torch.cuda.nvtx.range_pop()

        # return a list of per-utterance mel chunks [T_i, 100]
        out_list: List[torch.Tensor] = []
        for i in range(B):
            tgt_len = int(duration[i].item())
            out_list.append(noise_half[i, :tgt_len].detach())
        return out_list, None


# ----------------- High-level wrapper matching your src/f5tts.py -----------------

class F5TTS_TRT:
    """
    Drop-in replacement for your `src/f5tts.py:F5TTS`, but running DiT+Vocoder on TensorRT.
    Keeps the same .infer(...) signature and reuses your utils for pre/post-processing.
    """

    def __init__(
        self,
        *,
        config_path: Path,                  # kept for parity; used to read mel params if needed
        vocab_file: Path,                   # to get vocab size
        ckpt_file: Path,                    # PyTorch ckpt (only for text-embedding weights)
        tllm_engine_dir: Path,              # dir with TensorRT-LLM engine (config.json + rank0.engine)
        vocoder_trt_plan: Optional[Path] = None,  # plan file for Vocos (optional; fallback to PyTorch Vocos)
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cuda")

        # vocab size from your text vocab
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_size = sum(1 for _ in f)

        # TRT core
        self.core = _F5TRTCore(
            tllm_model_dir=tllm_engine_dir,
            text_vocab_size=vocab_size,
            model_ckpt_for_text_embed=ckpt_file,
            device=self.device,
        )

        # vocoder
        if vocoder_trt_plan is not None and Path(vocoder_trt_plan).exists():
            self.vocoder = VocosTensorRT(vocoder_trt_plan)
            self._vocoder_is_trt = True

        else:
            # fallback to your existing loader (HF / local torch model)
            self.vocoder = load_vocoder_fallback(vocoder_local_path=None, device=str(self.device))
            self._vocoder_is_trt = False

        self.target_sample_rate = 24000  # consistent with training/config

    # ---- helpers copying your original public API ----

    def export_wav(self, wav: torch.Tensor, file_wave: Union[str, Path], remove_silence=False):
        import soundfile as sf
        sf.write(str(file_wave), wav, self.target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(str(file_wave))

    def export_spectrogram(self, spec: torch.Tensor, file_spec: Union[str, Path]):
        save_spectrogram(spec, str(file_spec))

    @torch.inference_mode()
    def infer(
        self,
        ref_file: Union[str, Path],
        ref_text: str,
        gen_text: str,
        *,
        show_info=print,
        progress=None,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2.0,
        nfe_step=16,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave: Optional[Union[str, Path]] = None,
        file_spec: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        A light reimplementation of the minimal path from your utils:
        - use preprocess_ref_audio_text(...) to normalize inputs
        - build mel for ref, tokenize text via vocab.txt externally (kept in utils)
        - run TRT DiT -> mel
        - run TRT (or Torch) Vocos -> waveform
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 1) preprocess (reuses your utils)
        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        # 2) build reference mel and text indices using your utils functions
        #    We use your src/utils_infer functions for text->indices to keep parity.
        from .utils_infer import (
            text_to_vocab_idx_list,
            build_mel_from_audio,  # should produce [1, T, 100] float32 log-mel
        )

        # build ref mel
        ref_mel: torch.Tensor = build_mel_from_audio(ref_file, device=str(self.device))  # (1, T, 100)
        ref_len = ref_mel.shape[1]

        # concat prompt+target text per your pipeline
        full_text = ref_text + gen_text
        text_idx, target_len = text_to_vocab_idx_list(full_text, ref_len, device=str(self.device))
        # text_idx: (1, T) with -1 padding; target_len: int (estimated T_out)

        # 3) run DiT (TRT-LLM)
        gen_list, _ = self.core.sample(
            cond=ref_mel.to(self.device),
            text=text_idx.to(self.device),
            duration=torch.tensor([target_len], device=self.device, dtype=torch.int32),
            lens=torch.tensor([ref_len], device=self.device, dtype=torch.int32),
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            remove_input_padding=False,
            use_perf=False,
        )
        
        # take generated mel after the ref segment (ref_len : target_len)
        gen_mel = gen_list[0][ref_len:target_len]  # (T_out - T_ref, 100)
        gen_mel = gen_mel.unsqueeze(0).permute(0, 2, 1).contiguous().to(torch.float32)  # -> (1, 100, T)

        # 4) vocoder
        if self._vocoder_is_trt:
            wav = self.vocoder.decode(gen_mel).cpu().squeeze(0)  # (N,)
        else:
            wave = self.vocoder.decode(gen_mel)  # if your torch vocoder exposes decode(); adjust if needed
            wav = wave.cpu().squeeze(0)

        # normalize RMS to target
        rms = torch.sqrt(torch.mean(torch.square(wav)))
        if rms < target_rms:
            wav = wav * (target_rms / (rms + 1e-8))

        # 5) outputs
        spec_to_save = gen_mel.squeeze(0).permute(1, 0).cpu()  # (T, 100)
        if file_wave is not None:
            self.export_wav(wav.numpy(), file_wave, remove_silence)
        if file_spec is not None:
            self.export_spectrogram(spec_to_save, file_spec)

        return wav.numpy(), self.target_sample_rate, spec_to_save
