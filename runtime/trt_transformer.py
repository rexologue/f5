# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from trt_utils import create_model_runner

# === берем "нормальные" классы из твоего src/core/dit.py ===
from src.core.dit import (
    TimestepEmbedding,   # sinus -> MLP -> [B, H]
    TextEmbedding,       # text_num_embeds -> per-step text features [B, T, D_txt]
    InputEmbedding,      # concat(x, cond, text_embed) -> proj -> ConvPos -> +res
)

# AdaLN финальный — из modules (как в твоем DiT)
from src.core.modules import AdaLayerNorm_Final


# -----------------------------
# Вспомогательная обертка над TRT DiT
# -----------------------------
class _DiTRuntime:
    """
    Минимальная обёртка над TensorRT-LLM runner для DiT-ядра.

    ОЖИДАЕМЫЕ I/O (подстрой под свой engine при необходимости):
      inputs:  'x_embed', 't_embed', 'rope_cos', 'rope_sin', 'input_lengths'
      outputs: 'hidden'
    """
    def __init__(self, engine_dir: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.runner = create_model_runner(engine_dir, device=device)
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(
        self,
        x_embed: torch.Tensor,        # [B, T, H]
        t_embed: torch.Tensor,        # [B, H]
        rope_cos: torch.Tensor,       # [B, T, head_dim]
        rope_sin: torch.Tensor,       # [B, T, head_dim]
        input_lengths: torch.Tensor,  # [B]
        scale: float = 1.0,
    ) -> torch.Tensor:
        feeds: Dict[str, torch.Tensor] = {
            "x_embed": x_embed.to(self.dtype),
            "t_embed": t_embed.to(self.dtype),
            "rope_cos": rope_cos.to(self.dtype),
            "rope_sin": rope_sin.to(self.dtype),
            "input_lengths": input_lengths.to(torch.int32),
            # "scale": torch.tensor([scale], device=x_embed.device, dtype=self.dtype),  # если требуется движком
        }
        out = self.runner(feeds)
        hidden = out["hidden"]  # подстрой имя, если у тебя иначе
        return hidden.to(x_embed.dtype)  # [B, T, H]


# -----------------------------
# Основной модуль: DiT на TRT, эмбеддинги/нормы/проекция — Torch
# Совместим с интерфейсом твоего CFM.
# -----------------------------
class F5DiTTRT(nn.Module):
    """
    Совместим с CFM:

      forward(
        x, cond, text, time,
        mask=None, drop_audio_cond=False, drop_text=False,
        cfg_infer=False, cache=True
      ) -> [B, T, mel]  или [2B, T, mel] при cfg_infer=True
    """
    def __init__(
        self,
        ckpt_path: str,             # путь к EMA .pt (host веса для эмбеддингов / norm / proj / text_embed)
        trt_dit_dir: str,           # папка с TRT-ядром DiT
        *,
        mel_dim: int = 100,
        hidden: int = 1024,
        num_heads: int = 16,
        dim_head: int = 64,         # чтобы было явно
        text_num_embeds: int = 256,
        text_dim: Optional[int] = 512,
        # опции из твоего DiT/TextEmbedding:
        text_mask_padding: bool = True,
        text_embedding_average_upsampling: bool = False,
        conv_layers: int = 0,
        conv_mult: int = 2,
        # прочее:
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.hidden = hidden
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.text_dim = text_dim if text_dim is not None else mel_dim

        self.dtype = dtype
        self.device = torch.device(device)

        # --- Torch-часть (ровно как в src/core/dit.py) ---
        self.time_embed = TimestepEmbedding(hidden)
        self.text_embed = TextEmbedding(
            text_num_embeds=text_num_embeds,
            text_dim=self.text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
            conv_mult=conv_mult,
        )
        self.input_embed = InputEmbedding(mel_dim=self.mel_dim, text_dim=self.text_dim, out_dim=self.hidden)

        self.norm_out = AdaLayerNorm_Final(self.hidden)    # финальная модуляция
        self.proj_out = nn.Linear(self.hidden, self.mel_dim)

        # кэш текстовых эмбеддингов (как в DiT)
        self.text_cond: Optional[torch.Tensor] = None
        self.text_uncond: Optional[torch.Tensor] = None

        # заливка host-весов (если есть в чекпойнте)
        self._load_host_weights(ckpt_path)

        self.to(self.device).to(self.dtype)

        # --- TRT DiT ядро ---
        self.dit = _DiTRuntime(trt_dit_dir, device=device, dtype=dtype)

    # ---------------- utils ----------------

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def _rope_cos_sin(self, B: int, T: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Генерация RoPE cos/sin, форма под TRT: [B, T, head_dim].
        (Совместимо с вращением в твоих TRT-опах: rotate_every_two_3dim / apply_rotary_pos_emb_3dim)
        """
        pos = torch.arange(T, device=self.device, dtype=self.dtype)[None, :, None]  # [1,T,1]
        idx = torch.arange(head_dim, device=self.device, dtype=self.dtype)[None, None, :]
        freqs = torch.exp(-math.log(10000.0) * idx / max(1, head_dim - 1))
        ang = pos * freqs
        cos = torch.cos(ang).expand(B, T, head_dim).contiguous()
        sin = torch.sin(ang).expand(B, T, head_dim).contiguous()
        return cos, sin

    def _masked_lengths(self, mask: Optional[torch.Tensor], B: int, T: int) -> torch.Tensor:
        if mask is None:
            return torch.full((B,), T, device=self.device, dtype=torch.long)
        return mask.long().sum(-1)

    def _get_text_embed(
        self,
        text: torch.Tensor,           # [B, Nt] (ids, -1 = pad)
        seq_len: int,
        drop_text: bool,
        audio_mask: Optional[torch.Tensor],
        cache: bool,
    ) -> torch.Tensor:                # -> [B, T, text_dim]
        if cache:
            if drop_text:
                if self.text_uncond is None or self.text_uncond.shape[1] != seq_len:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True, audio_mask=audio_mask)
                return self.text_uncond
            else:
                if self.text_cond is None or self.text_cond.shape[1] != seq_len:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False, audio_mask=audio_mask)
                return self.text_cond
        else:
            return self.text_embed(text, seq_len, drop_text=drop_text, audio_mask=audio_mask)

    def _load_host_weights(self, ckpt_path: str):
        """
        Пытаемся аккуратно залить:
          - time_embed.time_mlp.[0/2].(weight|bias)
          - input_embed.proj.(weight|bias)
          - input_embed.conv_pos_embed.conv1d.{0,2}.(weight|bias)
          - norm_out.linear.(weight|bias)
          - proj_out.(weight|bias)
          - text_embed.text_embed.weight (если есть)
        """
        try:
            sd_all = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            return
        sd = sd_all.get("ema_model_state_dict", sd_all)

        with torch.no_grad():
            # time mlp (как в src/core/dit.py)
            pairs = [
                ("time_embed.time_mlp.0.weight", "ema_model.time_embed.time_mlp.0.weight"),
                ("time_embed.time_mlp.0.bias",   "ema_model.time_embed.time_mlp.0.bias"),
                ("time_embed.time_mlp.2.weight", "ema_model.time_embed.time_mlp.2.weight"),
                ("time_embed.time_mlp.2.bias",   "ema_model.time_embed.time_mlp.2.bias"),
            ]
            for dst, src in pairs:
                if src in sd:
                    module_name, param_name = dst.split(".", 1)
                    getattr(self, module_name).state_dict()[param_name].copy_(sd[src])

            # input_embed.proj
            for key in ("weight", "bias"):
                src = f"ema_model.input_embed.proj.{key}"
                if src in sd:
                    getattr(self.input_embed.proj, key).copy_(sd[src])

            # input_embed.conv_pos_embed.conv1d.{0,2}.(weight|bias)
            for idx in (0, 2):
                for key in ("weight", "bias"):
                    src = f"ema_model.input_embed.conv_pos_embed.conv1d.{idx}.{key}"
                    if src in sd:
                        getattr(self.input_embed.conv_pos_embed.conv1d[idx], key).copy_(sd[src])

            # norm_out.linear
            for key in ("weight", "bias"):
                src = f"ema_model.norm_out.linear.{key}"
                if src in sd:
                    getattr(self.norm_out.linear, key).copy_(sd[src])

            # proj_out
            for key in ("weight", "bias"):
                src = f"ema_model.proj_out.{key}"
                if src in sd:
                    getattr(self.proj_out, key).copy_(sd[src])

            # text embedding weights
            for cand in (
                "ema_model.text_embed.text_embed.weight",
                "ema_model.transformer.text_embed.text_embed.weight",
                "ema_model.text_embed.weight",
                "ema_model.transformer.text_embed.weight",
            ):
                if cand in sd and sd[cand].shape == self.text_embed.text_embed.weight.shape:
                    self.text_embed.text_embed.weight.copy_(sd[cand])
                    break

    # ---------------- forward ----------------

    @torch.inference_mode()
    def forward(
        self,
        *,
        x: torch.Tensor,                     # [B, T, mel]
        cond: torch.Tensor,                  # [B, T, mel]
        text: torch.Tensor,                  # [B, Nt] (ids, -1 = pad)
        time: torch.Tensor,                  # scalar / [B] / [B,256]
        mask: Optional[torch.Tensor] = None, # [B, T] bool
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Возвращает:
          - [B, T, mel] если cfg_infer=False
          - [2B, T, mel] если cfg_infer=True (cond ⊕ uncond)
        """
        B, T, _ = x.shape
        if time.ndim == 0:
            time = time.repeat(B)

        if time.ndim == 0:
            time = time.repeat(B)
        t_emb = self.time_embed(time)    

        head_dim = self.dim_head     # как в конфиге DiT/TRT (обычно 64)
        rope_cos, rope_sin = self._rope_cos_sin(B, T, head_dim)   # [B,T,head_dim]
        input_lengths = self._masked_lengths(mask, B, T)          # [B]

        def _one_pass(x_in: torch.Tensor, cond_in: torch.Tensor, drop_a: bool, drop_t: bool) -> torch.Tensor:
            # text -> [B,T,text_dim] (с теми же флагами и cache, как в твоём DiT)
            text_embed = self._get_text_embed(
                text=text,
                seq_len=T,
                drop_text=drop_t,
                audio_mask=mask,
                cache=cache,
            )                                                     # [B,T,D_txt]

            # [B,T,mel]*2 + [B,T,D_txt] -> [B,T,H]
            x_emb = self.input_embed(
                x=x_in.to(self.dtype),
                cond=cond_in.to(self.dtype),
                text_embed=text_embed.to(self.dtype),
                drop_audio_cond=drop_a,
            )

            # TRT DiT
            h = self.dit(
                x_embed=x_emb,
                t_embed=t_emb,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                input_lengths=input_lengths,
                scale=scale,
            )                                                     # [B,T,H]

            # финал Torch
            y = self.norm_out(h, t_emb)                           # [B,T,H]
            y = self.proj_out(y).to(x.dtype)                      # [B,T,mel]
            return y

        if not cfg_infer:
            return _one_pass(x, cond, drop_audio_cond, drop_text)

        # cfg_infer=True — в один проход формируем 2B (cond/uncond)
        # но чтобы не плодить гигантские конкатенации, делаем два вызова и склеиваем (для CFM это эквивалентно)
        y_cond   = _one_pass(x, cond, drop_a=False, drop_t=False)         # [B,T,mel]
        y_uncond = _one_pass(x, cond, drop_a=True,  drop_t=True)          # [B,T,mel]
        return torch.cat([y_cond, y_uncond], dim=0)                        # [2B,T,mel]
