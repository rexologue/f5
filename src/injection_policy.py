# -*- coding: utf-8 -*-
from __future__ import annotations
import dataclasses
from typing import Optional, Tuple, List, Any, Dict

import torch
import torch.nn as nn
import deepspeed

# === Ваши блоки ===
from .core.modules import Attention, FeedForward, DiTBlock

# ------------------------------------------------------------
# 1) ВСПОМОГАТЕЛЬНОЕ: алиасы и валидация путей для TP-фолбэка
# ------------------------------------------------------------

def ds_add_aliases(model: nn.Module) -> nn.Module:
    """
    Удобные алиасы, если хотите работать через injection_policy (TP).
    Но для kernel injection они не нужны.
    """
    for m in model.modules():
        if isinstance(m, Attention):
            if not hasattr(m, "out_proj"):
                m.out_proj = m.to_out[0]  # nn.Linear
        if isinstance(m, FeedForward):
            if not hasattr(m, "out_proj"):
                m.out_proj = m.ff[-1]    # nn.Linear
    # sanity
    for m in model.modules():
        if isinstance(m, DiTBlock):
            assert isinstance(m.attn.to_out[0], nn.Linear)
            assert isinstance(m.ff.ff[-1], nn.Linear)
            break
    return model

# Для TP (без кернелов) — указываем реальные пути, которые точно найдутся в named_modules():
INJ_DIT_TP = {
    DiTBlock: ("attn.to_out.0", "ff.ff.2")  # внимание-выход и "головной" выход FFN
}

# ------------------------------------------------------------
# 2) ПОЛИТИКА ДЛЯ KERNEL INJECTION (максимум на 1 GPU)
# ------------------------------------------------------------
# В 0.18.x пользовательскую политику для кернелов нельзя передать через
# параметр init_inference. Поэтому мы не кладём её в injection_policy.
# DeepSpeed попытается применить кернелы автоматически к слоям, которые узнает.
# Для ваших кастомных слоёв это не сработает «из коробки», но:
#  - на 1 GPU вы всё равно получите выгоду от cudagraph/triton/fused-кернелов,
#    если выставить правильные флаги (см. wrap_with_deepspeed_inference ниже);
#  - если всё же захотите «жёсткую» интеграцию с policy-классом на уровне
#    deepspeed.module_inject (как у HFBertLayerPolicy), придётся патчить сам
#    DeepSpeed (добавлять ваш policy в их registry). Это делается в исходниках
#    DeepSpeed (replace_policy.py) и не поддерживается как публичный API в 0.18.x.

# Ниже оставляю каркас "DiTPolicy" как справочный (для возможного патча DS).
# Он НИЖЕ НЕ ИСПОЛЬЗУЕТСЯ напрямую в init_inference, но показывает,
# какие тензоры обычно мапят в контейнер (QKV/Out и FFN).
@dataclasses.dataclass
class _ProjTensors:
    qkv: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    o: torch.Tensor
    ff_in: torch.Tensor
    ff_out: torch.Tensor
    qkv_b: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    o_b: Optional[torch.Tensor] = None
    ff_in_b: Optional[torch.Tensor] = None
    ff_out_b: Optional[torch.Tensor] = None

class DiTPolicyReference:
    """
    Справочная заготовка policy-класса (для модификации DS).
    Показывает, как собрать QKV/Out и FFN из ваших модулей.
    """
    def __init__(self, client_module: DiTBlock):
        self.m = client_module

    def attn_qkv_o(self) -> _ProjTensors:
        attn = self.m.attn
        # Ваши Q/K/V — это to_q, to_k, to_v (Linear), out — to_out[0] (Linear)
        Wq, Wk, Wv = attn.to_q.weight, attn.to_k.weight, attn.to_v.weight
        bq = getattr(attn.to_q, "bias", None)
        bk = getattr(attn.to_k, "bias", None)
        bv = getattr(attn.to_v, "bias", None)
        Wo = attn.to_out[0].weight
        bo = getattr(attn.to_out[0], "bias", None)

        return _ProjTensors(
            qkv=(Wq, Wk, Wv),
            o=Wo,
            ff_in=None, ff_out=None,
            qkv_b=(bq, bk, bv),
            o_b=bo
        )

    def ffn_in_out(self) -> _ProjTensors:
        ff = self.m.ff
        # Ваша FFN — это Sequential внутри FeedForward: [Linear, GELU, Dropout, Linear]
        W_in = ff.ff[0].weight
        b_in = getattr(ff.ff[0], "bias", None)
        W_out = ff.ff[-1].weight
        b_out = getattr(ff.ff[-1], "bias", None)
        return _ProjTensors(
            qkv=(None, None, None),
            o=None,
            ff_in=W_in, ff_out=W_out,
            ff_in_b=b_in, ff_out_b=b_out
        )

# ------------------------------------------------------------
# 3) ОБЁРТКА ВОКРУГ init_inference: правильно включаем KI и TP
# ------------------------------------------------------------

def _ds_init_kernel_inject(model: nn.Module,
                           dtype: torch.dtype = torch.float16,
                           use_triton: bool = True,
                           triton_autotune: bool = False) -> Tuple[Any, nn.Module]:
    """
    Попытка kernel injection на 1 GPU без user injection_policy.
    ВАЖНО: no 'injection_policy' здесь — иначе DS 0.18.x ругается.
    """
    # Некоторым кернелам DS нужен tuple-выход — у вас блоки возвращают Tensor,
    # поэтому ставим return_tuple=False. Маскинг некаузальный → triangular_masking=False.
    engine = deepspeed.init_inference(
        model=model,
        dtype=dtype,
        tensor_parallel={"tp_size": 1},
        replace_with_kernel_inject=True,   # включаем KI
        return_tuple=False,
        triangular_masking=False,          # у вас attention is_causal=False
        use_triton=use_triton,
        triton_autotune=triton_autotune,
    )
    return engine, engine.module

def _ds_init_tp_fallback(model: nn.Module,
                         dtype: torch.dtype = torch.float16) -> Tuple[Any, nn.Module]:
    """
    Фолбэк: чистый TP/AutoTP без кернелов с явной injection_policy.
    Тут как раз РАЗРЕШЕНО передавать injection_policy.
    """
    # на всякий случай убедимся, что пути существуют
    for name, child in model.named_modules():
        pass  # вызов пройдёт дерево; ошибки нам не нужны

    engine = deepspeed.init_inference(
        model=model,
        dtype=dtype,
        tensor_parallel={"tp_size": 1},
        # kernel_inject выключен => можно policy
        injection_policy=INJ_DIT_TP
    )
    return engine, engine.module

def wrap_with_deepspeed_inference(model: nn.Module,
                                  prefer_kernel_inject: bool = True,
                                  dtype: torch.dtype = torch.float16,
                                  use_triton: bool = True,
                                  triton_autotune: bool = False) -> Tuple[Any, nn.Module]:
    """
    1) Пытаемся включить kernel injection (без user policy).
    2) Если не взлетело — печатаем причину и аккуратно падаем на TP-путь.
    """
    if prefer_kernel_inject:
        try:
            return _ds_init_kernel_inject(model, dtype=dtype,
                                          use_triton=use_triton,
                                          triton_autotune=triton_autotune)
        except Exception as e:
            print(f"[DeepSpeed] kernel_inject не применился: {e}\nПробуем некернельный путь...")
    # TP/AutoTP без кернелов
    return _ds_init_tp_fallback(model, dtype=dtype)
