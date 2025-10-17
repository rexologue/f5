# runtime/setup/convert_checkpoint.py (ИСПРАВЛЕНО для vocab_size)
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import safetensors.torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm import str_dtype_to_torch

from utils import preload_libpython
preload_libpython()


def split_q_tp(v, tensor_parallel, rank):
    from tensorrt_llm.models.convert_utils import split
    return split(v, tensor_parallel, rank, dim=1).contiguous()


def split_q_bias_tp(v, tensor_parallel, rank):
    from tensorrt_llm.models.convert_utils import split
    return split(v, tensor_parallel, rank, dim=0).contiguous()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    from tensorrt_llm.models.convert_utils import split_matrix_tp as _split
    return _split(v, tensor_parallel, rank, dim=dim)


FACEBOOK_DIT_NAME_MAPPING = {
    r"^transformer_blocks\.(\d+)\.attn\.to_q\.weight$": r"transformer_blocks.\1.attn.to_q.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_q\.bias$": r"transformer_blocks.\1.attn.to_q.bias",
    r"^transformer_blocks\.(\d+)\.attn\.to_k\.weight$": r"transformer_blocks.\1.attn.to_k.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_k\.bias$": r"transformer_blocks.\1.attn.to_k.bias",
    r"^transformer_blocks\.(\d+)\.attn\.to_v\.weight$": r"transformer_blocks.\1.attn.to_v.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_v\.bias$": r"transformer_blocks.\1.attn.to_v.bias",
    r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.weight$": r"transformer_blocks.\1.attn.to_out.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.bias$": r"transformer_blocks.\1.attn.to_out.bias",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.0\.0\.weight$": r"transformer_blocks.\1.ff.project_in.weight",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.0\.0\.bias$": r"transformer_blocks.\1.ff.project_in.bias",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.2\.weight$": r"transformer_blocks.\1.ff.ff.weight",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.2\.bias$": r"transformer_blocks.\1.ff.ff.bias",
    r"^transformer_blocks\.(\d+)\.attn_norm\.linear\.weight$": r"transformer_blocks.\1.attn_norm.linear.weight",
    r"^transformer_blocks\.(\d+)\.attn_norm\.linear\.bias$": r"transformer_blocks.\1.attn_norm.linear.bias",
}


def _infer_vocab_size(state_dict: Dict[str, torch.Tensor]) -> tuple[int | None, str | None]:
    """Извлекаем vocab_size из text_embed весов."""
    candidates = [
        "ema_model.text_embed.text_embed.weight",
        "ema_model.transformer.text_embed.text_embed.weight",
        "text_embed.text_embed.weight",
    ]
    for key in candidates:
        if key in state_dict:
            weight = state_dict[key]
            if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
                # ВАЖНО: вычитаем 1, т.к. в коде используется +1 для filler token
                vocab_size = int(weight.shape[0]) - 1
                return vocab_size, key
    return None, None


def convert_timm_dit(args, mapping, dtype="float16") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    torch_dtype = str_dtype_to_torch(dtype)
    tensor_parallel = mapping.tp_size
    
    ckpt = torch.load(args.timm_ckpt, map_location="cpu")
    state_dict = ckpt.get("ema_model_state_dict", ckpt)
    
    # Извлекаем vocab_size
    vocab_size, vocab_source = _infer_vocab_size(state_dict)
    
    # Fallback на аргумент командной строки
    if vocab_size is None and args.vocab_size is not None and args.vocab_size > 0:
        vocab_size = args.vocab_size
        vocab_source = "cli_argument"
    
    # Если всё ещё None - ставим дефолт
    if vocab_size is None:
        print("⚠️  WARNING: vocab_size не найден, используем дефолт 2546")
        vocab_size = 2546
        vocab_source = "default"
    
    metadata = {
        "vocab_size": vocab_size,
        "vocab_source": vocab_source,
    }
    
    print(f"📊 Vocab size: {vocab_size} (источник: {vocab_source})")
    
    # Берём только transformer поддерево
    prefix = "ema_model.transformer."
    model_params = {
        k[len(prefix):] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
        if k.startswith("ema_model.transformer")
    }
    
    # Маппинг имён
    def map_name(src: str) -> str | None:
        for pat, repl in FACEBOOK_DIT_NAME_MAPPING.items():
            if re.match(pat, src):
                return re.sub(pat, repl, src)
        return None
    
    weights = {}
    for name, param in model_params.items():
        new_name = map_name(name)
        if new_name is None:
            continue
        weights[new_name] = param.contiguous().to(torch_dtype)
    
    # TP-шардинг
    for k in list(weights.keys()):
        v = weights[k]
        
        if re.match(r"^transformer_blocks\.\d+\.attn\.to_[qkv]\.weight$", k):
            weights[k] = split_q_tp(v, tensor_parallel, mapping.tp_rank)
        elif re.match(r"^transformer_blocks\.\d+\.attn\.to_[qkv]\.bias$", k):
            weights[k] = split_q_bias_tp(v, tensor_parallel, mapping.tp_rank)
        elif re.match(r"^transformer_blocks\.\d+\.attn\.to_out\.weight$", k):
            weights[k] = split_matrix_tp(v, tensor_parallel, mapping.tp_rank, dim=1)
        elif re.match(r"^transformer_blocks\.\d+\.ff\.project_in\.weight$", k):
            weights[k] = split_matrix_tp(v, tensor_parallel, mapping.tp_rank, dim=1)
        elif re.match(r"^transformer_blocks\.\d+\.ff\.project_in\.bias$", k):
            from tensorrt_llm.models.convert_utils import split
            weights[k] = split(v, tensor_parallel, mapping.tp_rank, dim=0).contiguous()
        elif re.match(r"^transformer_blocks\.\d+\.ff\.ff\.weight$", k):
            weights[k] = split_matrix_tp(v, tensor_parallel, mapping.tp_rank, dim=0)
    
    print(f"✅ Loaded {len(weights)} DiT block weights (TP={tensor_parallel}, rank={mapping.tp_rank})")
    return weights, metadata


def save_config(args, metadata: Dict[str, Any]):
    """Сохраняем config.json с ПРАВИЛЬНЫМ vocab_size."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    vocab_size = metadata.get("vocab_size")
    
    # ВАЖНО: vocab_size должен быть во ВСЕХ нужных местах
    config = {
        "architecture": "DiT",
        "dtype": args.dtype,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.depth,
        "num_attention_heads": args.num_heads,
        "dim_head": args.hidden_size // args.num_heads,
        "vocab_size": vocab_size,  # <- добавлено на верхний уровень
        "mapping": {
            "world_size": args.tp_size * args.pp_size,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
        },
        "builder_config": {
            "precision": args.dtype,
            "vocab_size": vocab_size,  # <- ОБЯЗАТЕЛЬНО здесь
        },
        "pretrained_config": {
            "vocab_size": vocab_size,  # <- и здесь для совместимости
        },
    }
    
    if metadata.get("vocab_source"):
        config["metadata"] = {"vocab_source": metadata["vocab_source"]}
    
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config сохранён: {config_path}")
    print(f"   vocab_size: {vocab_size}")


def convert_and_save(args, rank):
    mapping = Mapping(
        world_size=args.tp_size * args.pp_size,
        rank=rank,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )
    
    weights, metadata = convert_timm_dit(args, mapping, dtype=args.dtype)
    
    if rank == 0:
        save_config(args, metadata)
    
    safetensors.torch.save_file(
        weights,
        os.path.join(args.output_dir, f"rank{rank}.safetensors")
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timm_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./tllm_checkpoint_dit")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=22)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size")
    return parser.parse_args()


def main():
    args = parse_arguments()
    assert args.pp_size == 1, "PP не поддерживается"
    
    world_size = args.tp_size * args.pp_size
    
    print(f"🚀 Конвертация DiT checkpoint → TRT-LLM (TP={args.tp_size})")
    
    for rank in range(world_size):
        convert_and_save(args, rank)
    
    print(f"✅ Готово! Checkpoint сохранён в {args.output_dir}")


if __name__ == "__main__":
    main()