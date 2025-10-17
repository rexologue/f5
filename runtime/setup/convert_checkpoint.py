# runtime/setup/convert_checkpoint.py
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import traceback
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple

from utils import preload_libpython
preload_libpython()

import torch
import safetensors.torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm import str_dtype_to_torch
from tensorrt_llm.models.convert_utils import split, split_matrix_tp


def split_q_tp(v, n_head, n_hidden, tensor_parallel, rank):
    split_v = split(v, tensor_parallel, rank, dim=1)
    return split_v.contiguous()


def split_q_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    split_v = split(v, tensor_parallel, rank, dim=0)
    return split_v.contiguous()


# Маппим ТОЛЬКО то, что относится к DiT-блокам
FACEBOOK_DIT_NAME_MAPPING = {
    # attention out (Sequential -> weight/bias)
    r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.weight$": r"transformer_blocks.\1.attn.to_out.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.bias$":   r"transformer_blocks.\1.attn.to_out.bias",

    # feed-forward (Sequential -> project_in / ff)
    r"^transformer_blocks\.(\d+)\.ff\.ff\.0\.0\.weight$": r"transformer_blocks.\1.ff.project_in.weight",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.0\.0\.bias$":   r"transformer_blocks.\1.ff.project_in.bias",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.2\.weight$":    r"transformer_blocks.\1.ff.ff.weight",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.2\.bias$":      r"transformer_blocks.\1.ff.ff.bias",

    # если в чекпойнте уже «плоские» имена — тоже поддержим
    r"^transformer_blocks\.(\d+)\.attn\.to_out\.weight$": r"transformer_blocks.\1.attn.to_out.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_out\.bias$":   r"transformer_blocks.\1.attn.to_out.bias",
    r"^transformer_blocks\.(\d+)\.ff\.project_in\.weight$": r"transformer_blocks.\1.ff.project_in.weight",
    r"^transformer_blocks\.(\d+)\.ff\.project_in\.bias$":   r"transformer_blocks.\1.ff.project_in.bias",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.weight$":         r"transformer_blocks.\1.ff.ff.weight",
    r"^transformer_blocks\.(\d+)\.ff\.ff\.bias$":           r"transformer_blocks.\1.ff.ff.bias",

    # qkv
    r"^transformer_blocks\.(\d+)\.attn\.to_q\.weight$": r"transformer_blocks.\1.attn.to_q.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_q\.bias$":   r"transformer_blocks.\1.attn.to_q.bias",
    r"^transformer_blocks\.(\d+)\.attn\.to_k\.weight$": r"transformer_blocks.\1.attn.to_k.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_k\.bias$":   r"transformer_blocks.\1.attn.to_k.bias",
    r"^transformer_blocks\.(\d+)\.attn\.to_v\.weight$": r"transformer_blocks.\1.attn.to_v.weight",
    r"^transformer_blocks\.(\d+)\.attn\.to_v\.bias$":   r"transformer_blocks.\1.attn.to_v.bias",

    # AdaLayerNormZero for attn input
    r"^transformer_blocks\.(\d+)\.attn_norm\.linear\.weight$": r"transformer_blocks.\1.attn_norm.linear.weight",
    r"^transformer_blocks\.(\d+)\.attn_norm\.linear\.bias$":   r"transformer_blocks.\1.attn_norm.linear.bias",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="F5TTS_Base", choices=["F5TTS_Base"])
    parser.add_argument("--timm_ckpt", type=str, default="./ckpts/model_1200000.pt")
    parser.add_argument("--output_dir", type=str, default="./tllm_checkpoint_dit",
                        help="Where to save the TensorRT-LLM DiT checkpoint")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=22)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--fp8_linear", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override tokenizer vocabulary size when checkpoint metadata is unavailable.",
    )
    return parser.parse_args()


VOCAB_WEIGHT_CANDIDATES = (
    "ema_model.transformer.text_embed.text_embed.weight",
    "ema_model.text_embed.text_embed.weight",
    "transformer.text_embed.text_embed.weight",
    "text_embed.text_embed.weight",
)


def _infer_vocab_size(state_dict: Dict[str, torch.Tensor]) -> tuple[int | None, str | None]:
    for key in VOCAB_WEIGHT_CANDIDATES:
        if key in state_dict:
            weight = state_dict[key]
            if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
                return int(weight.shape[0]), key
    return None, None


def convert_timm_dit(args, mapping, dtype="float16") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    tik = time.time()
    torch_dtype = str_dtype_to_torch(dtype)
    tensor_parallel = mapping.tp_size

    ckpt = dict(torch.load(args.timm_ckpt))
    state_dict = ckpt.get("ema_model_state_dict", ckpt)

    vocab_size = args.vocab_size
    vocab_source = "cli"
    if vocab_size is None:
        vocab_size, vocab_source = _infer_vocab_size(state_dict)
    metadata: Dict[str, Any] = {
        "vocab_size": vocab_size,
        "vocab_source": vocab_source,
    }

    # берём только поддерево трансформера
    model_params = {k: v for k, v in state_dict.items() if k.startswith("ema_model.transformer")}
    prefix = "ema_model.transformer."
    model_params = {key[len(prefix):] if key.startswith(prefix) else key: value
                    for key, value in model_params.items()}

    # helper: regex map
    def map_name(src_name: str) -> str | None:
        for pat, repl in FACEBOOK_DIT_NAME_MAPPING.items():
            m = re.match(pat, src_name)
            if m:
                return re.sub(pat, repl, src_name)
        return None  # не относящееся к DiT — выкидываем

    weights: Dict[str, torch.Tensor] = {}
    for name, param in model_params.items():
        new_name = map_name(name)
        if new_name is None:
            continue
        v = param.contiguous().to(torch_dtype)
        weights[new_name] = v

    # TP-шардинг (как у тебя было; без доп. скейлов для Q/K)
    for k, v in list(weights.items()):
        if re.match(r"^transformer_blocks\.\d+\.attn\.to_[qkv]\.weight$", k):
            weights[k] = split_q_tp(v, args.num_heads, args.hidden_size, tensor_parallel, mapping.tp_rank)
        elif re.match(r"^transformer_blocks\.\d+\.attn\.to_[qkv]\.bias$", k):
            weights[k] = split_q_bias_tp(v, args.num_heads, args.hidden_size, tensor_parallel, mapping.tp_rank)
        elif re.match(r"^transformer_blocks\.\d+\.attn\.to_out\.weight$", k):
            weights[k] = split_matrix_tp(v, tensor_parallel, mapping.tp_rank, dim=1)
        # ff
        elif re.match(r"^transformer_blocks\.\d+\.ff\.project_in\.weight$", k):
            weights[k] = split_matrix_tp(v, tensor_parallel, mapping.tp_rank, dim=1)
        elif re.match(r"^transformer_blocks\.\d+\.ff\.project_in\.bias$", k):
            weights[k] = split(v, tensor_parallel, mapping.tp_rank, dim=0).contiguous()
        elif re.match(r"^transformer_blocks\.\d+\.ff\.ff\.weight$", k):
            weights[k] = split_matrix_tp(v, tensor_parallel, mapping.tp_rank, dim=0)
        elif re.match(r"^transformer_blocks\.\d+\.ff\.ff\.bias$", k):
            weights[k] = v
        # attn_norm.linear.* без TP
        elif re.match(r"^transformer_blocks\.\d+\.attn_norm\.linear\.(weight|bias)$", k):
            weights[k] = v

    print(f"Weights loaded (DiT only). Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time()-tik))}")
    return weights, metadata


def save_config(args, metadata: Dict[str, Any]):
    os.makedirs(args.output_dir, exist_ok=True)
    vocab_size = metadata.get("vocab_size")
    builder_config: Dict[str, Any] = {"precision": args.dtype}
    if vocab_size is not None:
        builder_config["vocab_size"] = int(vocab_size)

    config = {
        "architecture": "DiT",
        "dtype": args.dtype,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.depth,
        "num_attention_heads": args.num_heads,
        "dim_head": args.hidden_size // args.num_heads,
        "mapping": {
            "world_size": args.cp_size * args.tp_size * args.pp_size,
            "cp_size": args.cp_size,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
        },
        "builder_config": builder_config,
    }
    if vocab_size is not None:
        config.setdefault("tokenizer", {})["vocab_size"] = int(vocab_size)
        config.setdefault("pretrained_config", {})["vocab_size"] = int(vocab_size)
    if metadata.get("vocab_source"):
        config.setdefault("metadata", {})["vocab_source"] = metadata["vocab_source"]
    if args.fp8_linear:
        config["quantization"] = {"quant_algo": "FP8"}

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def convert_and_save(args, rank):
    mapping = Mapping(
        world_size=args.cp_size * args.tp_size * args.pp_size,
        rank=rank,
        cp_size=args.cp_size,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )

    weights, metadata = convert_timm_dit(args, mapping, dtype=args.dtype)
    if rank == 0:
        save_config(args, metadata)
    safetensors.torch.save_file(weights, os.path.join(args.output_dir, f"rank{rank}.safetensors"))


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()
                    exceptions.append(1)
            assert len(exceptions) == 0, "Checkpoint conversion failed, please check error log."


def main():
    args = parse_arguments()
    world_size = args.cp_size * args.tp_size * args.pp_size
    assert args.pp_size == 1, "PP is not supported yet."

    tik = time.time()
    print("Start DiT-only checkpoint conversion")
    execute(args.workers, [convert_and_save] * world_size, args)
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-tik))}")


if __name__ == "__main__":
    main()
