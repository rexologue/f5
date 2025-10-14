from __future__ import annotations

import os
import random
from pathlib import Path
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###########
# HELPERS #
###########


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib
        package_exists = importlib.util.find_spec(package_name) is not None

        return package_exists

    except Exception:
        return False


##################
# TENSOR HELPERS #
##################


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)

def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    """
    Char tokenizer, based on  extracted .txt file
    """
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    
    return text


#############
# TOKENIZER #
#############


def get_tokenizer(path_to_vocab: Path)-> tuple[dict[str, int], int]:
    with open(str(path_to_vocab), "r", encoding="utf-8") as f:
        vocab_char_map = {}

        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i

        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


def repetition_found(text, length=2, tolerance=10) -> bool:
    """
    Filter func for dirty data with many repetitions
    """
    pattern_count = defaultdict(int)

    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1

    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
        
    return False


# ------------------------------------------


def get_epss_timesteps(n, device, dtype) -> torch.Tensor:
    """
    Empirically pruned step for sampling
    """

    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])

    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    
    return dt * torch.tensor(t, device=device, dtype=dtype)
