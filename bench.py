# bench_f5tts_ljs.py
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Tuple

import tqdm

from ruaccent import RUAccent

# Подправь импорт под реальное расположение класса F5TTS
from src.f5tts import F5TTS  # например: from src.api import F5TTS


def read_metadata(metadata_path: Path) -> List[Tuple[str, str]]:
    """
    Читает LJS-подобный metadata.csv с форматом: id|text
    Возвращает список (id, text) без пустых или битых строк.
    """
    pairs: List[Tuple[str, str]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", maxsplit=1)
            if len(parts) != 2:
                continue
            ex_id = parts[0].strip()
            text = parts[1].strip()
            if ex_id and text:
                pairs.append((ex_id, text))
    return pairs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def format_seconds(s: float) -> str:
    # фиксированный формат для CSV (3 знака после запятой, точка как разделитель)
    return f"{s:.3f}"


def safe_len_seconds(wav, sr) -> float:
    try:
        return float(len(wav)) / float(sr)
    except Exception:
        return float("nan")


def load_ref_from_dir(ref_dir: Path) -> tuple[Path, str]:
    wav_path = ref_dir / "ref.wav"
    txt_path = ref_dir / "ref.txt"
    if not wav_path.exists():
        raise FileNotFoundError(f"В папке {ref_dir} не найден ref.wav")
    if not txt_path.exists():
        raise FileNotFoundError(f"В папке {ref_dir} не найден ref.txt")
    ref_text = txt_path.read_text(encoding="utf-8").strip()
    if not ref_text:
        raise ValueError(f"Файл {txt_path} пуст")
    return wav_path, ref_text


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TTFA/RTF для F5TTS на папке с LJSpeech-подобным metadata.csv"
    )
    parser.add_argument("--input", type=Path, required=True, help="Папка с metadata.csv")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Папка для результатов (по умолчанию: ./output)")
    parser.add_argument("--config", type=Path, required=True, help="Путь к config.yaml модели (F5 settings)")
    parser.add_argument("--vocab", type=Path, required=True, help="Путь к vocab файлу")
    parser.add_argument("--ckpt", type=Path, required=True, help="Путь к checkpoint модели")
    parser.add_argument("--vocoder", type=Path, required=True, help="Локальный путь к вокодеру")

    # Новый способ: папка с ref.wav и ref.txt
    parser.add_argument("--ref-dir", type=Path, default=None, help="Папка с ref.wav и ref.txt (UTF-8)")

    # Старый способ (оставлен для совместимости)
    parser.add_argument("--ref-wav", type=Path, default=None, help="Референс-аудио для голоса/стиля (если не используется --ref-dir)")
    parser.add_argument("--ref-text", type=str, default=None, help="Референс-текст (если не используется --ref-dir)")

    parser.add_argument("--device", type=str, default=None, help="Принудительное устройство: cuda|cpu|mps|xpu")
    parser.add_argument("--no-deepspeed", action="store_true", help="Отключить DeepSpeed kernel injection")
    parser.add_argument("--remove-silence", action="store_true", help="Удалить тишину на выходе после сохранения WAV")
    parser.add_argument("--nfe-step", type=int, default=32, help="nfe_step для infer()")
    parser.add_argument("--cfg-strength", type=float, default=2.0, help="cfg_strength для infer()")
    parser.add_argument("--speed", type=float, default=1.0, help="Скорость речи (time-stretch) для infer()")
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0, help="sway_sampling_coef для infer()")
    parser.add_argument("--fix-duration", type=float, default=None, help="Зафиксировать длительность (сек) или не задавать")
    parser.add_argument("--seed", type=int, default=None, help="Базовый сид для повторяемости")
    parser.add_argument("--warmup", action="store_true", help="Сделать один прогрев (не учитывается в метриках)")
    parser.add_argument("--limit", type=int, default=None, help="Обработать только первые N строк metadata.csv")

    args = parser.parse_args()

    # Разбираем референс
    if args.ref_dir is not None:
        ref_wav_path, ref_text_val = load_ref_from_dir(args.ref_dir)
    else:
        if args.ref_wav is None or args.ref_text is None:
            raise ValueError("Укажи либо --ref-dir, либо вместе --ref-wav и --ref-text")
        ref_wav_path = args.ref_wav
        ref_text_val = args.ref_text

    accentizer = RUAccent()
    accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)

    ref_text_val = accentizer.process_all(ref_text_val)

    input_dir: Path = args.input
    metadata_path = input_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Не найден {metadata_path}")

    all_pairs = read_metadata(metadata_path)
    if not all_pairs:
        raise RuntimeError(f"Файл {metadata_path} пуст или не содержит валидных строк")

    # Ограничение по числу сэмплов
    if args.limit is not None and args.limit >= 0:
        pairs = all_pairs[: args.limit]
    else:
        pairs = all_pairs

    out_root: Path = args.output
    out_wavs = out_root / "wavs"
    ensure_dir(out_wavs)

    info_csv = out_root / "info.csv"
    metrics_csv = out_root / "metrics.csv"
    ensure_dir(info_csv.parent)

    # Инициализация модели один раз
    tts = F5TTS(
        config_path=args.config,
        vocab_file=args.vocab,
        ckpt_file=args.ckpt,
        vocoder_local_path=args.vocoder,
        ode_method="euler",
        use_ema=True,
        device=args.device,
        use_deepspeed=not args.no_deepspeed,
    )

    # Необязательный прогрев: используем короткий валидный текст "тест" (НЕ "."),
    # чтобы не получить пустую токенизацию и ошибку min() по пустому тензору.
    if args.warmup:
        try:
            _ = tts.infer(
                ref_file=ref_wav_path,
                ref_text=ref_text_val,
                gen_text="тест",
                show_info=lambda *a, **k: None,
                progress=tqdm.tqdm,
                nfe_step=max(4, args.nfe_step // 4),
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=args.sway_sampling_coef,
                speed=args.speed,
                fix_duration=0.5 if args.fix_duration is None else min(args.fix_duration, 0.5),
                remove_silence=False,
                file_wave=None,
                file_spec=None,
                seed=args.seed if args.seed is not None else 1234,
            )
        except Exception as e:
            print(f"[warmup] пропущен из-за ошибки: {e}")

    # Основной бенчмарк
    times: List[float] = []
    info_lines: List[str] = []
    metrics_lines: List[str] = []
    metrics_header = "id|ttfa|dur_s|rtf|chars|chars_per_s"
    metrics_lines.append(metrics_header)

    pbar = tqdm.tqdm(pairs, desc="Benchmarking", dynamic_ncols=True)

    for ex_id, gen_text in pbar:
        # Детерминированный сид: фиксированный или от id
        seed = args.seed if args.seed is not None else (abs(hash(ex_id)) % (2**31 - 1) or 1)

        wav_path = out_wavs / f"{ex_id}.wav"

        try:
            t0 = time.perf_counter()
            _wav, _sr, _spec = tts.infer(
                ref_file=ref_wav_path,
                ref_text=ref_text_val,
                gen_text=accentizer.process_all(gen_text),
                show_info=lambda *a, **k: None,
                progress=tqdm.tqdm,  # совместимость с сигнатурой
                nfe_step=args.nfe_step,
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=args.sway_sampling_coef,
                speed=args.speed,
                fix_duration=args.fix_duration,
                remove_silence=args.remove_silence,
                file_wave=wav_path,
                file_spec=None,
                seed=seed,
            )
            t1 = time.perf_counter()
            ttfa = t1 - t0  # оффлайн: время до готового WAV
            times.append(ttfa)
            info_lines.append(f"{ex_id}|{format_seconds(ttfa)}")

            dur_s = safe_len_seconds(_wav, _sr)
            rtf = (ttfa / dur_s) if dur_s and dur_s > 0 else float("nan")
            chars = len(gen_text)
            cps = (chars / ttfa) if ttfa > 0 else float("nan")

            metrics_lines.append(
                f"{ex_id}|{format_seconds(ttfa)}|{format_seconds(dur_s)}|{format_seconds(rtf)}|{chars}|{cps:.2f}"
                if math.isfinite(cps) else
                f"{ex_id}|{format_seconds(ttfa)}|{format_seconds(dur_s)}|{format_seconds(rtf)}|{chars}|nan"
            )

            pbar.set_postfix_str(f"id={ex_id} ttfa={format_seconds(ttfa)}s rtf={format_seconds(rtf)}")
        except Exception as e:
            print(f"[{ex_id}] ошибка: {e}")
            info_lines.append(f"{ex_id}|nan")
            metrics_lines.append(f"{ex_id}|nan|nan|nan|{len(gen_text)}|nan")

    # Средние по успешным
    valid_times = [t for t in times if math.isfinite(t)]
    avg_ttfa = sum(valid_times) / len(valid_times) if valid_times else float("nan")
    info_lines.append(f"avg|{format_seconds(avg_ttfa) if math.isfinite(avg_ttfa) else 'nan'}")

    # Средний RTF по валидным строкам
    rtfs: List[float] = []
    for line in metrics_lines[1:]:  # пропустить header
        try:
            _, ttfa_s, dur_s, rtf_s, _chars, _cps = line.split("|")
            rtf_v = float(rtf_s)
            if math.isfinite(rtf_v):
                rtfs.append(rtf_v)
        except Exception:
            pass
    avg_rtf = (sum(rtfs) / len(rtfs)) if rtfs else float("nan")
    metrics_lines.append(
        f"avg|{format_seconds(avg_ttfa) if math.isfinite(avg_ttfa) else 'nan'}|nan|{format_seconds(avg_rtf) if math.isfinite(avg_rtf) else 'nan'}|nan|nan"
    )

    # Сохранение CSV
    with (out_root / "info.csv").open("w", encoding="utf-8") as f:
        f.write("\n".join(info_lines) + "\n")

    with (out_root / "metrics.csv").open("w", encoding="utf-8") as f:
        f.write("\n".join(metrics_lines) + "\n")

    print(f"\nГотово!")
    print(f"WAV: {out_wavs}")
    print(f"TTFA: {info_csv}")
    print(f"METRICS: {metrics_csv}")
    if math.isfinite(avg_ttfa):
        print(f"Средний TTFA: {format_seconds(avg_ttfa)} c")
    if math.isfinite(avg_rtf):
        print(f"Средний RTF: {format_seconds(avg_rtf)}")


if __name__ == "__main__":
    main()
