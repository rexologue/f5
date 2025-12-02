# bench_f5tts_multi_speakers_gigaam_ctc.py
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import tqdm
from ruaccent import RUAccent

# Подправь импорт под реальное расположение класса F5TTS
from src.f5tts import F5TTS  # например: from src.api import F5TTS


############################
# Text & speakers utilities
############################

def read_texts(texts_path: Path) -> list[str]:
    texts: list[str] = []
    with texts_path.open("r", encoding="utf-8") as f:
        for raw in f:
            t = raw.strip()
            if t:
                texts.append(t)
    return texts


@dataclass
class SpeakerRef:
    dir: Path
    wav: Path
    txt: Path
    text: str


def discover_speakers(refs_root: Path) -> list[SpeakerRef]:
    if not refs_root.exists() or not refs_root.is_dir():
        raise FileNotFoundError(f"Папка с референсами не найдена: {refs_root}")

    speakers: list[SpeakerRef] = []
    for sub in sorted([p for p in refs_root.iterdir() if p.is_dir()]):
        wav = sub / "ref.mp3"  # по условию — именно ref.mp3
        txt = sub / "ref.txt"
        if wav.exists() and txt.exists():
            ref_text = txt.read_text(encoding="utf-8").strip()
            if not ref_text:
                raise ValueError(f"Пустой ref.txt: {txt}")
            speakers.append(SpeakerRef(dir=sub, wav=wav, txt=txt, text=ref_text))

    if not speakers:
        raise RuntimeError(f"В {refs_root} нет подпапок с ref.mp3 и ref.txt")
    return speakers


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


#########################
# CER (character-level)
#########################

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def normalize_text(s: str, keep_spaces: bool = False) -> str:
    s = s.lower()
    s = _strip_accents(s)
    if keep_spaces:
        s = re.sub(r"[^a-zа-яё0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
    else:
        s = re.sub(r"[^a-zа-яё0-9]+", "", s)
    return s

def cer(ref: str, hyp: str, keep_spaces: bool = False) -> float:
    r = normalize_text(ref, keep_spaces=keep_spaces)
    h = normalize_text(hyp, keep_spaces=keep_spaces)
    if not r:
        return float("nan")

    # Левенштейн по символам
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,        # вставка
                dp[j - 1] + 1,    # удаление
                prev + cost       # замена/совпадение
            )
            prev = tmp
    return dp[m] / max(1, n)


############################
# GigaAM-CTC ASR wrapper
############################

class GigaAMCTC:
    """
    Обёртка над GigaAM ASR.
    model_type: "ctc", "v2_ctc", "rnnt", "v2_rnnt" — нам нужен CTC.
    """
    def __init__(self, model_type: str = "v2_ctc"):
        try:
            import gigaam  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Не найден пакет gigaam. Установи из репозитория salute-developers/GigaAM."
            ) from e
        self._gigaam = gigaam
        self.model = self._gigaam.load_model(model_type)

    def transcribe(self, wav_path: Path) -> str:
        # По README API: model.transcribe(audio_path) -> str
        # Короткие аудио (до ~30 c). Для наших синтов это подходит.
        text = self.model.transcribe(str(wav_path))
        return (text or "").strip()


def get_existing_indices(meta_csv: Path, csv_delim: str) -> Set[int]:
    """Получить множество уже обработанных индексов из metadata.csv"""
    existing_indices = set()
    if not meta_csv.exists():
        return existing_indices
    
    try:
        with meta_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=csv_delim)
            for row in reader:
                audio_path = row.get("audio_path", "")
                # Извлекаем индекс из имени файла: data/000001.wav -> 1
                match = re.search(r"/(\d+)\.wav$", audio_path)
                if match:
                    existing_indices.add(int(match.group(1)))
    except Exception as e:
        print(f"Предупреждение: не удалось прочитать существующий metadata.csv: {e}")
    
    return existing_indices


def write_metadata_row(meta_csv: Path, row_data: list, csv_delim: str, write_header: bool = False):
    """Записать одну строку в metadata.csv"""
    with meta_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=csv_delim)
        if write_header and f.tell() == 0:
            writer.writerow(["audio_path", "text", "text_accent", "speaker", "cer"])
        writer.writerow(row_data)


############################
# Main
############################

def main():
    parser = argparse.ArgumentParser(
        description="F5TTS → генерация по спикерам, ASR = GigaAM-CTC, метрика CER. Деление текстов поровну между спикерами."
    )

    # Модель синтеза
    parser.add_argument("--config", type=Path, required=True, help="Путь к config.yaml (F5TTS)")
    parser.add_argument("--vocab", type=Path, required=True, help="Путь к vocab (F5TTS)")
    parser.add_argument("--ckpt", type=Path, required=True, help="Путь к checkpoint (F5TTS)")
    parser.add_argument("--vocoder", type=Path, required=True, help="Локальный путь к вокодеру (F5TTS)")
    parser.add_argument("--device", type=str, default=None, help="cuda|cpu|mps|xpu для F5TTS")

    # Данные
    parser.add_argument("--refs-root", type=Path, required=True, help="Папка со спикерами: */ref.mp3 и */ref.txt")
    parser.add_argument("--texts", type=Path, required=True, help="Файл с текстами, по строке на пример")
    parser.add_argument("--speaker-idx", type=int, required=True, help="Номер спикера (1-based) из обнаруженных")
    parser.add_argument("--output", type=Path, required=True, help="Папка для результатов")

    # Генерация (F5TTS)
    parser.add_argument("--no-deepspeed", action="store_true", help="Отключить DeepSpeed kernel injection")
    parser.add_argument("--nfe-step", type=int, default=16, help="nfe_step для infer()")
    parser.add_argument("--cfg-strength", type=float, default=2.0, help="cfg_strength для infer()")
    parser.add_argument("--speed", type=float, default=1.0, help="Скорость речи (time-stretch)")
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0, help="sway_sampling_coef")
    parser.add_argument("--fix-duration", type=float, default=None, help="Фикс. длительность (сек) или None")
    parser.add_argument("--remove-silence", action="store_true", help="Удалять тишину на выходе")
    parser.add_argument("--seed", type=int, default=None, help="Сид для повторяемости")

    # CER / GigaAM
    parser.add_argument("--gigaam-variant", type=str, default="v2_ctc", choices=["ctc", "v2_ctc"], help="Какой CTC вариант GigaAM использовать")
    parser.add_argument("--cer-keep-spaces", action="store_true", help="CER с учётом пробелов")
    parser.add_argument("--csv-delim", type=str, default="|", help="Разделитель для metadata.csv (по умолчанию запятая)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число текстов выбранного спикера")
    parser.add_argument("--show-refs", action="store_true", help="Показать список обнаруженных спикеров и выйти")

    args = parser.parse_args()

    # 1) Спикеры
    speakers = discover_speakers(args.refs_root)
    n_spk = len(speakers)
    if args.show_refs:
        print("Найденные спикеры (1-based):")
        for i, sp in enumerate(speakers, start=1):
            print(f"{i:3d}: {sp.dir.name} | {sp.wav.name}, {sp.txt.name}")
        return

    if args.speaker_idx < 1 or args.speaker_idx > n_spk:
        raise ValueError(f"--speaker-idx вне диапазона 1..{n_spk}")

    # 2) Тексты (без перемешивания)
    all_texts = read_texts(args.texts)
    if not all_texts:
        raise RuntimeError(f"Файл {args.texts} пуст")

    per_spk = len(all_texts) // n_spk
    if per_spk == 0:
        raise RuntimeError(f"Текстов меньше, чем спикеров: {len(all_texts)} < {n_spk}")

    usable = all_texts[: per_spk * n_spk]
    idx0 = args.speaker_idx - 1
    start = idx0 * per_spk
    end = start + per_spk
    spk_texts = usable[start:end]
    if args.limit is not None:
        spk_texts = spk_texts[: max(0, args.limit)]
    if not spk_texts:
        raise RuntimeError("Для выбранного спикера не осталось текстов (после limit/деления)")

    # 3) Подготовка путей
    out_root = args.output
    out_data = out_root / "data"
    ensure_dir(out_data)
    meta_csv = out_root / "metadata.csv"

    # 4) Проверка уже сгенерированных примеров
    existing_indices = get_existing_indices(meta_csv, args.csv_delim)
    if existing_indices:
        print(f"Найдено {len(existing_indices)} уже обработанных примеров. Пропускаем их.")

    # 5) Акцент и модели
    accentizer = RUAccent()
    accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)

    tts = F5TTS(
        config_path=args.config,
        vocab_file=args.vocab,
        ckpt_file=args.ckpt,
        vocoder_local_path=args.vocoder,
        ode_method="euler",
        use_ema=True,
        device=args.device
    )

    asr = GigaAMCTC(model_type=args.gigaam_variant)
    speaker = speakers[idx0]
    ref_text_val = accentizer.process_all(speaker.text)

    # 6) Генерация + CER с инкрементальной записью
    processed_count = 0
    pbar = tqdm.tqdm(enumerate(spk_texts, start=1), total=len(spk_texts),
                     desc=f"Spk {args.speaker_idx}/{n_spk} ({speaker.dir.name})", dynamic_ncols=True)

    for i, gen_text in pbar:
        # Пропускаем уже обработанные
        if i in existing_indices:
            pbar.set_postfix_str("пропуск (уже обработан)")
            continue

        file_stem = f"{i:06d}"
        wav_path = out_data / f"{file_stem}.wav"
        audio_rel = f"data/{file_stem}.wav"
        seed = args.seed if args.seed is not None else i

        in_text = accentizer.process_all(gen_text)

        try:
            _wav, _sr, _spec = tts.infer(
                ref_file=speaker.wav,
                ref_text=ref_text_val,
                gen_text=in_text,
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
        except Exception as e:
            # Записываем ошибку в metadata
            write_metadata_row(
                meta_csv, 
                [audio_rel, gen_text, in_text, speaker.dir.name, "nan"], 
                args.csv_delim,
                write_header=(processed_count == 0 and not existing_indices)
            )
            processed_count += 1
            pbar.set_postfix_str(f"gen err: {e}")
            continue

        try:
            hyp = asr.transcribe(wav_path)
            value = cer(gen_text, hyp, keep_spaces=args.cer_keep_spaces) if hyp else float("nan")
            cer_str = f"{value:.4f}" if math.isfinite(value) else "nan"
        except Exception as e:
            value = float("nan")
            cer_str = "nan"
            pbar.set_postfix_str(f"asr err: {e}")

        # Записываем результат в metadata
        write_metadata_row(
            meta_csv, 
            [audio_rel, gen_text, in_text, speaker.dir.name, cer_str], 
            args.csv_delim,
            write_header=(processed_count == 0 and not existing_indices)
        )
        processed_count += 1

        if math.isfinite(value):
            pbar.set_postfix_str(f"cer={value:.3f}")
        else:
            pbar.set_postfix_str("cer=nan")

    print("\nГотово!")
    print(f"Сгенерированные WAV: {out_data}")
    print(f"Metadata: {meta_csv}")
    print(f"Спикеров обнаружено: {n_spk} (на спикера текстов: {per_spk}, использовано для выбранного: {len(spk_texts)})")
    print(f"Обработано примеров в этом запуске: {processed_count}")
    if existing_indices:
        print(f"Пропущено (уже обработаны ранее): {len(existing_indices)}")
    if len(all_texts) % n_spk != 0:
        print(f"Предупреждение: отброшено {len(all_texts) - per_spk * n_spk} текст(а/ов), чтобы сохранить равное распределение.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        sys.exit(130)
