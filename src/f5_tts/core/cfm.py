# -*- coding: utf-8 -*-
"""
CFM (Conditional Flow Matching) — инференс/обучение для F5-TTS.

Обозначения размерностей (ein-notation):
  b   — размер батча
  n   — длина последовательности мел-кадров (time frames)
  nt  — длина текстовой последовательности (в токенах)
  nw  — длина сырого аудио (в отсчётах)
  d   — число мел-каналов (mel bins), обычно 100

Идея:
  • На обучении: берём реальную мел-спектрограмму x1, шум x0 и случайный t∈[0,1],
    строим точку φ_t = (1−t)*x0 + t*x1 и учим сеть предсказывать «скорость» v* = x1 − x0
    (линейная геодезика flow matching). Лосс считаем только на замаскированном фрагменте
    (инфиллинг).
  • На инференсе: решаем ОДУ dx/dt = f_θ(t, x | cond, text) по небольшой сетке t
    (NFE≈16–32), где cond — «замороженный» аудиоконтекст (промпт), text — токены.
    Используем EPSS/Sway для удачного выбора шагов по времени и CFG для усиления
    подчинения тексту.
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
from torch import nn
from torchdiffeq import odeint
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# В проекте это внутренние модули:
from .modules import MelSpec
from .utils import (
    default,             # default(x, y) -> x if exists(x) else y
    exists,              # exists(x)     -> x is not None
    get_epss_timesteps,  # эмпирически подобранная сетка времени для малого NFE
    lens_to_mask,        # [b] -> [b, n] bool-паддинг маска по длинам
    list_str_to_idx,     # токенизация списка строк по словарю
    mask_from_frac_lengths,  # одна сплошная «дыра» в каждом примере по доле длины
)


class CFM(nn.Module):
    """
    Conditional Flow Matching обёртка вокруг трансформера (DiT).

    Параметры
    ---------
    transformer : nn.Module
        Бэкбон, который предсказывает векторное поле (скорость) по текущему состоянию x,
        времени t и условиям (cond, text). В F5 — DiT с режимами drop/cfg_infer/cache.
    odeint_kwargs : dict
        Аргументы в torchdiffeq.odeint. По умолчанию method="euler" (быстро и стабильно).
        Можно поставить "midpoint" (2-й порядок), будет медленнее.
    audio_drop_prob : float in [0,1]
        Вероятность вырубить аудио-контекст на обучении (часть CFG-тренировки).
    cond_drop_prob : float in [0,1]
        Вероятность сделать полностью uncond-пример (и аудио, и текст дропнуты).
    num_channels : int | None
        Число мел-каналов (d). Если None — берётся из MelSpec.n_mel_channels (обычно 100).
    mel_spec_module : nn.Module | None
        Готовый модуль мел-спектрограммы. Если None — создадим MelSpec(**mel_spec_kwargs).
    mel_spec_kwargs : dict
        Аргументы для MelSpec (sr, n_mels, hop_length и т.п.), если mel_spec_module не задан.
    frac_lengths_mask : tuple[float,float]
        Диапазон доли длины для «дырок» на обучении, напр. (0.7, 1.0) → скрываем 70–100% длины.
    vocab_char_map : dict[str,int] | None
        Словарь для токенизации текста. Если None — fallback (posymvol’no/байты).
    """

    def __init__(
        self,
        transformer: nn.Module,
        odeint_kwargs: dict = dict(method="euler"),  # можно: {'method': 'midpoint'}
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        num_channels: int | None = None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str, int] | None = None,
    ):
        super().__init__()

        # --- параметры инфиллинга/масок на обучении ---
        self.frac_lengths_mask = frac_lengths_mask

        # --- мел-спектрограмма (wav -> mel) ---
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels) # type: ignore
        self.num_channels = num_channels  # d

        # --- CFG-тренировка: вероятности дропа аудио/текста ---
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # --- бэкбон (DiT) ---
        self.transformer = transformer
        self.dim = getattr(self.transformer, "dim", None)  # скрытая размерность модели (для справки)

        # --- параметры интегратора ОДУ ---
        self.odeint_kwargs = odeint_kwargs

        # --- токенизация ---
        self.vocab_char_map = vocab_char_map

    # Удобно: текущий девайс модуля (GPU/CPU)
    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,                 # [b, n, d] мел или [b, nw] сырое аудио (будет конвертировано)
        text: torch.Tensor | list[str],     # [b, nt] ids или список из b строк
        duration: int | torch.Tensor,       # целевая длина(ы) в кадрах: int или [b]
        *,
        lens: torch.Tensor | None = None,   # [b], реальные длины промпта (в кадрах)
        steps: int = 32,                    # NFE: число шагов интегратора
        cfg_strength: float = 1.0,          # вес CFG на инференсе (0 → без CFG; 1..2 — типично)
        sway_sampling_coef: float | None = None,  # <0 → больше шагов в начале (смещаем сетку t)
        seed: int | None = None,            # сид для воспроизводимости шума
        max_duration: int = 4096,           # жёсткий потолок длины (безопасность/VRAM)
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,  # [b,d,n] -> [b,nw]
        use_epss: bool = True,              # применять EPSS (готовая сетка времени) при t_start=0
        no_ref_audio: bool = False,         # игнорировать аудиопромпт целиком (cond -> 0)
        duplicate_test: bool = False,       # диагностический старт не с нуля, а «ближе к cond»
        t_inter: float = 0.1,               # сколько «сразу пройти» (0.1 → старт не из чистого шума)
        edit_mask: torch.Tensor | None = None,  # доп. маска для частичного редактирования
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Инференс: решаем ОДУ от шума к данным с фиксированным аудиоконтекстом (promt cond) и текстом.

        Возвращает
        ----------
        out : torch.Tensor
            Если vocoder is None  → мел [b, n, d] (с «пришитым» промптом слева).
            Если vocoder задан   → wav [b, nw].
        trajectory : torch.Tensor
            Вся траектория интегрирования: [(steps+1), b, n_max, d].
        """

        self.eval()

        # --- 1) Приведение cond к мел-формату [b, n, d] ---
        if cond.ndim == 2:                            # пришёл wav [b, nw]
            cond = self.mel_spec(cond)                # -> [b, d, n] # type: ignore
            cond = cond.permute(0, 2, 1)              # -> [b, n, d]

            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype) # выравниваем dtype с моделью (fp16/bf16/fp32)

        batch, cond_seq_len = cond.shape[0], cond.shape[1]
        device = cond.device

        # --- 2) Реальные длины промпта (сколько слева считаем «контекстом») ---
        if not exists(lens):
            # Если не задано: считаем, что весь cond — реальный промпт
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # --- 3) Текст: ids или список строк ---
        if isinstance(text, list):
            # список из b строк → токенизируем
            text = list_str_to_idx(text, self.vocab_char_map).to(device)   # [b, nt] # type: ignore
            
            assert text.shape[0] == batch, "len(text_list) должен равняться batch size" # type: ignore

        # --- 4) Маска промпта cond_mask: True на [0: lens[i]) ---
        cond_mask = lens_to_mask(lens)  # [b, n_cond] (пока без паддинга) # type: ignore
        if edit_mask is not None:
            # Если редактируем кусок — ограничиваем, где можно читать контекст
            cond_mask = cond_mask & edit_mask

        # --- 5) duration: целевые длины в кадрах (по каждому элементу батча) ---
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        # Нижняя граница: покрыть весь текст и весь промпт + 1 кадр для генерации
        # (text может содержать -1 как паддинг; считаем только реальные токены)
        min_need = torch.maximum((text != -1).sum(dim=-1), lens) + 1 # type: ignore
        duration = torch.maximum(min_need, duration)
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()  # общая длина для паддинга внутри батча # type: ignore

        # --- 6) (опционально) режим диагностики: взять cond в середину, стартовать ближе к нему ---
        if duplicate_test:
            test_cond = F.pad(
                cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0
            )  # [b, n_max, d] cond "впаян" в середину

        # --- 7) Паддинг cond справа до max_duration; no_ref_audio - обнулить контекст полностью ---
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)  # -> [b, n_max, d]
        if no_ref_audio:
            cond = torch.zeros_like(cond)  # полностью игнорируем аудиопромпт

        # --- 8) Паддинг cond_mask до n_max и перевод к [b, n_max, 1] ---
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)  # [b, n_max] # type: ignore
        cond_mask = cond_mask.unsqueeze(-1)                                                 # [b, n_max, 1]

        # step_cond: на кадрах промпта — реальные значения, вне — ноль
        # Это фиксированная "маскированная" мел, которую подаём МОДЕЛИ на каждом шаге ОДУ.
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))  # [b, n_max, d]

        # --- 9) Attention/паддинг маска по целевой длине (для батча>1) ---
        if batch > 1:
            mask = lens_to_mask(duration)  # [b, n_max] # type: ignore
        else:
            mask = None  # экономим: для b=1 паддинг-маска не нужна

        # --- 10) Правая часть ОДУ: v(t, x) с поддержкой CFG ---
        def fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """
            f(t, x) = предсказанная моделью «скорость»/векторное поле на текущем шаге интегрирования.
            Здесь:
              • x — текущее «состояние» мел [b, n_max, d]
              • t — скалярное время (batched-скаляр)
              • step_cond, text, mask — фиксированы на протяжении всей траектории
            """
            if cfg_strength < 1e-5:
                # Без CFG: один условный прогон
                return self.transformer(
                    x=x, cond=step_cond, text=text, time=t,
                    mask=mask, drop_audio_cond=False, drop_text=False,
                    cache=True,          # включаем KV-кеш для ускорения последовательных шагов
                )

            # С CFG: трансформер внутри делает два прогона (cond/uncond) и склеивает по batch
            pred_cfg = self.transformer(
                x=x, cond=step_cond, text=text, time=t,
                mask=mask, cfg_infer=True,   # "сделай cond и uncond и верни [2b, ...]"
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)  # [b, n, d] и [b, n, d]
            return pred + (pred - null_pred) * cfg_strength    # классическая формула CFG

        # --- 11) Начальное состояние y0: шум N(0,1) под КАЖДУЮ длину в батче ---
        # Важно: для воспроизводимости одиночного вызова seed задаётся перед каждым элементом,
        # но для батча это даёт один и тот же шум на всех элементах — если нужно иначе,
        # заведите torch.Generator(device=...) и randn(..., generator=gen).
        y0_list: list[torch.Tensor] = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)

            y0_list.append(torch.randn(
                int(dur.item()) if isinstance(dur, torch.Tensor) else int(dur),
                self.num_channels, device=self.device, dtype=step_cond.dtype # type: ignore
            ))  # [dur_i, d]

        y0 = pad_sequence(y0_list, padding_value=0.0, batch_first=True)  # -> [b, n_max, d]

        # --- 12) Начальная точка по времени (t_start) и диагностический "смещённый" старт ---
        t_start = 0.0
        if duplicate_test:
            t_start = float(t_inter)            # например, 0.1
            y0 = (1 - t_start) * y0 + t_start * test_cond  # старт ближе к cond
            steps = int(steps * (1 - t_start))  # корректируем число шагов

        # --- 13) Сетка времени t: EPSS vs равномерная ---
        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)  # [(steps+1)]
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)

        # --- 14) Sway Sampling: деформируем сетку, смещая плотность шагов к ранним временам ---
        if sway_sampling_coef is not None:
            # t <- t + c * (cos(pi/2 * t) - 1 + t); при c<0 будет больше шагов в начале
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # --- 15) Решаем ОДУ (Euler / Midpoint / пр.), кеш внутри бэкбона ускоряет шаги ---
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)  # [(steps+1), b, n_max, d]
        self.transformer.clear_cache()  # вручную сбрасываем KV-кеш после интегрирования # type: ignore

        # --- 16) Берём финальное состояние и жёстко «пришиваем» промпт слева ---
        sampled = trajectory[-1]                                # [b, n_max, d]
        out = torch.where(cond_mask, cond, sampled)             # контекст остаётся ровно как во входе # type: ignore

        # --- 17) (опционально) Вокодер mel->wav ---
        if exists(vocoder):
            out = out.permute(0, 2, 1)  # [b, d, n]
            out = vocoder(out)          # [b, nw] # type: ignore

        return out, trajectory # type: ignore

    def forward(
        self,
        inp: torch.Tensor,                      # [b, n, d] мел или [b, nw] wav
        text: torch.Tensor | list[str],         # [b, nt] ids или список строк
        *,
        lens: torch.Tensor | None = None,       # [b], реальные длины (для построения масок)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Обучение: Conditional Flow Matching на линейной траектории с инфиллинг-маской.

        Шаги:
          1) inp -> mel (если пришёл wav), получаем x1.
          2) Сэмплируем шум x0 и t∈[0,1], строим φ_t = (1−t)*x0 + t*x1 и целевое поле flow = x1 − x0.
          3) Сэмплируем «дыру» rand_span_mask (непрерывная), формируем cond: контекст = x1 вне дыры, внутри — 0.
          4) Прогоняем трансформер (с вероятностями дропа аудио/текста для CFG-тренировки).
          5) MSE(pred, flow) считаем ТОЛЬКО внутри «дыры».

        Возвращает
        ----------
        loss : torch.Tensor
            Скаляр: средний MSE на маске инфиллинга.
        cond : torch.Tensor
            Маскированный контекст (X_m) — для отладки.
        pred : torch.Tensor
            Предсказанное моделью поле (скорость) на φ_t.
        """

        # --- 1) Приведение входа к mel [b, n, d] ---
        if inp.ndim == 2:                          # [b, nw] → wav
            inp = self.mel_spec(inp)               # [b, d, n] # type: ignore
            inp = inp.permute(0, 2, 1)             # [b, n, d]
            assert inp.shape[-1] == self.num_channels

        batch, seq_len = inp.shape[0], inp.shape[1]
        dtype, device = inp.dtype, self.device

        # --- 2) Текст: ids или список строк ---
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device) # type: ignore
            else:
                text = list_str_to_tensor(text).to(device) # type: ignore
            assert text.shape[0] == batch # type: ignore

        # --- 3) Длины и общая паддинг-маска ---
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)  # [b, n] (collate обычно и так паддит до max) # type: ignore

        # --- 4) Случайная «дыра» по доле длины для инфиллинга ---
        frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)  # [b, n] True на «дырке» # type: ignore
        if exists(mask):
            rand_span_mask &= mask

        # --- 5) Данные x1 и шум x0 ---
        x1 = inp                                        # [b, n, d]
        x0 = torch.randn_like(x1)                       # [b, n, d]

        # --- 6) Случайный момент времени t ∈ [0,1] ---
        time = torch.rand((batch,), dtype=dtype, device=device)  # [b]

        # --- 7) Точка на линейной траектории и целевое поле ---
        t = time.unsqueeze(-1).unsqueeze(-1)            # [b, 1, 1]
        phi = (1 - t) * x0 + t * x1                    # φ_t(x)  — [b, n, d]
        flow = x1 - x0                                  # v*      — [b, n, d] (константа по t)

        # --- 8) Маскированный аудиоконтекст cond: вне дыры — реальная мел, внутри — нули ---
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)  # [b, n, d]

        # --- 9) CFG-тренировка: иногда выключаем аудио и/или текст ---
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop (Voicebox)
        if random() < self.cond_drop_prob:                 # p_uncond → полностью uncond
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # --- 10) Прогон бэкбона: предсказываем поле на φ_t ---
        pred = self.transformer(
            x=phi, cond=cond, text=text, time=time,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text,
            mask=mask,  # паддинг-маска по длинам
        )  # [b, n, d]

        # --- 11) Лосс: MSE только на «дырке» (in-filling supervision) ---
        loss_map = F.mse_loss(pred, flow, reduction="none")  # [b, n, d]
        loss_on_hole = loss_map[rand_span_mask]              # выбираем только дырки → [K, d]
        loss = loss_on_hole.mean()                           # скаляр

        return loss, cond, pred
