"""Класс матча: повторяющаяся PD с компаундингом M(S) и шумом."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .action import Action
from .multiplier import Multiplier
from .strategy import Strategy


# Матрица выигрышей по умолчанию. Индексация: payoff[a, b, i] — выигрыш игрока
# `i` (0=A, 1=B), когда игрок A сыграл действие `a`, игрок B — действие `b`.
# Действия: 0 = COOPERATE, 1 = DEFECT. Канон: CC=3,3; CD=0,5; DC=5,0; DD=1,1.
DEFAULT_PAYOFF: np.ndarray = np.array(
    [
        [[3.0, 3.0], [0.0, 5.0]],  # A=C, B=C; A=C, B=D
        [[5.0, 0.0], [1.0, 1.0]],  # A=D, B=C; A=D, B=D
    ],
    dtype=np.float64,
)


NoiseMode = Literal["implementation", "perception"]


@dataclass
class GameResult:
    """Результат одного матча. Все массивы numpy.

    Длина `S_trajectory` равна T+1: индекс 0 — состояние до игры (S=0),
    индекс T — финальный накопленный счёт.

    `history_a`/`history_b` — *фактически сыгранные* действия (с учётом
    инверсии при шуме `implementation`).
    `perceived_by_a` — то, что A видел у B в opp_history; аналогично для B.
    В режиме `implementation` всегда `perceived_by_a == history_b` и
    `perceived_by_b == history_a` (ошибка видна всем). В режиме
    `perception` они могут отличаться от фактически сыгранного.
    """

    history_a: np.ndarray
    history_b: np.ndarray
    perceived_by_a: np.ndarray
    perceived_by_b: np.ndarray
    raw_payoffs_a: np.ndarray
    raw_payoffs_b: np.ndarray
    multiplied_payoffs_a: np.ndarray
    multiplied_payoffs_b: np.ndarray
    S_trajectory: np.ndarray
    M_trajectory: np.ndarray

    @property
    def total_a(self) -> float:
        return float(self.multiplied_payoffs_a.sum())

    @property
    def total_b(self) -> float:
        return float(self.multiplied_payoffs_b.sum())

    @property
    def total_raw_a(self) -> float:
        return float(self.raw_payoffs_a.sum())

    @property
    def total_raw_b(self) -> float:
        return float(self.raw_payoffs_b.sum())

    @property
    def coop_rate_a(self) -> float:
        if len(self.history_a) == 0:
            return 0.0
        return float((self.history_a == Action.COOPERATE).mean())

    @property
    def coop_rate_b(self) -> float:
        if len(self.history_b) == 0:
            return 0.0
        return float((self.history_b == Action.COOPERATE).mean())


class Game:
    """Один матч повторяющейся PD с компаундингом M(S).

    Механика раунда `t`:

        1. Игроки одновременно выбирают намерения a_int, b_int.
        2. Опционально срабатывает шум (см. `noise_mode`):
           - "implementation": с вероятностью `noise` фактически сыгранное
             действие инвертируется; оба игрока видят инвертированное.
           - "perception": фактически сыгранное равно намерению, но соперник
             с вероятностью `noise` наблюдает инвертированное действие.
        3. Сырые выигрыши r_a, r_b из матрицы по фактически сыгранным
           действиям.
        4. M(t) = multiplier(S(t)). Множенные выигрыши: p_a = r_a * M(t),
           p_b = r_b * M(t).
        5. S(t+1) = S(t) + p_a + p_b.

    Компаундинг получается рекурсивным: будущий множитель зависит от уже
    умноженных выигрышей всех предыдущих раундов.
    """

    def __init__(
        self,
        player_a: Strategy,
        player_b: Strategy,
        rounds: int,
        multiplier: Multiplier,
        noise: float = 0.0,
        noise_mode: NoiseMode = "implementation",
        seed: int | None = None,
        payoff: np.ndarray = DEFAULT_PAYOFF,
    ) -> None:
        if rounds < 0:
            raise ValueError("rounds должно быть >= 0")
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise должен быть в [0, 1]")
        if noise_mode not in ("implementation", "perception"):
            raise ValueError(f"некорректный noise_mode: {noise_mode!r}")
        if payoff.shape != (2, 2, 2):
            raise ValueError(f"ожидалась матрица shape (2,2,2), получено {payoff.shape}")
        self.player_a = player_a
        self.player_b = player_b
        self.rounds = int(rounds)
        self.multiplier = multiplier
        self.noise = float(noise)
        self.noise_mode = noise_mode
        self.seed = seed
        self.payoff = payoff

    def play(self) -> GameResult:
        T = self.rounds
        rng = np.random.default_rng(self.seed)

        history_a = np.zeros(T, dtype=np.int8)
        history_b = np.zeros(T, dtype=np.int8)
        perceived_by_a = np.zeros(T, dtype=np.int8)  # что A видел у B
        perceived_by_b = np.zeros(T, dtype=np.int8)  # что B видел у A
        raw_a = np.zeros(T, dtype=np.float64)
        raw_b = np.zeros(T, dtype=np.float64)
        mul_a = np.zeros(T, dtype=np.float64)
        mul_b = np.zeros(T, dtype=np.float64)
        S = np.zeros(T + 1, dtype=np.float64)
        M = np.zeros(T, dtype=np.float64)

        # Что каждый игрок видит у себя в собственной истории.
        # implementation: фактически сыгранное (включая ошибки).
        # perception:     намерение.
        my_view_a = np.zeros(T, dtype=np.int8)
        my_view_b = np.zeros(T, dtype=np.int8)

        self.player_a.reset()
        self.player_b.reset()

        for t in range(T):
            intended_a = int(
                self.player_a.play(my_view_a[:t], perceived_by_a[:t], t, T)
            )
            intended_b = int(
                self.player_b.play(my_view_b[:t], perceived_by_b[:t], t, T)
            )

            played_a = intended_a
            played_b = intended_b
            shown_a_to_b = intended_a
            shown_b_to_a = intended_b

            if self.noise > 0.0:
                if self.noise_mode == "implementation":
                    if rng.random() < self.noise:
                        played_a = 1 - played_a
                        shown_a_to_b = played_a
                    if rng.random() < self.noise:
                        played_b = 1 - played_b
                        shown_b_to_a = played_b
                else:  # perception
                    if rng.random() < self.noise:
                        shown_a_to_b = 1 - shown_a_to_b
                    if rng.random() < self.noise:
                        shown_b_to_a = 1 - shown_b_to_a

            # Что игроки видят у себя
            if self.noise_mode == "implementation":
                my_view_a[t] = played_a
                my_view_b[t] = played_b
            else:
                my_view_a[t] = intended_a
                my_view_b[t] = intended_b

            perceived_by_a[t] = shown_b_to_a
            perceived_by_b[t] = shown_a_to_b
            history_a[t] = played_a
            history_b[t] = played_b

            # Сырой выигрыш по фактически сыгранным действиям
            r_a = float(self.payoff[played_a, played_b, 0])
            r_b = float(self.payoff[played_a, played_b, 1])
            raw_a[t] = r_a
            raw_b[t] = r_b

            m = float(self.multiplier(S[t]))
            M[t] = m
            mul_a[t] = r_a * m
            mul_b[t] = r_b * m
            S[t + 1] = S[t] + mul_a[t] + mul_b[t]

        return GameResult(
            history_a=history_a,
            history_b=history_b,
            perceived_by_a=perceived_by_a,
            perceived_by_b=perceived_by_b,
            raw_payoffs_a=raw_a,
            raw_payoffs_b=raw_b,
            multiplied_payoffs_a=mul_a,
            multiplied_payoffs_b=mul_b,
            S_trajectory=S,
            M_trajectory=M,
        )
