"""Простые базовые стратегии: AllC, AllD, Random."""
from __future__ import annotations

import numpy as np

from engine.action import Action
from engine.strategy import Strategy


class AllC(Strategy):
    """Всегда кооперирует."""

    name = "AllC"

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        return Action.COOPERATE


class AllD(Strategy):
    """Всегда дефектирует."""

    name = "AllD"

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        return Action.DEFECT


class Random(Strategy):
    """Кооперирует с вероятностью `p`, дефектирует с вероятностью `1-p`.

    RNG не сбрасывается при `reset()` — случайная последовательность продолжается
    сквозь все матчи, что обеспечивает разнообразие при повторных играх в турнире.
    """

    name = "Random"

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p должен быть в [0, 1]")
        self.p = float(p)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # RNG намеренно не сбрасывается: каждый матч должен давать разные результаты.
        pass

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        return Action.COOPERATE if self._rng.random() < self.p else Action.DEFECT
