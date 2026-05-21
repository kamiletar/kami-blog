"""Зоопарк стратегий для турниров."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from engine.strategy import Strategy

from .basic import AllC, AllD, Random
from .reactive import (
    ContriteTfT,
    Friedman,
    GenerousTfT,
    GrimTrigger,
    Pavlov,
    SuspiciousTfT,
    TitForTat,
    TitForTwoTats,
)

__all__ = [
    "AllC",
    "AllD",
    "Random",
    "TitForTat",
    "SuspiciousTfT",
    "TitForTwoTats",
    "GrimTrigger",
    "Friedman",
    "GenerousTfT",
    "ContriteTfT",
    "Pavlov",
    "default_zoo",
]


StrategyFactory = Callable[[], Strategy]


def default_zoo(seed: int | None = None) -> list[tuple[str, StrategyFactory]]:
    """Стандартный набор стратегий для турниров.

    Каждый элемент — пара (имя, фабрика). Фабрика — callable без аргументов,
    возвращающий свежий экземпляр стратегии. Стохастические стратегии получают
    независимые подпорождающие seed-ы от мастер-`seed`-а, чтобы вся симуляция
    оставалась воспроизводимой.

    Friedman не включаем — это псевдоним GrimTrigger, чтобы не дублировать
    данные в турнире.
    """
    rng = np.random.default_rng(seed)
    sub_seeds = rng.integers(0, 2**31 - 1, size=8).tolist()

    return [
        ("AllC", lambda: AllC()),
        ("AllD", lambda: AllD()),
        ("Random", lambda s=int(sub_seeds[0]): Random(p=0.5, seed=s)),
        ("TfT", lambda: TitForTat()),
        ("SuspiciousTfT", lambda: SuspiciousTfT()),
        ("TFT2", lambda: TitForTwoTats()),
        ("Grim", lambda: GrimTrigger()),
        (
            "GenerousTfT",
            lambda s=int(sub_seeds[1]): GenerousTfT(forgiveness=0.1, seed=s),
        ),
        ("ContriteTfT", lambda: ContriteTfT()),
        ("Pavlov", lambda: Pavlov()),
    ]
