"""Игровой движок: действия, матрица выигрышей, множители, класс игры."""
from .action import Action
from .game import DEFAULT_PAYOFF, Game, GameResult
from .multiplier import (
    CappedMultiplier,
    ConstantMultiplier,
    LinearMultiplier,
    LogMultiplier,
    Multiplier,
    PowerMultiplier,
)
from .strategy import Strategy

__all__ = [
    "Action",
    "Strategy",
    "Game",
    "GameResult",
    "DEFAULT_PAYOFF",
    "Multiplier",
    "ConstantMultiplier",
    "LinearMultiplier",
    "PowerMultiplier",
    "LogMultiplier",
    "CappedMultiplier",
]
