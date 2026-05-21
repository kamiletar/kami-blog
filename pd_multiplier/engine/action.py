"""Действие игрока в раунде."""
from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """Действие игрока: кооперация или дефекция.

    Реализовано как IntEnum, чтобы значения можно было использовать
    как индексы в numpy-матрице выигрышей без явных преобразований.
    """

    COOPERATE = 0
    DEFECT = 1

    def __str__(self) -> str:
        return "C" if self is Action.COOPERATE else "D"
