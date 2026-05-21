"""Базовый класс стратегии."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .action import Action


class Strategy(ABC):
    """Базовый класс стратегии в повторяющейся PD.

    Атрибут `name` — короткое отображаемое имя стратегии (используется в
    легендах графиков и заголовках CSV; желательно без пробелов).

    Контракт:
        `reset()` вызывается перед каждой новой игрой — стратегия должна
        очистить любое внутреннее состояние (триггеры, накопители).

        `play(...)` вызывается на каждом раунде и возвращает Action.
    """

    name: str = "abstract"

    def reset(self) -> None:
        """Сбросить внутреннее состояние перед новой игрой.

        Базовая реализация — no-op. Стратегии без состояния могут не
        переопределять. RNG стохастических стратегий принято НЕ сбрасывать
        здесь: он инициализируется в `__init__` и продолжает работу через
        несколько матчей (см. комментарии в конкретных классах).
        """
        return None

    @abstractmethod
    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        """Выбрать действие в раунде `round_idx`.

        Аргументы:
            my_history: что игрок видит в собственной истории до этого раунда,
                shape (round_idx,), dtype int. Что именно сюда попадает,
                зависит от `noise_mode` движка: при `implementation` это
                фактически сыгранные действия (включая ошибки), при
                `perception` — намерения.
            opp_history: что игрок наблюдал у соперника, shape (round_idx,).
            round_idx: индекс текущего раунда, 0-based.
            total_rounds: общее число раундов в матче.

        Возвращает: Action.
        """
        ...

    def __repr__(self) -> str:
        return self.name
