"""Функции-множители M(S) для компаундинга выигрыша по накопленному счёту."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class Multiplier(ABC):
    """Абстрактный множитель M(S).

    Подклассы должны реализовать `__call__(S) -> float`.
    Поле `name` используется в логах и заголовках графиков.
    """

    name: str = "abstract"

    @abstractmethod
    def __call__(self, S: float) -> float:
        """Значение множителя для накопленного общего счёта `S`."""
        ...

    def __repr__(self) -> str:
        return self.name


class ConstantMultiplier(Multiplier):
    """`M(S) = c`. При `c=1` восстанавливает классическую PD (контрольный случай)."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = float(c)
        self.name = f"Constant({c:g})"

    def __call__(self, S: float) -> float:
        return self.c


class LinearMultiplier(Multiplier):
    """`M(S) = 1 + S / k`. Растёт без ограничений."""

    def __init__(self, k: float) -> None:
        if k <= 0:
            raise ValueError("k должно быть > 0")
        self.k = float(k)
        self.name = f"Linear({k:g})"

    def __call__(self, S: float) -> float:
        return 1.0 + S / self.k


class PowerMultiplier(Multiplier):
    """`M(S) = (1 + S / k) ** alpha`.

    При `alpha=1` совпадает с Linear. При `0 < alpha < 1` — сублинейный рост,
    при `alpha > 1` — суперлинейный.
    """

    def __init__(self, k: float, alpha: float) -> None:
        if k <= 0:
            raise ValueError("k должно быть > 0")
        self.k = float(k)
        self.alpha = float(alpha)
        self.name = f"Power({k:g},{alpha:g})"

    def __call__(self, S: float) -> float:
        return (1.0 + S / self.k) ** self.alpha


class LogMultiplier(Multiplier):
    """`M(S) = 1 + log(1 + S / k)`. Медленный, всегда возрастающий."""

    def __init__(self, k: float) -> None:
        if k <= 0:
            raise ValueError("k должно быть > 0")
        self.k = float(k)
        self.name = f"Log({k:g})"

    def __call__(self, S: float) -> float:
        return 1.0 + math.log1p(S / self.k)


class CappedMultiplier(Multiplier):
    """`M(S) = min(inner(S), cap)`. Обёртка над любым внутренним множителем."""

    def __init__(self, inner: Multiplier, cap: float) -> None:
        if cap <= 0:
            raise ValueError("cap должно быть > 0")
        self.inner = inner
        self.cap = float(cap)
        self.name = f"Capped({inner.name},cap={cap:g})"

    def __call__(self, S: float) -> float:
        return min(self.inner(S), self.cap)
