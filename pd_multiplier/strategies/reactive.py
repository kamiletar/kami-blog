"""Реактивные стратегии: TfT, Grim, Pavlov, и производные."""
from __future__ import annotations

import numpy as np

from engine.action import Action
from engine.strategy import Strategy


class TitForTat(Strategy):
    """Классический TfT Аксельрода.

    Первый ход — C, далее копирует последний наблюдаемый ход соперника.
    """

    name = "TfT"

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        if round_idx == 0:
            return Action.COOPERATE
        return Action(int(opp_history[-1]))


class SuspiciousTfT(Strategy):
    """TfT с подозрительным первым ходом.

    Первый ход — D, далее копирует последний наблюдаемый ход соперника.
    """

    name = "SuspiciousTfT"

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        if round_idx == 0:
            return Action.DEFECT
        return Action(int(opp_history[-1]))


class TitForTwoTats(Strategy):
    """Дефектирует только если соперник дефектировал в двух последних раундах подряд."""

    name = "TFT2"

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        if round_idx < 2:
            return Action.COOPERATE
        if opp_history[-1] == Action.DEFECT and opp_history[-2] == Action.DEFECT:
            return Action.DEFECT
        return Action.COOPERATE


class GrimTrigger(Strategy):
    """Grim Trigger (Friedman, 1971).

    Кооперирует до первой дефекции соперника, затем дефектирует навсегда.
    Реагирует на наблюдаемые ходы противника: в режиме `perception` одна
    ошибочная перцепция необратимо включает триггер.
    """

    name = "Grim"

    def __init__(self) -> None:
        self._triggered = False

    def reset(self) -> None:
        self._triggered = False

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        if round_idx == 0:
            return Action.COOPERATE
        if not self._triggered and opp_history[-1] == Action.DEFECT:
            self._triggered = True
        return Action.DEFECT if self._triggered else Action.COOPERATE


class Friedman(GrimTrigger):
    """Псевдоним GrimTrigger для совместимости с литературой."""

    name = "Friedman"


class GenerousTfT(Strategy):
    """Великодушный TfT.

    TfT, но прощает дефекцию соперника с вероятностью `forgiveness`:
    вместо ответной дефекции кооперирует.
    """

    name = "GenerousTfT"

    def __init__(self, forgiveness: float = 0.1, seed: int | None = None) -> None:
        if not 0.0 <= forgiveness <= 1.0:
            raise ValueError("forgiveness должен быть в [0, 1]")
        self.forgiveness = float(forgiveness)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # RNG не сбрасывается намеренно (см. Random).
        pass

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        if round_idx == 0:
            return Action.COOPERATE
        if opp_history[-1] == Action.COOPERATE:
            return Action.COOPERATE
        # Соперник предал — прощаем с вероятностью forgiveness
        if self._rng.random() < self.forgiveness:
            return Action.COOPERATE
        return Action.DEFECT


class Pavlov(Strategy):
    """Win-Stay Lose-Shift (Pavlov / Rapoport & Chammah).

    Первый ход — C. Затем:
    - если соперник кооперировал в прошлом раунде → повторить свой ход (stay).
    - если соперник дефектировал → переключиться (shift).

    Это эквивалентно классическому правилу «повторяй, если получил T или R;
    меняй, если получил S или P» при стандартной матрице PD.
    """

    name = "Pavlov"

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        if round_idx == 0:
            return Action.COOPERATE
        last_opp = int(opp_history[-1])
        last_my = int(my_history[-1])
        if last_opp == Action.COOPERATE:
            return Action(last_my)          # stay
        else:
            return Action(1 - last_my)      # shift


class ContriteTfT(Strategy):
    """Контритный TfT (Wu & Axelrod, 1995).

    TfT с механизмом «раскаяния»: если мой последний ход оказался D
    непреднамеренно (намеревался C, но движок сыграл D в режиме
    `implementation`), в следующем раунде прощаю ответную дефекцию соперника
    и кооперирую — не продолжая цикл взаимной мести.

    В режиме `perception` намерение и my_history совпадают, поэтому
    стратегия вырождается в обычный TfT.
    """

    name = "ContriteTfT"

    def __init__(self) -> None:
        self._intentions: list[int] = []

    def reset(self) -> None:
        self._intentions = []

    def play(
        self,
        my_history: np.ndarray,
        opp_history: np.ndarray,
        round_idx: int,
        total_rounds: int,
    ) -> Action:
        if round_idx == 0:
            intent = Action.COOPERATE
        else:
            last_intent = self._intentions[-1]
            last_played = int(my_history[-1])
            # Ошибка реализации: хотел C, но сыграл D
            was_mistake = last_intent == Action.COOPERATE and last_played == Action.DEFECT
            if was_mistake:
                # Раскаиваюсь: прощаю ответную дефекцию соперника
                intent = Action.COOPERATE
            else:
                intent = Action(int(opp_history[-1]))
        self._intentions.append(int(intent))
        return intent
