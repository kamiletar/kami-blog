"""Общие утилиты для экспериментов: турнирный движок, агрегация результатов."""
from __future__ import annotations

from itertools import product
from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from engine.game import Game
from engine.multiplier import Multiplier
from strategies import StrategyFactory


class MatchRecord(NamedTuple):
    """Итог одного матча в турнире."""

    multiplier_name: str
    strat_a: str
    strat_b: str
    repetition: int
    total_a: float
    total_b: float
    coop_rate_a: float
    coop_rate_b: float
    rounds: int


def run_round_robin(
    zoo: list[tuple[str, StrategyFactory]],
    multiplier: Multiplier,
    rounds: int,
    repetitions: int,
    noise: float = 0.0,
    noise_mode: str = "implementation",
    base_seed: int = 42,
) -> list[MatchRecord]:
    """Провести round-robin турнир.

    Каждый с каждым, включая self-play (A vs A — два независимых экземпляра).
    Повторяется `repetitions` раз с детерминированными seed-ами.
    Возвращает список MatchRecord.
    """
    rng = np.random.default_rng(base_seed)
    match_seeds = rng.integers(0, 2**31 - 1, size=repetitions * len(zoo) ** 2).tolist()
    seed_iter = iter(match_seeds)

    records: list[MatchRecord] = []
    pairs = list(product(range(len(zoo)), range(len(zoo))))

    for rep in tqdm(range(repetitions), desc=f"{multiplier.name}", leave=False):
        for i, j in pairs:
            name_a, factory_a = zoo[i]
            name_b, factory_b = zoo[j]
            seed = int(next(seed_iter))
            result = Game(
                factory_a(),
                factory_b(),
                rounds=rounds,
                multiplier=multiplier,
                noise=noise,
                noise_mode=noise_mode,  # type: ignore[arg-type]
                seed=seed,
            ).play()
            records.append(
                MatchRecord(
                    multiplier_name=multiplier.name,
                    strat_a=name_a,
                    strat_b=name_b,
                    repetition=rep,
                    total_a=result.total_a,
                    total_b=result.total_b,
                    coop_rate_a=result.coop_rate_a,
                    coop_rate_b=result.coop_rate_b,
                    rounds=rounds,
                )
            )
    return records


def records_to_df(records: list[MatchRecord]) -> pd.DataFrame:
    return pd.DataFrame(records)


def strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Суммарные метрики по стратегии.

    Считаем с точки зрения игрока A: стратегия выступает в роли A во всех
    матчах, где она записана как strat_a. Это даёт «личный» средний результат
    по всем соперникам за все повторения.
    """
    per_match = (
        df.groupby(["multiplier_name", "strat_a", "repetition"])
        .agg(
            match_total=("total_a", "mean"),
            match_coop=("coop_rate_a", "mean"),
        )
        .reset_index()
    )
    summary = (
        per_match.groupby(["multiplier_name", "strat_a"])
        .agg(
            mean_total=("match_total", "mean"),
            std_total=("match_total", "std"),
            mean_coop=("match_coop", "mean"),
        )
        .reset_index()
        .rename(columns={"strat_a": "strategy"})
    )
    summary["rank"] = summary.groupby("multiplier_name")["mean_total"].rank(
        ascending=False, method="min"
    )
    return summary.sort_values(["multiplier_name", "rank"])
