"""Эксперимент 3: чувствительность к горизонту.

Linear(100), без шума, T in {10, 50, 100, 200, 500, 1000},
100 повторений. Ищем T* — точку, где кооперативные стратегии
обгоняют дефектные по среднему выигрышу.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from engine.multiplier import LinearMultiplier
from experiments.common import records_to_df, run_round_robin
from strategies import default_zoo

SEED = 42
REPETITIONS = 100
MULTIPLIER = LinearMultiplier(100)
HORIZONS = [10, 50, 100, 200, 500, 1000]

COOP_STRATS = {"TfT", "GenerousTfT", "TFT2", "Pavlov", "ContriteTfT", "AllC"}
DEFECT_STRATS = {"AllD", "SuspiciousTfT"}


def run(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zoo = default_zoo(seed=seed)
    all_records = []
    for T in tqdm(HORIZONS, desc="E3 horizons"):
        records = run_round_robin(
            zoo=zoo,
            multiplier=MULTIPLIER,
            rounds=T,
            repetitions=REPETITIONS,
            noise=0.0,
            base_seed=seed,
        )
        for r in records:
            all_records.append(r._asdict() | {"horizon": T})

    raw_df = pd.DataFrame(all_records)

    # mean_total по (horizon, strategy)
    per_rep = (
        raw_df.groupby(["horizon", "strat_a", "repetition"])
        .agg(match_total=("total_a", "mean"))
        .reset_index()
    )
    summary_df = (
        per_rep.groupby(["horizon", "strat_a"])
        .agg(mean_total=("match_total", "mean"), std_total=("match_total", "std"))
        .reset_index()
        .rename(columns={"strat_a": "strategy"})
    )

    # T*: при каком T кооперативные впервые обгоняют дефектные
    coop_mean = (
        summary_df[summary_df.strategy.isin(COOP_STRATS)]
        .groupby("horizon")["mean_total"]
        .mean()
        .rename("coop_avg")
    )
    defect_mean = (
        summary_df[summary_df.strategy.isin(DEFECT_STRATS)]
        .groupby("horizon")["mean_total"]
        .mean()
        .rename("defect_avg")
    )
    crossover_df = pd.concat([coop_mean, defect_mean], axis=1).reset_index()
    crossover_df["coop_wins"] = crossover_df["coop_avg"] > crossover_df["defect_avg"]

    return raw_df, summary_df, crossover_df


def save(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    crossover_df: pd.DataFrame,
) -> None:
    out = Path(__file__).parent.parent / "output" / "data"
    out.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out / "exp3_raw.csv", index=False)
    summary_df.to_csv(out / "exp3_summary.csv", index=False)
    crossover_df.to_csv(out / "exp3_crossover.csv", index=False)
    print(f"E3 saved -> {out}")
