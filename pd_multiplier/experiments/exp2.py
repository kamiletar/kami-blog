"""Эксперимент 2: чувствительность к шуму.

Linear(100), noise in {0, 0.01, 0.05, 0.1, 0.2},
200 раундов, 100 повторений. Показываем, как Grim проседает
относительно прощающих стратегий при росте шума.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from engine.multiplier import LinearMultiplier
from experiments.common import records_to_df, run_round_robin, strategy_summary
from strategies import default_zoo

ROUNDS = 200
REPETITIONS = 100
SEED = 42
MULTIPLIER = LinearMultiplier(100)
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]


def run(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    zoo = default_zoo(seed=seed)
    all_records = []
    for noise in tqdm(NOISE_LEVELS, desc="E2 noise"):
        records = run_round_robin(
            zoo=zoo,
            multiplier=MULTIPLIER,
            rounds=ROUNDS,
            repetitions=REPETITIONS,
            noise=noise,
            noise_mode="implementation",
            base_seed=seed,
        )
        # Добавляем уровень шума как дополнительную колонку
        for r in records:
            all_records.append(r._asdict() | {"noise": noise})

    raw_df = pd.DataFrame(all_records)
    # Сводка: mean_total по (noise, strategy)
    per_rep = (
        raw_df.groupby(["noise", "strat_a", "repetition"])
        .agg(match_total=("total_a", "mean"))
        .reset_index()
    )
    summary_df = (
        per_rep.groupby(["noise", "strat_a"])
        .agg(mean_total=("match_total", "mean"), std_total=("match_total", "std"))
        .reset_index()
        .rename(columns={"strat_a": "strategy"})
    )
    return raw_df, summary_df


def save(raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    out = Path(__file__).parent.parent / "output" / "data"
    out.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out / "exp2_raw.csv", index=False)
    summary_df.to_csv(out / "exp2_summary.csv", index=False)
    print(f"E2 saved -> {out}")
