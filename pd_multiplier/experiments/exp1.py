"""Эксперимент 1: базовый round-robin турнир.

Round-robin, каждый с каждым включая self-play.
200 раундов, без шума, 100 повторений.
Множители: Constant(1), Linear(100), Power(100,0.5), Log(100), Capped(Linear(100), 50).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from engine.multiplier import (
    CappedMultiplier,
    ConstantMultiplier,
    LinearMultiplier,
    LogMultiplier,
    PowerMultiplier,
)
from experiments.common import records_to_df, run_round_robin, strategy_summary
from strategies import default_zoo

ROUNDS = 200
REPETITIONS = 100
SEED = 42

MULTIPLIERS = [
    ConstantMultiplier(1.0),
    LinearMultiplier(100),
    PowerMultiplier(100, 0.5),
    LogMultiplier(100),
    CappedMultiplier(LinearMultiplier(100), cap=50.0),
]


def run(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Запустить E1. Возвращает (raw_df, summary_df)."""
    zoo = default_zoo(seed=seed)
    all_records = []

    for mult in tqdm(MULTIPLIERS, desc="E1 множители"):
        records = run_round_robin(
            zoo=zoo,
            multiplier=mult,
            rounds=ROUNDS,
            repetitions=REPETITIONS,
            noise=0.0,
            base_seed=seed,
        )
        all_records.extend(records)

    raw_df = records_to_df(all_records)
    summary_df = strategy_summary(raw_df)
    return raw_df, summary_df


def save(raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    out = Path(__file__).parent.parent / "output" / "data"
    out.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out / "exp1_raw.csv", index=False)
    summary_df.to_csv(out / "exp1_summary.csv", index=False)
    print(f"Сохранено в {out}")
