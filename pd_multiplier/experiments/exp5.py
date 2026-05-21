"""Эксперимент 5: эволюционная динамика (репликатор).

Начальная популяция — равные доли всех стратегий.
На каждом поколении round-robin внутри популяции;
доля пропорциональна среднему фитнесу. 200 поколений.
Запускается для тех же четырёх множителей, что в E1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

from engine.game import Game
from engine.multiplier import (
    CappedMultiplier,
    ConstantMultiplier,
    LinearMultiplier,
    LogMultiplier,
    PowerMultiplier,
)
from strategies import default_zoo

ROUNDS = 200
GENERATIONS = 200
SEED = 42

MULTIPLIERS = [
    ConstantMultiplier(1.0),
    LinearMultiplier(100),
    PowerMultiplier(100, 0.5),
    LogMultiplier(100),
]


def _payoff_matrix(
    zoo: list[tuple[str, object]],
    multiplier,
    rounds: int,
    seed: int,
) -> np.ndarray:
    """Матрица средних выигрышей: payoff_mat[i, j] = выигрыш i против j."""
    n = len(zoo)
    mat = np.zeros((n, n))
    rng = np.random.default_rng(seed)
    for i, j in product(range(n), range(n)):
        _, fa = zoo[i]
        _, fb = zoo[j]
        s = int(rng.integers(0, 2**31 - 1))
        r = Game(fa(), fb(), rounds=rounds, multiplier=multiplier, seed=s).play()
        mat[i, j] = r.total_a
    return mat


def _replicator(
    payoff_mat: np.ndarray,
    generations: int,
) -> np.ndarray:
    """Репликаторная динамика. Возвращает историю долей shape (gens+1, n)."""
    n = payoff_mat.shape[0]
    freq = np.ones(n) / n
    history = np.zeros((generations + 1, n))
    history[0] = freq.copy()
    for g in range(generations):
        fitness = payoff_mat @ freq          # средний выигрыш каждой стратегии
        avg_fitness = freq @ fitness
        if avg_fitness < 1e-12:
            history[g + 1] = freq.copy()
            continue
        freq = freq * fitness / avg_fitness
        freq = np.clip(freq, 0, None)
        freq /= freq.sum()
        history[g + 1] = freq.copy()
    return history


def run(seed: int = SEED) -> list[dict]:
    """Запустить эволюционную симуляцию для всех множителей."""
    zoo = default_zoo(seed=seed)
    names = [name for name, _ in zoo]
    results = []
    for mult in tqdm(MULTIPLIERS, desc="E5 multipliers"):
        mat = _payoff_matrix(zoo, mult, ROUNDS, seed)
        history = _replicator(mat, GENERATIONS)
        results.append({
            "multiplier": mult.name,
            "names": names,
            "history": history,       # shape (GENERATIONS+1, n)
            "payoff_mat": mat,
        })
    return results


def save(dfs: list[dict]) -> None:
    out = Path(__file__).parent.parent / "output" / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in dfs:
        for gen in range(d["history"].shape[0]):
            for i, name in enumerate(d["names"]):
                rows.append({
                    "multiplier": d["multiplier"],
                    "generation": gen,
                    "strategy": name,
                    "share": float(d["history"][gen, i]),
                })
    pd.DataFrame(rows).to_csv(out / "exp5_evolution.csv", index=False)
    print(f"E5 saved -> {out}")
