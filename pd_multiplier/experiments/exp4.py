"""Эксперимент 4: AllC vs AllD парная дуэль.

Сравниваем кумулятивные выигрыши AllC и AllD при четырёх M(S).
Ищем точку кроссовера AllC >= AllD. Накладываем аналитическую кривую.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from engine.game import Game
from engine.multiplier import (
    ConstantMultiplier,
    LinearMultiplier,
    LogMultiplier,
    PowerMultiplier,
)
from strategies.basic import AllC, AllD

ROUNDS = 500
SEED = 42

MULTIPLIERS = [
    ConstantMultiplier(1.0),
    LinearMultiplier(100),
    PowerMultiplier(100, 0.5),
    LogMultiplier(100),
]


def _run_single(mult, seed: int) -> dict:
    """Сравниваем 4 сценария для каждого множителя.

    Сценарии (с точки зрения игрока A):
      CC  — AllC vs AllC: взаимная кооперация → максимальный компаундинг
      DD  — AllD vs AllD: взаимная дефекция
      DC  — AllD vs AllC: AllD эксплуатирует AllC (T=5/раунд)
      CD  — AllC vs AllD: AllC жертвует (S=0/раунд) — всегда 0 после умножения

    Это показывает: CC vs DC — вот где виден выигрыш от кооперации при M(S).
    В head-to-head AllC проигрывает AllD (CD < DC), но CC >> DD по совокупному
    выигрышу — это и есть механизм компаундинга.
    """
    r_cc = Game(AllC(), AllC(), rounds=ROUNDS, multiplier=mult, seed=seed).play()
    r_dd = Game(AllD(), AllD(), rounds=ROUNDS, multiplier=mult, seed=seed).play()
    r_dc = Game(AllD(), AllC(), rounds=ROUNDS, multiplier=mult, seed=seed).play()
    r_cd = Game(AllC(), AllD(), rounds=ROUNDS, multiplier=mult, seed=seed).play()
    return {
        "multiplier": mult.name,
        "rounds": np.arange(1, ROUNDS + 1),
        "cum_CC": np.cumsum(r_cc.multiplied_payoffs_a),
        "cum_DD": np.cumsum(r_dd.multiplied_payoffs_a),
        "cum_DC": np.cumsum(r_dc.multiplied_payoffs_a),
        "cum_CD": np.cumsum(r_cd.multiplied_payoffs_a),
        "S_CC": r_cc.S_trajectory,
        "M_CC": r_cc.M_trajectory,
    }


def run(seed: int = SEED) -> list[dict]:
    return [_run_single(m, seed) for m in MULTIPLIERS]


def save(dfs: list[dict]) -> None:
    out = Path(__file__).parent.parent / "output" / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in dfs:
        for t in range(ROUNDS):
            rows.append({
                "multiplier": d["multiplier"],
                "round": int(d["rounds"][t]),
                "cum_CC": float(d["cum_CC"][t]),
                "cum_DD": float(d["cum_DD"][t]),
                "cum_DC": float(d["cum_DC"][t]),
                "cum_CD": float(d["cum_CD"][t]),
            })
    pd.DataFrame(rows).to_csv(out / "exp4_cumulative.csv", index=False)
    print(f"E4 saved -> {out}")
