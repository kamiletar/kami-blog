"""Общие утилиты для графиков: стиль, палитры, вспомогательные функции."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["font.family"] = "DejaVu Sans"

FIGURES_DIR = Path(__file__).parent.parent / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Единый порядок стратегий для всех графиков
STRATEGY_ORDER = [
    "AllC", "TFT2", "GenerousTfT", "ContriteTfT",
    "TfT", "Pavlov", "Grim", "SuspiciousTfT", "Random", "AllD",
]

PALETTE = sns.color_palette("tab10", n_colors=10)
STRAT_COLOR = {s: PALETTE[i] for i, s in enumerate(STRATEGY_ORDER)}

MULT_ORDER = [
    "Constant(1)", "Linear(100)", "Power(100,0.5)", "Log(100)",
    "Capped(Linear(100),cap=50)",
]
MULT_PALETTE = sns.color_palette("viridis", n_colors=len(MULT_ORDER))
MULT_COLOR = {m: MULT_PALETTE[i] for i, m in enumerate(MULT_ORDER)}


def savefig(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    """Сохранить фигуру в FIGURES_DIR/name.png."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
