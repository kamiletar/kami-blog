"""CLI-диспетчер экспериментов.

Использование:
    python -m experiments.run --exp 1 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск эксперимента pd-multiplier")
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Номер эксперимента (1–5)")
    parser.add_argument("--seed", type=int, default=42, help="Мастер-seed (default: 42)")
    args = parser.parse_args()

    if args.exp == 1:
        from experiments.exp1 import run, save
        print("=== Эксперимент 1: базовый турнир ===")
        raw_df, summary_df = run(seed=args.seed)
        save(raw_df, summary_df)
        print("\nТоп-5 стратегий на Constant(1) [контроль ~ классический Аксельрод]:")
        ctrl = summary_df[summary_df.multiplier_name == "Constant(1)"].head(5)
        print(ctrl[["strategy", "mean_total", "std_total", "mean_coop", "rank"]].to_string(index=False))

    elif args.exp == 2:
        from experiments.exp2 import run, save
        print("=== Эксперимент 2: чувствительность к шуму ===")
        raw_df, summary_df = run(seed=args.seed)
        save(raw_df, summary_df)

    elif args.exp == 3:
        from experiments.exp3 import run, save
        print("=== Эксперимент 3: чувствительность к горизонту ===")
        raw_df, summary_df, crossover_df = run(seed=args.seed)
        save(raw_df, summary_df, crossover_df)
        print(crossover_df.to_string(index=False))

    elif args.exp == 4:
        from experiments.exp4 import run, save
        print("=== Эксперимент 4: AllC vs AllD парная дуэль ===")
        dfs = run(seed=args.seed)
        save(dfs)

    elif args.exp == 5:
        from experiments.exp5 import run, save
        print("=== Эксперимент 5: эволюционная динамика ===")
        dfs = run(seed=args.seed)
        save(dfs)


if __name__ == "__main__":
    main()
