# pd-multiplier

Симулятор повторяющейся дилеммы заключённого с механикой множителя общего счёта: на каждом раунде выигрыши обоих игроков умножаются на `M(S)`, где `S` — это сумма уже умноженных выигрышей за предыдущие раунды. Компаундинг получается рекурсивным.

Гипотеза: при нетривиальных `M(S)` кооперация становится выгодной даже против стратегий, не наказывающих дефекцию. Жёсткие непрощающие стратегии (Grim) проседают сильнее, чем в классической PD, из-за потерянного будущего множителя.

## Установка

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # Unix
pip install -e .[dev]
```

## Запуск экспериментов

```bash
python -m experiments.run --exp 1 --seed 42
python -m experiments.run --exp 2 --seed 42
# ... и так далее для exp 3, 4, 5
```

Результаты сохраняются в `output/data/expN_*.csv` и графики в `output/figures/`.

## Тесты

```bash
pytest -v
```

## Структура

```
engine/        — Action, Multiplier, Strategy ABC, Game, GameResult
strategies/    — зоопарк стратегий (AllC, AllD, TfT, Grim, Pavlov, …)
experiments/   — сценарии E1–E5
analysis/      — функции построения графиков
tests/         — pytest
output/        — сырые CSV и PNG-графики
```

## Лицензия

Тот же CC BY 4.0, что у материнского репозитория `kami-blog`.
