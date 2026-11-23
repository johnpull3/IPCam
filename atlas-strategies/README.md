# atlas-strategies

Paquete de estrategias de trading para validación previa a integrar en Atlas.

## Instalación

```bash
pip install -e ".[dev]"
```

## Correr el backtest sintético

```bash
python -m atlas_strategies.backtest.run_backtest
```

## Correr los tests

```bash
pytest -v
```

## Qué verifica Claude Code con esto

1. test_no_lookahead_vwap_strategy: truncar la serie no cambia señales pasadas.
2. test_sl_first_when_both_hit: conservador, SL primero.
3. test_size_position_*: step_size y min_notional de Binance respetados.
4. test_strategies_produce_trades: las 3 estrategias corren el backtest sin crash.

## Siguientes pasos

1. Clonar el repo, correr pytest -v.
2. Si todos los tests pasan, sustituir generate_synthetic_bars por datos reales de Binance Vision / API.
3. Añadir walk-forward (6 meses train / 2 meses test).
4. Solo entonces conectar al executor real de Atlas.
