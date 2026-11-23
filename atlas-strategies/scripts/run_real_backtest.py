"""Backtest sobre datos reales de Binance Vision usando el engine existente sin modificarlo.

Uso:
    python scripts/run_real_backtest.py --data real --path "$ATLAS_BINANCE_DATA" \
        --symbol BTCUSDT --timeframe 5m --start 2024-01 --end 2026-09

Notas:
- Carga mes a mes y resamplea por mes (los meses empiezan a 00:00 UTC, así que
  los buckets de 5m/15m/1h/4h nunca cruzan meses) para limitar memoria.
- El engine (run_backtest) se usa tal cual: funding fijo de 0.0001/8h, comisiones
  y slippage de common/executor.py. NO usa el funding real descargado.
- Max drawdown se calcula aquí a partir de result["equity_curve"].
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas_strategies.backtest.data_generator import generate_synthetic_bars  # noqa: E402
from atlas_strategies.backtest.run_backtest import run_backtest  # noqa: E402
from atlas_strategies.data.binance_vision import (  # noqa: E402
    _iter_months,
    load_klines_from_zip,
    resample,
)
from atlas_strategies.strategies.bollinger_rsi2_reversion import BollingerRsi2Reversion  # noqa: E402
from atlas_strategies.strategies.swing_reclaim import SwingReclaim  # noqa: E402
from atlas_strategies.strategies.vwap_ema_rsi_scalp import VwapEmaRsiScalp  # noqa: E402

STRATEGIES = [
    ("VWAP+EMA+RSI Scalp", VwapEmaRsiScalp),
    ("Bollinger+RSI2", BollingerRsi2Reversion),
    ("Swing Reclaim", SwingReclaim),
]


def load_real(path, symbol, timeframe, start, end):
    bars = []
    for month in _iter_months(start, end):
        m = load_klines_from_zip(symbol, month, month, path)
        bars.extend(m if timeframe == "1m" else resample(m, timeframe))
    return bars


def max_drawdown(curve, initial):
    peak, mdd = initial, 0.0
    for eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            mdd = max(mdd, (peak - eq) / peak)
    return mdd * 100


def metrics(result):
    trades = result["trades"]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    return {
        "n_trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2) if trades else None,
        "profit_factor": (round(gross_win / gross_loss, 3) if gross_loss > 0
                          else (None if not trades else float("inf"))),
        "return_pct": round(result["return_pct"], 2),
        "final_equity": round(result["final_equity"], 2),
        "max_drawdown_pct": round(max_drawdown(result["equity_curve"],
                                               result["initial_equity"]), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--path", default=os.environ.get("ATLAS_BINANCE_DATA"))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="5m", choices=["1m", "5m", "15m", "1h", "4h"])
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.data == "real":
        if not args.path:
            sys.exit("Falta --path o ATLAS_BINANCE_DATA")
        bars = load_real(os.path.expanduser(args.path), args.symbol,
                         args.timeframe, args.start, args.end)
        label = f"{args.symbol} {args.timeframe} {args.start}→{args.end}"
    else:
        bars = generate_synthetic_bars(n=20000)
        label = "sintético 20000 velas"
    print(f"{label}: {len(bars):,} velas cargadas en {time.time() - t0:.1f}s")

    results = {}
    for name, factory in STRATEGIES:
        t1 = time.time()
        res = run_backtest(factory(), bars)
        m = metrics(res)
        results[name] = m
        print(f"\n=== {name} ===  ({time.time() - t1:.1f}s)")
        for k, v in m.items():
            print(f"  {k:<17} {v}")

    out = args.out or os.path.join(
        "data", "reports",
        f"backtest_{args.symbol.lower()}_{args.timeframe}_{args.start}_{args.end}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"symbol": args.symbol, "timeframe": args.timeframe,
                   "start": args.start, "end": args.end, "bars": len(bars),
                   "engine_notes": "run_backtest sin cambios; funding fijo 0.0001/8h",
                   "results": results}, fh, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {out}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
