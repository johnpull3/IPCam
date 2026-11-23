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


def sharpe_per_trade(trades, initial):
    """mean/std de los retornos por trade cerrado × sqrt(252) (aproximación pedida).

    retorno_i = pnl_i / equity antes del trade (equity = inicial + PnL acumulado; ignora funding).
    """
    if len(trades) < 2:
        return None
    eq, rets = initial, []
    for t in trades:
        rets.append(t.pnl / eq if eq > 0 else 0.0)
        eq += t.pnl
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return round(mean / var ** 0.5 * 252 ** 0.5, 3) if var > 0 else None


def breakdown(trades):
    out = {"by_side": {}, "by_reason": {}, "by_tag": {}}
    for t in trades:
        for key, val in (("by_side", t.side.value), ("by_reason", t.reason), ("by_tag", t.tag)):
            d = out[key].setdefault(val, {"n": 0, "wins": 0, "pnl": 0.0})
            d["n"] += 1
            d["wins"] += t.pnl > 0
            d["pnl"] = round(d["pnl"] + t.pnl, 2)
    return out


def metrics(result):
    trades = result["trades"]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    sig = result.get("signals", 0)
    missed = result.get("missed_signals", 0)
    return {
        "execution_mode": result.get("execution_mode"),
        "signals": sig,
        "missed_signals": missed,
        "fill_rate_pct": round(100 * (sig - missed) / sig, 2) if sig else None,
        "n_trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2) if trades else None,
        "profit_factor": (round(gross_win / gross_loss, 3) if gross_loss > 0
                          else (None if not trades else float("inf"))),
        "return_pct": round(result["return_pct"], 2),
        "final_equity": round(result["final_equity"], 2),
        "max_drawdown_pct": round(max_drawdown(result["equity_curve"],
                                               result["initial_equity"]), 2),
        "sharpe_approx": sharpe_per_trade(trades, result["initial_equity"]),
        "breakdown": breakdown(trades),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--path", default=os.environ.get("ATLAS_BINANCE_DATA"))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="5m", choices=["1m", "5m", "15m", "1h", "4h"])
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--execution", choices=["taker", "maker"], default="taker")
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
    print(f"[{args.execution}] {label}: {len(bars):,} velas cargadas en {time.time() - t0:.1f}s")

    results = {}
    for name, factory in STRATEGIES:
        t1 = time.time()
        res = run_backtest(factory(), bars, execution_mode=args.execution)
        m = metrics(res)
        results[name] = m
        print(f"\n=== {name} ===  ({time.time() - t1:.1f}s)")
        for k, v in m.items():
            if k != "breakdown":
                print(f"  {k:<17} {v}")
        for k, groups in m["breakdown"].items():
            print(f"  {k}: " + ", ".join(f"{g}={d['n']} (wins {d['wins']}, pnl {d['pnl']})"
                                          for g, d in groups.items()))

    out = args.out or os.path.join(
        "data", "reports",
        f"backtest_{args.symbol.lower()}_{args.timeframe}_{args.start}_{args.end}_{args.execution}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"symbol": args.symbol, "timeframe": args.timeframe,
                   "start": args.start, "end": args.end, "bars": len(bars),
                   "execution": args.execution,
                   "engine_notes": "funding fijo 0.0001/8h",
                   "results": results}, fh, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {out}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
