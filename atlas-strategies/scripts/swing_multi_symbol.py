"""Swing Reclaim en varios símbolos con pool agregado de trades.

Uso:
    python scripts/swing_multi_symbol.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT DOGEUSDT \
        --timeframe 4h --start 2024-01 --end 2026-09 [--execution taker|maker]

Cada símbolo corre con su propia cuenta de 10.000. La fila TOTAL junta todos los
trades en un pool único (PF = suma ganancias / suma pérdidas, win rate global).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_real_backtest import load_real, metrics  # noqa: E402
from atlas_strategies.backtest.run_backtest import run_backtest  # noqa: E402
from atlas_strategies.strategies.swing_reclaim import SwingReclaim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.environ.get("ATLAS_BINANCE_DATA"))
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--execution", choices=["taker", "maker"], default="taker")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, pool = {}, []
    for sym in args.symbols:
        bars = load_real(os.path.expanduser(args.path), sym, args.timeframe, args.start, args.end)
        res = run_backtest(SwingReclaim(), bars, execution_mode=args.execution)
        m = metrics(res)
        m["bars"] = len(bars)
        rows[sym] = m
        pool.extend(res["trades"])

    wins = [t for t in pool if t.pnl > 0]
    gw = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in pool if t.pnl <= 0))
    total = {
        "n_trades": len(pool),
        "win_rate_pct": round(100 * len(wins) / len(pool), 2) if pool else None,
        "profit_factor": round(gw / gl, 3) if gl else None,
        "net_pnl": round(gw - gl, 2),
        "long": sum(1 for t in pool if t.side.value == "LONG"),
        "short": sum(1 for t in pool if t.side.value == "SHORT"),
        "tp": sum(1 for t in pool if t.reason == "TP"),
        "sl": sum(1 for t in pool if t.reason == "SL"),
    }

    print(f"{'Símbolo':<10}|{'Trades':>7}|{'Win %':>7}|{'PF':>7}|{'Ret %':>7}|{'MaxDD %':>8}")
    for sym, m in rows.items():
        print(f"{sym:<10}|{m['n_trades']:>7}|{m['win_rate_pct']!s:>7}|{m['profit_factor']!s:>7}|"
              f"{m['return_pct']:>7}|{m['max_drawdown_pct']:>8}")
    print(f"{'TOTAL':<10}|{total['n_trades']:>7}|{total['win_rate_pct']!s:>7}|"
          f"{total['profit_factor']!s:>7}|  pnl {total['net_pnl']}")
    print(f"pool: long {total['long']} / short {total['short']}  |  TP {total['tp']} / SL {total['sl']}")

    out = args.out or os.path.join(
        "data", "reports",
        f"swing_multi_{args.timeframe}_{args.start}_{args.end}_{args.execution}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"timeframe": args.timeframe, "start": args.start, "end": args.end,
                   "execution": args.execution, "per_symbol": rows, "total": total},
                  fh, indent=2, ensure_ascii=False)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
