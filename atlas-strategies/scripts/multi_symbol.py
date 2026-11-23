"""Corre una estrategia en varios símbolos y agrega los trades en un pool con bootstrap del PF.

Uso:
    python scripts/multi_symbol.py --strategy TrendPullback --timeframe 4h \
        --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT DOGEUSDT XRPUSDT --start 2024-01 --end 2026-09
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_real_backtest import load_real, max_drawdown  # noqa: E402
from atlas_strategies.backtest.run_backtest import run_backtest  # noqa: E402
from atlas_strategies.strategies.bollinger_rsi2_reversion import BollingerRsi2Reversion  # noqa: E402
from atlas_strategies.strategies.donchian_breakout import DonchianBreakoutStrategy  # noqa: E402
from atlas_strategies.strategies.swing_reclaim import SwingReclaim  # noqa: E402
from atlas_strategies.strategies.trend_pullback import TrendPullbackStrategy  # noqa: E402
from atlas_strategies.strategies.vwap_ema_rsi_scalp import VwapEmaRsiScalp  # noqa: E402

STRATEGIES = {
    "TrendPullback": TrendPullbackStrategy,
    "DonchianBreakout": DonchianBreakoutStrategy,
    "SwingReclaim": SwingReclaim,
    "VwapEmaRsi": VwapEmaRsiScalp,
    "BollingerRsi2": BollingerRsi2Reversion,
}


def trade_stats(pnls):
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    avg_w = gw / len(wins) if wins else None
    avg_l = gl / len(losses) if losses else None
    return {
        "n_trades": len(pnls),
        "win_rate_pct": round(100 * len(wins) / len(pnls), 2) if pnls else None,
        "profit_factor": round(gw / gl, 3) if gl else None,
        "avg_win": round(avg_w, 2) if avg_w is not None else None,
        "avg_loss": round(-avg_l, 2) if avg_l is not None else None,
        "avg_rr": round(avg_w / avg_l, 2) if avg_w and avg_l else None,
        "net_pnl": round(gw - gl, 2),
    }


def bootstrap_pf(pnls, n=1000, seed=0):
    if len(pnls) < 2:
        return None
    rng = random.Random(seed)
    pfs = []
    for _ in range(n):
        x = [rng.choice(pnls) for _ in pnls]
        gw = sum(v for v in x if v > 0)
        gl = -sum(v for v in x if v <= 0)
        pfs.append(gw / gl if gl > 0 else float("inf"))
    pfs.sort()
    return {"n": n, "ci95_low": round(pfs[int(0.025 * n)], 3),
            "ci95_high": round(pfs[int(0.975 * n) - 1], 3),
            "p_pf_lt_1_pct": round(100 * sum(p < 1 for p in pfs) / n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ap.add_argument("--path", default=os.environ.get("ATLAS_BINANCE_DATA"))
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--execution", choices=["taker", "maker"], default="taker")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    factory = STRATEGIES[args.strategy]
    rows, pool = {}, []
    for sym in args.symbols:
        bars = load_real(os.path.expanduser(args.path), sym, args.timeframe, args.start, args.end)
        res = run_backtest(factory(), bars, execution_mode=args.execution)
        pnls = [t.pnl for t in res["trades"]]
        row = trade_stats(pnls)
        row.update({
            "return_pct": round(res["return_pct"], 2),
            "final_equity": round(res["final_equity"], 2),
            "max_drawdown_pct": round(max_drawdown(res["equity_curve"], res["initial_equity"]), 2),
            "bars": len(bars),
            "by_reason": {r: sum(1 for t in res["trades"] if t.reason == r)
                          for r in sorted({t.reason for t in res["trades"]})},
            "long": sum(1 for t in res["trades"] if t.side.value == "LONG"),
            "short": sum(1 for t in res["trades"] if t.side.value == "SHORT"),
        })
        rows[sym] = row
        pool.extend(pnls)

    total = trade_stats(pool)
    total["bootstrap_pf"] = bootstrap_pf(pool, n=args.bootstrap)

    hdr = f"{'Símbolo':<9}|{'Trades':>6}|{'Win%':>6}|{'PF':>6}|{'Ret%':>7}|{'MaxDD%':>7}|{'AvgW':>8}|{'AvgL':>8}|{'AvgRR':>6}"
    print(f"{args.strategy} {args.timeframe} {args.start}→{args.end} [{args.execution}]")
    print(hdr)
    for sym, r in rows.items():
        print(f"{sym:<9}|{r['n_trades']:>6}|{r['win_rate_pct']!s:>6}|{r['profit_factor']!s:>6}|"
              f"{r['return_pct']:>7}|{r['max_drawdown_pct']:>7}|{r['avg_win']!s:>8}|"
              f"{r['avg_loss']!s:>8}|{r['avg_rr']!s:>6}  {r['by_reason']} L{r['long']}/S{r['short']}")
    print(f"{'TOTAL':<9}|{total['n_trades']:>6}|{total['win_rate_pct']!s:>6}|{total['profit_factor']!s:>6}|"
          f"{'':>7}|{'':>7}|{total['avg_win']!s:>8}|{total['avg_loss']!s:>8}|{total['avg_rr']!s:>6}  "
          f"pnl {total['net_pnl']}")
    print(f"Bootstrap PF: {total['bootstrap_pf']}")

    out = args.out or os.path.join(
        "data", "reports",
        f"multi_{args.strategy}_{args.timeframe}_{args.start}_{args.end}_{args.execution}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"strategy": args.strategy, "timeframe": args.timeframe,
                   "start": args.start, "end": args.end, "execution": args.execution,
                   "per_symbol": rows, "total": total}, fh, indent=2, ensure_ascii=False)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
