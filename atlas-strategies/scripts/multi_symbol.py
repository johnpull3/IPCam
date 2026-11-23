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

import datetime as dt  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

from run_real_backtest import load_real, max_drawdown  # noqa: E402
from atlas_strategies.data.binance_vision import _iter_months, _um_root, load_funding_from_zip  # noqa: E402
from atlas_strategies.backtest.run_backtest import run_backtest  # noqa: E402
from atlas_strategies.strategies.bollinger_rsi2_reversion import BollingerRsi2Reversion  # noqa: E402
from atlas_strategies.strategies.donchian_breakout import DonchianBreakoutStrategy  # noqa: E402
from atlas_strategies.strategies.swing_reclaim import SwingReclaim  # noqa: E402
from atlas_strategies.strategies.trend_pullback import TrendPullbackStrategy  # noqa: E402
from atlas_strategies.strategies.vwap_ema_rsi_scalp import VwapEmaRsiScalp  # noqa: E402

STRATEGIES = {
    "TrendPullback": TrendPullbackStrategy,
    "TrendPullbackRegime": lambda: TrendPullbackStrategy(regime_filter=True),
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


def bootstrap_pf_months(trades_by_month, n=1000, seed=0):
    """Remuestrea meses completos (todos los símbolos juntos) para respetar la correlación."""
    months = [m for m in trades_by_month if trades_by_month[m]]
    if len(months) < 2:
        return None
    rng = random.Random(seed)
    pfs = []
    for _ in range(n):
        x = [p for m in (rng.choice(months) for _ in months) for p in trades_by_month[m]]
        gw = sum(v for v in x if v > 0)
        gl = -sum(v for v in x if v <= 0)
        pfs.append(gw / gl if gl > 0 else float("inf"))
    pfs.sort()
    pos = sum(1 for m in months if sum(trades_by_month[m]) > 0)
    return {"n": n, "months": len(months), "positive_months": pos,
            "ci95_low": round(pfs[int(0.025 * n)], 3),
            "ci95_high": round(pfs[int(0.975 * n) - 1], 3),
            "p_pf_lt_1_pct": round(100 * sum(p < 1 for p in pfs) / n, 1)}


def available_range(base, symbol, start, end):
    kdir = os.path.join(_um_root(base), "monthly", "klines", symbol, "1m")
    pat = re.compile(rf"^{re.escape(symbol)}-1m-(\d{{4}}-\d{{2}})\.zip$")
    have = sorted(m.group(1) for f in os.listdir(kdir) if (m := pat.match(f)))
    want = [m for m in _iter_months(start, end)]
    inside = [m for m in want if m in have]
    if not inside:
        return None, None, want
    first = inside[0]
    missing = [m for m in want if m >= first and m not in have]
    return first, end, missing


def fixed8h_events(start, end, rate=0.0001):
    t0 = int(dt.datetime.strptime(start, "%Y-%m").replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    y, m = map(int, end.split("-"))
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    t1 = int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    return {t: rate for t in range(t0, t1, 8 * 3_600_000)}


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
    ap.add_argument("--funding", choices=["legacy", "fixed8h", "real"], default="legacy",
                    help="legacy: funding fijo histórico del engine; fixed8h: 0.0001 cada 8h con "
                         "calendario correcto; real: fundingRate descargado")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    factory = STRATEGIES[args.strategy]
    base = os.path.expanduser(args.path)
    rows, pool, pool_gross = {}, [], []
    by_month = defaultdict(list)
    for sym in args.symbols:
        first, last, missing = available_range(base, sym, args.start, args.end)
        if first is None or missing:
            print(f"{sym}: sin datos completos en {args.start}..{args.end} (faltan {missing[:5]}...)")
            continue
        bars = load_real(base, sym, args.timeframe, first, last)
        if args.funding == "real":
            events = load_funding_from_zip(sym, first, last, base)
        elif args.funding == "fixed8h":
            events = fixed8h_events(first, last)
        else:
            events = None
        res = run_backtest(factory(), bars, execution_mode=args.execution, funding_events=events)
        pnls = [t.pnl + t.funding for t in res["trades"]]  # PnL neto de funding
        gross = [t.pnl for t in res["trades"]]
        for t in res["trades"]:
            by_month[dt.datetime.fromtimestamp(t.exit_ts / 1000, dt.timezone.utc)
                     .strftime("%Y-%m")].append(t.pnl + t.funding)
        row = trade_stats(pnls)
        row.update({
            "first_month": first,
            "pf_gross_no_funding": trade_stats(gross)["profit_factor"],
            "funding_total": round(sum(t.funding for t in res["trades"]), 2),
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
        pool_gross.extend(gross)

    total = trade_stats(pool)
    total["pf_gross_no_funding"] = trade_stats(pool_gross)["profit_factor"]
    total["funding_total"] = round(sum(r["funding_total"] for r in rows.values()), 2)
    total["bootstrap_pf"] = bootstrap_pf(pool, n=args.bootstrap)
    total["bootstrap_pf_months"] = bootstrap_pf_months(by_month, n=args.bootstrap)

    hdr = f"{'Símbolo':<9}|{'Trades':>6}|{'Win%':>6}|{'PF':>6}|{'Ret%':>7}|{'MaxDD%':>7}|{'AvgW':>8}|{'AvgL':>8}|{'AvgRR':>6}"
    print(f"{args.strategy} {args.timeframe} {args.start}→{args.end} [{args.execution}] "
          f"funding={args.funding}  (PF/Win/Avg sobre PnL neto de funding)")
    print(hdr)
    for sym, r in rows.items():
        print(f"{sym:<9}|{r['n_trades']:>6}|{r['win_rate_pct']!s:>6}|{r['profit_factor']!s:>6}|"
              f"{r['return_pct']:>7}|{r['max_drawdown_pct']:>7}|{r['avg_win']!s:>8}|"
              f"{r['avg_loss']!s:>8}|{r['avg_rr']!s:>6}  desde {r['first_month']} "
              f"PFbruto {r['pf_gross_no_funding']} fund {r['funding_total']} {r['by_reason']}")
    print(f"{'TOTAL':<9}|{total['n_trades']:>6}|{total['win_rate_pct']!s:>6}|{total['profit_factor']!s:>6}|"
          f"{'':>7}|{'':>7}|{total['avg_win']!s:>8}|{total['avg_loss']!s:>8}|{total['avg_rr']!s:>6}  "
          f"pnl {total['net_pnl']}")
    print(f"PF bruto sin funding: {total['pf_gross_no_funding']}  funding total: {total['funding_total']}")
    print(f"Bootstrap PF (trades): {total['bootstrap_pf']}")
    print(f"Bootstrap PF (meses):  {total['bootstrap_pf_months']}")

    out = args.out or os.path.join(
        "data", "reports",
        f"multi_{args.strategy}_{args.timeframe}_{args.start}_{args.end}_{args.execution}"
        f"_funding-{args.funding}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"strategy": args.strategy, "timeframe": args.timeframe,
                   "start": args.start, "end": args.end, "execution": args.execution,
                   "funding": args.funding,
                   "per_symbol": rows, "total": total}, fh, indent=2, ensure_ascii=False)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
