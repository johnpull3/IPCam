"""Análisis de robustez de una estrategia sobre un universo de símbolos.

- Tabla por símbolo (PF neto de funding, trades, win %, retorno %, max DD %)
- Pool agregado + bootstrap por meses completos
- Recuento de símbolos con PF > 1.0 y > 1.15
- PF del pool por régimen (por mes de salida del trade)
- Correlación media entre retornos mensuales de los símbolos
- Evaluación del criterio de edge

Uso:
    python scripts/robustness.py --strategy TrendPullback --timeframe 4h \
        --start 2020-01 --end 2023-12 --funding real --bootstrap 2000 --symbols ...
"""
import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multi_symbol import (  # noqa: E402
    STRATEGIES, available_range, bootstrap_pf_months, fixed8h_events, trade_stats,
)
from run_real_backtest import load_real, max_drawdown  # noqa: E402
from atlas_strategies.backtest.run_backtest import run_backtest  # noqa: E402
from atlas_strategies.data.binance_vision import load_funding_from_zip  # noqa: E402

REGIMES = {
    "bull 2020-10..2021-11": ("2020-10", "2021-11"),
    "bear 2021-12..2022-12": ("2021-12", "2022-12"),
    "recovery 2023-01..2023-12": ("2023-01", "2023-12"),
}


def month_of(ts):
    return dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc).strftime("%Y-%m")


def monthly_returns(bars, equity_curve):
    """Retorno mensual a partir de la equity al cierre de cada mes."""
    last_eq = {}
    for b, eq in zip(bars[:-1], equity_curve):  # equity_curve[i] tras procesar bars[i+1]
        last_eq[month_of(b.ts)] = eq
    months = sorted(last_eq)
    rets, prev = {}, 10_000.0
    for m in months:
        rets[m] = last_eq[m] / prev - 1 if prev else 0.0
        prev = last_eq[m]
    return rets


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / (va * vb) ** 0.5 if va > 0 and vb > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="TrendPullback", choices=list(STRATEGIES))
    ap.add_argument("--path", default=os.environ.get("ATLAS_BINANCE_DATA"))
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--funding", choices=["legacy", "fixed8h", "real"], default="real")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = os.path.expanduser(args.path)
    factory = STRATEGIES[args.strategy]
    rows, pool, by_month, mrets, arets = {}, [], defaultdict(list), {}, {}
    for sym in args.symbols:
        first, last, missing = available_range(base, sym, args.start, args.end)
        if first is None or missing:
            print(f"!! {sym}: datos incompletos {missing[:5]}")
            continue
        bars = load_real(base, sym, args.timeframe, first, last)
        if args.funding == "real":
            events = load_funding_from_zip(sym, first, last, base)
        elif args.funding == "fixed8h":
            events = fixed8h_events(first, last)
        else:
            events = None
        res = run_backtest(factory(), bars, funding_events=events)
        net = [t.pnl + t.funding for t in res["trades"]]
        for t in res["trades"]:
            by_month[month_of(t.exit_ts)].append(t.pnl + t.funding)
        r = trade_stats(net)
        r.update({"first_month": first,
                  "return_pct": round(res["return_pct"], 2),
                  "max_drawdown_pct": round(max_drawdown(res["equity_curve"], 10_000.0), 2),
                  "funding_total": round(sum(t.funding for t in res["trades"]), 2)})
        rows[sym] = r
        pool.extend(net)
        mrets[sym] = monthly_returns(bars, res["equity_curve"])
        closes = {}
        for b in bars:
            closes[month_of(b.ts)] = b.close
        ms = sorted(closes)
        arets[sym] = {m: closes[m] / closes[p] - 1 for p, m in zip(ms, ms[1:])}
        print(f"{sym:<12} PF {r['profit_factor']!s:>6}  trades {r['n_trades']:>4}  "
              f"win {r['win_rate_pct']!s:>5}%  ret {r['return_pct']:>7}%  DD {r['max_drawdown_pct']:>6}%  "
              f"desde {first}", flush=True)

    total = trade_stats(pool)
    total["bootstrap_pf_months"] = bootstrap_pf_months(by_month, n=args.bootstrap)

    pfs = [r["profit_factor"] or 0 for r in rows.values()]
    n_gt1 = sum(1 for p in pfs if p > 1.0)
    n_gt115 = sum(1 for p in pfs if p > 1.15)

    regimes = {}
    for name, (a, b) in REGIMES.items():
        pnl = [p for m, ps in by_month.items() if a <= m <= b for p in ps]
        regimes[name] = trade_stats(pnl)

    syms = list(mrets)
    common = sorted(set.intersection(*(set(mrets[s]) for s in syms))) if syms else []
    corrs = [c for s1, s2 in combinations(syms, 2)
             if (c := pearson([mrets[s1][m] for m in common], [mrets[s2][m] for m in common]))
             is not None]
    mean_corr = round(sum(corrs) / len(corrs), 3) if corrs else None
    common_a = sorted(set.intersection(*(set(arets[s]) for s in syms))) if syms else []
    acorrs = [c for s1, s2 in combinations(syms, 2)
              if (c := pearson([arets[s1][m] for m in common_a], [arets[s2][m] for m in common_a]))
              is not None]
    mean_asset_corr = round(sum(acorrs) / len(acorrs), 3) if acorrs else None

    n = len(rows)
    boot = total["bootstrap_pf_months"] or {}
    reg_pos = sum(1 for r in regimes.values() if (r["profit_factor"] or 0) > 1.0)
    crit = {
        "pf_pool_gt_1.15": (total["profit_factor"] or 0) > 1.15,
        "bootstrap_months_low_gt_1.00": boot.get("ci95_low", 0) > 1.0,
        "pf_gt_1_in_at_least_12_of_20": n_gt1 >= 12,
        "pf_gt_1_in_at_least_2_of_3_regimes": reg_pos >= 2,
    }
    met = sum(crit.values())
    verdict = ("EDGE DEMOSTRADO" if met == 4 else "ZONA GRIS ALTA" if met == 3
               else "NO ES EDGE OPERATIVO")

    print(f"\nPOOL ({n} símbolos): PF {total['profit_factor']}  trades {total['n_trades']}  "
          f"win {total['win_rate_pct']}%  avgRR {total['avg_rr']}")
    print(f"Bootstrap meses ({args.bootstrap}): {boot}")
    print(f"Símbolos PF>1.0: {n_gt1}/{n}   PF>1.15: {n_gt115}/{n}")
    for k, r in regimes.items():
        print(f"  {k:<28} PF {r['profit_factor']!s:>6}  trades {r['n_trades']:>5}  win {r['win_rate_pct']}%")
    print(f"Correlación media retornos mensuales ESTRATEGIA: {mean_corr}  ({len(common)} meses "
          f"comunes, {len(corrs)} pares)")
    print(f"Correlación media retornos mensuales PRECIO activo: {mean_asset_corr}")
    if mean_corr is not None and mean_corr > 0.7:
        print("AVISO: correlación > 0.7, el bootstrap por meses sobreestima la independencia.")
    print("Criterios:", crit, f"-> {met}/4 -> {verdict}")

    out = args.out or os.path.join(
        "data", "reports",
        f"robustness_{args.strategy}_{args.timeframe}_{args.start}_{args.end}_{n}sym_funding-{args.funding}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"strategy": args.strategy, "timeframe": args.timeframe, "start": args.start,
                   "end": args.end, "funding": args.funding, "symbols": list(rows),
                   "per_symbol": rows, "pool": total, "n_pf_gt_1": n_gt1, "n_pf_gt_1_15": n_gt115,
                   "regimes": regimes, "mean_monthly_corr_strategy": mean_corr,
                   "mean_monthly_corr_asset_price": mean_asset_corr,
                   "criteria": crit, "criteria_met": met, "verdict": verdict},
                  fh, indent=2, ensure_ascii=False)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
