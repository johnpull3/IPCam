"""Diagnóstico de huecos (gaps) en las velas de 1m de Binance Vision.

Solo lectura: no modifica ni rellena datos.

Uso:
    python scripts/diagnose_gaps.py --base-path "G:\\Mi unidad\\Binance_Futures_Historicos"
    (o define ATLAS_BINANCE_DATA)

Opciones: --symbol BTCUSDT --start 2020-01 --end <último mes disponible>
          --out data/reports/btcusdt_gaps.json
"""
import argparse
import calendar
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas_strategies.data.binance_vision import (  # noqa: E402
    _iter_months,
    _um_root,
    load_klines_from_zip,
)

MIN_MS = 60_000

# Categorías por minutos FALTANTES dentro del hueco.
CATEGORIES = [
    ("<5min", 1, 4),
    ("5min-1h", 5, 60),
    ("1h-1dia", 61, 1440),
    (">1dia", 1441, None),
]


def fmt(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def month_bounds(month: str):
    y, m = map(int, month.split("-"))
    start = int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    days = calendar.monthrange(y, m)[1]
    return start, start + days * 1440 * MIN_MS, days * 1440


def categorize(missing: int) -> str:
    for name, lo, hi in CATEGORIES:
        if missing >= lo and (hi is None or missing <= hi):
            return name
    raise ValueError(missing)


def available_months(kdir: str, symbol: str):
    pat = re.compile(rf"^{re.escape(symbol)}-1m-(\d{{4}}-\d{{2}})\.zip$")
    return sorted(m.group(1) for f in os.listdir(kdir) if (m := pat.match(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", default=os.environ.get("ATLAS_BINANCE_DATA"))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default=None, help="por defecto, último mes disponible")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    if not args.base_path:
        sys.exit("Falta --base-path o ATLAS_BINANCE_DATA")
    base = os.path.expanduser(args.base_path)
    sym = args.symbol
    kdir = os.path.join(_um_root(base), "monthly", "klines", sym, "1m")
    avail = available_months(kdir, sym)
    if not avail:
        sys.exit(f"No hay ZIPs en {kdir}")
    end = args.end or avail[-1]
    out = args.out or os.path.join("data", "reports", f"{sym.lower()}_gaps.json")

    months = list(_iter_months(args.start, end))
    missing_files = [m for m in months if m not in avail]

    # Carga todo y calcula cobertura por mes.
    bars = []
    per_month = []
    for m in months:
        m_start, m_end, expected = month_bounds(m)
        if m in missing_files:
            per_month.append({"month": m, "expected": expected, "present": 0,
                              "coverage_pct": 0.0, "file_missing": True})
            continue
        mb = load_klines_from_zip(sym, m, m, base)
        unique_in_month = {b.ts for b in mb if m_start <= b.ts < m_end}
        per_month.append({
            "month": m,
            "expected": expected,
            "present": len(unique_in_month),
            "rows_in_file": len(mb),
            "rows_outside_month": sum(1 for b in mb if not (m_start <= b.ts < m_end)),
            "coverage_pct": round(100.0 * len(unique_in_month) / expected, 4),
            "file_missing": False,
        })
        bars.extend(mb)

    bars.sort(key=lambda b: b.ts)
    series_start, _, _ = month_bounds(months[0])
    _, series_end, _ = month_bounds(months[-1])

    # Gaps: incluye huecos al inicio/fin del rango esperado.
    ts_list = [b.ts for b in bars]
    duplicates = sum(1 for a, b in zip(ts_list, ts_list[1:]) if b == a)
    misaligned = sum(1 for t in ts_list if t % MIN_MS != 0)
    uniq = sorted(set(ts_list))
    points = [series_start - MIN_MS] + uniq + [series_end]

    gaps = []
    for a, b in zip(points, points[1:]):
        delta = b - a
        if delta > MIN_MS:
            missing = delta // MIN_MS - 1
            gaps.append({
                "last_bar_before": fmt(a) if a >= series_start else None,
                "first_bar_after": fmt(b) if b < series_end else None,
                "missing_from": fmt(a + MIN_MS),
                "missing_to": fmt(b - MIN_MS),
                "missing_minutes": missing,
                "duration_h": round(missing / 60, 2),
                "category": categorize(missing),
            })

    by_cat = {name: {"count": 0, "missing_minutes": 0} for name, _, _ in CATEGORIES}
    for g in gaps:
        by_cat[g["category"]]["count"] += 1
        by_cat[g["category"]]["missing_minutes"] += g["missing_minutes"]

    expected_total = sum(p["expected"] for p in per_month)
    present_total = sum(p["present"] for p in per_month)
    months_gt_99 = sum(1 for p in per_month if p["coverage_pct"] > 99.0)
    months_full = sum(1 for p in per_month if p["coverage_pct"] == 100.0)
    top = sorted(gaps, key=lambda g: g["missing_minutes"], reverse=True)[: args.top]

    report = {
        "symbol": sym,
        "interval": "1m",
        "source_dir": kdir,
        "range": {"start_month": months[0], "end_month": months[-1],
                  "start": fmt(series_start), "end_exclusive": fmt(series_end)},
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "totals": {
            "expected_bars": expected_total,
            "present_bars": present_total,
            "missing_bars": expected_total - present_total,
            "coverage_pct": round(100.0 * present_total / expected_total, 4),
            "duplicate_timestamps": duplicates,
            "misaligned_timestamps": misaligned,
        },
        "months": {
            "total": len(per_month),
            "missing_files": missing_files,
            "coverage_gt_99": months_gt_99,
            "coverage_gt_99_pct": round(100.0 * months_gt_99 / len(per_month), 2),
            "coverage_100": months_full,
        },
        "gap_categories_by_missing_minutes": {
            "definition": "<5min: 1-4 | 5min-1h: 5-60 | 1h-1dia: 61-1440 | >1dia: >1440 minutos faltantes",
            **by_cat,
        },
        "gap_count": len(gaps),
        "top_gaps": top,
        "per_month": per_month,
        "all_gaps": gaps,
    }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    t = report["totals"]
    print(f"{sym} 1m  {months[0]} → {months[-1]}  ({len(months)} meses)")
    print(f"Velas esperadas: {t['expected_bars']:,}  presentes: {t['present_bars']:,}  "
          f"faltan: {t['missing_bars']:,}  cobertura: {t['coverage_pct']}%")
    print(f"Duplicados: {duplicates}  desalineados: {misaligned}  "
          f"meses sin archivo: {missing_files or 'ninguno'}")
    print("Gaps por categoría (minutos faltantes):")
    for name, _, _ in CATEGORIES:
        c = by_cat[name]
        print(f"  {name:>8}: {c['count']:>5} gaps, {c['missing_minutes']:>8,} min")
    print(f"Meses con cobertura >99%: {months_gt_99}/{len(months)} "
          f"({report['months']['coverage_gt_99_pct']}%)  | al 100%: {months_full}")
    print("Top 10 gaps:")
    for i, g in enumerate(top[:10], 1):
        print(f"  {i:>2}. {g['missing_from']} → {g['missing_to']}  "
              f"{g['missing_minutes']:>6,} min ({g['duration_h']} h)")
    print(f"Informe completo: {out}")


if __name__ == "__main__":
    main()
