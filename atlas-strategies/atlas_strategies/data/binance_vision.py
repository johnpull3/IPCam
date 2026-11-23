"""Lectura de datos históricos de Binance Vision (futuros USD-M) desde disco local.

Estructura esperada (la misma que data.binance.vision):

    <base_path>/data/futures/um/monthly/klines/<SYMBOL>/1m/<SYMBOL>-1m-YYYY-MM.zip
    <base_path>/data/futures/um/monthly/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate-YYYY-MM.zip

`base_path` es la carpeta raíz (p. ej. "G:\\Mi unidad\\Binance_Futures_Historicos").
También se acepta que `base_path` apunte directamente a ".../data/futures/um".
"""
import csv
import io
import os
import zipfile
from typing import Dict, Iterator, List, Tuple

from atlas_strategies.common.models import Bar

TIMEFRAMES_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
}


def _um_root(base_path: str) -> str:
    candidate = os.path.join(base_path, "data", "futures", "um")
    if os.path.isdir(candidate):
        return candidate
    return base_path


def _parse_month(month: str) -> Tuple[int, int]:
    year, mon = month.split("-")
    y, m = int(year), int(mon)
    if not 1 <= m <= 12:
        raise ValueError(f"Mes inválido: {month!r} (formato YYYY-MM)")
    return y, m


def _iter_months(start_month: str, end_month: str) -> Iterator[str]:
    y, m = _parse_month(start_month)
    ey, em = _parse_month(end_month)
    if (y, m) > (ey, em):
        raise ValueError(f"start_month {start_month} es posterior a end_month {end_month}")
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _to_ms(ts: int) -> int:
    # Binance pasó algunos datasets a microsegundos; normalizamos a ms.
    return ts // 1000 if ts > 10**14 else ts


def _read_csv_rows(zip_path: str) -> List[List[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{zip_path}: se esperaba 1 CSV, hay {len(names)}")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            return [row for row in csv.reader(text) if row]


def _is_header(row: List[str]) -> bool:
    try:
        int(row[0])
        return False
    except ValueError:
        return True


def load_klines_from_zip(symbol: str, start_month: str, end_month: str,
                         base_path: str) -> List[Bar]:
    """Carga velas de 1m de los ZIP mensuales [start_month, end_month] (YYYY-MM, inclusive).

    Lanza FileNotFoundError si falta algún mes: no se rellenan huecos en silencio.
    """
    root = os.path.join(_um_root(base_path), "monthly", "klines", symbol, "1m")
    bars: List[Bar] = []
    for month in _iter_months(start_month, end_month):
        path = os.path.join(root, f"{symbol}-1m-{month}.zip")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        for row in _read_csv_rows(path):
            if _is_header(row):
                continue
            bars.append(Bar(
                ts=_to_ms(int(row[0])),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                closed=True,
            ))
    bars.sort(key=lambda b: b.ts)
    return bars


def load_funding_from_zip(symbol: str, start_month: str, end_month: str,
                          base_path: str) -> Dict[int, float]:
    """Devuelve {calc_time_ms: last_funding_rate} para los meses indicados."""
    root = os.path.join(_um_root(base_path), "monthly", "fundingRate", symbol)
    funding: Dict[int, float] = {}
    for month in _iter_months(start_month, end_month):
        path = os.path.join(root, f"{symbol}-fundingRate-{month}.zip")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        rows = _read_csv_rows(path)
        rate_col = 2
        if rows and _is_header(rows[0]):
            header = [h.strip().lower() for h in rows[0]]
            if "last_funding_rate" in header:
                rate_col = header.index("last_funding_rate")
            rows = rows[1:]
        for row in rows:
            funding[_to_ms(int(row[0]))] = float(row[rate_col])
    return dict(sorted(funding.items()))


def resample(bars_1m: List[Bar], target_tf: str) -> List[Bar]:
    """Agrega velas de 1m a 5m, 15m, 1h o 4h.

    open = primer open, high = max, low = min, close = último close,
    volume = suma, ts = inicio del bucket alineado al timeframe (UTC).
    """
    if target_tf not in TIMEFRAMES_MS or target_tf == "1m":
        raise ValueError(f"target_tf no soportado: {target_tf!r} (usa 5m, 15m, 1h o 4h)")
    tf_ms = TIMEFRAMES_MS[target_tf]

    out: List[Bar] = []
    cur: Bar = None
    for b in bars_1m:
        bucket = b.ts - (b.ts % tf_ms)
        if cur is None or bucket != cur.ts:
            if cur is not None:
                out.append(cur)
            cur = Bar(ts=bucket, open=b.open, high=b.high, low=b.low,
                      close=b.close, volume=b.volume, closed=True)
        else:
            cur.high = max(cur.high, b.high)
            cur.low = min(cur.low, b.low)
            cur.close = b.close
            cur.volume += b.volume
    if cur is not None:
        out.append(cur)
    return out
