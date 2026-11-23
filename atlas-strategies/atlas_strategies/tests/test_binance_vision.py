"""Tests de lectura de Binance Vision contra la carpeta real.

Configura la ruta raíz de los datos con la variable de entorno ATLAS_BINANCE_DATA, p. ej.:

    Windows:  set ATLAS_BINANCE_DATA=G:\\Mi unidad\\Binance_Futures_Historicos
    macOS:    export ATLAS_BINANCE_DATA=~/Library/CloudStorage/GoogleDrive-.../Mi unidad/Binance_Futures_Historicos

Si no está definida o no existe el ZIP del mes, los tests se saltan (skip).
"""
import os

import pytest

from atlas_strategies.data.binance_vision import (
    load_funding_from_zip,
    load_klines_from_zip,
    resample,
)

SYMBOL = "BTCUSDT"
MONTH = "2026-10"
BASE = os.path.expanduser(os.environ.get("ATLAS_BINANCE_DATA", ""))
ZIP = os.path.join(BASE, "data", "futures", "um", "monthly", "klines",
                   SYMBOL, "1m", f"{SYMBOL}-1m-{MONTH}.zip")

pytestmark = pytest.mark.skipif(
    not BASE or not os.path.isfile(ZIP),
    reason=f"Datos reales no disponibles (ATLAS_BINANCE_DATA / {SYMBOL}-1m-{MONTH}.zip)",
)


@pytest.fixture(scope="module")
def bars_1m():
    return load_klines_from_zip(SYMBOL, MONTH, MONTH, BASE)


def test_month_has_enough_1m_bars(bars_1m):
    assert len(bars_1m) > 40000


def test_bars_sorted_and_1m_spaced(bars_1m):
    gaps = [b.ts - a.ts for a, b in zip(bars_1m, bars_1m[1:])]
    assert all(g > 0 for g in gaps)
    assert sum(1 for g in gaps if g != 60_000) == 0, "hay huecos o duplicados en 1m"


def test_resample_5m_count(bars_1m):
    bars_5m = resample(bars_1m, "5m")
    # 28–31 días -> 8064–8928 velas de 5m
    assert 8000 <= len(bars_5m) <= 9000


def test_resample_preserves_volume(bars_1m):
    bars_5m = resample(bars_1m, "5m")
    assert sum(b.volume for b in bars_5m) == pytest.approx(
        sum(b.volume for b in bars_1m), rel=1e-9)


def test_resample_first_open_last_close(bars_1m):
    bars_5m = resample(bars_1m, "5m")
    assert bars_5m[0].open == bars_1m[0].open
    assert bars_5m[-1].close == bars_1m[-1].close


@pytest.mark.parametrize("tf,ms", [("5m", 300_000), ("15m", 900_000),
                                   ("1h", 3_600_000), ("4h", 14_400_000)])
def test_resample_alignment(bars_1m, tf, ms):
    out = resample(bars_1m, tf)
    assert all(b.ts % ms == 0 for b in out)
    for b in out:
        assert b.low <= min(b.open, b.close) <= max(b.open, b.close) <= b.high


def test_funding_loads():
    funding = load_funding_from_zip(SYMBOL, MONTH, MONTH, BASE)
    # BTCUSDT cobra cada 8h -> ~90-93 registros por mes
    assert 80 <= len(funding) <= 100
    assert all(abs(r) < 0.01 for r in funding.values())
