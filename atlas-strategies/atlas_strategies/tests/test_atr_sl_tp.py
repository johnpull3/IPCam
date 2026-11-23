import pytest

from atlas_strategies.backtest.data_generator import generate_synthetic_bars
from atlas_strategies.common.executor import SLIPPAGE, build_position
from atlas_strategies.common.models import Bar, Side, SymbolFilters
from atlas_strategies.data.binance_vision import resample
from atlas_strategies.strategies.bollinger_rsi2_reversion import BollingerRsi2Reversion
from atlas_strategies.strategies.vwap_ema_rsi_scalp import VwapEmaRsiScalp


def _bars_15m():
    # 1m sintético -> 15m, ts alineados a 15m (requisito de la EMA de 1h).
    return resample(generate_synthetic_bars(n=60000), "15m")


@pytest.mark.parametrize("factory", [VwapEmaRsiScalp, BollingerRsi2Reversion])
def test_signals_are_atr_based_with_rr(factory):
    strat = factory()
    sigs = [s for s in (strat.on_bar(b) for b in _bars_15m()) if s is not None]
    assert sigs, "sin señales en el sintético; el test no comprueba nada"
    for s in sigs:
        assert s.tp_price is None and s.sl_price is None
        assert s.rr == 2.0
        assert 0 < s.sl_pct < 0.2


def test_tp_is_rr_times_sl_distance():
    strat = BollingerRsi2Reversion()
    for b in _bars_15m():
        s = strat.on_bar(b)
        if s is not None:
            pos = build_position(s, b.close, 10_000, SymbolFilters(), b.ts, 0)
            if pos is None:
                continue
            sl_d = abs(pos.entry_price - pos.sl_price)
            tp_d = abs(pos.tp_price - pos.entry_price)
            assert tp_d == pytest.approx(2.0 * sl_d, rel=1e-9)
            return
    pytest.fail("no se construyó ninguna posición")


def test_trend_ema_only_updates_on_hour_close():
    strat = VwapEmaRsiScalp()
    hour = 3_600_000
    t0 = 1_700_000_000_000 - (1_700_000_000_000 % hour)
    for i in range(8):  # 2 horas de 15m
        strat.on_bar(Bar(ts=t0 + i * 900_000, open=1, high=1, low=1, close=float(i), volume=1))
    assert strat._trend_count == 2
    # La EMA vio solo los closes de las barras :45 (i=3 e i=7).
    assert strat.trend_ema.value == pytest.approx(3 + (7 - 3) * strat.trend_ema.k)
