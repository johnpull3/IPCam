import pytest

from atlas_strategies.backtest.data_generator import generate_synthetic_bars
from atlas_strategies.backtest.run_backtest import run_backtest
from atlas_strategies.common.executor import maker_fill_price
from atlas_strategies.common.models import Bar, Side, Signal
from atlas_strategies.strategies.swing_reclaim import SwingReclaim


def _bar(low, high):
    return Bar(ts=0, open=(low + high) / 2, high=high, low=low, close=(low + high) / 2, volume=1)


def test_maker_long_fills_only_if_low_touches_limit():
    sig = Signal(side=Side.LONG, risk_pct=0.01, sl_pct=0.01, rr=2.0)
    assert maker_fill_price(sig, 100.0, _bar(99.5, 101)) == 100.0
    assert maker_fill_price(sig, 100.0, _bar(100.0, 101)) == 100.0
    assert maker_fill_price(sig, 100.0, _bar(100.1, 101)) is None


def test_maker_short_fills_only_if_high_touches_limit():
    sig = Signal(side=Side.SHORT, risk_pct=0.01, sl_pct=0.01, rr=2.0)
    assert maker_fill_price(sig, 100.0, _bar(99, 100.5)) == 100.0
    assert maker_fill_price(sig, 100.0, _bar(99, 100.0)) == 100.0
    assert maker_fill_price(sig, 100.0, _bar(99, 99.9)) is None


def test_modes_account_signals():
    bars = generate_synthetic_bars(n=15000)
    for mode in ("taker", "maker"):
        r = run_backtest(SwingReclaim(), bars, execution_mode=mode)
        assert r["execution_mode"] == mode
        assert r["missed_signals"] <= r["signals"]
        if mode == "taker":
            assert r["missed_signals"] == 0


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        run_backtest(SwingReclaim(), generate_synthetic_bars(n=100), execution_mode="foo")
