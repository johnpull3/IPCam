import pytest

from atlas_strategies.backtest.data_generator import generate_synthetic_bars
from atlas_strategies.backtest.run_backtest import run_backtest
from atlas_strategies.common.models import Bar, Side
from atlas_strategies.data.binance_vision import resample
from atlas_strategies.strategies.donchian_breakout import DonchianBreakoutStrategy
from atlas_strategies.strategies.trend_pullback import TrendPullbackStrategy

NEW = [TrendPullbackStrategy, DonchianBreakoutStrategy]


def _bars_1h():
    return resample(generate_synthetic_bars(n=120_000), "1h")


# 1. No-lookahead estricto (mismo patrón que test_no_lookahead_strict)
@pytest.mark.parametrize("factory", NEW)
def test_no_lookahead_strict(factory):
    bars = _bars_1h()
    s_full, s_trunc = factory(), factory()
    full = [s_full.on_bar(b) for b in bars]
    trunc = [s_trunc.on_bar(b) for b in bars[:-1]]
    for i in range(len(trunc)):
        a, b = full[i], trunc[i]
        if (a is None) != (b is None):
            raise AssertionError(f"Lookahead detectado en barra {i}: full={a}, trunc={b}")
        if a is not None:
            assert a.side == b.side and a.tag == b.tag and a.sl_price == b.sl_price


# 2. El trailing stop nunca baja (LONG) / nunca sube (SHORT), engine + estrategia
@pytest.mark.parametrize("factory", NEW)
def test_trailing_stop_never_loosens(factory):
    history = {}

    class Recorder(factory):
        def on_bar(self, bar, position=None):
            if position is not None:
                history.setdefault((position.entry_ts, position.side), []).append(position.sl_price)
            return super().on_bar(bar, position)

    res = run_backtest(Recorder(), _bars_1h())
    assert res["n_trades"] > 0
    moved = 0
    for (_, side), stops in history.items():
        pairs = list(zip(stops, stops[1:]))
        moved += sum(1 for a, b in pairs if a != b)
        if side == Side.LONG:
            assert all(b >= a for a, b in pairs), stops
        else:
            assert all(b <= a for a, b in pairs), stops
    assert moved > 0, "el trailing nunca se movió; el test no comprobaría nada"


# 3. Donchian excluye la vela actual del canal
def test_donchian_excludes_current_bar():
    def mk(i, high, low, close):
        return Bar(ts=i * 3_600_000, open=close, high=high, low=low, close=close, volume=1)

    flat = [mk(i, 101.0, 99.0, 100.0) for i in range(30)]
    # Rompe el máximo previo (101) cerrando en 120 con high 150.
    breakout = mk(30, 150.0, 99.5, 120.0)
    s = DonchianBreakoutStrategy()
    for b in flat:
        assert s.on_bar(b) is None
    sig = s.on_bar(breakout)
    # Si el canal incluyera la vela actual (max=150), close 120 no rompería.
    assert sig is not None and sig.side == Side.LONG

    s2 = DonchianBreakoutStrategy()
    for b in flat:
        s2.on_bar(b)
    assert s2.on_bar(mk(30, 101.5, 99.5, 101.0)) is None  # close 101 no supera 101


# 4. Stop inicial de TrendPullback por debajo del low de la vela de señal (LONG)
def test_trend_pullback_initial_stop_below_signal_low():
    s = TrendPullbackStrategy()
    longs = shorts = 0
    for b in _bars_1h():
        sig = s.on_bar(b)
        if sig is None:
            continue
        if sig.side == Side.LONG:
            longs += 1
            assert sig.sl_price < b.low
        else:
            shorts += 1
            assert sig.sl_price > b.high
        assert sig.use_tp is False
    assert longs > 0


def test_existing_strategies_unaffected_by_manage_hook():
    from atlas_strategies.strategies.swing_reclaim import SwingReclaim
    assert not getattr(SwingReclaim(), "manages_position", False)
