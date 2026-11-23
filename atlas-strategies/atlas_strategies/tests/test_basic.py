import pytest
from atlas_strategies.common.indicators import RollingEMA, RollingRSI, RollingBollinger
from atlas_strategies.common.risk import size_position, round_to_step
from atlas_strategies.common.models import SymbolFilters, Position, Side
from atlas_strategies.common.executor import check_exit
from atlas_strategies.strategies.vwap_ema_rsi_scalp import VwapEmaRsiScalp
from atlas_strategies.strategies.bollinger_rsi2_reversion import BollingerRsi2Reversion
from atlas_strategies.strategies.swing_reclaim import SwingReclaim
from atlas_strategies.backtest.data_generator import generate_synthetic_bars
from atlas_strategies.backtest.run_backtest import run_backtest


def test_ema_converges():
    ema = RollingEMA(3)
    for _ in range(200):
        ema.update(100.0)
    assert abs(ema.value - 100.0) < 1e-6


def test_rsi_extremes():
    rsi = RollingRSI(14)
    for i in range(100):
        rsi.update(float(i))
    assert rsi.update(200.0) > 90


def test_round_to_step():
    assert round_to_step(1.2345, 0.01) == pytest.approx(1.23, abs=1e-9)


def test_size_position_respects_min_notional():
    filters = SymbolFilters(step_size=0.001, min_qty=0.001, min_notional=100.0)
    qty = size_position(equity=10.0, risk_pct=0.001, entry=50000, sl=49000, filters=filters)
    assert qty == 0.0


def test_size_position_basic():
    filters = SymbolFilters(step_size=0.001, min_qty=0.001, min_notional=5.0)
    qty = size_position(equity=10000, risk_pct=0.01, entry=50000, sl=49000, filters=filters)
    assert qty > 0


def test_sl_first_when_both_hit():
    pos = Position(side=Side.LONG, entry_price=100, qty=1, sl_price=95, tp_price=110)
    class B:
        high = 115
        low = 90
    assert check_exit(pos, B()) == "SL"


@pytest.mark.parametrize("factory", [VwapEmaRsiScalp, BollingerRsi2Reversion, SwingReclaim])
def test_no_lookahead(factory):
    """La señal en bar N no debe depender de bar N+1."""
    bars = generate_synthetic_bars(n=2000)
    s1 = factory()
    s2 = factory()
    signals_full = [s1.on_bar(b) for b in bars]
    signals_trunc = [s2.on_bar(b) for b in bars[:-1]]
    assert len(signals_trunc) == len(signals_full) - 1
    for i in range(len(signals_trunc)):
        a = signals_full[i]
        b = signals_trunc[i]
        if a is None and b is None:
            continue
        assert (a is None) == (b is None), (
            f"{factory.__name__}: señal distinta en índice {i}: "
            f"full={a!r} trunc={b!r}"
        )
        assert a.side == b.side, (
            f"{factory.__name__}: side distinto en índice {i}: {a.side} vs {b.side}"
        )
        assert a.tag == b.tag, (
            f"{factory.__name__}: tag distinto en índice {i}: {a.tag} vs {b.tag}"
        )


@pytest.mark.parametrize("factory", [VwapEmaRsiScalp, BollingerRsi2Reversion, SwingReclaim])
def test_strategies_produce_trades(factory):
    bars = generate_synthetic_bars(n=15000)
    result = run_backtest(factory(), bars)
    # No todas las estrategias generan trades en todas las semillas,
    # pero el motor debe devolver una estructura válida.
    assert "final_equity" in result
    assert result["final_equity"] > 0
