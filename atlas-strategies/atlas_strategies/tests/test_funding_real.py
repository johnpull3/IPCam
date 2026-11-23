import pytest

from atlas_strategies.backtest.run_backtest import run_backtest
from atlas_strategies.common.models import Bar, Side, Signal

H = 3_600_000


class OneShot:
    """Abre una posición en la primera vela y nunca sale (SL muy lejano)."""

    def __init__(self, side):
        self.side, self.done = side, False

    def on_bar(self, bar, _=None):
        if self.done:
            return None
        self.done = True
        sl = 1.0 if self.side == Side.LONG else 199.0
        return Signal(side=self.side, risk_pct=0.01, sl_price=sl, use_tp=False)


def _bars(n=12):
    return [Bar(ts=i * 4 * H, open=100, high=100, low=100, close=100, volume=1) for i in range(n)]


@pytest.mark.parametrize("side,rate,sign", [
    (Side.LONG, 0.001, -1), (Side.LONG, -0.001, +1),
    (Side.SHORT, 0.001, +1), (Side.SHORT, -0.001, -1),
])
def test_funding_sign(side, rate, sign):
    events = {8 * H + 1: rate, 16 * H + 3: rate}
    r = run_backtest(OneShot(side), _bars(), funding_events=events)
    eq = r["equity_curve"]
    delta = eq[-1] - r["initial_equity"]
    assert sign * delta > 0
    assert abs(abs(delta) - 2 * abs(rate) * 100 * _qty(r, eq)) < 1e-9


def _qty(r, eq):
    # qty implícita: el único cambio de equity es el funding (2 eventos, close=100)
    return abs(eq[-1] - r["initial_equity"]) / (2 * 0.001 * 100)


def test_no_events_no_funding():
    r = run_backtest(OneShot(Side.LONG), _bars(), funding_events={})
    assert r["equity_curve"][-1] == r["initial_equity"]


def test_events_outside_position_ignored():
    # Evento antes de la entrada (la posición entra al open de la vela 1, ts=4h)
    r = run_backtest(OneShot(Side.LONG), _bars(), funding_events={1 * H: 0.01})
    assert r["equity_curve"][-1] == r["initial_equity"]
