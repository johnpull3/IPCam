from typing import Optional
from atlas_strategies.common.models import Bar, Signal, Side
from atlas_strategies.common.indicators import (
    RollingEMA, RollingHighLow, RollingATR, RollingSMA
)


class SwingReclaim:
    def __init__(self, swing_lookback=20, sweep_min_pct=0.003, sweep_max_pct=0.010,
                 ema_len=50, vol_ma=20, risk_pct=0.005, rr=3.0):
        self.ema = RollingEMA(ema_len)
        self.hl = RollingHighLow(swing_lookback)
        self.atr = RollingATR(14)
        self.vol_ma = RollingSMA(vol_ma)
        self.sweep_min_pct = sweep_min_pct
        self.sweep_max_pct = sweep_max_pct
        self.risk_pct = risk_pct
        self.rr = rr

    def on_bar(self, bar: Bar, _unused=None) -> Optional[Signal]:
        if not bar.closed:
            return None

        ema = self.ema.update(bar.close)
        swing_low = self.hl.lowest_excluding_current()
        swing_high = self.hl.highest_excluding_current()
        atr = self.atr.update(bar.high, bar.low, bar.close)
        vol = self.vol_ma.update(bar.volume)

        # Actualiza el lookback DESPUÉS de leerlo (evita lookahead)
        self.hl.update(bar.high, bar.low)

        if ema is None or swing_low is None or swing_high is None or atr is None or vol is None:
            return None

        vol_ok = bar.volume > vol
        trend_up = bar.close > ema
        trend_down = bar.close < ema

        sweep_low = bar.low < swing_low * (1 - self.sweep_min_pct)
        not_deep_low = bar.low > swing_low * (1 - self.sweep_max_pct)
        reclaim_low = bar.close > swing_low

        if trend_up and sweep_low and not_deep_low and reclaim_low and vol_ok:
            sl = bar.low - 0.5 * atr
            return Signal(side=Side.LONG, risk_pct=self.risk_pct,
                          sl_price=sl, rr=self.rr, tag="swing_reclaim_long")

        sweep_high = bar.high > swing_high * (1 + self.sweep_min_pct)
        not_deep_high = bar.high < swing_high * (1 + self.sweep_max_pct)
        reclaim_high = bar.close < swing_high

        if trend_down and sweep_high and not_deep_high and reclaim_high and vol_ok:
            sl = bar.high + 0.5 * atr
            return Signal(side=Side.SHORT, risk_pct=self.risk_pct,
                          sl_price=sl, rr=self.rr, tag="swing_reclaim_short")

        return None
