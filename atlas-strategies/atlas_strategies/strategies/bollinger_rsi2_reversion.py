from typing import Optional
from atlas_strategies.common.models import Bar, Signal, Side
from atlas_strategies.common.indicators import RollingBollinger, RollingRSI, ADX


class BollingerRsi2Reversion:
    def __init__(self, bb_len=20, bb_std=2.0, rsi_len=2,
                 sl_pct=0.015, risk_pct=0.003, adx_max=30):
        self.bb = RollingBollinger(bb_len, bb_std)
        self.rsi = RollingRSI(rsi_len)
        self.adx = ADX(14)
        self.sl_pct = sl_pct
        self.risk_pct = risk_pct
        self.adx_max = adx_max

    def on_bar(self, bar: Bar, _unused=None) -> Optional[Signal]:
        if not bar.closed:
            return None

        upper, mid, lower = self.bb.update(bar.close)
        rsi = self.rsi.update(bar.close)
        adx = self.adx.update(bar.high, bar.low, bar.close)

        if upper is None or rsi is None or adx is None:
            return None
        if adx > self.adx_max:
            return None

        if bar.close < lower and rsi < 10:
            if (mid - bar.close) / bar.close >= 0.003:
                return Signal(side=Side.LONG, risk_pct=self.risk_pct,
                              sl_pct=self.sl_pct, tp_price=mid, tag="bb_rsi2_long")

        if bar.close > upper and rsi > 90:
            if (bar.close - mid) / bar.close >= 0.003:
                return Signal(side=Side.SHORT, risk_pct=self.risk_pct,
                              sl_pct=self.sl_pct, tp_price=mid, tag="bb_rsi2_short")

        return None
