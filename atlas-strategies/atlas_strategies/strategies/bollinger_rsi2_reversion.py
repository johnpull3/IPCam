from typing import Optional
from atlas_strategies.common.models import Bar, Signal, Side
from atlas_strategies.common.indicators import ADX, RollingATR, RollingBollinger, RollingRSI


class BollingerRsi2Reversion:
    """Reversión a la media con Bollinger + RSI(2), solo en rango (ADX <= adx_max).

    SL/TP basados en ATR: SL = atr_mult*ATR(14), TP = rr * distancia al SL.
    La banda media ya no es el objetivo.
    """

    def __init__(self, bb_len=20, bb_std=2.0, rsi_len=2,
                 atr_mult=1.5, rr=2.0, risk_pct=0.003, adx_max=30):
        self.bb = RollingBollinger(bb_len, bb_std)
        self.rsi = RollingRSI(rsi_len)
        self.adx = ADX(14)
        self.atr = RollingATR(14)
        self.atr_mult = atr_mult
        self.rr = rr
        self.risk_pct = risk_pct
        self.adx_max = adx_max

    def on_bar(self, bar: Bar, _unused=None) -> Optional[Signal]:
        if not bar.closed:
            return None

        upper, mid, lower = self.bb.update(bar.close)
        rsi = self.rsi.update(bar.close)
        adx = self.adx.update(bar.high, bar.low, bar.close)
        atr = self.atr.update(bar.high, bar.low, bar.close)

        if upper is None or rsi is None or adx is None or atr is None:
            return None
        if adx > self.adx_max or atr <= 0:
            return None

        sl_pct = self.atr_mult * atr / bar.close

        # El filtro de distancia mínima a la media se conserva como filtro de entrada.
        if bar.close < lower and rsi < 10:
            if (mid - bar.close) / bar.close >= 0.003:
                return Signal(side=Side.LONG, risk_pct=self.risk_pct,
                              sl_pct=sl_pct, rr=self.rr, tag="bb_rsi2_long")

        if bar.close > upper and rsi > 90:
            if (bar.close - mid) / bar.close >= 0.003:
                return Signal(side=Side.SHORT, risk_pct=self.risk_pct,
                              sl_pct=sl_pct, rr=self.rr, tag="bb_rsi2_short")

        return None
