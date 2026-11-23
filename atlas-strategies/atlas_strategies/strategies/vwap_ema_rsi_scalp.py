from typing import Optional
from atlas_strategies.common.models import Bar, Signal, Side
from atlas_strategies.common.indicators import RollingEMA, RollingRSI, SessionVWAP


class VwapEmaRsiScalp:
    def __init__(self, ema_fast=9, ema_slow=21, rsi_len=14,
                 sl_pct=0.008, rr=2.0, risk_pct=0.003):
        self.ema_fast = RollingEMA(ema_fast)
        self.ema_slow = RollingEMA(ema_slow)
        self.rsi = RollingRSI(rsi_len)
        self.vwap = SessionVWAP()
        self.sl_pct = sl_pct
        self.rr = rr
        self.risk_pct = risk_pct
        self._prev_ema_fast = None
        self._prev_ema_slow = None

    def on_bar(self, bar: Bar, _unused=None) -> Optional[Signal]:
        if not bar.closed:
            return None

        hlc3 = (bar.high + bar.low + bar.close) / 3
        vwap = self.vwap.update(bar.ts, hlc3, bar.volume)
        ema_f = self.ema_fast.update(bar.close)
        ema_s = self.ema_slow.update(bar.close)
        rsi = self.rsi.update(bar.close)

        prev_f, prev_s = self._prev_ema_fast, self._prev_ema_slow
        self._prev_ema_fast = ema_f
        self._prev_ema_slow = ema_s

        if rsi is None or prev_f is None or prev_s is None:
            return None

        trend_up = bar.close > vwap
        trend_down = bar.close < vwap

        long_cross = prev_f <= prev_s and ema_f > ema_s
        short_cross = prev_f >= prev_s and ema_f < ema_s

        if trend_up and long_cross and rsi > 50:
            return Signal(side=Side.LONG, risk_pct=self.risk_pct,
                          sl_pct=self.sl_pct, rr=self.rr, tag="vwap_ema_rsi_long")

        if trend_down and short_cross and rsi < 50:
            return Signal(side=Side.SHORT, risk_pct=self.risk_pct,
                          sl_pct=self.sl_pct, rr=self.rr, tag="vwap_ema_rsi_short")

        return None
