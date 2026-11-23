from typing import Optional
from atlas_strategies.common.models import Bar, Signal, Side
from atlas_strategies.common.indicators import (
    ADX, RollingATR, RollingEMA, RollingRSI, SessionVWAP
)

HOUR_MS = 3_600_000


class VwapEmaRsiScalp:
    """Cruce EMA rápida/lenta con filtros VWAP de sesión, EMA200 en 1h, ADX y RSI.

    Pensada para barras base de 15m (`base_tf_minutes`). La EMA200 de 1h se
    actualiza solo con el close de la barra que cierra cada hora (sin lookahead).
    SL/TP basados en ATR: SL = atr_mult*ATR(14), TP = rr * distancia al SL.
    """

    def __init__(self, ema_fast=9, ema_slow=21, rsi_len=14,
                 atr_mult=1.5, rr=2.0, risk_pct=0.003,
                 base_tf_minutes=15, trend_ema_len=200, adx_min=20):
        self.ema_fast = RollingEMA(ema_fast)
        self.ema_slow = RollingEMA(ema_slow)
        self.rsi = RollingRSI(rsi_len)
        self.vwap = SessionVWAP()
        self.atr = RollingATR(14)
        self.adx = ADX(14)
        self.atr_mult = atr_mult
        self.rr = rr
        self.risk_pct = risk_pct
        self.base_ms = base_tf_minutes * 60_000
        self.trend_ema = RollingEMA(trend_ema_len)
        self.trend_ema_len = trend_ema_len
        self._trend_count = 0
        self.adx_min = adx_min
        self._prev_ema_fast = None
        self._prev_ema_slow = None

    def _update_trend(self, bar: Bar):
        # La barra que cierra la hora es aquella cuyo fin coincide con un múltiplo de 1h.
        if (bar.ts + self.base_ms) % HOUR_MS == 0:
            self.trend_ema.update(bar.close)
            self._trend_count += 1

    def on_bar(self, bar: Bar, _unused=None) -> Optional[Signal]:
        if not bar.closed:
            return None

        hlc3 = (bar.high + bar.low + bar.close) / 3
        vwap = self.vwap.update(bar.ts, hlc3, bar.volume)
        ema_f = self.ema_fast.update(bar.close)
        ema_s = self.ema_slow.update(bar.close)
        rsi = self.rsi.update(bar.close)
        atr = self.atr.update(bar.high, bar.low, bar.close)
        adx = self.adx.update(bar.high, bar.low, bar.close)
        self._update_trend(bar)

        prev_f, prev_s = self._prev_ema_fast, self._prev_ema_slow
        self._prev_ema_fast = ema_f
        self._prev_ema_slow = ema_s

        if rsi is None or prev_f is None or prev_s is None or atr is None or adx is None:
            return None
        if self._trend_count < self.trend_ema_len:
            return None
        if adx <= self.adx_min or atr <= 0:
            return None

        ema_1h = self.trend_ema.value
        trend_up = bar.close > vwap and bar.close > ema_1h
        trend_down = bar.close < vwap and bar.close < ema_1h

        long_cross = prev_f <= prev_s and ema_f > ema_s
        short_cross = prev_f >= prev_s and ema_f < ema_s

        # SL relativo al precio de entrada: atr_mult * ATR expresado como fracción.
        sl_pct = self.atr_mult * atr / bar.close

        if trend_up and long_cross and rsi > 50:
            return Signal(side=Side.LONG, risk_pct=self.risk_pct,
                          sl_pct=sl_pct, rr=self.rr, tag="vwap_ema_rsi_long")

        if trend_down and short_cross and rsi < 50:
            return Signal(side=Side.SHORT, risk_pct=self.risk_pct,
                          sl_pct=sl_pct, rr=self.rr, tag="vwap_ema_rsi_short")

        return None
