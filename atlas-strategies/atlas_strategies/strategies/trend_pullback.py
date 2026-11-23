from typing import Optional, Union

from atlas_strategies.common.indicators import RollingATR, RollingEMA
from atlas_strategies.common.models import Bar, Position, PositionUpdate, Side, Signal


class TrendPullbackStrategy:
    """Pullback a la EMA20 a favor de la tendencia EMA50/EMA200 (pensada para 4h).

    Entrada LONG: EMA50 > EMA200, la vela toca la EMA20 (low <= EMA20 <= high) y cierra
    alcista. SHORT simétrico. Entrada al open siguiente (la pone el engine).
    Stop inicial: low - stop_atr*ATR (LONG) / high + stop_atr*ATR (SHORT).
    Trailing: con cada nuevo mejor close desde la entrada, stop = close -/+ stop_atr*ATR.
    Salida: trailing stop o cruce EMA50/EMA200 en contra. Sin TP.
    """

    manages_position = True

    def __init__(self, ema_fast=20, ema_mid=50, ema_slow=200, atr_len=14,
                 stop_atr=2.0, risk_pct=0.005):
        self.ema_fast = RollingEMA(ema_fast)
        self.ema_mid = RollingEMA(ema_mid)
        self.ema_slow = RollingEMA(ema_slow)
        self.atr = RollingATR(atr_len)
        self.warmup = ema_slow
        self.stop_atr = stop_atr
        self.risk_pct = risk_pct
        self._count = 0
        self._pos_key = None
        self._best_close = None

    def on_bar(self, bar: Bar, position: Optional[Position] = None
               ) -> Union[Signal, PositionUpdate, None]:
        if not bar.closed:
            return None
        e20 = self.ema_fast.update(bar.close)
        e50 = self.ema_mid.update(bar.close)
        e200 = self.ema_slow.update(bar.close)
        atr = self.atr.update(bar.high, bar.low, bar.close)
        self._count += 1

        if position is not None:
            return self._manage(bar, position, e50, e200, atr)

        if self._count < self.warmup or atr is None or atr <= 0:
            return None

        touches = bar.low <= e20 <= bar.high
        if e50 > e200 and touches and bar.close > bar.open:
            return Signal(side=Side.LONG, risk_pct=self.risk_pct,
                          sl_price=bar.low - self.stop_atr * atr, use_tp=False,
                          tag="trend_pullback_long")
        if e50 < e200 and touches and bar.close < bar.open:
            return Signal(side=Side.SHORT, risk_pct=self.risk_pct,
                          sl_price=bar.high + self.stop_atr * atr, use_tp=False,
                          tag="trend_pullback_short")
        return None

    def _manage(self, bar, position, e50, e200, atr) -> Optional[PositionUpdate]:
        key = (position.entry_ts, position.side)
        if key != self._pos_key:
            self._pos_key = key
            self._best_close = position.entry_price

        if position.side == Side.LONG:
            if e50 < e200:
                return PositionUpdate(exit=True)
            if atr is not None and bar.close > self._best_close:
                self._best_close = bar.close
                return PositionUpdate(new_stop=max(position.sl_price,
                                                   bar.close - self.stop_atr * atr))
        else:
            if e50 > e200:
                return PositionUpdate(exit=True)
            if atr is not None and bar.close < self._best_close:
                self._best_close = bar.close
                return PositionUpdate(new_stop=min(position.sl_price,
                                                   bar.close + self.stop_atr * atr))
        return None
