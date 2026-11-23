from collections import deque
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

    regime_filter (opcional, por defecto desactivado): EMA200 sobre velas DIARIAS construidas
    a partir de las barras recibidas. Una vela diaria se cierra con la barra cuyo final cae en
    00:00 UTC. LONG solo si EMA200d[d] > EMA200d[d-20] y close diario > EMA200d; SHORT simétrico.
    Usa solo el último día cerrado. Requiere `regime_ema_len` días + `regime_slope_days` de
    calentamiento; hasta entonces no se permiten entradas con el filtro activo.
    """

    DAY_MS = 86_400_000

    manages_position = True

    def __init__(self, ema_fast=20, ema_mid=50, ema_slow=200, atr_len=14,
                 stop_atr=2.0, risk_pct=0.005,
                 regime_filter=False, regime_ema_len=200, regime_slope_days=20):
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
        self.regime_filter = regime_filter
        self.regime_ema_len = regime_ema_len
        self.d_ema = RollingEMA(regime_ema_len)
        self.d_ema_hist: deque = deque(maxlen=regime_slope_days + 1)
        self.d_count = 0
        self.d_close: Optional[float] = None
        self._prev_ts: Optional[int] = None
        self._interval: Optional[int] = None

    def _update_daily(self, bar: Bar) -> None:
        if self._prev_ts is not None:
            iv = bar.ts - self._prev_ts
            if iv > 0:
                self._interval = iv if self._interval is None else min(self._interval, iv)
        self._prev_ts = bar.ts
        if self._interval and (bar.ts + self._interval) % self.DAY_MS == 0:
            self.d_ema_hist.append(self.d_ema.update(bar.close))
            self.d_count += 1
            self.d_close = bar.close

    def regime(self) -> Optional[str]:
        """'up', 'down', 'flat' o None si el filtro diario aún no está calentado."""
        if self.d_count < self.regime_ema_len or len(self.d_ema_hist) < self.d_ema_hist.maxlen:
            return None
        e_now, e_past = self.d_ema_hist[-1], self.d_ema_hist[0]
        if e_now > e_past and self.d_close > e_now:
            return "up"
        if e_now < e_past and self.d_close < e_now:
            return "down"
        return "flat"

    def on_bar(self, bar: Bar, position: Optional[Position] = None
               ) -> Union[Signal, PositionUpdate, None]:
        if not bar.closed:
            return None
        e20 = self.ema_fast.update(bar.close)
        e50 = self.ema_mid.update(bar.close)
        e200 = self.ema_slow.update(bar.close)
        atr = self.atr.update(bar.high, bar.low, bar.close)
        self._count += 1
        self._update_daily(bar)

        if position is not None:
            return self._manage(bar, position, e50, e200, atr)

        if self._count < self.warmup or atr is None or atr <= 0:
            return None

        touches = bar.low <= e20 <= bar.high
        allow_long = allow_short = True
        if self.regime_filter:
            reg = self.regime()
            allow_long, allow_short = reg == "up", reg == "down"
        if allow_long and e50 > e200 and touches and bar.close > bar.open:
            return Signal(side=Side.LONG, risk_pct=self.risk_pct,
                          sl_price=bar.low - self.stop_atr * atr, use_tp=False,
                          tag="trend_pullback_long")
        if allow_short and e50 < e200 and touches and bar.close < bar.open:
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
