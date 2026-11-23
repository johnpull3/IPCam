from collections import deque
from typing import Optional, Union

from atlas_strategies.common.indicators import RollingATR
from atlas_strategies.common.models import Bar, Position, PositionUpdate, Side, Signal


class DonchianBreakoutStrategy:
    """Ruptura del canal Donchian (pensada para 1h).

    Entrada LONG: close > máximo de los `entry_len` highs ANTERIORES (excluye la vela actual).
    SHORT simétrico con los lows.
    Stop inicial (LONG): el más alejado entre min(low de las últimas `stop_len` velas,
    incluida la de señal) y close - stop_atr*ATR. SHORT simétrico.
    Trailing: si close > entry + stop_atr*ATR, stop = close - stop_atr*ATR (solo sube).
    Salida: trailing stop o close cruzando en contra el Donchian(`exit_len`) anterior. Sin TP.
    """

    manages_position = True

    def __init__(self, entry_len=20, exit_len=10, stop_len=10, atr_len=14,
                 stop_atr=2.0, risk_pct=0.005):
        # Highs/lows de las N velas ANTERIORES a la actual.
        self.prev_highs_entry: deque = deque(maxlen=entry_len)
        self.prev_lows_entry: deque = deque(maxlen=entry_len)
        self.prev_highs_exit: deque = deque(maxlen=exit_len)
        self.prev_lows_exit: deque = deque(maxlen=exit_len)
        self.recent_lows: deque = deque(maxlen=stop_len)
        self.recent_highs: deque = deque(maxlen=stop_len)
        self.entry_len = entry_len
        self.exit_len = exit_len
        self.atr = RollingATR(atr_len)
        self.stop_atr = stop_atr
        self.risk_pct = risk_pct

    def on_bar(self, bar: Bar, position: Optional[Position] = None
               ) -> Union[Signal, PositionUpdate, None]:
        if not bar.closed:
            return None
        atr = self.atr.update(bar.high, bar.low, bar.close)
        # Canales con las velas ANTERIORES: se leen antes de añadir la actual.
        full_e = len(self.prev_highs_entry) == self.entry_len
        full_x = len(self.prev_highs_exit) == self.exit_len
        up20 = max(self.prev_highs_entry) if full_e else None
        dn20 = min(self.prev_lows_entry) if full_e else None
        up10 = max(self.prev_highs_exit) if full_x else None
        dn10 = min(self.prev_lows_exit) if full_x else None
        self.prev_highs_entry.append(bar.high)
        self.prev_lows_entry.append(bar.low)
        self.prev_highs_exit.append(bar.high)
        self.prev_lows_exit.append(bar.low)
        self.recent_lows.append(bar.low)
        self.recent_highs.append(bar.high)

        if position is not None:
            return self._manage(bar, position, up10, dn10, atr)

        if atr is None or atr <= 0 or up20 is None or dn20 is None:
            return None

        if bar.close > up20:
            sl = min(min(self.recent_lows), bar.close - self.stop_atr * atr)
            return Signal(side=Side.LONG, risk_pct=self.risk_pct, sl_price=sl,
                          use_tp=False, tag="donchian_long")
        if bar.close < dn20:
            sl = max(max(self.recent_highs), bar.close + self.stop_atr * atr)
            return Signal(side=Side.SHORT, risk_pct=self.risk_pct, sl_price=sl,
                          use_tp=False, tag="donchian_short")
        return None

    def _manage(self, bar, position, up10, dn10, atr) -> Optional[PositionUpdate]:
        if position.side == Side.LONG:
            if dn10 is not None and bar.close < dn10:
                return PositionUpdate(exit=True)
            if atr is not None and bar.close > position.entry_price + self.stop_atr * atr:
                return PositionUpdate(new_stop=max(position.sl_price,
                                                   bar.close - self.stop_atr * atr))
        else:
            if up10 is not None and bar.close > up10:
                return PositionUpdate(exit=True)
            if atr is not None and bar.close < position.entry_price - self.stop_atr * atr:
                return PositionUpdate(new_stop=min(position.sl_price,
                                                   bar.close + self.stop_atr * atr))
        return None
