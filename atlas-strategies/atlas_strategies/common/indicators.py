import math
from collections import deque
from typing import List, Optional


class RollingEMA:
    def __init__(self, period: int):
        self.period = period
        self.k = 2.0 / (period + 1)
        self.value: Optional[float] = None

    def update(self, price: float) -> float:
        if self.value is None:
            self.value = price
        else:
            self.value = price * self.k + self.value * (1 - self.k)
        return self.value


class RollingSMA:
    def __init__(self, period: int):
        self.period = period
        self.buf: deque = deque(maxlen=period)

    def update(self, price: float) -> Optional[float]:
        self.buf.append(price)
        if len(self.buf) < self.period:
            return None
        return sum(self.buf) / self.period


class RollingRSI:
    def __init__(self, period: int):
        self.period = period
        self.prev_price: Optional[float] = None
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self._warmup_gains: List[float] = []
        self._warmup_losses: List[float] = []

    def update(self, price: float) -> Optional[float]:
        if self.prev_price is None:
            self.prev_price = price
            return None

        change = price - self.prev_price
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self.prev_price = price

        if self.avg_gain is None:
            self._warmup_gains.append(gain)
            self._warmup_losses.append(loss)
            if len(self._warmup_gains) < self.period:
                return None
            self.avg_gain = sum(self._warmup_gains) / self.period
            self.avg_loss = sum(self._warmup_losses) / self.period
        else:
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        if self.avg_loss == 0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - 100.0 / (1.0 + rs)


class RollingBollinger:
    def __init__(self, period: int, k: float = 2.0):
        self.period = period
        self.k = k
        self.buf: deque = deque(maxlen=period)

    def update(self, price: float):
        self.buf.append(price)
        if len(self.buf) < self.period:
            return None, None, None
        mid = sum(self.buf) / self.period
        var = sum((x - mid) ** 2 for x in self.buf) / self.period
        std = math.sqrt(var)
        return mid + self.k * std, mid, mid - self.k * std


class SessionVWAP:
    def __init__(self):
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.day: Optional[int] = None

    def update(self, ts_ms: int, hlc3: float, volume: float) -> float:
        day = ts_ms // 86_400_000
        if self.day != day:
            self.cum_pv = 0.0
            self.cum_v = 0.0
            self.day = day
        self.cum_pv += hlc3 * volume
        self.cum_v += volume
        if self.cum_v <= 0:
            return hlc3
        return self.cum_pv / self.cum_v


class RollingATR:
    def __init__(self, period: int):
        self.period = period
        self.prev_close: Optional[float] = None
        self.values: deque = deque(maxlen=period)
        self.atr: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.prev_close = close

        if self.atr is None:
            self.values.append(tr)
            if len(self.values) < self.period:
                return None
            self.atr = sum(self.values) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        return self.atr


class RollingHighLow:
    def __init__(self, period: int):
        self.period = period
        self.highs: deque = deque(maxlen=period + 1)
        self.lows: deque = deque(maxlen=period + 1)

    def update(self, high: float, low: float):
        self.highs.append(high)
        self.lows.append(low)

    def highest_excluding_current(self) -> Optional[float]:
        if len(self.highs) < 2:
            return None
        return max(list(self.highs)[:-1])

    def lowest_excluding_current(self) -> Optional[float]:
        if len(self.lows) < 2:
            return None
        return min(list(self.lows)[:-1])


class ADX:
    def __init__(self, period: int = 14):
        self.period = period
        self.prev_high: Optional[float] = None
        self.prev_low: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.tr_buf: deque = deque(maxlen=period)
        self.dm_plus_buf: deque = deque(maxlen=period)
        self.dm_minus_buf: deque = deque(maxlen=period)
        self.dx_buf: deque = deque(maxlen=period)
        self.adx: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self.prev_high is None:
            self.prev_high, self.prev_low, self.prev_close = high, low, close
            return None

        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        up = high - self.prev_high
        down = self.prev_low - low
        dm_plus = up if up > down and up > 0 else 0.0
        dm_minus = down if down > up and down > 0 else 0.0

        self.prev_high, self.prev_low, self.prev_close = high, low, close
        self.tr_buf.append(tr)
        self.dm_plus_buf.append(dm_plus)
        self.dm_minus_buf.append(dm_minus)

        if len(self.tr_buf) < self.period:
            return None

        tr_sum = sum(self.tr_buf)
        if tr_sum == 0:
            return None
        di_plus = 100 * sum(self.dm_plus_buf) / tr_sum
        di_minus = 100 * sum(self.dm_minus_buf) / tr_sum
        di_sum = di_plus + di_minus
        dx = 0.0 if di_sum == 0 else 100 * abs(di_plus - di_minus) / di_sum

        self.dx_buf.append(dx)
        if len(self.dx_buf) < self.period:
            return None

        self.adx = sum(self.dx_buf) / self.period
        return self.adx
