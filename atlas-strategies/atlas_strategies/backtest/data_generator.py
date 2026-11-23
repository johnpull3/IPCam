import random
from atlas_strategies.common.models import Bar


def generate_synthetic_bars(n: int = 20000, start_price: float = 50000,
                             seed: int = 42) -> list[Bar]:
    """Genera velas OHLCV sintéticas con fases trend up / trend down / range."""
    rng = random.Random(seed)
    bars = []
    price = start_price
    ts = 1_700_000_000_000
    for i in range(n):
        phase = (i // 500) % 3
        drift = 0.0002 if phase == 0 else (-0.0002 if phase == 1 else 0.0)
        ret = drift + rng.gauss(0, 0.003)
        o = price
        c = o * (1 + ret)
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.001)))
        l = min(o, c) * (1 - abs(rng.gauss(0, 0.001)))
        v = abs(rng.gauss(100, 30)) + 10
        bars.append(Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v))
        price = c
        ts += 60_000  # 1 min
    return bars
