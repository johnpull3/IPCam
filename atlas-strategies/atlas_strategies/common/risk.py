import math
from .models import SymbolFilters


def round_to_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


def size_position(equity: float, risk_pct: float, entry: float, sl: float,
                  filters: SymbolFilters, max_leverage: float = 5.0) -> float:
    if equity <= 0 or entry <= 0:
        return 0.0

    distance = abs(entry - sl)
    if distance <= 0:
        return 0.0

    risk_amount = equity * risk_pct
    qty = risk_amount / distance
    qty = round_to_step(qty, filters.step_size)

    if qty < filters.min_qty:
        return 0.0

    notional = qty * entry
    if notional < filters.min_notional:
        return 0.0

    if notional / max_leverage > equity:
        return 0.0

    return qty
