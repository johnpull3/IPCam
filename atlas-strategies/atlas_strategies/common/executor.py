from typing import Optional
from .models import Signal, Position, SymbolFilters, Side
from .risk import size_position


TAKER_FEE = 0.0004
SLIPPAGE = 0.0002
MAKER_FEE = 0.0002

EXECUTION_MODES = ("taker", "maker")


def maker_fill_price(signal: Signal, signal_close: float, next_bar) -> Optional[float]:
    """LIMIT al close de la vela de señal. Se llena solo si la vela siguiente toca el precio.

    LONG: low <= limit. SHORT: high >= limit. Devuelve el precio límite o None si no llena.
    """
    limit = signal_close
    if signal.side == Side.LONG:
        return limit if next_bar.low <= limit else None
    return limit if next_bar.high >= limit else None


def build_position(signal: Signal, entry_open: float, equity: float,
                   filters: SymbolFilters, entry_ts: int, entry_index: int,
                   slippage: float = SLIPPAGE) -> Optional[Position]:
    if signal.side == Side.LONG:
        fill = entry_open * (1 + slippage)
        if signal.sl_price is not None:
            sl = signal.sl_price
        elif signal.sl_pct is not None:
            sl = fill * (1 - signal.sl_pct)
        else:
            return None

        if signal.tp_price is not None:
            tp = signal.tp_price
        elif signal.rr is not None:
            tp = fill + signal.rr * (fill - sl)
        else:
            tp = fill * 1.10
    else:
        fill = entry_open * (1 - slippage)
        if signal.sl_price is not None:
            sl = signal.sl_price
        elif signal.sl_pct is not None:
            sl = fill * (1 + signal.sl_pct)
        else:
            return None

        if signal.tp_price is not None:
            tp = signal.tp_price
        elif signal.rr is not None:
            tp = fill - signal.rr * (sl - fill)
        else:
            tp = fill * 0.90

    qty = size_position(equity, signal.risk_pct, fill, sl, filters)
    if qty <= 0:
        return None

    return Position(
        side=signal.side,
        entry_price=fill,
        qty=qty,
        sl_price=sl,
        tp_price=tp,
        tag=signal.tag,
        entry_ts=entry_ts,
        entry_index=entry_index,
    )


def check_exit(position: Position, bar) -> Optional[str]:
    if position.side == Side.LONG:
        hit_sl = bar.low <= position.sl_price
        hit_tp = bar.high >= position.tp_price
    else:
        hit_sl = bar.high >= position.sl_price
        hit_tp = bar.low <= position.tp_price

    if hit_sl:
        return "SL"
    if hit_tp:
        return "TP"
    return None


def exit_pnl(position: Position, exit_price: float, fee_rate: float = TAKER_FEE) -> float:
    if position.side == Side.LONG:
        gross = (exit_price - position.entry_price) * position.qty
    else:
        gross = (position.entry_price - exit_price) * position.qty

    fees = (position.entry_price + exit_price) * position.qty * fee_rate
    return gross - fees
