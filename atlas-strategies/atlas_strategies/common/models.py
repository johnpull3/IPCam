from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


@dataclass
class Signal:
    side: Side
    risk_pct: float
    sl_pct: Optional[float] = None
    rr: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    tag: str = ""
    use_tp: bool = True  # False: sin TP fijo (salida por trailing stop o señal)


@dataclass
class Position:
    side: Side
    entry_price: float
    qty: float
    sl_price: float
    tp_price: Optional[float]
    tag: str = ""
    entry_ts: int = 0
    entry_index: int = 0


@dataclass
class SymbolFilters:
    step_size: float = 0.001
    min_qty: float = 0.001
    min_notional: float = 5.0
    tick_size: float = 0.1


@dataclass
class Trade:
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str
    tag: str
    entry_ts: int
    exit_ts: int
    bars_held: int


@dataclass
class PositionUpdate:
    """Respuesta de una estrategia que gestiona su posición abierta.

    new_stop: nuevo stop propuesto (el engine solo lo acepta si no afloja el actual).
    exit: True para cerrar al open de la vela siguiente (motivo "SIGNAL").
    """
    new_stop: Optional[float] = None
    exit: bool = False
