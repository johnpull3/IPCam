from atlas_strategies.common.models import SymbolFilters, Trade, Side
from atlas_strategies.common.executor import (
    EXECUTION_MODES, MAKER_FEE, TAKER_FEE, build_position, check_exit, exit_pnl, maker_fill_price
)
from atlas_strategies.backtest.data_generator import generate_synthetic_bars


def run_backtest(strategy, bars, filters=None, initial_equity=10000.0,
                 funding_rate_per_8h=0.0001, execution_mode="taker"):
    """execution_mode: "taker" (MARKET al open siguiente, TAKER_FEE + SLIPPAGE)
    o "maker" (LIMIT al close de la señal, llena si la vela siguiente lo toca, MAKER_FEE sin slippage)."""
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"execution_mode inválido: {execution_mode!r}")
    filters = filters or SymbolFilters()
    fee_rate = MAKER_FEE if execution_mode == "maker" else TAKER_FEE
    signals = 0
    missed_signals = 0
    equity = initial_equity
    position = None
    trades = []
    equity_curve = []
    bars_held = 0
    next_funding_ts = bars[0].ts + 8 * 3600 * 1000

    for i in range(len(bars) - 1):
        bar = bars[i]
        next_bar = bars[i + 1]

        if position is None:
            signal = strategy.on_bar(bar)
            if signal is not None:
                signals += 1
                if execution_mode == "maker":
                    limit = maker_fill_price(signal, bar.close, next_bar)
                    if limit is None:
                        missed_signals += 1
                    else:
                        position = build_position(
                            signal, limit, equity, filters,
                            entry_ts=next_bar.ts, entry_index=i + 1, slippage=0.0
                        )
                else:
                    position = build_position(
                        signal, next_bar.open, equity, filters,
                        entry_ts=next_bar.ts, entry_index=i + 1
                    )
                if position:
                    bars_held = 0

        if position is not None:
            bars_held += 1

            # Funding cada 8h
            if next_bar.ts >= next_funding_ts:
                if position.side == Side.LONG:
                    equity -= position.qty * next_bar.close * funding_rate_per_8h
                else:
                    equity += position.qty * next_bar.close * funding_rate_per_8h
                next_funding_ts += 8 * 3600 * 1000

            reason = check_exit(position, next_bar)
            if reason is not None:
                exit_price = position.sl_price if reason == "SL" else position.tp_price
                pnl = exit_pnl(position, exit_price, fee_rate)
                equity += pnl
                trades.append(Trade(
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    qty=position.qty,
                    pnl=pnl,
                    reason=reason,
                    tag=position.tag,
                    entry_ts=position.entry_ts,
                    exit_ts=next_bar.ts,
                    bars_held=bars_held,
                ))
                position = None

        equity_curve.append(equity)

    return {
        "final_equity": equity,
        "initial_equity": initial_equity,
        "return_pct": (equity / initial_equity - 1) * 100,
        "n_trades": len(trades),
        "execution_mode": execution_mode,
        "signals": signals,
        "missed_signals": missed_signals,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def summarize(result):
    trades = result["trades"]
    if not trades:
        print("  Sin trades.")
        return
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100
    total_win = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses))
    pf = total_win / total_loss if total_loss > 0 else float("inf")
    print(f"  Equity final:  {result['final_equity']:.2f}")
    print(f"  Retorno:       {result['return_pct']:.2f}%")
    print(f"  Trades:        {len(trades)}")
    print(f"  Win rate:      {win_rate:.1f}%")
    print(f"  Profit factor: {pf:.2f}")


if __name__ == "__main__":
    from atlas_strategies.strategies.vwap_ema_rsi_scalp import VwapEmaRsiScalp
    from atlas_strategies.strategies.bollinger_rsi2_reversion import BollingerRsi2Reversion
    from atlas_strategies.strategies.swing_reclaim import SwingReclaim

    print("Generando velas sintéticas...")
    bars = generate_synthetic_bars(n=20000)

    for name, strat in [
        ("VWAP+EMA+RSI Scalp", VwapEmaRsiScalp()),
        ("Bollinger+RSI2",     BollingerRsi2Reversion()),
        ("Swing Reclaim",      SwingReclaim()),
    ]:
        print(f"\n=== {name} ===")
        result = run_backtest(strat, bars)
        summarize(result)
