"""Backtesting engine orchestration."""

from datetime import datetime
import polars as pl

from .types import ExecutionMode
from .models import CommissionModel, SlippageModel
from .datafeed import DataFeed
from .broker import Broker
from .strategy import Strategy


class Engine:
    """Backtesting engine."""

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        initial_cash: float = 100000.0,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
        account_type: str = "cash",
        initial_margin: float = 0.5,
        maintenance_margin: float = 0.25,
    ):
        self.feed = feed
        self.strategy = strategy
        self.execution_mode = execution_mode
        self.broker = Broker(
            initial_cash=initial_cash,
            commission_model=commission_model,
            slippage_model=slippage_model,
            execution_mode=execution_mode,
            account_type=account_type,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
        )
        self.equity_curve: list[tuple[datetime, float]] = []

    def run(self) -> dict:
        """Run backtest and return results."""
        self.strategy.on_start(self.broker)

        for timestamp, assets_data, context in self.feed:
            prices = {a: d["close"] for a, d in assets_data.items() if d.get("close")}
            opens = {a: d.get("open", d.get("close")) for a, d in assets_data.items()}
            volumes = {a: d.get("volume", 0) for a, d in assets_data.items()}
            signals = {a: d.get("signals", {}) for a, d in assets_data.items()}

            self.broker._update_time(timestamp, prices, opens, volumes, signals)

            if self.execution_mode == ExecutionMode.NEXT_BAR:
                # Next-bar mode: process pending orders at open price
                self.broker._process_orders(use_open=True)
                # Strategy generates new orders
                self.strategy.on_data(timestamp, assets_data, context, self.broker)
                # New orders will be processed next bar
            else:
                # Same-bar mode: process before and after strategy
                self.broker._process_orders()
                self.strategy.on_data(timestamp, assets_data, context, self.broker)
                self.broker._process_orders()

            self.equity_curve.append((timestamp, self.broker.get_account_value()))

        self.strategy.on_end(self.broker)
        return self._generate_results()

    def _generate_results(self) -> dict:
        """Generate backtest results."""
        if not self.equity_curve:
            return {}

        initial = self.broker.initial_cash
        final = self.equity_curve[-1][1]
        total_return = (final - initial) / initial

        peak = initial
        max_dd = 0.0
        for _, value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        winning = [t for t in self.broker.trades if t.pnl > 0]
        losing = [t for t in self.broker.trades if t.pnl <= 0]

        return {
            "initial_cash": initial,
            "final_value": final,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "num_trades": len(self.broker.trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.broker.trades) if self.broker.trades else 0,
            "total_commission": sum(f.commission for f in self.broker.fills),
            "total_slippage": sum(f.slippage for f in self.broker.fills),
            "trades": self.broker.trades,
            "equity_curve": self.equity_curve,
            "fills": self.broker.fills,
        }


# === Convenience Function ===

def run_backtest(
    prices: pl.DataFrame | str,
    strategy: Strategy,
    signals: pl.DataFrame | str | None = None,
    context: pl.DataFrame | str | None = None,
    initial_cash: float = 100000.0,
    commission_model: CommissionModel | None = None,
    slippage_model: SlippageModel | None = None,
    execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
) -> dict:
    """Run a backtest with minimal setup."""
    feed = DataFeed(
        prices_path=prices if isinstance(prices, str) else None,
        signals_path=signals if isinstance(signals, str) else None,
        context_path=context if isinstance(context, str) else None,
        prices_df=prices if isinstance(prices, pl.DataFrame) else None,
        signals_df=signals if isinstance(signals, pl.DataFrame) else None,
        context_df=context if isinstance(context, pl.DataFrame) else None,
    )
    engine = Engine(
        feed, strategy, initial_cash,
        commission_model=commission_model,
        slippage_model=slippage_model,
        execution_mode=execution_mode,
    )
    return engine.run()


# Backward compatibility: BacktestEngine was renamed to Engine in v0.2.0
BacktestEngine = Engine
