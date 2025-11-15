"""VectorBT adapters for cross-platform validation."""
import time
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

# Prioritize open-source vectorbt for correctness validation
# VectorBT Pro should only be used for performance benchmarking
try:
    import vectorbt as vbt
    HAS_FREE = True
    HAS_PRO = False
    # Check if Pro is also available (for performance tests)
    try:
        import vectorbtpro
        HAS_PRO = True
    except ImportError:
        pass
except ImportError:
    vbt = None
    HAS_PRO = False
    HAS_FREE = False

from .base import BacktestResult, PlatformAdapter, Trade


class VectorBTBaseAdapter(PlatformAdapter):
    """Base adapter for VectorBT (shared logic between Pro and free)."""

    def __init__(self, name: str, is_pro: bool):
        super().__init__(name)
        self.is_pro = is_pro

    def run_backtest(
        self,
        signals: list,
        data: pd.DataFrame,  # VectorBT expects pandas
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0,
        **kwargs
    ) -> BacktestResult:
        """Run backtest using VectorBT."""
        start_time = time.time()

        # Convert signals to entries/exits
        entries, exits, sizes = self._signals_to_entries_exits(signals, data)

        # Run portfolio simulation
        if self.is_pro:
            portfolio = self._run_pro(data, entries, exits, sizes, initial_capital, commission, slippage)
        else:
            portfolio = self._run_free(data, entries, exits, sizes, initial_capital, commission, slippage)

        execution_time = time.time() - start_time

        # Convert to standardized format
        trades = self._convert_trades(portfolio, signals)
        equity_curve = self._convert_equity_curve(portfolio)
        metrics = self._extract_metrics(portfolio)

        return BacktestResult(
            platform=self.name,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            execution_time=execution_time,
            config={
                'initial_capital': initial_capital,
                'commission': commission,
                'slippage': slippage,
            }
        )

    def _signals_to_entries_exits(self, signals: list, data: pd.DataFrame):
        """Convert signals to VectorBT entries/exits format."""
        # Initialize boolean arrays
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        sizes = pd.Series(np.nan, index=data.index)

        for signal in signals:
            # Find matching timestamp in index
            try:
                idx = data.index.get_loc(signal.timestamp)
            except KeyError:
                # Signal timestamp not in data, skip
                continue

            if signal.action == 'BUY':
                entries.iloc[idx] = True
                sizes.iloc[idx] = signal.quantity if signal.quantity else 100
            elif signal.action == 'SELL':
                # VectorBT handles shorts differently
                # For now, treat SELL as exit (or short entry if supported)
                exits.iloc[idx] = True
            elif signal.action == 'CLOSE':
                exits.iloc[idx] = True

        return entries, exits, sizes

    def _run_pro(self, data, entries, exits, sizes, initial_capital, commission, slippage):
        """Run using VectorBT Pro."""
        portfolio = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=entries,
            exits=exits,
            size=sizes,
            size_type='amount',  # Fixed quantity
            init_cash=initial_capital,
            fees=commission,
            slippage=slippage,
        )
        return portfolio

    def _run_free(self, data, entries, exits, sizes, initial_capital, commission, slippage):
        """Run using VectorBT free version."""
        portfolio = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=entries,
            exits=exits,
            size=sizes,
            size_type='amount',
            init_cash=initial_capital,
            fees=commission,
            slippage=slippage,
        )
        return portfolio

    def _convert_trades(self, portfolio, signals: list) -> list[Trade]:
        """Convert VectorBT trades to standardized format."""
        trades = []

        try:
            vbt_trades = portfolio.trades.records_readable

            for _, row in vbt_trades.iterrows():
                # Find matching signal
                signal_id = ''
                entry_time = row['Entry Index']
                for sig in signals:
                    if sig.timestamp == entry_time:
                        signal_id = sig.signal_id
                        break

                # Handle exit (might be None for open positions)
                exit_time = row.get('Exit Index')
                if pd.isna(exit_time):
                    exit_time = None

                trade = Trade(
                    entry_time=entry_time,
                    entry_price=row['Avg Entry Price'],
                    exit_time=exit_time,
                    exit_price=row.get('Avg Exit Price') if not pd.isna(row.get('Avg Exit Price')) else None,
                    symbol='AAPL',  # TODO: Extract from data or Column index
                    quantity=row['Size'],
                    pnl=row.get('PnL'),
                    commission=row.get('Entry Fees', 0.0) + row.get('Exit Fees', 0.0),
                    signal_id=signal_id,
                )
                trades.append(trade)

        except Exception as e:
            # VectorBT API might differ between versions
            print(f"Warning: Could not extract trades from VectorBT: {e}")

        return trades

    def _convert_equity_curve(self, portfolio) -> pl.DataFrame:
        """Extract equity curve from VectorBT portfolio."""
        try:
            equity = portfolio.value()
            df = pd.DataFrame({
                'timestamp': equity.index,
                'equity': equity.values
            })
            return pl.from_pandas(df)
        except Exception:
            # Fallback
            return pl.DataFrame({'timestamp': [], 'equity': []})

    def _extract_metrics(self, portfolio) -> dict:
        """Extract performance metrics from VectorBT portfolio."""
        try:
            stats = portfolio.stats()
            return {
                'total_return': stats.get('Total Return [%]', 0.0),
                'sharpe_ratio': stats.get('Sharpe Ratio', 0.0),
                'max_drawdown': stats.get('Max Drawdown [%]', 0.0),
                'win_rate': stats.get('Win Rate [%]', 0.0),
                'total_trades': stats.get('Total Trades', 0),
            }
        except Exception:
            return {}

    def supports_stop_loss(self) -> bool:
        """VectorBT Pro supports stop loss."""
        return self.is_pro

    def supports_take_profit(self) -> bool:
        """VectorBT Pro supports take profit."""
        return self.is_pro

    def supports_trailing_stop(self) -> bool:
        """VectorBT Pro supports trailing stop."""
        return self.is_pro


class VectorBTProAdapter(VectorBTBaseAdapter):
    """Adapter for VectorBT Pro (commercial version)."""

    def __init__(self):
        if not HAS_PRO:
            raise ImportError("VectorBT Pro not installed. Install with: pip install vectorbt-pro")
        super().__init__("vectorbt_pro", is_pro=True)


class VectorBTFreeAdapter(VectorBTBaseAdapter):
    """Adapter for VectorBT free (open source version)."""

    def __init__(self):
        if not HAS_FREE and not HAS_PRO:
            raise ImportError("Neither VectorBT Pro nor free version installed")
        # If only Pro is installed, use it in "free" mode
        super().__init__("vectorbt_free", is_pro=HAS_PRO)
