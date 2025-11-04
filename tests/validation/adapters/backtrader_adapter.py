"""Backtrader adapter for cross-platform validation."""
import time
from datetime import datetime

import backtrader as bt
import pandas as pd
import polars as pl

from .base import BacktestResult, PlatformAdapter, Trade


class SignalStrategy(bt.Strategy):
    """Backtrader strategy that executes based on pre-generated signals."""

    def __init__(self, signals: list):
        self.signals = {(s.timestamp, s.symbol): s for s in signals}
        self.executed_signals = set()
        self.trades_log = []

    def next(self):
        """Process each bar."""
        current_time = self.data.datetime.datetime()

        # Check for matching signal
        for data_feed in self.datas:
            symbol = data_feed._name
            key = (current_time, symbol)

            if key in self.signals and key not in self.executed_signals:
                signal = self.signals[key]
                self._execute_signal(signal, data_feed)
                self.executed_signals.add(key)

    def _execute_signal(self, signal, data_feed):
        """Execute signal via backtrader."""
        if signal.action == 'BUY':
            size = signal.quantity or 100
            if signal.stop_loss or signal.take_profit:
                # Create bracket order
                self.buy_bracket(
                    data=data_feed,
                    size=size,
                    stopprice=signal.stop_loss,
                    limitprice=signal.take_profit,
                )
            else:
                self.buy(data=data_feed, size=size)

        elif signal.action == 'SELL':
            size = signal.quantity or 100
            if signal.stop_loss or signal.take_profit:
                self.sell_bracket(
                    data=data_feed,
                    size=size,
                    stopprice=signal.stop_loss,
                    limitprice=signal.take_profit,
                )
            else:
                self.sell(data=data_feed, size=size)

        elif signal.action == 'CLOSE':
            self.close(data=data_feed)

    def notify_trade(self, trade):
        """Track completed trades."""
        if trade.isclosed:
            self.trades_log.append({
                'entry_time': bt.num2date(trade.dtopen),
                'entry_price': trade.price,
                'exit_time': bt.num2date(trade.dtclose),
                'exit_price': trade.priceclosed,
                'quantity': trade.size,
                'pnl': trade.pnl,
                'commission': trade.commission,
            })


class BacktraderAdapter(PlatformAdapter):
    """Adapter for Backtrader backtesting platform."""

    def __init__(self):
        super().__init__("backtrader")

    def run_backtest(
        self,
        signals: list,
        data: pd.DataFrame,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0,
        **kwargs
    ) -> BacktestResult:
        """Run backtest using Backtrader."""
        start_time = time.time()

        # Create cerebro engine
        cerebro = bt.Cerebro()

        # Add data feed
        bt_data = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(bt_data)

        # Add strategy with signals
        cerebro.addstrategy(SignalStrategy, signals=signals)

        # Set initial capital
        cerebro.broker.setcash(initial_capital)

        # Set commission
        cerebro.broker.setcommission(commission=commission)

        # Run backtest
        strategies = cerebro.run()
        strategy = strategies[0]

        execution_time = time.time() - start_time

        # Convert results
        trades = self._convert_trades(strategy.trades_log, signals)
        equity_curve = self._get_equity_curve(cerebro)
        metrics = self._calculate_metrics(cerebro, initial_capital)

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

    def _convert_trades(self, trades_log: list, signals: list) -> list[Trade]:
        """Convert backtrader trades to standardized format."""
        trades = []

        for trade_data in trades_log:
            # Find matching signal
            signal_id = ''
            for sig in signals:
                if sig.timestamp == trade_data['entry_time']:
                    signal_id = sig.signal_id
                    break

            trade = Trade(
                entry_time=trade_data['entry_time'],
                entry_price=trade_data['entry_price'],
                exit_time=trade_data['exit_time'],
                exit_price=trade_data['exit_price'],
                symbol='UNKNOWN',  # Backtrader doesn't store this easily
                quantity=trade_data['quantity'],
                pnl=trade_data['pnl'],
                commission=trade_data['commission'],
                signal_id=signal_id,
            )
            trades.append(trade)

        return trades

    def _get_equity_curve(self, cerebro) -> pl.DataFrame:
        """Extract equity curve from backtrader."""
        # Backtrader doesn't provide this easily
        # Would need to add analyzer
        final_value = cerebro.broker.getvalue()
        return pl.DataFrame({
            'timestamp': [datetime.now()],
            'equity': [final_value]
        })

    def _calculate_metrics(self, cerebro, initial_capital: float) -> dict:
        """Calculate performance metrics."""
        final_value = cerebro.broker.getvalue()
        total_return = (final_value / initial_capital - 1) * 100

        return {
            'total_return': total_return,
            'final_value': final_value,
        }

    def supports_stop_loss(self) -> bool:
        """Backtrader supports stop loss via bracket orders."""
        return True

    def supports_take_profit(self) -> bool:
        """Backtrader supports take profit via bracket orders."""
        return True

    def supports_trailing_stop(self) -> bool:
        """Backtrader supports trailing stop via StopTrail order."""
        return True
