"""Bracket order example: Entry with automatic stop-loss and take-profit.

This example demonstrates how to use bracket orders in ml4t.backtest.
A bracket order consists of:
1. Entry order (market or limit)
2. Take-profit order (limit sell)
3. Stop-loss order (stop sell)

When the entry fills, both exit orders become active. When either
exit order fills, the other is automatically cancelled.

Key concepts:
- Using broker.submit_bracket() for OCO (one-cancels-other) exits
- Automatic position management with predefined risk/reward
- Clean exit handling without manual order tracking
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.backtest import (
    DataFeed,
    Engine,
    ExecutionMode,
    Strategy,
)


def generate_trending_data(n_bars: int = 252, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic data with trending periods.

    Creates data that has clear trends to demonstrate bracket order behavior:
    - Some trends hit take-profit
    - Some trends reverse and hit stop-loss
    """
    np.random.seed(seed)

    rows = []
    base_date = datetime(2020, 1, 1)

    # Create a mix of trending and mean-reverting periods
    price = 100.0

    for i in range(n_bars):
        timestamp = base_date + timedelta(days=i)

        # Regime changes every ~50 bars
        regime = (i // 50) % 3
        if regime == 0:
            drift = 0.002  # Uptrend
        elif regime == 1:
            drift = -0.001  # Mild downtrend
        else:
            drift = 0.0  # Sideways

        # Daily return with regime drift
        daily_return = drift + np.random.normal(0, 0.015)
        price *= 1 + daily_return

        # Generate OHLC
        volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = price * (1 + np.random.normal(0, 0.005))

        rows.append(
            {
                "timestamp": timestamp,
                "asset": "SPY",
                "open": open_price,
                "high": max(high, open_price, price),
                "low": min(low, open_price, price),
                "close": price,
                "volume": np.random.randint(1000000, 5000000),
            }
        )

    return pl.DataFrame(rows).sort("timestamp")


class BracketOrderStrategy(Strategy):
    """Strategy using bracket orders for defined risk/reward.

    Entry: Buy when price makes a new 20-day high
    Exit: Bracket order with 3% take-profit, 2% stop-loss (1.5:1 reward/risk)
    """

    def __init__(
        self,
        lookback: int = 20,
        take_profit_pct: float = 0.03,
        stop_loss_pct: float = 0.02,
        position_size: int = 100,
    ):
        """Initialize strategy.

        Args:
            lookback: Days for high/low calculation
            take_profit_pct: Take-profit as decimal (0.03 = 3%)
            stop_loss_pct: Stop-loss as decimal (0.02 = 2%)
            position_size: Number of shares per trade
        """
        self.lookback = lookback
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.position_size = position_size
        self.price_history: list[float] = []
        self.in_trade = False

    def on_data(self, timestamp, data, context, broker):
        """Execute trading logic with bracket orders."""
        bar = data.get("SPY")
        if bar is None:
            return

        price = bar.get("close", 0)
        if price <= 0:
            return

        # Update price history
        self.price_history.append(price)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)

        # Need full lookback period
        if len(self.price_history) < self.lookback:
            return

        # Check if we have a position (via bracket orders)
        position = broker.get_position("SPY")

        if position is None and not self.in_trade:
            # Check for entry signal: new 20-day high
            current_high = max(self.price_history)

            if price >= current_high:
                # Calculate bracket levels
                take_profit = price * (1 + self.take_profit_pct)
                stop_loss = price * (1 - self.stop_loss_pct)

                # Submit bracket order
                result = broker.submit_bracket(
                    asset="SPY",
                    quantity=self.position_size,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                )

                if result:
                    entry, tp, sl = result
                    self.in_trade = True
                    print(
                        f"[{timestamp.date()}] BRACKET ENTRY @ ${price:.2f}"
                        f" | TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}"
                    )

        elif position is None and self.in_trade:
            # Position was closed by bracket exit
            self.in_trade = False


def main():
    """Run the bracket order strategy backtest."""
    # Generate sample data
    print("Generating sample data with trending periods...")
    df = generate_trending_data(n_bars=252)

    # Create data feed
    feed = DataFeed(prices_df=df)

    # Create strategy with 3% take-profit, 2% stop-loss
    strategy = BracketOrderStrategy(
        lookback=20,
        take_profit_pct=0.03,  # 3% take-profit
        stop_loss_pct=0.02,  # 2% stop-loss
        position_size=100,
    )

    # Run backtest
    print("\nRunning backtest...")
    initial_cash = 100_000.0
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_mode=ExecutionMode.NEXT_BAR,
    )
    result = engine.run()

    # Print results
    print("\n" + "=" * 50)
    print("BRACKET ORDER STRATEGY RESULTS")
    print("=" * 50)
    print(f"Initial Capital:  ${initial_cash:,.2f}")
    print(f"Final Value:      ${result['final_value']:,.2f}")
    print(f"Total Return:     {result['total_return']:.2%}")
    print(f"Sharpe Ratio:     {result['sharpe']:.2f}")
    print(f"Max Drawdown:     {result['max_drawdown_pct']:.2%}")

    # Trade analysis
    trades = engine.broker.trades
    print(f"\nTotal Trades:     {len(trades)}")

    if trades:
        # Analyze exit reasons
        tp_exits = [t for t in trades if "take_profit" in t.exit_reason.lower()]
        sl_exits = [t for t in trades if "stop_loss" in t.exit_reason.lower()]
        other_exits = [t for t in trades if t not in tp_exits and t not in sl_exits]

        print(f"Take-Profit Exits: {len(tp_exits)}")
        print(f"Stop-Loss Exits:   {len(sl_exits)}")
        print(f"Other Exits:       {len(other_exits)}")

        # Win rate and average P&L
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        print(f"\nWin Rate:          {win_rate:.1%}")
        print(f"Avg Win:           ${avg_win:.2f}")
        print(f"Avg Loss:          ${avg_loss:.2f}")

        if avg_loss != 0:
            profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses))
            print(f"Profit Factor:     {profit_factor:.2f}")


if __name__ == "__main__":
    main()
