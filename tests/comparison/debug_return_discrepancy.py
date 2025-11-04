"""
Debug return discrepancy between QEngine and Backtrader

Investigate why identical trades produce different returns.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project paths
qengine_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(qengine_src))


def load_spy_data():
    """Load SPY data."""
    spy_path = projects_dir / "spy_order_flow" / "spy_features.parquet"
    df = pd.read_parquet(spy_path)
    df = df[["timestamp", "open", "high", "low", "last", "volume"]].copy()
    df = df.rename(columns={"last": "close"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df.head(390)


def debug_qengine_calculation():
    """Debug QEngine calculation step by step."""
    print("\n" + "=" * 60)
    print("DEBUGGING QENGINE CALCULATION")
    print("=" * 60)

    data = load_spy_data()

    # Calculate MAs
    data["ma_short"] = data["close"].rolling(window=20).mean()
    data["ma_long"] = data["close"].rolling(window=50).mean()

    initial_capital = 10000.0
    cash = initial_capital
    shares = 0
    position = 0

    trades = []
    portfolio_values = []

    for i, (timestamp, row) in enumerate(data.iterrows()):
        # Skip if MAs not ready
        if pd.isna(row["ma_short"]) or pd.isna(row["ma_long"]):
            portfolio_values.append(cash + shares * row["close"])
            continue

        prev_ma_short = data["ma_short"].iloc[i - 1] if i > 0 else row["ma_short"]
        prev_ma_long = data["ma_long"].iloc[i - 1] if i > 0 else row["ma_long"]

        # Buy signal: short MA crosses above long MA
        if row["ma_short"] > row["ma_long"] and prev_ma_short <= prev_ma_long and position <= 0:
            if cash > 0:
                shares_to_buy = cash / row["close"]
                trades.append(
                    {
                        "timestamp": timestamp,
                        "action": "BUY",
                        "price": row["close"],
                        "shares": shares_to_buy,
                        "cash_before": cash,
                        "shares_before": shares,
                    },
                )
                shares += shares_to_buy
                cash = 0  # All cash invested
                position = 1
                print(f"BUY: {timestamp} @ ${row['close']:.2f} - Shares: {shares_to_buy:.4f}")

        # Sell signal: short MA crosses below long MA
        elif row["ma_short"] < row["ma_long"] and prev_ma_short >= prev_ma_long and position >= 0:
            if shares > 0:
                cash_received = shares * row["close"]
                trades.append(
                    {
                        "timestamp": timestamp,
                        "action": "SELL",
                        "price": row["close"],
                        "shares": shares,
                        "cash_before": cash,
                        "shares_before": shares,
                    },
                )
                cash += cash_received
                shares = 0  # All shares sold
                position = 0
                print(
                    f"SELL: {timestamp} @ ${row['close']:.2f} - Cash received: ${cash_received:.2f}",
                )

        # Track portfolio value
        portfolio_value = cash + shares * row["close"]
        portfolio_values.append(portfolio_value)

    # Final calculations
    final_value = cash + shares * data["close"].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    print("\nQEngine Summary:")
    print(f"  Initial capital: ${initial_capital:,.2f}")
    print(f"  Final cash: ${cash:.2f}")
    print(f"  Final shares: {shares:.4f}")
    print(f"  Final share value: ${shares * data['close'].iloc[-1]:.2f}")
    print(f"  Final total value: ${final_value:.2f}")
    print(f"  Total return: {total_return:.6f}%")
    print(f"  Number of trades: {len(trades)}")

    return trades, final_value, total_return


def debug_backtrader_calculation():
    """Debug Backtrader calculation."""
    print("\n" + "=" * 60)
    print("DEBUGGING BACKTRADER CALCULATION")
    print("=" * 60)

    try:
        import backtrader as bt

        data = load_spy_data()
        initial_capital = 10000.0

        class DetailedMomentumStrategy(bt.Strategy):
            params = (("short_window", 20), ("long_window", 50))

            def __init__(self):
                self.ma_short = bt.indicators.SMA(self.data.close, period=self.params.short_window)
                self.ma_long = bt.indicators.SMA(self.data.close, period=self.params.long_window)
                self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)
                self.trade_details = []

            def next(self):
                current_value = self.broker.getvalue()
                current_cash = self.broker.getcash()

                if not self.position:
                    if self.crossover > 0:  # Golden cross
                        size = current_cash / self.data.close[0]
                        self.buy(size=size)
                        self.trade_details.append(
                            {
                                "timestamp": self.data.datetime.datetime(0),
                                "action": "BUY",
                                "price": self.data.close[0],
                                "size": size,
                                "cash_before": current_cash,
                                "value_before": current_value,
                            },
                        )
                        print(
                            f"BUY: {self.data.datetime.datetime(0)} @ ${self.data.close[0]:.2f} - Size: {size:.4f}",
                        )
                else:
                    if self.crossover < 0:  # Death cross
                        size = self.position.size
                        self.sell(size=size)
                        cash_received = size * self.data.close[0]
                        self.trade_details.append(
                            {
                                "timestamp": self.data.datetime.datetime(0),
                                "action": "SELL",
                                "price": self.data.close[0],
                                "size": size,
                                "cash_before": current_cash,
                                "value_before": current_value,
                            },
                        )
                        print(
                            f"SELL: {self.data.datetime.datetime(0)} @ ${self.data.close[0]:.2f} - Cash received: ${cash_received:.2f}",
                        )

        # Create cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(DetailedMomentumStrategy)

        # Prepare and add data
        data_bt_format = data.copy().reset_index()
        data_bt = bt.feeds.PandasData(
            dataname=data_bt_format,
            datetime="timestamp",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
        )
        cerebro.adddata(data_bt)

        # Set broker parameters
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=0.0)

        # Run
        strategies = cerebro.run()
        strategy = strategies[0]

        # Results
        final_value = cerebro.broker.getvalue()
        final_cash = cerebro.broker.getcash()
        total_return = (final_value / initial_capital - 1) * 100

        print("\nBacktrader Summary:")
        print(f"  Initial capital: ${initial_capital:,.2f}")
        print(f"  Final cash: ${final_cash:.2f}")
        print(f"  Final portfolio value: ${final_value:.2f}")
        print(f"  Total return: {total_return:.6f}%")
        print(f"  Number of trades: {len(strategy.trade_details)}")

        return strategy.trade_details, final_value, total_return

    except Exception as e:
        print(f"Backtrader error: {e}")
        return [], 0, 0


def compare_trade_by_trade():
    """Compare trades side by side."""
    print("\n" + "=" * 80)
    print("TRADE-BY-TRADE COMPARISON")
    print("=" * 80)

    qengine_trades, qengine_final, qengine_return = debug_qengine_calculation()
    bt_trades, bt_final, bt_return = debug_backtrader_calculation()

    print(f"\n{'Trade':<6} {'QEngine Price':<15} {'Backtrader Price':<18} {'Difference':<12}")
    print("-" * 60)

    for i in range(min(len(qengine_trades), len(bt_trades))):
        qe_trade = qengine_trades[i]
        bt_trade = bt_trades[i]

        price_diff = abs(qe_trade["price"] - bt_trade["price"])
        print(
            f"{i + 1:<6} ${qe_trade['price']:<14.2f} ${bt_trade['price']:<17.2f} ${price_diff:<11.2f}",
        )

    print("\nFINAL COMPARISON:")
    print(f"  QEngine final value:  ${qengine_final:.6f}")
    print(f"  Backtrader final value: ${bt_final:.6f}")
    print(f"  Difference: ${abs(qengine_final - bt_final):.6f}")
    print(f"  QEngine return:  {qengine_return:.6f}%")
    print(f"  Backtrader return: {bt_return:.6f}%")
    print(f"  Return difference: {abs(qengine_return - bt_return):.6f}%")

    # Investigate potential causes
    print("\nPOTENTIAL CAUSES:")
    if abs(qengine_final - bt_final) > 1:
        print("  • Significant value difference suggests different position sizing")
    if len(qengine_trades) != len(bt_trades):
        print("  • Different number of trades")
    else:
        print("  • Same number of trades - likely rounding or calculation differences")


if __name__ == "__main__":
    compare_trade_by_trade()
