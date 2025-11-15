"""
Debug Framework Discrepancy

Investigate why ml4t.backtest and Backtrader produce different results
with identical data and strategy logic.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project paths
ml4t.backtest_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(ml4t.backtest_src))


def load_test_data():
    """Load the exact same test data."""
    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"
    df = pd.read_parquet(wiki_path)

    aapl = df[df["ticker"] == "AAPL"].copy()
    aapl["date"] = pd.to_datetime(aapl["date"])
    aapl = aapl.set_index("date").sort_index()
    test_data = aapl.loc["2015-01-01":"2016-12-31"].copy()

    return test_data


def debug_backtest_signals(data):
    """Debug ml4t.backtest signal generation step by step."""
    print("=" * 60)
    print("DEBUGGING ML4T.BACKTEST SIGNALS")
    print("=" * 60)

    data_copy = data.copy()
    data_copy["ma_20"] = data_copy["close"].rolling(window=20, min_periods=20).mean()
    data_copy["ma_50"] = data_copy["close"].rolling(window=50, min_periods=50).mean()
    data_copy = data_copy.dropna()

    print(f"Data after MA calculation: {len(data_copy)} rows")
    print(f"First date with signals: {data_copy.index[0]}")

    cash = 10000.0
    shares = 0.0
    position = 0
    trades = []

    prev_ma_20 = None
    prev_ma_50 = None

    signals_found = []

    for i, (date, row) in enumerate(data_copy.iterrows()):
        current_ma_20 = row["ma_20"]
        current_ma_50 = row["ma_50"]
        current_price = row["close"]

        signal = 0
        signal_type = ""

        if prev_ma_20 is not None and prev_ma_50 is not None:
            # Check for crossover
            if prev_ma_20 <= prev_ma_50 and current_ma_20 > current_ma_50:
                signal = 1
                signal_type = "GOLDEN_CROSS"
            elif prev_ma_20 > prev_ma_50 and current_ma_20 <= current_ma_50:
                signal = -1
                signal_type = "DEATH_CROSS"

        if signal != 0:
            signals_found.append(
                {
                    "date": date,
                    "signal": signal,
                    "type": signal_type,
                    "ma_20": current_ma_20,
                    "ma_50": current_ma_50,
                    "prev_ma_20": prev_ma_20,
                    "prev_ma_50": prev_ma_50,
                    "price": current_price,
                    "position_before": position,
                },
            )

        # Execute trades
        if signal == 1 and position == 0 and cash > 0:
            shares = cash / current_price
            cash = 0.0
            position = 1
            trades.append(
                {
                    "date": date,
                    "action": "BUY",
                    "price": current_price,
                    "shares": shares,
                },
            )

        elif signal == -1 and position == 1 and shares > 0:
            cash = shares * current_price
            shares = 0.0
            position = 0
            trades.append(
                {
                    "date": date,
                    "action": "SELL",
                    "price": current_price,
                    "cash_received": cash,
                },
            )

        prev_ma_20 = current_ma_20
        prev_ma_50 = current_ma_50

    final_value = cash + shares * data_copy["close"].iloc[-1]

    print("\nml4t.backtest Results:")
    print(f"  Signals detected: {len(signals_found)}")
    print(f"  Trades executed: {len(trades)}")
    print(f"  Final value: ${final_value:,.2f}")
    print(f"  Return: {(final_value / 10000 - 1) * 100:.2f}%")

    print("\nFirst 5 signals:")
    for i, sig in enumerate(signals_found[:5]):
        print(
            f"  {i + 1}. {sig['date'].strftime('%Y-%m-%d')}: {sig['type']} - MA20={sig['ma_20']:.2f}, MA50={sig['ma_50']:.2f}",
        )

    print("\nFirst 5 trades:")
    for i, trade in enumerate(trades[:5]):
        print(
            f"  {i + 1}. {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} @ ${trade['price']:.2f}",
        )

    return signals_found, trades, final_value


def debug_backtrader_signals(data):
    """Debug Backtrader signal generation step by step."""
    print("\n" + "=" * 60)
    print("DEBUGGING BACKTRADER SIGNALS")
    print("=" * 60)

    try:
        import backtrader as bt

        class DebuggingStrategy(bt.Strategy):
            def __init__(self):
                self.ma_20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
                self.ma_50 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
                self.crossover = bt.indicators.CrossOver(self.ma_20, self.ma_50)
                self.signals_found = []
                self.trades_made = []
                self.day_count = 0

            def next(self):
                self.day_count += 1

                # Record crossover signals
                if self.crossover[0] != 0:
                    signal_type = "GOLDEN_CROSS" if self.crossover[0] > 0 else "DEATH_CROSS"
                    self.signals_found.append(
                        {
                            "date": self.data.datetime.date(0),
                            "signal": self.crossover[0],
                            "type": signal_type,
                            "ma_20": self.ma_20[0],
                            "ma_50": self.ma_50[0],
                            "price": self.data.close[0],
                            "position_before": 1 if self.position else 0,
                        },
                    )

                if not self.position:
                    if self.crossover > 0:
                        size = self.broker.getcash() / self.data.close[0]
                        self.buy(size=size)
                        self.trades_made.append(
                            {
                                "date": self.data.datetime.date(0),
                                "action": "BUY",
                                "price": self.data.close[0],
                                "shares": size,
                            },
                        )
                else:
                    if self.crossover < 0:
                        self.close()
                        self.trades_made.append(
                            {
                                "date": self.data.datetime.date(0),
                                "action": "SELL",
                                "price": self.data.close[0],
                                "cash_received": self.position.size * self.data.close[0],
                            },
                        )

        # Setup Cerebro with detailed debugging
        cerebro = bt.Cerebro()
        cerebro.addstrategy(DebuggingStrategy)

        # Prepare identical data
        bt_data = data.reset_index()
        bt_data = bt_data.rename(columns={"date": "datetime"})

        data_feed = bt.feeds.PandasData(
            dataname=bt_data,
            datetime="datetime",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
        cerebro.adddata(data_feed)

        cerebro.broker.setcash(10000.0)
        cerebro.broker.setcommission(commission=0.0)

        # Run
        strategies = cerebro.run()
        strategy = strategies[0]

        final_value = cerebro.broker.getvalue()

        print("Backtrader Results:")
        print(f"  Days processed: {strategy.day_count}")
        print(f"  Signals detected: {len(strategy.signals_found)}")
        print(f"  Trades executed: {len(strategy.trades_made)}")
        print(f"  Final value: ${final_value:,.2f}")
        print(f"  Return: {(final_value / 10000 - 1) * 100:.2f}%")

        print("\nFirst 5 signals:")
        for i, sig in enumerate(strategy.signals_found[:5]):
            print(
                f"  {i + 1}. {sig['date']}: {sig['type']} - MA20={sig['ma_20']:.2f}, MA50={sig['ma_50']:.2f}",
            )

        print("\nFirst 5 trades:")
        for i, trade in enumerate(strategy.trades_made[:5]):
            print(f"  {i + 1}. {trade['date']}: {trade['action']} @ ${trade['price']:.2f}")

        return strategy.signals_found, strategy.trades_made, final_value

    except Exception as e:
        print(f"Backtrader debug failed: {e}")
        return [], [], 0


def compare_signals(qe_signals, bt_signals):
    """Compare signals between frameworks."""
    print("\n" + "=" * 60)
    print("SIGNAL COMPARISON")
    print("=" * 60)

    print(f"ml4t.backtest signals: {len(qe_signals)}")
    print(f"Backtrader signals: {len(bt_signals)}")

    # Compare first few signals
    print("\nSide-by-side comparison (first 10):")
    print(
        f"{'Date':<12} {'QE Type':<12} {'BT Type':<12} {'QE MA20':<10} {'BT MA20':<10} {'Match':<6}",
    )
    print("-" * 70)

    matches = 0
    max_compare = min(len(qe_signals), len(bt_signals), 10)

    for i in range(max_compare):
        qe_sig = qe_signals[i]
        bt_sig = bt_signals[i]

        qe_date = (
            qe_sig["date"].strftime("%Y-%m-%d")
            if hasattr(qe_sig["date"], "strftime")
            else str(qe_sig["date"])
        )
        bt_date = str(bt_sig["date"])

        match = "✓" if (qe_date == bt_date and qe_sig["type"] == bt_sig["type"]) else "✗"
        if match == "✓":
            matches += 1

        print(
            f"{qe_date:<12} {qe_sig['type']:<12} {bt_sig['type']:<12} {qe_sig['ma_20']:<10.2f} {bt_sig['ma_20']:<10.2f} {match:<6}",
        )

    if max_compare > 0:
        match_rate = matches / max_compare * 100
        print(f"\nSignal match rate: {matches}/{max_compare} ({match_rate:.1f}%)")

    return matches


def main():
    """Debug the discrepancy between frameworks."""
    print("DEBUGGING FRAMEWORK DISCREPANCY")
    print("Using identical AAPL data 2015-2016")
    print("Moving Average Crossover Strategy (20/50)")

    # Load identical data
    data = load_test_data()
    print(f"\nLoaded {len(data)} rows of AAPL data")

    # Debug both frameworks
    qe_signals, qe_trades, qe_final = debug_ml4t.backtest_signals(data)
    bt_signals, bt_trades, bt_final = debug_backtrader_signals(data)

    # Compare results
    compare_signals(qe_signals, bt_signals)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"ml4t.backtest final value: ${qe_final:,.2f}")
    print(f"Backtrader final value: ${bt_final:,.2f}")
    print(f"Difference: ${abs(qe_final - bt_final):.2f}")
    print(f"ml4t.backtest trades: {len(qe_trades)}")
    print(f"Backtrader trades: {len(bt_trades)}")


if __name__ == "__main__":
    main()
