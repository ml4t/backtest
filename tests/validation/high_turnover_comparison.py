"""High-Turnover Cross-Framework Validation.

Test specification:
- 500 symbols
- Always 25 positions (fixed portfolio size)
- Random signals (reproducible seed)
- High turnover (rebalance every N days)

Measures:
- Trade count (should match across frameworks)
- Returns (should match within small tolerance)
- Execution time (framework speed comparison)
- Memory usage (framework memory efficiency)

Run:
    python tests/validation/high_turnover_comparison.py
"""

import gc
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed, memory measurements disabled")

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


# ==============================================================================
# DATA GENERATION
# ==============================================================================


def generate_synthetic_data(
    n_symbols: int = 500,
    n_days: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Args:
        n_symbols: Number of symbols
        n_days: Number of trading days
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: timestamp, asset_id, open, high, low, close, volume
    """
    np.random.seed(seed)

    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start_date, periods=n_days, freq="B")

    # Generate data for each symbol
    data = []
    for i in range(n_symbols):
        asset_id = f"SYM{i:04d}"

        # Start price around $100 with random walk
        prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
        prices = np.maximum(prices, 10)  # Floor at $10

        for j, date in enumerate(dates):
            close = prices[j]
            open_price = close * (1 + np.random.randn() * 0.01)
            high = max(open_price, close) * (1 + abs(np.random.randn()) * 0.02)
            low = min(open_price, close) * (1 - abs(np.random.randn()) * 0.02)
            volume = int(np.random.uniform(10000, 1000000))

            data.append(
                {
                    "timestamp": date,
                    "asset_id": asset_id,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": volume,
                }
            )

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} bars for {n_symbols} symbols over {n_days} days")
    return df


def generate_random_signals(
    df: pd.DataFrame,
    n_positions: int = 25,
    rebalance_days: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate random signals for high-turnover testing.

    Strategy: Every rebalance_days, randomly select n_positions symbols to hold.

    Args:
        df: Price data
        n_positions: Number of positions to hold
        rebalance_days: Rebalance frequency
        seed: Random seed

    Returns:
        DataFrame with columns: timestamp, asset_id, signal (1=long, 0=flat)
    """
    np.random.seed(seed)

    dates = sorted(df["timestamp"].unique())
    symbols = sorted(df["asset_id"].unique())

    signals = []
    current_positions = set()

    for i, date in enumerate(dates):
        # Rebalance every N days
        if i % rebalance_days == 0:
            # Randomly select n_positions symbols
            current_positions = set(np.random.choice(symbols, n_positions, replace=False))

        # Generate signals for all symbols
        for symbol in symbols:
            signal = 1 if symbol in current_positions else 0
            signals.append({"timestamp": date, "asset_id": symbol, "signal": signal})

    signals_df = pd.DataFrame(signals)
    print(
        f"Generated signals: {rebalance_days}-day rebalance, "
        f"{n_positions} positions, {len(signals_df)} signal points"
    )
    return signals_df


# ==============================================================================
# FRAMEWORK ADAPTERS
# ==============================================================================


@dataclass
class BacktestResult:
    """Standardized result format."""

    framework: str
    initial_capital: float
    final_value: float
    total_return: float
    num_trades: int
    execution_time: float
    memory_mb: Optional[float]
    trades_df: pd.DataFrame


def run_ml4t_backtest(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    initial_capital: float = 100_000,
) -> BacktestResult:
    """Run backtest on ml4t.backtest."""
    from ml4t.backtest.core.event import MarketEvent
    from ml4t.backtest.core.types import AssetId
    from ml4t.backtest.data.polars_feed import PolarsDataFeed
    from ml4t.backtest.engine import BacktestEngine
    from ml4t.backtest.execution.broker import SimulationBroker
    from ml4t.backtest.execution.commission import PercentageCommission
    from ml4t.backtest.portfolio.portfolio import Portfolio
    from ml4t.backtest.strategy.base import Strategy

    # Convert to Polars
    price_pl = pl.from_pandas(price_df)
    signals_pl = pl.from_pandas(signals_df)

    # Join signals to price data
    data_pl = price_pl.join(
        signals_pl,
        on=["timestamp", "asset_id"],
        how="left",
    ).fill_null(0)

    # Create feed
    feed = PolarsDataFeed(data_pl)

    # Strategy
    class SignalFollower(Strategy):
        def __init__(self, broker: SimulationBroker, portfolio: Portfolio):
            super().__init__(broker, portfolio)
            self.target_positions = 25
            self.position_size = 1.0 / self.target_positions

        def on_market_event(self, event: MarketEvent):
            asset_id = event.asset_id
            signal = event.data.get("signal", 0)

            current_position = self.broker.get_position(asset_id)
            has_position = current_position.quantity > 0 if current_position else False

            if signal == 1 and not has_position:
                # Enter position
                allocation = self.position_size
                self.buy_percent(asset_id, allocation, event)
            elif signal == 0 and has_position:
                # Exit position
                self.close_position(asset_id, event)

    # Portfolio and broker
    portfolio = Portfolio(initial_capital=initial_capital)
    broker = SimulationBroker(
        portfolio=portfolio,
        commission_model=PercentageCommission(rate=0.001),
    )

    # Strategy
    strategy = SignalFollower(broker, portfolio)

    # Engine
    engine = BacktestEngine(
        data_feed=feed,
        strategy=strategy,
        broker=broker,
        portfolio=portfolio,
    )

    # Measure
    gc.collect()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None

    start = time.perf_counter()
    engine.run()
    end = time.perf_counter()

    mem_after = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None
    memory_mb = mem_after - mem_before if mem_before and mem_after else None

    # Extract trades
    trades_list = []
    for trade in broker.trade_tracker.trades:
        trades_list.append(
            {
                "timestamp": trade.entry_timestamp,
                "action": "ENTRY",
                "asset_id": trade.asset_id,
                "quantity": trade.quantity,
                "price": trade.entry_price,
            }
        )
        if trade.exit_timestamp:
            trades_list.append(
                {
                    "timestamp": trade.exit_timestamp,
                    "action": "EXIT",
                    "asset_id": trade.asset_id,
                    "quantity": trade.quantity,
                    "price": trade.exit_price,
                }
            )

    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

    return BacktestResult(
        framework="ml4t.backtest",
        initial_capital=initial_capital,
        final_value=portfolio.equity,
        total_return=(portfolio.equity / initial_capital - 1) * 100,
        num_trades=len(broker.trade_tracker.trades),
        execution_time=end - start,
        memory_mb=memory_mb,
        trades_df=trades_df,
    )


def run_vectorbt_backtest(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    initial_capital: float = 100_000,
) -> BacktestResult:
    """Run backtest on VectorBT Pro."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        import vectorbt as vbt

    # Pivot price data to wide format
    price_wide = price_df.pivot(index="timestamp", columns="asset_id", values="close")

    # Pivot signals to wide format
    signals_wide = signals_df.pivot(index="timestamp", columns="asset_id", values="signal")

    # Align
    signals_wide = signals_wide.reindex(price_wide.index).fillna(0).astype(bool)

    # Measure
    gc.collect()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None

    start = time.perf_counter()
    pf = vbt.Portfolio.from_signals(
        close=price_wide,
        entries=signals_wide,
        exits=~signals_wide,
        size=initial_capital / 25,  # Equal weight 25 positions
        size_type="value",
        fees=0.001,
        freq="D",
    )
    end = time.perf_counter()

    mem_after = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None
    memory_mb = mem_after - mem_before if mem_before and mem_after else None

    # Extract trades
    try:
        trades = pf.trades.records_readable
        trades_df = pd.DataFrame(
            {
                "timestamp": trades["Entry Timestamp"],
                "action": "TRADE",
                "asset_id": trades["Column"],
                "quantity": trades["Size"],
                "price": trades["Avg Entry Price"],
            }
        )
    except Exception:
        trades_df = pd.DataFrame()

    return BacktestResult(
        framework="VectorBT",
        initial_capital=initial_capital,
        final_value=pf.final_value,
        total_return=pf.total_return * 100,
        num_trades=pf.trades.count(),
        execution_time=end - start,
        memory_mb=memory_mb,
        trades_df=trades_df,
    )


def run_backtrader_backtest(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    initial_capital: float = 100_000,
) -> BacktestResult:
    """Run backtest on Backtrader."""
    import backtrader as bt

    # Merge signals into price data
    merged = price_df.merge(signals_df, on=["timestamp", "asset_id"], how="left")
    merged["signal"] = merged["signal"].fillna(0)

    # Create multi-asset data feed
    class PandasData(bt.feeds.PandasData):
        lines = ("signal",)
        params = (("signal", -1),)

    cerebro = bt.Cerebro()

    # Add data for each symbol
    for asset_id in merged["asset_id"].unique():
        symbol_data = merged[merged["asset_id"] == asset_id].copy()
        symbol_data = symbol_data.set_index("timestamp")
        data = PandasData(dataname=symbol_data, name=asset_id)
        cerebro.adddata(data)

    # Strategy
    class SignalFollower(bt.Strategy):
        def __init__(self):
            self.trades = []

        def next(self):
            for i, data in enumerate(self.datas):
                signal = data.signal[0]
                position = self.getposition(data).size

                target_size = initial_capital / 25 / data.close[0]

                if signal == 1 and position == 0:
                    self.order_target_size(data, target=target_size)
                    self.trades.append(
                        {
                            "timestamp": data.datetime.date(0),
                            "action": "BUY",
                            "asset_id": data._name,
                            "quantity": target_size,
                            "price": data.close[0],
                        }
                    )
                elif signal == 0 and position > 0:
                    self.order_target_size(data, target=0)
                    self.trades.append(
                        {
                            "timestamp": data.datetime.date(0),
                            "action": "SELL",
                            "asset_id": data._name,
                            "quantity": position,
                            "price": data.close[0],
                        }
                    )

    cerebro.addstrategy(SignalFollower)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_coc(True)  # Same-bar execution

    # Measure
    gc.collect()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None

    start = time.perf_counter()
    results = cerebro.run()
    end = time.perf_counter()

    mem_after = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else None
    memory_mb = mem_after - mem_before if mem_before and mem_after else None

    strategy = results[0]
    trades_df = pd.DataFrame(strategy.trades) if strategy.trades else pd.DataFrame()

    return BacktestResult(
        framework="Backtrader",
        initial_capital=initial_capital,
        final_value=cerebro.broker.getvalue(),
        total_return=(cerebro.broker.getvalue() / initial_capital - 1) * 100,
        num_trades=len(strategy.trades),
        execution_time=end - start,
        memory_mb=memory_mb,
        trades_df=trades_df,
    )


# ==============================================================================
# MAIN COMPARISON
# ==============================================================================


def run_comparison():
    """Run cross-framework comparison."""
    print("=" * 80)
    print("HIGH-TURNOVER CROSS-FRAMEWORK VALIDATION")
    print("=" * 80)
    print()

    # Parameters
    N_SYMBOLS = 500
    N_DAYS = 252
    N_POSITIONS = 25
    REBALANCE_DAYS = 5
    INITIAL_CAPITAL = 100_000
    SEED = 42

    print(f"Test Parameters:")
    print(f"  Symbols: {N_SYMBOLS}")
    print(f"  Days: {N_DAYS}")
    print(f"  Positions: {N_POSITIONS}")
    print(f"  Rebalance: Every {REBALANCE_DAYS} days")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Random Seed: {SEED}")
    print()

    # Generate data
    print("[1/4] Generating synthetic data...")
    price_df = generate_synthetic_data(N_SYMBOLS, N_DAYS, SEED)
    signals_df = generate_random_signals(price_df, N_POSITIONS, REBALANCE_DAYS, SEED)
    print()

    # Save data for inspection
    output_dir = Path(__file__).parent / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    price_df.to_parquet(output_dir / "price_data.parquet")
    signals_df.to_parquet(output_dir / "signals_data.parquet")
    print(f"Data saved to {output_dir}/")
    print()

    # Run backtests
    results = {}

    print("[2/4] Running ml4t.backtest...")
    try:
        results["ml4t"] = run_ml4t_backtest(price_df, signals_df, INITIAL_CAPITAL)
        print(f"  ✓ Completed in {results['ml4t'].execution_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
    print()

    print("[3/4] Running VectorBT Pro...")
    try:
        results["vectorbt"] = run_vectorbt_backtest(price_df, signals_df, INITIAL_CAPITAL)
        print(f"  ✓ Completed in {results['vectorbt'].execution_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
    print()

    print("[4/4] Running Backtrader...")
    try:
        results["backtrader"] = run_backtrader_backtest(price_df, signals_df, INITIAL_CAPITAL)
        print(f"  ✓ Completed in {results['backtrader'].execution_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
    print()

    # Compare results
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()

    print(f"{'Framework':<20} {'Trades':<10} {'Return':<12} {'Time (s)':<12} {'Memory (MB)':<12}")
    print("-" * 80)

    for name, result in results.items():
        memory_str = f"{result.memory_mb:.1f}" if result.memory_mb else "N/A"
        print(
            f"{result.framework:<20} "
            f"{result.num_trades:<10} "
            f"{result.total_return:>10.2f}% "
            f"{result.execution_time:>10.2f}  "
            f"{memory_str:>10}"
        )

    print()

    # Analysis
    if len(results) > 1:
        print("ANALYSIS:")
        print()

        # Trade count variance
        trade_counts = [r.num_trades for r in results.values()]
        trade_std = np.std(trade_counts)
        trade_mean = np.mean(trade_counts)
        trade_cv = (trade_std / trade_mean * 100) if trade_mean > 0 else 0

        print(f"Trade Count Variance: {trade_cv:.2f}% (std: {trade_std:.1f}, mean: {trade_mean:.1f})")

        # Return variance
        returns = [r.total_return for r in results.values()]
        return_std = np.std(returns)
        return_mean = np.mean(returns)

        print(f"Return Variance: {return_std:.4f}% (mean: {return_mean:.2f}%)")

        # Speed comparison
        times = {name: r.execution_time for name, r in results.items()}
        fastest = min(times.values())
        print()
        print("Speed Comparison (vs fastest):")
        for name, t in times.items():
            ratio = t / fastest
            print(f"  {results[name].framework}: {ratio:.2f}x")

        # Memory comparison
        memories = {name: r.memory_mb for name, r in results.items() if r.memory_mb}
        if memories:
            print()
            print("Memory Usage:")
            for name, m in memories.items():
                print(f"  {results[name].framework}: {m:.1f} MB")

    # Save results
    results_file = output_dir / "comparison_summary.txt"
    with open(results_file, "w") as f:
        f.write(f"High-Turnover Cross-Framework Validation\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Symbols: {N_SYMBOLS}\n")
        f.write(f"  Days: {N_DAYS}\n")
        f.write(f"  Positions: {N_POSITIONS}\n")
        f.write(f"  Rebalance: Every {REBALANCE_DAYS} days\n\n")
        f.write(f"Results:\n")
        for name, result in results.items():
            f.write(f"\n{result.framework}:\n")
            f.write(f"  Trades: {result.num_trades}\n")
            f.write(f"  Return: {result.total_return:.2f}%\n")
            f.write(f"  Time: {result.execution_time:.2f}s\n")
            f.write(f"  Memory: {result.memory_mb:.1f} MB\n" if result.memory_mb else "")

    print(f"\nResults saved to {results_file}")

    # Save trade logs
    for name, result in results.items():
        if not result.trades_df.empty:
            trades_file = output_dir / f"trades_{name}.csv"
            result.trades_df.to_csv(trades_file, index=False)
            print(f"Trades saved to {trades_file}")


if __name__ == "__main__":
    run_comparison()
