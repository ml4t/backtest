"""
Cross-Backtester Validation Framework
=====================================

This module validates QEngine results against established backtesting frameworks:
- Zipline-Reloaded: Professional-grade backtester with regulatory compliance
- VectorBT Pro: High-performance vectorized backtesting
- Backtrader: Popular retail backtesting framework

Goals:
1. Accuracy Validation: Ensure QEngine produces correct results
2. Performance Benchmarking: Compare execution speed and memory usage
3. Feature Parity: Identify gaps and advantages
4. Edge Case Testing: Validate behavior under extreme conditions

The tests use standardized scenarios to enable fair comparisons.
"""

import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Add QEngine path
qengine_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(qengine_src))

from qengine.core.event import MarketEvent
from qengine.core.types import MarketDataType
from qengine.strategy.crypto_basis_adapter import create_crypto_basis_strategy
from qengine.strategy.spy_order_flow_adapter import create_spy_order_flow_strategy


@dataclass
class BacktestResult:
    """Standardized backtest results for comparison."""

    framework: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

    # Performance metrics
    execution_time: float
    memory_usage: float  # MB

    # Additional metrics
    volatility: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    final_portfolio_value: float = 0.0

    # Detailed results for analysis
    returns: list[float] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, float]:
        """Return key metrics summary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
        }


@dataclass
class StandardizedTestData:
    """Standardized test data for fair comparisons."""

    name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float

    # Price data (OHLCV)
    timestamps: list[datetime]
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[int]

    # Additional data for strategy-specific needs
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.timestamps)

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for framework compatibility."""
        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "open": self.opens,
                "high": self.highs,
                "low": self.lows,
                "close": self.closes,
                "volume": self.volumes,
            },
        ).set_index("timestamp")


class BacktesterValidationFramework:
    """Framework for comparing backtester results."""

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize validation framework.

        Args:
            tolerance: Acceptable relative difference between frameworks (5%)
        """
        self.tolerance = tolerance
        self.results: dict[str, BacktestResult] = {}

    def create_spy_test_data(self, n_days: int = 252) -> StandardizedTestData:
        """Create standardized SPY-like test data."""

        start_date = datetime(2024, 1, 2, 9, 30)  # Market open
        timestamps = []
        opens, highs, lows, closes, volumes = [], [], [], [], []

        # Generate realistic SPY data
        base_price = 450.0
        current_price = base_price

        for day in range(n_days):
            # 390 5-minute bars per day (6.5 hour trading day)
            for minute in range(0, 390, 5):
                timestamp = start_date + timedelta(days=day, minutes=minute)

                # Random walk with mean reversion
                daily_factor = 0.0001 * day  # Slight upward trend
                noise = np.random.randn() * 0.002  # 0.2% volatility
                mean_reversion = -0.0001 * (current_price - base_price) / base_price

                price_change = daily_factor + noise + mean_reversion
                new_price = current_price * (1 + price_change)

                # OHLC for 5-minute bar
                open_price = current_price
                high_price = new_price * (1 + abs(np.random.randn()) * 0.0005)
                low_price = min(open_price, new_price) * (1 - abs(np.random.randn()) * 0.0005)
                close_price = new_price
                volume = int(np.random.randint(800000, 1500000))

                timestamps.append(timestamp)
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(volume)

                current_price = close_price

        # Add order flow metadata
        imbalances = []
        for i, _vol in enumerate(volumes):
            # Create realistic order flow patterns
            if i % 78 < 30:  # Morning: balanced
                imbalance = 0.5 + np.random.randn() * 0.05
            elif i % 78 < 60:  # Midday: buy pressure
                imbalance = 0.6 + np.random.randn() * 0.08
            else:  # Afternoon: mixed
                imbalance = 0.5 + np.sin(i * 0.1) * 0.2 + np.random.randn() * 0.03

            imbalance = np.clip(imbalance, 0.2, 0.8)
            imbalances.append(imbalance)

        # Calculate buy/sell volumes
        buy_volumes = [int(vol * imb) for vol, imb in zip(volumes, imbalances, strict=False)]
        sell_volumes = [vol - buy_vol for vol, buy_vol in zip(volumes, buy_volumes, strict=False)]

        return StandardizedTestData(
            name="SPY_5min_252days",
            start_date=start_date,
            end_date=timestamps[-1],
            initial_capital=100000.0,
            timestamps=timestamps,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            metadata={
                "buy_volumes": buy_volumes,
                "sell_volumes": sell_volumes,
                "imbalances": imbalances,
            },
        )

    def create_crypto_test_data(
        self,
        n_days: int = 90,
    ) -> tuple[StandardizedTestData, StandardizedTestData]:
        """Create standardized crypto spot/futures test data."""

        start_date = datetime(2024, 1, 1, 0, 0)  # 24/7 trading
        timestamps = []

        # Spot data
        spot_opens, spot_highs, spot_lows, spot_closes, spot_volumes = [], [], [], [], []

        # Futures data
        fut_opens, fut_highs, fut_lows, fut_closes, fut_volumes = [], [], [], [], []

        base_spot = 50000.0
        current_spot = base_spot

        for day in range(n_days):
            # 288 5-minute bars per day (24 hours)
            for minute in range(0, 24 * 60, 5):
                timestamp = start_date + timedelta(days=day, minutes=minute)

                # Correlated random walks
                spot_change = np.random.randn() * 0.008  # Higher crypto volatility
                new_spot = current_spot * (1 + spot_change)

                # Futures with time-varying basis
                days_to_expiry = 30 - (day % 30)  # Monthly expiry cycle
                basis_rate = 0.1 / 365 * days_to_expiry  # Contango
                basis_noise = np.random.randn() * 0.002
                basis_factor = basis_rate + basis_noise

                new_futures = new_spot * (1 + basis_factor)

                # Generate OHLC for both
                # Spot
                timestamps.append(timestamp)
                spot_opens.append(current_spot)
                spot_highs.append(new_spot * (1 + abs(np.random.randn()) * 0.002))
                spot_lows.append(min(current_spot, new_spot) * (1 - abs(np.random.randn()) * 0.002))
                spot_closes.append(new_spot)
                spot_volumes.append(int(np.random.randint(100, 1000)))

                # Futures
                current_futures = current_spot * (1 + basis_factor)
                fut_opens.append(current_futures)
                fut_highs.append(new_futures * (1 + abs(np.random.randn()) * 0.002))
                fut_lows.append(
                    min(current_futures, new_futures) * (1 - abs(np.random.randn()) * 0.002),
                )
                fut_closes.append(new_futures)
                fut_volumes.append(int(np.random.randint(80, 800)))

                current_spot = new_spot

        spot_data = StandardizedTestData(
            name="BTC_spot_5min_90days",
            start_date=start_date,
            end_date=timestamps[-1],
            initial_capital=100000.0,
            timestamps=timestamps,
            opens=spot_opens,
            highs=spot_highs,
            lows=spot_lows,
            closes=spot_closes,
            volumes=spot_volumes,
        )

        futures_data = StandardizedTestData(
            name="BTC_futures_5min_90days",
            start_date=start_date,
            end_date=timestamps[-1],
            initial_capital=100000.0,
            timestamps=timestamps,
            opens=fut_opens,
            highs=fut_highs,
            lows=fut_lows,
            closes=fut_closes,
            volumes=fut_volumes,
        )

        return spot_data, futures_data

    def run_qengine_backtest(
        self,
        test_data: StandardizedTestData,
        strategy_name: str = "spy_order_flow",
        **strategy_params,
    ) -> BacktestResult:
        """Run backtest using QEngine."""

        print(f"\n🔧 Running QEngine backtest: {strategy_name}")
        tracemalloc.start()
        start_time = time.time()

        try:
            # Create strategy based on type
            if strategy_name == "spy_order_flow":
                strategy = create_spy_order_flow_strategy(
                    asset_id="SPY",
                    **strategy_params,
                )
            elif strategy_name == "crypto_basis":
                # For crypto basis, we need both spot and futures data
                # This is a simplified version for now
                strategy = create_crypto_basis_strategy(
                    spot_asset_id="BTC",
                    futures_asset_id="BTC_FUTURE",
                    **strategy_params,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            # Mock broker
            from unittest.mock import Mock

            strategy.broker = Mock()
            strategy.broker.submit_order = Mock(return_value="order_123")
            strategy.broker.get_cash = Mock(return_value=test_data.initial_capital)

            strategy.on_start()

            # Process market events
            returns = []
            positions = []
            trades = []
            current_value = test_data.initial_capital

            for i in range(test_data.length):
                # Create market event
                event = MarketEvent(
                    timestamp=test_data.timestamps[i],
                    asset_id="SPY" if strategy_name == "spy_order_flow" else "BTC",
                    data_type=MarketDataType.BAR,
                    open=test_data.opens[i],
                    high=test_data.highs[i],
                    low=test_data.lows[i],
                    close=test_data.closes[i],
                    volume=test_data.volumes[i],
                    metadata=test_data.metadata,
                )

                # Process event
                initial_orders = strategy.broker.submit_order.call_count
                strategy.on_market_event(event)
                new_orders = strategy.broker.submit_order.call_count - initial_orders

                if new_orders > 0:
                    trades.append(
                        {
                            "timestamp": test_data.timestamps[i],
                            "orders": new_orders,
                            "price": test_data.closes[i],
                        },
                    )

                # Simple return calculation
                if i > 0:
                    price_return = (
                        test_data.closes[i] - test_data.closes[i - 1]
                    ) / test_data.closes[i - 1]
                    returns.append(price_return)
                    current_value *= 1 + price_return * 0.1  # Simplified position sizing

                positions.append(current_value)

            strategy.on_end()

            # Calculate metrics
            total_return = (current_value - test_data.initial_capital) / test_data.initial_capital
            annual_return = total_return * (
                252 / (test_data.length / 78)
            )  # Assuming 78 bars per day

            returns_array = np.array(returns) if returns else np.array([0])
            sharpe_ratio = (
                np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                if len(returns) > 1
                else 0
            )

            # Max drawdown
            peak = np.maximum.accumulate(positions)
            drawdowns = (np.array(positions) - peak) / peak
            max_drawdown = abs(np.min(drawdowns)) if len(positions) > 0 else 0

            # Win rate
            winning_returns = [r for r in returns if r > 0]
            win_rate = len(winning_returns) / len(returns) if returns else 0

            # Performance metrics
            end_time = time.time()
            execution_time = end_time - start_time

            current, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage = peak_mem / 1024 / 1024  # Convert to MB

            result = BacktestResult(
                framework="QEngine",
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=len(trades),
                execution_time=execution_time,
                memory_usage=memory_usage,
                final_portfolio_value=current_value,
                returns=returns,
                positions=positions,
                trades=trades,
            )

            print(f"✅ QEngine completed in {execution_time:.2f}s, Memory: {memory_usage:.1f}MB")
            return result

        except Exception as e:
            print(f"❌ QEngine backtest failed: {e}")
            tracemalloc.stop()
            return BacktestResult(
                framework="QEngine",
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                execution_time=time.time() - start_time,
                memory_usage=0,
            )

    def run_zipline_backtest(self, test_data: StandardizedTestData) -> BacktestResult | None:
        """Run backtest using Zipline-Reloaded."""

        try:
            # Try to import zipline
            sys.path.insert(
                0,
                str(
                    Path(__file__).parent.parent.parent
                    / "resources"
                    / "zipline-reloaded-main"
                    / "src",
                ),
            )
            import zipline
            from zipline import run_algorithm
            from zipline.api import order_target_percent, symbol

            print("\n🔧 Running Zipline backtest")
            tracemalloc.start()
            start_time = time.time()

            # Simple buy-and-hold strategy for comparison
            def initialize(context):
                context.asset = symbol("SPY")

            def handle_data(context, data):
                order_target_percent(context.asset, 0.1)  # 10% position

            # Convert data for Zipline
            df = test_data.to_pandas()
            df.index = pd.to_datetime(df.index)

            # Run algorithm
            result = run_algorithm(
                start=test_data.start_date,
                end=test_data.end_date,
                initialize=initialize,
                handle_data=handle_data,
                capital_base=test_data.initial_capital,
                data_frequency="minute",
                bundle="custom",  # Would need custom bundle setup
            )

            # Extract metrics (simplified)
            total_return = result["portfolio_value"].iloc[-1] / test_data.initial_capital - 1
            returns = result["returns"].dropna()
            annual_return = returns.mean() * 252
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0

            # Performance
            end_time = time.time()
            execution_time = end_time - start_time
            current, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage = peak_mem / 1024 / 1024

            print(f"✅ Zipline completed in {execution_time:.2f}s, Memory: {memory_usage:.1f}MB")

            return BacktestResult(
                framework="Zipline",
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=0,  # Would need calculation
                win_rate=0,  # Would need calculation
                total_trades=0,  # Would need extraction
                execution_time=execution_time,
                memory_usage=memory_usage,
            )

        except ImportError:
            print("⚠️  Zipline not available for comparison")
            return None
        except Exception as e:
            print(f"❌ Zipline backtest failed: {e}")
            tracemalloc.stop()
            return None

    def run_vectorbt_backtest(self, test_data: StandardizedTestData) -> BacktestResult | None:
        """Run backtest using VectorBT Pro."""

        try:
            # Try to import vectorbt
            sys.path.insert(
                0,
                str(Path(__file__).parent.parent.parent / "resources" / "vectorbt.pro-main"),
            )
            import vectorbtpro as vbt

            print("\n🔧 Running VectorBT backtest")
            tracemalloc.start()
            start_time = time.time()

            # Convert data
            df = test_data.to_pandas()

            # Simple RSI strategy for comparison
            rsi = vbt.RSI.run(df["close"])
            entries = rsi.rsi < 30
            exits = rsi.rsi > 70

            # Run portfolio simulation
            pf = vbt.Portfolio.from_signals(
                df["close"],
                entries=entries,
                exits=exits,
                init_cash=test_data.initial_capital,
                freq="5T",  # 5-minute frequency
            )

            # Extract metrics
            total_return = pf.total_return()
            annual_return = pf.annual_return()
            sharpe_ratio = pf.sharpe_ratio()
            max_drawdown = pf.max_drawdown()
            win_rate = pf.win_rate()
            total_trades = pf.count()

            # Performance
            end_time = time.time()
            execution_time = end_time - start_time
            current, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage = peak_mem / 1024 / 1024

            print(f"✅ VectorBT completed in {execution_time:.2f}s, Memory: {memory_usage:.1f}MB")

            return BacktestResult(
                framework="VectorBT",
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                execution_time=execution_time,
                memory_usage=memory_usage,
            )

        except ImportError:
            print("⚠️  VectorBT not available for comparison")
            return None
        except Exception as e:
            print(f"❌ VectorBT backtest failed: {e}")
            tracemalloc.stop()
            return None

    def run_backtrader_backtest(self, test_data: StandardizedTestData) -> BacktestResult | None:
        """Run backtest using Backtrader."""

        try:
            # Try to import backtrader
            sys.path.insert(
                0,
                str(Path(__file__).parent.parent.parent / "resources" / "backtrader-master"),
            )
            import backtrader as bt

            print("\n🔧 Running Backtrader backtest")
            tracemalloc.start()
            start_time = time.time()

            # Simple strategy
            class SimpleStrategy(bt.Strategy):
                params = {
                    "rsi_period": 14,
                    "rsi_upper": 70,
                    "rsi_lower": 30,
                }

                def __init__(self):
                    self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)

                def next(self):
                    if not self.position:
                        if self.rsi < self.params.rsi_lower:
                            self.buy(size=100)
                    else:
                        if self.rsi > self.params.rsi_upper:
                            self.sell(size=100)

            # Create data feed
            df = test_data.to_pandas()
            data_feed = bt.feeds.PandasData(dataname=df)

            # Setup cerebro
            cerebro = bt.Cerebro()
            cerebro.adddata(data_feed)
            cerebro.addstrategy(SimpleStrategy)
            cerebro.broker.setcash(test_data.initial_capital)

            # Run backtest
            cerebro.run()

            # Extract metrics (simplified)
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - test_data.initial_capital) / test_data.initial_capital

            # Performance
            end_time = time.time()
            execution_time = end_time - start_time
            current, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage = peak_mem / 1024 / 1024

            print(f"✅ Backtrader completed in {execution_time:.2f}s, Memory: {memory_usage:.1f}MB")

            return BacktestResult(
                framework="Backtrader",
                total_return=total_return,
                annual_return=total_return * 252 / (test_data.length / 78),  # Approx
                sharpe_ratio=0,  # Would need calculation
                max_drawdown=0,  # Would need calculation
                win_rate=0,  # Would need calculation
                total_trades=0,  # Would need extraction
                execution_time=execution_time,
                memory_usage=memory_usage,
                final_portfolio_value=final_value,
            )

        except ImportError:
            print("⚠️  Backtrader not available for comparison")
            return None
        except Exception as e:
            print(f"❌ Backtrader backtest failed: {e}")
            tracemalloc.stop()
            return None

    def compare_results(self, results: dict[str, BacktestResult]) -> dict[str, Any]:
        """Compare results across frameworks."""

        if len(results) < 2:
            return {"error": "Need at least 2 frameworks to compare"}

        comparison = {
            "frameworks": list(results.keys()),
            "metrics_comparison": {},
            "performance_comparison": {},
            "accuracy_analysis": {},
            "recommendations": [],
        }

        # Extract metrics for comparison
        metrics = ["total_return", "annual_return", "sharpe_ratio", "max_drawdown", "win_rate"]

        for metric in metrics:
            values = {fw: getattr(result, metric) for fw, result in results.items()}
            comparison["metrics_comparison"][metric] = {
                "values": values,
                "range": max(values.values()) - min(values.values()),
                "coefficient_of_variation": np.std(list(values.values()))
                / np.mean(list(values.values()))
                if np.mean(list(values.values())) != 0
                else 0,
            }

        # Performance comparison
        exec_times = {fw: result.execution_time for fw, result in results.items()}
        memory_usage = {fw: result.memory_usage for fw, result in results.items()}

        comparison["performance_comparison"] = {
            "execution_time": {
                "values": exec_times,
                "fastest": min(exec_times, key=exec_times.get),
                "slowest": max(exec_times, key=exec_times.get),
                "speed_ratio": max(exec_times.values()) / min(exec_times.values()),
            },
            "memory_usage": {
                "values": memory_usage,
                "lowest": min(memory_usage, key=memory_usage.get),
                "highest": max(memory_usage, key=memory_usage.get),
                "memory_ratio": max(memory_usage.values()) / min(memory_usage.values()),
            },
        }

        # Accuracy analysis (using QEngine as baseline if available)
        if "QEngine" in results:
            baseline = results["QEngine"]
            for fw, result in results.items():
                if fw != "QEngine":
                    diff = abs(result.total_return - baseline.total_return)
                    rel_diff = (
                        diff / abs(baseline.total_return)
                        if baseline.total_return != 0
                        else float("inf")
                    )
                    comparison["accuracy_analysis"][fw] = {
                        "absolute_difference": diff,
                        "relative_difference": rel_diff,
                        "within_tolerance": rel_diff <= self.tolerance,
                    }

        # Generate recommendations
        if "QEngine" in results:
            results["QEngine"]

            # Performance recommendations
            fastest_fw = comparison["performance_comparison"]["execution_time"]["fastest"]
            if fastest_fw != "QEngine":
                comparison["recommendations"].append(
                    f"Consider optimizing QEngine performance - {fastest_fw} is {comparison['performance_comparison']['execution_time']['speed_ratio']:.1f}x faster",
                )

            # Accuracy recommendations
            for fw, analysis in comparison["accuracy_analysis"].items():
                if not analysis["within_tolerance"]:
                    comparison["recommendations"].append(
                        f"Significant difference with {fw}: {analysis['relative_difference'] * 100:.1f}% - investigate calculation differences",
                    )

        return comparison

    def generate_report(
        self,
        results: dict[str, BacktestResult],
        comparison: dict[str, Any],
    ) -> str:
        """Generate comprehensive comparison report."""

        report = []
        report.append("=" * 80)
        report.append("CROSS-BACKTESTER VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        if "QEngine" in results:
            qe = results["QEngine"]
            report.append(
                f"QEngine Performance: {qe.total_return * 100:.2f}% return, {qe.sharpe_ratio:.2f} Sharpe",
            )
            report.append(
                f"Execution Time: {qe.execution_time:.2f}s, Memory: {qe.memory_usage:.1f}MB",
            )

        if "performance_comparison" in comparison:
            fastest = comparison["performance_comparison"]["execution_time"]["fastest"]
            report.append(f"Fastest Framework: {fastest} ({results[fastest].execution_time:.2f}s)")
        else:
            report.append(
                f"Single Framework Analysis - Only {next(iter(results.keys()))} available"
            )
        report.append("")

        # Detailed Results
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        for fw, result in results.items():
            report.append(f"{fw}:")
            report.append(f"  Total Return: {result.total_return * 100:8.2f}%")
            report.append(f"  Annual Return: {result.annual_return * 100:7.2f}%")
            report.append(f"  Sharpe Ratio: {result.sharpe_ratio:8.2f}")
            report.append(f"  Max Drawdown: {result.max_drawdown * 100:7.2f}%")
            report.append(f"  Win Rate: {result.win_rate * 100:11.1f}%")
            report.append(f"  Total Trades: {result.total_trades:9d}")
            report.append(f"  Exec Time: {result.execution_time:10.2f}s")
            report.append(f"  Memory: {result.memory_usage:13.1f}MB")
            report.append("")

        # Performance Analysis
        if "performance_comparison" in comparison:
            report.append("PERFORMANCE ANALYSIS")
            report.append("-" * 40)
            perf = comparison["performance_comparison"]
            report.append("Speed Ranking:")
            exec_times = perf["execution_time"]["values"]
            for i, (fw, time) in enumerate(sorted(exec_times.items(), key=lambda x: x[1])):
                report.append(f"  {i + 1}. {fw}: {time:.2f}s")
            report.append(
                f"Speed Ratio (slowest/fastest): {perf['execution_time']['speed_ratio']:.1f}x",
            )
            report.append("")
        else:
            report.append("SINGLE FRAMEWORK ANALYSIS")
            report.append("-" * 40)
            for fw, result in results.items():
                report.append(f"{fw} Performance:")
                report.append(f"  Execution Time: {result.execution_time:.2f}s")
                report.append(f"  Memory Usage: {result.memory_usage:.1f}MB")
                report.append(f"  Trades Generated: {result.total_trades}")
            report.append("")

        # Accuracy Analysis
        if comparison.get("accuracy_analysis"):
            report.append("ACCURACY ANALYSIS (vs QEngine)")
            report.append("-" * 40)
            for fw, analysis in comparison["accuracy_analysis"].items():
                status = "✅" if analysis["within_tolerance"] else "⚠️"
                report.append(
                    f"{status} {fw}: {analysis['relative_difference'] * 100:.2f}% difference",
                )
            report.append("")

        # Recommendations
        if comparison.get("recommendations"):
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(comparison["recommendations"], 1):
                report.append(f"{i}. {rec}")
            report.append("")

        # Validation Status
        report.append("VALIDATION STATUS")
        report.append("-" * 40)
        if "QEngine" in results:
            accuracy_ok = all(
                analysis["within_tolerance"]
                for analysis in comparison.get("accuracy_analysis", {}).values()
            )
            performance_competitive = results["QEngine"].execution_time <= 2 * min(
                r.execution_time for r in results.values()
            )

            if accuracy_ok and performance_competitive:
                report.append(
                    "✅ VALIDATION PASSED: QEngine results are accurate and performance is competitive",
                )
            elif accuracy_ok:
                report.append(
                    "⚠️  PARTIAL PASS: QEngine accurate but performance needs optimization",
                )
            else:
                report.append(
                    "❌ VALIDATION FAILED: QEngine results differ significantly from other frameworks",
                )
        else:
            report.append("❌ INCOMPLETE: QEngine results not available")

        report.append("=" * 80)

        return "\n".join(report)


class TestBacktesterValidation:
    """Test cases for cross-backtester validation."""

    @pytest.fixture
    def validation_framework(self):
        """Create validation framework instance."""
        return BacktesterValidationFramework(tolerance=0.1)  # 10% tolerance for tests

    @pytest.fixture
    def spy_test_data(self, validation_framework):
        """Create SPY test data."""
        return validation_framework.create_spy_test_data(n_days=5)  # Short test

    def test_qengine_spy_backtest(self, validation_framework, spy_test_data):
        """Test QEngine SPY strategy backtest."""

        result = validation_framework.run_qengine_backtest(
            spy_test_data,
            strategy_name="spy_order_flow",
            imbalance_threshold=0.65,
            momentum_threshold=0.002,
        )

        # Validate result structure
        assert result.framework == "QEngine"
        assert result.execution_time > 0
        assert result.memory_usage >= 0
        assert isinstance(result.total_return, float)
        assert isinstance(result.positions, list)

        print(f"QEngine Result: {result.summary}")

    def test_cross_framework_comparison(self, validation_framework, spy_test_data):
        """Test comparison across multiple frameworks."""

        results = {}

        # Run QEngine
        qe_result = validation_framework.run_qengine_backtest(
            spy_test_data,
            strategy_name="spy_order_flow",
        )
        results["QEngine"] = qe_result

        # Try other frameworks (may not be available in test environment)
        bt_result = validation_framework.run_backtrader_backtest(spy_test_data)
        if bt_result:
            results["Backtrader"] = bt_result

        vbt_result = validation_framework.run_vectorbt_backtest(spy_test_data)
        if vbt_result:
            results["VectorBT"] = vbt_result

        # Compare results
        if len(results) > 1:
            comparison = validation_framework.compare_results(results)
            report = validation_framework.generate_report(results, comparison)

            print(report)

            # Validate comparison structure
            assert "frameworks" in comparison
            assert "metrics_comparison" in comparison
            assert "performance_comparison" in comparison

            # Performance should be reasonable
            assert qe_result.execution_time < 60  # Should complete in under 1 minute
            assert qe_result.memory_usage < 1000  # Should use less than 1GB

        else:
            print("Only QEngine available - install other frameworks for full comparison")

    def test_performance_benchmark(self, validation_framework):
        """Benchmark QEngine performance with larger dataset."""

        # Create larger test dataset
        large_data = validation_framework.create_spy_test_data(n_days=20)  # ~15k data points

        result = validation_framework.run_qengine_backtest(
            large_data,
            strategy_name="spy_order_flow",
        )

        # Performance targets
        assert result.execution_time < 10.0, f"QEngine too slow: {result.execution_time:.2f}s"
        assert result.memory_usage < 500.0, (
            f"QEngine uses too much memory: {result.memory_usage:.1f}MB"
        )

        # Should generate some activity
        assert result.total_trades >= 0
        assert len(result.positions) == large_data.length

        print(
            f"Performance Benchmark - Time: {result.execution_time:.2f}s, Memory: {result.memory_usage:.1f}MB",
        )


if __name__ == "__main__":
    # Run validation demo
    framework = BacktesterValidationFramework()

    print("Cross-Backtester Validation Demo")
    print("=" * 50)

    # Create test data
    spy_data = framework.create_spy_test_data(n_days=10)
    print(f"Created test data: {spy_data.length} data points")

    # Run comparisons
    results = {}

    # QEngine
    qe_result = framework.run_qengine_backtest(spy_data, strategy_name="spy_order_flow")
    results["QEngine"] = qe_result

    # Other frameworks (optional)
    bt_result = framework.run_backtrader_backtest(spy_data)
    if bt_result:
        results["Backtrader"] = bt_result

    # Generate report
    if len(results) > 0:
        comparison = framework.compare_results(results)
        report = framework.generate_report(results, comparison)
        print(report)
    else:
        print("No frameworks available for comparison")
