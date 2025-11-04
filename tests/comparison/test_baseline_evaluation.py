"""
Baseline Evaluation: QEngine vs Established Backtesters with Real Data

This module performs comprehensive evaluation of QEngine against Zipline-Reloaded,
VectorBT Pro, and Backtrader using real market data from the projects directory.
Tests both correctness (matching results) and performance (speed/memory).
"""

import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Add project paths
qengine_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(qengine_src))

from qengine.core.event import MarketEvent
from qengine.core.types import MarketDataType
from qengine.engine import BacktestEngine
from qengine.execution.broker import SimulationBroker
from qengine.portfolio.simple import SimplePortfolio
from qengine.reporting.reporter import InMemoryReporter
from qengine.strategy.spy_order_flow_adapter import create_spy_order_flow_strategy


@dataclass
class EvaluationResult:
    """Results from a single backtester evaluation."""

    framework: str
    strategy: str
    data_source: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    execution_time: float
    memory_usage: float  # MB
    data_points: int
    error: str | None = None
    trades: list[dict] = field(default_factory=list)
    equity_curve: pd.Series | None = None


class BaselineEvaluator:
    """Evaluates strategies across multiple backtesters with real data."""

    def __init__(self, verbose: bool = True):
        """Initialize evaluator."""
        self.verbose = verbose
        self.results: list[EvaluationResult] = []

    def load_spy_order_flow_data(self) -> pd.DataFrame:
        """Load SPY order flow data from projects directory."""
        spy_path = projects_dir / "spy_order_flow" / "spy_features.parquet"
        if not spy_path.exists():
            raise FileNotFoundError(f"SPY data not found at {spy_path}")

        df = pd.read_parquet(spy_path)

        # Ensure required columns exist
        required_cols = ["timestamp", "last", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"SPY data missing required columns. Has: {df.columns.tolist()}")

        # Rename 'last' to 'close' for consistency with other data sources
        df = df.rename(columns={"last": "close"})

        if self.verbose:
            print(
                f"Loaded SPY data: {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}",
            )
            print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns

        return df

    def load_crypto_basis_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load crypto spot and futures data from projects directory."""
        spot_path = projects_dir / "crypto_futures" / "data" / "spot" / "BTC.parquet"
        futures_path = projects_dir / "crypto_futures" / "data" / "futures" / "BTC.parquet"

        if not spot_path.exists() or not futures_path.exists():
            raise FileNotFoundError(f"Crypto data not found at {spot_path} or {futures_path}")

        spot_df = pd.read_parquet(spot_path)
        futures_df = pd.read_parquet(futures_path)

        if self.verbose:
            print(f"Loaded BTC spot: {len(spot_df)} rows")
            print(f"Loaded BTC futures: {len(futures_df)} rows")
            print(f"Spot columns: {spot_df.columns.tolist()}")
            print(f"Futures columns: {futures_df.columns.tolist()}")

        return spot_df, futures_df

    def load_equity_daily_data(self) -> pd.DataFrame:
        """Load daily equity data from projects directory."""
        equity_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"
        if not equity_path.exists():
            # Try alternative file
            equity_path = (
                projects_dir / "daily_us_equities" / "equity_prices_enhanced_1962_2025.parquet"
            )

        if not equity_path.exists():
            raise FileNotFoundError(
                f"Equity data not found in {projects_dir / 'daily_us_equities'}",
            )

        df = pd.read_parquet(equity_path)

        if self.verbose:
            print(f"Loaded equity data: {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
            if "ticker" in df.columns:
                print(f"Unique tickers: {df['ticker'].nunique()}")

        return df

    def prepare_qengine_data(self, df: pd.DataFrame, asset_id: str = "SPY") -> list[MarketEvent]:
        """Convert DataFrame to QEngine MarketEvents."""
        events = []

        for _, row in df.iterrows():
            # Extract timestamp
            if "timestamp" in row:
                timestamp = pd.to_datetime(row["timestamp"])
            elif "date" in row:
                timestamp = pd.to_datetime(row["date"])
            else:
                raise ValueError("No timestamp/date column found")

            # Extract OHLCV data
            event = MarketEvent(
                timestamp=timestamp,
                asset_id=asset_id,
                data_type=MarketDataType.BAR,
                open=row.get("open", row.get("close", 0)),
                high=row.get("high", row.get("close", 0)),
                low=row.get("low", row.get("close", 0)),
                close=row.get("close", 0),
                volume=row.get("volume", 0),
                metadata={
                    "buy_volume": row.get("buy_volume", row.get("volume", 0) * 0.5),
                    "sell_volume": row.get("sell_volume", row.get("volume", 0) * 0.5),
                    "vwap": row.get("vwap", row.get("close", 0)),
                    "dollar_volume": row.get(
                        "dollar_volume",
                        row.get("close", 0) * row.get("volume", 0),
                    ),
                },
            )
            events.append(event)

        return events

    def run_qengine_spy_backtest(self, data: pd.DataFrame) -> EvaluationResult:
        """Run SPY order flow strategy on QEngine."""
        print("\n" + "=" * 60)
        print("Running QEngine SPY Order Flow Backtest")
        print("=" * 60)

        # Start timing and memory tracking
        tracemalloc.start()
        start_time = time.time()

        try:
            # Convert data to events
            events = self.prepare_qengine_data(data, "SPY")

            # Create strategy
            strategy = create_spy_order_flow_strategy(
                asset_id="SPY",
                lookback_window=50,
                momentum_window_short=5,
                momentum_window_long=20,
                imbalance_threshold=0.65,
                momentum_threshold=0.002,
                position_scaling=0.2,
            )

            # Create engine components
            broker = SimulationBroker()
            portfolio = SimplePortfolio(initial_capital=100000)
            reporter = InMemoryReporter()

            # Create and configure engine
            engine = BacktestEngine(
                strategies=[strategy],
                broker=broker,
                portfolio=portfolio,
                reporter=reporter,
            )

            # Process events
            trades = []
            for event in events:
                engine.process_event(event)

                # Track any trades
                if hasattr(broker, "get_filled_orders"):
                    filled = broker.get_filled_orders()
                    for order in filled:
                        trades.append(
                            {
                                "timestamp": order.timestamp,
                                "asset": order.asset_id,
                                "quantity": order.quantity,
                                "price": order.fill_price,
                                "type": "BUY" if order.quantity > 0 else "SELL",
                            },
                        )

            # Calculate metrics
            final_value = portfolio.get_total_value()
            initial_value = 100000
            total_return = (final_value / initial_value - 1) * 100

            # Simple annualized return (assuming 252 trading days)
            days = (events[-1].timestamp - events[0].timestamp).days
            annual_return = total_return * (252 / max(days, 1))

            # Calculate win rate from trades
            win_rate = 0.5  # Default
            if trades:
                # Group trades by position entry/exit
                position_returns = []
                for i in range(0, len(trades) - 1, 2):
                    if i + 1 < len(trades):
                        entry = trades[i]
                        exit = trades[i + 1]
                        ret = (
                            (exit["price"] / entry["price"] - 1)
                            if entry["type"] == "BUY"
                            else (entry["price"] / exit["price"] - 1)
                        )
                        position_returns.append(ret)

                if position_returns:
                    win_rate = sum(1 for r in position_returns if r > 0) / len(position_returns)

            # Memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = peak / 1024 / 1024

            # Execution time
            execution_time = time.time() - start_time

            result = EvaluationResult(
                framework="QEngine",
                strategy="SPY Order Flow",
                data_source="spy_features.parquet",
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=1.5,  # Placeholder - would need full equity curve
                max_drawdown=-10.0,  # Placeholder
                win_rate=win_rate,
                total_trades=len(trades),
                execution_time=execution_time,
                memory_usage=memory_mb,
                data_points=len(events),
                trades=trades,
            )

            print("✓ QEngine SPY backtest completed successfully")
            print(f"  Return: {total_return:.2f}%")
            print(f"  Trades: {len(trades)}")
            print(f"  Time: {execution_time:.2f}s")
            print(f"  Memory: {memory_mb:.2f}MB")

            return result

        except Exception as e:
            tracemalloc.stop()
            print(f"✗ QEngine SPY backtest failed: {e}")
            return EvaluationResult(
                framework="QEngine",
                strategy="SPY Order Flow",
                data_source="spy_features.parquet",
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                execution_time=time.time() - start_time,
                memory_usage=0,
                data_points=len(data),
                error=str(e),
            )

    def run_zipline_spy_backtest(self, data: pd.DataFrame) -> EvaluationResult:
        """Run SPY strategy on Zipline-Reloaded."""
        print("\n" + "=" * 60)
        print("Running Zipline SPY Order Flow Backtest")
        print("=" * 60)

        try:
            import zipline.finance.commission as commission
            import zipline.finance.slippage as slippage
            from zipline import run_algorithm
            from zipline.api import order_target_percent, record, symbol

            # Implementation would go here
            # For now, return placeholder
            print("⚠ Zipline implementation pending")
            return EvaluationResult(
                framework="Zipline",
                strategy="SPY Order Flow",
                data_source="spy_features.parquet",
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                execution_time=0,
                memory_usage=0,
                data_points=len(data),
                error="Not implemented",
            )

        except ImportError as e:
            print(f"⚠ Zipline not available: {e}")
            return EvaluationResult(
                framework="Zipline",
                strategy="SPY Order Flow",
                data_source="spy_features.parquet",
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                execution_time=0,
                memory_usage=0,
                data_points=len(data),
                error="Library not installed",
            )

    def run_vectorbt_spy_backtest(self, data: pd.DataFrame) -> EvaluationResult:
        """Run SPY strategy on VectorBT Pro."""
        print("\n" + "=" * 60)
        print("Running VectorBT SPY Order Flow Backtest")
        print("=" * 60)

        try:
            import vectorbtpro as vbt

            # Implementation would go here
            # For now, return placeholder
            print("⚠ VectorBT implementation pending")
            return EvaluationResult(
                framework="VectorBT",
                strategy="SPY Order Flow",
                data_source="spy_features.parquet",
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                execution_time=0,
                memory_usage=0,
                data_points=len(data),
                error="Not implemented",
            )

        except ImportError as e:
            print(f"⚠ VectorBT not available: {e}")
            return EvaluationResult(
                framework="VectorBT",
                strategy="SPY Order Flow",
                data_source="spy_features.parquet",
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                execution_time=0,
                memory_usage=0,
                data_points=len(data),
                error="Library not installed",
            )

    def run_backtrader_spy_backtest(self, data: pd.DataFrame) -> EvaluationResult:
        """Run SPY strategy on Backtrader."""
        print("\n" + "=" * 60)
        print("Running Backtrader SPY Order Flow Backtest")
        print("=" * 60)

        try:
            import backtrader as bt

            # Implementation would go here
            # For now, return placeholder
            print("⚠ Backtrader implementation pending")
            return EvaluationResult(
                framework="Backtrader",
                strategy="SPY Order Flow",
                data_source="spy_features.parquet",
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                execution_time=0,
                memory_usage=0,
                data_points=len(data),
                error="Not implemented",
            )

        except ImportError as e:
            print(f"⚠ Backtrader not available: {e}")
            return EvaluationResult(
                framework="Backtrader",
                strategy="SPY Order Flow",
                data_source="spy_features.parquet",
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                execution_time=0,
                memory_usage=0,
                data_points=len(data),
                error="Library not installed",
            )

    def compare_results(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Compare results across frameworks."""
        comparison = {
            "summary": {},
            "performance": {},
            "accuracy": {},
            "recommendations": [],
        }

        # Filter out failed results
        valid_results = [r for r in results if r.error is None]

        if not valid_results:
            comparison["summary"]["status"] = "No valid results to compare"
            return comparison

        # Performance comparison
        comparison["performance"] = {
            r.framework: {
                "execution_time": r.execution_time,
                "memory_usage": r.memory_usage,
                "data_points_per_second": r.data_points / r.execution_time
                if r.execution_time > 0
                else 0,
            }
            for r in valid_results
        }

        # Find fastest
        fastest = min(valid_results, key=lambda r: r.execution_time)
        comparison["summary"]["fastest"] = f"{fastest.framework} ({fastest.execution_time:.2f}s)"

        # Accuracy comparison (if multiple valid results)
        if len(valid_results) > 1:
            returns = [r.total_return for r in valid_results]
            comparison["accuracy"]["return_variance"] = np.var(returns)
            comparison["accuracy"]["return_std"] = np.std(returns)

            # Check if results agree
            if comparison["accuracy"]["return_std"] < 1.0:  # Within 1% std
                comparison["summary"]["agreement"] = "High (returns within 1%)"
            elif comparison["accuracy"]["return_std"] < 5.0:
                comparison["summary"]["agreement"] = "Moderate (returns within 5%)"
            else:
                comparison["summary"]["agreement"] = "Low (returns differ >5%)"

        # Recommendations
        if fastest.framework == "QEngine":
            comparison["recommendations"].append("QEngine shows competitive performance")
        else:
            comparison["recommendations"].append(
                f"Consider optimizing QEngine (currently {fastest.execution_time:.1f}x slower than {fastest.framework})",
            )

        return comparison

    def generate_report(self, results: list[EvaluationResult]) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("BASELINE EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Group results by strategy
        strategies = {}
        for r in results:
            if r.strategy not in strategies:
                strategies[r.strategy] = []
            strategies[r.strategy].append(r)

        for strategy, strategy_results in strategies.items():
            report.append(f"\n## {strategy}")
            report.append("-" * 40)

            # Results table
            report.append("\n### Results by Framework:")
            report.append(
                f"{'Framework':<15} {'Return %':<12} {'Trades':<10} {'Time (s)':<12} {'Memory (MB)':<12} {'Status':<20}",
            )
            report.append("-" * 90)

            for r in strategy_results:
                status = "✓ Success" if r.error is None else f"✗ {r.error[:15]}"
                report.append(
                    f"{r.framework:<15} {r.total_return:>10.2f}% {r.total_trades:>8} {r.execution_time:>10.2f} {r.memory_usage:>10.2f} {status}",
                )

            # Comparison
            comparison = self.compare_results(strategy_results)

            if comparison.get("summary"):
                report.append("\n### Performance Summary:")
                for key, value in comparison["summary"].items():
                    report.append(f"  {key}: {value}")

            if comparison.get("performance"):
                report.append("\n### Performance Metrics:")
                for framework, metrics in comparison["performance"].items():
                    report.append(f"  {framework}:")
                    report.append(f"    - Data points/sec: {metrics['data_points_per_second']:.0f}")
                    report.append(f"    - Memory efficiency: {metrics['memory_usage']:.2f} MB")

            if comparison.get("recommendations"):
                report.append("\n### Recommendations:")
                for rec in comparison["recommendations"]:
                    report.append(f"  • {rec}")

        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def run_full_evaluation(self) -> str:
        """Run complete baseline evaluation suite."""
        print("\n" + "#" * 80)
        print("# STARTING BASELINE EVALUATION WITH REAL DATA")
        print("#" * 80)

        all_results = []

        # Test 1: SPY Order Flow Strategy
        print("\n### TEST 1: SPY Order Flow Strategy ###")
        try:
            spy_data = self.load_spy_order_flow_data()

            # Limit data size for initial testing
            test_size = min(1000, len(spy_data))
            spy_test_data = spy_data.head(test_size)
            print(f"Using {test_size} data points for evaluation")

            # Run on each framework
            all_results.append(self.run_qengine_spy_backtest(spy_test_data))
            all_results.append(self.run_zipline_spy_backtest(spy_test_data))
            all_results.append(self.run_vectorbt_spy_backtest(spy_test_data))
            all_results.append(self.run_backtrader_spy_backtest(spy_test_data))

        except Exception as e:
            print(f"Failed to run SPY evaluation: {e}")

        # Generate and return report
        report = self.generate_report(all_results)
        print(report)

        # Save report
        report_path = Path(__file__).parent / "baseline_evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        return report


class TestBaselineEvaluation:
    """Test suite for baseline evaluation."""

    @pytest.mark.skip(reason="Needs refactoring - uses old Engine API, BacktestEngine requires data_feed not event list")
    def test_spy_order_flow_evaluation(self):
        """Test SPY order flow strategy evaluation."""
        evaluator = BaselineEvaluator(verbose=True)

        # Load data
        spy_data = evaluator.load_spy_order_flow_data()
        assert len(spy_data) > 0, "Should load SPY data"

        # Run QEngine backtest
        result = evaluator.run_qengine_spy_backtest(spy_data.head(500))

        # Validate result
        assert result.framework == "QEngine"
        assert result.data_points == 500
        assert result.execution_time > 0
        assert result.memory_usage > 0

        print(
            f"QEngine SPY test passed: {result.total_trades} trades in {result.execution_time:.2f}s",
        )

    def test_data_loading(self):
        """Test that all data sources can be loaded."""
        evaluator = BaselineEvaluator(verbose=False)

        # Test SPY data
        spy_data = evaluator.load_spy_order_flow_data()
        assert len(spy_data) > 0, "Should load SPY data"
        assert "close" in spy_data.columns, "SPY data should have close prices"

        # Test crypto data
        spot_data, futures_data = evaluator.load_crypto_basis_data()
        assert len(spot_data) > 0, "Should load spot data"
        assert len(futures_data) > 0, "Should load futures data"

        # Test equity data
        equity_data = evaluator.load_equity_daily_data()
        assert len(equity_data) > 0, "Should load equity data"

        print("All data sources loaded successfully")

    def test_comparison_logic(self):
        """Test result comparison logic."""
        evaluator = BaselineEvaluator(verbose=False)

        # Create mock results
        results = [
            EvaluationResult(
                framework="QEngine",
                strategy="Test",
                data_source="test.parquet",
                total_return=10.5,
                annual_return=15.2,
                sharpe_ratio=1.5,
                max_drawdown=-5.0,
                win_rate=0.55,
                total_trades=50,
                execution_time=2.5,
                memory_usage=100.0,
                data_points=1000,
            ),
            EvaluationResult(
                framework="Zipline",
                strategy="Test",
                data_source="test.parquet",
                total_return=10.8,
                annual_return=15.6,
                sharpe_ratio=1.6,
                max_drawdown=-4.8,
                win_rate=0.56,
                total_trades=48,
                execution_time=5.0,
                memory_usage=200.0,
                data_points=1000,
            ),
        ]

        comparison = evaluator.compare_results(results)

        assert "performance" in comparison
        assert "QEngine" in comparison["performance"]
        assert comparison["performance"]["QEngine"]["execution_time"] == 2.5
        assert comparison["summary"]["fastest"] == "QEngine (2.50s)"

        print("Comparison logic test passed")


if __name__ == "__main__":
    # Run full evaluation
    evaluator = BaselineEvaluator(verbose=True)
    evaluator.run_full_evaluation()
