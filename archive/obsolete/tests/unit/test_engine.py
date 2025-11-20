"""Unit tests for the BacktestEngine."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import polars as pl
import pytest

from ml4t.backtest.core.event import EventType, MarketEvent, OrderEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide, OrderType
from ml4t.backtest.engine import BacktestEngine, BacktestResults


class TestBacktestEngine:
    """Test suite for BacktestEngine."""

    @pytest.fixture
    def mock_data_feed(self):
        """Create a mock data feed."""
        feed = Mock()
        feed.is_exhausted = False
        feed.initialize = Mock()
        feed.reset = Mock()
        feed.get_time_range = Mock(return_value=(datetime(2024, 1, 1), datetime(2024, 12, 31)))
        return feed

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy."""
        strategy = Mock()
        strategy.on_start = Mock()
        strategy.on_end = Mock()
        strategy.on_market_event = Mock()
        strategy.on_fill_event = Mock()
        strategy.reset = Mock()
        return strategy

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = Mock()
        broker.initialize = Mock()
        broker.finalize = Mock()
        broker.on_order_event = Mock()
        broker.get_trades = Mock(return_value=pl.DataFrame())
        broker.reset = Mock()
        return broker

    @pytest.fixture
    def mock_portfolio(self):
        """Create a mock portfolio."""
        portfolio = Mock()
        # Portfolio no longer has initialize/finalize methods (handled in __init__)
        portfolio.on_fill_event = Mock()
        portfolio.update_market_value = Mock()
        # New API: properties and methods
        portfolio.get_all_positions = Mock(return_value={})  # dict[AssetId, Quantity]
        portfolio.returns = pl.Series("returns", [0.0])  # Property, not method
        portfolio.get_performance_metrics = Mock(return_value={})  # New method name
        portfolio.equity = 100000.0  # Property for total portfolio value
        portfolio.reset = Mock()
        return portfolio

    @pytest.fixture
    def mock_reporter(self):
        """Create a mock reporter."""
        reporter = Mock()
        reporter.on_start = Mock()
        reporter.on_end = Mock()
        reporter.on_event = Mock()
        reporter.get_report = Mock(return_value={})
        reporter.reset = Mock()
        return reporter

    @pytest.fixture
    def engine(self, mock_data_feed, mock_strategy, mock_broker, mock_portfolio, mock_reporter):
        """Create a BacktestEngine with mocked components."""
        return BacktestEngine(
            data_feed=mock_data_feed,
            strategy=mock_strategy,
            broker=mock_broker,
            portfolio=mock_portfolio,
            reporter=mock_reporter,
            initial_capital=100000.0,
        )

    def test_engine_initialization(self, engine, mock_data_feed, mock_strategy):
        """Test engine is initialized correctly."""
        assert engine.data_feed == mock_data_feed
        assert engine.strategy == mock_strategy
        assert engine.initial_capital == 100000.0
        assert engine.currency == "USD"
        assert engine.events_processed == 0
        assert engine.start_time is None
        assert engine.end_time is None

    def test_engine_creates_default_components(self, mock_data_feed, mock_strategy):
        """Test engine creates default broker, portfolio, and reporter if not provided."""
        engine = BacktestEngine(data_feed=mock_data_feed, strategy=mock_strategy)

        assert engine.broker is not None
        assert engine.portfolio is not None
        assert engine.reporter is not None

        # Check types
        from ml4t.backtest.execution.broker import SimulationBroker
        from ml4t.backtest.portfolio.portfolio import Portfolio
        from ml4t.backtest.reporting.reporter import InMemoryReporter

        assert isinstance(engine.broker, SimulationBroker)
        assert isinstance(engine.portfolio, Portfolio)
        assert isinstance(engine.reporter, InMemoryReporter)

    def test_event_handler_setup(self, engine):
        """Test event handlers are properly wired up."""
        # Check Clock has subscriptions (replaces event_bus)
        assert engine.clock is not None
        assert len(engine.clock._subscribers) > 0

        # Verify handler setup was called during initialization
        # Components should be subscribed to Clock events
        assert EventType.MARKET in engine.clock._subscribers
        assert EventType.FILL in engine.clock._subscribers
        assert EventType.ORDER in engine.clock._subscribers

    def test_run_basic_flow(
        self,
        engine,
        mock_data_feed,
        mock_strategy,
        mock_broker,
        mock_portfolio,
    ):
        """Test basic run flow with a single market event."""
        # Setup data feed to return one event then exhaust
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
        )

        mock_data_feed.get_next_event = Mock(side_effect=[market_event, None])
        mock_data_feed.is_exhausted = False

        # Run engine (Clock-driven, no event_bus needed)
        results = engine.run()

        # Verify initialization calls (Portfolio no longer has initialize/finalize methods)
        mock_strategy.on_start.assert_called_once()
        mock_broker.initialize.assert_called_once()

        # Verify finalization calls
        mock_strategy.on_end.assert_called_once()
        mock_broker.finalize.assert_called_once()

        # Check results structure
        assert isinstance(results, dict)
        assert "trades" in results
        assert "positions" in results
        assert "returns" in results
        assert "metrics" in results
        assert "events_processed" in results
        assert results["initial_capital"] == 100000.0

    def test_run_with_date_range(self, engine, mock_data_feed):
        """Test run with specific date range."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        mock_data_feed.get_next_event = Mock(return_value=None)
        mock_data_feed.is_exhausted = True

        engine.run(start_date=start_date, end_date=end_date)

        # DataFeed no longer has initialize method - date range is handled by engine

    def test_run_with_max_events(self, mock_strategy, mock_broker, mock_portfolio, mock_reporter):
        """Test run stops at max_events limit."""
        from ml4t.backtest.data.feed import DataFeed

        # Create a simple mock data feed class that works with Clock
        class SimpleMockDataFeed(DataFeed):
            def __init__(self, events):
                self.events = events
                self.index = 0

            def get_next_event(self):
                if self.index >= len(self.events):
                    return None
                event = self.events[self.index]
                self.index += 1
                return event

            def peek_next_timestamp(self):
                if self.index >= len(self.events):
                    return None
                return self.events[self.index].timestamp

            @property
            def is_exhausted(self):
                return self.index >= len(self.events)

            def reset(self):
                self.index = 0

            def seek(self, timestamp):
                pass

        # Create multiple market events
        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, i),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                close=100.0 + i,
            )
            for i in range(10)
        ]

        # Create engine with real mock datafeed
        mock_data_feed = SimpleMockDataFeed(events)
        engine = BacktestEngine(
            data_feed=mock_data_feed,
            strategy=mock_strategy,
            broker=mock_broker,
            portfolio=mock_portfolio,
            reporter=mock_reporter,
            initial_capital=100000.0,
        )

        # Run engine with max_events limit (Clock-driven)
        results = engine.run(max_events=5)

        # Should process exactly 5 events
        assert results["events_processed"] == 5

    def test_reset(
        self,
        engine,
        mock_data_feed,
        mock_strategy,
        mock_broker,
        mock_portfolio,
        mock_reporter,
    ):
        """Test engine reset functionality."""
        # Set some state
        engine.events_processed = 100
        engine.start_time = datetime.now()
        engine.end_time = datetime.now()

        # Reset
        engine.reset()

        # Verify all components were reset
        mock_data_feed.reset.assert_called_once()
        mock_strategy.reset.assert_called_once()
        mock_broker.reset.assert_called_once()
        mock_portfolio.reset.assert_called_once()
        mock_reporter.reset.assert_called_once()

        # Verify state was cleared
        assert engine.events_processed == 0
        assert engine.start_time is None
        assert engine.end_time is None

    def test_compile_results(self, engine, mock_broker, mock_portfolio, mock_reporter):
        """Test results compilation."""
        # Setup mock returns with new API
        mock_broker.get_trades.return_value = pl.DataFrame({"trade_id": [1, 2]})
        mock_portfolio.get_all_positions.return_value = {"AAPL": 100.0}  # New API: dict
        mock_portfolio.returns = pl.Series("returns", [0.01, 0.02])  # New API: property
        mock_portfolio.get_performance_metrics.return_value = {"sharpe_ratio": 1.5}  # New API
        mock_portfolio.equity = 110000.0  # New API: property for total value
        mock_reporter.get_report.return_value = {"events": 100}

        engine.start_time = datetime.now()
        engine.end_time = engine.start_time + timedelta(seconds=10)
        engine.events_processed = 1000

        results = engine._compile_results()

        assert results["initial_capital"] == 100000.0
        assert results["final_value"] == 110000.0
        assert results["total_return"] == pytest.approx(10.0, rel=1e-9)  # 10% return
        assert results["events_processed"] == 1000
        assert results["report"] == {"events": 100}
        assert "duration_seconds" in results
        assert "events_per_second" in results


class TestBacktestResults:
    """Test suite for BacktestResults."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results dictionary."""
        return {
            "trades": pl.DataFrame({"trade_id": [1, 2, 3], "pnl": [100.0, -50.0, 75.0]}),
            "positions": pl.DataFrame({"asset_id": ["AAPL", "GOOGL"], "quantity": [100, 50]}),
            "returns": pl.Series([0.01, -0.005, 0.02]),
            "metrics": {"sharpe_ratio": 1.5, "max_drawdown": -0.1},
            "total_return": 15.0,
            "events_processed": 10000,
            "duration_seconds": 5.5,
        }

    @pytest.fixture
    def results(self, sample_results):
        """Create BacktestResults instance."""
        return BacktestResults(sample_results)

    def test_results_initialization(self, results, sample_results):
        """Test BacktestResults initialization."""
        assert results.results == sample_results
        assert isinstance(results.trades, pl.DataFrame)
        assert isinstance(results.positions, pl.DataFrame)
        assert isinstance(results.returns, pl.Series)
        assert isinstance(results.metrics, dict)

    def test_properties(self, results):
        """Test BacktestResults properties."""
        assert results.total_return == 15.0
        assert results.sharpe_ratio == 1.5
        assert results.max_drawdown == -0.1

    def test_win_rate(self, results):
        """Test win rate calculation."""
        # 2 winning trades out of 3
        assert results.win_rate == pytest.approx(66.67, rel=0.01)

    def test_win_rate_empty_trades(self):
        """Test win rate with no trades."""
        results = BacktestResults({"trades": pl.DataFrame()})
        assert results.win_rate == 0.0

    def test_summary(self, results):
        """Test summary generation."""
        summary = results.summary()
        assert "Total Return: 15.00%" in summary
        assert "Sharpe Ratio: 1.50" in summary
        assert "Max Drawdown: -0.10%" in summary
        assert "Win Rate: 66.67%" in summary
        assert "Total Trades: 3" in summary
        assert "Events Processed: 10,000" in summary

    def test_to_dict(self, results, sample_results):
        """Test conversion to dictionary."""
        assert results.to_dict() == sample_results

    @patch("builtins.open", create=True)
    def test_save_parquet(self, mock_open, results):
        """Test saving results to parquet."""
        with patch.object(results.trades, "write_parquet") as mock_write_trades:
            with patch.object(results.positions, "write_parquet") as mock_write_positions:
                results.save("results.parquet")

                mock_write_trades.assert_called_once_with("results_trades.parquet")
                mock_write_positions.assert_called_once_with("results_positions.parquet")

    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_save_json(self, mock_json_dump, mock_open, results):
        """Test saving results to JSON."""
        results.save("results.json")

        mock_open.assert_called_once_with("results.json", "w")
        mock_json_dump.assert_called_once()

    def test_save_unsupported_format(self, results):
        """Test saving with unsupported format raises error."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            results.save("results.txt")


class TestIntegrationScenarios:
    """Integration tests for complete backtest scenarios."""

    def test_simple_buy_and_hold_scenario(self):
        """Test a simple buy and hold strategy scenario."""
        from ml4t.backtest.data.feed import DataFeed
        from ml4t.backtest.strategy.base import Strategy

        # Create a simple data feed
        class SimpleDataFeed(DataFeed):
            def __init__(self):
                self.prices = [100.0, 101.0, 102.0, 101.5, 103.0]
                self.index = 0

            def initialize(self, start_date=None, end_date=None):
                self.index = 0

            def get_next_event(self):
                if self.index >= len(self.prices):
                    return None

                event = MarketEvent(
                    timestamp=datetime(2024, 1, 1) + timedelta(days=self.index),
                    asset_id="TEST",
                    data_type=MarketDataType.BAR,
                    close=self.prices[self.index],
                )
                self.index += 1
                return event

            @property
            def is_exhausted(self):
                return self.index >= len(self.prices)

            def reset(self):
                self.index = 0

            def peek_next_timestamp(self):
                if self.index >= len(self.prices):
                    return None
                return datetime(2024, 1, 1) + timedelta(days=self.index)

            def seek(self, timestamp):
                # Simple implementation - find the day index
                base = datetime(2024, 1, 1)
                days = (timestamp - base).days
                self.index = max(0, min(days, len(self.prices) - 1))

        # Create a simple buy and hold strategy
        class SimpleBuyAndHold(Strategy):
            def __init__(self):
                super().__init__(name="SimpleBuyAndHold")  # Call parent __init__ to set up _execution_mode
                self.bought = False
                self.portfolio = None
                self.clock = None

            def on_start(self, portfolio, clock):
                self.portfolio = portfolio
                self.clock = clock
                self.bought = False

            def on_market_event(self, event, context=None):
                if not self.bought and self.clock:
                    # Generate buy signal
                    order = OrderEvent(
                        timestamp=event.timestamp,
                        asset_id=event.asset_id,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=100,
                        order_id=f"ORDER_{event.timestamp}",
                    )
                    self.clock.publish(order)
                    self.bought = True

            def on_fill_event(self, event):
                pass

            def on_end(self):
                pass

            def on_event(self, event, pit_data=None):
                # General event handler - delegate to specific handlers
                if hasattr(self, "on_market_event"):
                    self.on_market_event(event)

            def reset(self):
                self.bought = False

        # Run backtest
        engine = BacktestEngine(
            data_feed=SimpleDataFeed(),
            strategy=SimpleBuyAndHold(),
            initial_capital=10000.0,
        )

        results = engine.run()

        # Verify results
        assert results["events_processed"] > 0
        assert results["initial_capital"] == 10000.0
        assert "final_value" in results
        assert "total_return" in results
