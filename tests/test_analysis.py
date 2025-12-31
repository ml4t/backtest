"""Tests for the analysis module - trade statistics and backtest analysis."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.backtest import DataFeed, Engine, Strategy
from ml4t.backtest.analysis import (
    BacktestAnalyzer,
    TradeStatistics,
    to_equity_dataframe,
    to_returns_series,
    to_trade_record,
    to_trade_records,
)
from ml4t.backtest.types import Trade

# === Test Fixtures ===


@pytest.fixture
def sample_winning_trade() -> Trade:
    """A profitable long trade."""
    return Trade(
        asset="AAPL",
        entry_time=datetime(2024, 1, 10, 10, 0),
        exit_time=datetime(2024, 1, 15, 15, 30),
        entry_price=150.0,
        exit_price=160.0,
        quantity=100,
        pnl=1000.0,
        pnl_percent=6.67,
        bars_held=5,
        commission=10.0,
        slippage=5.0,
        entry_signals={"momentum": 0.8},
        exit_signals={"momentum": -0.2},
        max_favorable_excursion=8.0,
        max_adverse_excursion=-2.0,
    )


@pytest.fixture
def sample_losing_trade() -> Trade:
    """A losing long trade."""
    return Trade(
        asset="MSFT",
        entry_time=datetime(2024, 1, 20, 9, 30),
        exit_time=datetime(2024, 1, 25, 16, 0),
        entry_price=400.0,
        exit_price=380.0,
        quantity=50,
        pnl=-1000.0,
        pnl_percent=-5.0,
        bars_held=5,
        commission=8.0,
        slippage=4.0,
        entry_signals={"momentum": 0.6},
        exit_signals={"momentum": -0.5},
        max_favorable_excursion=2.0,
        max_adverse_excursion=-6.0,
    )


@pytest.fixture
def sample_short_trade() -> Trade:
    """A profitable short trade."""
    return Trade(
        asset="TSLA",
        entry_time=datetime(2024, 2, 1, 10, 0),
        exit_time=datetime(2024, 2, 5, 14, 0),
        entry_price=250.0,
        exit_price=240.0,
        quantity=-100,  # Negative = short
        pnl=1000.0,
        pnl_percent=4.0,
        bars_held=4,
        commission=12.0,
        slippage=6.0,
        entry_signals={},
        exit_signals={},
        max_favorable_excursion=5.0,
        max_adverse_excursion=-3.0,
    )


@pytest.fixture
def mixed_trades(
    sample_winning_trade: Trade,
    sample_losing_trade: Trade,
    sample_short_trade: Trade,
) -> list[Trade]:
    """Mix of winning/losing, long/short trades."""
    return [sample_winning_trade, sample_losing_trade, sample_short_trade]


@pytest.fixture
def sample_equity_curve() -> list[float]:
    """Sample equity curve with gains and losses."""
    return [100000, 101000, 100500, 102000, 103500, 103000, 105000]


@pytest.fixture
def sample_timestamps() -> list[datetime]:
    """Timestamps matching sample_equity_curve."""
    base = datetime(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(7)]


# === Tests for to_trade_record ===


class TestToTradeRecord:
    """Tests for converting Trade to diagnostic record format."""

    def test_basic_long_trade(self, sample_winning_trade: Trade):
        """Test conversion of a basic winning long trade."""
        record = to_trade_record(sample_winning_trade)

        assert record["symbol"] == "AAPL"
        assert record["entry_price"] == 150.0
        assert record["exit_price"] == 160.0
        assert record["pnl"] == 1000.0
        assert record["direction"] == "long"
        assert record["quantity"] == 100
        assert record["timestamp"] == sample_winning_trade.exit_time
        assert record["entry_timestamp"] == sample_winning_trade.entry_time
        assert record["fees"] == 10.0
        assert record["slippage"] == 5.0

    def test_short_trade(self, sample_short_trade: Trade):
        """Test conversion of a short trade."""
        record = to_trade_record(sample_short_trade)

        assert record["direction"] == "short"
        assert record["quantity"] == 100  # Absolute value
        assert record["pnl"] == 1000.0

    def test_metadata_fields(self, sample_winning_trade: Trade):
        """Test that metadata fields are correctly populated."""
        record = to_trade_record(sample_winning_trade)

        metadata = record["metadata"]
        assert metadata["entry_signals"] == {"momentum": 0.8}
        assert metadata["exit_signals"] == {"momentum": -0.2}
        assert metadata["bars_held"] == 5
        assert metadata["pnl_percent"] == 6.67
        assert metadata["mfe"] == 8.0
        assert metadata["mae"] == -2.0

    def test_duration_calculation(self, sample_winning_trade: Trade):
        """Test that duration is correctly calculated."""
        record = to_trade_record(sample_winning_trade)

        expected_duration = sample_winning_trade.exit_time - sample_winning_trade.entry_time
        assert record["duration"] == expected_duration


# === Tests for to_trade_records ===


class TestToTradeRecords:
    """Tests for batch conversion of trades."""

    def test_empty_list(self):
        """Test with no trades."""
        records = to_trade_records([])
        assert records == []

    def test_multiple_trades(self, mixed_trades: list[Trade]):
        """Test conversion of multiple trades."""
        records = to_trade_records(mixed_trades)

        assert len(records) == 3
        assert records[0]["symbol"] == "AAPL"
        assert records[1]["symbol"] == "MSFT"
        assert records[2]["symbol"] == "TSLA"

    def test_preserves_order(self, mixed_trades: list[Trade]):
        """Test that trade order is preserved."""
        records = to_trade_records(mixed_trades)

        for i, (record, trade) in enumerate(zip(records, mixed_trades)):
            assert record["pnl"] == trade.pnl, f"Trade {i} PnL mismatch"


# === Tests for to_returns_series ===


class TestToReturnsSeries:
    """Tests for equity to returns conversion."""

    def test_basic_returns(self, sample_equity_curve: list[float]):
        """Test basic returns calculation."""
        returns = to_returns_series(sample_equity_curve)

        assert len(returns) == len(sample_equity_curve) - 1
        assert returns.dtype == pl.Float64
        assert returns.name == "returns"

        # First return: (101000 - 100000) / 100000 = 0.01
        assert pytest.approx(returns[0], rel=1e-4) == 0.01

    def test_empty_curve(self):
        """Test with empty equity curve."""
        returns = to_returns_series([])

        assert len(returns) == 0
        assert returns.dtype == pl.Float64

    def test_single_value(self):
        """Test with single value (no returns possible)."""
        returns = to_returns_series([100000])

        assert len(returns) == 0

    def test_numpy_array_input(self, sample_equity_curve: list[float]):
        """Test with numpy array input."""
        arr = np.array(sample_equity_curve)
        returns = to_returns_series(arr)

        assert len(returns) == len(sample_equity_curve) - 1

    def test_declining_equity(self):
        """Test with declining equity (negative returns)."""
        equity = [100000, 95000, 90000]
        returns = to_returns_series(equity)

        assert returns[0] < 0  # First return is negative
        assert returns[1] < 0  # Second return is negative


# === Tests for to_equity_dataframe ===


class TestToEquityDataframe:
    """Tests for equity curve DataFrame conversion."""

    def test_with_timestamps(
        self, sample_equity_curve: list[float], sample_timestamps: list[datetime]
    ):
        """Test DataFrame creation with timestamps."""
        df = to_equity_dataframe(sample_equity_curve, sample_timestamps)

        assert len(df) == len(sample_equity_curve)
        assert "timestamp" in df.columns
        assert "equity" in df.columns
        assert "returns" in df.columns
        assert df["timestamp"].to_list() == sample_timestamps

    def test_without_timestamps(self, sample_equity_curve: list[float]):
        """Test DataFrame creation without timestamps (uses bar index)."""
        df = to_equity_dataframe(sample_equity_curve)

        assert len(df) == len(sample_equity_curve)
        assert "bar" in df.columns
        assert "equity" in df.columns
        assert "returns" in df.columns
        assert df["bar"].to_list() == list(range(len(sample_equity_curve)))

    def test_empty_history(self):
        """Test with empty equity history."""
        df = to_equity_dataframe([])

        assert len(df) == 0
        assert "equity" in df.columns
        assert "returns" in df.columns

    def test_returns_calculation(self, sample_equity_curve: list[float]):
        """Test that returns are correctly calculated in DataFrame."""
        df = to_equity_dataframe(sample_equity_curve)

        # First return should be 0 (no prior value)
        assert df["returns"][0] == 0.0

        # Second return: (101000 - 100000) / 100000 = 0.01
        assert pytest.approx(df["returns"][1], rel=1e-4) == 0.01


# === Tests for TradeStatistics ===


class TestTradeStatistics:
    """Tests for trade statistics computation."""

    def test_from_empty_trades(self):
        """Test statistics from empty trade list."""
        stats = TradeStatistics.from_trades([])

        assert stats.n_trades == 0
        assert stats.n_winners == 0
        assert stats.n_losers == 0
        assert stats.win_rate == 0.0
        assert stats.profit_factor is None
        assert stats.avg_winner is None
        assert stats.avg_loser is None
        assert stats.total_pnl == 0.0

    def test_single_winning_trade(self, sample_winning_trade: Trade):
        """Test statistics from single winning trade."""
        stats = TradeStatistics.from_trades([sample_winning_trade])

        assert stats.n_trades == 1
        assert stats.n_winners == 1
        assert stats.n_losers == 0
        assert stats.win_rate == 1.0
        assert stats.avg_winner == 1000.0
        assert stats.avg_loser is None  # No losers
        assert stats.total_pnl == 1000.0
        assert stats.pnl_std == 0.0  # Single trade, no std

    def test_single_losing_trade(self, sample_losing_trade: Trade):
        """Test statistics from single losing trade."""
        stats = TradeStatistics.from_trades([sample_losing_trade])

        assert stats.n_trades == 1
        assert stats.n_winners == 0
        assert stats.n_losers == 1
        assert stats.win_rate == 0.0
        assert stats.avg_winner is None  # No winners
        assert stats.avg_loser == -1000.0
        assert stats.total_pnl == -1000.0

    def test_mixed_trades(self, mixed_trades: list[Trade]):
        """Test statistics from mixed trades."""
        stats = TradeStatistics.from_trades(mixed_trades)

        assert stats.n_trades == 3
        assert stats.n_winners == 2  # winning_trade + short_trade
        assert stats.n_losers == 1
        assert pytest.approx(stats.win_rate, rel=1e-4) == 2 / 3
        assert stats.total_pnl == 1000.0  # 1000 + (-1000) + 1000

    def test_expectancy_calculation(self, mixed_trades: list[Trade]):
        """Test expectancy property calculation."""
        stats = TradeStatistics.from_trades(mixed_trades)

        # expectancy = win_rate * avg_winner + (1 - win_rate) * avg_loser
        expected = stats.win_rate * stats.avg_winner + (1 - stats.win_rate) * stats.avg_loser
        assert pytest.approx(stats.expectancy, rel=1e-4) == expected

    def test_expectancy_no_losers(self, sample_winning_trade: Trade):
        """Test expectancy when no losers (falls back to avg_pnl)."""
        stats = TradeStatistics.from_trades([sample_winning_trade])

        # With no losers, expectancy falls back to avg_pnl
        assert stats.expectancy == stats.avg_pnl

    def test_payoff_ratio_calculation(self, mixed_trades: list[Trade]):
        """Test payoff ratio (avg_winner / |avg_loser|)."""
        stats = TradeStatistics.from_trades(mixed_trades)

        expected = stats.avg_winner / abs(stats.avg_loser)
        assert pytest.approx(stats.payoff_ratio, rel=1e-4) == expected

    def test_payoff_ratio_no_losers(self, sample_winning_trade: Trade):
        """Test payoff ratio when no losers."""
        stats = TradeStatistics.from_trades([sample_winning_trade])

        assert stats.payoff_ratio is None

    def test_profit_factor_calculation(self, mixed_trades: list[Trade]):
        """Test profit factor (gross_profit / gross_loss)."""
        stats = TradeStatistics.from_trades(mixed_trades)

        # Gross profit: 1000 + 1000 = 2000
        # Gross loss: |-1000| = 1000
        assert pytest.approx(stats.profit_factor, rel=1e-4) == 2.0

    def test_profit_factor_no_losers(self, sample_winning_trade: Trade):
        """Test profit factor when no losers."""
        stats = TradeStatistics.from_trades([sample_winning_trade])

        assert stats.profit_factor is None

    def test_summary_format(self, mixed_trades: list[Trade]):
        """Test summary string generation."""
        stats = TradeStatistics.from_trades(mixed_trades)
        summary = stats.summary()

        assert "Trade Statistics" in summary
        assert "Total Trades: 3" in summary
        assert "Win Rate:" in summary
        assert "Profit Factor:" in summary
        assert "Expectancy:" in summary

    def test_summary_no_trades(self):
        """Test summary with no trades."""
        stats = TradeStatistics.from_trades([])
        summary = stats.summary()

        assert "Total Trades: 0" in summary
        assert "Win Rate: 0.00%" in summary

    def test_to_dict(self, mixed_trades: list[Trade]):
        """Test dictionary export."""
        stats = TradeStatistics.from_trades(mixed_trades)
        d = stats.to_dict()

        assert d["n_trades"] == 3
        assert d["win_rate"] == stats.win_rate
        assert d["expectancy"] == stats.expectancy
        assert d["payoff_ratio"] == stats.payoff_ratio
        assert "total_commission" in d
        assert "total_slippage" in d

    def test_commission_slippage_totals(self, mixed_trades: list[Trade]):
        """Test commission and slippage totals."""
        stats = TradeStatistics.from_trades(mixed_trades)

        # Sum of commissions: 10 + 8 + 12 = 30
        assert stats.total_commission == 30.0
        # Sum of slippage: 5 + 4 + 6 = 15
        assert stats.total_slippage == 15.0

    def test_max_winner_loser(self, mixed_trades: list[Trade]):
        """Test max winner and max loser."""
        stats = TradeStatistics.from_trades(mixed_trades)

        assert stats.max_winner == 1000.0  # Both winners have pnl=1000
        assert stats.max_loser == -1000.0


# === Tests for BacktestAnalyzer ===


def generate_simple_prices(start: datetime, periods: int) -> pl.DataFrame:
    """Generate simple uptrending price data."""
    rows = []
    for i in range(periods):
        ts = start + timedelta(days=i)
        price = 100 * (1 + 0.01 * i)  # 1% daily return
        rows.append(
            {
                "timestamp": ts,
                "asset": "SPY",
                "open": price * 0.999,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000000.0,
            }
        )
    return pl.DataFrame(rows)


class SimpleBuyAndHoldStrategy(Strategy):
    """Simple buy and hold for testing."""

    def __init__(self, asset: str = "SPY"):
        self.asset = asset
        self.bought = False

    def on_data(self, timestamp, data, context, broker):
        if not self.bought and self.asset in data:
            price = data[self.asset]["close"]
            qty = int(broker.account.cash * 0.95 / price)
            if qty > 0:
                broker.submit_order(self.asset, qty)
                self.bought = True


class SimpleLongShortStrategy(Strategy):
    """Strategy that creates trades for testing."""

    def __init__(self):
        self.bar_count = 0
        self.position = 0

    def on_data(self, timestamp, data, context, broker):
        self.bar_count += 1
        if "SPY" not in data:
            return

        # Buy on bar 2
        if self.bar_count == 2 and self.position == 0:
            broker.submit_order("SPY", 100)
            self.position = 100

        # Sell on bar 6
        if self.bar_count == 6 and self.position > 0:
            broker.submit_order("SPY", -100)
            self.position = 0


class TestBacktestAnalyzer:
    """Tests for BacktestAnalyzer class."""

    @pytest.fixture
    def completed_engine(self) -> Engine:
        """Engine that has completed a backtest with trades."""
        prices = generate_simple_prices(datetime(2024, 1, 1), 10)
        feed = DataFeed(prices_df=prices)
        strategy = SimpleLongShortStrategy()
        engine = Engine(feed, strategy, initial_cash=100000)
        engine.run()
        return engine

    @pytest.fixture
    def buy_hold_engine(self) -> Engine:
        """Engine with buy and hold (no closed trades)."""
        prices = generate_simple_prices(datetime(2024, 1, 1), 10)
        feed = DataFeed(prices_df=prices)
        strategy = SimpleBuyAndHoldStrategy()
        engine = Engine(feed, strategy, initial_cash=100000)
        engine.run()
        return engine

    def test_initialization(self, completed_engine: Engine):
        """Test analyzer initialization."""
        analyzer = BacktestAnalyzer(completed_engine)

        assert analyzer.engine is completed_engine
        assert analyzer.broker is completed_engine.broker
        assert analyzer._trade_stats is None  # Not computed yet

    def test_trades_property(self, completed_engine: Engine):
        """Test trades property returns broker trades."""
        analyzer = BacktestAnalyzer(completed_engine)

        trades = analyzer.trades
        assert isinstance(trades, list)
        assert len(trades) == 1  # SimpleLongShortStrategy creates 1 trade

    def test_equity_history_property(self, completed_engine: Engine):
        """Test equity history property."""
        analyzer = BacktestAnalyzer(completed_engine)

        equity = analyzer.equity_history
        assert isinstance(equity, list)
        assert len(equity) > 0
        assert all(isinstance(v, int | float) for v in equity)

    def test_trade_statistics_caching(self, completed_engine: Engine):
        """Test that trade statistics are cached."""
        analyzer = BacktestAnalyzer(completed_engine)

        stats1 = analyzer.trade_statistics()
        stats2 = analyzer.trade_statistics()

        assert stats1 is stats2  # Same object (cached)

    def test_get_trade_records(self, completed_engine: Engine):
        """Test get_trade_records returns diagnostic format."""
        analyzer = BacktestAnalyzer(completed_engine)

        records = analyzer.get_trade_records()
        assert isinstance(records, list)
        assert len(records) == 1

        # Verify record structure
        record = records[0]
        assert "symbol" in record
        assert "pnl" in record
        assert "metadata" in record

    def test_get_returns_series(self, completed_engine: Engine):
        """Test get_returns_series."""
        analyzer = BacktestAnalyzer(completed_engine)

        returns = analyzer.get_returns_series()
        assert isinstance(returns, pl.Series)
        assert len(returns) > 0

    def test_get_equity_dataframe(self, completed_engine: Engine):
        """Test get_equity_dataframe."""
        analyzer = BacktestAnalyzer(completed_engine)

        df = analyzer.get_equity_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert "equity" in df.columns
        assert "returns" in df.columns

    def test_get_trades_dataframe(self, completed_engine: Engine):
        """Test get_trades_dataframe with trades."""
        analyzer = BacktestAnalyzer(completed_engine)

        df = analyzer.get_trades_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1

        # Check expected columns
        expected_cols = [
            "asset",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "quantity",
            "pnl",
            "direction",
        ]
        for col in expected_cols:
            assert col in df.columns

    def test_get_trades_dataframe_empty(self, buy_hold_engine: Engine):
        """Test get_trades_dataframe with no closed trades."""
        analyzer = BacktestAnalyzer(buy_hold_engine)

        df = analyzer.get_trades_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_summary(self, completed_engine: Engine):
        """Test summary generation."""
        analyzer = BacktestAnalyzer(completed_engine)

        summary = analyzer.summary()
        assert "Backtest Summary" in summary
        assert "Initial Capital:" in summary
        assert "Final Value:" in summary
        assert "Total Return:" in summary
        assert "Trade Statistics" in summary

    def test_summary_with_no_trades(self, buy_hold_engine: Engine):
        """Test summary with no completed trades."""
        analyzer = BacktestAnalyzer(buy_hold_engine)

        summary = analyzer.summary()
        assert "Backtest Summary" in summary
        assert "Total Trades: 0" in summary
