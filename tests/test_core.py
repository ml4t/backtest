"""Test the minimal core backtesting engine."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import (
    Broker,
    DataFeed,
    Engine,
    ExecutionMode,
    Fill,
    OrderSide,
    OrderType,
    Strategy,
    run_backtest,
)
from ml4t.backtest.config import (
    BacktestConfig,
    CommissionType,
    DataFrequency,
    SlippageType,
)
from ml4t.backtest.models import PercentageCommission, VolumeShareSlippage
from ml4t.specs.market_data import FeedSpec

# === Test Data Generators ===


def generate_prices(
    assets: list[str],
    start: datetime,
    periods: int,
    base_prices: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data."""
    if base_prices is None:
        base_prices = dict.fromkeys(assets, 100.0)

    rows = []
    for i in range(periods):
        ts = start + timedelta(days=i)
        for asset in assets:
            base = base_prices[asset]
            # Simple trend with noise
            price = base * (1 + 0.001 * i + 0.01 * (i % 5 - 2))
            rows.append(
                {
                    "timestamp": ts,
                    "asset": asset,
                    "open": price * 0.998,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000000.0,
                }
            )

    return pl.DataFrame(rows)


def generate_signals(
    assets: list[str],
    start: datetime,
    periods: int,
    signal_names: list[str] = None,
) -> pl.DataFrame:
    """Generate synthetic signals."""
    if signal_names is None:
        signal_names = ["ml_score", "momentum"]
    rows = []
    for i in range(periods):
        ts = start + timedelta(days=i)
        for asset in assets:
            row = {"timestamp": ts, "asset": asset}
            for sig in signal_names:
                # Oscillating signals
                row[sig] = 0.5 + 0.3 * ((i + hash(asset)) % 10 - 5) / 5
            rows.append(row)

    return pl.DataFrame(rows)


def generate_context(
    start: datetime,
    periods: int,
) -> pl.DataFrame:
    """Generate synthetic context data (VIX, etc.)."""
    rows = []
    for i in range(periods):
        ts = start + timedelta(days=i)
        rows.append(
            {
                "timestamp": ts,
                "vix": 15 + 5 * (i % 7 - 3),  # Oscillating VIX
                "spy": 450 + i * 0.5,
            }
        )

    return pl.DataFrame(rows)


# === Test Strategies ===


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold for testing."""

    def __init__(self, asset: str):
        self.asset = asset
        self.bought = False

    def on_data(self, timestamp, data, context, broker):
        if not self.bought and self.asset in data:
            broker.submit_order(self.asset, 100)
            self.bought = True


class SignalBasedStrategy(Strategy):
    """Buy on high signal, sell on low signal."""

    def __init__(self, threshold_buy: float = 0.7, threshold_sell: float = 0.3):
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell

    def on_data(self, timestamp, data, context, broker):
        for asset, asset_data in data.items():
            signals = asset_data.get("signals", {})
            ml_score = signals.get("ml_score", 0.5)

            pos = broker.get_position(asset)

            if pos is None and ml_score > self.threshold_buy:
                # Buy 10% of portfolio
                value = broker.get_account_value() * 0.1
                price = asset_data["close"]
                qty = value / price
                broker.submit_order(asset, qty)

            elif pos is not None and ml_score < self.threshold_sell:
                broker.close_position(asset)


class VIXFilterStrategy(Strategy):
    """Only trade when VIX is below threshold."""

    def __init__(self, vix_threshold: float = 20):
        self.vix_threshold = vix_threshold

    def on_data(self, timestamp, data, context, broker):
        vix = context.get("vix", 15)

        if vix > self.vix_threshold:
            # High VIX - close all positions
            for asset in list(broker.positions.keys()):
                broker.close_position(asset)
            return

        # Low VIX - buy assets with good signals
        for asset, asset_data in data.items():
            signals = asset_data.get("signals", {})
            ml_score = signals.get("ml_score", 0.5)

            pos = broker.get_position(asset)
            if pos is None and ml_score > 0.6:
                price = asset_data["close"]
                qty = 1000 / price
                broker.submit_order(asset, qty)


class BracketOrderStrategy(Strategy):
    """Test bracket orders with stop loss and take profit."""

    def __init__(self, asset: str):
        self.asset = asset
        self.entered = False

    def on_data(self, timestamp, data, context, broker):
        if not self.entered and self.asset in data:
            price = data[self.asset]["close"]
            # Entry with 5% take profit and 2% stop loss
            broker.submit_bracket(
                self.asset,
                quantity=100,
                take_profit=price * 1.05,
                stop_loss=price * 0.98,
            )
            self.entered = True


class BuyThenSellStrategy(Strategy):
    """Buy on first bar and fully exit on second bar."""

    def __init__(self, asset: str, quantity: float):
        self.asset = asset
        self.quantity = quantity
        self.bar_count = 0

    def on_data(self, timestamp, data, context, broker):
        if self.asset not in data:
            return
        self.bar_count += 1
        if self.bar_count == 1:
            broker.submit_order(self.asset, self.quantity)
        elif self.bar_count == 2:
            broker.close_position(self.asset)


class VolatilityAdjustedStopStrategy(Strategy):
    """Test updating stop orders based on volatility."""

    def __init__(self, asset: str):
        self.asset = asset
        self.stop_order_id = None
        self.entry_order_id = None

    def on_data(self, timestamp, data, context, broker):
        if self.asset not in data:
            return

        asset_data = data[self.asset]
        price = asset_data["close"]
        vol = asset_data.get("signals", {}).get("volatility", price * 0.02)

        pos = broker.get_position(self.asset)

        if pos is None and self.entry_order_id is None:
            # Enter position
            entry = broker.submit_order(self.asset, 100)
            self.entry_order_id = entry.order_id

            # Set initial stop at 2x volatility below entry
            stop = broker.submit_order(
                self.asset, 100, OrderSide.SELL, OrderType.STOP, stop_price=price - 2 * vol
            )
            self.stop_order_id = stop.order_id

        elif pos is not None and self.stop_order_id:
            # Update stop based on current volatility
            new_stop = price - 2 * vol
            broker.update_order(self.stop_order_id, stop_price=new_stop)


# === Tests ===


class TestDataFeed:
    """Test DataFeed functionality."""

    def test_basic_iteration(self):
        prices = generate_prices(["AAPL", "GOOG"], datetime(2024, 1, 1), 10)
        feed = DataFeed(prices_df=prices)

        count = 0
        for ts, data, _context in feed:
            assert isinstance(ts, datetime)
            assert "AAPL" in data
            assert "GOOG" in data
            assert data["AAPL"]["close"] is not None
            count += 1

        assert count == 10

    def test_with_signals(self):
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 5)
        signals = generate_signals(["AAPL"], datetime(2024, 1, 1), 5, ["ml_score"])
        feed = DataFeed(prices_df=prices, signals_df=signals)

        for _ts, data, _context in feed:
            assert "ml_score" in data["AAPL"]["signals"]

    def test_with_context(self):
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 5)
        context_df = generate_context(datetime(2024, 1, 1), 5)
        feed = DataFeed(prices_df=prices, context_df=context_df)

        for _ts, _data, context in feed:
            assert "vix" in context
            assert "spy" in context

    def test_unsorted_inputs_align_prices_signals_and_context(self):
        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 1, 2)
        prices = pl.DataFrame(
            {
                "timestamp": [ts2, ts1, ts2, ts1],
                "asset": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "open": [102.0, 100.0, 202.0, 200.0],
                "high": [103.0, 101.0, 203.0, 201.0],
                "low": [101.0, 99.0, 201.0, 199.0],
                "close": [102.5, 100.5, 202.5, 200.5],
                "volume": [1000.0, 900.0, 1100.0, 950.0],
            }
        )
        signals = pl.DataFrame(
            {
                "timestamp": [ts2, ts1, ts2, ts1],
                "asset": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "ml_score": [0.7, 0.5, 0.8, 0.4],
            }
        )
        context_df = pl.DataFrame(
            {
                "timestamp": [ts2, ts1],
                "vix": [18.0, 16.0],
            }
        )

        feed = DataFeed(prices_df=prices, signals_df=signals, context_df=context_df)
        rows = list(feed)

        assert [ts for ts, *_ in rows] == [ts1, ts2]

        first_ts, first_data, first_context = rows[0]
        assert first_ts == ts1
        assert first_data["AAPL"]["close"] == 100.5
        assert first_data["MSFT"]["close"] == 200.5
        assert first_data["AAPL"]["signals"]["ml_score"] == 0.5
        assert first_data["MSFT"]["signals"]["ml_score"] == 0.4
        assert first_context["vix"] == 16.0

        second_ts, second_data, second_context = rows[1]
        assert second_ts == ts2
        assert second_data["AAPL"]["signals"]["ml_score"] == 0.7
        assert second_data["MSFT"]["signals"]["ml_score"] == 0.8
        assert second_context["vix"] == 18.0


class TestBroker:
    """Test Broker functionality."""

    def test_submit_order(self):
        broker = Broker(initial_cash=100000)
        broker._current_time = datetime(2024, 1, 1)
        broker._current_prices = {"AAPL": 150.0}
        broker._current_volumes = {"AAPL": 1000000}
        broker._current_signals = {}

        order = broker.submit_order("AAPL", 100)
        assert order.order_id == "ORD-1"
        assert order.quantity == 100
        assert len(broker.pending_orders) == 1

    def test_market_order_fill(self):
        broker = Broker(initial_cash=100000)
        broker._current_time = datetime(2024, 1, 1)
        broker._current_prices = {"AAPL": 150.0}
        broker._current_volumes = {"AAPL": 1000000}
        broker._current_signals = {}

        broker.submit_order("AAPL", 100)
        broker._process_orders()

        assert len(broker.fills) == 1
        assert broker.fills[0].price == 150.0
        assert broker.get_position("AAPL").quantity == 100
        assert broker.cash == 100000 - 100 * 150

    def test_commission_model(self):
        commission = PercentageCommission(rate=0.001)
        broker = Broker(initial_cash=100000, commission_model=commission)
        broker._current_time = datetime(2024, 1, 1)
        broker._current_prices = {"AAPL": 100.0}
        broker._current_volumes = {"AAPL": 1000000}
        broker._current_signals = {}

        broker.submit_order("AAPL", 100)
        broker._process_orders()

        # Commission = 100 * 100 * 0.001 = 10
        assert broker.fills[0].commission == 10.0
        assert broker.cash == 100000 - 10000 - 10

    def test_slippage_model(self):
        slippage = VolumeShareSlippage(impact_factor=0.5)
        broker = Broker(initial_cash=100000, slippage_model=slippage)
        broker._current_time = datetime(2024, 1, 1)
        broker._current_prices = {"AAPL": 100.0}
        broker._current_volumes = {"AAPL": 1000}  # Small volume
        broker._current_signals = {}

        broker.submit_order("AAPL", 100)  # 10% of volume
        broker._process_orders()

        # Should have slippage
        assert broker.fills[0].slippage > 0
        assert broker.fills[0].price > 100.0

    def test_update_order(self):
        broker = Broker(initial_cash=100000)
        broker._current_time = datetime(2024, 1, 1)

        order = broker.submit_order("AAPL", 100, OrderSide.SELL, OrderType.STOP, stop_price=95.0)

        # Update stop price
        result = broker.update_order(order.order_id, stop_price=90.0)
        assert result is True
        assert order.stop_price == 90.0

    def test_bracket_order(self):
        broker = Broker(initial_cash=100000)
        broker._current_time = datetime(2024, 1, 1)
        broker._current_prices = {"AAPL": 100.0}
        broker._current_volumes = {"AAPL": 1000000}
        broker._current_signals = {}

        entry, tp, sl = broker.submit_bracket("AAPL", 100, take_profit=110, stop_loss=90)

        assert entry.order_type == OrderType.MARKET
        assert tp.order_type == OrderType.LIMIT
        assert tp.limit_price == 110
        assert sl.order_type == OrderType.STOP
        assert sl.stop_price == 90
        assert tp.parent_id == entry.order_id
        assert sl.parent_id == entry.order_id

    def test_position_bars_held(self):
        broker = Broker(initial_cash=100000)
        broker._current_time = datetime(2024, 1, 1)
        broker._current_prices = {"AAPL": 100.0}
        broker._current_volumes = {"AAPL": 1000000}
        broker._current_signals = {}

        broker.submit_order("AAPL", 100)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos.bars_held == 0

        # Simulate next bar (prices, opens, highs, lows, volumes, signals)
        broker._update_time(
            datetime(2024, 1, 2),
            {"AAPL": 101.0},  # prices (close)
            {"AAPL": 101.0},  # opens
            {"AAPL": 102.0},  # highs
            {"AAPL": 100.0},  # lows
            {"AAPL": 1000000},  # volumes
            {},  # signals
        )
        assert pos.bars_held == 1

        broker._update_time(
            datetime(2024, 1, 3),
            {"AAPL": 102.0},  # prices (close)
            {"AAPL": 102.0},  # opens
            {"AAPL": 103.0},  # highs
            {"AAPL": 101.0},  # lows
            {"AAPL": 1000000},  # volumes
            {},  # signals
        )
        assert pos.bars_held == 2

    def test_order_rejection_insufficient_cash(self):
        """Test that orders are rejected when insufficient cash (Phase 2 validation)."""
        broker = Broker(initial_cash=10000, allow_short_selling=False)  # Only $10k
        broker._current_time = datetime(2024, 1, 1)
        broker._current_prices = {"AAPL": 150.0}
        broker._current_volumes = {"AAPL": 1000000}
        broker._current_signals = {}

        # Try to buy 100 shares @ $150 = $15,000 (more than we have)
        order = broker.submit_order("AAPL", 100)
        assert len(broker.pending_orders) == 1

        # Process orders - should reject due to insufficient cash
        broker._process_orders()

        # Order should be rejected
        from ml4t.backtest import OrderStatus

        assert order.status == OrderStatus.REJECTED
        assert len(broker.pending_orders) == 0  # Removed from pending
        assert len(broker.fills) == 0  # No fill
        assert broker.get_position("AAPL") is None  # No position
        assert broker.cash == 10000  # Cash unchanged


class TestEngine:
    """Test Engine functionality."""

    def test_buy_and_hold(self):
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 20, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        engine = Engine(feed, strategy)
        results = engine.run()

        assert results.metrics["num_trades"] == 0  # Still holding
        assert len(engine.broker.positions) == 1
        assert engine.broker.get_position("AAPL").quantity == 100

    def test_signal_based_strategy(self):
        prices = generate_prices(["AAPL", "GOOG"], datetime(2024, 1, 1), 30)
        signals = generate_signals(["AAPL", "GOOG"], datetime(2024, 1, 1), 30)
        feed = DataFeed(prices_df=prices, signals_df=signals)

        strategy = SignalBasedStrategy(threshold_buy=0.6, threshold_sell=0.4)
        engine = Engine(feed, strategy)
        results = engine.run()

        # Should have some trades
        assert len(results.equity_curve) == 30
        assert results.metrics["final_value"] > 0

    def test_vix_filter_strategy(self):
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 20)
        signals = generate_signals(["AAPL"], datetime(2024, 1, 1), 20)
        context = generate_context(datetime(2024, 1, 1), 20)
        feed = DataFeed(prices_df=prices, signals_df=signals, context_df=context)

        strategy = VIXFilterStrategy(vix_threshold=18)
        engine = Engine(feed, strategy)
        results = engine.run()

        assert len(results.equity_curve) == 20

    def test_with_commission(self):
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(
            commission_type=CommissionType.PER_SHARE,
            commission_per_share=0.01,
            commission_minimum=1.0,
        )
        engine = Engine(feed, strategy, config)
        results = engine.run()

        assert results.metrics["total_commission"] >= 1.0

    def test_convenience_function(self):
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(
            initial_cash=50000,
            commission_type=CommissionType.NONE,
            slippage_type=SlippageType.NONE,
        )
        results = run_backtest(
            prices=prices,
            strategy=strategy,
            config=config,
        )

        assert results.metrics["initial_cash"] == 50000
        assert len(results.equity_curve) == 10

    def test_reports_activity_and_portfolio_state_metrics(self):
        prices = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "asset": ["AAPL", "AAPL", "AAPL"],
                "open": [100.0, 110.0, 110.0],
                "high": [100.0, 110.0, 110.0],
                "low": [100.0, 110.0, 110.0],
                "close": [100.0, 110.0, 110.0],
                "volume": [1000.0, 1000.0, 1000.0],
            }
        )
        engine = Engine(
            DataFeed(prices_df=prices),
            BuyThenSellStrategy("AAPL", quantity=10),
            BacktestConfig(
                initial_cash=100000.0,
                execution_mode=ExecutionMode.SAME_BAR,
                commission_type=CommissionType.NONE,
                slippage_type=SlippageType.NONE,
            ),
        )

        results = engine.run()
        portfolio_state = results.to_portfolio_state_dataframe()

        assert results.metrics["num_fills"] == 2
        assert results.metrics["num_rebalance_events"] == 2
        assert results.metrics["unique_symbols_traded"] == 1
        assert results.metrics["total_filled_notional"] == pytest.approx(2100.0)
        assert results.metrics["avg_turnover"] == pytest.approx(
            ((1000.0 / 100000.0) + (1100.0 / 100100.0)) / 3.0
        )
        assert results.metrics["max_turnover"] == pytest.approx(1100.0 / 100100.0)
        assert results.metrics["avg_open_positions"] == pytest.approx(1.0 / 3.0)
        assert results.metrics["max_open_positions"] == 1

        assert portfolio_state.columns == [
            "timestamp",
            "equity",
            "cash",
            "gross_exposure",
            "net_exposure",
            "open_positions",
        ]
        assert portfolio_state["open_positions"].to_list() == [1, 0, 0]
        assert portfolio_state["gross_exposure"].to_list() == [1000.0, 0.0, 0.0]

    def test_activity_metrics_prefer_explicit_rebalance_ids(self):
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "asset": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [100.0, 101.0],
                "low": [100.0, 101.0],
                "close": [100.0, 101.0],
                "volume": [1000.0, 1000.0],
            }
        )
        engine = Engine(
            DataFeed(prices_df=prices),
            BuyAndHoldStrategy("AAPL"),
            BacktestConfig(
                initial_cash=100000.0,
                execution_mode=ExecutionMode.SAME_BAR,
                commission_type=CommissionType.NONE,
                slippage_type=SlippageType.NONE,
            ),
        )

        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 1, 2)
        engine.portfolio_state = [
            (ts1, 100000.0, 100000.0, 0.0, 0.0, 0),
            (ts2, 100100.0, 99100.0, 1000.0, 1000.0, 1),
        ]
        engine.broker.fills = [
            Fill(
                order_id="ORD-1",
                rebalance_id="rebalance-1",
                asset="AAPL",
                side=OrderSide.BUY,
                quantity=5.0,
                price=100.0,
                timestamp=ts1,
            ),
            Fill(
                order_id="ORD-2",
                rebalance_id="rebalance-1",
                asset="MSFT",
                side=OrderSide.BUY,
                quantity=2.0,
                price=200.0,
                timestamp=ts2,
            ),
        ]

        metrics = engine._build_activity_metrics()

        assert metrics["num_rebalance_events"] == 1


class TestTradeRecording:
    """Test trade recording with signals."""

    def test_trade_has_signals(self):
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        signals = generate_signals(["AAPL"], datetime(2024, 1, 1), 10, ["ml_score"])

        class QuickTrade(Strategy):
            def __init__(self):
                self.state = "init"

            def on_data(self, timestamp, data, context, broker):
                if self.state == "init":
                    broker.submit_order("AAPL", 100)
                    self.state = "holding"
                elif self.state == "holding":
                    broker.close_position("AAPL")
                    self.state = "done"

        feed = DataFeed(prices_df=prices, signals_df=signals)
        engine = Engine(feed, QuickTrade())
        results = engine.run()

        assert len(results.trades) == 1
        trade = results.trades[0]
        assert trade.bars_held >= 1
        # Note: entry_signals/exit_signals are not Trade fields
        # Signals are available in context during on_data, not stored on Trade
        assert trade.symbol == "AAPL"
        assert trade.pnl != 0  # Should have some P&L


class TestMultiAsset:
    """Test multi-asset scenarios."""

    def test_multiple_positions(self):
        assets = ["AAPL", "GOOG", "MSFT", "AMZN"]
        prices = generate_prices(assets, datetime(2024, 1, 1), 20)
        signals = generate_signals(assets, datetime(2024, 1, 1), 20)

        class MultiAssetStrategy(Strategy):
            def on_data(self, timestamp, data, context, broker):
                for asset in data:
                    if broker.get_position(asset) is None:
                        broker.submit_order(asset, 10)

        feed = DataFeed(prices_df=prices, signals_df=signals)
        engine = Engine(feed, MultiAssetStrategy())
        engine.run()

        assert len(engine.broker.positions) == 4
        for asset in assets:
            assert engine.broker.get_position(asset) is not None


class TestEngineFromConfig:
    """Tests for Engine.from_config() factory method."""

    def test_from_config_percentage_commission(self):
        """Test from_config with percentage commission."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(
            commission_type=CommissionType.PERCENTAGE,
            commission_rate=0.001,  # 0.1%
        )
        engine = Engine.from_config(feed, strategy, config)
        results = engine.run()

        # Verify commission was applied
        assert results.metrics["total_commission"] > 0

    def test_from_config_per_share_commission(self):
        """Test from_config with per-share commission."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(
            commission_type=CommissionType.PER_SHARE,
            commission_per_share=0.01,
            commission_minimum=1.0,
        )
        engine = Engine.from_config(feed, strategy, config)
        results = engine.run()

        # Should have minimum commission applied
        assert results.metrics["total_commission"] >= 1.0

    def test_from_config_no_commission(self):
        """Test from_config with no commission."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(commission_type=CommissionType.NONE)
        engine = Engine.from_config(feed, strategy, config)
        results = engine.run()

        assert results.metrics["total_commission"] == 0.0

    def test_from_config_percentage_slippage(self):
        """Test from_config with percentage slippage."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(
            slippage_type=SlippageType.PERCENTAGE,
            slippage_rate=0.001,  # 0.1%
        )
        engine = Engine.from_config(feed, strategy, config)
        results = engine.run()

        assert results.metrics["total_slippage"] > 0

    def test_from_config_fixed_slippage(self):
        """Test from_config with fixed slippage."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(
            slippage_type=SlippageType.FIXED,
            slippage_fixed=0.05,  # $0.05 per share
        )
        engine = Engine.from_config(feed, strategy, config)
        results = engine.run()

        assert results.metrics["total_slippage"] == pytest.approx(5.0)
        assert results.trades[0].entry_slippage == pytest.approx(0.05)
        assert results.trades[0].exit_slippage == 0.0

    def test_from_config_execution_mode_same_bar(self):
        """Test from_config with SAME_BAR execution mode."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(execution_mode=ExecutionMode.SAME_BAR)
        engine = Engine.from_config(feed, strategy, config)

        assert engine.execution_mode == ExecutionMode.SAME_BAR

    def test_from_config_execution_mode_next_bar(self):
        """Test from_config with NEXT_BAR execution mode."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR)
        engine = Engine.from_config(feed, strategy, config)

        assert engine.execution_mode == ExecutionMode.NEXT_BAR

    def test_from_config_margin_account(self):
        """Test from_config with margin account."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(
            allow_short_selling=True,
            allow_leverage=True,
            initial_margin=0.4,
        )
        engine = Engine.from_config(feed, strategy, config)
        results = engine.run()

        assert results.metrics["final_value"] > 0


class TestNextBarExecutionMode:
    """Tests for NEXT_BAR execution mode."""

    def test_next_bar_mode_order_delayed(self):
        """Test that orders submitted in NEXT_BAR mode fill next bar."""

        class TrackingStrategy(Strategy):
            def __init__(self):
                self.bar_count = 0
                self.order_bar = None
                self.fill_bar = None

            def on_data(self, timestamp, data, context, broker):
                self.bar_count += 1
                if self.bar_count == 2:
                    broker.submit_order("AAPL", 100)
                    self.order_bar = self.bar_count

                # Check if position exists (order filled)
                pos = broker.get_position("AAPL")
                if pos is not None and self.fill_bar is None:
                    self.fill_bar = self.bar_count

        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10, {"AAPL": 100})
        feed = DataFeed(prices_df=prices)
        strategy = TrackingStrategy()

        config = BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR)
        engine = Engine(feed, strategy, config)
        engine.run()

        # Order placed on bar 2, should fill on bar 3
        assert strategy.order_bar == 2
        assert strategy.fill_bar == 3


class TestRunBacktestWithConfig:
    """Tests for run_backtest() convenience function with config."""

    def test_run_backtest_with_config_object(self):
        """Test run_backtest with BacktestConfig object."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10)
        strategy = BuyAndHoldStrategy("AAPL")

        config = BacktestConfig(
            initial_cash=50000,
            commission_type=CommissionType.NONE,
            slippage_type=SlippageType.NONE,  # Must disable slippage too
        )
        results = run_backtest(prices=prices, strategy=strategy, config=config)

        assert results.metrics["initial_cash"] == 50000

    def test_run_backtest_with_string_preset(self):
        """Test run_backtest with string preset name."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10)
        strategy = BuyAndHoldStrategy("AAPL")

        results = run_backtest(prices=prices, strategy=strategy, config="default")

        assert results.equity_curve is not None
        assert len(results.equity_curve) == 10

    def test_run_backtest_preserves_raw_predictions_dataframe(self):
        """Test raw input predictions are available on the result surface."""
        prices = generate_prices(["AAPL"], datetime(2024, 1, 1), 10)
        signals = generate_signals(["AAPL"], datetime(2024, 1, 1), 10, ["ml_score"])
        strategy = BuyAndHoldStrategy("AAPL")

        result = run_backtest(prices=prices, signals=signals, strategy=strategy, config="default")

        assert result.predictions is not None
        assert result.to_predictions_dataframe().equals(signals)

    def test_run_backtest_uses_feed_spec_for_runtime_config(self):
        """Feed metadata should populate runtime config when explicit config is unset."""

        class PrepareTrackingStrategy(Strategy):
            def __init__(self):
                self.seen_config = None

            def on_prepare(self, broker, timestamps, config=None):
                self.seen_config = config

            def on_data(self, timestamp, data, context, broker):
                return None

        prices = pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 9, 31)],
                "ticker": ["AAPL", "AAPL"],
                "open_px": [100.0, 101.0],
                "high_px": [101.0, 102.0],
                "low_px": [99.0, 100.0],
                "close_px": [100.5, 101.5],
                "vol": [1000.0, 1100.0],
            }
        )
        strategy = PrepareTrackingStrategy()

        result = run_backtest(
            prices=prices,
            strategy=strategy,
            config=BacktestConfig(),
            feed_spec=FeedSpec(
                timestamp_col="ts",
                entity_col="ticker",
                open_col="open_px",
                high_col="high_px",
                low_col="low_px",
                close_col="close_px",
                volume_col="vol",
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency="minute",
            ),
        )

        assert strategy.seen_config is not None
        assert strategy.seen_config.calendar == "NYSE"
        assert strategy.seen_config.timezone == "America/New_York"
        assert strategy.seen_config.data_frequency == DataFrequency.MINUTE_1
        assert result.config is not None
        assert result.config.calendar == "NYSE"

    def test_run_backtest_preserves_explicit_runtime_config_over_feed_spec(self):
        """Explicit runtime config should not be overwritten by feed metadata."""

        class PrepareTrackingStrategy(Strategy):
            def __init__(self):
                self.seen_config = None

            def on_prepare(self, broker, timestamps, config=None):
                self.seen_config = config

            def on_data(self, timestamp, data, context, broker):
                return None

        prices = pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 9, 31)],
                "ticker": ["AAPL", "AAPL"],
                "open_px": [100.0, 101.0],
                "high_px": [101.0, 102.0],
                "low_px": [99.0, 100.0],
                "close_px": [100.5, 101.5],
                "vol": [1000.0, 1100.0],
            }
        )
        strategy = PrepareTrackingStrategy()

        result = run_backtest(
            prices=prices,
            strategy=strategy,
            config=BacktestConfig(timezone="UTC", data_frequency=DataFrequency.DAILY),
            feed_spec=FeedSpec(
                timestamp_col="ts",
                entity_col="ticker",
                open_col="open_px",
                high_col="high_px",
                low_col="low_px",
                close_col="close_px",
                volume_col="vol",
                calendar="NYSE",
                timezone="America/New_York",
                data_frequency="minute",
            ),
        )

        assert strategy.seen_config is not None
        assert strategy.seen_config.calendar == "NYSE"
        assert strategy.seen_config.timezone == "UTC"
        assert strategy.seen_config.data_frequency == DataFrequency.DAILY
        assert result.config is not None
        assert result.config.timezone == "UTC"
        assert result.config.data_frequency == DataFrequency.DAILY


class TestEmptyDataFeed:
    """Tests for edge cases with empty or minimal data."""

    def test_empty_data_returns_empty_results(self):
        """Test engine with no data bars."""
        # Create empty price DataFrame with correct schema
        empty_prices = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "asset": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

        feed = DataFeed(prices_df=empty_prices)
        strategy = BuyAndHoldStrategy("AAPL")
        engine = Engine(feed, strategy)
        results = engine.run()

        # Should return empty or minimal results without error
        assert results.metrics.get("num_trades", 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
