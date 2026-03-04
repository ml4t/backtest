"""Tests for trade cost decomposition and short P&L correctness.

Verifies:
- Direction-aware pnl_percent for short trades
- MFE/MAE tracking for short positions
- Trade computed properties: gross_pnl, net_return, total_slippage_cost, cost_drag
- Decomposition identity: gross_pnl - fees == pnl (when slippage is baked into prices)
- Futures multiplier propagation
- Parquet roundtrip of new fields
- Backward compatibility with old Parquet files
"""

from datetime import datetime

import pytest

from ml4t.backtest.types import Position, Trade

# === Position.pnl_percent short fix ===


class TestPositionPnlPercentShort:
    def test_long_positive_return(self):
        """Long position, price goes up → positive return."""
        pos = Position("AAPL", 100.0, 100.0, datetime(2024, 1, 1))
        assert pos.pnl_percent(110.0) == pytest.approx(0.10)

    def test_long_negative_return(self):
        """Long position, price goes down → negative return."""
        pos = Position("AAPL", 100.0, 100.0, datetime(2024, 1, 1))
        assert pos.pnl_percent(90.0) == pytest.approx(-0.10)

    def test_short_profitable(self):
        """Short position, price goes down → positive return."""
        pos = Position("AAPL", -100.0, 100.0, datetime(2024, 1, 1))
        assert pos.pnl_percent(90.0) == pytest.approx(0.10)

    def test_short_loss(self):
        """Short position, price goes up → negative return."""
        pos = Position("AAPL", -100.0, 100.0, datetime(2024, 1, 1))
        assert pos.pnl_percent(110.0) == pytest.approx(-0.10)

    def test_short_at_entry(self):
        """Short position, price unchanged → zero return."""
        pos = Position("AAPL", -100.0, 100.0, datetime(2024, 1, 1))
        assert pos.pnl_percent(100.0) == pytest.approx(0.0)


# === Trade pnl_percent sign ===


class TestTradePnlPercentSign:
    def test_short_profitable_trade(self):
        """Short at 100, exit at 90 → pnl_percent should be +0.10."""
        trade = Trade(
            symbol="TEST",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=100.0,
            exit_price=90.0,
            quantity=-100.0,
            pnl=1000.0,
            pnl_percent=0.10,  # Direction-aware: positive for profitable short
            bars_held=1,
        )
        assert trade.pnl_percent > 0  # Profitable short = positive pnl_percent

    def test_short_losing_trade(self):
        """Short at 100, exit at 110 → pnl_percent should be -0.10."""
        trade = Trade(
            symbol="TEST",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=100.0,
            exit_price=110.0,
            quantity=-100.0,
            pnl=-1000.0,
            pnl_percent=-0.10,  # Direction-aware: negative for losing short
            bars_held=1,
        )
        assert trade.pnl_percent < 0  # Losing short = negative pnl_percent


# === MFE/MAE tracking for shorts ===


class TestShortMFEMAE:
    def test_short_mfe_positive_when_price_drops(self):
        """For short positions, MFE should be positive when price drops (favorable)."""
        pos = Position("TEST", -100.0, 100.0, datetime(2024, 1, 1))
        # Price drops to 90 → favorable for short → MFE should be positive
        # use_low_for_lwm=True uses bar_low for MFE source (OHLC mode)
        pos.update_water_marks(95.0, bar_high=100.0, bar_low=90.0, use_low_for_lwm=True)
        assert pos.max_favorable_excursion > 0, (
            f"Short MFE should be positive when price drops, got {pos.max_favorable_excursion}"
        )
        assert pos.max_favorable_excursion == pytest.approx(0.10)  # (100-90)/100

    def test_short_mae_negative_when_price_rises(self):
        """For short positions, MAE should be negative when price rises (adverse)."""
        pos = Position("TEST", -100.0, 100.0, datetime(2024, 1, 1))
        # Price rises to 110 → adverse for short → MAE should be negative
        # use_high_for_hwm=True uses bar_high for MAE source (OHLC mode)
        pos.update_water_marks(105.0, bar_high=110.0, bar_low=100.0, use_high_for_hwm=True)
        assert pos.max_adverse_excursion < 0, (
            f"Short MAE should be negative when price rises, got {pos.max_adverse_excursion}"
        )
        assert pos.max_adverse_excursion == pytest.approx(-0.10)  # -(110-100)/100

    def test_short_mfe_not_always_zero(self):
        """Regression: MFE was always 0 for shorts due to wrong pnl_percent sign."""
        pos = Position("TEST", -100.0, 100.0, datetime(2024, 1, 1))
        pos.update_water_marks(95.0, bar_high=100.0, bar_low=90.0)
        pos.update_water_marks(92.0, bar_high=96.0, bar_low=88.0)
        assert pos.max_favorable_excursion != 0.0, "Short MFE should not be stuck at zero"

    def test_short_mae_not_always_zero(self):
        """Regression: MAE was always 0 for shorts due to wrong pnl_percent sign."""
        pos = Position("TEST", -100.0, 100.0, datetime(2024, 1, 1))
        pos.update_water_marks(105.0, bar_high=110.0, bar_low=102.0)
        assert pos.max_adverse_excursion != 0.0, "Short MAE should not be stuck at zero"


# === Trade computed properties ===


class TestTradeComputedProperties:
    @pytest.fixture
    def long_trade(self):
        return Trade(
            symbol="AAPL",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 5),
            entry_price=100.0,
            exit_price=110.0,
            quantity=100.0,
            pnl=980.0,  # 1000 gross - 20 fees
            pnl_percent=0.10,
            bars_held=4,
            fees=20.0,
            slippage=0.05,
            entry_slippage=0.03,
            multiplier=1.0,
        )

    @pytest.fixture
    def short_trade(self):
        return Trade(
            symbol="TSLA",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 5),
            entry_price=100.0,
            exit_price=90.0,
            quantity=-100.0,
            pnl=980.0,  # 1000 gross - 20 fees
            pnl_percent=0.10,
            bars_held=4,
            fees=20.0,
            slippage=0.05,
            entry_slippage=0.03,
            multiplier=1.0,
        )

    @pytest.fixture
    def futures_trade(self):
        return Trade(
            symbol="ES",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 5),
            entry_price=5000.0,
            exit_price=5010.0,
            quantity=2.0,
            pnl=991.0,  # (10 * 2 * 50) - 9 fees = 1000 - 9
            pnl_percent=0.002,  # 10/5000
            bars_held=4,
            fees=9.0,
            slippage=0.25,
            entry_slippage=0.25,
            multiplier=50.0,
        )

    def test_gross_pnl_long(self, long_trade):
        assert long_trade.gross_pnl == pytest.approx(1000.0)  # (110-100)*100*1

    def test_gross_pnl_short(self, short_trade):
        # (90-100) * (-100) * 1 = (-10)*(-100) = 1000
        assert short_trade.gross_pnl == pytest.approx(1000.0)

    def test_gross_pnl_futures(self, futures_trade):
        # (5010-5000) * 2 * 50 = 1000
        assert futures_trade.gross_pnl == pytest.approx(1000.0)

    def test_net_pnl_alias(self, long_trade):
        assert long_trade.net_pnl == long_trade.pnl

    def test_gross_return_alias(self, long_trade):
        assert long_trade.gross_return == long_trade.pnl_percent

    def test_net_return_long(self, long_trade):
        # 980 / (100 * 100 * 1) = 0.098
        assert long_trade.net_return == pytest.approx(0.098)

    def test_net_return_short(self, short_trade):
        # 980 / (100 * 100 * 1) = 0.098
        assert short_trade.net_return == pytest.approx(0.098)

    def test_net_return_futures(self, futures_trade):
        # 991 / (5000 * 2 * 50) = 991/500000 = 0.001982
        assert futures_trade.net_return == pytest.approx(991.0 / 500000.0)

    def test_total_slippage_cost_long(self, long_trade):
        # (0.03 + 0.05) * 100 * 1 = 8.0
        assert long_trade.total_slippage_cost == pytest.approx(8.0)

    def test_total_slippage_cost_short(self, short_trade):
        # (0.03 + 0.05) * 100 * 1 = 8.0 (uses abs(quantity))
        assert short_trade.total_slippage_cost == pytest.approx(8.0)

    def test_total_slippage_cost_futures(self, futures_trade):
        # (0.25 + 0.25) * 2 * 50 = 50.0
        assert futures_trade.total_slippage_cost == pytest.approx(50.0)

    def test_cost_drag_long(self, long_trade):
        # (20 + 8) / (100 * 100 * 1) = 28/10000 = 0.0028
        assert long_trade.cost_drag == pytest.approx(28.0 / 10000.0)

    def test_cost_drag_futures(self, futures_trade):
        # (9 + 50) / (5000 * 2 * 50) = 59/500000
        assert futures_trade.cost_drag == pytest.approx(59.0 / 500000.0)

    def test_cost_drag_zero_notional(self):
        trade = Trade(
            symbol="X",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=0.0,
            exit_price=0.0,
            quantity=0.0,
            pnl=0.0,
            pnl_percent=0.0,
            bars_held=1,
        )
        assert trade.cost_drag == 0.0

    def test_net_return_zero_notional(self):
        trade = Trade(
            symbol="X",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=0.0,
            exit_price=0.0,
            quantity=0.0,
            pnl=0.0,
            pnl_percent=0.0,
            bars_held=1,
        )
        assert trade.net_return == 0.0


# === Decomposition identity ===


class TestDecompositionIdentity:
    def test_gross_minus_fees_equals_pnl(self):
        """When slippage is baked into fill prices, gross_pnl - fees == pnl."""
        trade = Trade(
            symbol="TEST",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=100.0,
            exit_price=110.0,
            quantity=100.0,
            pnl=980.0,  # 1000 - 20 fees
            pnl_percent=0.10,
            bars_held=1,
            fees=20.0,
        )
        assert trade.gross_pnl - trade.fees == pytest.approx(trade.pnl)

    def test_gross_minus_fees_short(self):
        """Decomposition identity holds for short trades."""
        trade = Trade(
            symbol="TEST",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=100.0,
            exit_price=90.0,
            quantity=-100.0,
            pnl=985.0,  # 1000 - 15 fees
            pnl_percent=0.10,
            bars_held=1,
            fees=15.0,
        )
        assert trade.gross_pnl - trade.fees == pytest.approx(trade.pnl)

    def test_gross_minus_fees_futures(self):
        """Decomposition identity holds for futures."""
        trade = Trade(
            symbol="ES",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=5000.0,
            exit_price=5010.0,
            quantity=2.0,
            pnl=991.0,  # (10*2*50) - 9 = 1000-9
            pnl_percent=0.002,
            bars_held=1,
            fees=9.0,
            multiplier=50.0,
        )
        assert trade.gross_pnl - trade.fees == pytest.approx(trade.pnl)


# === Entry slippage field ===


class TestEntrySlippage:
    def test_position_entry_slippage_default(self):
        pos = Position("TEST", 100.0, 100.0, datetime(2024, 1, 1))
        assert pos.entry_slippage == 0.0

    def test_position_entry_slippage_set(self):
        pos = Position("TEST", 100.0, 100.0, datetime(2024, 1, 1), entry_slippage=0.05)
        assert pos.entry_slippage == 0.05

    def test_trade_entry_slippage_default(self):
        trade = Trade(
            symbol="TEST",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=100.0,
            exit_price=110.0,
            quantity=100.0,
            pnl=1000.0,
            pnl_percent=0.10,
            bars_held=1,
        )
        assert trade.entry_slippage == 0.0

    def test_trade_multiplier_default(self):
        trade = Trade(
            symbol="TEST",
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 2),
            entry_price=100.0,
            exit_price=110.0,
            quantity=100.0,
            pnl=1000.0,
            pnl_percent=0.10,
            bars_held=1,
        )
        assert trade.multiplier == 1.0


# === Parquet roundtrip ===


class TestParquetRoundtrip:
    def test_new_fields_survive_roundtrip(self, tmp_path):
        """entry_slippage and multiplier survive write/read."""
        from ml4t.backtest.result import BacktestResult

        trades = [
            Trade(
                symbol="ES",
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 2),
                entry_price=5000.0,
                exit_price=5010.0,
                quantity=2.0,
                pnl=991.0,
                pnl_percent=0.002,
                bars_held=1,
                fees=9.0,
                slippage=0.25,
                entry_slippage=0.25,
                multiplier=50.0,
            )
        ]
        result = BacktestResult(
            trades=trades,
            equity_curve=[(datetime(2024, 1, 1), 100000.0)],
            fills=[],
            metrics={"initial_cash": 100000.0},
        )

        result.to_parquet(tmp_path / "test_result")
        loaded = BacktestResult.from_parquet(tmp_path / "test_result")

        assert len(loaded.trades) == 1
        t = loaded.trades[0]
        assert t.entry_slippage == pytest.approx(0.25)
        assert t.multiplier == pytest.approx(50.0)
        assert t.gross_pnl == pytest.approx(1000.0)

    def test_backward_compat_missing_fields(self, tmp_path):
        """Old Parquet files without entry_slippage/multiplier load with defaults."""
        import polars as pl

        # Write a Parquet without the new columns (simulates old format)
        old_df = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "entry_time": [datetime(2024, 1, 1)],
                "exit_time": [datetime(2024, 1, 2)],
                "entry_price": [100.0],
                "exit_price": [110.0],
                "quantity": [100.0],
                "direction": ["long"],
                "pnl": [1000.0],
                "pnl_percent": [0.10],
                "bars_held": [1],
                "fees": [0.0],
                "slippage": [0.0],
                "mfe": [0.12],
                "mae": [-0.03],
                "exit_reason": ["signal"],
                "status": ["closed"],
            }
        )

        result_dir = tmp_path / "old_result"
        result_dir.mkdir()
        old_df.write_parquet(result_dir / "trades.parquet")

        from ml4t.backtest.result import BacktestResult

        loaded = BacktestResult.from_parquet(result_dir)
        assert len(loaded.trades) == 1
        t = loaded.trades[0]
        assert t.entry_slippage == 0.0  # Default
        assert t.multiplier == 1.0  # Default
        assert t.gross_pnl == pytest.approx(1000.0)


# === Integration: actual backtest with shorts ===


class TestShortBacktestIntegration:
    def test_short_trade_pnl_percent_positive_for_profit(self):
        """Full engine run with short trade produces correct pnl_percent sign."""
        import polars as pl

        from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

        class ShortStrategy(Strategy):
            def on_data(self, timestamp, data, context, broker):
                for asset, bar in data.items():
                    signals = bar.get("signals", {})
                    signal = signals.get("signal", 0)
                    if signal == -1 and asset not in broker.positions:
                        broker.submit_order(asset, -100)
                    elif signal == 1 and asset in broker.positions:
                        broker.submit_order(asset, 100)

        # Extra bar after close signal so SAME_BAR mode can process the order
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, d) for d in range(1, 7)],
                "asset": ["TEST"] * 6,
                "open": [100.0, 100.0, 95.0, 92.0, 90.0, 90.0],
                "high": [101.0, 100.5, 96.0, 93.0, 91.0, 91.0],
                "low": [99.0, 94.0, 91.0, 89.0, 88.0, 89.0],
                "close": [100.0, 95.0, 92.0, 90.0, 90.0, 90.0],
                "volume": [1000] * 6,
            }
        )
        signals = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, d) for d in range(1, 7)],
                "asset": ["TEST"] * 6,
                "signal": [-1, 0, 0, 0, 1, 0],
            }
        )

        config = BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            allow_short_selling=True,
        )
        feed = DataFeed(prices_df=prices, signals_df=signals)
        result = Engine(feed, ShortStrategy(), config).run()

        trades = [t for t in result.trades if t.status == "closed"]
        assert len(trades) == 1
        trade = trades[0]

        # Short at 100, exit at 90 → profitable → pnl_percent > 0
        assert trade.pnl > 0, f"Short trade should be profitable, got pnl={trade.pnl}"
        assert trade.pnl_percent > 0, (
            f"Profitable short should have positive pnl_percent, got {trade.pnl_percent}"
        )
        assert trade.direction == "short"
        assert trade.quantity < 0

    def test_short_mfe_mae_nonzero(self):
        """Short trade MFE/MAE should not be stuck at zero."""
        import polars as pl

        from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

        class ShortStrategy(Strategy):
            def on_data(self, timestamp, data, context, broker):
                for asset, bar in data.items():
                    signals = bar.get("signals", {})
                    signal = signals.get("signal", 0)
                    if signal == -1 and asset not in broker.positions:
                        broker.submit_order(asset, -100)
                    elif signal == 1 and asset in broker.positions:
                        broker.submit_order(asset, 100)

        # Extra bar after close signal for SAME_BAR order processing
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, d) for d in range(1, 9)],
                "asset": ["TEST"] * 8,
                "open": [100.0, 100.0, 97.0, 95.0, 105.0, 93.0, 90.0, 90.0],
                "high": [101.0, 101.0, 98.0, 96.0, 108.0, 94.0, 91.0, 91.0],
                "low": [99.0, 96.0, 94.0, 93.0, 95.0, 91.0, 88.0, 89.0],
                "close": [100.0, 97.0, 95.0, 95.0, 103.0, 92.0, 90.0, 90.0],
                "volume": [1000] * 8,
            }
        )
        signals = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, d) for d in range(1, 9)],
                "asset": ["TEST"] * 8,
                "signal": [-1, 0, 0, 0, 0, 0, 1, 0],
            }
        )

        config = BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            allow_short_selling=True,
        )
        feed = DataFeed(prices_df=prices, signals_df=signals)
        result = Engine(feed, ShortStrategy(), config).run()

        trades = [t for t in result.trades if t.status == "closed"]
        assert len(trades) == 1
        trade = trades[0]

        # MFE should be positive (price dropped to 88 → favorable for short)
        assert trade.mfe > 0, f"Short MFE should be positive, got {trade.mfe}"
        # MAE should be negative (price rose to 108 → adverse for short)
        assert trade.mae < 0, f"Short MAE should be negative, got {trade.mae}"
