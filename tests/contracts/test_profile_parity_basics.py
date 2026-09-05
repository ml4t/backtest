from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest.config import BacktestConfig
from ml4t.backtest.engine import run_backtest
from ml4t.backtest.profiles import list_profiles
from ml4t.backtest.risk.position.dynamic import TrailingStop
from ml4t.backtest.strategy import Strategy
from ml4t.backtest.types import OrderSide, StopLevelBasis


def _prices() -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows = []
    for i, (open_, close) in enumerate([(100.0, 101.0), (110.0, 111.0)]):
        ts = start + timedelta(days=i)
        rows.append(
            {
                "timestamp": ts,
                "asset": "AAPL",
                "open": open_,
                "high": max(open_, close),
                "low": min(open_, close),
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    return pl.DataFrame(rows)


class _BuyOnce(Strategy):
    def __init__(self) -> None:
        self.done = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if not self.done:
            broker.submit_order("AAPL", 1.0)
            self.done = True


class _BuyOnMissingClose(Strategy):
    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp.day == 2:
            broker.submit_order("AAPL", 1.0)


class _ExerciseVectorbtShortCollateral(Strategy):
    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp.day == 1:
            broker.submit_order("B", 5.0, side=OrderSide.SELL)
        elif timestamp.day == 2:
            broker.submit_order("A", 5.0)
            broker.submit_order("B", 5.0, side=OrderSide.SELL)


def _flat_two_asset_prices() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, day),
                "asset": asset,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1_000_000.0,
            }
            for day in (1, 2, 3)
            for asset in ("A", "B")
        ]
    )


def _prices_with_missing_middle() -> pl.DataFrame:
    return (
        _prices()
        .vstack(
            pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 3)],
                    "asset": ["AAPL"],
                    "open": [120.0],
                    "high": [120.0],
                    "low": [120.0],
                    "close": [120.0],
                    "volume": [1_000_000.0],
                }
            )
        )
        .with_columns(
            pl.when(pl.col("timestamp") == datetime(2024, 1, 2))
            .then(float("nan"))
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("open", "high", "low", "close")
        )
    )


class _BuyWithTrailingStop(Strategy):
    def __init__(self) -> None:
        self.done = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if not self.done:
            broker.set_position_rules(TrailingStop(pct=0.1))
            broker.submit_order("AAPL", 1.0)
            self.done = True


def _backtrader_trailing_prices() -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows = []
    bars = [
        (100.0, 101.0, 99.0, 100.0),
        (110.0, 111.0, 94.0, 95.0),
        (95.0, 96.0, 85.0, 90.0),
        (100.0, 101.0, 99.0, 100.0),
    ]
    for offset, (open_, high, low, close) in enumerate(bars):
        rows.append(
            {
                "timestamp": start + timedelta(days=offset),
                "asset": "AAPL",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    return pl.DataFrame(rows)


def test_profile_registry_has_expected_core_profiles() -> None:
    assert list_profiles() == ["backtrader", "default", "lean", "realistic", "vectorbt", "zipline"]


def test_string_preset_and_explicit_preset_config_match() -> None:
    by_name = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="vectorbt")
    by_config = run_backtest(
        prices=_prices(),
        strategy=_BuyOnce(),
        config=BacktestConfig.from_preset("vectorbt"),
    )

    assert by_name.metrics["final_value"] == by_config.metrics["final_value"]
    assert by_name.trades[0].entry_price == by_config.trades[0].entry_price


def test_profiles_enforce_expected_entry_timing_contract() -> None:
    vbt = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="vectorbt")
    bt = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="backtrader")
    zl = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="zipline")
    lean = run_backtest(prices=_prices(), strategy=_BuyOnce(), config="lean")

    assert vbt.trades[0].entry_price == 101.0  # same-bar close
    assert bt.trades[0].entry_price == 110.0  # next-bar open with zero default slippage
    assert zl.trades[0].entry_price == 110.0  # configured next-bar open with zero slippage
    assert lean.trades[0].entry_price == 110.0  # DefaultBrokerageModel has null slippage


def test_vectorbt_profile_drops_signal_order_with_missing_close() -> None:
    result = run_backtest(
        prices=_prices_with_missing_middle(),
        strategy=_BuyOnMissingClose(),
        config="vectorbt",
    )

    assert result.fills == []
    assert len(result.rejected_orders) == 1
    assert result.rejected_orders[0].rejection_code == "price_unavailable"


def test_vectorbt_strict_locks_short_collateral_from_new_exposure() -> None:
    config = BacktestConfig.from_preset("vectorbt_strict")
    config.initial_cash = 1_000.0

    result = run_backtest(
        prices=_flat_two_asset_prices(),
        strategy=_ExerciseVectorbtShortCollateral(),
        config=config,
    )

    assert [(fill.asset, fill.quantity, fill.side.value) for fill in result.fills] == [
        ("B", 5.0, "sell"),
        ("A", 5.0, "buy"),
    ]


def test_zipline_profile_defers_pending_order_across_stale_bar() -> None:
    result = run_backtest(
        prices=_prices_with_missing_middle(),
        strategy=_BuyOnce(),
        config="zipline",
    )

    assert result.fills[0].timestamp == datetime(2024, 1, 3)
    assert result.fills[0].price == 120.0


def test_backtrader_profile_uses_signal_price_stop_basis() -> None:
    cfg = BacktestConfig.from_preset("backtrader")
    assert cfg.stop_level_basis == StopLevelBasis.SIGNAL_PRICE


def test_backtrader_profile_trailing_stop_uses_signal_close_before_fill() -> None:
    result = run_backtest(
        prices=_backtrader_trailing_prices(),
        strategy=_BuyWithTrailingStop(),
        config="backtrader",
    )

    assert result.trades[0].entry_price == 110.0
    assert result.trades[0].exit_price == 90.0


def test_backtrader_profile_parity_order_knobs() -> None:
    cfg = BacktestConfig.from_preset("backtrader")
    assert cfg.rebalance_headroom_pct == 1.0
    assert cfg.missing_price_policy.value == "use_last"
    assert cfg.late_asset_policy.value == "allow"
    assert cfg.late_asset_min_bars == 1


def test_zipline_profile_parity_order_knobs() -> None:
    cfg = BacktestConfig.from_preset("zipline")
    assert cfg.rebalance_headroom_pct == 1.0
    assert cfg.missing_price_policy.value == "use_last"
    assert cfg.late_asset_policy.value == "allow"


def test_zipline_strict_uses_credit_short_cash_policy() -> None:
    """Zipline_strict must use 'credit' so longs and shorts are cash-checked equally."""
    cfg = BacktestConfig.from_preset("zipline_strict")
    assert cfg.short_cash_policy.value == "credit"


def test_profile_registry_has_lean_profile() -> None:
    """LEAN profile must be a core profile, not just a strict variant."""
    cfg = BacktestConfig.from_preset("lean")
    assert cfg.preset_name == "lean"
    assert cfg.execution_mode.value == "next_bar"
    assert cfg.execution_price.value == "open"
    assert cfg.fill_ordering.value == "sequential"
    assert cfg.commission_per_share == 0.005
    assert cfg.commission_minimum == 1.0
    assert cfg.slippage_type.value == "none"
    assert cfg.initial_margin == 0.5
    assert cfg.long_maintenance_margin == 0.5
    assert cfg.short_maintenance_margin == 0.5
    assert cfg.rebalance_headroom_pct == 0.9975


def test_lean_profile_is_independent_of_backtrader() -> None:
    """LEAN profile must remain distinct from Backtrader on execution semantics."""
    lean = BacktestConfig.from_preset("lean")
    bt = BacktestConfig.from_preset("backtrader")
    assert lean.fill_ordering.value == "sequential"
    assert bt.fill_ordering.value == "fifo"
    assert lean.commission_per_share == 0.005
    assert bt.commission_rate == 0.0


def test_quantconnect_alias_resolves_to_lean() -> None:
    """The 'quantconnect' alias must resolve to the lean profile."""
    cfg = BacktestConfig.from_preset("quantconnect")
    assert cfg.preset_name == "quantconnect"
    lean = BacktestConfig.from_preset("lean")
    assert cfg.fill_ordering == lean.fill_ordering
    assert cfg.allow_leverage == lean.allow_leverage


def test_ibkr_us_stocks_fixed_alias_resolves() -> None:
    cfg = BacktestConfig.from_preset("ibkr:us:stocks:fixed")
    canonical = BacktestConfig.from_preset("ibkr_us_stocks_fixed")
    assert cfg.preset_name == "ibkr:us:stocks:fixed"
    assert cfg.commission_type == canonical.commission_type
    assert cfg.commission_per_share == canonical.commission_per_share
    assert cfg.commission_minimum == canonical.commission_minimum
    assert cfg.slippage_type == canonical.slippage_type
