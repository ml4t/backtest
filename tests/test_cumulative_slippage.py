"""Cumulative volume impact is an explicit option, separate from native per-fill linear impact."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode, ExecutionPrice, SlippageType
from ml4t.backtest.execution import SharedVolumeParticipationLimit


def _run(model, *, repeat=False, volumes=None, quoted=False, shared_limit=True):
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]]
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 3,
            "close": [100.0] * 3,
            "volume": volumes if volumes is not None else [100.0] * 3,
        }
    )
    fields = {}
    if quoted:
        frame = frame.with_columns(pl.lit(101.0).alias("ask"), pl.lit(10.0).alias("ask_size"))
        fields = {"ask_col": "ask", "ask_size_col": "ask_size"}

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 1 if quoted else 10)
                broker.submit_order("TEST", 1 if quoted else 10)
            elif timestamp == times[1] and repeat:
                broker.submit_order("TEST", 10)

    return Engine(
        DataFeed(prices_df=frame, **fields),
        Submit(),
        BacktestConfig(
            calendar=None,
            initial_cash=10000,
            execution_mode=ExecutionMode.NEXT_BAR,
            slippage_type=SlippageType(model),
            slippage_rate=0.1,
            execution_price=ExecutionPrice.ASK if quoted else ExecutionPrice.CLOSE,
            next_bar_execution_price=ExecutionPrice.ASK if quoted else ExecutionPrice.CLOSE,
        ),
        execution_limits=SharedVolumeParticipationLimit(max_participation=1)
        if shared_limit
        else None,
    ).run()


def test_existing_per_fill_linear_volume_slippage_remains_a_distinct_method():
    result = _run("volume_based")
    assert result.to_fills_dataframe()["price"].to_list() == [101.0, 101.0]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 9980.0, 9980.0]


def test_cumulative_quadratic_option_matches_actual_zipline_multi_order_fills_and_nav():
    result = _run("cumulative_volume_share")
    assert result.to_fills_dataframe()["price"].to_list() == pytest.approx([100.1, 100.4])
    assert result.to_equity_dataframe()["equity"].to_list() == pytest.approx(
        [10000.0, 9995.0, 9995.0]
    )
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == pytest.approx(
        [10000.0, 7995.0, 7995.0]
    )


def test_cumulative_participation_resets_on_the_next_actual_bar():
    result = _run("cumulative_volume_share", repeat=True)
    assert result.to_fills_dataframe()["price"].to_list() == pytest.approx([100.1, 100.4, 100.1])
    assert result.to_equity_dataframe()["equity"].to_list() == pytest.approx(
        [10000.0, 9995.0, 9994.0]
    )


def test_cumulative_impact_uses_traded_volume_instead_of_quoted_depth():
    result = _run("cumulative_volume_share", quoted=True)
    fills = result.to_fills_dataframe()
    assert fills["available_size"].to_list() == [10.0, 10.0]
    assert fills["price"].to_list() == pytest.approx([101.00101, 101.00404])


def test_missing_traded_volume_cannot_be_replaced_with_a_zero_impact_execution():
    result = _run("cumulative_volume_share", volumes=[100.0, None, 100.0], shared_limit=False)
    fills = result.to_fills_dataframe()
    assert fills["timestamp"].dt.day().to_list() == [8, 8]
    assert fills["price"].to_list() == pytest.approx([100.1, 100.4])


def test_cumulative_selector_and_coefficient_survive_strict_config_round_trip():
    config = BacktestConfig(
        slippage_type=SlippageType("cumulative_volume_share"), slippage_rate=0.1
    )
    restored = BacktestConfig.from_dict(config.to_dict(), strict=True)
    assert restored.slippage_type.value == "cumulative_volume_share"
    assert restored.slippage_rate == 0.1


def test_cumulative_context_tracks_committed_volume_without_a_participation_limit():
    result = _run("cumulative_volume_share", shared_limit=False)
    assert result.to_fills_dataframe()["price"].to_list() == pytest.approx([100.1, 100.4])


def test_volume_impact_coefficients_are_not_reported_as_flat_transaction_cost_percentages():
    config = BacktestConfig(
        slippage_type=SlippageType("cumulative_volume_share"), slippage_rate=0.1
    )
    assert not any("Total transaction cost" in message for message in config.validate(warn=False))
    assert "coefficient=0.1" in config.describe()
