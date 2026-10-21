"""Runtime account projections must report native marks and actual contract units."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("inf"), float("nan"), True, None, 10**400])
def test_invalid_contract_multiplier_cannot_create_a_free_or_reversed_notional_position(multiplier):
    started = []

    class Buy(Strategy):
        def on_start(self, broker):
            started.append(True)

        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("A", 1)

    with pytest.raises(ValueError, match="multiplier must be finite and positive"):
        Engine(
            DataFeed(
                prices_df=pl.DataFrame(
                    {
                        "timestamp": [datetime(2025, 1, 6, 21, tzinfo=UTC)],
                        "asset": ["A"],
                        "close": [100.0],
                    }
                )
            ),
            Buy(),
            BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR),
            contract_specs={"A": ContractSpec("A", multiplier=multiplier)},
        ).run()
    assert started == []


@pytest.mark.parametrize(
    "side,multiplier,cash,last_equity",
    [
        (1, 1.0, 99798.8, 100018.8),
        (-1, 1.0, 100198.8, 99978.8),
        (1, 50.0, 89998.8, 100998.8),
        (-1, 50.0, 109998.8, 98998.8),
    ],
)
def test_account_snapshot_keeps_fees_signed_exposure_and_current_marks_in_native_units(
    side, multiplier, cash, last_equity
):
    from dataclasses import FrozenInstanceError

    from ml4t.backtest.config import CommissionType

    snapshots = []
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7]]

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("A", side * 2)
            snapshots.append(broker.get_account_snapshot())
            with pytest.raises(FrozenInstanceError):
                snapshots[-1].cash = 0

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {"timestamp": times, "asset": ["A"] * 2, "close": [100.0, 110.0]}
            )
        ),
        Trade(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            execution_mode=ExecutionMode.SAME_BAR,
            immediate_fill=True,
            allow_short_selling=True,
            commission_type=CommissionType.PER_ORDER,
            commission_per_share=0.1,
            commission_per_trade=1,
        ),
        contract_specs={"A": ContractSpec("A", multiplier=multiplier)},
    ).run()
    assert [s.timestamp for s in snapshots] == times
    assert [s.cash for s in snapshots] == pytest.approx([cash, cash])
    assert [s.equity for s in snapshots] == pytest.approx([99998.8, last_equity])
    assert [s.net_exposure for s in snapshots] == [side * 200 * multiplier, side * 220 * multiplier]
    assert [s.positions_value for s in snapshots] == pytest.approx(
        [side * 200 * multiplier, side * 220 * multiplier]
    )
    assert [s.gross_exposure for s in snapshots] == [200 * multiplier, 220 * multiplier]
    assert result.to_fills_dataframe()["commission"].to_list() == [1.2]


def test_contract_input_and_returned_metadata_cannot_change_bound_execution_units():
    specification = ContractSpec("A")
    seen = []

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            specification.multiplier = 2
            returned = broker.get_contract_spec("A")
            returned.multiplier = 3
            assert broker.get_multiplier("A") == 1
            seen.append(broker.submit_order("A", 1).order_id)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [datetime(2025, 1, 6, 21, tzinfo=UTC)],
                    "asset": ["A"],
                    "close": [100.0],
                }
            )
        ),
        Trade(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        contract_specs={"A": specification},
    ).run()
    assert seen == ["ORD-1"]
    assert result.to_fills_dataframe()["multiplier"].to_list() == [1.0]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [99900.0]


@pytest.mark.parametrize("false_valued", [False, True])
def test_holdings_preserve_declared_currency_even_when_contract_spec_is_false_valued(false_valued):
    class DeclaredContract(ContractSpec):
        def __bool__(self):
            return not false_valued

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("A", 2)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [datetime(2025, 1, 6, 21, tzinfo=UTC)],
                    "asset": ["A"],
                    "close": [100.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        contract_specs={"A": DeclaredContract("A", multiplier=50.0, currency="USD")},
        holdings_sampling="bar",
    ).run()
    holdings = result.to_holdings_dataframe()
    assert holdings["currency"].to_list() == ["USD"]
    assert holdings["quantity"].to_list() == [2.0]
    assert holdings["multiplier"].to_list() == [50.0]
    assert holdings["signed_notional"].to_list() == [10000.0]
    assert holdings["liquidation_value"].to_list() == [10000.0]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [90000.0]
    assert result.to_portfolio_state_dataframe()["equity"].to_list() == [100000.0]


def test_position_value_is_observed_directly_instead_of_subtracting_large_cash_from_nav():
    seen = []

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("A", 1)
            seen.append(broker.get_account_snapshot().positions_value)

    Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [datetime(2025, 1, 6, 21, tzinfo=UTC)],
                    "asset": ["A"],
                    "close": [1.0],
                }
            )
        ),
        Trade(),
        BacktestConfig(
            calendar=None,
            initial_cash=1e18,
            execution_mode=ExecutionMode.SAME_BAR,
            immediate_fill=True,
        ),
    ).run()
    assert seen == [1.0]


def test_bound_and_recorded_settings_are_independent_of_caller_and_completed_engine_edits():
    from ml4t.backtest.config import ExecutionPrice, SlippageType

    feed = DataFeed(
        prices_df=pl.DataFrame(
            {"timestamp": [datetime(2025, 1, 6, 21, tzinfo=UTC)], "asset": ["A"], "close": [100.0]}
        ),
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
            "calendar": None,
        },
    )
    settings = BacktestConfig(
        feed_spec=feed.feed_spec,
        calendar=None,
        initial_cash=1000,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.CLOSE,
        slippage_type=SlippageType.SPREAD,
        slippage_spread_by_asset={"A": 2.0},
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("A", 1)

    engine = Engine(feed, Buy(), settings)
    settings.initial_cash = 1_000_000
    settings.slippage_spread_by_asset["A"] = 20.0
    result = engine.run()
    assert result.config.initial_cash == 1000
    assert result.config.slippage_spread_by_asset == {"A": 2.0}
    assert result.to_fills_dataframe()["price"].to_list() == [101.0]
    engine.config.initial_cash = 2000
    engine.config.slippage_spread_by_asset["A"] = 4.0
    assert result.config.initial_cash == 1000
    assert result.config.slippage_spread_by_asset == {"A": 2.0}


def test_get_fills_returns_detached_native_records_without_mutating_observed_execution():
    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            first = broker.submit_order("A", 2)
            broker.submit_order("A", 3)
            selected = broker.get_fills(first.order_id)
            assert [fill.quantity for fill in selected] == [2]
            selected[0].quantity = 1000
            selected.clear()
            assert [fill.quantity for fill in broker.get_fills()] == [2, 3]
            assert broker.get_fills("unknown") == []

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [datetime(2025, 1, 6, 21, tzinfo=UTC)],
                    "asset": ["A"],
                    "close": [100.0],
                }
            )
        ),
        Trade(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR, immediate_fill=True),
    ).run()
    assert [fill.quantity for fill in result.fills] == [2, 3]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [99500]
