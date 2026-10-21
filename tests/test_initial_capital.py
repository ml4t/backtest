"""Real Engine regressions for costs incurred before the first recorded mark."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, DataFrequency, ExecutionMode
from ml4t.backtest.result import BacktestResult


class BuyFirstBar(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if not broker.positions:
            broker.submit_order("TEST", 10)


@pytest.mark.parametrize("bars", [1, 3])
@pytest.mark.parametrize("calendar", [None, "NYSE", "CME_Equity"])
def test_first_bar_fee_is_an_investment_loss_not_initial_capital(tmp_path, bars, calendar):
    timestamps = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3, 6]][:bars]
    prices = pl.DataFrame(
        {"timestamp": timestamps, "asset": ["TEST"] * bars, "close": [100.0] * bars}
    )
    config = BacktestConfig(
        initial_cash=10000.0,
        calendar=None,
        execution_mode=ExecutionMode.SAME_BAR,
        commission_type=CommissionType.PER_TRADE,
        commission_per_trade=10.0,
    )
    result = Engine(DataFeed(prices_df=prices), BuyFirstBar(), config).run()
    assert result.to_fills_dataframe()["commission"].to_list() == [10.0]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [8990.0] * bars
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        assert actual.metrics["initial_cash"] == 10000.0
        assert actual.metrics["total_return"] == pytest.approx(-0.001)
        assert actual.metrics["max_drawdown"] == pytest.approx(0.001)
        assert actual.equity_curve == list(zip(timestamps, [9990.0] * bars))
        equity = actual.to_equity_dataframe()
        assert equity["return"].to_list() == pytest.approx([-0.001] + [0.0] * (bars - 1))
        assert equity["cumulative_return"].to_list() == pytest.approx([-0.001] * bars)
        assert equity["drawdown"].to_list() == pytest.approx([-0.001] * bars)
        assert equity["high_water_mark"].to_list() == [10000.0] * bars
        assert actual.to_daily_returns(calendar=calendar).to_list() == pytest.approx(
            [-0.001] + [0.0] * (bars - 1)
        )
    assert result.equity.initial_value == 10000.0
    assert result.equity.returns.tolist() == pytest.approx([-0.001] + [0.0] * (bars - 1))
    assert result.equity.drawdown_series().tolist() == pytest.approx([-0.001] * bars)


def test_legacy_artifact_does_not_invent_unrecorded_initial_capital(tmp_path):
    import json

    timestamps = [datetime(2025, 1, 2, tzinfo=UTC), datetime(2025, 1, 3, tzinfo=UTC)]
    result = BacktestResult([], list(zip(timestamps, [9990.0, 9990.0])), [], {})
    result.to_parquet(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 2
    manifest.pop("initial_capital")
    manifest_path.write_text(json.dumps(manifest))
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.initial_capital is None
    assert restored.to_returns_series().to_list() == [0.0, 0.0]


def test_all_cash_observations_remain_real_zero_returns(tmp_path):
    timestamps = [datetime(2025, 1, day, tzinfo=UTC) for day in [2, 3, 6]]
    prices = pl.DataFrame({"timestamp": timestamps, "asset": ["TEST"] * 3, "close": [100.0] * 3})

    class AllCash(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    result = Engine(
        DataFeed(prices_df=prices), AllCash(), BacktestConfig(initial_cash=10000, calendar=None)
    ).run()
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        assert actual.initial_capital == 10000
        assert actual.to_equity_dataframe()["timestamp"].to_list() == timestamps
        assert actual.to_returns_series().to_list() == [0.0] * 3
        assert actual.to_fills_dataframe().is_empty()


def test_session_drawdown_uses_the_observed_path_instead_of_reordering_high_and_low():
    dates = [datetime(2025, 1, 2, hour, tzinfo=UTC) for hour in [15, 16, 17]]
    prices = pl.DataFrame(
        {"timestamp": dates, "asset": ["TEST"] * 3, "close": [100.0, 90.0, 110.0]}
    )

    class Invest(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if not broker.positions:
                broker.submit_order("TEST", 100)

    result = Engine(
        DataFeed(prices_df=prices),
        Invest(),
        BacktestConfig(
            initial_cash=10000,
            calendar="NYSE",
            data_frequency=DataFrequency.HOURLY,
            execution_mode=ExecutionMode.SAME_BAR,
        ),
    ).run()
    assert result.to_portfolio_state_dataframe()["equity"].to_list() == [10000, 9000, 11000]
    assert result.to_daily_pnl(session_aligned=True)["intra_session_dd"].to_list() == pytest.approx(
        [-0.1]
    )


def test_session_drawdown_excludes_external_cash_flows_and_keeps_first_fee(tmp_path):
    dates = [datetime(2025, 1, 2, hour, tzinfo=UTC) for hour in [15, 16, 17]]
    prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 3, "close": [100.0] * 3})
    flows = pl.DataFrame(
        {
            "timestamp": dates[1:],
            "event_id": ["deposit", "withdraw"],
            "kind": ["delta", "delta"],
            "value": [10000.0, -10000.0],
        }
    )
    result = Engine(
        DataFeed(prices_df=prices, capital_flows=flows),
        BuyFirstBar(),
        BacktestConfig(
            initial_cash=10000,
            calendar="NYSE",
            data_frequency=DataFrequency.HOURLY,
            execution_mode=ExecutionMode.SAME_BAR,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=10,
        ),
    ).run()
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        assert actual.to_portfolio_state_dataframe()["equity"].to_list() == [9990, 19990, 9990]
        daily = actual.to_daily_pnl(session_aligned=True)
        assert daily["intra_session_dd"].to_list() == pytest.approx([-0.001])
        assert daily["return_pct"].to_list() == pytest.approx([-0.001])
        assert daily["pnl"].to_list() == pytest.approx([-10])
