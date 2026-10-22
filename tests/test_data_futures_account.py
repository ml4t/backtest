"""Consume actual optional Data preparation without rebuilding its units or identities."""

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType


def test_actual_prepared_contract_units_and_currency_feed_the_cme_variation_account():
    futures = pytest.importorskip("ml4t.data.futures")
    times = [datetime(2025, 3, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(4)]
    source = pl.DataFrame(
        {
            "timestamp": times,
            "asset": [17] * 4,
            "symbol": ["ESM25"] * 4,
            "close": [4000.0, 4000.0, 4010.0, 4010.0],
            "volume": [100.0] * 4,
            "currency": ["USD"] * 4,
            "price_quote_unit": ["index_points"] * 4,
            "mark_kind": ["trade_close"] * 4,
            "available_at": times,
            "source": ["declared-example"] * 4,
            "revision_id": ["r1"] * 4,
        }
    )
    definitions = pl.DataFrame(
        [
            {
                "asset": 17,
                "symbol": "ESM25",
                "root": "ES",
                "exchange": "CME",
                "instrument_type": "future",
                "valid_from": date(2025, 1, 1),
                "available_at": datetime(2025, 1, 1, tzinfo=UTC),
                "source": "declared-example",
                "revision_id": "r1",
                "expiration": date(2025, 6, 20),
                "last_trade_date": date(2025, 6, 20),
                "currency": "USD",
                "multiplier": 50.0,
                "tick_size": 0.25,
                "price_quote_unit": "index_points",
                "settlement_type": "cash",
                "volume_unit": "contracts",
            }
        ]
    )
    prepared = futures.prepare_futures_inputs(
        source,
        definitions,
        as_of=times[-1],
        feed_spec=FeedSpec(
            time_contract_version=2, calendar="CME_Equity", timestamp_semantics="event_time"
        ),
        roll_strategy=futures.HighestVolumeRoll(),
    )
    facts = prepared.bars.row(0, named=True)
    asset = f"declared-example:{facts['asset']}"
    native_bars = prepared.bars.with_columns(
        pl.concat_str(pl.col("source"), pl.lit(":"), pl.col("asset").cast(pl.String)).alias("asset")
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order(asset, 1)

    result = Engine(
        DataFeed(prices_df=native_bars, feed_spec=prepared.metadata["normalized_feed_spec"]),
        Buy(),
        BacktestConfig(
            initial_cash=100000,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
            fixed_margin_schedule={asset: (15000, 12000)},
        ),
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=facts["normalized_multiplier"],
                tick_size=facts["normalized_tick_size"],
                currency=facts["instrument_currency"],
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
        },
        holdings_sampling="bar",
    ).run()
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [
        100000,
        99998,
        100498,
        100498,
    ]
    assert result.to_holdings_dataframe()["asset"].to_list() == [asset] * 3
    assert result.to_futures_settlements_dataframe()["cash_delta"].to_list() == [500]
    assert result.instrument_accounting[asset]["currency"] == "USD"
