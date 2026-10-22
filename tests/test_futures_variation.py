"""Public Engine acceptance for opt-in variation cash; independent dollar oracles."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType

TIMES = [datetime(2025, 3, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(4)]


def run_future(quantity, *, mode="on_mark", settlements=None, legacy=False):
    reads = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("ES", quantity)
            before = broker.cash
            for _ in range(3):
                broker.get_account_value()
            assert broker.cash == before
            reads.append(broker.cash)

    options = (
        {} if legacy else {"accounting_method": "futures_variation", "settlement_policy": mode}
    )
    spec = ContractSpec(
        "ES", asset_class=AssetClass.FUTURE, multiplier=50, currency="USD", **options
    )
    frame = pl.DataFrame(
        {"timestamp": TIMES, "asset": ["ES"] * 4, "close": [4000.0, 4000.0, 4010.0, 4010.0]}
    )
    config = BacktestConfig(
        calendar=None,
        initial_cash=100000,
        allow_leverage=True,
        allow_short_selling=True,
        commission_type=CommissionType.PER_TRADE,
        commission_per_trade=2,
        slippage_rate=0,
        fixed_margin_schedule={"ES": (15000, 12000)},
    )
    kwargs = {} if settlements is None else {"settlement_events": settlements}
    engine = Engine(
        DataFeed(prices_df=frame),
        Buy(),
        config,
        contract_specs={"ES": spec},
        holdings_sampling="bar",
        **kwargs,
    )
    return engine, engine.run(), reads


@pytest.mark.parametrize("quantity", [1, -1])
def test_mark_variation_has_no_principal_cash_movement_and_reads_cannot_pay_again(quantity):
    engine, result, reads = run_future(quantity)
    assert reads == [100000, 99998, 99998 + quantity * 500, 99998 + quantity * 500]
    assert result.to_equity_dataframe()["equity"].to_list() == reads
    assert engine.broker.get_position("ES").quantity == quantity
    holdings = result.to_holdings_dataframe()
    assert holdings["signed_notional"][-1] == quantity * 200500
    assert holdings["liquidation_value"][-1] == 0


@pytest.mark.parametrize("quantity", [1, -1])
def test_scheduled_variation_keeps_unpaid_pnl_out_of_cash_until_declared_payment(quantity):
    engine, result, reads = run_future(quantity, mode="scheduled")
    assert reads == [100000, 99998, 99998, 99998]
    assert result.to_equity_dataframe()["equity"].to_list() == [
        100000,
        99998,
        99998 + quantity * 500,
        99998 + quantity * 500,
    ]
    assert result.to_holdings_dataframe()["liquidation_value"][-1] == quantity * 500


@pytest.mark.parametrize("quantity", [1, -1])
def test_explicit_settlement_transfers_pnl_once_without_changing_nav(quantity):
    event = pl.DataFrame(
        [
            {
                "timestamp": TIMES[3],
                "asset": "ES",
                "available_at": TIMES[2],
                "settlement_price": 4010.0,
                "currency": "USD",
                "source": "declared-settlement",
                "event_id": "day3",
                "revision_id": "r1",
            }
        ]
    )
    _, result, reads = run_future(quantity, mode="scheduled", settlements=event)
    assert reads[-2:] == [99998, 99998 + quantity * 500]
    assert result.to_equity_dataframe()["equity"].to_list()[-2:] == [99998 + quantity * 500] * 2


def test_default_signed_notional_account_does_not_change_when_variation_is_added():
    _, result, reads = run_future(1, legacy=True)
    assert reads == [100000, -100002, -100002, -100002]
    assert result.to_equity_dataframe()["equity"].to_list() == [100000, 99998, 100498, 100498]
