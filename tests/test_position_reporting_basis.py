"""Reporting basis must not replace the gross entry used for native P&L."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode, ExecutionPrice

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/zipline_position_basis/reference.json").read_text()
)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("multiplier", [1, 50])
def test_commission_adjusted_basis_preserves_reduce_and_flip_reporting_without_changing_pnl(
    side, multiplier
):
    reference = _REFERENCE["traces"][str(side)]
    times = [datetime.fromisoformat(row["timestamp"]) for row in reference]
    trace = []

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pos = broker.get_position("A")
            trace.append(
                (pos.quantity, pos.commission_adjusted_cost_basis, pos.entry_price)
                if pos is not None
                else (0, 0, 0)
            )
            i = times.index(timestamp)
            if i < 4:
                broker.submit_order("A", side * [10, 10, -5, -30][i])

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["A"] * 5,
                    "close": [90.0, 100.0, 110.0, 120.0, 130.0],
                }
            )
        ),
        Trade(),
        BacktestConfig(
            calendar=None,
            initial_cash=10000 * multiplier,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
            commission_type=CommissionType.PER_ORDER,
            commission_per_share=0.1 * multiplier,
        ),
        contract_specs={"A": ContractSpec("A", multiplier=multiplier)},
    ).run()
    assert [row[0] for row in trace] == [row["quantity"] for row in reference]
    assert [row[1] for row in trace] == pytest.approx([row["cost_basis"] for row in reference])
    assert [row[2] for row in trace] == [0, 100, 105, 105, 130]
    account = result.to_portfolio_state_dataframe()
    assert account["cash"].to_list() == pytest.approx(
        [row["cash"] * multiplier for row in reference]
    )
    assert account["equity"].to_list() == pytest.approx(
        [row["equity"] * multiplier for row in reference]
    )
    # Fees on reductions/reversals affect the reporting basis, but cannot be charged twice.
    assert sum(fill.commission for fill in result.fills) == pytest.approx(5.5 * multiplier)
    assert sum(trade.pnl for trade in result.trades) == pytest.approx(
        (reference[-1]["equity"] - 10000) * multiplier
    )
