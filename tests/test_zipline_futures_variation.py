"""Actual original Future traces and exported raw contracts must reach native accounting."""

import hashlib
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl
import pytest

from ml4t.backtest import AssetClass, BacktestResult, ContractSpec, DataFeed
from ml4t.backtest.compat.zipline import PreparedAssetLookup, run_algorithm
from ml4t.backtest.compat.zipline.finance import commission, slippage

ROOT = Path(__file__).parent / "fixtures/futures_variation"
REFERENCE = json.loads((ROOT / "reference.json").read_text())


def snapshot(actual=False):
    with tarfile.open(ROOT / "snapshot.tar.gz") as archive:
        if actual:
            storage = pytest.importorskip("ml4t.data.storage")
            with TemporaryDirectory(prefix="ml4t-future-variation-") as directory:
                archive.extractall(directory, filter="data")
                return storage.read_snapshot(Path(directory) / "snapshot")
        manifest = json.loads(archive.extractfile("snapshot/manifest.json").read())
        frames = {}
        for name, entry in manifest["content"]["tables"].items():
            source = archive.extractfile("snapshot/" + entry["files"][0]["path"]).read()
            assert hashlib.sha256(source).hexdigest() == entry["files"][0]["sha256"]
            frames[name] = pl.read_parquet(io.BytesIO(source))
        return SimpleNamespace(
            manifest=manifest, tables=frames, snapshot_id=manifest["snapshot_id"]
        )


@pytest.mark.parametrize("actual", [False, True], ids=["captured", "installed-data"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_original_future_variation_cash_closing_basis_positions_and_fills(case, actual, tmp_path):
    captured = snapshot(actual)
    lookup = PreparedAssetLookup(captured)
    future = lookup.assets[0]
    raw = captured.tables["daily_bars"]
    clocks = pl.DataFrame(
        {
            "session": raw["timestamp"].unique().sort(),
            "timestamp": [datetime.fromisoformat(row["timestamp"]) for row in case["trace"]],
        }
    )
    prices = (
        raw.rename({"timestamp": "session"})
        .join(clocks, on="session")
        .with_columns(pl.col("timestamp").alias("available_at"))
    )
    feed = DataFeed(
        prices_df=prices,
        input_manifest=captured.manifest,
        feed_spec={
            "calendar": "us_futures",
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
            "data_frequency": "daily",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )
    trace = []
    plan = {int(key): quantity for key, quantity in case["order_plan"].items()}

    def initialize(context):
        context.set_benchmark(None)
        context.set_commission(us_futures=commission.PerFutureTrade(case.get("fee_per_order", 0)))
        context.set_slippage(us_futures=slippage.NoSlippage())
        context.step = 0

    def handle(context, data):
        if context.step in plan:
            context.order(future, plan[context.step])
        trace.append(
            {
                "timestamp": context.get_datetime().isoformat(),
                "cash": context.portfolio.cash,
                "nav": context.portfolio.portfolio_value,
                "positions": [
                    {
                        "sid": int(asset),
                        "quantity": p.amount,
                        "basis": p.cost_basis,
                        "mark": p.last_sale_price,
                    }
                    for asset, p in context.portfolio.positions.items()
                ],
            }
        )
        context.step += 1

    performance = run_algorithm(
        feed=feed,
        asset_lookup=lookup,
        initialize=initialize,
        handle_data=handle,
        capital_base=100000,
        contract_specs={
            future.key: ContractSpec(
                future.key,
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                tick_size=future.tick_size,
                currency="USD",
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
        },
    )
    assert performance["cash"].to_list() == case["cash"]
    assert performance["portfolio_value"].to_list() == case["nav"]
    assert performance["positions_value"].to_list() == [0] * len(case["nav"])
    assert trace == case["trace"]
    fills = [
        {
            "timestamp": fill["dt"].isoformat(),
            "sid": int(fill["sid"]),
            "quantity": fill["amount"],
            "price": fill["price"],
        }
        for transactions in performance["transactions"]
        for fill in transactions
    ]
    assert fills == case["fills"]
    result = performance.attrs["backtest_result"]
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_portfolio_state_dataframe().equals(result.to_portfolio_state_dataframe())
    assert restored.to_futures_settlements_dataframe().equals(
        result.to_futures_settlements_dataframe()
    )
