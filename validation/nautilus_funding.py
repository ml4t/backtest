"""Compare native perpetual funding payments on the frozen real crypto bundle."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from importlib import import_module
from math import fsum
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import pandas as pd
import polars as pl
from real_strategy_runner import load_bundle
from reference_workloads import CASH, verify

FUNDING_BUNDLE = "2acee3c8542043266e6f9c0dfc9434c95b4469ef6fcaf0556efc569fc9721cdd"
POSITIONS = {"BTCUSDT": 1, "ETHUSDT": -1}
SECOND = 10**9
Row = tuple[int, str, float, float]


def load_funding(source: Path) -> list[Row]:
    bundle = load_bundle(source)
    if bundle["manifest"]["bundle_sha256"] != FUNDING_BUNDLE:
        raise ValueError(f"Expected the frozen funding bundle {FUNDING_BUNDLE}")
    market = (
        bundle["market"]
        .filter(pl.col("symbol").is_in(POSITIONS))
        .select("timestamp", "symbol", pl.col("open").round(2).alias("mark"))
    )
    funding = bundle["funding"].filter(pl.col("symbol").is_in(POSITIONS))
    frame = market.join(funding, on=["timestamp", "symbol"], how="inner", validate="1:1")
    if (
        frame.height != market.height
        or frame.height != funding.height
        or set(frame["funding_interval_hours"]) != {8}
        or frame.null_count().sum_horizontal()[0]
        or frame["mark"].min() <= 0
    ):
        raise ValueError("The frozen market and funding series do not align")
    return (
        frame.select(pl.col("timestamp").dt.epoch("ns"), "symbol", "mark", "funding_rate")
        .sort("timestamp", "symbol")
        .rows()
    )


def expected_funding(rows: list[Row]) -> dict:
    initial = rows[0][0]
    fills = sorted(
        (time + SECOND, symbol, POSITIONS[symbol], round(mark * 100))
        for time, symbol, mark, _ in rows
        if time == initial
    )
    funding = [
        (time, symbol, round(-POSITIONS[symbol] * mark * rate, 8))
        for time, symbol, mark, rate in rows
        if time != initial
    ]
    return {"fills": fills, "funding": funding, "positions": POSITIONS}


def run_ml4t_funding(rows: list[Row]) -> tuple[dict, dict]:
    from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
    from ml4t.backtest.config import ExecutionMode, ExecutionPrice

    opening = rows[0][0] + SECOND

    class Hold(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if pd.Timestamp(timestamp).value == opening:
                for symbol, qty in POSITIONS.items():
                    broker.submit_order(symbol, qty)

    marks = [(time, symbol, mark) for time, symbol, mark, _ in rows]
    marks.extend((opening, symbol, mark) for time, symbol, mark, _ in rows if time == rows[0][0])
    market = pl.DataFrame(marks, schema=["timestamp", "asset", "close"], orient="row").with_columns(
        pl.from_epoch("timestamp", time_unit="ns"),
        *(pl.col("close").alias(name) for name in ("open", "high", "low")),
        pl.lit(1000).alias("volume"),
    )
    funding = (
        pl.DataFrame(rows, schema=["timestamp", "asset", "mark", "rate"], orient="row")
        .with_columns(pl.from_epoch("timestamp", time_unit="ns"))
        .drop("mark")
    )
    engine = Engine(
        DataFeed(prices_df=market),
        Hold(),
        BacktestConfig(
            initial_cash=CASH,
            calendar=None,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.CLOSE,
        ),
        funding_df=funding,
    )
    start_wall, start_cpu = perf_counter(), process_time()
    result = engine.run()
    timing = {
        "run_wall_seconds": perf_counter() - start_wall,
        "run_cpu_seconds": process_time() - start_cpu,
    }
    state = result.to_portfolio_state_dataframe()
    opening_notional = sum(
        POSITIONS[symbol] * mark for time, symbol, mark, _ in rows if time == rows[0][0]
    )
    return {
        "funding_balance_change": state["cash"][-1] - CASH + opening_notional,
        "fills": sorted(
            (
                pd.Timestamp(f.timestamp).value,
                f.asset,
                int(f.quantity) * (1 if f.side.value == "buy" else -1),
                round(f.price * 100),
            )
            for f in result.fills
        ),
        "funding": sorted(
            (pd.Timestamp(f.timestamp).value, f.asset, round(f.cash_delta, 8))
            for f in result.funding_payments
            if f.quantity
        ),
        "positions": {
            asset: int(p.quantity) if (p := engine.broker.get_position(asset)) else 0
            for asset in POSITIONS
        },
    }, timing


def run_nautilus_funding(rows: list[Row]) -> tuple[dict, dict]:
    backtest: Any = import_module("nautilus_trader.backtest")
    model: Any = import_module("nautilus_trader.model")
    strategy: Any = import_module("nautilus_trader.trading").Strategy
    logger = import_module("nautilus_trader.common").LoggerConfig
    fee_model = import_module("nautilus_trader.execution").FixedFeeModel
    currency, venue = model.Currency.from_str("USDT"), model.Venue("SIM")
    instruments = {}
    for symbol in POSITIONS:
        sym = model.Symbol(symbol)
        instruments[symbol] = model.CryptoPerpetual(
            model.InstrumentId(sym, venue),
            sym,
            model.Currency.from_str(symbol[:-4]),
            currency,
            currency,
            False,
            2,
            0,
            model.Price(0.01, 2),
            model.Quantity.from_int(1),
            0,
            0,
        )
    opening = rows[0][0] + SECOND

    class Hold(strategy):
        def on_start(self):
            for instrument in instruments.values():
                self.subscribe_quotes(instrument.id)

        def on_quote(self, quote):
            if quote.ts_event == opening:
                asset = str(quote.instrument_id.symbol)
                self.submit_order(
                    self.order_factory.market(
                        quote.instrument_id,
                        model.OrderSide.BUY if POSITIONS[asset] > 0 else model.OrderSide.SELL,
                        model.Quantity.from_int(1),
                    )
                )

    engine = backtest.BacktestEngine(
        backtest.BacktestEngineConfig(
            logging=logger(bypass_logging=True),
            run_analysis=False,
        )
    )
    try:
        engine.add_venue(
            venue,
            model.OmsType.NETTING,
            model.AccountType.MARGIN,
            [model.Money(CASH, currency)],
            base_currency=currency,
            fee_model=fee_model(model.Money(0, currency)),
        )
        for instrument in instruments.values():
            engine.add_instrument(instrument)
        events = []
        size = model.Quantity.from_int(1000)
        for time, symbol, mark, rate in rows:
            ident = instruments[symbol].id
            price = model.Price(mark, 2)
            events.extend(
                (
                    model.QuoteTick(ident, price, price, size, size, time, time),
                    model.MarkPriceUpdate(ident, price, time, time),
                    model.FundingRateUpdate(ident, Decimal(str(rate)), time, time, interval=480),
                )
            )
            if time == rows[0][0]:
                events.append(model.QuoteTick(ident, price, price, size, size, opening, opening))
        engine.add_data(events)
        engine.add_strategy(Hold())
        start_wall, start_cpu = perf_counter(), process_time()
        engine.run()
        timing = {
            "run_wall_seconds": perf_counter() - start_wall,
            "run_cpu_seconds": process_time() - start_cpu,
        }
        return {
            "funding_balance_change": float(
                engine.cache.account_for_venue(venue).balance_total(currency)
            )
            - CASH,
            "fills": sorted(
                (
                    pd.Timestamp(f.ts_event).value,
                    str(f.instrument_id).split(".")[0],
                    int(float(f.last_qty)) * (1 if f.order_side == "BUY" else -1),
                    round(float(f.last_px) * 100),
                )
                for f in engine.generate_fills_report().itertuples()
            ),
            "funding": sorted(
                (
                    adjustment.ts_event,
                    str(position.instrument_id.symbol),
                    round(float(adjustment.pnl_change), 8),
                )
                for position in engine.cache.positions_open()
                for adjustment in position.adjustments()
                if adjustment.pnl_change is not None
            ),
            "positions": {
                symbol: int(engine.portfolio.net_position(inst.id))
                for symbol, inst in instruments.items()
            },
        }, timing
    finally:
        engine.dispose()


def funding_worker(source: Path, side: str) -> dict:
    rows = load_funding(source)
    actual, timing = run_ml4t_funding(rows) if side == "ml4t" else run_nautilus_funding(rows)
    observed = actual.pop("funding")
    balance_change = actual.pop("funding_balance_change")
    wanted = expected_funding(rows)
    expected = wanted.pop("funding")
    verify(actual, wanted)
    if len(observed) != len(expected):
        raise ValueError(f"Funding payment count: {len(observed)} vs {len(expected)}")
    for i, (a, b) in enumerate(zip(observed, expected, strict=True)):
        if a[:2] != b[:2] or abs(a[2] - b[2]) > 1.00001e-8:
            raise ValueError(f"Funding payment {i}: {a} vs {b}")
    if abs(balance_change - fsum(row[2] for row in observed)) > len(observed) * 1e-8:
        raise ValueError("Funding adjustments do not reconcile to the account balance")
    actual["funding"] = expected
    return {
        "observed_funding": observed,
        "observed_funding_balance_change": balance_change,
        "input_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "rows": len(rows),
        "ledger": actual,
        "timing": timing,
    }
