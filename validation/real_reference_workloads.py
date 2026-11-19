#!/usr/bin/env python3
"""Compare native strategy and execution paths on frozen real market data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import distributions, version
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from typing import Any

import pandas as pd
import polars as pl
from nautilus_funding import FUNDING_BUNDLE, funding_worker
from pysystemtrade_trend import trend_worker
from real_strategy_runner import load_bundle
from reference_workloads import (
    CASH,
    NAUTILUS_VERSION,
    PYSYSTEMTRADE_COMMIT,
    SAMPLE_COUNT,
    verify,
    verify_pysystemtrade_source,
)

BUNDLES = {
    "etfs": "01f38079ce47821a5379d3769e86f4a2170b88033108153bfc0f928698e946db",
    "fx_pairs": "48922296cea03bfc9867a9c4e1a1d352a0bd54c0b48671e3ca8d1fdddb95d762",
}
UNIVERSE = {
    "etfs": ("SPY", "QQQ", "IWM", "TLT"),
    "fx_pairs": ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD"),
}
FUTURES_CASE = "pysystemtrade-trend"
NAUTILUS_FUTURES_CASE = "nautilus-dated-futures"
TREND_CASES = {FUTURES_CASE, NAUTILUS_FUTURES_CASE}
FUNDING_CASE = "crypto_perps_funding"
FUNDING_SIDES = ("ml4t", "nautilus")
FUTURES_SIDES = ("ml4t", "pysystemtrade")
QUOTE_SIDES = ("ml4t", "nautilus-python", "nautilus-rust-indicators", "nautilus-rust")
FAST, SLOW = 10, 50


@dataclass
class Ema:
    period: int
    value: float = 0.0
    count: int = 0
    seed_sum: float = 0.0

    def update(self, price: float) -> float:
        self.count += 1
        if self.count <= self.period:
            self.seed_sum += price
            if self.count == self.period:
                self.value = self.seed_sum / self.period
        else:
            alpha = 2.0 / (self.period + 1)
            self.value = alpha * price + (1 - alpha) * self.value
        return self.value


class Cross:
    def __init__(self) -> None:
        self.fast = Ema(FAST)
        self.slow = Ema(SLOW)
        self.previous: bool | None = None
        self.observations = 0

    def signal(self, above: bool) -> int:
        self.observations += 1
        if self.observations < SLOW:
            return 0
        previous, self.previous = self.previous, above
        return 0 if previous is None or previous == above else (1 if above else -1)

    def update(self, price: float) -> int:
        return self.signal(self.fast.update(price) > self.slow.update(price))


@dataclass
class Quotes:
    rows: list[tuple[int, str, int]]
    precision: int
    size: int

    @property
    def scale(self) -> int:
        return 10**self.precision


def load_quotes(case: str, bundle: Path) -> Quotes:
    inputs = load_bundle(bundle)
    if inputs["manifest"]["bundle_sha256"] != BUNDLES[case]:
        raise ValueError(f"Expected the frozen {case} bundle {BUNDLES[case]}")
    precision, size = (2, 10) if case == "etfs" else (5, 1000)
    frame = (
        inputs["market"]
        .filter(pl.col("symbol").is_in(UNIVERSE[case]))
        .select(
            pl.col("timestamp").dt.epoch("ns"),
            "symbol",
            (pl.col("close") * 10**precision).round().cast(pl.Int64).alias("price"),
        )
        .sort("timestamp", "symbol")
    )
    if (
        set(frame["symbol"].unique()) != set(UNIVERSE[case])
        or frame.null_count().sum_horizontal()[0]
        or frame.select("timestamp", "symbol").is_duplicated().any()
        or frame["price"].min() <= 0
    ):
        raise ValueError(
            "The frozen quote projection must have one valid price per timestamp/asset"
        )
    return Quotes(frame.rows(), precision, size)


def expected_quotes(quotes: Quotes) -> dict:
    # pandas is independent of both incremental strategy implementations.
    frame = pd.DataFrame(quotes.rows, columns=pd.Index(["timestamp", "symbol", "price"]))

    def seeded_ema(prices: pd.Series, period: int) -> pd.Series:
        tail = prices.iloc[period - 1 :].copy()
        tail.iloc[0] = prices.iloc[:period].mean()
        return tail.ewm(span=period, adjust=False).mean()

    fills = []
    for symbol, rows in frame.groupby("symbol"):
        prices = rows.price / quotes.scale
        slow = seeded_ema(prices, SLOW)
        active = seeded_ema(prices, FAST).reindex(slow.index) > slow
        delta = active.astype(int).diff().fillna(0).astype(int)
        for i in delta.index[delta != 0]:
            fills.append(
                (int(rows.timestamp[i]), symbol, int(delta[i]) * quotes.size, int(rows.price[i]), 0)
            )
    return {
        "fills": sorted(fills),
        "positions": {
            symbol: sum(qty for _, asset, qty, _, _ in fills if asset == symbol)
            for symbol in frame.symbol.unique()
        },
    }


def elapsed(started: tuple[float, float]) -> dict[str, float]:
    return {
        "run_wall_seconds": perf_counter() - started[0],
        "run_cpu_seconds": process_time() - started[1],
    }


def run_ml4t_quotes(quotes: Quotes, case: str) -> tuple[dict, dict]:
    from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
    from ml4t.backtest.config import ExecutionMode, ExecutionPrice

    class Crossover(Strategy):
        def on_data(self, timestamp, data, context, broker):
            for symbol, bar in data.items():
                delta = states[symbol].update(bar["close"])
                if delta:
                    broker.submit_order(symbol, delta * quotes.size)

    states = {symbol: Cross() for symbol in UNIVERSE[case]}
    frame = (
        pl.DataFrame(quotes.rows, schema=["timestamp", "asset", "close"], orient="row")
        .with_columns(
            pl.from_epoch("timestamp", time_unit="ns"),
            (pl.col("close") / quotes.scale).alias("close"),
        )
        .with_columns(
            *(pl.col("close").alias(name) for name in ("open", "high", "low")),
            pl.lit(100_000).alias("volume"),
        )
    )
    engine = Engine(
        DataFeed(prices_df=frame),
        Crossover(),
        BacktestConfig(
            initial_cash=CASH,
            calendar=None,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.CLOSE,
        ),
        contract_specs={
            symbol: ContractSpec(
                symbol,
                asset_class=AssetClass.FOREX if case == "fx_pairs" else AssetClass.EQUITY,
                tick_size=1 / quotes.scale,
            )
            for symbol in states
        },
    )
    started = perf_counter(), process_time()
    result = engine.run()
    times = elapsed(started)
    if sum(state.observations for state in states.values()) != len(quotes.rows):
        raise ValueError("Backtest did not deliver every quote to the decision rule")
    return {
        "fills": sorted(
            (
                pd.Timestamp(f.timestamp).value,
                f.asset,
                f.quantity if f.side.value == "buy" else -f.quantity,
                round(f.price * quotes.scale),
                round(f.commission * 100),
            )
            for f in result.fills
        ),
        "positions": {
            symbol: (p.quantity if (p := engine.broker.get_position(symbol)) else 0)
            for symbol in states
        },
    }, times


def run_nautilus_quotes(quotes: Quotes, case: str, side: str) -> tuple[dict, dict]:
    backtest: Any = import_module("nautilus_trader.backtest")
    model: Any = import_module("nautilus_trader.model")
    trading: Any = import_module("nautilus_trader.trading")
    ema: Any = import_module("nautilus_trader.indicators").ExponentialMovingAverage
    logger_config = import_module("nautilus_trader.common").LoggerConfig
    fee_model = import_module("nautilus_trader.execution").FixedFeeModel

    usd = model.Currency.from_str("USD")
    venue = model.Venue("SIM")
    instruments = {}
    for symbol in UNIVERSE[case]:
        sym = model.Symbol(symbol)
        ident = model.InstrumentId(sym, venue)
        tick = model.Price(1 / quotes.scale, quotes.precision)
        if case == "fx_pairs":
            instruments[symbol] = model.CurrencyPair(
                ident,
                sym,
                model.Currency.from_str(symbol.split("_")[0]),
                usd,
                quotes.precision,
                0,
                tick,
                model.Quantity.from_int(1),
                0,
                0,
            )
        else:
            instruments[symbol] = model.Equity(ident, sym, usd, quotes.precision, tick, 0, 0)

    class Crossover(trading.Strategy):
        def configure(self, symbol):
            self.instrument_id = instruments[symbol].id
            self.cross = Cross()
            self.fast_native = ema(FAST, model.PriceType.MID)
            self.slow_native = ema(SLOW, model.PriceType.MID)

        def on_start(self):
            if side == "nautilus-rust-indicators":
                self.register_indicator_for_quote_ticks(self.instrument_id, self.fast_native)
                self.register_indicator_for_quote_ticks(self.instrument_id, self.slow_native)
            self.subscribe_quotes(self.instrument_id)

        def on_quote(self, quote):
            if side == "nautilus-python":
                delta = self.cross.update(float(quote.bid_price))
            else:
                delta = self.cross.signal(self.fast_native.value > self.slow_native.value)
            if delta:
                self.submit_order(
                    self.order_factory.market(
                        self.instrument_id,
                        model.OrderSide.BUY if delta > 0 else model.OrderSide.SELL,
                        model.Quantity.from_int(quotes.size),
                    )
                )

    engine = backtest.BacktestEngine(
        backtest.BacktestEngineConfig(
            logging=logger_config(bypass_logging=True),
            run_analysis=False,
        )
    )
    try:
        engine.add_venue(
            venue,
            model.OmsType.NETTING,
            model.AccountType.MARGIN,
            [model.Money(CASH, usd)],
            base_currency=usd,
            fee_model=fee_model(model.Money(0, usd)),
        )
        for instrument in instruments.values():
            engine.add_instrument(instrument)
        size = model.Quantity.from_int(100_000)
        ticks = []
        for timestamp, symbol, units in quotes.rows:
            price = model.Price(units / quotes.scale, quotes.precision)
            ticks.append(
                model.QuoteTick(
                    instruments[symbol].id, price, price, size, size, timestamp, timestamp
                )
            )
        engine.add_data(ticks)
        python_strategies = []
        for i, (symbol, instrument) in enumerate(instruments.items()):
            tag = f"{i:03}"
            strategy_id = model.StrategyId(f"Crossover-{tag}")
            if side == "nautilus-rust":
                engine.add_builtin_strategy(
                    "EmaCross",
                    trading.EmaCrossConfig(
                        instrument.id,
                        model.Quantity.from_int(quotes.size),
                        FAST,
                        SLOW,
                        strategy_id,
                        tag,
                    ),
                )
            else:
                strategy = Crossover(
                    trading.StrategyConfig(
                        strategy_id=strategy_id,
                        order_id_tag=tag,
                    ),
                )
                strategy.configure(symbol)
                python_strategies.append(strategy)
                engine.add_strategy(strategy)
        started = perf_counter(), process_time()
        engine.run()
        times = elapsed(started)
        if python_strategies and sum(s.cross.observations for s in python_strategies) != len(ticks):
            raise ValueError("Nautilus did not deliver every quote to the Python strategy")
        return {
            "fills": sorted(
                (
                    pd.Timestamp(row.ts_event).value,
                    str(row.instrument_id).split(".")[0],
                    int(float(row.last_qty)) * (1 if row.order_side == "BUY" else -1),
                    round(float(row.last_px) * quotes.scale),
                    round(float(str(row.commission).split()[0]) * 100),
                )
                for row in engine.generate_fills_report().itertuples()
            ),
            "positions": {
                symbol: int(engine.portfolio.net_position(instrument.id))
                for symbol, instrument in instruments.items()
            },
        }, times
    finally:
        engine.dispose()


def quote_worker(case: str, bundle: Path, side: str) -> dict:
    quotes = load_quotes(case, bundle)
    ledger, timing = (
        run_ml4t_quotes(quotes, case) if side == "ml4t" else run_nautilus_quotes(quotes, case, side)
    )
    verify(ledger, expected_quotes(quotes))
    return {
        "input_sha256": hashlib.sha256(json.dumps(quotes.rows).encode()).hexdigest(),
        "rows": len(quotes.rows),
        "ledger": ledger,
        "timing": timing,
    }


def collect(case: str, source: Path, sides: tuple[str, ...], samples: int) -> dict:
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        POLARS_MAX_THREADS="1",
        PYTHONHASHSEED="0",
    )

    def child(side: str) -> dict:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            case,
            str(source),
            "--worker",
            side,
        ]
        before, start = os.getloadavg(), perf_counter()
        out = json.loads(subprocess.check_output(command, env=env, text=True, timeout=180))
        out["timing"].update(
            process_wall_seconds=perf_counter() - start,
            load_before=before,
            load_after=os.getloadavg(),
        )
        return out

    results: dict[str, dict[str, Any]] = {
        side: {"correctness": child(side), "samples": []} for side in sides
    }
    if len({record["correctness"]["input_sha256"] for record in results.values()}) != 1:
        raise ValueError("Input projections differ between sides")
    fields = ("ledger", "forecast") if case in TREND_CASES else ("ledger",)
    for field in fields:
        reference = results[sides[0]]["correctness"][field]
        if any(record["correctness"][field] != reference for record in results.values()):
            raise ValueError(f"Comparison sides disagree on {field}")
    stability_fields = (
        (*fields, "observed_funding", "observed_funding_balance_change")
        if case == FUNDING_CASE
        else fields
    )
    for i in range(samples):
        for side in sides[i % len(sides) :] + sides[: i % len(sides)]:
            evidence = child(side)
            record = results[side]
            for key in ("input_sha256", "rows", *stability_fields, "observed_cash_cents"):
                if evidence.get(key) != record["correctness"].get(key):
                    raise ValueError(f"{side} changed {key} during measurement")
            record["samples"].append(evidence["timing"])
    for record in results.values():
        if samples:
            record["medians"] = {
                key: median(sample[key] for sample in record["samples"])
                for key in (
                    "run_wall_seconds",
                    "run_cpu_seconds",
                    "process_wall_seconds",
                    *(("forecast_wall_seconds",) if case in TREND_CASES else ()),
                )
            }
    return {
        "case": case,
        "bundle_sha256": FUNDING_BUNDLE if case == FUNDING_CASE else BUNDLES.get(case),
        "pysystemtrade_commit": PYSYSTEMTRADE_COMMIT if case in TREND_CASES else None,
        "harness_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "real_reference_workloads.py",
                "pysystemtrade_trend.py",
                "nautilus_funding.py",
                "nautilus_dated_futures.py",
                "reference_workloads.py",
                "real_strategy_runner.py",
            )
        },
        "platform": platform.platform(),
        "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "sides": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", choices=(*BUNDLES, FUTURES_CASE, NAUTILUS_FUTURES_CASE, FUNDING_CASE)
    )
    parser.add_argument("source", type=Path, help="Frozen bundle or pinned pysystemtrade checkout")
    parser.add_argument(
        "--measure", action="store_true", help="Run ten isolated repetitions per side"
    )
    parser.add_argument(
        "--worker",
        choices=tuple(dict.fromkeys(QUOTE_SIDES + FUTURES_SIDES + FUNDING_SIDES)),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    sides = {
        FUTURES_CASE: FUTURES_SIDES,
        NAUTILUS_FUTURES_CASE: FUNDING_SIDES,
        FUNDING_CASE: FUNDING_SIDES,
    }.get(args.case, QUOTE_SIDES)
    if args.worker:
        if args.worker not in sides:
            parser.error(f"{args.worker} is not a side of {args.case}")
        if args.worker.startswith("nautilus") and version("nautilus-trader") != NAUTILUS_VERSION:
            raise ValueError(f"Expected NautilusTrader {NAUTILUS_VERSION}")
        if args.case in TREND_CASES:
            if args.worker == "pysystemtrade":
                sys.path.insert(0, str(source))
            result = trend_worker(source, args.worker)
        elif args.case == FUNDING_CASE:
            result = funding_worker(source, args.worker)
        else:
            result = quote_worker(args.case, source, args.worker)
        print(json.dumps(result, allow_nan=False))
        return
    if args.case in TREND_CASES:
        verify_pysystemtrade_source(source)
    report = collect(args.case, source, sides, SAMPLE_COUNT if args.measure else 0)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for side, record in report["sides"].items():
        ledger = record["correctness"]["ledger"]
        extra = (
            f", {len(ledger['funding'])} funding payments"
            if args.case == FUNDING_CASE
            else f", {len(ledger['equity_cents'])} daily marks"
            if args.case in TREND_CASES
            else ""
        )
        print(
            f"{args.case} {side}: {len(ledger['fills'])} fills{extra} matched {record.get('medians', '')}"
        )


if __name__ == "__main__":
    main()
