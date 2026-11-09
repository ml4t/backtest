#!/usr/bin/env python3
"""Run optional native NautilusTrader and pysystemtrade comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import distribution, distributions, version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import pandas as pd

CASH = 1_000_000
FUTURES_FEE = 2
VALUE_PER_POINT = 50
NAUTILUS_VERSION = "2.0.0rc8"
PYSYSTEMTRADE_VERSION = "1.8.3"
PYSYSTEMTRADE_COMMIT = "bbf5cf823e16be1cca27493a94099eb8cf33f6e7"
SAMPLE_COUNT = 10


@dataclass(frozen=True)
class Workload:
    reference: str
    interval: str
    assets: int
    bars: int


WORKLOADS = {
    "nautilus-daily": Workload("nautilus", "DAY", 50, 252),
    "nautilus-minute": Workload("nautilus", "MINUTE", 20, 1950),
    "pysystemtrade-futures": Workload("pysystemtrade", "DAY", 12, 504),
}


@dataclass(frozen=True)
class Tape:
    index: pd.DatetimeIndex
    prices: dict[str, list[int]]
    targets: dict[str, list[int]]
    futures: bool


def make_tape(workload: Workload) -> Tape:
    """Generate deterministic cent prices and targets known before the close."""
    index = pd.date_range(
        "2022-01-03 16:00",
        periods=workload.bars,
        freq="B" if workload.interval == "DAY" else "min",
        tz="UTC",
    )
    futures = workload.reference == "pysystemtrade"
    prices, targets = {}, {}
    for asset in range(workload.assets):
        symbol = f"S{asset:03}"
        prices[symbol] = [
            (10_000 if futures else 5_000) + 81 * asset + (41 * i + 13 * i * asset) % 907
            for i in range(workload.bars)
        ]
        sequence = (0, 2, 1, -2, -1, 0) if futures else (0, 5, 2, 8, 1, 0)
        targets[symbol] = [
            0 if not i else sequence[((i - 1) // (7 + asset % 5) + asset + 1) % len(sequence)]
            for i in range(workload.bars)
        ]
    return Tape(index, prices, targets, futures)


def expected_ledger(tape: Tape) -> dict[str, Any]:
    """Independent integer-cent ledger for fills, final cash, and marks."""
    cash = CASH * 100
    multiplier = VALUE_PER_POINT if tape.futures else 1
    positions = dict.fromkeys(tape.prices, 0)
    fills = []
    equity = []
    for i in range(len(tape.index)):
        inventory = 0
        for symbol, prices in tape.prices.items():
            price, position = prices[i], positions[symbol]
            target = tape.targets[symbol][i]
            change = target - position
            if tape.futures and i:
                cash += position * (price - prices[i - 1]) * multiplier
            if change:
                fee = abs(change) * FUTURES_FEE * 100 if tape.futures else 0
                cash -= fee if tape.futures else change * price
                fills.append((i, symbol, change, price, fee))
            positions[symbol] = target
            inventory += target * price if not tape.futures else 0
        equity.append(cash + inventory)
    result = {"fills": fills, "positions": positions}
    if tape.futures:
        result["equity_cents"] = equity
    else:
        result["cash_cents"] = cash
    return result


def run_ml4t(tape: Tape) -> tuple[dict[str, Any], float]:
    import polars as pl

    from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
    from ml4t.backtest.config import CommissionType, ExecutionMode, ExecutionPrice

    timestamps = tape.index.to_pydatetime().tolist()
    row_by_time = {stamp: i for i, stamp in enumerate(timestamps)}
    prices = pl.DataFrame(
        {
            "timestamp": timestamps * len(tape.prices),
            "asset": [symbol for symbol in tape.prices for _ in timestamps],
            "open": [(price - 2) / 100 for series in tape.prices.values() for price in series],
            "high": [(price + 3) / 100 for series in tape.prices.values() for price in series],
            "low": [(price - 4) / 100 for series in tape.prices.values() for price in series],
            "close": [price / 100 for series in tape.prices.values() for price in series],
            "volume": [10_000] * (len(timestamps) * len(tape.prices)),
        }
    )

    class Targets(Strategy):
        def on_data(self, timestamp, data, context, broker):
            i = row_by_time[timestamp]
            for symbol, targets in tape.targets.items():
                position = broker.get_position(symbol)
                change = targets[i] - (position.quantity if position else 0)
                if change:
                    broker.submit_order(symbol, change)

    contracts = (
        {
            symbol: ContractSpec(
                symbol,
                asset_class=AssetClass.FUTURE,
                multiplier=VALUE_PER_POINT,
                margin=1000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for symbol in tape.prices
        }
        if tape.futures
        else None
    )
    engine = Engine(
        DataFeed(prices_df=prices),
        Targets(),
        BacktestConfig(
            initial_cash=CASH,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.CLOSE,
            allow_short_selling=tape.futures,
            allow_leverage=tape.futures,
            commission_type=CommissionType.PER_SHARE if tape.futures else CommissionType.NONE,
            commission_per_share=FUTURES_FEE if tape.futures else 0,
        ),
        contract_specs=contracts,
    )
    started = perf_counter()
    result = engine.run()
    seconds = perf_counter() - started
    native = result.to_portfolio_state_dataframe()
    fills = [
        (
            row_by_time[fill.timestamp],
            fill.asset,
            fill.quantity if fill.side.value == "buy" else -fill.quantity,
            round(fill.price * 100),
            round(fill.commission * 100),
        )
        for fill in result.fills
    ]
    positions = engine.broker.get_positions()
    ledger = {
        "fills": sorted(fills),
        "positions": {
            symbol: positions[symbol].quantity if symbol in positions else 0
            for symbol in tape.prices
        },
    }
    if tape.futures:
        equity = [round(x * 100) for x in native["equity"]]
        if [round(x * 100) for x in native["cash"]] != equity:
            raise ValueError("Backtest cash and equity differ on the daily settled workload")
        ledger["equity_cents"] = equity
    else:
        ledger["cash_cents"] = round(native["cash"][-1] * 100)
    return ledger, seconds


def run_nautilus(tape: Tape, interval: str) -> tuple[dict[str, Any], float]:
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import (
        AccountType,
        Bar,
        BarType,
        Currency,
        Equity,
        InstrumentId,
        Money,
        OmsType,
        OrderSide,
        Price,
        Quantity,
        Symbol,
        Venue,
    )
    from nautilus_trader.trading import Strategy

    usd = Currency.from_str("USD")
    venue = Venue("XNAS")
    instruments = {
        symbol: Equity(
            InstrumentId(Symbol(symbol), venue),
            Symbol(symbol),
            usd,
            2,
            Price(0.01, 2),
            0,
            0,
            lot_size=Quantity.from_int(1),
        )
        for symbol in tape.prices
    }
    bar_types = {
        symbol: BarType.from_str(f"{instrument.id}-1-{interval}-LAST-EXTERNAL")
        for symbol, instrument in instruments.items()
    }
    timestamps = [stamp.value for stamp in tape.index]
    row_by_time = {stamp: i for i, stamp in enumerate(timestamps)}

    class Targets(Strategy):
        def on_start(self):
            for bar_type in bar_types.values():
                self.subscribe_bars(bar_type)

        def on_bar(self, bar):
            instrument_id = bar.bar_type.instrument_id
            symbol = str(instrument_id.symbol)
            target = tape.targets[symbol][row_by_time[bar.ts_event]]
            delta = target - int(self.portfolio.net_position(instrument_id))
            if delta:
                self.submit_order(
                    self.order_factory.market(
                        instrument_id,
                        OrderSide.BUY if delta > 0 else OrderSide.SELL,
                        Quantity.from_int(abs(delta)),
                    )
                )

    engine = BacktestEngine(
        BacktestEngineConfig(logging=LoggerConfig(bypass_logging=True), run_analysis=False)
    )
    try:
        engine.add_venue(
            venue,
            OmsType.NETTING,
            AccountType.CASH,
            [Money(CASH, usd)],
            fee_model=FixedFeeModel(Money(0, usd)),
            bar_execution=True,
        )
        for instrument in instruments.values():
            engine.add_instrument(instrument)
        engine.add_data(
            [
                Bar(
                    bar_types[symbol],
                    Price((close - 2) / 100, 2),
                    Price((close + 3) / 100, 2),
                    Price((close - 4) / 100, 2),
                    Price(close / 100, 2),
                    Quantity.from_int(10_000),
                    timestamps[i],
                    timestamps[i],
                )
                for i in range(len(timestamps))
                for symbol, prices in tape.prices.items()
                for close in (prices[i],)
            ]
        )
        engine.add_strategy(Targets())
        started = perf_counter()
        engine.run()
        seconds = perf_counter() - started
        fills = engine.generate_fills_report()
        records = [
            (
                row_by_time[pd.Timestamp(row.ts_event).value],
                str(row.instrument_id).split(".")[0],
                float(row.last_qty) * (1 if row.order_side == "BUY" else -1),
                round(float(row.last_px) * 100),
                round(float(str(row.commission).split()[0]) * 100),
            )
            for row in fills.itertuples()
        ]
        account = engine.cache.account_for_venue(venue)
        return {
            "fills": sorted(records),
            "positions": {
                symbol: float(engine.portfolio.net_position(instrument.id))
                for symbol, instrument in instruments.items()
            },
            "cash_cents": round(float(account.balance_total(usd)) * 100),
        }, seconds
    finally:
        engine.dispose()


def run_pysystemtrade(tape: Tape) -> tuple[dict[str, Any], float]:
    from sysobjects.instruments import instrumentCosts
    from systems.accounts.pandl_calculators.pandl_cash_costs import (
        pandlCalculationWithCashCostsAndFills,
    )

    calculations = {
        symbol: pandlCalculationWithCashCostsAndFills(
            price=pd.Series([price / 100 for price in prices], index=tape.index),
            positions=pd.Series(tape.targets[symbol], index=tape.index),
            value_per_point=VALUE_PER_POINT,
            raw_costs=instrumentCosts(value_of_block_commission=FUTURES_FEE),
            rolls_per_year=0,
            vol_normalise_currency_costs=False,
            roundpositions=False,
            delayfill=False,
        )
        for symbol, prices in tape.prices.items()
    }
    started = perf_counter()
    native = {
        symbol: (
            calc.net_pandl_in_instrument_currency(),
            calc.fills,
            calc.costs_pandl_in_instrument_currency(),
        )
        for symbol, calc in calculations.items()
    }
    seconds = perf_counter() - started
    pnl = pd.DataFrame({symbol: outputs[0] for symbol, outputs in native.items()})
    if not pnl.index.equals(tape.index) or pnl.isna().any().any():
        raise ValueError("pysystemtrade did not return a complete aligned P&L series")
    equity = [round(x * 100) for x in (CASH + pnl.sum(axis=1).cumsum())]
    row_by_time = {stamp: i for i, stamp in enumerate(tape.index)}
    return {
        "fills": sorted(
            (
                row_by_time[f.date],
                symbol,
                float(f.qty),
                round(f.price * 100),
                round(-costs.loc[f.date] * 100),
            )
            for symbol, (_, fills, costs) in native.items()
            for f in fills
        ),
        "positions": {
            symbol: float(calc.positions.iloc[-1]) for symbol, calc in calculations.items()
        },
        "equity_cents": equity,
    }, seconds


def verify(actual: dict, expected: dict) -> None:
    for field, wanted in expected.items():
        observed = actual[field]
        if observed != wanted:
            if isinstance(wanted, list):
                first = next((i for i, (a, b) in enumerate(zip(observed, wanted)) if a != b), None)
                if first is not None:
                    raise ValueError(
                        f"{field}[{first}]: actual {observed[first]!r}; expected {wanted[first]!r}"
                    )
                raise ValueError(f"{field}: actual length {len(observed)}; expected {len(wanted)}")
            raise ValueError(f"{field}: actual {observed!r}; expected {wanted!r}")


def worker(workload: Workload, side: str) -> dict:
    tape = make_tape(workload)
    if side == "ml4t":
        package = "ml4t-backtest"
        ledger, elapsed = run_ml4t(tape)
    elif workload.reference == "nautilus":
        package = "nautilus-trader"
        if version(package) != NAUTILUS_VERSION:
            raise ValueError(f"Expected {package} {NAUTILUS_VERSION}; found {version(package)}")
        ledger, elapsed = run_nautilus(tape, workload.interval)
    else:
        package = "pysystemtrade"
        if version(package) != PYSYSTEMTRADE_VERSION:
            raise ValueError(
                f"Expected {package} {PYSYSTEMTRADE_VERSION}; found {version(package)}"
            )
        direct_url = distribution(package).read_text("direct_url.json")
        provenance = json.loads(direct_url) if direct_url else {}
        if provenance.get("vcs_info", {}).get("commit_id") != PYSYSTEMTRADE_COMMIT:
            raise ValueError(f"Install pysystemtrade from Git commit {PYSYSTEMTRADE_COMMIT}")
        ledger, elapsed = run_pysystemtrade(tape)
    verify(ledger, expected_ledger(tape))
    input_sha256 = hashlib.sha256(
        json.dumps(
            {
                "timestamps": [ts.isoformat() for ts in tape.index],
                "prices_cents": tape.prices,
                "target_quantities": tape.targets,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "input_sha256": input_sha256,
        "package": package,
        "version": version(package),
        "native_seconds": elapsed,
        "ledger": ledger,
    }


def collect(case: str, samples: int) -> dict:
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        POLARS_MAX_THREADS="1",
        PYTHONHASHSEED="0",
    )
    sides: dict[str, dict] = {}

    def load_average() -> tuple[float, float, float] | None:
        return os.getloadavg() if hasattr(os, "getloadavg") else None

    def run_child(side: str) -> tuple[dict, float]:
        started = perf_counter()
        output = subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), case, "--worker", side],
            env=env,
            text=True,
            timeout=300,
        )
        elapsed = perf_counter() - started
        return json.loads(output), elapsed

    for side in ("reference", "ml4t"):
        evidence, _ = run_child(side)
        sides[side] = {"correctness": evidence, "measured": []}
    if (
        sides["reference"]["correctness"]["input_sha256"]
        != sides["ml4t"]["correctness"]["input_sha256"]
    ):
        raise ValueError("The two engines did not run on the same input")

    for i in range(samples):
        order = ("reference", "ml4t") if i % 2 == 0 else ("ml4t", "reference")
        for side in order:
            load_before = load_average()
            evidence, wall_seconds = run_child(side)
            load_after = load_average()
            previous = sides[side]["correctness"]
            if evidence["input_sha256"] != previous["input_sha256"]:
                raise ValueError(f"Input changed during the {side} repetitions")
            verify(evidence["ledger"], previous["ledger"])
            sides[side]["measured"].append(
                {
                    "native_seconds": evidence["native_seconds"],
                    "process_seconds": wall_seconds,
                    "load_before": load_before,
                    "load_after": load_after,
                }
            )
    for evidence in sides.values():
        if samples:
            evidence["medians"] = {
                field: median(row[field] for row in evidence["measured"])
                for field in ("native_seconds", "process_seconds")
            }
    return {
        "case": case,
        "workload": asdict(WORKLOADS[case]),
        "logical_cpus": os.cpu_count(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in sorted(distributions(), key=lambda d: d.metadata["Name"].lower())
        },
        "python": sys.version,
        "platform": platform.platform(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": samples,
        "sides": sides,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=WORKLOADS)
    parser.add_argument("--worker", choices=("reference", "ml4t"), help=argparse.SUPPRESS)
    parser.add_argument("--measure", action="store_true", help="Run ten isolated timed pairs.")
    parser.add_argument(
        "--output", type=Path, help="Write full ledger and sample evidence to JSON."
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(WORKLOADS[args.case], args.worker), allow_nan=False))
        return
    report = collect(args.case, SAMPLE_COUNT if args.measure else 0)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for side, evidence in report["sides"].items():
        result = evidence["correctness"]
        print(
            f"{args.case} {side}: {result['package']} {result['version']}, "
            f"{len(result['ledger']['fills'])} fills matched"
            f"{', ' + str(evidence['medians']) if args.measure else ''}"
        )


if __name__ == "__main__":
    main()
