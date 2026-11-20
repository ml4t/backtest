"""Dated futures trend and roll comparison against the pinned pysystemtrade source."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import pandas as pd
import polars as pl
from nautilus_dated_futures import run_nautilus_trend
from reference_workloads import CASH, verify

ROOTS = ("SP500", "GOLD")
START, END = "2022-01-20", "2022-03-25"
ROLL = pd.Timestamp("2022-03-01")
SIGNAL_START = "2020-01-01"


@dataclass
class TrendInput:
    dates: pd.DatetimeIndex
    signal_prices: dict[str, pd.Series]
    prices: dict[str, pd.Series]
    legs: dict[str, tuple[str, str]]
    multipliers: dict[str, int]
    fees: dict[str, float]
    source_files: dict[str, str]


def load_trend(source: Path) -> TrendInput:
    files: dict[str, str] = {}

    def read(relative: str, **kwargs: Any) -> pd.DataFrame:
        file = source / "data" / "futures" / relative
        files[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
        return pd.read_csv(file, **kwargs)

    instruments = read("csvconfig/instrumentconfig.csv", index_col="Instrument")
    date_args = {"index_col": "DATETIME", "parse_dates": ["DATETIME"]}

    def daily(frame: pd.DataFrame) -> pd.DataFrame:
        # Keep the last complete source row, so roll legs share one physical observation.
        days = pd.DatetimeIndex(pd.Timestamp(stamp).normalize() for stamp in frame.index)
        selected = frame.groupby(days).tail(1)
        return selected.set_axis(
            pd.DatetimeIndex(pd.Timestamp(stamp).normalize() for stamp in selected.index)
        )

    raw = {
        root: daily(read(f"multiple_prices_csv/{root}.csv", **date_args).loc[START:END])
        for root in ROOTS
    }
    dates = raw[ROOTS[0]].index.intersection(raw[ROOTS[1]].index).sort_values()
    signals, prices, legs, multipliers, fees = {}, {}, {}, {}, {}
    if ROLL not in dates:
        raise ValueError("Both markets must have observations on the scheduled roll")
    for root, frame in raw.items():
        frame = frame.loc[dates]
        adjusted = daily(read(f"adjusted_prices_csv/{root}.csv", **date_args).loc[SIGNAL_START:END])
        signals[root] = adjusted["price"].dropna()
        if not dates.isin(signals[root].index).all():
            raise ValueError(f"{root}: missing signal observation on an execution date")
        old_id = int(frame.loc[ROLL, "PRICE_CONTRACT"])
        new_id = int(frame.loc[ROLL, "FORWARD_CONTRACT"])
        if old_id == new_id:
            raise ValueError(f"{root}: distinct contract IDs required")
        legs[root] = (f"{root}_{old_id}", f"{root}_{new_id}")
        for asset, contract, active in (
            (legs[root][0], old_id, dates <= ROLL),
            (legs[root][1], new_id, dates >= ROLL),
        ):
            observed = (
                frame["PRICE"]
                .where(contract == frame.PRICE_CONTRACT)
                .combine_first(frame["FORWARD"].where(contract == frame.FORWARD_CONTRACT))
            )
            selected = observed.loc[active]
            if selected.isna().any() or (selected <= 0).any():
                raise ValueError(f"{asset}: missing raw contract mark in its execution interval")
            prices[asset] = selected
            multipliers[asset] = int(instruments.loc[root, "Pointsize"])
            fees[asset] = float(instruments.loc[root, "PerBlock"])
    return TrendInput(dates, signals, prices, legs, multipliers, fees, files)


def forecasts(inputs: TrendInput, side: str) -> dict[str, pd.Series]:
    if side == "pysystemtrade":
        ewmac = import_module("systems.provided.rules.ewmac").ewmac_forecast_with_defaults
        return {
            root: ewmac(price, Lfast=16, Lslow=64) for root, price in inputs.signal_prices.items()
        }

    def forecast(price: pd.Series) -> pd.Series:
        # The same public EWMAC rule expressed with ML4T's Polars data stack.
        x = pl.col("price")
        frame = pl.DataFrame({"price": price.to_numpy()}).with_columns(
            x.diff().ewm_std(span=35, min_samples=10).clip(lower_bound=1e-10).alias("vol")
        )
        value = frame.select(
            (
                (x.ewm_mean(span=16) - x.ewm_mean(span=64))
                / pl.max_horizontal(
                    "vol",
                    pl.col("vol")
                    .rolling_quantile(
                        0.05, interpolation="linear", window_size=500, min_samples=100
                    )
                    .forward_fill(),
                )
            ).alias("forecast")
        )["forecast"]
        return pd.Series(value.to_numpy(), index=price.index)

    return {root: forecast(price) for root, price in inputs.signal_prices.items()}


def positions(inputs: TrendInput, signal: dict[str, pd.Series]) -> dict[str, pd.Series]:
    out = {}
    for root, (old, new) in inputs.legs.items():
        # The signal is from the previous observed session. Hold size constant on the roll.
        lag = signal[root].shift().reindex(inputs.dates)
        if lag.isna().any():
            raise ValueError(f"{root}: signal warmup incomplete")
        target = ((lag > 0).astype(int) - (lag < 0).astype(int)) * 2
        target.loc[ROLL] = target.shift().loc[ROLL]
        target.iloc[0] = 0
        out[old] = target.where(target.index < ROLL, 0).loc[inputs.prices[old].index]
        out[new] = target.loc[inputs.prices[new].index]
        if not target.loc[ROLL]:
            raise ValueError(f"{root}: test must carry exposure through the roll")
    return out


def expected_trend(inputs: TrendInput, targets: dict[str, pd.Series]) -> dict:
    cash = float(CASH)
    fills, equity = [], []
    held = dict.fromkeys(targets, 0)
    marks: dict[str, float] = {}
    for stamp in inputs.dates:
        for asset, prices in inputs.prices.items():
            if stamp not in prices.index:
                continue
            px = float(prices.loc[stamp])
            prior = held[asset]
            qty = int(targets[asset].loc[stamp]) - prior
            cash += prior * (px - marks.get(asset, px)) * inputs.multipliers[asset]
            marks[asset] = px
            if qty:
                fee = abs(qty) * inputs.fees[asset]
                cash -= fee
                fills.append((stamp.value, asset, qty, round(px * 100), round(fee * 100)))
            held[asset] += qty
        equity.append(round(cash * 100))
    return {"fills": sorted(fills), "positions": held, "equity_cents": equity}


def run_pysystemtrade_trend(inputs: TrendInput, targets: dict[str, pd.Series]) -> dict:
    accounting = import_module("systems.accounts.pandl_calculators.pandl_cash_costs")
    instruments = import_module("sysobjects.instruments")
    pnl, fills = {}, []
    for asset, prices in inputs.prices.items():
        target = targets[asset]
        if target.index[0] > inputs.dates[0]:
            # Native position differences require an initial flat observation.
            target = target.reindex(inputs.dates[inputs.dates <= target.index[-1]], fill_value=0)
        calc = accounting.pandlCalculationWithCashCostsAndFills(
            price=prices,
            positions=target,
            value_per_point=inputs.multipliers[asset],
            raw_costs=instruments.instrumentCosts(value_of_block_commission=inputs.fees[asset]),
            rolls_per_year=0,
            vol_normalise_currency_costs=False,
            roundpositions=False,
            delayfill=False,
        )
        pnl[asset] = calc.net_pandl_in_instrument_currency()
        costs = calc.costs_pandl_in_instrument_currency()
        fills.extend(
            (f.date.value, asset, int(f.qty), round(f.price * 100), round(-costs.loc[f.date] * 100))
            for f in calc.fills
        )
    return {
        "fills": sorted(fills),
        "positions": {
            asset: sum(qty for _, symbol, qty, _, _ in fills if symbol == asset)
            for asset in targets
        },
        "equity_cents": [round(x * 100) for x in CASH + pd.DataFrame(pnl).sum(axis=1).cumsum()],
    }


def run_ml4t_trend(inputs: TrendInput, targets: dict[str, pd.Series]) -> dict:
    from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy
    from ml4t.backtest.config import ExecutionMode, ExecutionPrice

    by_time = {stamp.to_pydatetime(): {} for stamp in inputs.dates}
    for asset, series in targets.items():
        for stamp, qty in zip(pd.DatetimeIndex(series.index), series, strict=True):
            by_time[stamp.to_pydatetime()][asset] = int(qty)

    class DatedTargets(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == ROLL.to_pydatetime():
                for root, (old, new) in inputs.legs.items():
                    broker.roll_position(old, new, roll_id=root)
                return
            for asset, wanted in by_time[timestamp].items():
                position = broker.get_position(asset)
                delta = wanted - (position.quantity if position else 0)
                if delta:
                    broker.submit_order(asset, delta)

    rows = [
        (stamp.to_pydatetime(), asset, price)
        for asset, series in inputs.prices.items()
        for stamp, price in zip(pd.DatetimeIndex(series.index), series, strict=True)
    ]
    prices = pl.DataFrame(rows, schema=["timestamp", "asset", "close"], orient="row").with_columns(
        *(pl.col("close").alias(name) for name in ("open", "high", "low")),
        pl.lit(100_000).alias("volume"),
    )

    class ContractFees:
        def calculate(self, asset: str, quantity: float, price: float) -> float:
            del price
            return abs(quantity) * inputs.fees[asset]

    engine = Engine(
        DataFeed(prices_df=prices),
        DatedTargets(),
        BacktestConfig(
            initial_cash=CASH,
            calendar=None,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.CLOSE,
        ),
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=inputs.multipliers[asset],
                margin=5000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset in inputs.prices
        },
        commission_model=ContractFees(),
    )
    result = engine.run()
    rolls = result.to_rolls_dataframe()
    if rolls.height != 2 * len(ROOTS) or set(rolls["roll_state"]) != {"completed"}:
        raise ValueError("Backtest did not complete both legs of every roll")
    state = result.to_portfolio_state_dataframe()
    equity = [round(x * 100) for x in state["equity"]]
    if [round(x * 100) for x in state["cash"]] != equity:
        raise ValueError("Backtest did not settle the futures cash and equity at each mark")
    return {
        "fills": sorted(
            (
                pd.Timestamp(f.timestamp).value,
                f.asset,
                int(f.quantity) * (1 if f.side.value == "buy" else -1),
                round(f.price * 100),
                round(f.commission * 100),
            )
            for f in result.fills
        ),
        "positions": {
            asset: int(p.quantity) if (p := engine.broker.get_position(asset)) else 0
            for asset in inputs.prices
        },
        "equity_cents": equity,
    }


def trend_worker(source: Path, side: str) -> dict:
    inputs = load_trend(source)
    start_wall, start_cpu = perf_counter(), process_time()
    signal = forecasts(inputs, side)
    targets = positions(inputs, signal)
    forecast_wall = perf_counter() - start_wall
    extra = {}
    if side == "nautilus":
        actual, cash = run_nautilus_trend(inputs, targets)
        extra["observed_cash_cents"] = cash
    elif side == "pysystemtrade":
        actual = run_pysystemtrade_trend(inputs, targets)
    else:
        actual = run_ml4t_trend(inputs, targets)
        extra["observed_cash_cents"] = actual["equity_cents"]
    timing = {
        "run_wall_seconds": perf_counter() - start_wall,
        "run_cpu_seconds": process_time() - start_cpu,
        "forecast_wall_seconds": forecast_wall,
    }
    verify(actual, expected_trend(inputs, targets))
    return {
        "input_sha256": hashlib.sha256(
            json.dumps(inputs.source_files, sort_keys=True).encode()
        ).hexdigest(),
        "source_files": inputs.source_files,
        **extra,
        "rows": len(inputs.dates),
        "ledger": actual,
        "forecast": {
            root: [round(float(x), 8) for x in values.shift().loc[inputs.dates]]
            for root, values in signal.items()
        },
        "timing": timing,
    }
