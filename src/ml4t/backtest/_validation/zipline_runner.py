"""Shared Zipline helpers for internal validation workflows."""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ZiplineRunResult:
    """Standardized Zipline run artifacts for benchmark consumers."""

    final_value: float
    num_trades: int
    trades_df: pd.DataFrame | None
    positions_df: pd.DataFrame | None
    transactions_df: pd.DataFrame | None
    orders_df: pd.DataFrame | None
    setup_time_sec: float = 0.0


def load_zipline_modules() -> dict[str, Any]:
    """Import and return the Zipline modules used by the validation runner."""
    zipline_api = importlib.import_module("zipline.api")
    return {
        "run_algorithm": importlib.import_module("zipline").run_algorithm,
        "get_datetime": zipline_api.get_datetime,
        "order_target": zipline_api.order_target,
        "set_commission": zipline_api.set_commission,
        "set_max_leverage": zipline_api.set_max_leverage,
        "set_slippage": zipline_api.set_slippage,
        "sid": zipline_api.sid,
        "ingest": importlib.import_module("zipline.data.bundles").ingest,
        "register": importlib.import_module("zipline.data.bundles").register,
        "NoCommission": importlib.import_module("zipline.finance.commission").NoCommission,
        "PerDollar": importlib.import_module("zipline.finance.commission").PerDollar,
        "SlippageModel": importlib.import_module("zipline.finance.slippage").SlippageModel,
        "get_calendar": importlib.import_module("zipline.utils.calendar_utils").get_calendar,
    }


def normalize_target_lookup(
    target_lookup_raw: dict[pd.Timestamp, dict[str, float]],
) -> dict[pd.Timestamp, dict[str, float]]:
    """Normalize target timestamps to tz-naive daily lookup keys."""
    target_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for ts, targets in target_lookup_raw.items():
        ts_naive = ts.tz_convert(None) if ts.tz is not None else ts
        target_lookup[pd.Timestamp(ts_naive).normalize()] = targets
    return target_lookup


def flatten_result_column(results: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Flatten Zipline's list-valued result columns into a DataFrame."""
    records: list[dict[str, object]] = []
    if column_name not in results.columns:
        return pd.DataFrame()

    for dt, payload in results[column_name].items():
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict):
                record = dict(item)
            elif hasattr(item, "to_dict"):
                record = item.to_dict()
            elif hasattr(item, "items"):
                record = dict(item.items())
            else:
                continue
            record["dt"] = dt
            records.append(record)

    return pd.DataFrame(records)


def transactions_to_trade_log(transactions: pd.DataFrame) -> pd.DataFrame | None:
    """Convert Zipline transactions into completed round-trip trades."""
    if transactions.empty:
        return None

    transactions = transactions.copy()
    transactions["dt"] = pd.to_datetime(transactions["dt"])
    amount_col = "amount" if "amount" in transactions.columns else "filled"
    price_col = "price" if "price" in transactions.columns else "last_sale_price"
    symbol_col = (
        "symbol"
        if "symbol" in transactions.columns
        else "sid"
        if "sid" in transactions.columns
        else "asset"
        if "asset" in transactions.columns
        else None
    )
    if (
        symbol_col is None
        or amount_col not in transactions.columns
        or price_col not in transactions.columns
    ):
        return None

    trade_records: list[dict[str, object]] = []
    for symbol in transactions[symbol_col].dropna().unique():
        symbol_txns = transactions[transactions[symbol_col] == symbol].sort_values("dt")

        running_pos = 0.0
        entry_time = None
        entry_price = None
        entry_size = 0.0

        for _, row in symbol_txns.iterrows():
            amount = float(row[amount_col])
            price = float(row[price_col])
            prev_pos = running_pos
            running_pos += amount

            if prev_pos == 0 and running_pos != 0:
                entry_time = row["dt"]
                entry_price = price
                entry_size = amount
            elif prev_pos != 0 and running_pos == 0:
                assert entry_time is not None
                assert entry_price is not None
                pnl = (price - entry_price) * entry_size
                trade_records.append(
                    {
                        "entry_date": entry_time,
                        "exit_date": row["dt"],
                        "asset": str(symbol),
                        "side": "long" if entry_size > 0 else "short",
                        "quantity": abs(entry_size),
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                    }
                )
                entry_time = None
                entry_price = None
            elif prev_pos != 0 and running_pos != 0 and (prev_pos > 0) != (running_pos > 0):
                assert entry_time is not None
                assert entry_price is not None
                pnl = (price - entry_price) * entry_size
                trade_records.append(
                    {
                        "entry_date": entry_time,
                        "exit_date": row["dt"],
                        "asset": str(symbol),
                        "side": "long" if entry_size > 0 else "short",
                        "quantity": abs(entry_size),
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                    }
                )
                entry_time = row["dt"]
                entry_price = price
                entry_size = running_pos

    if not trade_records:
        return None
    return pd.DataFrame(trade_records).sort_values("entry_date").reset_index(drop=True)


def run_zipline_target_shares(
    *,
    modules: dict[str, Any],
    config: Any,
    price_data: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    target_lookup_raw: dict[pd.Timestamp, dict[str, float]],
    logger: Any | None = None,
) -> ZiplineRunResult:
    """Run a canonical target-share strategy through Zipline Reloaded."""
    log = logger or (lambda *_args, **_kwargs: None)

    run_algorithm = modules["run_algorithm"]
    get_datetime = modules["get_datetime"]
    order_target = modules["order_target"]
    set_commission = modules["set_commission"]
    set_max_leverage = modules["set_max_leverage"]
    set_slippage = modules["set_slippage"]
    sid = modules["sid"]
    ingest = modules["ingest"]
    register = modules["register"]
    NoCommission = modules["NoCommission"]
    PerDollar = modules["PerDollar"]
    SlippageModel = modules["SlippageModel"]
    get_calendar = modules["get_calendar"]

    nyse = get_calendar("XNYS")
    asset_names = sorted(price_data.keys())
    target_lookup = normalize_target_lookup(target_lookup_raw)

    first_df = price_data[asset_names[0]]
    start_date = first_df.index[0]
    end_date = first_df.index[-1]
    nyse_sessions = nyse.sessions_in_range(
        pd.Timestamp(start_date).tz_localize(None) if pd.Timestamp(start_date).tz else start_date,
        pd.Timestamp(end_date).tz_localize(None) if pd.Timestamp(end_date).tz else end_date,
    )

    class OpenPriceSlippage(SlippageModel):
        @staticmethod
        def process_order(data, order):
            open_px = data.current(order.asset, "open")
            if config.slippage_pct > 0:
                if order.amount > 0:
                    open_px = open_px * (1.0 + config.slippage_pct)
                elif order.amount < 0:
                    open_px = open_px * (1.0 - config.slippage_pct)
            return (open_px, order.amount)

    def make_multi_asset_ingest(price_data_dict, asset_list):
        def ingest_func(
            _environ,
            asset_db_writer,
            _minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            _cache,
            show_progress,
            _output_dir,
        ):
            sessions = calendar.sessions_in_range(start_session, end_session)
            sessions = pd.DatetimeIndex(sessions).tz_localize(None)

            equities_df = pd.DataFrame(
                {
                    "symbol": asset_list,
                    "asset_name": [f"Asset {name}" for name in asset_list],
                    "exchange": ["NYSE"] * len(asset_list),
                }
            )
            asset_db_writer.write(equities=equities_df)

            bar_data = []
            for asset_sid, asset_name in enumerate(asset_list):
                df = price_data_dict[asset_name].copy()
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                trading_df = (
                    df.reindex(sessions).ffill().bfill()[["open", "high", "low", "close", "volume"]]
                )
                if len(trading_df) > 0:
                    bar_data.append((asset_sid, trading_df))

            daily_bar_writer.write(bar_data, show_progress=show_progress)
            adjustment_writer.write()

        return ingest_func

    bundle_sig_input = (
        "|".join(asset_names) + f"|{pd.Timestamp(dates[0]).date()}|{pd.Timestamp(dates[-1]).date()}"
    )
    bundle_sig = hashlib.md5(bundle_sig_input.encode("utf-8")).hexdigest()[:10]
    bundle_name = f"bench_multi_{len(asset_names)}_{config.n_bars}_{bundle_sig}"
    start_session = nyse_sessions[0]
    end_session = (
        nyse_sessions[-1]
        if len(nyse_sessions) <= config.n_bars
        else nyse_sessions[config.n_bars - 1]
    )

    zipline_root = Path(os.environ.get("ZIPLINE_ROOT", Path.home() / ".zipline"))
    bundle_dir = zipline_root / "data" / bundle_name
    bundle_exists = False
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        bundle_runs = sorted(path for path in bundle_dir.iterdir() if path.is_dir())
        if bundle_runs:
            latest_run = bundle_runs[-1]
            assets_ok = any(latest_run.glob("assets-*.sqlite"))
            if assets_ok:
                bundle_exists = True
            else:
                shutil.rmtree(bundle_dir, ignore_errors=True)

    setup_time_sec = 0.0
    if not bundle_exists:
        bundle_start = time.perf_counter()
        register(
            bundle_name,
            make_multi_asset_ingest(price_data, asset_names),
            calendar_name="XNYS",
            start_session=start_session,
            end_session=end_session,
        )
        ingest(bundle_name, show_progress=False)
        setup_time_sec = time.perf_counter() - bundle_start
        log(f"  Bundle creation: {setup_time_sec:.1f}s (one-time setup)")
    else:
        log(f"  Using cached bundle: {bundle_name}")
        register(
            bundle_name,
            make_multi_asset_ingest(price_data, asset_names),
            calendar_name="XNYS",
            start_session=start_session,
            end_session=end_session,
        )

    algo_state = {
        "target_lookup": target_lookup,
        "asset_names": asset_names,
    }

    def initialize(context):
        context.state = algo_state
        context.assets = [sid(i) for i in range(len(context.state["asset_names"]))]
        context.asset_map = {
            name: context.assets[i] for i, name in enumerate(context.state["asset_names"])
        }
        context.name_by_asset = {asset: name for name, asset in context.asset_map.items()}
        if config.commission_pct > 0:
            set_commission(PerDollar(cost=config.commission_pct))
        else:
            set_commission(NoCommission())
        if config.zipline_max_leverage is not None:
            set_max_leverage(config.zipline_max_leverage)
        set_slippage(OpenPriceSlippage())

    def handle_data(context, data):
        dt = get_datetime()
        dt_naive = dt.tz_convert(None) if dt.tz else dt
        dt_normalized = dt_naive.normalize()

        targets = context.state["target_lookup"].get(dt_normalized, {})
        active_names = set(targets.keys())
        for asset_obj, position in context.portfolio.positions.items():
            if position.amount != 0:
                asset_name = context.name_by_asset.get(asset_obj)
                if asset_name is not None:
                    active_names.add(asset_name)

        for asset_name in sorted(active_names):
            asset = context.asset_map[asset_name]
            if not data.can_trade(asset):
                continue

            current_pos = context.portfolio.positions[asset].amount
            target = targets.get(asset_name, 0.0)
            if current_pos != target:
                order_target(asset, target)

    results = run_algorithm(
        start=start_session,
        end=end_session,
        initialize=initialize,
        handle_data=handle_data,
        capital_base=config.initial_cash,
        bundle=bundle_name,
        data_frequency="daily",
    )

    positions = flatten_result_column(results, "positions")
    transactions = flatten_result_column(results, "transactions")
    orders = flatten_result_column(results, "orders")
    trades_df = transactions_to_trade_log(transactions)
    num_trades = (
        len(trades_df) if trades_df is not None else int(len(orders)) if not orders.empty else 0
    )

    return ZiplineRunResult(
        final_value=float(results["portfolio_value"].iloc[-1]),
        num_trades=num_trades,
        trades_df=trades_df,
        positions_df=positions if not positions.empty else None,
        transactions_df=transactions if not transactions.empty else None,
        orders_df=orders if not orders.empty else None,
        setup_time_sec=setup_time_sec,
    )
