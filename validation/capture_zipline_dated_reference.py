"""Actual published Zipline continuous selection and explicit expiry cleanup traces."""

import importlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pandas as pd

zipline: Any = importlib.import_module("zipline")
api: Any = importlib.import_module("zipline.api")
bundle_module: Any = importlib.import_module("zipline.data.bundles")
commission: Any = importlib.import_module("zipline.finance.commission")
slippage: Any = importlib.import_module("zipline.finance.slippage")
calendars: Any = importlib.import_module("zipline.utils.calendar_utils")
run_algorithm = cast(Any, zipline).run_algorithm
continuous_future = api.continuous_future
get_datetime = api.get_datetime
order = api.order
set_benchmark = api.set_benchmark
set_commission = api.set_commission
set_slippage = api.set_slippage
sid = api.sid
bundles = bundle_module.bundles
ingest = bundle_module.ingest
register = bundle_module.register
unregister = bundle_module.unregister
NoCommission = commission.NoCommission
NoSlippage = slippage.NoSlippage
get_calendar = calendars.get_calendar

assert version("zipline-reloaded") == "3.2.0"
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
calendar = get_calendar("us_futures")
days = calendar.sessions_in_range("2025-07-07", "2025-07-14")
name = "ml4t_dated_reference_20261022"
if name in bundles:
    unregister(name)


def prepare(_env, assets, _minute, daily, adjustments, _calendar, _start, _end, *_args):
    assets.write(
        futures=pd.DataFrame(
            {
                "symbol": ["TESTN25", "TESTU25"],
                "root_symbol": ["TEST"] * 2,
                "exchange": ["CME"] * 2,
                "start_date": [days[0]] * 2,
                "end_date": pd.to_datetime(["2025-07-15", "2025-09-15"]),
                "notice_date": pd.to_datetime(["2025-07-10", "2025-09-10"]),
                "expiration_date": pd.to_datetime(["2025-07-15", "2025-09-15"]),
                "auto_close_date": pd.to_datetime(["2025-07-10", "2025-09-10"]),
                "multiplier": [50.0] * 2,
                "tick_size": [0.25] * 2,
            },
            index=pd.Index([17, 23]),
        ),
        root_symbols=pd.DataFrame({"root_symbol": ["TEST"], "exchange": ["CME"]}),
        exchanges=pd.DataFrame(
            {"exchange": ["CME"], "canonical_name": ["CME"], "country_code": ["US"]}
        ),
    )
    frames = []
    for asset, offset, volumes in [
        (17, 0, [100, 40, 30, 20, 10, 10]),
        (23, 10, [50, 80, 90, 100, 110, 110]),
    ]:
        prices = [4000.0 + offset + i * 10 for i in range(len(days))]
        frame = pd.DataFrame(dict.fromkeys(["open", "high", "low", "close"], prices), index=days)
        frame["volume"] = volumes
        frames.append((asset, frame))
    daily.write(frames, show_progress=False)
    adjustments.write()


register(name, prepare, calendar_name="us_futures", start_session=days[0], end_session=days[-1])
ingest(name)
report: dict[str, Any] = {
    "package": "zipline-reloaded==3.2.0",
    "source_revision": "cf649ac06fddce76f6caa2403cd8494ac84f72ad",
    "wheel_sha256": "1d7a3c4f547c02e9a3c3cc1f8ea2394641124c1fdb35ef27cdf30fa488d3899b",
    "calendar": "us_futures",
    "timing_claim": False,
    "cases": [],
}


def capture(case):
    trace = []

    def initialize(context):
        set_benchmark(None)
        set_commission(us_futures=NoCommission())
        set_slippage(us_futures=NoSlippage())
        context.cf = continuous_future(
            "TEST", offset=0, roll=case if case != "expiry" else "calendar", adjustment=None
        )
        context.step = 0

    def handle(context, data):
        contract = data.current(context.cf, "contract")
        if context.step == 0:
            order(sid(17), 1)
        trace.append(
            {
                "timestamp": get_datetime().isoformat(),
                "selected_sid": int(contract),
                "continuous_price": float(data.current(context.cf, "price")),
                "cash": float(context.portfolio.cash),
                "nav": float(context.portfolio.portfolio_value),
                "positions": [
                    {
                        "sid": int(asset),
                        "quantity": float(p.amount),
                        "mark": float(p.last_sale_price),
                    }
                    for asset, p in context.portfolio.positions.items()
                ],
            }
        )
        context.step += 1

    perf = run_algorithm(
        start=days[0],
        end=days[-1],
        initialize=initialize,
        handle_data=handle,
        bundle=name,
        trading_calendar=calendar,
        data_frequency="daily",
        capital_base=100000,
    )
    report["cases"].append(
        {
            "name": case,
            "trace": trace,
            "cash": perf.ending_cash.tolist(),
            "nav": perf.portfolio_value.tolist(),
            "fills": [
                {
                    "timestamp": f["dt"].isoformat(),
                    "sid": int(f["sid"]),
                    "quantity": float(f["amount"]),
                    "price": float(f["price"]),
                }
                for transactions in perf.transactions
                for f in transactions
            ],
        }
    )
    print(case, [(r["timestamp"], r["selected_sid"], r["cash"]) for r in trace], flush=True)


for case in ["calendar", "volume", "expiry"]:
    capture(case)

(root / "reference.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
