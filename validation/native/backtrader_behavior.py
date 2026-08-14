#!/usr/bin/env python3
"""Measure frozen Backtrader behavior without importing ML4T comparison code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="invalid escape sequence.*", category=SyntaxWarning)
    bt: Any = importlib.import_module("backtrader")

VERSION = "1.9.78.123"
ARTIFACT = "backtrader-1.9.78.123-py2.py3-none-any.whl"
ARTIFACT_SHA256 = "9a07a516b0de9155539a35c56e9404d8711dd7020b3d37b30495e83e1b9d5dfd"
SOURCE = "https://pypi.org/project/backtrader/1.9.78.123/"


def _frame(
    opens: list[float],
    *,
    closes: list[float] | None = None,
    lows: list[float] | None = None,
    highs: list[float] | None = None,
    dates: list[str] | None = None,
) -> pd.DataFrame:
    closes = closes or opens
    lows = lows or [min(open_, close) for open_, close in zip(opens, closes, strict=True)]
    highs = highs or [max(open_, close) for open_, close in zip(opens, closes, strict=True)]
    index = (
        pd.to_datetime(dates)
        if dates is not None
        else pd.date_range("2024-01-01", periods=len(opens))
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1_000.0},
        index=index,
    )


def _feed(frame: pd.DataFrame, name: str = "asset") -> Any:
    return bt.feeds.PandasData(dataname=frame, name=name)


def _run(
    strategy: type,
    *frames: tuple[str, pd.DataFrame],
    cash: float = 1_000.0,
    configure: Any = None,
    **strategy_kwargs: Any,
) -> tuple[Any, Any]:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    if configure is not None:
        configure(cerebro.broker)
    for name, frame in frames:
        cerebro.adddata(_feed(frame, name))
    cerebro.addstrategy(strategy, **strategy_kwargs)
    instance = cerebro.run()[0]
    return cerebro, instance


def _terminal(order: Any) -> bool:
    return order.status in {order.Completed, order.Margin, order.Rejected, order.Canceled}


def _defaults() -> dict[str, Any]:
    broker = bt.Cerebro().broker
    commission = bt.CommInfoBase()
    return {
        "cash": broker.p.cash,
        "checksubmit": broker.p.checksubmit,
        "cheat_on_close": broker.p.coc,
        "cheat_on_open": broker.p.coo,
        "commission": commission.p.commission,
        "leverage": commission.p.leverage,
        "short_cash": broker.p.shortcash,
        "slippage_fixed": broker.p.slip_fixed,
        "slippage_percent": broker.p.slip_perc,
    }


def _timing(coc: bool = False) -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def next(self) -> None:
            if len(self) == 1:
                self.buy(size=1)

        def notify_order(self, order: Any) -> None:
            if order.status == order.Completed:
                self.events.append(
                    {
                        "date": bt.num2date(order.executed.dt).date().isoformat(),
                        "price": order.executed.price,
                        "size": order.executed.size,
                    }
                )

    def configure(broker: Any) -> None:
        broker.set_coc(coc)

    _, strategy = _run(
        Strategy,
        ("asset", _frame([100.0, 110.0, 120.0], closes=[100.0, 105.0, 120.0])),
        configure=configure,
    )
    return strategy.events


def _target_percent(target: float, commission: float = 0.0) -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def next(self) -> None:
            if len(self) == 1:
                self.order_target_percent(target=target)

        def notify_order(self, order: Any) -> None:
            if _terminal(order):
                self.events.append(
                    {
                        "status": order.getstatusname(),
                        "created_size": order.created.size,
                        "executed_size": order.executed.size,
                        "price": order.executed.price,
                        "commission": order.executed.comm,
                    }
                )

    def configure(broker: Any) -> None:
        broker.setcommission(commission=commission)

    _, strategy = _run(
        Strategy,
        ("asset", _frame([100.0, 100.0])),
        configure=configure,
    )
    return strategy.events


def _cash_and_margin(leverage: float) -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def next(self) -> None:
            if len(self) == 1:
                self.buy(size=20)

        def notify_order(self, order: Any) -> None:
            if _terminal(order):
                self.events.append({"status": order.getstatusname(), "size": order.executed.size})

    def configure(broker: Any) -> None:
        broker.setcommission(leverage=leverage)

    _, strategy = _run(
        Strategy,
        ("asset", _frame([100.0, 100.0])),
        configure=configure,
    )
    return strategy.events


def _submission_sequence(reverse: bool) -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def next(self) -> None:
            if len(self) == 1:
                data_sequence = list(reversed(self.datas)) if reverse else self.datas
                for data in data_sequence:
                    self.order_target_percent(data=data, target=0.6)

        def notify_order(self, order: Any) -> None:
            if _terminal(order):
                self.events.append(
                    {
                        "asset": order.data._name,
                        "created_size": order.created.size,
                        "reference": order.ref,
                        "status": order.getstatusname(),
                    }
                )

    _, strategy = _run(
        Strategy,
        ("a", _frame([100.0, 100.0])),
        ("b", _frame([100.0, 100.0])),
        cash=1_000.0,
    )
    return [
        {
            "asset": event["asset"],
            "created_size": event["created_size"],
            "status": event["status"],
        }
        for event in sorted(strategy.events, key=lambda event: event["reference"])
    ]


def _signal_price_stop() -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def next(self) -> None:
            if len(self) == 1:
                self.buy(size=1)
                self.sell(size=1, exectype=bt.Order.Stop, price=self.data.close[0] * 0.9)

        def notify_order(self, order: Any) -> None:
            if order.status == order.Completed:
                self.events.append(
                    {
                        "created_price": order.created.price,
                        "executed_price": order.executed.price,
                        "size": order.executed.size,
                    }
                )

    _, strategy = _run(
        Strategy,
        (
            "asset",
            _frame(
                [100.0, 110.0, 95.0],
                closes=[100.0, 100.0, 95.0],
                highs=[101.0, 111.0, 96.0],
                lows=[99.0, 85.0, 94.0],
            ),
        ),
    )
    return strategy.events


def _trailing_stop() -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def next(self) -> None:
            if len(self) == 1:
                self.buy(size=1)
                self.sell(size=1, exectype=bt.Order.StopTrail, trailpercent=0.1)

        def notify_order(self, order: Any) -> None:
            if order.status == order.Completed:
                self.events.append(
                    {
                        "date": bt.num2date(order.executed.dt).date().isoformat(),
                        "executed_price": order.executed.price,
                        "size": order.executed.size,
                    }
                )

    _, strategy = _run(
        Strategy,
        (
            "asset",
            _frame(
                [100.0, 110.0, 95.0, 100.0],
                closes=[100.0, 95.0, 90.0, 100.0],
                highs=[101.0, 111.0, 96.0, 101.0],
                lows=[99.0, 94.0, 85.0, 99.0],
            ),
        ),
    )
    return strategy.events


def _short_cash(short_cash: bool) -> dict[str, Any]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.event: dict[str, Any] = {}

        def next(self) -> None:
            if len(self) == 1:
                self.sell(size=1)

        def notify_order(self, order: Any) -> None:
            if order.status == order.Completed:
                self.event = {
                    "cash": self.broker.getcash(),
                    "position": self.position.size,
                    "price": order.executed.price,
                }

    def configure(broker: Any) -> None:
        broker.set_shortcash(short_cash)

    _, strategy = _run(
        Strategy,
        ("asset", _frame([100.0, 100.0, 100.0])),
        configure=configure,
    )
    return strategy.event


def _missing_and_late_bars() -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def prenext(self) -> None:
            self._record("prenext")

        def nextstart(self) -> None:
            self._record("nextstart")

        def next(self) -> None:
            self._record("next")

        def _record(self, callback: str) -> None:
            self.events.append(
                {
                    "callback": callback,
                    "date": self.datetime.date().isoformat(),
                    "lengths": [len(data) for data in self.datas],
                    "values": [float(data.close[0]) if len(data) else None for data in self.datas],
                }
            )

    _, strategy = _run(
        Strategy,
        (
            "a",
            _frame(
                [100.0, 120.0],
                dates=["2024-01-01", "2024-01-03"],
            ),
        ),
        (
            "b",
            _frame(
                [200.0, 210.0, 220.0],
                dates=["2024-01-01", "2024-01-02", "2024-01-03"],
            ),
        ),
    )
    return strategy.events


def _late_start() -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def prenext(self) -> None:
            self._record("prenext")

        def nextstart(self) -> None:
            self._record("nextstart")

        def next(self) -> None:
            self._record("next")

        def _record(self, callback: str) -> None:
            self.events.append(
                {
                    "callback": callback,
                    "date": self.datetime.date().isoformat(),
                    "lengths": [len(data) for data in self.datas],
                }
            )

    _, strategy = _run(
        Strategy,
        ("a", _frame([100.0, 100.0, 100.0])),
        (
            "b",
            _frame([100.0, 100.0], dates=["2024-01-02", "2024-01-03"]),
        ),
    )
    return strategy.events


def _final_bar_order() -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def next(self) -> None:
            if len(self) == 2:
                self.buy(size=1)

        def notify_order(self, order: Any) -> None:
            if order.status == order.Completed:
                self.events.append({"price": order.executed.price})

    _, strategy = _run(Strategy, ("asset", _frame([100.0, 110.0])))
    return strategy.events


def _trade_record() -> list[dict[str, Any]]:
    class Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def next(self) -> None:
            if len(self) == 1:
                self.buy(size=2)
            elif len(self) == 2:
                self.close()

        def notify_trade(self, trade: Any) -> None:
            if trade.isclosed:
                self.events.append(
                    {
                        "commission": trade.commission,
                        "entry_price": trade.price,
                        "is_long": trade.long,
                        "pnl": trade.pnl,
                        "pnl_after_commission": trade.pnlcomm,
                    }
                )

    def configure(broker: Any) -> None:
        broker.setcommission(commission=0.01)

    _, strategy = _run(
        Strategy,
        ("asset", _frame([100.0, 110.0, 120.0, 130.0])),
        configure=configure,
    )
    return strategy.events


def _source_location(function: Any) -> dict[str, Any]:
    source_file = inspect.getsourcefile(function)
    if source_file is None:
        raise RuntimeError(f"Could not locate source for {function}")
    package_root = Path(bt.__file__).resolve().parent.parent
    relative_path = Path(source_file).resolve().relative_to(package_root).as_posix()
    return {
        "artifact": ARTIFACT,
        "artifact_sha256": ARTIFACT_SHA256,
        "line": inspect.getsourcelines(function)[1],
        "module": function.__module__,
        "path": relative_path,
        "qualname": function.__qualname__,
        "source": SOURCE,
    }


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and isinstance(expected, (float, int)):
        return math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(_equal(left, right) for left, right in zip(actual, expected, strict=True))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(_equal(actual[key], expected[key]) for key in actual)
    return bool(actual == expected)


def run() -> dict[str, Any]:
    """Execute Backtrader directly and return framework-native evidence."""
    actual_version = importlib.metadata.version("backtrader")
    if actual_version != VERSION:
        raise RuntimeError(f"backtrader version differs: {actual_version} != {VERSION}")

    measurements = [
        (
            "defaults",
            _defaults(),
            {
                "cash": 10_000.0,
                "checksubmit": True,
                "cheat_on_close": False,
                "cheat_on_open": False,
                "commission": 0.0,
                "leverage": 1.0,
                "short_cash": True,
                "slippage_fixed": 0.0,
                "slippage_percent": 0.0,
            },
        ),
        (
            "next_bar_open_and_gap",
            _timing(),
            [{"date": "2024-01-02", "price": 110.0, "size": 1}],
        ),
        (
            "cheat_on_close",
            _timing(coc=True),
            [{"date": "2024-01-01", "price": 100.0, "size": 1}],
        ),
        (
            "integer_target_percent",
            _target_percent(0.5),
            [
                {
                    "status": "Completed",
                    "created_size": 5,
                    "executed_size": 5,
                    "price": 100.0,
                    "commission": 0.0,
                }
            ],
        ),
        (
            "commission_headroom",
            {"full": _target_percent(1.0, 0.001), "headroom": _target_percent(0.998, 0.001)},
            {
                "full": [
                    {
                        "status": "Margin",
                        "created_size": 10,
                        "executed_size": 0,
                        "price": 0.0,
                        "commission": 0.0,
                    }
                ],
                "headroom": [
                    {
                        "status": "Completed",
                        "created_size": 9,
                        "executed_size": 9,
                        "price": 100.0,
                        "commission": 0.9000000000000001,
                    }
                ],
            },
        ),
        (
            "cash_rejection_and_configured_leverage",
            {"default": _cash_and_margin(1.0), "leverage_2": _cash_and_margin(2.0)},
            {
                "default": [{"status": "Margin", "size": 0}],
                "leverage_2": [{"status": "Completed", "size": 20}],
            },
        ),
        (
            "submission_sequence",
            {"forward": _submission_sequence(False), "reverse": _submission_sequence(True)},
            {
                "forward": [
                    {"asset": "a", "created_size": 6, "status": "Completed"},
                    {"asset": "b", "created_size": 6, "status": "Margin"},
                ],
                "reverse": [
                    {"asset": "b", "created_size": 6, "status": "Completed"},
                    {"asset": "a", "created_size": 6, "status": "Margin"},
                ],
            },
        ),
        (
            "signal_price_stop_basis",
            _signal_price_stop(),
            [
                {"created_price": 100.0, "executed_price": 110.0, "size": 1},
                {"created_price": 90.0, "executed_price": 90.0, "size": -1},
            ],
        ),
        (
            "trailing_stop_signal_close_and_lagged",
            _trailing_stop(),
            [
                {"date": "2024-01-02", "executed_price": 110.0, "size": 1},
                {"date": "2024-01-03", "executed_price": 90.0, "size": -1},
            ],
        ),
        (
            "short_cash",
            {"credit": _short_cash(True), "debit": _short_cash(False)},
            {
                "credit": {"cash": 1_100.0, "position": -1, "price": 100.0},
                "debit": {"cash": 900.0, "position": -1, "price": 100.0},
            },
        ),
        (
            "missing_bar_uses_last_value",
            _missing_and_late_bars(),
            [
                {
                    "callback": "nextstart",
                    "date": "2024-01-01",
                    "lengths": [1, 1],
                    "values": [100.0, 200.0],
                },
                {
                    "callback": "nextstart",
                    "date": "2024-01-02",
                    "lengths": [1, 2],
                    "values": [100.0, 210.0],
                },
                {
                    "callback": "next",
                    "date": "2024-01-03",
                    "lengths": [2, 3],
                    "values": [120.0, 220.0],
                },
            ],
        ),
        (
            "late_feed_start",
            _late_start(),
            [
                {"callback": "prenext", "date": "2024-01-01", "lengths": [1, 0]},
                {"callback": "nextstart", "date": "2024-01-02", "lengths": [2, 1]},
                {"callback": "next", "date": "2024-01-03", "lengths": [3, 2]},
            ],
        ),
        ("final_bar_market_order", _final_bar_order(), []),
        (
            "trade_record",
            _trade_record(),
            [
                {
                    "commission": 4.6,
                    "entry_price": 110.0,
                    "is_long": True,
                    "pnl": 20.0,
                    "pnl_after_commission": 15.4,
                }
            ],
        ),
    ]
    checks = [
        {"id": check_id, "actual": actual, "expected": expected, "passed": _equal(actual, expected)}
        for check_id, actual, expected in measurements
    ]
    return {
        "schema_version": 1,
        "framework": "backtrader",
        "package": "backtrader",
        "version": actual_version,
        "artifact": ARTIFACT,
        "artifact_sha256": ARTIFACT_SHA256,
        "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_locations": {
            "broker_next": _source_location(bt.brokers.BackBroker.next),
            "order_execute": _source_location(bt.Order.execute),
            "target_percent": _source_location(bt.Strategy.order_target_percent),
            "target_size": _source_location(bt.CommInfoBase.getsize),
        },
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = run()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Backtrader native behavior run failed: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [check["id"] for check in evidence["checks"] if not check["passed"]]
    if failed:
        print(f"Backtrader native behavior differs: {failed}", file=sys.stderr)
        return 1
    print(f"Backtrader native behavior passed: {len(evidence['checks'])} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
