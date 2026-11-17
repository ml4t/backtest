"""Compare fill-aware IOC execution on one checksummed Bybit XRPUSDT hour.

Install NautilusTrader 2.0.0rc8 separately. This compares the displayed-depth
replay API with Nautilus's native L2 matcher, outside the bar-based Engine.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from ml4t.specs import OrderSide

from ml4t.backtest.execution.depth import DepthConsumption, DisplayedDepthBook

BOOK_HASH = "35ffe2b0316acb706da4cf5d3c0964bb330da75d3f145449688416522194ab05"
TRADE_HASH = "69e42bf37e69d61107204f36363ef5fda2462676f142d8b92a3cc3b290075d80"
START = 1_733_011_200_000
SCALE = 10_000
FEE = Decimal("0.00055")
CENT_SCALE = Decimal("0.00000001")
BUY, SELL = OrderSide.BUY, OrderSide.SELL


def validate_archive(path: Path, expected_hash: str) -> None:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
            raise ValueError(f"Unexpected archive bytes: {path}")


def read_book(path: Path, end: int):
    validate_archive(path, BOOK_HASH)
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as stream:
        previous = 0
        sequence = None
        for line in stream:
            msg = json.loads(line)
            ts, cts = msg["ts"], msg["cts"]
            if ts >= end:
                break
            raw = msg["data"]
            snapshot = msg["type"] == "snapshot"
            if (
                raw["s"] != "XRPUSDT"
                or msg["topic"] != "orderbook.500.XRPUSDT"
                or not ts >= START
                or not previous < ts
                or cts > ts
                or msg["type"] not in {"snapshot", "delta"}
                or (not snapshot and (sequence is None or raw["u"] != sequence + 1))
            ):
                raise ValueError(f"Invalid Bybit update at {ts}")
            edits = []
            for side, key in ((BUY, "b"), (SELL, "a")):
                for price, size in raw[key]:
                    ticks = Decimal(price) * SCALE
                    if ticks != int(ticks) or int(size) < 0:
                        raise ValueError(f"Off-grid Bybit update at {ts}")
                    edits.append((side, int(ticks), int(size)))
            yield ts, cts, raw["u"], snapshot, edits
            previous, sequence = ts, raw["u"]


def audit_trades(path: Path, end: int, book_clock: list[int]) -> dict:
    from bisect import bisect_left

    validate_archive(path, TRADE_HASH)
    previous = Decimal(0)
    same_ms = set()
    matched_ms = 0
    count = 0
    seen_ids = set()
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "trdMatchID" not in reader.fieldnames:
            raise ValueError("Missing Bybit trade identity")
        for row in reader:
            ts = Decimal(row["timestamp"]) * 1000
            if ts >= end:
                break
            if (
                not ts >= START
                or ts < previous
                or row["symbol"] != "XRPUSDT"
                or row["side"] not in ("Buy", "Sell")
                or row["trdMatchID"] in seen_ids
            ):
                raise ValueError(f"Invalid Bybit trade at {ts}")
            seen_ids.add(row["trdMatchID"])
            count += 1
            ms = int(ts)
            index = bisect_left(book_clock, ms)
            if index < len(book_clock) and book_clock[index] == ms:
                matched_ms += 1
                same_ms.add(ms)
            previous = ts
    return {
        "trades": count,
        "distinct_book_engine_ms_with_trade": len(same_ms),
        "trades_in_same_ms_as_book_engine_update": matched_ms,
        "shared_sequence": False,
        "matching_assumption": (
            "The public trade archive has no shared sequence or RPI indicator. "
            "Same-millisecond observations cannot be causally ordered. "
            "Only book updates drive the marketable IOC comparison."
        ),
    }


@dataclass
class IocStrategy:
    """Submit the next child using the first child's actual fills and resulting position."""

    end: int
    next_group: int = START + 1000
    next_child: int = 0
    side: OrderSide = BUY
    child: int = 0
    first_quantity: int = 0
    first_limit: int | None = None
    position: int = 0

    def on_fill(self, child: int, price: int, quantity: int) -> None:
        self.position += quantity if self.side is BUY else -quantity
        if child == self.child - 1 and self.next_child:
            self.first_quantity += quantity
            self.first_limit = price

    def on_book(self, ts: int) -> tuple[int, OrderSide, int, int | None] | None:
        if self.next_child and ts >= self.next_child:
            self.next_child = 0
            if self.first_quantity:
                quantity = (
                    self.first_quantity // 2
                    if abs(self.position) > 500_000
                    else self.first_quantity
                )
                if quantity:
                    self.child += 1
                    return self.child - 1, self.side, quantity, self.first_limit
        if self.next_group <= ts < self.end - 1000:
            group = (self.next_group - START) // 1000
            self.next_group += 1000
            self.side = BUY if group % 2 else SELL
            self.next_child = ts + 250
            self.first_quantity = 0
            self.first_limit = None
            self.child += 1
            return self.child - 1, self.side, 250_000, None
        return None


def fee(price: int, quantity: int) -> str:
    return str((Decimal(price * quantity) / SCALE * FEE).quantize(CENT_SCALE))


def stream_digest(orders, fills) -> str:
    records = [(child, order, fills.get(child, [])) for child, order in sorted(orders.items())]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def run(book_path: Path, trade_path: Path, minutes: int) -> dict:
    if version("nautilus_trader") != "2.0.0rc8":
        raise ValueError("This comparison requires NautilusTrader 2.0.0rc8")
    backtest, common, execution, model, risk, trading = (
        import_module(f"nautilus_trader.{module}")
        for module in ("backtest", "common", "execution", "model", "risk", "trading")
    )
    if not 1 <= minutes <= 60:
        raise ValueError("Minutes must be in [1, 60]")
    end = START + minutes * 60_000
    venue = model.Venue("BYBIT")
    symbol = model.Symbol("XRPUSDT")
    instrument_id = model.InstrumentId(symbol, venue)
    quote = model.Currency.from_str("USDT")
    side_map = {BUY: model.OrderSide.BUY, SELL: model.OrderSide.SELL}
    modes = (
        DepthConsumption.REUSE,
        DepthConsumption.REFRESH_SIZE_AT_MATCH,
        DepthConsumption.CARRY_CONSUMPTION,
    )
    local = {
        mode: (DisplayedDepthBook(mode), IocStrategy(end), {}, defaultdict(list)) for mode in modes
    }
    batches = []
    px = {}
    book_clock = []
    edits_count = 0
    for ts, cts, seq, snapshot, edits in read_book(book_path, end):
        book_clock.append(cts)
        edits_count += len(edits)
        ns = ts * 1_000_000
        deltas = [model.OrderBookDelta.clear(instrument_id, seq, ns, ns)] if snapshot else []
        for i, (side, ticks, size) in enumerate(edits):
            if ticks not in px:
                px[ticks] = model.Price.from_str(f"{Decimal(ticks) / SCALE:.4f}")
            deltas.append(
                model.OrderBookDelta(
                    instrument_id,
                    model.BookAction.UPDATE if size else model.BookAction.DELETE,
                    model.BookOrder(side_map[side], px[ticks], model.Quantity.from_int(size), 0),
                    int(model.RecordFlag.F_LAST) if i == len(edits) - 1 else 0,
                    seq,
                    ns,
                    ns,
                )
            )
        batches.append(model.OrderBookDeltas(instrument_id, deltas))
        for book, strategy, orders, fills in local.values():
            book.update(seq, edits, snapshot=snapshot)
            if (order := strategy.on_book(ts)) is not None:
                child, side, qty, limit = order
                orders[child] = (ts, side, qty, limit)
                for price, size in book.sweep(side, qty, limit=limit):
                    strategy.on_fill(child, price, size)
                    fills[child].append((ts, price, size, fee(price, size)))

    class NativeStrategy(trading.Strategy):
        def __init__(self):
            super().__init__()
            self.strategy = IocStrategy(end)
            self.orders = {}
            self.fills = defaultdict(list)
            self.index_by_id = {}

        def on_start(self):
            self.subscribe_book_deltas(instrument_id, model.BookType.L2_MBP)

        def on_book_deltas(self, deltas):
            ts = deltas.ts_event // 1_000_000
            decision = self.strategy.on_book(ts)
            if decision is None:
                return
            child, side, qty, limit = decision
            self.orders[child] = (ts, side, qty, limit)
            arguments = (instrument_id, side_map[side], model.Quantity.from_int(qty))
            if limit is None:
                order = self.order_factory.market(*arguments, time_in_force=model.TimeInForce.IOC)
            else:
                order = self.order_factory.limit(
                    *arguments, px[limit], time_in_force=model.TimeInForce.IOC
                )
            self.index_by_id[str(order.client_order_id)] = child
            self.submit_order(order)

        def on_order_filled(self, event):
            child = self.index_by_id[str(event.client_order_id)]
            price = int(event.last_px.as_decimal() * SCALE)
            size = int(event.last_qty.as_decimal())
            ts = event.ts_event // 1_000_000
            self.fills[child].append(
                (ts, price, size, str(event.commission.as_decimal().quantize(CENT_SCALE)))
            )
            self.strategy.on_fill(child, price, size)

    outcomes = {}
    for mode in modes[:2]:
        strategy = NativeStrategy()
        engine = backtest.BacktestEngine(
            backtest.BacktestEngineConfig(
                logging=common.LoggerConfig(bypass_logging=True),
                run_analysis=False,
                risk_engine=risk.RiskEngineConfig(bypass=True),
            )
        )
        try:
            engine.add_venue(
                venue,
                model.OmsType.NETTING,
                model.AccountType.MARGIN,
                [model.Money(1e9, quote)],
                book_type=model.BookType.L2_MBP,
                liquidity_consumption=mode is DepthConsumption.REFRESH_SIZE_AT_MATCH,
                trade_execution=False,
                fee_model=execution.MakerTakerFeeModel(FEE, FEE),
            )
            engine.add_instrument(
                model.CryptoPerpetual(
                    instrument_id,
                    symbol,
                    model.Currency.from_str("XRP"),
                    quote,
                    quote,
                    False,
                    4,
                    0,
                    model.Price.from_str("0.0001"),
                    model.Quantity.from_int(1),
                    0,
                    0,
                )
            )
            engine.add_data(batches)
            engine.add_strategy(strategy)
            engine.run()
            _, local_strategy, orders, fills = local[mode]
            ids = sorted(orders.keys() | strategy.orders.keys())
            mismatch = [
                child
                for child in ids
                if orders.get(child) != strategy.orders.get(child)
                or fills.get(child, []) != strategy.fills.get(child, [])
            ]
            native_position = sum(int(pos.signed_qty) for pos in engine.cache.positions_open())
            if native_position != strategy.strategy.position:
                raise AssertionError("Nautilus account position disagrees with the native fills")
            order_residuals = [
                order[2] - sum(fill[2] for fill in fills.get(child, ()))
                for child, order in orders.items()
            ]
            outcomes[mode.value] = {
                "local_orders": len(orders),
                "native_orders": len(strategy.orders),
                "local_fills": sum(map(len, fills.values())),
                "native_fills": sum(map(len, strategy.fills.values())),
                "local_position": local_strategy.position,
                "native_position_from_fills": strategy.strategy.position,
                "native_account_position": native_position,
                "local_orders_with_remainder": sum(residual > 0 for residual in order_residuals),
                "local_requested_lots": sum(order[2] for order in orders.values()),
                "local_unfilled_lots": sum(order_residuals),
                "local_filled_lots": sum(fill[2] for rows in fills.values() for fill in rows),
                "local_execution_stream_sha256": stream_digest(orders, fills),
                "native_execution_stream_sha256": stream_digest(strategy.orders, strategy.fills),
                "order_or_fill_mismatches": len(mismatch),
                "first_mismatch": (
                    {
                        "child": mismatch[0],
                        "local_order": orders.get(mismatch[0]),
                        "native_order": strategy.orders.get(mismatch[0]),
                        "local_fills": fills.get(mismatch[0], []),
                        "native_fills": strategy.fills.get(mismatch[0], []),
                    }
                    if mismatch
                    else None
                ),
            }
        finally:
            engine.dispose()

    _, conservative, orders, fills = local[DepthConsumption.CARRY_CONSUMPTION]
    outcomes[DepthConsumption.CARRY_CONSUMPTION.value] = {
        "local_orders": len(orders),
        "local_fills": sum(map(len, fills.values())),
        "local_position": conservative.position,
        "local_requested_lots": sum(order[2] for order in orders.values()),
        "local_filled_lots": sum(fill[2] for rows in fills.values() for fill in rows),
    }
    return {
        "minutes": minutes,
        "nautilus_version": "2.0.0rc8",
        "book_sha256": BOOK_HASH,
        "trades_sha256": TRADE_HASH,
        "order_policy": "IOC, zero insertion latency, next child after 250 ms at the next book update",
        "book_updates": len(batches),
        "price_level_updates": edits_count,
        "trade_audit": audit_trades(trade_path, end, sorted(book_clock)),
        "cases": outcomes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", type=Path)
    parser.add_argument("trades", type=Path)
    parser.add_argument("--minutes", type=int, default=60)
    args = parser.parse_args()
    result = run(args.book, args.trades, args.minutes)
    print(json.dumps(result, indent=2))
    if any(case.get("order_or_fill_mismatches", 0) for case in result["cases"].values()):
        raise SystemExit(1)
