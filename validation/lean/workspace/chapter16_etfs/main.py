# region imports
from AlgorithmImports import *
# endregion

import csv
import math
from pathlib import Path


class Ml4tPercentFeeModel(FeeModel):
    def __init__(self, rate: float):
        self.rate = rate

    def GetOrderFee(self, parameters: OrderFeeParameters) -> OrderFee:
        order = parameters.Order
        security = parameters.Security
        reference_price = float(security.Open) if float(security.Open) > 0 else float(security.Price)
        fee = abs(float(order.AbsoluteQuantity)) * reference_price * self.rate
        return OrderFee(CashAmount(fee, "USD"))


class Ml4tCaseStudyParity(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2015, 12, 28)
        self.set_end_date(2023, 12, 29)
        self.set_cash(1000000.0)
        self.set_brokerage_model(BrokerageName.DEFAULT, AccountType.MARGIN)
        self._targets = {}
        self._rebalance_dates = set()
        base_path = Path(__file__).resolve().parent
        self._equity_path = base_path / "ml4t_daily_equity.csv"
        self._order_events_path = base_path / "ml4t_order_events.csv"
        self._initialize_artifact_files()

        with (base_path / "weights.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row["timestamp"]
                if key not in self._targets:
                    self._targets[key] = {}
                self._targets[key][row["asset"]] = float(row["target_weight"])
        self._rebalance_dates = {
            line.strip()
            for line in (base_path / "rebalance_dates.csv").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        self._asset_order = []
        self._symbols = {}
        fee_model = Ml4tPercentFeeModel(0.001)
        with (base_path / "asset_symbols.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                asset = row["asset"].strip()
                ticker = row["ticker"].strip()
                if not asset or not ticker:
                    continue
                security = self.add_equity(ticker, Resolution.DAILY)
                security.set_leverage(2.0)
                security.set_fee_model(fee_model)
                security.set_slippage_model(ConstantSlippageModel(0))
                self._asset_order.append(asset)
                self._symbols[asset] = security.symbol
        if self._symbols:
            self.set_benchmark(self._symbols[self._asset_order[0]])

    def _initialize_artifact_files(self):
        with self._equity_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "equity", "cash", "total_fees", "holdings_value"]
            )
        with self._order_events_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "timestamp",
                    "symbol",
                    "status",
                    "direction",
                    "fill_quantity",
                    "fill_price",
                    "fee",
                    "message",
                    "order_id",
                ]
            )

    def _append_csv_row(self, path: Path, row: list[object]) -> None:
        with path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

    def _target_quantity(self, target_weight: float, price: float) -> int:
        raw_qty = (target_weight * float(self.portfolio.total_portfolio_value)) / price
        if raw_qty >= 0:
            return int(math.floor(raw_qty + 1e-12))
        return -int(math.floor(abs(raw_qty) + 1e-12))

    def on_data(self, data: Slice):
        key = self.time.strftime("%Y-%m-%d")
        if key not in self._rebalance_dates:
            return
        targets = self._targets.get(key, {})

        bars = data.bars
        for asset in self._asset_order:
            symbol = self._symbols[asset]
            if bars is None or symbol not in bars:
                continue
            price = float(bars[symbol].close)
            if price <= 0:
                continue
            target_qty = self._target_quantity(targets.get(asset, 0.0), price)
            current_qty = int(self.portfolio[symbol].quantity)
            delta = target_qty - current_qty
            if delta != 0:
                self.market_order(symbol, delta)

        self._append_csv_row(
            self._equity_path,
            [
                key,
                float(self.portfolio.total_portfolio_value),
                float(self.portfolio.cash),
                float(self.portfolio.total_fees),
                float(self.portfolio.total_holdings_value),
            ],
        )

    def on_order_event(self, order_event: OrderEvent):
        fee_amount = 0.0
        if order_event.order_fee and order_event.order_fee.value is not None:
            fee_amount = float(order_event.order_fee.value.amount)

        message = str(order_event.message or "").replace("\n", " ").strip()
        symbol = order_event.symbol.value if order_event.symbol is not None else ""
        self._append_csv_row(
            self._order_events_path,
            [
                self.time.strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                str(order_event.status),
                str(order_event.direction),
                float(order_event.fill_quantity),
                float(order_event.fill_price),
                fee_amount,
                message,
                int(order_event.order_id),
            ],
        )
