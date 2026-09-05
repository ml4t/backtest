# ruff: noqa: F403, F405, I001, SIM114, SIM401
# region imports
from AlgorithmImports import *
# endregion

import json
from pathlib import Path


class LeanNativeBehavior(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2024, 1, 2)
        self.set_end_date(2024, 1, 11)
        self.set_cash(10_000)
        self.set_brokerage_model(BrokerageName.DEFAULT, AccountType.MARGIN)
        self.case = self.get_parameter("case")
        ticker = "MLMISS" if self.case == "fill_forward" else "MLNATV"
        security = self.add_equity(ticker, Resolution.DAILY, fill_forward=True)
        security.set_leverage(2.0)
        self.symbol = security.symbol
        self.security = security
        self.events = []
        self.observations = []
        self.submitted = []
        self.target_quantity = None

        if self.case == "explicit_costs":
            security.set_fee_model(ConstantFeeModel(1.0, "USD"))
            security.set_slippage_model(ConstantSlippageModel(0.001))

        self.models = {
            "brokerage": type(self.brokerage_model).__name__,
            "buying_power": type(security.buying_power_model).__name__,
            "fee": type(security.fee_model).__name__,
            "fill": type(security.fill_model).__name__,
            "leverage": float(security.leverage),
            "slippage": type(security.slippage_model).__name__,
        }

    def _submit(self, quantity):
        ticket = self.market_order(self.symbol, quantity)
        self.submitted.append(
            {
                "order_id": int(ticket.order_id),
                "quantity": float(quantity),
                "status": str(ticket.status),
            }
        )

    def on_data(self, data: Slice):
        bar = data.bars[self.symbol] if self.symbol in data.bars else None
        self.observations.append(
            {
                "close": float(bar.close) if bar is not None else None,
                "date": self.time.strftime("%Y-%m-%d"),
                "fill_forward": bool(bar.is_fill_forward) if bar is not None else None,
                "open": float(bar.open) if bar is not None else None,
                "volume": float(bar.volume) if bar is not None else None,
            }
        )

        first = self.time.strftime("%Y-%m-%d") == "2024-01-02"
        final = self.time.strftime("%Y-%m-%d") == "2024-01-11"
        if first:
            if self.case in {"timing", "default_models", "explicit_costs"}:
                self._submit(1)
            elif self.case == "target_sizing":
                self.target_quantity = float(self.calculate_order_quantity(self.symbol, 1.0))
                self._submit(self.target_quantity)
            elif self.case == "submission_sequence":
                self._submit(1)
                self._submit(-1)
            elif self.case == "buying_power_allowed":
                self._submit(150)
            elif self.case == "buying_power_rejected":
                self._submit(250)
            elif self.case == "buying_power_sequence":
                self._submit(150)
                self._submit(150)
            elif self.case == "default_full_fill":
                self._submit(150)
            elif self.case == "terminal_holding":
                self._submit(1)
            elif self.case == "liquidation":
                self._submit(1)

        if self.case == "fill_forward" and self.time.strftime("%Y-%m-%d") == "2024-01-03":
            self._submit(1)

        if self.case == "liquidation" and self.time.strftime("%Y-%m-%d") == "2024-01-10":
            tickets = self.liquidate(self.symbol)
            for ticket in tickets:
                self.submitted.append(
                    {
                        "order_id": int(ticket.order_id),
                        "quantity": float(ticket.quantity),
                        "status": str(ticket.status),
                    }
                )

        if final and self.case == "final_bar_order":
            self._submit(1)

    def on_order_event(self, order_event: OrderEvent):
        fee = 0.0
        if order_event.order_fee and order_event.order_fee.value is not None:
            fee = float(order_event.order_fee.value.amount)
        self.events.append(
            {
                "direction": str(order_event.direction),
                "event_time_utc": order_event.utc_time.strftime("%Y-%m-%d %H:%M:%S"),
                "fee": fee,
                "fill_price": float(order_event.fill_price),
                "fill_quantity": float(order_event.fill_quantity),
                "message": str(order_event.message or ""),
                "order_id": int(order_event.order_id),
                "status": str(order_event.status),
                "time": self.time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def on_end_of_algorithm(self):
        payload = {
            "case": self.case,
            "cash": float(self.portfolio.cash),
            "events": self.events,
            "models": self.models,
            "observations": self.observations,
            "position": float(self.portfolio[self.symbol].quantity),
            "submitted": self.submitted,
            "target_quantity": self.target_quantity,
            "total_fees": float(self.portfolio.total_fees),
            "total_portfolio_value": float(self.portfolio.total_portfolio_value),
        }
        output = Path(__file__).resolve().parent / f"lean_native_{self.case}.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
