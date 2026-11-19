"""Execute both dated-contract roll legs in Nautilus and reconcile its daily marked equity."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import pandas as pd
from reference_workloads import CASH

if TYPE_CHECKING:
    from pysystemtrade_trend import TrendInput

SESSION_NS = 86400 * 10**9
EXPIRATIONS = {
    "SP500_20220300": "2022-03-18 13:30:00",
    "SP500_20220600": "2022-06-17 13:30:00",
    "GOLD_20220400": "2022-04-27",
    "GOLD_20220600": "2022-06-28",
}


def run_nautilus_trend(inputs: TrendInput, targets: dict[str, pd.Series]) -> tuple[dict, list[int]]:
    backtest: Any = import_module("nautilus_trader.backtest")
    model: Any = import_module("nautilus_trader.model")
    strategy: Any = import_module("nautilus_trader.trading").Strategy
    logger = import_module("nautilus_trader.common").LoggerConfig
    fee_model = import_module("nautilus_trader.execution").PerContractFeeModel
    usd = model.Currency.from_str("USD")
    venues = {"SP500": model.Venue("XCME"), "GOLD": model.Venue("XCEC")}
    instruments = {}
    for root, legs in inputs.legs.items():
        for asset in legs:
            sym = model.Symbol(asset)
            instruments[asset] = model.FuturesContract(
                model.InstrumentId(sym, venues[root]),
                sym,
                model.AssetClass.INDEX if root == "SP500" else model.AssetClass.COMMODITY,
                root,
                pd.Timestamp("2021-01-01").value,
                pd.Timestamp(EXPIRATIONS[asset]).value,
                usd,
                2,
                model.Price(0.25 if root == "SP500" else 0.10, 2),
                model.Quantity.from_int(inputs.multipliers[asset]),
                model.Quantity.from_int(1),
                0,
                0,
            )

    by_time: dict[int, list[tuple[str, float]]] = {stamp.value: [] for stamp in inputs.dates}
    for asset, series in inputs.prices.items():
        for stamp, price in zip(pd.DatetimeIndex(series.index), series, strict=True):
            by_time[stamp.value].append((asset, price))
    equity, cash_marks = [], []
    marks: dict[str, float] = {}

    class DatedTargets(strategy):
        def on_start(self):
            for instrument in instruments.values():
                self.subscribe_quotes(instrument.id)

        def on_quote(self, quote):
            time = quote.ts_event
            if time % SESSION_NS:
                # Sample after all orders at the shared session label can execute.
                cash = sum(
                    float(self.cache.account_for_venue(venue).balance_total(usd))
                    for venue in venues.values()
                )
                unrealized = sum(
                    float(
                        position.unrealized_pnl(
                            model.Price(marks[str(position.instrument_id.symbol)], 2)
                        )
                    )
                    for position in self.cache.positions_open()
                )
                cash_marks.append(round(cash * 100))
                equity.append(round((cash + unrealized) * 100))
                return
            ident = quote.instrument_id
            asset = str(ident.symbol)
            marks[asset] = float(quote.bid_price)
            wanted = int(targets[asset].loc[pd.Timestamp(time)])
            delta = wanted - int(self.portfolio.net_position(ident))
            if delta:
                self.submit_order(
                    self.order_factory.market(
                        ident,
                        model.OrderSide.BUY if delta > 0 else model.OrderSide.SELL,
                        model.Quantity.from_int(abs(delta)),
                    )
                )

    engine = backtest.BacktestEngine(
        backtest.BacktestEngineConfig(
            logging=logger(bypass_logging=True),
            run_analysis=False,
        )
    )
    try:
        for root, venue in venues.items():
            engine.add_venue(
                venue,
                model.OmsType.NETTING,
                model.AccountType.MARGIN,
                [model.Money(CASH / len(venues), usd)],
                base_currency=usd,
                fee_model=fee_model(model.Money(inputs.fees[inputs.legs[root][0]], usd)),
            )
        for instrument in instruments.values():
            engine.add_instrument(instrument)

        size = model.Quantity.from_int(1000)

        def tick(time: int, asset: str, price: float) -> Any:
            price = model.Price(price, 2)
            return model.QuoteTick(instruments[asset].id, price, price, size, size, time, time)

        data = []
        for time, rows in by_time.items():
            data.extend(tick(time, asset, price) for asset, price in rows)
            data.append(tick(time + 1, *rows[0]))
        engine.add_data(data)
        engine.add_strategy(DatedTargets())
        engine.run()
        return {
            "fills": sorted(
                (
                    pd.Timestamp(f.ts_event).value,
                    str(f.instrument_id).split(".")[0],
                    int(float(f.last_qty)) * (1 if f.order_side == "BUY" else -1),
                    round(float(f.last_px) * 100),
                    round(float(str(f.commission).split()[0]) * 100),
                )
                for f in engine.generate_fills_report().itertuples()
            ),
            "positions": {
                asset: int(engine.portfolio.net_position(inst.id))
                for asset, inst in instruments.items()
            },
            "equity_cents": equity,
        }, cash_marks
    finally:
        engine.dispose()
