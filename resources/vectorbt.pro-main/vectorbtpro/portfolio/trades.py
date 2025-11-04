# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module for working with trade records.

This class encapsulates detailed information on trades executed within vectorbtpro.
In vectorbtpro, a trade is defined as a sequence of orders that begins with an opening order
and may optionally end with a closing order. Each pair of opposing orders is represented by
a trade with associated profit and loss (PnL) details for quick performance assessment.
A key advantage of this representation is the ability to aggregate consecutive trades—allowing,
for example, single-order trades to be combined into positions or multiple positions to be merged
into a single summary of a symbol's performance.

!!! warning
    All classes return both closed and open trades/positions, which may skew performance results.
    To consider only closed trades/positions, explicitly query the `status_closed` attribute.

## Trade types

There are three main types of trades:

### Entry trades

Entry trades are generated from orders that open or add to a position.

For example, if a single large buy order is followed by 100 smaller sell orders,
a single trade is created with the entry details copied from the buy order and
the exit details computed as a size-weighted average of all sell orders.
Conversely, if 100 smaller buy orders are followed by a single sell order,
100 trades are created, each carrying the entry details from the respective
buy order and exit details representing a size-based fraction of the sell order.

Use `vectorbtpro.portfolio.trades.EntryTrades.from_orders` to build entry trades from orders.
They are also accessible via `vectorbtpro.portfolio.base.Portfolio.entry_trades`.

### Exit trades

Exit trades are generated from orders that close or reduce a position.

Use `vectorbtpro.portfolio.trades.ExitTrades.from_orders` to construct exit trades from orders.
They are also available as `vectorbtpro.portfolio.base.Portfolio.exit_trades`.

### Positions

Positions are created by aggregating a sequence of entry or exit trades.

Use `vectorbtpro.portfolio.trades.Positions.from_trades` to build positions from entry or exit trades.
They are also accessible via `vectorbtpro.portfolio.base.Portfolio.positions`.

## Examples

Increasing position:

```pycon
>>> from vectorbtpro import *

>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., 1., 1., 1., -4.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.readable
   Entry Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0               0       0   1.0               0            0              1.0
1               1       0   1.0               1            1              2.0
2               2       0   1.0               2            2              3.0
3               3       0   1.0               3            3              4.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees   PnL  \\
0         1.0              4           4             5.0       0.25  2.75
1         1.0              4           4             5.0       0.25  1.75
2         1.0              4           4             5.0       0.25  0.75
3         1.0              4           4             5.0       0.25 -0.25

   Return Direction  Status  Position Id
0  2.7500      Long  Closed            0
1  0.8750      Long  Closed            0
2  0.2500      Long  Closed            0
3 -0.0625      Long  Closed            0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.readable
   Exit Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0              0       0   4.0               0            0              2.5

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         4.0              4           4             5.0        1.0  5.0

   Return Direction  Status  Position Id
0     0.5      Long  Closed            0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.readable
   Position Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0            0       0   4.0               0            0              2.5

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         4.0              4           4             5.0        1.0  5.0

   Return Direction  Status
0     0.5      Long  Closed

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

Decreasing position:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([4., -1., -1., -1., -1.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.readable
   Entry Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0               0       0   4.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              4           4             3.5        4.0  5.0

   Return Direction  Status  Position Id
0    1.25      Long  Closed            0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.readable
   Exit Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0              0       0   1.0               0            0              1.0
1              1       0   1.0               0            0              1.0
2              2       0   1.0               0            0              1.0
3              3       0   1.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees   PnL  \\
0        0.25              1           1             2.0        1.0 -0.25
1        0.25              2           2             3.0        1.0  0.75
2        0.25              3           3             4.0        1.0  1.75
3        0.25              4           4             5.0        1.0  2.75

   Return Direction  Status  Position Id
0   -0.25      Long  Closed            0
1    0.75      Long  Closed            0
2    1.75      Long  Closed            0
3    2.75      Long  Closed            0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.readable
   Position Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0            0       0   4.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              4           4             3.5        4.0  5.0

   Return Direction  Status
0    1.25      Long  Closed

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

Multiple reversing positions:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., -2., 2., -2., 1.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.readable
   Entry Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0               0       0   1.0               0            0              1.0
1               1       0   1.0               1            1              2.0
2               2       0   1.0               2            2              3.0
3               3       0   1.0               3            3              4.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              1           1             2.0        0.5 -0.5
1         0.5              2           2             3.0        0.5 -2.0
2         0.5              3           3             4.0        0.5  0.0
3         0.5              4           4             5.0        1.0 -2.5

   Return Direction  Status  Position Id
0  -0.500      Long  Closed            0
1  -1.000     Short  Closed            1
2   0.000      Long  Closed            2
3  -0.625     Short  Closed            3

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.readable
   Exit Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0              0       0   1.0               0            0              1.0
1              1       0   1.0               1            1              2.0
2              2       0   1.0               2            2              3.0
3              3       0   1.0               3            3              4.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              1           1             2.0        0.5 -0.5
1         0.5              2           2             3.0        0.5 -2.0
2         0.5              3           3             4.0        0.5  0.0
3         0.5              4           4             5.0        1.0 -2.5

   Return Direction  Status  Position Id
0  -0.500      Long  Closed            0
1  -1.000     Short  Closed            1
2   0.000      Long  Closed            2
3  -0.625     Short  Closed            3

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.readable
   Position Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0            0       0   1.0               0            0              1.0
1            1       0   1.0               1            1              2.0
2            2       0   1.0               2            2              3.0
3            3       0   1.0               3            3              4.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0              1           1             2.0        0.5 -0.5
1         0.5              2           2             3.0        0.5 -2.0
2         0.5              3           3             4.0        0.5  0.0
3         0.5              4           4             5.0        1.0 -2.5

   Return Direction  Status
0  -0.500      Long  Closed
1  -1.000     Short  Closed
2   0.000      Long  Closed
3  -0.625     Short  Closed

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

Open position:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., 0., 0., 0., 0.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.readable
   Entry Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0               0       0   1.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0             -1           4             5.0        0.0  3.0

   Return Direction Status  Position Id
0     3.0      Long   Open            0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.readable
   Exit Trade Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0              0       0   1.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0             -1           4             5.0        0.0  3.0

   Return Direction Status  Position Id
0     3.0      Long   Open            0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.readable
   Position Id  Column  Size  Entry Order Id  Entry Index  Avg Entry Price  \\
0            0       0   1.0               0            0              1.0

   Entry Fees  Exit Order Id  Exit Index  Avg Exit Price  Exit Fees  PnL  \\
0         1.0             -1           4             5.0        0.0  3.0

   Return Direction Status
0     3.0      Long   Open

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

Get trade count, trade PnL, and winning trade PnL:

```pycon
>>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.])
>>> size = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
>>> trades = vbt.Portfolio.from_orders(price, size).trades

>>> trades.count()
6

>>> trades.pnl.sum()
-3.0

>>> trades.winning.count()
2

>>> trades.winning.pnl.sum()
1.5
```

Get count and PnL of trades with a duration longer than 2 days:

```pycon
>>> mask = (trades.records['exit_idx'] - trades.records['entry_idx']) > 2
>>> trades_filtered = trades.apply_mask(mask)
>>> trades_filtered.count()
2

>>> trades_filtered.pnl.sum()
-3.0
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Trades.metrics`.

```pycon
>>> price = vbt.RandomData.pull(
...     ['a', 'b'],
...     start=datetime(2020, 1, 1),
...     end=datetime(2020, 3, 1),
...     seed=vbt.symbol_dict(a=42, b=43)
... ).get()
```

[=100% "100%"]{: .candystripe .candystripe-animate }

```pycon
>>> size = pd.DataFrame({
...     'a': np.random.randint(-1, 2, size=len(price.index)),
...     'b': np.random.randint(-1, 2, size=len(price.index)),
... }, index=price.index, columns=price.columns)
>>> pf = vbt.Portfolio.from_orders(price, size, fees=0.01, init_cash="auto")

>>> pf.trades['a'].stats()
Start                          2019-12-31 23:00:00+00:00
End                            2020-02-29 23:00:00+00:00
Period                                  61 days 00:00:00
First Trade Start              2019-12-31 23:00:00+00:00
Last Trade End                 2020-02-29 23:00:00+00:00
Coverage                                60 days 00:00:00
Overlap Coverage                        49 days 00:00:00
Total Records                                       19.0
Total Long Trades                                    2.0
Total Short Trades                                  17.0
Total Closed Trades                                 18.0
Total Open Trades                                    1.0
Open Trade PnL                                    16.063
Win Rate [%]                                   61.111111
Max Win Streak                                      11.0
Max Loss Streak                                      7.0
Best Trade [%]                                  3.526377
Worst Trade [%]                                -6.543679
Avg Winning Trade [%]                           2.225861
Avg Losing Trade [%]                           -3.601313
Avg Winning Trade Duration    32 days 19:38:10.909090909
Avg Losing Trade Duration                5 days 00:00:00
Profit Factor                                   1.022425
Expectancy                                      0.028157
SQN                                             0.039174
Name: agg_stats, dtype: object
```

Positions share almost identical metrics with trades:

```pycon
>>> pf.positions['a'].stats()
Start                         2019-12-31 23:00:00+00:00
End                           2020-02-29 23:00:00+00:00
Period                                 61 days 00:00:00
First Trade Start             2019-12-31 23:00:00+00:00
Last Trade End                2020-02-29 23:00:00+00:00
Coverage                               60 days 00:00:00
Overlap Coverage                        0 days 00:00:00
Total Records                                       5.0
Total Long Trades                                   2.0
Total Short Trades                                  3.0
Total Closed Trades                                 4.0
Total Open Trades                                   1.0
Open Trade PnL                                38.356823
Win Rate [%]                                        0.0
Max Win Streak                                      0.0
Max Loss Streak                                     4.0
Best Trade [%]                                -1.529613
Worst Trade [%]                               -6.543679
Avg Winning Trade [%]                               NaN
Avg Losing Trade [%]                          -3.786739
Avg Winning Trade Duration                          NaT
Avg Losing Trade Duration               4 days 00:00:00
Profit Factor                                       0.0
Expectancy                                    -5.446748
SQN                                           -1.794214
Name: agg_stats, dtype: object
```

To also include open trades/positions when calculating metrics such as win rate, pass `incl_open=True`:

```pycon
>>> pf.trades['a'].stats(settings=dict(incl_open=True))
Start                         2019-12-31 23:00:00+00:00
End                           2020-02-29 23:00:00+00:00
Period                                 61 days 00:00:00
First Trade Start             2019-12-31 23:00:00+00:00
Last Trade End                2020-02-29 23:00:00+00:00
Coverage                               60 days 00:00:00
Overlap Coverage                       49 days 00:00:00
Total Records                                      19.0
Total Long Trades                                   2.0
Total Short Trades                                 17.0
Total Closed Trades                                18.0
Total Open Trades                                   1.0
Open Trade PnL                                   16.063
Win Rate [%]                                  61.111111
Max Win Streak                                     12.0
Max Loss Streak                                     7.0
Best Trade [%]                                 3.526377
Worst Trade [%]                               -6.543679
Avg Winning Trade [%]                          2.238896
Avg Losing Trade [%]                          -3.601313
Avg Winning Trade Duration             33 days 18:00:00
Avg Losing Trade Duration               5 days 00:00:00
Profit Factor                                  1.733143
Expectancy                                     0.872096
SQN                                            0.804714
Name: agg_stats, dtype: object
```

`Trades.stats` also supports (re-)grouping:

```pycon
>>> pf.trades.stats(group_by=True)
Start                          2019-12-31 23:00:00+00:00
End                            2020-02-29 23:00:00+00:00
Period                                  61 days 00:00:00
First Trade Start              2019-12-31 23:00:00+00:00
Last Trade End                 2020-02-29 23:00:00+00:00
Coverage                                61 days 00:00:00
Overlap Coverage                        61 days 00:00:00
Total Records                                         37
Total Long Trades                                      5
Total Short Trades                                    32
Total Closed Trades                                   35
Total Open Trades                                      2
Open Trade PnL                                  1.336259
Win Rate [%]                                   37.142857
Max Win Streak                                        11
Max Loss Streak                                       10
Best Trade [%]                                  3.526377
Worst Trade [%]                                -8.710238
Avg Winning Trade [%]                           1.907799
Avg Losing Trade [%]                           -3.259135
Avg Winning Trade Duration    28 days 14:46:09.230769231
Avg Losing Trade Duration               14 days 00:00:00
Profit Factor                                   0.340493
Expectancy                                     -1.292596
SQN                                            -2.509223
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Trades.subplots`.

The `Trades` class provides two subplots based on `Trades.plot` and `Trades.plot_pnl`:

```pycon
>>> pf.trades['a'].plots().show()
```

![](/assets/images/api/trades_plots.light.svg#only-light){: .iimg loading=lazy }
![](/assets/images/api/trades_plots.dark.svg#only-dark){: .iimg loading=lazy }
"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.indexes import stack_indexes
from vectorbtpro.base.reshaping import broadcast_to, to_1d_array, to_2d_array, to_pd_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic.enums import WType, range_dt
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.enums import TradeDirection, TradeStatus, trade_dt
from vectorbtpro.portfolio.orders import Orders
from vectorbtpro.records.decorators import (
    attach_fields,
    attach_shortcut_properties,
    override_field_config,
)
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.array_ import max_rel_rescale, min_rel_rescale
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import Config, HybridConfig, ReadonlyConfig, merge_dicts
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.template import Rep, RepEval, RepFunc

__all__ = [
    "Trades",
    "EntryTrades",
    "ExitTrades",
    "Positions",
]

__pdoc__ = {}

# ############# Trades ############# #

trades_field_config = ReadonlyConfig(
    dict(
        dtype=trade_dt,
        settings={
            "id": dict(title="Trade Id"),
            "idx": dict(name="exit_idx"),  # remap field of Records
            "start_idx": dict(name="entry_idx"),  # remap field of Ranges
            "end_idx": dict(name="exit_idx"),  # remap field of Ranges
            "size": dict(title="Size"),
            "entry_order_id": dict(title="Entry Order Id", mapping="ids"),
            "entry_idx": dict(title="Entry Index", mapping="index"),
            "entry_price": dict(title="Avg Entry Price"),
            "entry_fees": dict(title="Entry Fees"),
            "exit_order_id": dict(title="Exit Order Id", mapping="ids"),
            "exit_idx": dict(title="Exit Index", mapping="index"),
            "exit_price": dict(title="Avg Exit Price"),
            "exit_fees": dict(title="Exit Fees"),
            "pnl": dict(title="PnL"),
            "return": dict(title="Return", hovertemplate="$title: %{customdata[$index]:,%}"),
            "direction": dict(title="Direction", mapping=TradeDirection),
            "status": dict(title="Status", mapping=TradeStatus),
            "parent_id": dict(title="Position Id", mapping="ids"),
        },
    )
)
"""_"""

__pdoc__["trades_field_config"] = f"""Field configuration for `Trades`.

```python
{trades_field_config.prettify_doc()}
```
"""

trades_attach_field_config = ReadonlyConfig(
    {
        "return": dict(attach="returns"),
        "direction": dict(attach_filters=True),
        "status": dict(attach_filters=True, on_conflict="ignore"),
    }
)
"""_"""

__pdoc__[
    "trades_attach_field_config"
] = f"""Configuration mapping for attaching extra properties to `Trades`.

```python
{trades_attach_field_config.prettify_doc()}
```
"""

trades_shortcut_config = ReadonlyConfig(
    dict(
        ranges=dict(),
        long_view=dict(),
        short_view=dict(),
        winning=dict(),
        losing=dict(),
        winning_streak=dict(obj_type="mapped_array"),
        losing_streak=dict(obj_type="mapped_array"),
        win_rate=dict(obj_type="red_array"),
        profit_factor=dict(obj_type="red_array", method_kwargs=dict(use_returns=False)),
        rel_profit_factor=dict(
            obj_type="red_array",
            method_name="get_profit_factor",
            method_kwargs=dict(
                use_returns=True, wrap_kwargs=dict(name_or_index="rel_profit_factor")
            ),
        ),
        expectancy=dict(obj_type="red_array", method_kwargs=dict(use_returns=False)),
        rel_expectancy=dict(
            obj_type="red_array",
            method_name="get_expectancy",
            method_kwargs=dict(use_returns=True, wrap_kwargs=dict(name_or_index="rel_expectancy")),
        ),
        sqn=dict(obj_type="red_array", method_kwargs=dict(use_returns=False)),
        rel_sqn=dict(
            obj_type="red_array",
            method_name="get_sqn",
            method_kwargs=dict(use_returns=True, wrap_kwargs=dict(name_or_index="rel_sqn")),
        ),
        best_price=dict(obj_type="mapped_array"),
        worst_price=dict(obj_type="mapped_array"),
        best_price_idx=dict(obj_type="mapped_array"),
        worst_price_idx=dict(obj_type="mapped_array"),
        expanding_best_price=dict(obj_type="array"),
        expanding_worst_price=dict(obj_type="array"),
        mfe=dict(obj_type="mapped_array"),
        mfe_returns=dict(
            obj_type="mapped_array",
            method_name="get_mfe",
            method_kwargs=dict(use_returns=True),
        ),
        mae=dict(obj_type="mapped_array"),
        mae_returns=dict(
            obj_type="mapped_array",
            method_name="get_mae",
            method_kwargs=dict(use_returns=True),
        ),
        expanding_mfe=dict(obj_type="array"),
        expanding_mfe_returns=dict(
            obj_type="array",
            method_name="get_expanding_mfe",
            method_kwargs=dict(use_returns=True),
        ),
        expanding_mae=dict(obj_type="array"),
        expanding_mae_returns=dict(
            obj_type="array",
            method_name="get_expanding_mae",
            method_kwargs=dict(use_returns=True),
        ),
        edge_ratio=dict(obj_type="red_array"),
        running_edge_ratio=dict(obj_type="array"),
    )
)
"""_"""

__pdoc__[
    "trades_shortcut_config"
] = f"""Configuration for shortcut properties associated with `Trades`.

```python
{trades_shortcut_config.prettify_doc()}
```
"""

TradesT = tp.TypeVar("TradesT", bound="Trades")


@attach_shortcut_properties(trades_shortcut_config)
@attach_fields(trades_attach_field_config)
@override_field_config(trades_field_config)
class Trades(Ranges):
    """Class for representing trade-like records, including entry trades, exit trades, and positions.

    Requires `records_arr` to have all fields defined in `vectorbtpro.portfolio.enums.trade_dt`.

    !!! info
        For default settings, see `vectorbtpro._settings.trades`.
    """

    @property
    def field_config(self) -> Config:
        """Field configuration.

        Returns:
            Config: Field configuration for the `Trades` class.
        """
        return self._field_config

    def get_ranges(self, **kwargs) -> Ranges:
        """Return trade records as a `vectorbtpro.generic.ranges.Ranges` instance.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_records`.

        Returns:
            Ranges: New instance of `vectorbtpro.generic.ranges.Ranges` constructed from trade record fields.

                Has the `vectorbtpro.generic.enums.range_dt` dtype.
        """
        new_records_arr = np.empty(self.values.shape, dtype=range_dt)
        new_records_arr["id"][:] = self.get_field_arr("id").copy()
        new_records_arr["col"][:] = self.get_field_arr("col").copy()
        new_records_arr["start_idx"][:] = self.get_field_arr("entry_idx").copy()
        new_records_arr["end_idx"][:] = self.get_field_arr("exit_idx").copy()
        new_records_arr["status"][:] = self.get_field_arr("status").copy()
        return Ranges.from_records(
            self.wrapper,
            new_records_arr,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            **kwargs,
        )

    # ############# Views ############# #

    def get_long_view(self: TradesT, **kwargs) -> TradesT:
        """Return trade records filtered for long trades.

        Args:
            **kwargs: Keyword arguments for `Trades.apply_mask`.

        Returns:
            Trades: New instance of trade records containing only long trades.
        """
        filter_mask = self.get_field_arr("direction") == TradeDirection.Long
        return self.apply_mask(filter_mask, **kwargs)

    def get_short_view(self: TradesT, **kwargs) -> TradesT:
        """Return trade records filtered for short trades.

        Args:
            **kwargs: Keyword arguments for `Trades.apply_mask`.

        Returns:
            Trades: New instance of trade records containing only short trades.
        """
        filter_mask = self.get_field_arr("direction") == TradeDirection.Short
        return self.apply_mask(filter_mask, **kwargs)

    # ############# Stats ############# #

    def get_winning(self: TradesT, **kwargs) -> TradesT:
        """Return winning trades with positive profit and loss.

        Args:
            **kwargs: Keyword arguments for `Trades.apply_mask`.

        Returns:
            Trades: Trade records where profit and loss (pnl) is greater than 0.
        """
        filter_mask = self.get_field_arr("pnl") > 0.0
        return self.apply_mask(filter_mask, **kwargs)

    def get_losing(self: TradesT, **kwargs) -> TradesT:
        """Return losing trades with negative profit and loss.

        Args:
            **kwargs: Keyword arguments for `Trades.apply_mask`.

        Returns:
            Trades: Trade records where profit and loss (pnl) is less than 0.
        """
        filter_mask = self.get_field_arr("pnl") < 0.0
        return self.apply_mask(filter_mask, **kwargs)

    def get_winning_streak(self, **kwargs) -> MappedArray:
        """Return the winning streak count for each trade in the current column.

        Args:
            **kwargs: Keyword arguments for `Trades.apply`.

        Returns:
            MappedArray: Array of winning streak counts for each trade.

        See:
            `vectorbtpro.portfolio.nb.records.trade_winning_streak_nb`
        """
        return self.apply(nb.trade_winning_streak_nb, dtype=int_, **kwargs)

    def get_losing_streak(self, **kwargs) -> MappedArray:
        """Return the losing streak count for each trade in the current column.

        Args:
            **kwargs: Keyword arguments for `Trades.apply`.

        Returns:
            MappedArray: Array of losing streak counts for each trade.

        See:
            `vectorbtpro.portfolio.nb.records.trade_losing_streak_nb`
        """
        return self.apply(nb.trade_losing_streak_nb, dtype=int_, **kwargs)

    def get_win_rate(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the rate of winning trades.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.reduce`.

        Returns:
            MaybeSeries: Series containing the winning trade rates.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="win_rate"), wrap_kwargs)
        return self.get_map_field("pnl").reduce(
            nb.win_rate_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_profit_factor(
        self,
        use_returns: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the profit factor based on trade returns or profit and loss.

        Args:
            use_returns (bool): Whether to use trade returns instead of profit and loss.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.reduce`.

        Returns:
            MaybeSeries: Series containing profit factor values.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="profit_factor"), wrap_kwargs)
        if use_returns:
            mapped_arr = self.get_map_field("return")
        else:
            mapped_arr = self.get_map_field("pnl")
        return mapped_arr.reduce(
            nb.profit_factor_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_expectancy(
        self,
        use_returns: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the average profitability (expectancy) from trades.

        Args:
            use_returns (bool): Whether to use trade returns instead of profit and loss.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.reduce`.

        Returns:
            MaybeSeries: Series containing expectancy values.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="expectancy"), wrap_kwargs)
        if use_returns:
            mapped_arr = self.get_map_field("return")
        else:
            mapped_arr = self.get_map_field("pnl")
        return mapped_arr.reduce(
            nb.expectancy_reduce_nb,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_sqn(
        self,
        ddof: int = 1,
        use_returns: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the System Quality Number (SQN) based on trades.

        Args:
            ddof (int): Delta degrees of freedom.
            use_returns (bool): Whether to use trade returns instead of profit and loss.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.reduce`.

        Returns:
            MaybeSeries: Series containing SQN values.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="sqn"), wrap_kwargs)
        if use_returns:
            mapped_arr = self.get_map_field("return")
        else:
            mapped_arr = self.get_map_field("pnl")
        return mapped_arr.reduce(
            nb.sqn_reduce_nb,
            ddof,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_best_price(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        **kwargs,
    ) -> MappedArray:
        """Return the best price computed from trade data.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            **kwargs: Keyword arguments for `Trades.apply`.

        Returns:
            MappedArray: Array of best prices.

        See:
            `vectorbtpro.portfolio.nb.records.best_price_nb`
        """
        return self.apply(
            nb.best_price_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            max_duration,
            **kwargs,
        )

    def get_worst_price(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        **kwargs,
    ) -> MappedArray:
        """Return the worst price computed from trade data.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            **kwargs: Keyword arguments for `Trades.apply`.

        Returns:
            MappedArray: Array of worst prices.

        See:
            `vectorbtpro.portfolio.nb.records.worst_price_nb`
        """
        return self.apply(
            nb.worst_price_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            max_duration,
            **kwargs,
        )

    def get_best_price_idx(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        relative: bool = True,
        **kwargs,
    ) -> MappedArray:
        """Return the index of the best price for each trade.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            relative (bool): Whether to return a relative index.
            **kwargs: Keyword arguments for `Trades.apply`.

        Returns:
            MappedArray: Array of indices for the best price locations.

        See:
            `vectorbtpro.portfolio.nb.records.best_price_idx_nb`
        """
        return self.apply(
            nb.best_price_idx_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            max_duration,
            relative,
            dtype=int_,
            **kwargs,
        )

    def get_worst_price_idx(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        relative: bool = True,
        **kwargs,
    ) -> MappedArray:
        """Return the index of the worst price for each trade.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            relative (bool): Whether to return a relative index.
            **kwargs: Keyword arguments for `Trades.apply`.

        Returns:
            MappedArray: Array of indices for the worst price locations.

        See:
            `vectorbtpro.portfolio.nb.records.worst_price_idx_nb`
        """
        return self.apply(
            nb.worst_price_idx_nb,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open,
            exit_price_close,
            max_duration,
            relative,
            dtype=int_,
            **kwargs,
        )

    def get_expanding_best_price(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        clean_index_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return an expanding best price series.

        Computes the expanding best price over the duration of trades using the corresponding nb function.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Wrapped Series or DataFrame of expanding best prices.

        See:
            `vectorbtpro.portfolio.nb.records.expanding_best_price_nb`
        """
        func = jit_reg.resolve_option(nb.expanding_best_price_nb, jitted)
        out = func(
            self.values,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
        )
        if clean_index_kwargs is None:
            clean_index_kwargs = {}
        new_columns = stack_indexes(
            (
                self.wrapper.columns[self.get_field_arr("col")],
                pd.Index(self.get_field_arr("id"), name="id"),
            ),
            **clean_index_kwargs,
        )
        if wrap_kwargs is None:
            wrap_kwargs = {}
        return self.wrapper.wrap(
            out,
            group_by=False,
            index=pd.RangeIndex(stop=len(out)),
            columns=new_columns,
            **wrap_kwargs,
        )

    def get_expanding_worst_price(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        clean_index_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return an expanding worst price series.

        Computes the expanding worst price over the duration of trades using the corresponding nb function.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Wrapped Series or DataFrame of expanding worst prices.

        See:
            `vectorbtpro.portfolio.nb.records.expanding_worst_price_nb`
        """
        func = jit_reg.resolve_option(nb.expanding_worst_price_nb, jitted)
        out = func(
            self.values,
            self._open,
            self._high,
            self._low,
            self._close,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
        )
        if clean_index_kwargs is None:
            clean_index_kwargs = {}
        new_columns = stack_indexes(
            (
                self.wrapper.columns[self.get_field_arr("col")],
                pd.Index(self.get_field_arr("id"), name="id"),
            ),
            **clean_index_kwargs,
        )
        if wrap_kwargs is None:
            wrap_kwargs = {}
        return self.wrapper.wrap(
            out,
            group_by=False,
            index=pd.RangeIndex(stop=len(out)),
            columns=new_columns,
            **wrap_kwargs,
        )

    def get_mfe(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        use_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Return the maximum favorable excursion (MFE) values.

        Computes the MFE for trades based on price movements and trade directions.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            use_returns (bool): Flag indicating whether to compute using return-based values.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Trades.map_array`.

        Returns:
            MappedArray: Mapped array containing the computed MFE values.

        See:
            `vectorbtpro.portfolio.nb.records.mfe_nb`
        """
        best_price = self.resolve_shortcut_attr(
            "best_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            jitted=jitted,
            chunked=chunked,
        )
        func = jit_reg.resolve_option(nb.mfe_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        mfe = func(
            self.get_field_arr("size"),
            self.get_field_arr("direction"),
            self.get_field_arr("entry_price"),
            best_price.values,
            use_returns=use_returns,
        )
        return self.map_array(mfe, **kwargs)

    def get_mae(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        use_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Return the maximum adverse excursion (MAE) values.

        Computes the MAE for trades based on price movements and trade directions.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            use_returns (bool): Flag indicating whether to compute using return-based values.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Trades.map_array`.

        Returns:
            MappedArray: Mapped array containing the computed MAE values.

        See:
            `vectorbtpro.portfolio.nb.records.mae_nb`
        """
        worst_price = self.resolve_shortcut_attr(
            "worst_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            jitted=jitted,
            chunked=chunked,
        )
        func = jit_reg.resolve_option(nb.mae_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        mae = func(
            self.get_field_arr("size"),
            self.get_field_arr("direction"),
            self.get_field_arr("entry_price"),
            worst_price.values,
            use_returns=use_returns,
        )
        return self.map_array(mae, **kwargs)

    def get_expanding_mfe(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        use_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Return an expanding MFE series.

        Computes the expanding maximum favorable excursion (MFE) using an expanding best price as a reference.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            use_returns (bool): Flag indicating whether to compute using return-based values.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Trades.get_expanding_best_price`.

        Returns:
            SeriesFrame: Wrapped Series or DataFrame containing the computed expanding MFE values.

        See:
            `vectorbtpro.portfolio.nb.records.expanding_mfe_nb`
        """
        expanding_best_price = self.resolve_shortcut_attr(
            "expanding_best_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            jitted=jitted,
            **kwargs,
        )
        func = jit_reg.resolve_option(nb.expanding_mfe_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.values,
            expanding_best_price.values,
            use_returns=use_returns,
        )
        return ArrayWrapper.from_obj(expanding_best_price).wrap(out)

    def get_expanding_mae(
        self,
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        use_returns: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Return an expanding MAE series.

        Computes the expanding maximum adverse excursion (MAE) using an expanding worst price as a reference.

        Args:
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            use_returns (bool): Flag indicating whether to compute using return-based values.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Trades.get_expanding_worst_price`.

        Returns:
            SeriesFrame: Wrapped Series or DataFrame containing the computed expanding MAE values.

        See:
            `vectorbtpro.portfolio.nb.records.expanding_mae_nb`
        """
        expanding_worst_price = self.resolve_shortcut_attr(
            "expanding_worst_price",
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            jitted=jitted,
            **kwargs,
        )
        func = jit_reg.resolve_option(nb.expanding_mae_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.values,
            expanding_worst_price.values,
            use_returns=use_returns,
        )
        return ArrayWrapper.from_obj(expanding_worst_price).wrap(out)

    def get_edge_ratio(
        self,
        volatility: tp.Optional[tp.ArrayLike] = None,
        window: int = 14,
        wtype: tp.Union[int, str] = "wilder",
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute edge ratio.

        Args:
            volatility (Optional[ArrayLike]): Volatility values used in the edge ratio calculation.

                If None, an ATR is computed when high and low are available;
                otherwise, a rolling standard deviation is used.
            window (int): Window size for the volatility calculation.
            wtype (Union[int, str]): Weighting type.

                Mapped using `vectorbtpro.generic.enums.WType` if provided as a string.
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            SeriesFrame: Computed edge ratio.

        See:
            `vectorbtpro.portfolio.nb.records.edge_ratio_nb`
        """
        if self._close is None:
            raise ValueError("Must provide close")

        if volatility is None:
            if isinstance(wtype, str):
                wtype = map_enum_fields(wtype, WType)
            if self._high is not None and self._low is not None:
                from vectorbtpro.indicators.nb import atr_nb

                if self._high is None or self._low is None:
                    raise ValueError("Must provide high and low for ATR calculation")

                volatility = atr_nb(
                    high=to_2d_array(self._high),
                    low=to_2d_array(self._low),
                    close=to_2d_array(self._close),
                    window=window,
                    wtype=wtype,
                )[1]
            else:
                from vectorbtpro.indicators.nb import msd_nb

                volatility = msd_nb(
                    close=to_2d_array(self._close),
                    window=window,
                    wtype=wtype,
                )
        else:
            volatility = broadcast_to(volatility, self.wrapper, to_pd=False, keep_flex=True)
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.edge_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.values,
            col_map,
            self._open,
            self._high,
            self._low,
            self._close,
            volatility,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
        )
        if wrap_kwargs is None:
            wrap_kwargs = {}
        return self.wrapper.wrap_reduced(out, group_by=group_by, **wrap_kwargs)

    def get_running_edge_ratio(
        self,
        volatility: tp.Optional[tp.ArrayLike] = None,
        window: int = 14,
        wtype: tp.Union[int, str] = "wilder",
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        incl_shorter: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute running edge ratio.

        Args:
            volatility (Optional[ArrayLike]): Volatility values used in the edge ratio calculation.

                If None, an ATR is computed when high and low are available;
                otherwise, a rolling standard deviation is used.
            window (int): Window size for the volatility calculation.
            wtype (Union[int, str]): Weighting type.

                Mapped using `vectorbtpro.generic.enums.WType` if provided as a string.
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            incl_shorter (bool): Whether to include trades shorter than the current duration step.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Computed running edge ratio.

        See:
            `vectorbtpro.portfolio.nb.records.running_edge_ratio_nb`
        """
        if self._close is None:
            raise ValueError("Must provide close")

        if volatility is None:
            if isinstance(wtype, str):
                wtype = map_enum_fields(wtype, WType)
            if self._high is not None and self._low is not None:
                from vectorbtpro.indicators.nb import atr_nb

                if self._high is None or self._low is None:
                    raise ValueError("Must provide high and low for ATR calculation")

                volatility = atr_nb(
                    high=to_2d_array(self._high),
                    low=to_2d_array(self._low),
                    close=to_2d_array(self._close),
                    window=window,
                    wtype=wtype,
                )[1]
            else:
                from vectorbtpro.indicators.nb import msd_nb

                volatility = msd_nb(
                    close=to_2d_array(self._close),
                    window=window,
                    wtype=wtype,
                )
        else:
            volatility = broadcast_to(volatility, self.wrapper, to_pd=False, keep_flex=True)
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.running_edge_ratio_nb, jitted)
        out = func(
            self.values,
            col_map,
            self._open,
            self._high,
            self._low,
            self._close,
            volatility,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            incl_shorter=incl_shorter,
        )
        if wrap_kwargs is None:
            wrap_kwargs = {}
        return self.wrapper.wrap(
            out, group_by=group_by, index=pd.RangeIndex(stop=len(out)), **wrap_kwargs
        )

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `Trades.stats`.

        Merges the defaults from `vectorbtpro.generic.ranges.Ranges.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.trades`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        trades_stats_cfg = settings["trades"]["stats"]

        return merge_dicts(Ranges.stats_defaults.__get__(self), trades_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            first_trade_start=dict(
                title="First Trade Start",
                calc_func="entry_idx.min",
                wrap_kwargs=dict(to_index=True),
                tags=["trades", "index"],
            ),
            last_trade_end=dict(
                title="Last Trade End",
                calc_func="exit_idx.max",
                wrap_kwargs=dict(to_index=True),
                tags=["trades", "index"],
            ),
            coverage=dict(
                title="Coverage",
                calc_func="coverage",
                overlapping=False,
                normalize=False,
                apply_to_timedelta=True,
                tags=["ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="coverage",
                overlapping=True,
                normalize=False,
                apply_to_timedelta=True,
                tags=["ranges", "coverage"],
            ),
            total_records=dict(title="Total Records", calc_func="count", tags="records"),
            total_long_trades=dict(
                title="Total Long Trades", calc_func="direction_long.count", tags=["trades", "long"]
            ),
            total_short_trades=dict(
                title="Total Short Trades",
                calc_func="direction_short.count",
                tags=["trades", "short"],
            ),
            total_closed_trades=dict(
                title="Total Closed Trades",
                calc_func="status_closed.count",
                tags=["trades", "closed"],
            ),
            total_open_trades=dict(
                title="Total Open Trades", calc_func="status_open.count", tags=["trades", "open"]
            ),
            open_trade_pnl=dict(
                title="Open Trade PnL", calc_func="status_open.pnl.sum", tags=["trades", "open"]
            ),
            win_rate=dict(
                title="Win Rate [%]",
                calc_func="status_closed.get_win_rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            winning_streak=dict(
                title="Max Win Streak",
                calc_func=RepEval(
                    "'winning_streak.max' if incl_open else 'status_closed.winning_streak.max'"
                ),
                wrap_kwargs=dict(dtype=pd.Int64Dtype()),
                tags=RepEval("['trades', *incl_open_tags, 'streak']"),
            ),
            losing_streak=dict(
                title="Max Loss Streak",
                calc_func=RepEval(
                    "'losing_streak.max' if incl_open else 'status_closed.losing_streak.max'"
                ),
                wrap_kwargs=dict(dtype=pd.Int64Dtype()),
                tags=RepEval("['trades', *incl_open_tags, 'streak']"),
            ),
            best_trade=dict(
                title="Best Trade [%]",
                calc_func=RepEval("'returns.max' if incl_open else 'status_closed.returns.max'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            worst_trade=dict(
                title="Worst Trade [%]",
                calc_func=RepEval("'returns.min' if incl_open else 'status_closed.returns.min'"),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            avg_winning_trade=dict(
                title="Avg Winning Trade [%]",
                calc_func=RepEval(
                    "'winning.returns.mean' if incl_open else 'status_closed.winning.returns.mean'"
                ),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags, 'winning']"),
            ),
            avg_losing_trade=dict(
                title="Avg Losing Trade [%]",
                calc_func=RepEval(
                    "'losing.returns.mean' if incl_open else 'status_closed.losing.returns.mean'"
                ),
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['trades', *incl_open_tags, 'losing']"),
            ),
            avg_winning_trade_duration=dict(
                title="Avg Winning Trade Duration",
                calc_func=RepEval(
                    "'winning.avg_duration' if incl_open else 'status_closed.winning.get_avg_duration'"
                ),
                fill_wrap_kwargs=True,
                tags=RepEval("['trades', *incl_open_tags, 'winning', 'duration']"),
            ),
            avg_losing_trade_duration=dict(
                title="Avg Losing Trade Duration",
                calc_func=RepEval(
                    "'losing.avg_duration' if incl_open else 'status_closed.losing.get_avg_duration'"
                ),
                fill_wrap_kwargs=True,
                tags=RepEval("['trades', *incl_open_tags, 'losing', 'duration']"),
            ),
            profit_factor=dict(
                title="Profit Factor",
                calc_func=RepEval(
                    "'profit_factor' if incl_open else 'status_closed.get_profit_factor'"
                ),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            expectancy=dict(
                title="Expectancy",
                calc_func=RepEval("'expectancy' if incl_open else 'status_closed.get_expectancy'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            sqn=dict(
                title="SQN",
                calc_func=RepEval("'sqn' if incl_open else 'status_closed.get_sqn'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
            edge_ratio=dict(
                title="Edge Ratio",
                calc_func=RepEval("'edge_ratio' if incl_open else 'status_closed.get_edge_ratio'"),
                tags=RepEval("['trades', *incl_open_tags]"),
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_pnl(
        self,
        column: tp.Optional[tp.Column] = None,
        group_by: tp.GroupByLike = False,
        pct_scale: bool = False,
        marker_size_range: tp.Tuple[float, float] = (7, 14),
        opacity_range: tp.Tuple[float, float] = (0.75, 0.9),
        closed_trace_kwargs: tp.KwargsLike = None,
        closed_profit_trace_kwargs: tp.KwargsLike = None,
        closed_loss_trace_kwargs: tp.KwargsLike = None,
        open_trace_kwargs: tp.KwargsLike = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot trades' profit and loss (PnL) or returns.

        Args:
            column (Optional[Column]): Identifier of the column or group to plot.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            pct_scale (bool): Flag to display the y-axis on a percentage scale.
            marker_size_range (Tuple[float, float]): Range for marker sizes.
            opacity_range (Tuple[float, float]): Range for marker opacities.
            closed_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for closed trade markers.
            closed_profit_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for closed trade with profit markers.
            closed_loss_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for closed trade with loss markers.
            open_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for open trade markers.
            hline_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for the horizontal line.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Plotly figure object containing the plot of trade PnL or returns.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> index = pd.date_range("2020", periods=7)
            >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.], index=index)
            >>> orders = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.trades.plot_pnl().show()
            ```

            ![](/assets/images/api/trades_plot_pnl.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/trades_plot_pnl.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import get_domain, make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=group_by)

        if closed_trace_kwargs is None:
            closed_trace_kwargs = {}
        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        marker_size_range = tuple(marker_size_range)
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]

        if fig is None:
            fig = make_figure()
        def_layout_kwargs = {xaxis: {}, yaxis: {}}
        if pct_scale:
            def_layout_kwargs[yaxis]["tickformat"] = ".2%"
            def_layout_kwargs[yaxis]["title"] = "Return"
        else:
            def_layout_kwargs[yaxis]["title"] = "PnL"
        fig.update_layout(**def_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        y_domain = get_domain(yref, fig)

        if self_col.count() > 0:
            # Extract information
            exit_idx = self_col.get_map_field_to_index("exit_idx")
            pnl = self_col.get_field_arr("pnl")
            returns = self_col.get_field_arr("return")
            status = self_col.get_field_arr("status")

            valid_mask = ~np.isnan(returns)
            neutral_mask = (pnl == 0) & valid_mask
            profit_mask = (pnl > 0) & valid_mask
            loss_mask = (pnl < 0) & valid_mask

            marker_size = min_rel_rescale(np.abs(returns), marker_size_range)
            opacity = max_rel_rescale(np.abs(returns), opacity_range)

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask, name, color, kwargs):
                if np.any(mask):
                    if self_col.get_field_setting("parent_id", "ignore", False):
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    else:
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "parent_id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    _kwargs = merge_dicts(
                        dict(
                            x=exit_idx[mask],
                            y=returns[mask] if pct_scale else pnl[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=marker_size[mask],
                                opacity=opacity[mask],
                                line=dict(width=1, color=adjust_lightness(color)),
                            ),
                            name=name,
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Neutral scatter
            _plot_scatter(
                neutral_mask,
                "Closed",
                plotting_cfg["contrast_color_schema"]["gray"],
                closed_trace_kwargs,
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                "Closed - Profit",
                plotting_cfg["contrast_color_schema"]["green"],
                closed_profit_trace_kwargs,
            )

            # Plot Closed - Loss scatter
            _plot_scatter(
                closed_loss_mask,
                "Closed - Loss",
                plotting_cfg["contrast_color_schema"]["red"],
                closed_loss_trace_kwargs,
            )

            # Plot Open scatter
            _plot_scatter(
                open_mask,
                "Open",
                plotting_cfg["contrast_color_schema"]["orange"],
                open_trace_kwargs,
            )

        # Plot zeroline
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot_returns(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot trade returns using `Trades.plot_pnl`.

        Args:
            *args: Positional arguments for `Trades.plot_pnl`.
            **kwargs: Keyword arguments for `Trades.plot_pnl`.

        Returns:
            BaseFigure: Plotly figure object containing the plot of trade returns.
        """
        return self.plot_pnl(
            *args,
            pct_scale=True,
            **kwargs,
        )

    def plot_against_pnl(
        self,
        field: tp.Union[str, tp.Array1d, MappedArray],
        field_label: tp.Optional[str] = None,
        column: tp.Optional[tp.Column] = None,
        group_by: tp.GroupByLike = False,
        pct_scale: bool = False,
        field_pct_scale: bool = False,
        closed_trace_kwargs: tp.KwargsLike = None,
        closed_profit_trace_kwargs: tp.KwargsLike = None,
        closed_loss_trace_kwargs: tp.KwargsLike = None,
        open_trace_kwargs: tp.KwargsLike = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        vline_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot a field against PnL or returns.

        Args:
            field (Union[str, Array1d, MappedArray]): Field to be plotted.

                The field can be specified as a string, a mapped array, or a 1-dimensional array.
            field_label (Optional[str]): Label for the field.
            column (Optional[Column]): Identifier of the column or group to plot.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            pct_scale (bool): Flag to display the y-axis on a percentage scale.
            field_pct_scale (bool): Flag to display the y-axis on a percentage scale.
            closed_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for closed trade markers.
            closed_profit_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for closed trade with profit markers.
            closed_loss_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for closed trade with loss markers.
            open_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for open trade markers.
            hline_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for the horizontal line.
            vline_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for the vertical line.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure with the plotted field against PnL or returns.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> index = pd.date_range("2020", periods=10)
            >>> price = pd.Series([1., 2., 3., 4., 5., 6., 5., 3., 2., 1.], index=index)
            >>> orders = pd.Series([1., -0.5, 0., -0.5, 2., 0., -0.5, -0.5, 0., -0.5], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> trades = pf.trades
            >>> trades.plot_against_pnl("MFE").show()
            ```

            ![](/assets/images/api/trades_plot_against_pnl.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/trades_plot_against_pnl.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import get_domain, make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=group_by)

        if closed_trace_kwargs is None:
            closed_trace_kwargs = {}
        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]

        if isinstance(field, str):
            if field_label is None:
                field_label = field
            field = getattr(self_col, field.lower())
        if isinstance(field, MappedArray):
            field = field.values
        if field_label is None:
            field_label = "Field"

        if fig is None:
            fig = make_figure()
        def_layout_kwargs = {xaxis: {}, yaxis: {}}
        if pct_scale:
            def_layout_kwargs[xaxis]["tickformat"] = ".2%"
            def_layout_kwargs[xaxis]["title"] = "Return"
        else:
            def_layout_kwargs[xaxis]["title"] = "PnL"
        if field_pct_scale:
            def_layout_kwargs[yaxis]["tickformat"] = ".2%"
        def_layout_kwargs[yaxis]["title"] = field_label
        fig.update_layout(**def_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        y_domain = get_domain(yref, fig)

        if self_col.count() > 0:
            # Extract information
            pnl = self_col.get_field_arr("pnl")
            returns = self_col.get_field_arr("return")
            status = self_col.get_field_arr("status")

            valid_mask = ~np.isnan(returns)
            neutral_mask = (pnl == 0) & valid_mask
            profit_mask = (pnl > 0) & valid_mask
            loss_mask = (pnl < 0) & valid_mask

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask, name, color, kwargs):
                if np.any(mask):
                    if self_col.get_field_setting("parent_id", "ignore", False):
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    else:
                        customdata, hovertemplate = self_col.prepare_customdata(
                            incl_fields=["id", "parent_id", "exit_idx", "pnl", "return"], mask=mask
                        )
                    _kwargs = merge_dicts(
                        dict(
                            x=returns[mask] if pct_scale else pnl[mask],
                            y=field[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=7,
                                line=dict(width=1, color=adjust_lightness(color)),
                            ),
                            name=name,
                            customdata=customdata,
                            hovertemplate=hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Neutral scatter
            _plot_scatter(
                neutral_mask,
                "Closed",
                plotting_cfg["contrast_color_schema"]["gray"],
                closed_trace_kwargs,
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                "Closed - Profit",
                plotting_cfg["contrast_color_schema"]["green"],
                closed_profit_trace_kwargs,
            )

            # Plot Closed - Loss scatter
            _plot_scatter(
                closed_loss_mask,
                "Closed - Loss",
                plotting_cfg["contrast_color_schema"]["red"],
                closed_loss_trace_kwargs,
            )

            # Plot Open scatter
            _plot_scatter(
                open_mask,
                "Open",
                plotting_cfg["contrast_color_schema"]["orange"],
                open_trace_kwargs,
            )

        # Plot zerolines
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                hline_shape_kwargs,
            )
        )
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref=xref,
                    yref="paper",
                    x0=0,
                    y0=y_domain[0],
                    x1=0,
                    y1=y_domain[1],
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                vline_shape_kwargs,
            )
        )
        return fig

    def plot_mfe(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot MFE.

        Args:
            *args: Positional arguments for `Trades.plot_against_pnl`.
            **kwargs: Keyword arguments for `Trades.plot_against_pnl`.

        Returns:
            BaseFigure: Updated figure with the MFE plot.
        """
        return self.plot_against_pnl(
            *args,
            field="mfe",
            field_label="MFE",
            **kwargs,
        )

    def plot_mfe_returns(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot MFE returns.

        Args:
            *args: Positional arguments for `Trades.plot_against_pnl`.
            **kwargs: Keyword arguments for `Trades.plot_against_pnl`.

        Returns:
            BaseFigure: Figure object with the plotted MFE returns.
        """
        return self.plot_against_pnl(
            *args,
            field="mfe_returns",
            field_label="MFE Return",
            pct_scale=True,
            field_pct_scale=True,
            **kwargs,
        )

    def plot_mae(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot MAE.

        Args:
            *args: Positional arguments for `Trades.plot_against_pnl`.
            **kwargs: Keyword arguments for `Trades.plot_against_pnl`.

        Returns:
            BaseFigure: Figure object with the plotted MAE.
        """
        return self.plot_against_pnl(
            *args,
            field="mae",
            field_label="MAE",
            **kwargs,
        )

    def plot_mae_returns(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot MAE returns.

        Args:
            *args: Positional arguments for `Trades.plot_against_pnl`.
            **kwargs: Keyword arguments for `Trades.plot_against_pnl`.

        Returns:
            BaseFigure: Figure object with the plotted MAE returns.
        """
        return self.plot_against_pnl(
            *args,
            field="mae_returns",
            field_label="MAE Return",
            pct_scale=True,
            field_pct_scale=True,
            **kwargs,
        )

    def plot_expanding(
        self,
        field: tp.Union[str, tp.Array1d, MappedArray],
        field_label: tp.Optional[str] = None,
        column: tp.Optional[tp.Column] = None,
        group_by: tp.GroupByLike = False,
        plot_bands: bool = False,
        colorize: tp.Union[bool, str, tp.Callable] = "last",
        field_pct_scale: bool = False,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot projections of an expanding field.

        Args:
            field (Union[str, Array1d, MappedArray]): Field to be plotted.

                Can also be provided as a two-dimensional array.
            field_label (Optional[str]): Label for the field.
            column (Optional[Column]): Identifier of the column or group to plot.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            plot_bands (bool): Plot computed bands if True.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            colorize (Union[bool, str, Callable]): Strategy for colorizing projections or bands.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            field_pct_scale (bool): Flag to display the y-axis on a percentage scale.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.

        Returns:
            BaseFigure: Figure object with the plotted expanding field projections.

        Examples:
            ```pycon
            >>> index = pd.date_range("2020", periods=10)
            >>> price = pd.Series([1., 2., 3., 2., 4., 5., 6., 5., 6., 7.], index=index)
            >>> orders = pd.Series([1., 0., 0., -2., 0., 0., 2., 0., 0., -1.], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.trades.plot_expanding("MFE").show()
            ```

            ![](/assets/images/api/trades_plot_expanding.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/trades_plot_expanding.dark.svg#only-dark){: .iimg loading=lazy }
        """
        if column is not None:
            self_col = self.select_col(column=column, group_by=group_by)
        else:
            self_col = self

        if isinstance(field, str):
            if field_label is None:
                field_label = field
            if not field.lower().startswith("expanding_"):
                field = "expanding_" + field
            field = getattr(self_col, field.lower())
        if isinstance(field, MappedArray):
            field = field.values
        if field_label is None:
            field_label = "Field"
        field = to_pd_array(field)

        fig = field.vbt.plot_projections(
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            plot_bands=plot_bands,
            colorize=colorize,
            **kwargs,
        )
        xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]
        if field_label is not None and "title" not in kwargs.get(yaxis, {}):
            fig.update_layout(**{yaxis: dict(title=field_label)})
        if field_pct_scale and "tickformat" not in kwargs.get(yaxis, {}):
            fig.update_layout(**{yaxis: dict(tickformat=".2%")})
        return fig

    def plot_expanding_mfe(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot expanding MFE.

        Args:
            *args: Positional arguments for `Trades.plot_expanding`.
            **kwargs: Keyword arguments for `Trades.plot_expanding`.

        Returns:
            BaseFigure: Figure object with the plotted expanding MFE.
        """
        return self.plot_expanding(
            *args,
            field="expanding_mfe",
            field_label="MFE",
            **kwargs,
        )

    def plot_expanding_mfe_returns(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot expanding MFE returns.

        Args:
            *args: Positional arguments for `Trades.plot_expanding`.
            **kwargs: Keyword arguments for `Trades.plot_expanding`.

        Returns:
            BaseFigure: Figure object with the plotted expanding MFE returns.
        """
        return self.plot_expanding(
            *args,
            field="expanding_mfe_returns",
            field_label="MFE Return",
            field_pct_scale=True,
            **kwargs,
        )

    def plot_expanding_mae(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot expanding MAE.

        Args:
            *args: Positional arguments for `Trades.plot_expanding`.
            **kwargs: Keyword arguments for `Trades.plot_expanding`.

        Returns:
            BaseFigure: Figure object with the plotted expanding MAE.
        """
        return self.plot_expanding(
            *args,
            field="expanding_mae",
            field_label="MAE",
            **kwargs,
        )

    def plot_expanding_mae_returns(self, *args, **kwargs) -> tp.BaseFigure:
        """Plot expanding MAE returns.

        Args:
            *args: Positional arguments for `Trades.plot_expanding`.
            **kwargs: Keyword arguments for `Trades.plot_expanding`.

        Returns:
            BaseFigure: Figure object with the plotted expanding MAE returns.
        """
        return self.plot_expanding(
            *args,
            field="expanding_mae_returns",
            field_label="MAE Return",
            field_pct_scale=True,
            **kwargs,
        )

    def plot_running_edge_ratio(
        self,
        column: tp.Optional[tp.Column] = None,
        volatility: tp.Optional[tp.ArrayLike] = None,
        window: int = 14,
        wtype: tp.Union[int, str] = "wilder",
        entry_price_open: bool = False,
        exit_price_close: bool = False,
        max_duration: tp.Optional[int] = None,
        incl_shorter: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot the running edge ratio for a specified column or group.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            volatility (Optional[ArrayLike]): Volatility values used in the edge ratio calculation.

                If None, an ATR is computed when high and low are available;
                otherwise, a rolling standard deviation is used.
            window (int): Window size for the volatility calculation.
            wtype (Union[int, str]): Weighting type.

                Mapped using `vectorbtpro.generic.enums.WType` if provided as a string.
            entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
            exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
            max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
            incl_shorter (bool): Whether to include trades shorter than the current duration step.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            hline_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for the horizontal line.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`.

        Returns:
            BaseFigure: Figure object with the plotted running edge ratio.
        """
        from vectorbtpro.utils.figure import get_domain

        running_edge_ratio = self.resolve_shortcut_attr(
            "running_edge_ratio",
            volatility=volatility,
            window=window,
            wtype=wtype,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            incl_shorter=incl_shorter,
            group_by=group_by,
            jitted=jitted,
        )
        running_edge_ratio = self.select_col_from_obj(
            running_edge_ratio, column, wrapper=self.wrapper.regroup(group_by)
        )
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(name="Edge Ratio"),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = running_edge_ratio.vbt.plot_against(1, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=1.0,
                    x1=x_domain[1],
                    y1=1.0,
                ),
                hline_shape_kwargs,
            )
        )
        return fig

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_ohlc: bool = True,
        plot_close: bool = True,
        plot_markers: bool = True,
        plot_zones: bool = True,
        plot_by_type: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        entry_trace_kwargs: tp.KwargsLike = None,
        exit_trace_kwargs: tp.KwargsLike = None,
        exit_profit_trace_kwargs: tp.KwargsLike = None,
        exit_loss_trace_kwargs: tp.KwargsLike = None,
        active_trace_kwargs: tp.KwargsLike = None,
        profit_shape_kwargs: tp.KwargsLike = None,
        loss_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot trades on a Plotly figure.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_ohlc (bool): Whether to plot the OHLC data.
            plot_close (bool): Whether to plot the close price if OHLC data is not plotted.
            plot_markers (bool): Display markers for trade entry and exit points.
            plot_zones (bool): Display zones indicating profit and loss areas.
            plot_by_type (bool): Plot exit markers categorized by trade type (neutral, profit, or loss).

                If False, marker appearance is controlled using `exit_trace_kwargs`.
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            ohlc_trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the close price.
            entry_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for entry markers.
            exit_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for exit markers.
            exit_profit_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for profit exit markers.
            exit_loss_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for loss exit markers.
            active_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for active trade markers.
            profit_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for profit zone shapes.
            loss_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for loss zone shapes.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated Plotly figure with the plotted trades.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> index = pd.date_range("2020", periods=7)
            >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.], index=index)
            >>> size = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, size)
            >>> pf.trades.plot().show()
            ```

            ![](/assets/images/api/trades_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/trades_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if entry_trace_kwargs is None:
            entry_trace_kwargs = {}
        if exit_trace_kwargs is None:
            exit_trace_kwargs = {}
        if exit_profit_trace_kwargs is None:
            exit_profit_trace_kwargs = {}
        if exit_loss_trace_kwargs is None:
            exit_loss_trace_kwargs = {}
        if active_trace_kwargs is None:
            active_trace_kwargs = {}
        if profit_shape_kwargs is None:
            profit_shape_kwargs = {}
        if loss_shape_kwargs is None:
            loss_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if (
            plot_ohlc
            and self_col._open is not None
            and self_col._high is not None
            and self_col._low is not None
            and self_col._close is not None
        ):
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            entry_idx = self_col.get_map_field_to_index("entry_idx", minus_one_to_zero=True)
            entry_price = self_col.get_field_arr("entry_price")
            exit_idx = self_col.get_map_field_to_index("exit_idx")
            exit_price = self_col.get_field_arr("exit_price")
            pnl = self_col.get_field_arr("pnl")
            status = self_col.get_field_arr("status")
            duration = to_1d_array(
                self_col.wrapper.arr_to_timedelta(
                    self_col.duration.values,
                    to_pd=True,
                    silence_warnings=True,
                ).astype(str)
            )

            if plot_markers:
                # Plot Entry markers
                if self_col.get_field_setting("parent_id", "ignore", False):
                    entry_customdata, entry_hovertemplate = self_col.prepare_customdata(
                        incl_fields=[
                            "id",
                            "entry_order_id",
                            "entry_idx",
                            "size",
                            "entry_price",
                            "entry_fees",
                            "direction",
                        ]
                    )
                else:
                    entry_customdata, entry_hovertemplate = self_col.prepare_customdata(
                        incl_fields=[
                            "id",
                            "entry_order_id",
                            "parent_id",
                            "entry_idx",
                            "size",
                            "entry_price",
                            "entry_fees",
                            "direction",
                        ]
                    )
                _entry_trace_kwargs = merge_dicts(
                    dict(
                        x=entry_idx,
                        y=entry_price,
                        mode="markers",
                        marker=dict(
                            symbol="square",
                            color=plotting_cfg["contrast_color_schema"]["blue"],
                            size=7,
                            line=dict(
                                width=1,
                                color=adjust_lightness(
                                    plotting_cfg["contrast_color_schema"]["blue"]
                                ),
                            ),
                        ),
                        name="Entry",
                        customdata=entry_customdata,
                        hovertemplate=entry_hovertemplate,
                    ),
                    entry_trace_kwargs,
                )
                entry_scatter = go.Scatter(**_entry_trace_kwargs)
                fig.add_trace(entry_scatter, **add_trace_kwargs)

                # Plot end markers
                def _plot_end_markers(mask, name, color, kwargs, incl_status=False) -> None:
                    if np.any(mask):
                        if self_col.get_field_setting("parent_id", "ignore", False):
                            exit_customdata, exit_hovertemplate = self_col.prepare_customdata(
                                incl_fields=[
                                    "id",
                                    "exit_order_id",
                                    "exit_idx",
                                    "size",
                                    "exit_price",
                                    "exit_fees",
                                    "pnl",
                                    "return",
                                    "direction",
                                    *(("status",) if incl_status else ()),
                                ],
                                append_info=[(duration, "Duration")],
                                mask=mask,
                            )
                        else:
                            exit_customdata, exit_hovertemplate = self_col.prepare_customdata(
                                incl_fields=[
                                    "id",
                                    "exit_order_id",
                                    "parent_id",
                                    "exit_idx",
                                    "size",
                                    "exit_price",
                                    "exit_fees",
                                    "pnl",
                                    "return",
                                    "direction",
                                    *(("status",) if incl_status else ()),
                                ],
                                append_info=[(duration, "Duration")],
                                mask=mask,
                            )
                        _kwargs = merge_dicts(
                            dict(
                                x=exit_idx[mask],
                                y=exit_price[mask],
                                mode="markers",
                                marker=dict(
                                    symbol="square",
                                    color=color,
                                    size=7,
                                    line=dict(width=1, color=adjust_lightness(color)),
                                ),
                                name=name,
                                customdata=exit_customdata,
                                hovertemplate=exit_hovertemplate,
                            ),
                            kwargs,
                        )
                        scatter = go.Scatter(**_kwargs)
                        fig.add_trace(scatter, **add_trace_kwargs)

                if plot_by_type:
                    # Plot Exit markers
                    _plot_end_markers(
                        (status == TradeStatus.Closed) & (pnl == 0.0),
                        "Exit",
                        plotting_cfg["contrast_color_schema"]["gray"],
                        exit_trace_kwargs,
                    )

                    # Plot Exit - Profit markers
                    _plot_end_markers(
                        (status == TradeStatus.Closed) & (pnl > 0.0),
                        "Exit - Profit",
                        plotting_cfg["contrast_color_schema"]["green"],
                        exit_profit_trace_kwargs,
                    )

                    # Plot Exit - Loss markers
                    _plot_end_markers(
                        (status == TradeStatus.Closed) & (pnl < 0.0),
                        "Exit - Loss",
                        plotting_cfg["contrast_color_schema"]["red"],
                        exit_loss_trace_kwargs,
                    )

                    # Plot Active markers
                    _plot_end_markers(
                        status == TradeStatus.Open,
                        "Active",
                        plotting_cfg["contrast_color_schema"]["orange"],
                        active_trace_kwargs,
                    )
                else:
                    # Plot Exit markers
                    _plot_end_markers(
                        np.full(len(status), True),
                        "Exit",
                        plotting_cfg["contrast_color_schema"]["pink"],
                        exit_trace_kwargs,
                        incl_status=True,
                    )

            if plot_zones:
                # Plot profit zones
                self_col.winning.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    add_shape_kwargs=merge_dicts(
                        dict(
                            yref=Rep("yref"),
                            y0=RepFunc(lambda record: record["entry_price"]),
                            y1=RepFunc(lambda record: record["exit_price"]),
                            fillcolor=plotting_cfg["contrast_color_schema"]["green"],
                        ),
                        profit_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

                # Plot loss zones
                self_col.losing.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    add_shape_kwargs=merge_dicts(
                        dict(
                            yref=Rep("yref"),
                            y0=RepFunc(lambda record: record["entry_price"]),
                            y1=RepFunc(lambda record: record["exit_price"]),
                            fillcolor=plotting_cfg["contrast_color_schema"]["red"],
                        ),
                        loss_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `Trades.plots`.

        Merges the defaults from `vectorbtpro.generic.ranges.Ranges.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.trades`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        trades_plots_cfg = settings["trades"]["plots"]

        return merge_dicts(Ranges.plots_defaults.__get__(self), trades_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="Trades",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="plot",
                tags="trades",
            ),
            plot_pnl=dict(
                title="Trade PnL",
                yaxis_kwargs=dict(title="Trade PnL"),
                check_is_not_grouped=True,
                plot_func="plot_pnl",
                tags="trades",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Trades.override_field_config_doc(__pdoc__)
Trades.override_metrics_doc(__pdoc__)
Trades.override_subplots_doc(__pdoc__)

# ############# EntryTrades ############# #

entry_trades_field_config = ReadonlyConfig(
    dict(
        settings={"id": dict(title="Entry Trade Id"), "idx": dict(name="entry_idx")}
    )  # remap field of Records,
)
"""_"""

__pdoc__["entry_trades_field_config"] = f"""Field configuration for `EntryTrades`.

```python
{entry_trades_field_config.prettify_doc()}
```
"""

EntryTradesT = tp.TypeVar("EntryTradesT", bound="EntryTrades")


@override_field_config(entry_trades_field_config)
class EntryTrades(Trades):
    """Class representing entry trade records, extending `Trades`.

    Field configuration is overridden using `entry_trades_field_config`.

    Requires `records_arr` to have all fields defined in `vectorbtpro.portfolio.enums.trade_dt`.
    """

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_orders(
        cls: tp.Type[EntryTradesT],
        orders: Orders,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> EntryTradesT:
        """Build an `EntryTrades` instance from `vectorbtpro.portfolio.orders.Orders`.

        Args:
            orders (vectorbtpro.portfolio.orders.Orders): Orders instance from which to derive entry trades.
            open (Optional[ArrayLike]): Open prices.

                If None, uses `orders._open`.
            high (Optional[ArrayLike]): High prices.

                If None, uses `orders._high`.
            low (Optional[ArrayLike]): Low prices.

                If None, uses `orders._low`.
            close (Optional[ArrayLike]): Close prices.

                If None, uses `orders._close`.
            init_position (ArrayLike): Initial position.
            init_price (ArrayLike): Initial position price.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `EntryTrades.from_records`.

        Returns:
            EntryTrades: Constructed `EntryTrades` instance.

        See:
            `vectorbtpro.portfolio.nb.records.get_entry_trades_nb`
        """
        if open is None:
            open = orders._open
        if high is None:
            high = orders._high
        if low is None:
            low = orders._low
        if close is None:
            close = orders._close
        func = jit_reg.resolve_option(nb.get_entry_trades_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        trade_records_arr = func(
            orders.values,
            to_2d_array(orders.wrapper.wrap(close, group_by=False)),
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
            sim_start=None if sim_start is None else to_1d_array(sim_start),
            sim_end=None if sim_end is None else to_1d_array(sim_end),
        )
        return cls.from_records(
            orders.wrapper,
            trade_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )

    def plot_signals(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_ohlc: bool = True,
        plot_close: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        long_entry_trace_kwargs: tp.KwargsLike = None,
        short_entry_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot entry trade signals.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_ohlc (bool): Whether to plot the OHLC data.
            plot_close (bool): Whether to plot the close price if OHLC data is not plotted.
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            ohlc_trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the close price.
            long_entry_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for long entry markers.
            short_entry_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for short entry markers.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated or newly created figure.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> index = pd.date_range("2020", periods=7)
            >>> price = pd.Series([1, 2, 3, 2, 3, 4, 3], index=index)
            >>> orders = pd.Series([1, 0, -1, 0, -1, 2, -2], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.entry_trades.plot_signals().show()
            ```

            ![](/assets/images/api/entry_trades_plot_signals.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/entry_trades_plot_signals.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if long_entry_trace_kwargs is None:
            long_entry_trace_kwargs = {}
        if short_entry_trace_kwargs is None:
            short_entry_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if (
            plot_ohlc
            and self_col._open is not None
            and self_col._high is not None
            and self_col._low is not None
            and self_col._close is not None
        ):
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            entry_idx = self_col.get_map_field_to_index("entry_idx", minus_one_to_zero=True)
            entry_price = self_col.get_field_arr("entry_price")
            direction = self_col.get_field_arr("direction")

            def _plot_entry_markers(mask, name, color, kwargs):
                if np.any(mask):
                    entry_customdata, entry_hovertemplate = self_col.prepare_customdata(
                        incl_fields=[
                            "id",
                            "entry_order_id",
                            "parent_id",
                            "entry_idx",
                            "size",
                            "entry_price",
                            "entry_fees",
                            "pnl",
                            "return",
                            "status",
                        ],
                        mask=mask,
                    )
                    _kwargs = merge_dicts(
                        dict(
                            x=entry_idx[mask],
                            y=entry_price[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color="rgba(0, 0, 0, 0)",
                                size=15,
                                line=dict(
                                    color=color,
                                    width=2,
                                ),
                            ),
                            name=name,
                            customdata=entry_customdata,
                            hovertemplate=entry_hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Long Entry markers
            _plot_entry_markers(
                direction == TradeDirection.Long,
                "Long Entry",
                plotting_cfg["contrast_color_schema"]["green"],
                long_entry_trace_kwargs,
            )

            # Plot Short Entry markers
            _plot_entry_markers(
                direction == TradeDirection.Short,
                "Short Entry",
                plotting_cfg["contrast_color_schema"]["red"],
                short_entry_trace_kwargs,
            )

        return fig


EntryTrades.override_field_config_doc(__pdoc__)


# ############# ExitTrades ############# #

exit_trades_field_config = ReadonlyConfig(dict(settings={"id": dict(title="Exit Trade Id")}))
"""_"""

__pdoc__["exit_trades_field_config"] = f"""Field configuration for `ExitTrades`.

```python
{exit_trades_field_config.prettify_doc()}
```
"""

ExitTradesT = tp.TypeVar("ExitTradesT", bound="ExitTrades")


@override_field_config(exit_trades_field_config)
class ExitTrades(Trades):
    """Class representing exit trade records, extending `Trades`.

    Field configuration is overridden using `exit_trades_field_config`.

    Requires `records_arr` to have all fields defined in `vectorbtpro.portfolio.enums.trade_dt`.
    """

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_orders(
        cls: tp.Type[ExitTradesT],
        orders: Orders,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> ExitTradesT:
        """Build an `ExitTrades` instance from `Orders`.

        Args:
            orders (vectorbtpro.portfolio.orders.Orders): Orders instance from which to derive exit trades.
            open (Optional[ArrayLike]): Open prices.

                If None, uses `orders._open`.
            high (Optional[ArrayLike]): High prices.

                If None, uses `orders._high`.
            low (Optional[ArrayLike]): Low prices.

                If None, uses `orders._low`.
            close (Optional[ArrayLike]): Close prices.

                If None, uses `orders._close`.
            init_position (ArrayLike): Initial position.
            init_price (ArrayLike): Initial position price.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `ExitTrades.from_records`.

        Returns:
            ExitTrades: An `ExitTrades` instance generated from the provided orders.

        See:
            `vectorbtpro.portfolio.nb.records.get_exit_trades_nb`
        """
        if open is None:
            open = orders._open
        if high is None:
            high = orders._high
        if low is None:
            low = orders._low
        if close is None:
            close = orders._close
        func = jit_reg.resolve_option(nb.get_exit_trades_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        trade_records_arr = func(
            orders.values,
            to_2d_array(orders.wrapper.wrap(close, group_by=False)),
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
            sim_start=None if sim_start is None else to_1d_array(sim_start),
            sim_end=None if sim_end is None else to_1d_array(sim_end),
        )
        return cls.from_records(
            orders.wrapper,
            trade_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )

    def plot_signals(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_ohlc: bool = True,
        plot_close: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        long_exit_trace_kwargs: tp.KwargsLike = None,
        short_exit_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot exit trade signals.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_ohlc (bool): Whether to plot the OHLC data.
            plot_close (bool): Whether to plot the close price if OHLC data is not plotted.
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            ohlc_trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the close price.
            long_exit_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for long exit markers.
            short_exit_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for short exit markers.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Plotly figure with exit trade signals plotted.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> index = pd.date_range("2020", periods=7)
            >>> price = pd.Series([1, 2, 3, 2, 3, 4, 3], index=index)
            >>> orders = pd.Series([1, 0, -1, 0, -1, 2, -2], index=index)
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.exit_trades.plot_signals().show()
            ```

            ![](/assets/images/api/exit_trades_plot_signals.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/exit_trades_plot_signals.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if long_exit_trace_kwargs is None:
            long_exit_trace_kwargs = {}
        if short_exit_trace_kwargs is None:
            short_exit_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if (
            plot_ohlc
            and self_col._open is not None
            and self_col._high is not None
            and self_col._low is not None
            and self_col._close is not None
        ):
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            exit_idx = self_col.get_map_field_to_index("exit_idx", minus_one_to_zero=True)
            exit_price = self_col.get_field_arr("exit_price")
            direction = self_col.get_field_arr("direction")

            def _plot_exit_markers(mask, name, color, kwargs):
                if np.any(mask):
                    exit_customdata, exit_hovertemplate = self_col.prepare_customdata(
                        incl_fields=[
                            "id",
                            "exit_order_id",
                            "parent_id",
                            "exit_idx",
                            "size",
                            "exit_price",
                            "exit_fees",
                            "pnl",
                            "return",
                            "status",
                        ],
                        mask=mask,
                    )
                    _kwargs = merge_dicts(
                        dict(
                            x=exit_idx[mask],
                            y=exit_price[mask],
                            mode="markers",
                            marker=dict(
                                symbol="circle",
                                color=color,
                                size=8,
                            ),
                            name=name,
                            customdata=exit_customdata,
                            hovertemplate=exit_hovertemplate,
                        ),
                        kwargs,
                    )
                    scatter = go.Scatter(**_kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Long Exit markers
            _plot_exit_markers(
                direction == TradeDirection.Long,
                "Long Exit",
                plotting_cfg["contrast_color_schema"]["green"],
                long_exit_trace_kwargs,
            )

            # Plot Short Exit markers
            _plot_exit_markers(
                direction == TradeDirection.Short,
                "Short Exit",
                plotting_cfg["contrast_color_schema"]["red"],
                short_exit_trace_kwargs,
            )

        return fig


ExitTrades.override_field_config_doc(__pdoc__)


# ############# Positions ############# #

positions_field_config = ReadonlyConfig(
    dict(
        settings={
            "id": dict(title="Position Id"),
            "parent_id": dict(title="Parent Id", ignore=True),
        }
    ),
)
"""_"""

__pdoc__["positions_field_config"] = f"""Field configuration for `Positions`.

```python
{positions_field_config.prettify_doc()}
```
"""

PositionsT = tp.TypeVar("PositionsT", bound="Positions")


@override_field_config(positions_field_config)
class Positions(Trades):
    """Class representing position records, extending `Trades`.

    Field configuration is overridden using `positions_field_config`.

    Requires `records_arr` to have all fields defined in `vectorbtpro.portfolio.enums.trade_dt`.
    """

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_trades(
        cls: tp.Type[PositionsT],
        trades: Trades,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> PositionsT:
        """Construct a `Positions` instance from a given `Trades` instance.

        Converts the provided trades into position records by applying default price arrays
        when necessary and processing the data with optionally JIT-compiled and chunked functions.

        Args:
            trades (Trades): Source trades instance from which positions are derived.
            open (Optional[ArrayLike]): Open prices.

                If None, uses `trades._open`.
            high (Optional[ArrayLike]): High prices.

                If None, uses `trades._high`.
            low (Optional[ArrayLike]): Low prices.

                If None, uses `trades._low`.
            close (Optional[ArrayLike]): Close prices.

                If None, uses `trades._close`.
            init_position (ArrayLike): Initial position.
            init_price (ArrayLike): Initial position price.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Positions.from_records`.

        Returns:
            Positions: New instance of `Positions` created from the source trades.

        See:
            `vectorbtpro.portfolio.nb.records.get_positions_nb`
        """
        if open is None:
            open = trades._open
        if high is None:
            high = trades._high
        if low is None:
            low = trades._low
        if close is None:
            close = trades._close
        func = jit_reg.resolve_option(nb.get_positions_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        position_records_arr = func(trades.values, trades.col_mapper.col_map)
        return cls.from_records(
            trades.wrapper,
            position_records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )


Positions.override_field_config_doc(__pdoc__)
