# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module for named tuples and enumerated types used in portfolio management."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.utils.formatting import prettify_doc

__pdoc__all__ = __all__ = [
    "RejectedOrderError",
    "PriceType",
    "ValPriceType",
    "InitCashMode",
    "CallSeqType",
    "PendingConflictMode",
    "AccumulationMode",
    "ConflictMode",
    "DirectionConflictMode",
    "OppositeEntryMode",
    "DeltaFormat",
    "TimeDeltaFormat",
    "StopLadderMode",
    "StopEntryPrice",
    "StopExitPrice",
    "StopExitType",
    "StopUpdateMode",
    "SizeType",
    "Direction",
    "LeverageMode",
    "PriceAreaVioMode",
    "OrderStatus",
    "OrderStatusInfo",
    "status_info_desc",
    "OrderSide",
    "OrderType",
    "LimitOrderPrice",
    "TradeDirection",
    "TradeStatus",
    "TradesType",
    "OrderPriceStatus",
    "PositionFeature",
    "PriceArea",
    "NoPriceArea",
    "AccountState",
    "ExecState",
    "SimulationOutput",
    "SimulationContext",
    "GroupContext",
    "RowContext",
    "SegmentContext",
    "OrderContext",
    "PostOrderContext",
    "FlexOrderContext",
    "Order",
    "NoOrder",
    "OrderResult",
    "SignalSegmentContext",
    "SignalContext",
    "PostSignalContext",
    "FSInOutputs",
    "FOInOutputs",
    "order_fields",
    "order_dt",
    "fs_order_fields",
    "fs_order_dt",
    "trade_fields",
    "trade_dt",
    "log_fields",
    "log_dt",
    "alloc_range_fields",
    "alloc_range_dt",
    "alloc_point_fields",
    "alloc_point_dt",
    "main_info_fields",
    "main_info_dt",
    "limit_info_fields",
    "limit_info_dt",
    "sl_info_fields",
    "sl_info_dt",
    "tsl_info_fields",
    "tsl_info_dt",
    "tp_info_fields",
    "tp_info_dt",
    "time_info_fields",
    "time_info_dt",
]

__pdoc__ = {}


# ############# Errors ############# #


class RejectedOrderError(Exception):
    """Exception raised when an order is rejected."""

    pass


# ############# Enums ############# #


class PriceTypeT(tp.NamedTuple):
    Open: int = -np.inf
    Close: int = np.inf
    NextOpen: int = -1
    NextClose: int = -2
    NextValidOpen: int = -3
    NextValidClose: int = -4


PriceType = PriceTypeT()
"""_"""

__pdoc__["PriceType"] = f"""Price type enumeration.

```python
{prettify_doc(PriceType)}
```

Fields:
    Open: Open price.

        Replaced by `-np.inf`.
    Close: Close price.

        Replaced by `np.inf`.
    NextOpen: Next open price.

        Replaced by `-np.inf` with `from_ago` set to 1.
    NextClose: Next close price.

        Replaced by `np.inf` with `from_ago` set to 1.
    NextValidOpen: Next valid (non-NA) open price.

        Replaced by `-np.inf` with `from_ago` set to the distance to the previous valid value.
    NextValidClose: Next valid (non-NA) close price.

        Replaced by `np.inf` with `from_ago` set to the distance to the previous valid value.
"""


class ValPriceTypeT(tp.NamedTuple):
    Latest: int = -np.inf
    Price: int = np.inf


ValPriceType = ValPriceTypeT()
"""_"""

__pdoc__["ValPriceType"] = f"""Asset valuation price type enumeration.

```python
{prettify_doc(ValPriceType)}
```

Fields:
    Latest: Latest price.

        Replaced by `-np.inf`.
    Price: Order price.

        Replaced by `np.inf`.
"""


class InitCashModeT(tp.NamedTuple):
    Auto: int = -1
    AutoAlign: int = -2


InitCashMode = InitCashModeT()
"""_"""

__pdoc__["InitCashMode"] = f"""Initial cash mode enumeration.

```python
{prettify_doc(InitCashMode)}
```

Fields:
    Auto: Initial cash is treated as infinite during simulation, then set to the total cash spent.
    AutoAlign: Initial cash is set uniformly to the total cash spent across all columns.
"""


class CallSeqTypeT(tp.NamedTuple):
    Default: int = 0
    Reversed: int = 1
    Random: int = 2
    Auto: int = 3


CallSeqType = CallSeqTypeT()
"""_"""

__pdoc__["CallSeqType"] = f"""Call sequence type enumeration.

```python
{prettify_doc(CallSeqType)}
```

Fields:
    Default: Calls are executed from left to right.
    Reversed: Calls are executed from right to left.
    Random: Calls are executed in a random order.
    Auto: Calls are dynamically ordered based on order value.
"""


class PendingConflictModeT(tp.NamedTuple):
    KeepIgnore: int = 0
    KeepExecute: int = 1
    CancelIgnore: int = 2
    CancelExecute: int = 3


PendingConflictMode = PendingConflictModeT()
"""_"""

__pdoc__["PendingConflictMode"] = f"""Conflict mode enumeration for pending signals.

This setting determines the action when an executable signal occurs during the pending phase.

```python
{prettify_doc(PendingConflictMode)}
```

Fields:
    KeepIgnore: Retain the pending signal while canceling the user-defined signal.
    KeepExecute: Retain the pending signal and execute the user-defined signal.
    CancelIgnore: Cancel the pending signal and ignore the user-defined signal.
    CancelExecute: Cancel the pending signal and execute the user-defined signal.
"""


class AccumulationModeT(tp.NamedTuple):
    Disabled: int = 0
    Both: int = 1
    AddOnly: int = 2
    RemoveOnly: int = 3


AccumulationMode = AccumulationModeT()
"""_"""

__pdoc__["AccumulationMode"] = f"""Accumulation mode enumeration.

```python
{prettify_doc(AccumulationMode)}
```

Enables gradual adjustment of positions.

Fields:
    Disabled: Disables accumulation (also accepts False).
    Both: Allows both increasing and decreasing the position (also accepts True).
    AddOnly: Permits only adding to the position.
    RemoveOnly: Permits only reducing the position.

!!! note
    Accumulation behaves differently for exits and opposite entries: exits reduce the current
    position without initiating a reverse position, whereas opposite entries reduce the current
    position by the same amount and, once fully closed, commence building the opposite position.

    The behavior for opposite entries can be modified using `OppositeEntryMode` and
    for stop orders using `StopExitType`.
"""


class ConflictModeT(tp.NamedTuple):
    Ignore: int = 0
    Entry: int = 1
    Exit: int = 2
    Adjacent: int = 3
    Opposite: int = 4


ConflictMode = ConflictModeT()
"""_"""

__pdoc__["ConflictMode"] = f"""Conflict mode enumeration.

```python
{prettify_doc(ConflictMode)}
```

Determines the action when both entry and exit signals occur simultaneously.

Fields:
    Ignore: Ignore both signals.
    Entry: Execute the entry signal.
    Exit: Execute the exit signal.
    Adjacent: Execute the signal adjacent to the current position
        (applies only when already in a position; otherwise ignored).
    Opposite: Execute the signal opposite to the current position
        (applies only when already in a position; otherwise ignored).
"""


class DirectionConflictModeT(tp.NamedTuple):
    Ignore: int = 0
    Long: int = 1
    Short: int = 2
    Adjacent: int = 3
    Opposite: int = 4


DirectionConflictMode = DirectionConflictModeT()
"""_"""

__pdoc__["DirectionConflictMode"] = f"""Direction conflict mode enumeration.

```python
{prettify_doc(DirectionConflictMode)}
```

Determines the behavior when both long and short entry signals occur simultaneously.

Fields:
    Ignore: Ignore both entry signals.
    Long: Execute the long entry signal.
    Short: Execute the short entry signal.
    Adjacent: Execute the adjacent entry signal.

        Takes effect only when in position; otherwise, it is ignored.
    Opposite: Execute the opposite entry signal.

        Takes effect only when in position; otherwise, it is ignored.
"""


class OppositeEntryModeT(tp.NamedTuple):
    Ignore: int = 0
    Close: int = 1
    CloseReduce: int = 2
    Reverse: int = 3
    ReverseReduce: int = 4


OppositeEntryMode = OppositeEntryModeT()
"""_"""

__pdoc__["OppositeEntryMode"] = f"""Opposite entry mode enumeration.

```python
{prettify_doc(OppositeEntryMode)}
```

Determines the behavior when an entry signal in the opposite direction is received before an exit signal.

Fields:
    Ignore: Ignore the opposite entry signal.
    Close: Close the current position.
    CloseReduce: Close the current position or reduce it if accumulation is enabled.
    Reverse: Reverse the current position.
    ReverseReduce: Reverse the current position or reduce it if accumulation is enabled.
"""


class DeltaFormatT(tp.NamedTuple):
    Absolute: int = 0
    Percent: int = 1
    Percent100: int = 2
    Target: int = 3


DeltaFormat = DeltaFormatT()
"""_"""

__pdoc__["DeltaFormat"] = f"""Delta format enumeration.

```python
{prettify_doc(DeltaFormat)}
```

Specifies the format in which a delta value is provided.

Fields:
    Absolute: Represents the absolute difference between the initial and target values.
    Percent: Represents a percentage difference, where 0.1 indicates a 10% change applied to the initial value.
    Percent100: Represents a percentage difference, where 0.1 indicates a 0.1% change applied to the initial value.
    Target: Represents the target value.
"""


class TimeDeltaFormatT(tp.NamedTuple):
    Rows: int = 0
    Index: int = 1


TimeDeltaFormat = TimeDeltaFormatT()
"""_"""

__pdoc__["TimeDeltaFormat"] = f"""Time delta format enumeration.

```python
{prettify_doc(TimeDeltaFormat)}
```

Specifies the format in which a time delta value is represented.

Fields:
    Rows: Row-based format where a value of 1 indicates that one simulation step (row) has passed.

        Does not require an index.
    Index: Index-based format where a value of 1 indicates that one position in the index has passed.

        If the index is datetime-like, 1 corresponds to one nanosecond; the index must be provided.
"""


class StopLadderModeT(tp.NamedTuple):
    Disabled: int = 0
    Uniform: int = 1
    Weighted: int = 2
    AdaptUniform: int = 3
    AdaptWeighted: int = 4
    Dynamic: int = 5


StopLadderMode = StopLadderModeT()
"""_"""

__pdoc__["StopLadderMode"] = f"""Stop ladder mode enumeration.

```python
{prettify_doc(StopLadderMode)}
```

Specifies the configuration of the stop ladder mechanism.

Fields:
    Disabled: Disables the stop ladder (can also be provided as False).
    Uniform: Enables the stop ladder with a uniform exit size (can also be provided as True).
    Weighted: Enables the stop ladder with a stop-weighted exit size.
    AdaptUniform: Enables the stop ladder with a uniform exit size that adapts to the current position.
    AdaptWeighted: Enables the stop ladder with a stop-weighted exit size that adapts to the current position.
    Dynamic: Enables the stop ladder without using stop arrays.

!!! note
    When disabled, ensure that stop arrays broadcast against the target shape.
    When enabled, ensure that rows in stop arrays represent steps in the ladder.
"""


class StopEntryPriceT(tp.NamedTuple):
    ValPrice: int = -1
    Open: int = -2
    Price: int = -3
    FillPrice: int = -4
    Close: int = -5


StopEntryPrice = StopEntryPriceT()
"""_"""

__pdoc__["StopEntryPrice"] = f"""Stop entry price enumeration.

```python
{prettify_doc(StopEntryPrice)}
```

Specifies the price to use as an initial stop price.

Fields:
    ValPrice: Represents the asset's valuation price.
    Open: Represents the open price.
    Price: Represents the order price.
    FillPrice: Represents the filled order price (with slippage applied).
    Close: Represents the close price.

!!! note
    Each flag is negative; if a positive value is provided, it is used directly as a price.
"""


class StopExitPriceT(tp.NamedTuple):
    Stop: int = -1
    HardStop: int = -2
    Close: int = -3


StopExitPrice = StopExitPriceT()
"""_"""

__pdoc__["StopExitPrice"] = f"""Stop exit price enumeration.

```python
{prettify_doc(StopExitPrice)}
```

Specifies the price to use when exiting a position upon a stop signal.

Fields:
    Stop: Uses the stop price.

        If the target price is first reached by the open price, the open price is used.
        The same applies to the close price if OHLC is not available.

        This option is more conservative.
    HardStop: Uses a hard stop price, applying the stop value regardless of whether the target price
        is first hit by the open or close price.

        This option is more optimistic.
    Close: Uses the close price.

!!! note
    Each flag is negative; if a positive value is provided, it is used directly as a price.
"""


class StopExitTypeT(tp.NamedTuple):
    Close: int = 0
    CloseReduce: int = 1
    Reverse: int = 2
    ReverseReduce: int = 3


StopExitType = StopExitTypeT()
"""_"""

__pdoc__["StopExitType"] = f"""Stop exit type enumeration.

```python
{prettify_doc(StopExitType)}
```

Specifies the method for exiting the current position when a stop signal is triggered.

Fields:
    Close: Closes the current position.
    CloseReduce: Closes or reduces the current position if accumulation is enabled.
    Reverse: Reverses the current position.
    ReverseReduce: Reverses or reduces the current position if accumulation is enabled.
"""


class StopUpdateModeT(tp.NamedTuple):
    Keep: int = 0
    Override: int = 1
    OverrideNaN: int = 2


StopUpdateMode = StopUpdateModeT()
"""_"""

__pdoc__["StopUpdateMode"] = f"""Stop update mode enumeration.

```python
{prettify_doc(StopUpdateMode)}
```

Specifies how to handle an existing stop when a new entry or accumulation occurs.

Fields:
    Keep: Retain the old stop.
    Override: Replace the old stop if the new stop is not NaN.
    OverrideNaN: Replace the old stop even if the new stop is NaN.
"""


class SizeTypeT(tp.NamedTuple):
    Amount: int = 0
    Value: int = 1
    Percent: int = 2
    Percent100: int = 3
    ValuePercent: int = 4
    ValuePercent100: int = 5
    TargetAmount: int = 6
    TargetValue: int = 7
    TargetPercent: int = 8
    TargetPercent100: int = 9


SizeType = SizeTypeT()
"""_"""

__pdoc__["SizeType"] = f"""Size type for trading operations enumeration.

```python
{prettify_doc(SizeType)}
```

Fields:
    Amount: Specifies the number of assets to trade.
    Value: Specifies the monetary value of assets to trade.

        It is converted to `SizeType.Amount` using `ExecState.val_price`.
    Percent: Specifies the percentage of available resources to use for trading, where 0.01 represents 1%.

          * When long buying, applies to `ExecState.free_cash`
          * When long selling, applies to `ExecState.position`
          * When short selling, applies to `ExecState.free_cash`
          * When short buying, applies to `ExecState.free_cash`, `ExecState.debt`, and `ExecState.locked_cash`
          * When reversing, applies to the final position
    Percent100: Equivalent to `SizeType.Percent` with a scale where 1.0 represents 1%.
    ValuePercent: Represents a percentage of the total value using `ExecState.value`.

        Converted to `SizeType.Value`.
    ValuePercent100: Equivalent to `SizeType.ValuePercent` with 1.0 representing 1%.
    TargetAmount: Specifies the target number of assets (target position) using `ExecState.position`.

        Converted to `SizeType.Amount`.
    TargetValue: Specifies the target asset value using `ExecState.val_price`.

        Converted to `SizeType.TargetAmount`.
    TargetPercent: Specifies the target percentage of total value using `ExecState.value`.

        Converted to `SizeType.TargetValue`.
    TargetPercent100: Equivalent to `SizeType.TargetPercent` where 1.0 represents 1%.
"""


class DirectionT(tp.NamedTuple):
    LongOnly: int = 0
    ShortOnly: int = 1
    Both: int = 2


Direction = DirectionT()
"""_"""

__pdoc__["Direction"] = f"""Position direction enumeration.

```python
{prettify_doc(Direction)}
```

Fields:
    LongOnly: Indicates long-only positions.
    ShortOnly: Indicates short-only positions.
    Both: Indicates both long and short positions.
"""


class LeverageModeT(tp.NamedTuple):
    Lazy: int = 0
    Eager: int = 1


LeverageMode = LeverageModeT()
"""_"""

__pdoc__["LeverageMode"] = f"""Leverage mode enumeration.

```python
{prettify_doc(LeverageMode)}
```

Fields:
    Lazy: Leverage is applied only when free cash is exhausted.
    Eager: Leverage is applied to each order.
"""


class PriceAreaVioModeT(tp.NamedTuple):
    Ignore: int = 0
    Cap: int = 1
    Error: int = 2


PriceAreaVioMode = PriceAreaVioModeT()
"""_"""

__pdoc__["PriceAreaVioMode"] = f"""Price area violation mode enumeration.

```python
{prettify_doc(PriceAreaVioMode)}
```

Fields:
    Ignore: Ignores any violation.
    Cap: Caps the price to prevent a violation.
    Error: Raises an error upon violation.
"""


class OrderStatusT(tp.NamedTuple):
    Filled: int = 0
    Ignored: int = 1
    Rejected: int = 2


OrderStatus = OrderStatusT()
"""_"""

__pdoc__["OrderStatus"] = f"""Order status enumeration.

```python
{prettify_doc(OrderStatus)}
```

Fields:
    Filled: Indicates the order has been filled.
    Ignored: Indicates the order has been ignored.
    Rejected: Indicates the order has been rejected.
"""


class OrderStatusInfoT(tp.NamedTuple):
    SizeNaN: int = 0
    PriceNaN: int = 1
    ValPriceNaN: int = 2
    ValueNaN: int = 3
    ValueZeroNeg: int = 4
    SizeZero: int = 5
    NoCash: int = 6
    NoOpenPosition: int = 7
    MaxSizeExceeded: int = 8
    RandomEvent: int = 9
    CantCoverFees: int = 10
    MinSizeNotReached: int = 11
    PartialFill: int = 12


OrderStatusInfo = OrderStatusInfoT()
"""_"""

__pdoc__["OrderStatusInfo"] = f"""Order status information enumeration.

```python
{prettify_doc(OrderStatusInfo)}
```

Fields:
    SizeNaN: Size is NaN.
    PriceNaN: Price is NaN.
    ValPriceNaN: Asset valuation price is NaN.
    ValueNaN: Asset/group value is NaN.
    ValueZeroNeg: Asset/group value is zero or negative.
    SizeZero: Size is zero.
    NoCash: Not enough cash.
    NoOpenPosition: No open position to reduce/close.
    MaxSizeExceeded: Size is greater than maximum allowed.
    RandomEvent: Random event happened.
    CantCoverFees: Not enough cash to cover fees.
    MinSizeNotReached: Final size is less than minimum allowed.
    PartialFill: Final size is less than requested.
"""

status_info_desc = [
    "Size is NaN",
    "Price is NaN",
    "Asset valuation price is NaN",
    "Asset/group value is NaN",
    "Asset/group value is zero or negative",
    "Size is zero",
    "Not enough cash",
    "No open position to reduce/close",
    "Size is greater than maximum allowed",
    "Random event happened",
    "Not enough cash to cover fees",
    "Final size is less than minimum allowed",
    "Final size is less than requested",
]
"""_"""

__pdoc__["status_info_desc"] = f"""Order status descriptions.

```python
{prettify_doc(status_info_desc)}
```
"""


class OrderSideT(tp.NamedTuple):
    Buy: int = 0
    Sell: int = 1


OrderSide = OrderSideT()
"""_"""

__pdoc__["OrderSide"] = f"""Order side enumeration.

```python
{prettify_doc(OrderSide)}
```

Fields:
    Buy: Buy order.
    Sell: Sell order.
"""


class OrderTypeT(tp.NamedTuple):
    Market: int = 0
    Limit: int = 1


OrderType = OrderTypeT()
"""_"""

__pdoc__["OrderType"] = f"""Order type enumeration.

```python
{prettify_doc(OrderType)}
```

Fields:
    Market: Market order.
    Limit: Limit order.
"""


class LimitOrderPriceT(tp.NamedTuple):
    Limit: int = -1
    HardLimit: int = -2
    AutoLimit: int = -3
    Close: int = -4


LimitOrderPrice = LimitOrderPriceT()
"""_"""

__pdoc__["LimitOrderPrice"] = f"""Limit order price enumeration.

```python
{prettify_doc(LimitOrderPrice)}
```

Determines which price to use when executing a limit order.

Fields:
    Limit: Limit price.

        If the target price is first reached at the opening, the open price is used.
        The same applies to the close price if OHLC is not available.

        This option is more optimistic.
    HardLimit: Hard limit price, where the stop price is applied regardless of the open or close price trigger.

        This option is more conservative.
    AutoLimit: Behaves like `Limit` if reversal is enabled, otherwise behaves like `HardLimit`.
    Close: Close price.

!!! note
    Each flag is negative; if a positive value is provided, it is used directly as the price.
"""


class TradeDirectionT(tp.NamedTuple):
    Long: int = 0
    Short: int = 1


TradeDirection = TradeDirectionT()
"""_"""

__pdoc__["TradeDirection"] = f"""Trade direction enumeration.

```python
{prettify_doc(TradeDirection)}
```

Fields:
    Long: Long trade.
    Short: Short trade.
"""


class TradeStatusT(tp.NamedTuple):
    Open: int = 0
    Closed: int = 1


TradeStatus = TradeStatusT()
"""_"""

__pdoc__["TradeStatus"] = f"""Trade status enumeration.

```python
{prettify_doc(TradeStatus)}
```

Fields:
    Open: Open trade.
    Closed: Closed trade.
"""


class TradesTypeT(tp.NamedTuple):
    Trades: int = 0
    EntryTrades: int = 1
    ExitTrades: int = 2
    Positions: int = 3


TradesType = TradesTypeT()
"""_"""

__pdoc__["TradesType"] = f"""Trades type enumeration.

```python
{prettify_doc(TradesType)}
```

Fields:
    Trades: Trades of type `vectorbtpro.portfolio.trades.Trades`.
    EntryTrades: Trades of type `vectorbtpro.portfolio.trades.EntryTrades`.
    ExitTrades: Trades of type `vectorbtpro.portfolio.trades.ExitTrades`.
    Positions: Trades of type `vectorbtpro.portfolio.trades.Positions`.
"""


class OrderPriceStatusT(tp.NamedTuple):
    OK: int = 0
    AboveHigh: int = 1
    BelowLow: int = 2
    Unknown: int = 3


OrderPriceStatus = OrderPriceStatusT()
"""_"""

__pdoc__["OrderPriceStatus"] = f"""Order price status enumeration.

```python
{prettify_doc(OrderPriceStatus)}
```

Fields:
    OK: Order price is within the OHLC range.
    AboveHigh: Order price is above the high.
    BelowLow: Order price is below the low.
    Unknown: High and/or low are unknown.
"""


class PositionFeatureT(tp.NamedTuple):
    EntryPrice: int = 0
    ExitPrice: int = 1


PositionFeature = PositionFeatureT()
"""_"""

__pdoc__["PositionFeature"] = f"""Position feature enumeration.

```python
{prettify_doc(PositionFeature)}
```

Fields:
    EntryPrice: Entry price.
    ExitPrice: Exit price.
"""


# ############# Named tuples ############# #


class PriceArea(tp.NamedTuple):
    open: float
    high: float
    low: float
    close: float


__pdoc__["PriceArea"] = """Price area defined by four boundaries.

Used together with `PriceAreaVioMode`.
"""
__pdoc__["PriceArea.open"] = "Open price of the bar."
__pdoc__["PriceArea.high"] = """High price of the bar.

A violation occurs when the adjusted price exceeds this value.
"""
__pdoc__["PriceArea.low"] = """Low price of the bar.

A violation is triggered when the adjusted price falls below this value.
"""
__pdoc__["PriceArea.close"] = """Close price of the bar.

A violation is triggered when the adjusted price goes beyond this value.
"""

NoPriceArea = PriceArea(open=np.nan, high=np.nan, low=np.nan, close=np.nan)
"""_"""

__pdoc__["NoPriceArea"] = (
    "`PriceArea` instance with NaN values, representing the absence of a valid price area."
)


class AccountState(tp.NamedTuple):
    cash: float
    position: float
    debt: float
    locked_cash: float
    free_cash: float


__pdoc__["AccountState"] = "Named tuple representing the state of the account."
__pdoc__["AccountState.cash"] = """Cash.

Value is per group if cash sharing is enabled, or per column otherwise.
"""
__pdoc__["AccountState.position"] = """Position.

Value is reported per column.
"""
__pdoc__["AccountState.debt"] = """Debt.

Value is reported per column.
"""
__pdoc__["AccountState.locked_cash"] = """Locked cash.

Value is reported per column.
"""
__pdoc__["AccountState.free_cash"] = """Free cash.

Value is per group if cash sharing is enabled, or per column otherwise.
"""


class ExecState(tp.NamedTuple):
    cash: float
    position: float
    debt: float
    locked_cash: float
    free_cash: float
    val_price: float
    value: float


__pdoc__["ExecState"] = "Named tuple representing the state before or after order execution."
__pdoc__["ExecState.cash"] = "See `AccountState.cash`."
__pdoc__["ExecState.position"] = "See `AccountState.position`."
__pdoc__["ExecState.debt"] = "See `AccountState.debt`."
__pdoc__["ExecState.locked_cash"] = "See `AccountState.locked_cash`."
__pdoc__["ExecState.free_cash"] = "See `AccountState.free_cash`."
__pdoc__["ExecState.val_price"] = "Valuation price for the current column."
__pdoc__["ExecState.value"] = (
    "Value for the current column, or per group when cash sharing is enabled."
)


class SimulationOutput(tp.NamedTuple):
    order_records: tp.RecordArray2d
    log_records: tp.RecordArray2d
    cash_deposits: tp.Array2d
    cash_earnings: tp.Array2d
    call_seq: tp.Optional[tp.Array2d]
    in_outputs: tp.Optional[tp.NamedTuple]
    sim_start: tp.Optional[tp.Array1d]
    sim_end: tp.Optional[tp.Array1d]


__pdoc__["SimulationOutput"] = "Named tuple representing the output of a simulation."
__pdoc__["SimulationOutput.order_records"] = """Flattened order records.

Must adhere to the `order_dt` dtype.
"""
__pdoc__["SimulationOutput.log_records"] = """Flattened log records.

Must adhere to the `log_dt` dtype.
"""
__pdoc__["SimulationOutput.cash_deposits"] = """Cash deposits and withdrawals for each timestamp.

If not tracked, a zero array with shape `(1, 1)` is returned.
"""
__pdoc__["SimulationOutput.cash_earnings"] = """Cash earnings added or removed at each timestamp.

If not tracked, a zero array with shape `(1, 1)` is returned.
"""
__pdoc__["SimulationOutput.call_seq"] = """Call sequence.

If not tracked, it is set to None.
"""
__pdoc__["SimulationOutput.in_outputs"] = """Named tuple containing in-place output objects.

If not tracked, this value is None.
"""
__pdoc__["SimulationOutput.sim_start"] = """Simulation start for each column.

Use `vectorbtpro.generic.nb.sim_range.prepare_ungrouped_sim_range_nb` to ungroup the array.

If not tracked, this value is None.
"""
__pdoc__["SimulationOutput.sim_end"] = """Simulation end for each column.

Use `vectorbtpro.generic.nb.sim_range.prepare_ungrouped_sim_range_nb` to ungroup the array.

If not tracked, this value is None.
"""


class SimulationContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    last_pos_info: tp.RecordArray

    sim_start: tp.Array1d
    sim_end: tp.Array1d


__pdoc__["SimulationContext"] = """Named tuple representing the simulation context.

Contains general simulation information for use in other contexts, and
is passed to `pre_sim_func_nb` and `post_sim_func_nb`.
"""
__pdoc__["SimulationContext.target_shape"] = """Target shape of the simulation data.

A tuple with exactly two elements: the number of rows and columns.

Examples:
    One day of minute data for three assets yields a `target_shape` of `(1440, 3)`,
    where the first axis corresponds to rows (minutes) and the second axis corresponds to columns (assets).
"""
__pdoc__["SimulationContext.group_lens"] = """Number of columns in each group.

If columns are not grouped, `group_lens` consists of ones (one column per group).

!!! note
    Modifying this array may lead to results that are inconsistent with
    those produced by `vectorbtpro.portfolio.base.Portfolio`.

Examples:
    In pairs trading, `group_lens` would be `np.array([2])`, whereas three independent columns
    would be represented by `np.array([1, 1, 1])`.
"""
__pdoc__["SimulationContext.cash_sharing"] = "Indicates whether cash sharing is enabled."
__pdoc__["SimulationContext.call_seq"] = """Default sequence of calls per segment.

Controls the order in which `order_func_nb` is executed within each segment.

It has the shape of `SimulationContext.target_shape`, and each value must be in the range `[0, group_len)`.

If not provided, this field is set to None.

!!! note
    To use `sort_call_seq_1d_nb`, the sequence must be generated using `CallSeqType.Default`.

    To modify the call sequence dynamically, change `SegmentContext.call_seq_now` in place.

Examples:
    Default call sequence for three data points and two groups with three columns each:

    ```python
    np.array([
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2]
    ])
    ```
"""
__pdoc__[
    "SimulationContext.init_cash"
] = """Initial capital per column, or per group with cash sharing.

Flexible indexing is applied using `vectorbtpro.base.flex_indexing.flex_select_1d_pc_nb` and
the array must broadcast to shape `(group_lens.shape[0],)` with cash sharing,
or `(target_shape[1],)` otherwise.

!!! note
    Modifying this array may yield results inconsistent with those produced
    by `vectorbtpro.portfolio.base.Portfolio`.

Examples:
    For three columns each with $100 of starting capital, grouping them as one group of two columns and
    one group of one column yields `init_cash` of `np.array([200, 100])` with cash sharing, and
    `np.array([100, 100, 100])` without cash sharing.
"""
__pdoc__["SimulationContext.init_position"] = """Initial position per column.

Flexible indexing is applied using `vectorbtpro.base.flex_indexing.flex_select_1d_pc_nb` and
the array must broadcast to shape `(target_shape[1],)`.

!!! note
    Modifying this array may yield results inconsistent with those produced
    by `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__["SimulationContext.init_price"] = """Initial position price per column.

Uses flexible indexing via `vectorbtpro.base.flex_indexing.flex_select_1d_pc_nb`.

The array must broadcast to shape `(target_shape[1],)`.

!!! note
    Changing this array may produce results inconsistent with those
    of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.cash_deposits"
] = """Cash to be deposited or withdrawn per column (or per group with cash sharing).

Uses flexible indexing via `vectorbtpro.base.flex_indexing.flex_select_nb`.

The array must broadcast to shape `(target_shape[0], group_lens.shape[0])`.

Cash is deposited or withdrawn immediately after `pre_segment_func_nb`.
It can be modified within `pre_segment_func_nb`.

!!! note
    To modify the array in place, ensure that the full shape of the array is constructed.
"""
__pdoc__["SimulationContext.cash_earnings"] = """Earnings to be added per column.

Uses flexible indexing via `vectorbtpro.base.flex_indexing.flex_select_nb`.

The array must broadcast to shape `SimulationContext.target_shape`.

Earnings are added immediately before `post_segment_func_nb` and are included in the group value.
They can be modified in `pre_segment_func_nb` or `post_order_func_nb`.

!!! note
    To modify the array in place, ensure that the full array shape is constructed.
"""
__pdoc__[
    "SimulationContext.segment_mask"
] = """Boolean mask indicating whether a segment should be executed.

A segment is defined as a sequence of `order_func_nb` calls within the same group and row.
If a segment is inactive, callbacks within it will not be executed.
You can override this by enabling `SimulationContext.call_pre_segment` and
`SimulationContext.call_post_segment`.

Uses flexible indexing via `vectorbtpro.base.flex_indexing.flex_select_nb`.

The mask must broadcast to shape `(target_shape[0], group_lens.shape[0])`.

!!! note
    To modify the array in place, ensure that an array of the full shape is constructed.

Examples:
    Consider two groups with two columns each and the following activity mask:

    ```python
    np.array([[ True, False],
              [False,  True]])
    ```

    The first group is only executed in the first row and the second group is only executed in the second row.
"""
__pdoc__["SimulationContext.call_pre_segment"] = (
    """Indicates whether to call `pre_segment_func_nb` regardless of the `SimulationContext.segment_mask`."""
)
__pdoc__[
    "SimulationContext.call_post_segment"
] = """Indicates whether to call `post_segment_func_nb` regardless of `SimulationContext.segment_mask`.

This flag enables additional operations at the end of each segment, such as
writing user-defined arrays (e.g., returns).
"""
__pdoc__["SimulationContext.index"] = """Time index in integer (nanosecond) format.

If the index is datetime-like, it is assumed to be in UTC.
Preset simulation methods automatically format any datetime index as UTC without converting the actual time
(e.g., `12:00 +02:00` becomes `12:00 +00:00`) to avoid timezone conversion issues.
"""
__pdoc__["SimulationContext.freq"] = (
    """Frequency of the time index in integer (nanosecond) format."""
)
__pdoc__["SimulationContext.open"] = """Open price.

Replaces `Order.price` when it is `-np.inf`.

Exhibits behavior similar to `SimulationContext.close`.
"""
__pdoc__["SimulationContext.high"] = """High price.

Exhibits behavior similar to `SimulationContext.close`.
"""
__pdoc__["SimulationContext.low"] = """Low price.

Exhibits behavior similar to `SimulationContext.close`.
"""
__pdoc__["SimulationContext.close"] = """Close price at each bar.

Replaces `Order.price` when it is `np.inf`.

Serves as a boundary; see `PriceArea.close`.

Uses flexible indexing with `vectorbtpro.base.flex_indexing.flex_select_nb`.

The array must broadcast to shape `SimulationContext.target_shape`.

!!! note
    To modify the array in place, ensure that an array of the full shape is provided.
"""
__pdoc__["SimulationContext.bm_close"] = """Benchmark close price at each bar.

The array must broadcast to shape `SimulationContext.target_shape`.
"""
__pdoc__["SimulationContext.ffill_val_price"] = """Tracks valuation price only if it is known.

If not, an unknown `SimulationContext.close` will result in NaN for the valuation price
at the subsequent timestamp.
"""
__pdoc__["SimulationContext.update_value"] = """Update group value after each filled order.

If False, the value remains constant for all columns in the group (calculated only once before
executing any order). The change is marginal and primarily influenced by transaction costs and slippage.
"""
__pdoc__["SimulationContext.fill_pos_info"] = """Determines whether to fill position information.

Disable this feature to improve simulation speed for simple use cases.
"""
__pdoc__[
    "SimulationContext.track_value"
] = """Tracks value metrics, such as the current valuation price, value, and return.

If False, `SimulationContext.last_val_price`, `SimulationContext.last_value`, and
`SimulationContext.last_return` will remain NaN, and open position statistics will not be updated.
Consequently, `SizeType.Value`, `SizeType.TargetValue`, and `SizeType.TargetPercent` cannot be used.

Disable to improve simulation speed in simple cases.
"""
__pdoc__["SimulationContext.order_records"] = """Order records per column.

This 2-dimensional array contains records of type `order_dt`.

It is initially populated with empty records (which contain placeholder random data), and is then
gradually updated with order data. The number of empty records is determined by `max_order_records` —
typically matching the number of rows, which implies a maximum of one order record per element.
Adjust `max_order_records` lower if not every `order_func_nb` call results in a filled order to
conserve memory, or higher if multiple orders per element are expected.

Use `SimulationContext.order_counts` to obtain the count of filled orders per column.
To retrieve all order records filled thus far in a column, use `order_records[:order_counts[col], col]`.

Examples:
    Before filling, an order record appears as:

    ```python
    np.array([(-8070450532247928832, -8070450532247928832, 4, 0., 0., 0., 5764616306889786413)]
    ```

    After filling, it becomes:

    ```python
    np.array([(0, 0, 1, 50., 1., 0., 1)]
    ```
"""
__pdoc__["SimulationContext.order_counts"] = """Number of filled order records in each column.

Corresponds to `SimulationContext.order_records` and has shape `(target_shape[1],)`.

Examples:
    `order_counts` of `np.array([2, 100, 0])` indicates that the latest filled order is
    `order_records[1, 0]` in the first column, `order_records[99, 1]` in the second column,
    and that no orders have been filled in the third column (`order_records[0, 2]` remains empty).

    !!! note
        Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__["SimulationContext.log_records"] = """Log records per column.

Similar to `SimulationContext.order_records` but containing records of type `log_dt` and
associated with `SimulationContext.log_counts`.
"""
__pdoc__["SimulationContext.log_counts"] = """Number of filled log records in each column.

Analogous to `SimulationContext.order_counts`, this array reflects the count of log records of type `log_dt`.

!!! note
    Changing this array may produce results inconsistent with those of `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__["SimulationContext.in_outputs"] = """Named tuple with in-place output objects.

May contain objects of arbitrary shape and type, and is returned as part of `SimulationOutput`.
"""
__pdoc__[
    "SimulationContext.last_cash"
] = """Latest cash for each column (or per group when cash sharing is enabled).

At the initial timestamp, it holds the initial capital.
It is updated immediately after `order_func_nb`.

!!! note
    Modifying this array may lead to inconsistencies with `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__["SimulationContext.last_position"] = """Latest position for each column.

At the initial timestamp, it holds the initial position.
The array has shape `(target_shape[1],)` and is updated immediately after `order_func_nb`.

!!! note
    Modifying this array may result in inconsistencies with `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__["SimulationContext.last_debt"] = """Latest debt per column from leverage or shorting.

Shape:
    `(target_shape[1],)`

Updated immediately after `order_func_nb`.

!!! note
    Modifying this array may cause inconsistencies with `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_locked_cash"
] = """Latest locked cash per column from leverage or shorting.

Shape:
    `(target_shape[1],)`

Updated immediately after `order_func_nb`.

!!! note
    Modifying this array may cause inconsistencies with `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_free_cash"
] = """Latest free cash per column (or per group with cash sharing).

Free cash never exceeds the initial level since every operation incurs a cost.

Shape:
    `(target_shape[1],)`

Updated immediately after `order_func_nb`.

!!! note
    Modifying this array may cause inconsistencies with `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__["SimulationContext.last_val_price"] = """Latest valuation price per column.

Shape:
    `(target_shape[1],)`

Used for `SizeType.Value`, `SizeType.TargetValue`, and `SizeType.TargetPercent`.

This value is multiplied by the current position to compute the column's total value
(see `SimulationContext.last_value`).

Update sequence:

* Updated before `pre_segment_func_nb` using `SimulationContext.open`.
* Updated immediately after `pre_segment_func_nb`, allowing in-place overrides.
* If `SimulationContext.update_value` is True, updated after `order_func_nb` using the filled order price.
* Updated before `post_segment_func_nb` using `SimulationContext.close`.

If `SimulationContext.ffill_val_price` is enabled, the price is updated only when not NaN.
For example, a close value of `[1, 2, np.nan, np.nan, 5]` results in `[1, 2, 2, 2, 5]`.

!!! note
    Use only finite values; `-np.inf` and `np.inf` are not allowed.

Examples:
    Consider 10 units in column 1 and 20 units in column 2. The current open price of them is
    $40 and $50 respectively, which is also the default valuation price in the current row,
    available as `last_val_price` in `pre_segment_func_nb`. If both columns are in the same group
    with cash sharing, the group is valued at $1400 before any `order_func_nb` is called, and can
    be later accessed via `OrderContext.value_now`.
"""
__pdoc__[
    "SimulationContext.last_value"
] = """Latest value per column (or per group with cash sharing).

Computed as (valuation price * current position) + cash.

For groups with cash sharing, column values are summed to determine the group's total value.

Update sequence:

* Updated before `pre_segment_func_nb`.
* Updated immediately after `pre_segment_func_nb`.
* If `SimulationContext.update_value` is True, updated after `order_func_nb` using the filled order price.
* Updated before `post_segment_func_nb`.

!!! note
    Modifying this array may cause inconsistencies with `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__[
    "SimulationContext.last_return"
] = """Latest return per column (or per group with cash sharing).

Shape:
    Same as `SimulationContext.last_value`.

Calculated by comparing the current `last_value` with its previous row value.

Updated each time `last_value` is refreshed.

!!! note
    Modifying this array may cause inconsistencies with `vectorbtpro.portfolio.base.Portfolio`.
"""
__pdoc__["SimulationContext.last_pos_info"] = """Latest position record for each column.

A 1-dimensional array of records of type `trade_dt` with shape `(target_shape[1],)`.

If `SimulationContext.init_position` is nonzero, the corresponding column's position record
is pre-populated with `entry_price` set to `SimulationContext.init_price` and `entry_idx` set to -1.

Notes:

* `entry_price` and `exit_price` represent the average entry and exit prices, respectively.
* Average exit price excludes open statistics (unlike `vectorbtpro.portfolio.trades.Positions`).
* Fields `pnl` and `return` are computed as if the position were closed and are recalculated at:
    * Before and after `pre_segment_func_nb` using `last_val_price`.
    * After `order_func_nb`.
    * Before `post_segment_func_nb`.

!!! note
    In open positions, `exit_price` retains the average reduction price and
    does not reflect the latest valuation.
"""
__pdoc__[
    "SimulationContext.sim_start"
] = """Start of the simulation per column or group (without cash sharing).

In-place modifications will not affect the current simulation.
"""
__pdoc__[
    "SimulationContext.sim_start"
] = """End of the simulation per column or group (without cash sharing).

In-place modifications will affect the current simulation if the value is lower than the initial one.
"""


class GroupContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    last_pos_info: tp.RecordArray

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int


__pdoc__["GroupContext"] = """Named tuple representing a group context.

A group is a set of adjacent columns that are related (e.g., by shared capital).
In each row, columns in the same group form a single segment.

This tuple includes all fields from `SimulationContext` along with additional fields
describing the current group.

Used in `pre_group_func_nb` and `post_group_func_nb`.

Examples:
    Consider a configuration with a group of three columns, one of two columns, and one standalone column:

    | group | group_len | from_col | to_col |
    | ----- | --------- | -------- | ------ |
    | 0     | 3         | 0        | 3      |
    | 1     | 2         | 3        | 5      |
    | 2     | 1         | 5        | 6      |
"""
for field in GroupContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["GroupContext." + field] = f"See `SimulationContext.{field}`."
__pdoc__["GroupContext.group"] = """Index of the current group.

Range:
    `[0, group_lens.shape[0])`
"""
__pdoc__["GroupContext.group_len"] = """Number of columns in the current group.

Scalar value equivalent to `group_lens[group]`.
"""
__pdoc__["GroupContext.from_col"] = """Index of the first column in the current group.

Range:
    `[0, target_shape[1])`
"""
__pdoc__["GroupContext.to_col"] = """Index of the last column in the current group plus one.

Range: `[1, target_shape[1] + 1)`

When columns are not grouped, equals `from_col + 1`.

!!! warning
    In the last group, `to_col` points to a column that does not exist.
"""


class RowContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    last_pos_info: tp.RecordArray

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    i: int


__pdoc__["RowContext"] = """Named tuple representing the context of a row.

A row corresponds to a bar during which segments are executed.

Includes all fields from `SimulationContext` plus additional fields specific to the current row.

Passed to `pre_row_func_nb` and `post_row_func_nb`.
"""
for field in RowContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["RowContext." + field] = f"See `SimulationContext.{field}`."
__pdoc__["RowContext.i"] = """Index of the current row.

Range: `[0, target_shape[0])`
"""


class SegmentContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    last_pos_info: tp.RecordArray

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int

    call_seq_now: tp.Optional[tp.Array1d]


__pdoc__["SegmentContext"] = """Named tuple representing the context of a segment.

A segment is an intersection of groups and rows that defines the processing order of elements.

Includes all fields from `SimulationContext`, `GroupContext`, and `RowContext` plus additional
fields for the current segment.

Passed to `pre_segment_func_nb` and `post_segment_func_nb`.
"""
for field in SegmentContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["SegmentContext." + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__["SegmentContext." + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__["SegmentContext." + field] = f"See `RowContext.{field}`."
__pdoc__["SegmentContext.call_seq_now"] = """Sequence of call indices within the current segment.

Shape:
    `(group_len,)`

Each value indicates the position of the column in the group to call next.
Processing always proceeds from left to right.

You can override `call_seq_now` using `pre_segment_func_nb`.

Examples:
    `[2, 0, 1]` calls column 2 first, then column 0, and finally column 1.
"""


class OrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    last_pos_info: tp.RecordArray

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int

    call_seq_now: tp.Optional[tp.Array1d]

    col: int
    call_idx: int
    cash_now: float
    position_now: float
    debt_now: float
    locked_cash_now: float
    free_cash_now: float
    val_price_now: float
    value_now: float
    return_now: float
    pos_info_now: tp.Record


__pdoc__["OrderContext"] = """Named tuple representing the context of an order.

Includes all fields from `SegmentContext` along with additional fields that describe the current order state.

Passed to `order_func_nb`.
"""
for field in OrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["OrderContext." + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__["OrderContext." + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__["OrderContext." + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__["OrderContext." + field] = f"See `SegmentContext.{field}`."
__pdoc__["OrderContext.col"] = """Current column index.

Valid indices are in the range `[0, target_shape[1])` and always lie within `[from_col, to_col)`.
"""
__pdoc__["OrderContext.call_idx"] = """Current call index from `SegmentContext.call_seq_now`.

Valid values are in the range `[0, group_len)`.
"""
__pdoc__["OrderContext.cash_now"] = "`SimulationContext.last_cash` for the current column or group."
__pdoc__["OrderContext.position_now"] = "`SimulationContext.last_position` for the current column."
__pdoc__["OrderContext.debt_now"] = "`SimulationContext.last_debt` for the current column."
__pdoc__["OrderContext.locked_cash_now"] = (
    "`SimulationContext.last_locked_cash` for the current column."
)
__pdoc__["OrderContext.free_cash_now"] = (
    "`SimulationContext.last_free_cash` for the current column or group."
)
__pdoc__["OrderContext.val_price_now"] = (
    "`SimulationContext.last_val_price` for the current column."
)
__pdoc__["OrderContext.value_now"] = (
    "`SimulationContext.last_value` for the current column or group."
)
__pdoc__["OrderContext.return_now"] = (
    "`SimulationContext.last_return` for the current column or group."
)
__pdoc__["OrderContext.pos_info_now"] = "`SimulationContext.last_pos_info` for the current column."


class PostOrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    last_pos_info: tp.RecordArray

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int

    call_seq_now: tp.Optional[tp.Array1d]

    col: int
    call_idx: int
    cash_before: float
    position_before: float
    debt_before: float
    locked_cash_before: float
    free_cash_before: float
    val_price_before: float
    value_before: float
    order_result: "OrderResult"
    cash_now: float
    position_now: float
    debt_now: float
    locked_cash_now: float
    free_cash_now: float
    val_price_now: float
    value_now: float
    return_now: float
    pos_info_now: tp.Record


__pdoc__["PostOrderContext"] = """Named tuple representing the post-order context.

Contains all fields from `OrderContext` along with additional fields representing the order result and
the state prior to execution.

Passed to `post_order_func_nb`.
"""
for field in PostOrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `SegmentContext.{field}`."
    elif field in OrderContext._fields:
        __pdoc__["PostOrderContext." + field] = f"See `OrderContext.{field}`."
__pdoc__["PostOrderContext.cash_before"] = "`OrderContext.cash_now` value before order execution."
__pdoc__["PostOrderContext.position_before"] = (
    "`OrderContext.position_now` value before order execution."
)
__pdoc__["PostOrderContext.debt_before"] = "`OrderContext.debt_now` value before order execution."
__pdoc__["PostOrderContext.locked_cash_before"] = (
    "`OrderContext.locked_cash_now` value before order execution."
)
__pdoc__["PostOrderContext.free_cash_before"] = (
    "`OrderContext.free_cash_now` value before order execution."
)
__pdoc__["PostOrderContext.val_price_before"] = (
    "`OrderContext.val_price_now` value before order execution."
)
__pdoc__["PostOrderContext.value_before"] = "`OrderContext.value_now` value before order execution."
__pdoc__["PostOrderContext.order_result"] = """Order result of type `OrderResult`.

Indicates whether the order was filled, ignored, or rejected.
"""
__pdoc__["PostOrderContext.cash_now"] = "`OrderContext.cash_now` value after order execution."
__pdoc__["PostOrderContext.position_now"] = (
    "`OrderContext.position_now` value after order execution."
)
__pdoc__["PostOrderContext.debt_now"] = "`OrderContext.debt_now` value after order execution."
__pdoc__["PostOrderContext.locked_cash_now"] = (
    "`OrderContext.locked_cash_now` value after order execution."
)
__pdoc__["PostOrderContext.free_cash_now"] = (
    "`OrderContext.free_cash_now` value after order execution."
)
__pdoc__[
    "PostOrderContext.val_price_now"
] = """`OrderContext.val_price_now` value after order execution.

If `SimulationContext.update_value` is enabled, this value is updated with the fill price,
representing the most recently known price; otherwise, it remains unchanged.
"""
__pdoc__["PostOrderContext.value_now"] = """`OrderContext.value_now` value after order execution.

If `SimulationContext.update_value` is enabled, this value is updated with the new cash and column value;
otherwise, it remains unchanged.
"""
__pdoc__["PostOrderContext.return_now"] = "`OrderContext.return_now` value after order execution."
__pdoc__["PostOrderContext.pos_info_now"] = (
    "`OrderContext.pos_info_now` value after order execution."
)


class FlexOrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d
    cash_deposits: tp.FlexArray2d
    cash_earnings: tp.FlexArray2d
    segment_mask: tp.FlexArray2d
    call_pre_segment: bool
    call_post_segment: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    bm_close: tp.FlexArray2d
    ffill_val_price: bool
    update_value: bool
    fill_pos_info: bool
    track_value: bool

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d
    last_pos_info: tp.RecordArray

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: None
    call_idx: int


__pdoc__["FlexOrderContext"] = """Named tuple representing the flexible order context.

It includes all fields from `SegmentContext` along with the current call index.

Passed to `flex_order_func_nb`.
"""
for field in FlexOrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["FlexOrderContext." + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__["FlexOrderContext." + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__["FlexOrderContext." + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__["FlexOrderContext." + field] = f"See `SegmentContext.{field}`."
__pdoc__["FlexOrderContext.call_idx"] = "Index of the current call."


class Order(tp.NamedTuple):
    size: float = np.inf
    price: float = np.inf
    size_type: int = SizeType.Amount
    direction: int = Direction.Both
    fees: float = 0.0
    fixed_fees: float = 0.0
    slippage: float = 0.0
    min_size: float = np.nan
    max_size: float = np.nan
    size_granularity: float = np.nan
    leverage: float = 1.0
    leverage_mode: int = LeverageMode.Lazy
    reject_prob: float = 0.0
    price_area_vio_mode: int = PriceAreaVioMode.Ignore
    allow_partial: bool = True
    raise_reject: bool = False
    log: bool = False


__pdoc__["Order"] = """Named tuple representing an order.

!!! note
    Due to issues with Numba default handling in named tuples,
    use `vectorbtpro.portfolio.nb.core.order_nb` to create an order.
"""
__pdoc__["Order.size"] = """Size in units.

Behavior depends on `Order.size_type` and `Order.direction`.

For fixed sizes:

* Specify a numeric value to buy or sell a fixed amount or value.
* Use `np.inf` to buy using all available cash, or `-np.inf` to sell using all free cash.
    If `Order.direction` is not `Direction.Both`, `-np.inf` will close the position.
* Use `np.nan` or 0 to skip.

For target sizes:

* Specify a number to buy or sell an amount relative to the current position or value.
* Use 0 to close the current position.
* Use `np.nan` to skip.
"""
__pdoc__["Order.price"] = """Price per unit.

The final price is adjusted for slippage.

* If set to `-np.inf`, it will be replaced by the current open price.
* If set to `np.inf`, it will be replaced by the current close price.

!!! note
    Ensure that timestamps fall between (and ideally do not include) the current open and close prices.
"""
__pdoc__["Order.size_type"] = "See `SizeType`."
__pdoc__["Order.direction"] = "See `Direction`."
__pdoc__["Order.fees"] = """Fees as a percentage of the order value.

Negative values (e.g. -0.05) indicate earning 5% per trade instead of paying a fee.

!!! note
    0.01 represents 1%.
"""
__pdoc__["Order.fixed_fees"] = """Fixed fee amount charged for this order.

As with `Order.fees`, it can be negative.
"""
__pdoc__["Order.slippage"] = """Slippage expressed as a percentage of `Order.price`.

This represents a penalty applied to the order's price.

!!! note
    0.01 corresponds to 1%.
"""
__pdoc__["Order.min_size"] = """Minimum allowable size for an order.

Depends on `Order.size_type`. Values below this threshold will be rejected.
"""
__pdoc__["Order.max_size"] = """Maximum allowable size for an order.

Depends on `Order.size_type`. Values exceeding this limit may be partially filled.
"""
__pdoc__["Order.size_granularity"] = """Specifies the granularity of the order size.

For example, a granularity of 1.0 enforces integer-like quantities;
placing an order for 12.5 shares will result in ordering exactly 12.0 shares.

!!! note
    The filled size is still represented as a floating-point number.
"""
__pdoc__["Order.leverage"] = "Leverage factor."
__pdoc__["Order.leverage_mode"] = "See `LeverageMode`."
__pdoc__[
    "Order.reject_prob"
] = """Probability of order rejection, used to simulate random rejection events.

This can help test the robustness of the order management system.
"""
__pdoc__["Order.price_area_vio_mode"] = "See `PriceAreaVioMode`."
__pdoc__["Order.allow_partial"] = """Indicates whether partial filling of the order is permitted.

If False, orders that cannot be fully filled will be rejected.

Note: This does not apply when `Order.size` is `np.inf`.
"""
__pdoc__[
    "Order.raise_reject"
] = """Specifies whether an exception should be raised if the order is rejected.

If True, the simulation terminates upon order rejection.
"""
__pdoc__["Order.log"] = """Determines whether the order should be logged by creating a log record.

Remember to increase `max_log_records` if logging is enabled.
"""

NoOrder = Order(
    size=np.nan,
    price=np.nan,
    size_type=-1,
    direction=-1,
    fees=np.nan,
    fixed_fees=np.nan,
    slippage=np.nan,
    min_size=np.nan,
    max_size=np.nan,
    size_granularity=np.nan,
    leverage=1.0,
    leverage_mode=LeverageMode.Lazy,
    reject_prob=np.nan,
    price_area_vio_mode=-1,
    allow_partial=False,
    raise_reject=False,
    log=False,
)
"""_"""

__pdoc__["NoOrder"] = "Sentinel order indicating that no order should be processed."


class OrderResult(tp.NamedTuple):
    size: float
    price: float
    fees: float
    side: int
    status: int
    status_info: int


__pdoc__["OrderResult"] = """Named tuple representing the result of an order execution."""
__pdoc__["OrderResult.size"] = "Filled size of the order."
__pdoc__["OrderResult.price"] = "Filled price per unit, adjusted for slippage."
__pdoc__["OrderResult.fees"] = "Total fees paid for the order."
__pdoc__["OrderResult.side"] = "See `OrderSide`."
__pdoc__["OrderResult.status"] = "See `OrderStatus`."
__pdoc__["OrderResult.status_info"] = "See `OrderStatusInfo`."


class SignalSegmentContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    track_cash_deposits: bool
    cash_deposits_out: tp.Array2d
    track_cash_earnings: bool
    cash_earnings_out: tp.Array2d
    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d

    last_pos_info: tp.Array1d
    last_limit_info: tp.Array1d
    last_sl_info: tp.Array1d
    last_tsl_info: tp.Array1d
    last_tp_info: tp.Array1d
    last_td_info: tp.Array1d
    last_dt_info: tp.Array1d

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int


__pdoc__[
    "SignalSegmentContext"
] = """Named tuple representing the context for a simulation segment in a from-signals simulation.

This context includes OHLC data and additional internal simulation details computed at the start
of the simulation. For accessing other information, such as order size, consider using templates.

Passed to `post_segment_func_nb`.
"""
for field in SignalSegmentContext._fields:
    if field in SimulationContext._fields:
        __pdoc__["SignalSegmentContext." + field] = f"See `SimulationContext.{field}`."
for field in SignalSegmentContext._fields:
    if field in GroupContext._fields:
        __pdoc__["SignalSegmentContext." + field] = f"See `GroupContext.{field}`."
for field in SignalSegmentContext._fields:
    if field in RowContext._fields:
        __pdoc__["SignalSegmentContext." + field] = f"See `RowContext.{field}`."
__pdoc__[
    "SignalSegmentContext.track_cash_deposits"
] = """Indicates whether cash deposits are tracked.

Becomes True if any value in `cash_deposits` is nonzero.
"""
__pdoc__["SignalSegmentContext.cash_deposits_out"] = "See `SimulationOutput.cash_deposits`."
__pdoc__[
    "SignalSegmentContext.track_cash_earnings"
] = """Indicates whether cash earnings are tracked.

Becomes True if any value in `cash_earnings` is nonzero.
"""
__pdoc__["SignalSegmentContext.cash_earnings_out"] = "See `SimulationOutput.cash_earnings`."
__pdoc__["SignalSegmentContext.in_outputs"] = "See `FSInOutputs`."
__pdoc__[
    "SignalSegmentContext.last_limit_info"
] = """Record of type `limit_info_dt` for each column.

Accessible via `c.limit_info_dt[field][col]`.
"""
__pdoc__["SignalSegmentContext.last_sl_info"] = """Record of type `sl_info_dt` for each column.

Accessible via `c.last_sl_info[field][col]`.
"""
__pdoc__["SignalSegmentContext.last_tsl_info"] = """Record of type `tsl_info_dt` for each column.

Accessible via `c.last_tsl_info[field][col]`.
"""
__pdoc__["SignalSegmentContext.last_tp_info"] = """Record of type `tp_info_dt` for each column.

Accessible via `c.last_tp_info[field][col]`.
"""
__pdoc__["SignalSegmentContext.last_td_info"] = """Record of type `time_info_dt` for each column.

Accessible via `c.last_td_info[field][col]`.
"""
__pdoc__["SignalSegmentContext.last_dt_info"] = """Record of type `time_info_dt` for each column.

Accessible via `c.last_dt_info[field][col]`.
"""


class SignalContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    track_cash_deposits: bool
    cash_deposits_out: tp.Array2d
    track_cash_earnings: bool
    cash_earnings_out: tp.Array2d
    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d

    last_pos_info: tp.Array1d
    last_limit_info: tp.Array1d
    last_sl_info: tp.Array1d
    last_tsl_info: tp.Array1d
    last_tp_info: tp.Array1d
    last_td_info: tp.Array1d
    last_dt_info: tp.Array1d

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    col: int


__pdoc__[
    "SignalContext"
] = """Named tuple representing the context of an element in a from-signals simulation.

Contains all fields from `SignalSegmentContext` with an additional field `col` representing the column.

Passed to `signal_func_nb` and `adjust_func_nb`.
"""
for field in SignalContext._fields:
    if field in SignalSegmentContext._fields:
        __pdoc__["SignalContext." + field] = f"See `SignalSegmentContext.{field}`."
__pdoc__["SignalContext.col"] = "See `OrderContext.col`."


class PostSignalContext(tp.NamedTuple):
    target_shape: tp.Shape
    group_lens: tp.GroupLens
    cash_sharing: bool
    index: tp.Optional[tp.Array1d]
    freq: tp.Optional[int]
    open: tp.FlexArray2d
    high: tp.FlexArray2d
    low: tp.FlexArray2d
    close: tp.FlexArray2d
    init_cash: tp.FlexArray1d
    init_position: tp.FlexArray1d
    init_price: tp.FlexArray1d

    order_records: tp.RecordArray2d
    order_counts: tp.Array1d
    log_records: tp.RecordArray2d
    log_counts: tp.Array1d

    track_cash_deposits: bool
    cash_deposits_out: tp.Array2d
    track_cash_earnings: bool
    cash_earnings_out: tp.Array2d
    in_outputs: tp.Optional[tp.NamedTuple]

    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_locked_cash: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_return: tp.Array1d

    last_pos_info: tp.Array1d
    last_limit_info: tp.Array1d
    last_sl_info: tp.Array1d
    last_tsl_info: tp.Array1d
    last_tp_info: tp.Array1d
    last_td_info: tp.Array1d
    last_dt_info: tp.Array1d

    sim_start: tp.Array1d
    sim_end: tp.Array1d

    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    col: int

    cash_before: float
    position_before: float
    debt_before: float
    locked_cash_before: float
    free_cash_before: float
    val_price_before: float
    value_before: float

    order_result: "OrderResult"


__pdoc__[
    "PostSignalContext"
] = """Named tuple representing the context after an order has been processed in a from-signals simulation.

Contains all fields from `SignalContext` along with previous balance fields and the order result.

Passed to `post_signal_func_nb`.
"""
for field in PostSignalContext._fields:
    if field in SignalContext._fields:
        __pdoc__["PostSignalContext." + field] = f"See `SignalContext.{field}`."
__pdoc__["PostSignalContext.cash_before"] = "`ExecState.cash` balance before execution."
__pdoc__["PostSignalContext.position_before"] = "`ExecState.position` value before execution."
__pdoc__["PostSignalContext.debt_before"] = "`ExecState.debt` value before execution."
__pdoc__["PostSignalContext.locked_cash_before"] = "`ExecState.free_cash` value before execution."
__pdoc__["PostSignalContext.free_cash_before"] = "`ExecState.val_price` value before execution."
__pdoc__["PostSignalContext.val_price_before"] = "`ExecState.value` value before execution."
__pdoc__["PostSignalContext.order_result"] = "See `PostOrderContext.order_result`."


# ############# In-place outputs ############# #


class FOInOutputs(tp.NamedTuple):
    cash: tp.Array2d
    position: tp.Array2d
    debt: tp.Array2d
    locked_cash: tp.Array2d
    free_cash: tp.Array2d
    value: tp.Array2d
    returns: tp.Array2d


__pdoc__["FOInOutputs"] = (
    "Named tuple representing the in-place outputs for simulation based on orders."
)
__pdoc__["FOInOutputs.cash"] = """See `AccountState.cash`.

Follows groups if cash sharing is enabled; otherwise, follows columns.

Populated when `save_state` is True; otherwise, has shape `(0, 0)`.
"""
__pdoc__["FOInOutputs.position"] = """See `AccountState.position`.

Follows columns.

Populated when `save_state` is True; otherwise, has shape `(0, 0)`.
"""
__pdoc__["FOInOutputs.debt"] = """See `AccountState.debt`.

Follows columns.

Populated when `save_state` is True; otherwise, has shape `(0, 0)`.
"""
__pdoc__["FOInOutputs.locked_cash"] = """See `AccountState.locked_cash`.

Follows columns.

Populated when `save_state` is True; otherwise, shape is `(0, 0)`.
"""
__pdoc__["FOInOutputs.free_cash"] = """See `AccountState.free_cash`.

Follows groups if cash sharing is enabled; otherwise, follows columns.

Populated when `save_state` is True; otherwise, shape is `(0, 0)`.
"""
__pdoc__["FOInOutputs.value"] = """Value.

Follows groups if cash sharing is enabled; otherwise, follows columns.

Populated when `fill_value` is True; otherwise, shape is `(0, 0)`.
"""
__pdoc__["FOInOutputs.returns"] = """Returns.

Follows groups if cash sharing is enabled; otherwise, follows columns.

Populated when `save_returns` is True; otherwise, shape is `(0, 0)`.
"""


class FSInOutputs(tp.NamedTuple):
    cash: tp.Array2d
    position: tp.Array2d
    debt: tp.Array2d
    locked_cash: tp.Array2d
    free_cash: tp.Array2d
    value: tp.Array2d
    returns: tp.Array2d


__pdoc__["FSInOutputs"] = (
    "Named tuple representing the in-place outputs for simulation based on signals."
)
__pdoc__["FSInOutputs.cash"] = "See `FOInOutputs.cash`."
__pdoc__["FSInOutputs.position"] = "See `FOInOutputs.position`."
__pdoc__["FSInOutputs.debt"] = "See `FOInOutputs.debt`."
__pdoc__["FSInOutputs.locked_cash"] = "See `FOInOutputs.locked_cash`."
__pdoc__["FSInOutputs.free_cash"] = "See `FOInOutputs.free_cash`."
__pdoc__["FSInOutputs.value"] = "See `FOInOutputs.value`."
__pdoc__["FSInOutputs.returns"] = "See `FOInOutputs.returns`."

# ############# Records ############# #

order_fields = [
    ("id", int_),
    ("col", int_),
    ("idx", int_),
    ("size", float_),
    ("price", float_),
    ("fees", float_),
    ("side", int_),
]
"""Field definitions for the NumPy dtype `order_dt`."""

order_dt = np.dtype(order_fields, align=True)
"""_"""

__pdoc__["order_dt"] = f"""NumPy dtype for order records.

```python
{prettify_doc(order_dt)}
```

Fields:
    id: Order ID.
    col: Column index.
    idx: Index of the order in the column.
    size: Size of the order.
    price: Price of the order.
    fees: Fees associated with the order.
    side: Side of the order (buy/sell).

        See `OrderSide`.
"""

fs_order_fields = [
    ("id", int_),
    ("col", int_),
    ("signal_idx", int_),
    ("creation_idx", int_),
    ("idx", int_),
    ("size", float_),
    ("price", float_),
    ("fees", float_),
    ("side", int_),
    ("type", int_),
    ("stop_type", int_),
]
"""Field definitions for the NumPy dtype `fs_order_dt`."""

fs_order_dt = np.dtype(fs_order_fields, align=True)
"""_"""

__pdoc__["fs_order_dt"] = f"""NumPy dtype for order records generated from signals.

```python
{prettify_doc(fs_order_dt)}
```

Fields:
    id: Order ID.
    col: Column index.
    signal_idx: Index of the signal that generated the order.
    creation_idx: Index of the order creation.
    idx: Index of the order in the column.
    size: Size of the order.
    price: Price of the order.
    fees: Fees associated with the order.
    side: Side of the order (buy/sell).

        See `OrderSide`.
    type: Type of the order (market/limit).

        See `OrderType`.
    stop_type: Type of stop loss for the order (SL/TP/etc.).

        See `vectorbtpro.signals.enums.StopType`.
"""

trade_fields = [
    ("id", int_),
    ("col", int_),
    ("size", float_),
    ("entry_order_id", int_),
    ("entry_idx", int_),
    ("entry_price", float_),
    ("entry_fees", float_),
    ("exit_order_id", int_),
    ("exit_idx", int_),
    ("exit_price", float_),
    ("exit_fees", float_),
    ("pnl", float_),
    ("return", float_),
    ("direction", int_),
    ("status", int_),
    ("parent_id", int_),
]
"""Field definitions for the NumPy dtype `trade_dt`."""

trade_dt = np.dtype(trade_fields, align=True)
"""_"""

__pdoc__["trade_dt"] = f"""NumPy dtype for trade records.

```python
{prettify_doc(trade_dt)}
```

Fields:
    id: Trade ID.
    col: Column index.
    size: Size of the trade.
    entry_order_id: ID of the entry order.
    entry_idx: Row index of the entry order.
    entry_price: Size-weighted average entry price of the trade.
    entry_fees: Fees associated with the entry order.
    exit_order_id: ID of the exit order.
    exit_idx: Row index of the exit order.
    exit_price: Size-weighted average exit price of the trade.
    exit_fees: Fees associated with the exit order.
    pnl: Profit and loss from the trade.
    return: Return from the trade.
    direction: Direction of the trade (buy/sell).

        See `vectorbtpro.portfolio.enums.TradeDirection`.
    status: Status of the trade (open/closed).

        See `vectorbtpro.portfolio.enums.TradeStatus`.
    parent_id: ID of the parent trade or position.
"""

log_fields = [
    ("id", int_),
    ("group", int_),
    ("col", int_),
    ("idx", int_),
    ("price_area_open", float_),
    ("price_area_high", float_),
    ("price_area_low", float_),
    ("price_area_close", float_),
    ("st0_cash", float_),
    ("st0_position", float_),
    ("st0_debt", float_),
    ("st0_locked_cash", float_),
    ("st0_free_cash", float_),
    ("st0_val_price", float_),
    ("st0_value", float_),
    ("req_size", float_),
    ("req_price", float_),
    ("req_size_type", int_),
    ("req_direction", int_),
    ("req_fees", float_),
    ("req_fixed_fees", float_),
    ("req_slippage", float_),
    ("req_min_size", float_),
    ("req_max_size", float_),
    ("req_size_granularity", float_),
    ("req_leverage", float_),
    ("req_leverage_mode", int_),
    ("req_reject_prob", float_),
    ("req_price_area_vio_mode", int_),
    ("req_allow_partial", np.bool_),
    ("req_raise_reject", np.bool_),
    ("req_log", np.bool_),
    ("res_size", float_),
    ("res_price", float_),
    ("res_fees", float_),
    ("res_side", int_),
    ("res_status", int_),
    ("res_status_info", int_),
    ("st1_cash", float_),
    ("st1_position", float_),
    ("st1_debt", float_),
    ("st1_locked_cash", float_),
    ("st1_free_cash", float_),
    ("st1_val_price", float_),
    ("st1_value", float_),
    ("order_id", int_),
]
"""Field definitions for the NumPy dtype `log_dt`."""

log_dt = np.dtype(log_fields, align=True)
"""_"""

__pdoc__["log_dt"] = f"""NumPy dtype for log records.

```python
{prettify_doc(log_dt)}
```

Fields:
    id: Log ID.
    group: Group index.
    col: Column index.
    idx: Row index.
    price_area_open: Open price.
    price_area_high: High price.
    price_area_low: Low price.
    price_area_close: Close price.
    st0_cash: Initial cash value.
    st0_position: Initial position.
    st0_debt: Initial debt value.
    st0_locked_cash: Initial locked cash value.
    st0_free_cash: Initial free cash value.
    st0_val_price: Initial valuation price.
    st0_value: Initial value.
    req_size: Requested size for the order.
    req_price: Requested price for the order.
    req_size_type: Size type for the order.

        See `SizeType`.
    req_direction: Direction for the order.

        See `Direction`.
    req_fees: Requested fees for the order.
    req_fixed_fees: Requested fixed fees for the order.
    req_slippage: Requested slippage for the order.
    req_min_size: Requested minimum size for the order.
    req_max_size: Requested maximum size for the order.
    req_size_granularity: Requested size granularity for the order.
    req_leverage: Requested leverage for the order.
    req_leverage_mode: Requested leverage mode for the order.

        See `LeverageMode`.
    req_reject_prob: Requested rejection probability for the order.
    req_price_area_vio_mode: Requested price area violation mode.

        See `PriceAreaVioMode`.
    req_allow_partial: Requested flag indicating if partial filling is allowed.

        See `AllowPartial`.
    req_raise_reject: Requested flag indicating if rejection should raise an exception.

        See `RaiseReject`.
    req_log: Requested flag indicating if logging is enabled.

        See `Log`.
    res_size: Resulting size of the order after execution.
    res_price: Resulting price of the order after execution.
    res_fees: Resulting fees of the order after execution.
    res_side: Resulting side of the resulting order.

        See `OrderSide`.
    res_status: Status of the resulting order.

        See `OrderStatus`.
    res_status_info: Status information of the resulting order.

        See `OrderStatusInfo`.
    st1_cash: Final cash value after execution.
    st1_position: Final position after execution.
    st1_debt: Final debt value after execution.
    st1_locked_cash: Final locked cash value after execution.
    st1_free_cash: Final free cash value after execution.
    st1_val_price: Final valuation price after execution.
    st1_value: Final value after execution.
    order_id: ID of the order.
"""

alloc_range_fields = [
    ("id", int_),
    ("col", int_),
    ("start_idx", int_),
    ("end_idx", int_),
    ("alloc_idx", int_),
    ("status", int_),
]
"""Field definitions for the NumPy dtype `alloc_range_dt`."""

alloc_range_dt = np.dtype(alloc_range_fields, align=True)
"""_"""

__pdoc__["alloc_range_dt"] = f"""NumPy dtype for allocation range records.

```python
{prettify_doc(alloc_range_dt)}
```

Fields:
    id: Allocation range ID.
    col: Column index.
    start_idx: Start row index of the allocation range.
    end_idx: End row index of the allocation range.
    alloc_idx: Row index of the allocation.
    status: Status of the allocation range.

        See `vectorbtpro.generic.enums.RangeStatus`.
"""

alloc_point_fields = [
    ("id", int_),
    ("col", int_),
    ("alloc_idx", int_),
]
"""Field definitions for the NumPy dtype `alloc_point_dt`."""

alloc_point_dt = np.dtype(alloc_point_fields, align=True)
"""_"""

__pdoc__["alloc_point_dt"] = f"""NumPy dtype for allocation point records.

```python
{prettify_doc(alloc_point_dt)}
```

Fields:
    id: Allocation point ID.
    col: Column index.
    alloc_idx: Row index of the allocation.
"""

# ############# Info records ############# #

main_info_fields = [
    ("bar_zone", int_),
    ("signal_idx", int_),
    ("creation_idx", int_),
    ("idx", int_),
    ("val_price", float_),
    ("price", float_),
    ("size", float_),
    ("size_type", int_),
    ("direction", int_),
    ("type", int_),
    ("stop_type", int_),
]
"""Field definitions for the NumPy dtype `main_info_dt`."""

main_info_dt = np.dtype(main_info_fields, align=True)
"""_"""

__pdoc__["main_info_dt"] = f"""NumPy dtype for main information records.

```python
{prettify_doc(main_info_dt)}
```

Fields:
    bar_zone: See `vectorbtpro.generic.enums.BarZone`.
    signal_idx: Row where signal was placed.
    creation_idx: Row where order was created.
    i: Row from which order information was taken.
    val_price: Valuation price.
    price: Requested price.
    size: Order size.
    size_type: See `SizeType`.
    direction: See `Direction`.
    type: See `OrderType`.
    stop_type: See `vectorbtpro.signals.enums.StopType`.
"""

limit_info_fields = [
    ("signal_idx", int_),
    ("creation_idx", int_),
    ("init_idx", int_),
    ("init_price", float_),
    ("init_size", float_),
    ("init_size_type", int_),
    ("init_direction", int_),
    ("init_stop_type", int_),
    ("delta", float_),
    ("delta_format", int_),
    ("tif", np.int64),
    ("expiry", np.int64),
    ("time_delta_format", int_),
    ("reverse", float_),
    ("order_price", float_),
]
"""Field definitions for the NumPy dtype `limit_info_dt`."""

limit_info_dt = np.dtype(limit_info_fields, align=True)
"""_"""

__pdoc__["limit_info_dt"] = f"""NumPy dtype for limit information records.

```python
{prettify_doc(limit_info_dt)}
```

Fields:
    signal_idx: Row index for the signal.
    creation_idx: Row index for limit creation.
    init_idx: Initial order row index from which information is taken.
    init_price: Initial price.
    init_size: Order size.
    init_size_type: See `SizeType`.
    init_direction: See `Direction`.
    init_stop_type: See `vectorbtpro.signals.enums.StopType`.
    delta: Price delta from the initial price.
    delta_format: See `DeltaFormat`.
    tif: Time in force as an integer. Use `-1` to disable.
    expiry: Expiry time as an integer. Use `-1` to disable.
    time_delta_format: See `TimeDeltaFormat`.
    reverse: Flag indicating if price hit detection should be reversed.
    order_price: See `LimitOrderPrice`.
"""

sl_info_fields = [
    ("init_idx", int_),
    ("init_price", float_),
    ("init_position", float_),
    ("stop", float_),
    ("exit_price", float_),
    ("exit_size", float_),
    ("exit_size_type", int_),
    ("exit_type", int_),
    ("order_type", int_),
    ("limit_delta", float_),
    ("delta_format", int_),
    ("ladder", int_),
    ("step", int_),
    ("step_idx", int_),
]
"""Field definitions for the NumPy dtype `sl_info_dt`."""

sl_info_dt = np.dtype(sl_info_fields, align=True)
"""_"""

__pdoc__["sl_info_dt"] = f"""NumPy dtype for SL information records.

```python
{prettify_doc(sl_info_dt)}
```

Fields:
    init_idx: Row index for initial order.
    init_price: Initial order price.
    init_position: Initial position.
    stop: Updated stop value.
    exit_price: See `StopExitPrice`.
    exit_size: Order size.
    exit_size_type: See `SizeType`.
    exit_type: See `StopExitType`.
    order_type: See `OrderType`.
    limit_delta: Delta from the hit price, applicable only for `StopType.Limit`.
    delta_format: See `DeltaFormat`.
    ladder: See `StopLadderMode`.
    step: Ladder step count (number of stop executions).
    step_idx: Row index corresponding to the ladder step.
"""

tsl_info_fields = [
    ("init_idx", int_),
    ("init_price", float_),
    ("init_position", float_),
    ("peak_idx", int_),
    ("peak_price", float_),
    ("stop", float_),
    ("th", float_),
    ("exit_price", float_),
    ("exit_size", float_),
    ("exit_size_type", int_),
    ("exit_type", int_),
    ("order_type", int_),
    ("limit_delta", float_),
    ("delta_format", int_),
    ("ladder", int_),
    ("step", int_),
    ("step_idx", int_),
]
"""Field definitions for the NumPy dtype `tsl_info_dt`."""

tsl_info_dt = np.dtype(tsl_info_fields, align=True)
"""_"""

__pdoc__["tsl_info_dt"] = f"""NumPy dtype for TSL information records.

```python
{prettify_doc(tsl_info_dt)}
```

Fields:
    init_idx: Row index for the initial order.
    init_price: Initial order price.
    init_position: Initial position.
    peak_idx: Row index at which the peak price occurred.
    peak_price: Peak price value.
    stop: Updated stop value.
    th: Updated threshold value.
    exit_price: See `StopExitPrice`.
    exit_size: Order size.
    exit_size_type: See `SizeType`.
    exit_type: See `StopExitType`.
    order_type: See `OrderType`.
    limit_delta: Delta from the hit price, applicable only for `StopType.Limit`.
    delta_format: See `DeltaFormat`.
    ladder: See `StopLadderMode`.
    step: Ladder step count (number of stop executions).
    step_idx: Row index corresponding to the ladder step.
"""

tp_info_fields = [
    ("init_idx", int_),
    ("init_price", float_),
    ("init_position", float_),
    ("stop", float_),
    ("exit_price", float_),
    ("exit_size", float_),
    ("exit_size_type", int_),
    ("exit_type", int_),
    ("order_type", int_),
    ("limit_delta", float_),
    ("delta_format", int_),
    ("ladder", int_),
    ("step", int_),
    ("step_idx", int_),
]
"""Field definitions for the NumPy dtype `tp_info_dt`."""

tp_info_dt = np.dtype(tp_info_fields, align=True)
"""_"""

__pdoc__["tp_info_dt"] = f"""NumPy dtype for TP information records.

```python
{prettify_doc(tp_info_dt)}
```

Fields:
    init_idx: Initial order row index.
    init_price: Initial order price.
    init_position: Initial position.
    stop: Updated stop value.
    exit_price: See `StopExitPrice`.
    exit_size: Order size.
    exit_size_type: See `SizeType`.
    exit_type: See `StopExitType`.
    order_type: See `OrderType`.
    limit_delta: Delta from the hit price, applicable only for `StopType.Limit`.
    delta_format: See `DeltaFormat`.
    ladder: See `StopLadderMode`.
    step: Ladder step count (number of stop executions).
    step_idx: Row index corresponding to the ladder step.
"""

time_info_fields = [
    ("init_idx", int_),
    ("init_position", float_),
    ("stop", np.int64),
    ("exit_price", float_),
    ("exit_size", float_),
    ("exit_size_type", int_),
    ("exit_type", int_),
    ("order_type", int_),
    ("limit_delta", float_),
    ("delta_format", int_),
    ("time_delta_format", int_),
    ("ladder", int_),
    ("step", int_),
    ("step_idx", int_),
]
"""Field definitions for the NumPy dtype `time_info_dt`."""

time_info_dt = np.dtype(time_info_fields, align=True)
"""_"""

__pdoc__["time_info_dt"] = f"""NumPy dtype for time information records.

```python
{prettify_doc(time_info_dt)}
```

Fields:
    init_idx: Initial row index.
    init_position: Initial position.
    stop: Updated stop value.
    exit_price: See `StopExitPrice`.
    exit_size: Order size.
    exit_size_type: See `SizeType`.
    exit_type: See `StopExitType`.
    order_type: See `OrderType`.
    limit_delta: Delta from the hit price, applicable only for `StopType.Limit`.
    delta_format: See `DeltaFormat` (used only for `StopType.Limit`).
    time_delta_format: See `TimeDeltaFormat`.
    ladder: See `StopLadderMode`.
    step: Ladder step count (number of stop executions).
    step_idx: Row index corresponding to the ladder step.
"""
