# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled context helper functions for portfolio simulation."""

from vectorbtpro.base.flex_indexing import flex_select_col_nb
from vectorbtpro.portfolio.nb import records as pf_records_nb
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.portfolio.nb.iter_ import select_nb
from vectorbtpro.records import nb as records_nb

# ############# Position ############# #


@register_jitted
def get_col_position_nb(c: tp.NamedTuple, col: int) -> float:
    """Return the position of a specified column from the context.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        float: Position value of the specified column.
    """
    return c.last_position[col]


@register_jitted
def get_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Return the position of the current column based on the provided context.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Position value of the current column.
    """
    return get_col_position_nb(c, c.col)


@register_jitted
def col_in_position_nb(c: tp.NamedTuple, col: int) -> bool:
    """Return whether a specified column is currently in a position.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        bool: True if the column's position is non-zero, otherwise False.
    """
    position = get_col_position_nb(c, col)
    return position != 0


@register_jitted
def in_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Return whether the current column is in a position.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if the current column's position is non-zero, otherwise False.
    """
    return col_in_position_nb(c, c.col)


@register_jitted
def col_in_long_position_nb(c: tp.NamedTuple, col: int) -> bool:
    """Return whether a specified column is in a long position.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        bool: True if the column's position is greater than 0, otherwise False.
    """
    position = get_col_position_nb(c, col)
    return position > 0


@register_jitted
def in_long_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Return whether the current column is in a long position.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if the current column's position is positive, otherwise False.
    """
    return col_in_long_position_nb(c, c.col)


@register_jitted
def col_in_short_position_nb(c: tp.NamedTuple, col: int) -> bool:
    """Return whether a specified column is in a short position.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        bool: True if the column's position is less than 0, otherwise False.
    """
    position = get_col_position_nb(c, col)
    return position < 0


@register_jitted
def in_short_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Return whether the current column is in a short position.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if the current column's position is negative, otherwise False.
    """
    return col_in_short_position_nb(c, c.col)


@register_jitted
def get_n_active_positions_nb(
    c: tp.Union[
        GroupContext,
        SegmentContext,
        OrderContext,
        PostOrderContext,
        FlexOrderContext,
        SignalSegmentContext,
        SignalContext,
        PostSignalContext,
    ],
    all_groups: bool = False,
) -> int:
    """Return the number of active positions in the current group or across all groups.

    Args:
        c (Union[GroupContext, SegmentContext, OrderContext, PostOrderContext, FlexOrderContext, SignalSegmentContext, SignalContext, PostSignalContext]):
            Relevant context.
        all_groups (bool): Flag indicating whether to count active positions across all groups.

    Returns:
        int: Total number of active positions.
    """
    n_active_positions = 0
    if all_groups:
        for col in range(c.target_shape[1]):
            if c.last_position[col] != 0:
                n_active_positions += 1
    else:
        for col in range(c.from_col, c.to_col):
            if c.last_position[col] != 0:
                n_active_positions += 1
    return n_active_positions


# ############# Cash ############# #


@register_jitted
def get_col_cash_nb(c: tp.NamedTuple, col: int) -> float:
    """Return the cash for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        float: Cash value for the specified column.

    Raises:
        ValueError: If cash sharing is enabled, indicating that cash cannot be retrieved for a single column.
    """
    if c.cash_sharing:
        raise ValueError(
            "Cannot get cash of a single column from a group with cash sharing. "
            "Use get_group_cash_nb."
        )
    return c.last_cash[col]


@register_jitted
def get_group_cash_nb(c: tp.NamedTuple, group: int) -> float:
    """Return the cash for a specified group.

    Args:
        c (NamedTuple): Context.
        group (int): Index of the group.

    Returns:
        float: Total cash value for the specified group.
    """
    if c.cash_sharing:
        return c.last_cash[group]
    cash = 0.0
    from_col = 0
    for g in range(len(c.group_lens)):
        to_col = from_col + c.group_lens[g]
        if g == group:
            for col in range(from_col, to_col):
                cash += c.last_cash[col]
            break
        from_col = to_col
    return cash


@register_jitted
def get_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Return the cash for the current column or group based on cash sharing.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Cash value of the current column if cash sharing is disabled;
            otherwise, the cash value of the current group.
    """
    if c.cash_sharing:
        return get_group_cash_nb(c, c.group)
    return get_col_cash_nb(c, c.col)


# ############# Debt ############# #


@register_jitted
def get_col_debt_nb(c: tp.NamedTuple, col: int) -> float:
    """Return the debt for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        float: Debt value of the specified column.
    """
    return c.last_debt[col]


@register_jitted
def get_debt_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Return the debt for the current column.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Debt value of the current column.
    """
    return get_col_debt_nb(c, c.col)


# ############# Locked cash ############# #


@register_jitted
def get_col_locked_cash_nb(c: tp.NamedTuple, col: int) -> float:
    """Return the locked cash for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        float: Locked cash value of the specified column.
    """
    return c.last_locked_cash[col]


@register_jitted
def get_locked_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Return the locked cash for the current column.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Locked cash value of the current column.
    """
    return get_col_locked_cash_nb(c, c.col)


# ############# Free cash ############# #


@register_jitted
def get_col_free_cash_nb(c: tp.NamedTuple, col: int) -> float:
    """Return the free cash for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        float: Free cash value of the specified column.

    Raises:
        ValueError: If cash sharing is enabled, indicating that free cash cannot be retrieved
            for a single column.
    """
    if c.cash_sharing:
        raise ValueError(
            "Cannot get free cash of a single column from a group with cash sharing. "
            "Use get_group_free_cash_nb."
        )
    return c.last_free_cash[col]


@register_jitted
def get_group_free_cash_nb(c: tp.NamedTuple, group: int) -> float:
    """Return the free cash for a specified group.

    Args:
        c (NamedTuple): Context.
        group (int): Index of the group.

    Returns:
        float: Total free cash value for the specified group.
    """
    if c.cash_sharing:
        return c.last_free_cash[group]
    free_cash = 0.0
    from_col = 0
    for g in range(len(c.group_lens)):
        to_col = from_col + c.group_lens[g]
        if g == group:
            for col in range(from_col, to_col):
                free_cash += c.last_free_cash[col]
            break
        from_col = to_col
    return free_cash


@register_jitted
def get_free_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Return the free cash for the current column or group based on cash sharing.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Free cash value of the current column if cash sharing is disabled;
            otherwise, the free cash value of the current group.
    """
    if c.cash_sharing:
        return get_group_free_cash_nb(c, c.group)
    return get_col_free_cash_nb(c, c.col)


@register_jitted
def col_has_free_cash_nb(c: tp.NamedTuple, col: int) -> float:
    """Return whether a specified column has free cash available.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        bool: True if the free cash value is greater than zero, otherwise False.
    """
    return get_col_free_cash_nb(c, col) > 0


@register_jitted
def group_has_free_cash_nb(c: tp.NamedTuple, group: int) -> float:
    """Return whether a specified group has free cash available.

    Args:
        c (NamedTuple): Context.
        group (int): Index of the group.

    Returns:
        bool: True if the total free cash is greater than zero, otherwise False.
    """
    return get_group_free_cash_nb(c, group) > 0


@register_jitted
def has_free_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Return whether the current column or group with cash sharing has free cash available.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if free cash is available, otherwise False.
    """
    if c.cash_sharing:
        return group_has_free_cash_nb(c, c.group)
    return col_has_free_cash_nb(c, c.col)


# ############# Valuation price ############# #


@register_jitted
def get_col_val_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Return the valuation price for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        float: Valuation price of the specified column.
    """
    return c.last_val_price[col]


@register_jitted
def get_val_price_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Return the valuation price for the current column.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Valuation price of the current column.
    """
    return get_col_val_price_nb(c, c.col)


# ############# Value ############# #


@register_jitted
def get_col_value_nb(c: tp.NamedTuple, col: int) -> float:
    """Retrieve the value of a specified column from the context.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column to retrieve the value from.

    Returns:
        float: Value at the specified column.

    Raises:
        ValueError: If cash sharing is enabled, indicating that the value cannot be retrieved
            for a single column.
    """
    if c.cash_sharing:
        raise ValueError(
            "Cannot get value of a single column from a group with cash sharing. "
            "Use get_group_value_nb."
        )
    return c.last_value[col]


@register_jitted
def get_group_value_nb(c: tp.NamedTuple, group: int) -> float:
    """Retrieve the aggregated value of a specified group from the context.

    Args:
        c (NamedTuple): Context.
        group (int): Index of the group to retrieve the value for.

    Returns:
        float: Total value aggregated from all columns in the specified group.
    """
    if c.cash_sharing:
        return c.last_value[group]
    value = 0.0
    from_col = 0
    for g in range(len(c.group_lens)):
        to_col = from_col + c.group_lens[g]
        if g == group:
            for col in range(from_col, to_col):
                value += c.last_value[col]
            break
        from_col = to_col
    return value


@register_jitted
def get_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Retrieve the value of the current column or group based on cash sharing.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Value of the current column, or if cash sharing is enabled, the aggregated group value.
    """
    if c.cash_sharing:
        return get_group_value_nb(c, c.group)
    return get_col_value_nb(c, c.col)


# ############# Leverage ############# #


@register_jitted
def get_col_leverage_nb(c: tp.NamedTuple, col: int) -> float:
    """Calculate the leverage of a specified column based on its financial metrics.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column for which to calculate leverage.

    Returns:
        float: Leverage computed as debt divided by locked cash, incremented by one if
            a position exists, or NaN if locked cash is zero.
    """
    position = get_col_position_nb(c, col)
    debt = get_col_debt_nb(c, col)
    locked_cash = get_col_locked_cash_nb(c, col)
    if locked_cash == 0:
        return np.nan
    leverage = debt / locked_cash
    if position > 0:
        leverage += 1
    return leverage


@register_jitted
def get_leverage_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Retrieve the leverage of the current column from the context.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Leverage of the current column.
    """
    return get_col_leverage_nb(c, c.col)


# ############# Allocation ############# #


@register_jitted
def get_col_position_value_nb(c: tp.NamedTuple, col: int) -> float:
    """Calculate the position value for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        float: Product of the column's position and its valuation price, or 0.0 if the position is zero.
    """
    position = get_col_position_nb(c, col)
    val_price = get_col_val_price_nb(c, col)
    if position == 0:
        return 0.0
    return position * val_price


@register_jitted
def get_group_position_value_nb(c: tp.NamedTuple, group: int) -> float:
    """Aggregate the position value for a specified group by summing the values of its constituent columns.

    Args:
        c (NamedTuple): Context.
        group (int): Index of the group to calculate the total position value for.

    Returns:
        float: Aggregated position value of the specified group.
    """
    value = 0.0
    from_col = 0
    for g in range(len(c.group_lens)):
        to_col = from_col + c.group_lens[g]
        if g == group:
            for col in range(from_col, to_col):
                value += get_col_position_value_nb(c, col)
            break
        from_col = to_col
    return value


@register_jitted
def get_position_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Retrieve the position value of the current column from the context.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Position value of the current column.
    """
    return get_col_position_value_nb(c, c.col)


@register_jitted
def get_col_allocation_nb(c: tp.NamedTuple, col: int, group: tp.Optional[int] = None) -> float:
    """Calculate the allocation of a specified column within its group.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.
        group (Optional[int]): Index of the group.

            If not provided, the group is determined based on the column index.

    Returns:
        float: Allocation ratio computed as the column's position value divided by
            the group's total value, or 0.0 if the position value is zero and NaN if the
            group value is non-positive.
    """
    position_value = get_col_position_value_nb(c, col)
    if group is None:
        from_col = 0
        found = False
        for _group in range(len(c.group_lens)):
            to_col = from_col + c.group_lens[_group]
            if from_col <= col < to_col:
                found = True
                break
            from_col = to_col
        if not found:
            raise ValueError("Column out of bounds")
    else:
        _group = group
    value = get_group_value_nb(c, _group)
    if position_value == 0:
        return 0.0
    if value <= 0:
        return np.nan
    return position_value / value


@register_jitted
def get_allocation_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Retrieve the allocation of the current column within its current group.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Allocation ratio for the current column.
    """
    return get_col_allocation_nb(c, c.col, group=c.group)


# ############# Orders ############# #


@register_jitted
def get_col_order_count_nb(c: tp.NamedTuple, col: int) -> int:
    """Retrieve the number of order records for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        int: Number of order records for the specified column.
    """
    return c.order_counts[col]


@register_jitted
def get_order_count_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> int:
    """Retrieve the number of order records for the current column from the context.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        int: Count of order records for the current column.
    """
    return get_col_order_count_nb(c, c.col)


@register_jitted
def get_col_order_records_nb(c: tp.NamedTuple, col: int) -> tp.RecordArray:
    """Retrieve the order records for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        RecordArray: Order records for the specified column.
    """
    order_count = get_col_order_count_nb(c, col)
    return c.order_records[:order_count, col]


@register_jitted
def get_order_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> tp.RecordArray:
    """Retrieve the order records for the current column from the context.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        RecordArray: Order records for the current column.
    """
    return get_col_order_records_nb(c, c.col)


@register_jitted
def col_has_orders_nb(c: tp.NamedTuple, col: int) -> bool:
    """Determine whether any order records exist for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        bool: True if there is at least one order record, otherwise False.
    """
    return get_col_order_count_nb(c, col) > 0


@register_jitted
def has_orders_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Determine whether any order records exist for the current column.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if there is at least one order record for the current column, otherwise False.
    """
    return col_has_orders_nb(c, c.col)


@register_jitted
def get_col_last_order_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Retrieve the last order record for a specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Index of the column.

    Returns:
        Record: Last order record for the specified column.

    Raises:
        ValueError: If there are no orders for the specified column.
    """
    if not col_has_orders_nb(c, col):
        raise ValueError("There are no orders. Check for any orders first.")
    return get_col_order_records_nb(c, col)[-1]


@register_jitted
def get_last_order_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Retrieve the last order record for the current column from the context.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        Record: Last order record for the current column.
    """
    return get_col_last_order_nb(c, c.col)


# ############# Order result ############# #


@register_jitted
def order_filled_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ],
) -> bool:
    """Determine if the order has been filled.

    Args:
        c (Union[PostOrderContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if the order status is filled, otherwise False.
    """
    return c.order_result.status == OrderStatus.Filled


@register_jitted
def order_opened_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ],
) -> bool:
    """Determine if the order has opened a new position.

    Args:
        c (Union[PostOrderContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if the order either reversed a position or changed the position from
            zero to non-zero, otherwise False.
    """
    position_now = get_position_nb(c)
    return order_reversed_position_nb(c) or (c.position_before == 0 and position_now != 0)


@register_jitted
def order_increased_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ],
) -> bool:
    """Determine if the order has opened a new position or increased an existing position.

    Args:
        c (Union[PostOrderContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if the order resulted in a new position or increased the absolute size of
            an existing position, otherwise False.
    """
    position_now = get_position_nb(c)
    return order_opened_position_nb(c) or (
        np.sign(position_now) == np.sign(c.position_before)
        and abs(position_now) > abs(c.position_before)
    )


@register_jitted
def order_decreased_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ],
) -> bool:
    """Determine if the order has decreased or closed an existing position.

    Args:
        c (Union[PostOrderContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if the order reduced the position size, closed the position, or
            reversed it, otherwise False.
    """
    position_now = get_position_nb(c)
    return (
        order_closed_position_nb(c)
        or order_reversed_position_nb(c)
        or (
            np.sign(position_now) == np.sign(c.position_before)
            and abs(position_now) < abs(c.position_before)
        )
    )


@register_jitted
def order_closed_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ],
) -> bool:
    """Determine if the order has completely closed an existing position.

    Args:
        c (Union[PostOrderContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if there was an existing position and the current position is zero,
            indicating the position has been closed, otherwise False.
    """
    position_now = get_position_nb(c)
    return c.position_before != 0 and position_now == 0


@register_jitted
def order_reversed_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ],
) -> bool:
    """Check whether the order has reversed an existing position.

    Args:
        c (Union[PostOrderContext, PostSignalContext]):
            Relevant context.

    Returns:
        bool: True if the order reverses an existing position, otherwise False.
    """
    position_now = get_position_nb(c)
    return (
        c.position_before != 0
        and position_now != 0
        and np.sign(c.position_before) != np.sign(position_now)
    )


# ############# Limit orders ############# #


@register_jitted
def get_col_limit_info_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get limit order information for the specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Column index from which to retrieve the limit order information.

    Returns:
        Record: Limit order information for the specified column.
    """
    return c.last_limit_info[col]


@register_jitted
def get_limit_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get limit order information for the current column.

    Args:
        c (Union[SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        Record: Limit order information for the current column.
    """
    return get_col_limit_info_nb(c, c.col)


@register_jitted
def get_col_limit_target_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get target price of the limit order for the specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Column index for retrieving the limit order target price.

    Returns:
        float: Target price of the limit order or NaN if no active position.
    """
    if not col_in_position_nb(c, col):
        return np.nan
    limit_info = get_col_limit_info_nb(c, col)
    return get_limit_info_target_price_nb(limit_info)


@register_jitted
def get_limit_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get target price of the limit order for the current column.

    Args:
        c (Union[SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Target price of the limit order or NaN if no active position.
    """
    return get_col_limit_target_price_nb(c, c.col)


# ############# Stop orders ############# #


@register_jitted
def get_col_sl_info_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get stop-loss (SL) order information for the specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Column index from which to retrieve the SL order information.

    Returns:
        Record: Stop-loss order information for the specified column.
    """
    return c.last_sl_info[col]


@register_jitted
def get_sl_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get stop-loss (SL) order information for the current column.

    Args:
        c (Union[SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        Record: Stop-loss order information for the current column.
    """
    return get_col_sl_info_nb(c, c.col)


@register_jitted
def get_col_sl_target_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get target price of the stop-loss (SL) order for the specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Column index for retrieving the SL target price.

    Returns:
        float: Target price of the SL order or NaN if no active position.
    """
    if not col_in_position_nb(c, col):
        return np.nan
    position = get_col_position_nb(c, col)
    sl_info = get_col_sl_info_nb(c, col)
    return get_sl_info_target_price_nb(sl_info, position)


@register_jitted
def get_sl_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get target price of the stop-loss (SL) order for the current column.

    Args:
        c (Union[SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Target price of the SL order or NaN if no active position.
    """
    return get_col_sl_target_price_nb(c, c.col)


@register_jitted
def get_col_tsl_info_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get trailing stop-loss (TSL/TTP) order information for the specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Column index from which to retrieve the TSL/TTP order information.

    Returns:
        Record: Trailing stop-loss order information for the specified column.
    """
    return c.last_tsl_info[col]


@register_jitted
def get_tsl_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get trailing stop-loss (TSL/TTP) order information for the current column.

    Args:
        c (Union[SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        Record: Trailing stop-loss order information for the current column.
    """
    return get_col_tsl_info_nb(c, c.col)


@register_jitted
def get_col_tsl_target_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get target price of the trailing stop-loss (TSL/TTP) order for the specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Column index for retrieving the TSL/TTP target price.

    Returns:
        float: Target price of the TSL/TTP order or NaN if no active position.
    """
    if not col_in_position_nb(c, col):
        return np.nan
    position = get_col_position_nb(c, col)
    tsl_info = get_col_tsl_info_nb(c, col)
    return get_tsl_info_target_price_nb(tsl_info, position)


@register_jitted
def get_tsl_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get target price of the trailing stop-loss (TSL/TTP) order for the current column.

    Args:
        c (Union[SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Target price of the TSL/TTP order or NaN if no active position.
    """
    return get_col_tsl_target_price_nb(c, c.col)


@register_jitted
def get_col_tp_info_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get take-profit (TP) order information for the specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Column index from which to retrieve the TP order information.

    Returns:
        Record: Take-profit order information for the specified column.
    """
    return c.last_tp_info[col]


@register_jitted
def get_tp_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get take-profit (TP) order information for the current column.

    Args:
        c (Union[SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        Record: Take-profit order information for the current column.
    """
    return get_col_tp_info_nb(c, c.col)


@register_jitted
def get_col_tp_target_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get target price of the take-profit (TP) order for the specified column.

    Args:
        c (NamedTuple): Context.
        col (int): Column index for retrieving the TP target price.

    Returns:
        float: Target price of the TP order or NaN if no active position.
    """
    if not col_in_position_nb(c, col):
        return np.nan
    position = get_col_position_nb(c, col)
    tp_info = get_col_tp_info_nb(c, col)
    return get_tp_info_target_price_nb(tp_info, position)


@register_jitted
def get_tp_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get target price of the take-profit (TP) order for the current column.

    Args:
        c (Union[SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        float: Target price of the TP order or NaN if no active position.
    """
    return get_col_tp_target_price_nb(c, c.col)


# ############# Trades ############# #


@register_jitted
def get_col_entry_trade_records_nb(
    c: tp.NamedTuple,
    col: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get entry trade records for the specified column up to the current point.

    Args:
        c (NamedTuple): Context.
        col (int): Column index for which to retrieve entry trade records.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

    Returns:
        Array1d: Entry trade records for the specified column.
    """
    order_records = get_col_order_records_nb(c, col)
    col_map = records_nb.col_map_nb(order_records["col"], c.target_shape[1])
    close = flex_select_col_nb(c.close, col)
    entry_trades = pf_records_nb.get_entry_trades_nb(
        order_records,
        close[: c.i + 1],
        col_map,
        init_position=init_position,
        init_price=init_price,
    )
    return entry_trades


@register_jitted
def get_entry_trade_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get entry trade records for the current column up to the current point.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

    Returns:
        Array1d: Entry trade records for the current column.
    """
    return get_col_entry_trade_records_nb(
        c, c.col, init_position=init_position, init_price=init_price
    )


@register_jitted
def get_col_exit_trade_records_nb(
    c: tp.NamedTuple,
    col: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get exit trade records for the specified column up to the current point.

    Args:
        c (NamedTuple): Context.
        col (int): Column index for which to retrieve exit trade records.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

    Returns:
        Array1d: Exit trade records for the specified column.
    """
    order_records = get_col_order_records_nb(c, col)
    col_map = records_nb.col_map_nb(order_records["col"], c.target_shape[1])
    close = flex_select_col_nb(c.close, col)
    exit_trades = pf_records_nb.get_exit_trades_nb(
        order_records,
        close[: c.i + 1],
        col_map,
        init_position=init_position,
        init_price=init_price,
    )
    return exit_trades


@register_jitted
def get_exit_trade_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get exit trade records for the current column up to the current point.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

    Returns:
        Array1d: Exit trade records for the current column.
    """
    return get_col_exit_trade_records_nb(
        c, c.col, init_position=init_position, init_price=init_price
    )


@register_jitted
def get_col_position_records_nb(
    c: tp.NamedTuple,
    col: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get position records for the specified column up to the current point.

    Args:
        c (NamedTuple): Context.
        col (int): Column index for which to retrieve position records.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

    Returns:
        Array1d: Position records for the specified column.
    """
    exit_trade_records = get_col_exit_trade_records_nb(
        c, col, init_position=init_position, init_price=init_price
    )
    col_map = records_nb.col_map_nb(exit_trade_records["col"], c.target_shape[1])
    position_records = pf_records_nb.get_positions_nb(exit_trade_records, col_map)
    return position_records


@register_jitted
def get_position_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get position records for the current column up to the current point.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

    Returns:
        Array1d: Position records for the current column.
    """
    return get_col_position_records_nb(c, c.col, init_position=init_position, init_price=init_price)


@register_jitted
def stop_group_sim_nb(c: tp.NamedTuple, group: int) -> None:
    """Stop simulation for a specific group.

    This function sets the simulation end index for the specified group to the next row.

    Args:
        c (NamedTuple): Context.
        group (int): Index of the group for which the simulation is to be stopped.

    Returns:
        None: This function modifies the context in place.
    """
    c.sim_end[group] = c.i + 1


@register_jitted
def stop_sim_nb(
    c: tp.Union[
        SegmentContext,
        OrderContext,
        PostOrderContext,
        FlexOrderContext,
        SignalSegmentContext,
        SignalContext,
        PostSignalContext,
    ],
) -> None:
    """Stop simulation for the current group.

    Args:
        c (Union[SegmentContext, OrderContext, PostOrderContext, FlexOrderContext, SignalSegmentContext, SignalContext, PostSignalContext]):
            Relevant context.

    Returns:
        None: This function modifies the context in place.
    """
    stop_group_sim_nb(c, c.group)


# ############# Ordering ############# #


@register_jitted
def get_exec_state_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    val_price: tp.Optional[int] = None,
) -> ExecState:
    """Compute the execution state from the simulation context.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.
        val_price (Optional[float]): Valuation price of the asset.

    Returns:
        ExecState: Updated execution state with attributes such as cash, position, debt,
            locked cash, free cash, value price, and overall value.
    """
    if val_price is not None:
        _val_price = float(val_price)
        value = float(
            update_value_nb(
                cash_before=get_cash_nb(c),
                cash_now=get_cash_nb(c),
                position_before=get_position_nb(c),
                position_now=get_position_nb(c),
                val_price_before=get_val_price_nb(c),
                val_price_now=_val_price,
                value_before=get_value_nb(c),
            )
        )
    else:
        _val_price = float(get_val_price_nb(c))
        value = float(get_value_nb(c))
    return ExecState(
        cash=get_cash_nb(c),
        position=get_position_nb(c),
        debt=get_debt_nb(c),
        locked_cash=get_locked_cash_nb(c),
        free_cash=get_free_cash_nb(c),
        val_price=_val_price,
        value=value,
    )


@register_jitted
def get_price_area_nb(c: tp.NamedTuple) -> PriceArea:
    """Retrieve price area values from the simulation context.

    Args:
        c (NamedTuple): Context.

    Returns:
        PriceArea: Selected price area with open, high, low, and
            close values for the current simulation step.
    """
    return PriceArea(
        open=select_nb(c, c.open, i=c.i),
        high=select_nb(c, c.high, i=c.i),
        low=select_nb(c, c.low, i=c.i),
        close=select_nb(c, c.close, i=c.i),
    )


@register_jitted
def get_order_size_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    size: float,
    size_type: int = SizeType.Amount,
    val_price: tp.Optional[int] = None,
) -> float:
    """Calculate the order size based on the simulation context and provided parameters.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.
        size (float): Order size.
        size_type (int): Type of order size.

            Percent sizes are not supported. See `vectorbtpro.portfolio.enums.SizeType`.
        val_price (Optional[float]): Valuation price of the asset.

    Returns:
        float: Computed order size.
    """
    exec_state = get_exec_state_nb(c, val_price=val_price)
    if size_type == SizeType.Percent100 or size_type == SizeType.Percent:
        raise ValueError("Size type Percent(100) is not supported")
    return resolve_size_nb(
        size=size,
        size_type=size_type,
        position=get_position_nb(c),
        val_price=exec_state.val_price,
        value=exec_state.value,
    )[0]


@register_jitted
def get_order_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    size: float,
    size_type: int = SizeType.Amount,
    direction: int = Direction.Both,
    val_price: tp.Optional[int] = None,
) -> float:
    """Calculate the approximate order value from the execution state and order parameters.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.
        size (float): Order size.
        size_type (int): Type of order size.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (int): Order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        val_price (Optional[float]): Valuation price of the asset.

    Returns:
        float: Approximated order value.
    """
    exec_state = get_exec_state_nb(c, val_price=val_price)
    return approx_order_value_nb(
        exec_state,
        size=size,
        size_type=size_type,
        direction=direction,
    )


@register_jitted
def get_order_result_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    order: Order,
    val_price: tp.Optional[float] = None,
    update_value: bool = False,
) -> tp.Tuple[OrderResult, ExecState]:
    """Obtain the order result and updated execution state without modifying the simulation state.

    Args:
        c (Union[OrderContext, PostOrderContext, SignalContext, PostSignalContext]):
            Relevant context.
        order (Order): Order to execute.

            See `vectorbtpro.portfolio.enums.Order`.
        val_price (Optional[float]): Valuation price of the asset.
        update_value (bool): Flag to update portfolio value with each order.

    Returns:
        Tuple[OrderResult, ExecState]: Tuple containing the order result and
            the updated execution state.
    """
    exec_state = get_exec_state_nb(c, val_price=val_price)
    price_area = get_price_area_nb(c)
    return execute_order_nb(
        exec_state=exec_state,
        order=order,
        price_area=price_area,
        update_value=update_value,
    )
