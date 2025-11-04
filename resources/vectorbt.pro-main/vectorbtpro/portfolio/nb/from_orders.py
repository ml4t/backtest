# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for portfolio simulation based on orders."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns.nb import get_return_nb
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        open=base_ch.flex_array_gl_slicer,
        high=base_ch.flex_array_gl_slicer,
        low=base_ch.flex_array_gl_slicer,
        close=base_ch.flex_array_gl_slicer,
        init_cash=base_ch.FlexArraySlicer(),
        init_position=base_ch.flex_1d_array_gl_slicer,
        init_price=base_ch.flex_1d_array_gl_slicer,
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        cash_earnings=base_ch.flex_array_gl_slicer,
        cash_dividends=base_ch.flex_array_gl_slicer,
        size=base_ch.flex_array_gl_slicer,
        price=base_ch.flex_array_gl_slicer,
        size_type=base_ch.flex_array_gl_slicer,
        direction=base_ch.flex_array_gl_slicer,
        fees=base_ch.flex_array_gl_slicer,
        fixed_fees=base_ch.flex_array_gl_slicer,
        slippage=base_ch.flex_array_gl_slicer,
        min_size=base_ch.flex_array_gl_slicer,
        max_size=base_ch.flex_array_gl_slicer,
        size_granularity=base_ch.flex_array_gl_slicer,
        leverage=base_ch.flex_array_gl_slicer,
        leverage_mode=base_ch.flex_array_gl_slicer,
        reject_prob=base_ch.flex_array_gl_slicer,
        price_area_vio_mode=base_ch.flex_array_gl_slicer,
        allow_partial=base_ch.flex_array_gl_slicer,
        raise_reject=base_ch.flex_array_gl_slicer,
        log=base_ch.flex_array_gl_slicer,
        val_price=base_ch.flex_array_gl_slicer,
        from_ago=base_ch.flex_array_gl_slicer,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        call_seq=base_ch.array_gl_slicer,
        auto_call_seq=None,
        ffill_val_price=None,
        update_value=None,
        save_state=None,
        save_value=None,
        save_returns=None,
        skip_empty=None,
        max_order_records=None,
        max_log_records=None,
    ),
    **portfolio_ch.merge_sim_outs_config,
)
@register_jitted(cache=True, tags={"can_parallel"})
def from_orders_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    open: tp.FlexArray2dLike = np.nan,
    high: tp.FlexArray2dLike = np.nan,
    low: tp.FlexArray2dLike = np.nan,
    close: tp.FlexArray2dLike = np.nan,
    init_cash: tp.FlexArray1dLike = 100.0,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    cash_dividends: tp.FlexArray2dLike = 0.0,
    size: tp.FlexArray2dLike = np.inf,
    price: tp.FlexArray2dLike = np.inf,
    size_type: tp.FlexArray2dLike = SizeType.Amount,
    direction: tp.FlexArray2dLike = Direction.Both,
    fees: tp.FlexArray2dLike = 0.0,
    fixed_fees: tp.FlexArray2dLike = 0.0,
    slippage: tp.FlexArray2dLike = 0.0,
    min_size: tp.FlexArray2dLike = np.nan,
    max_size: tp.FlexArray2dLike = np.nan,
    size_granularity: tp.FlexArray2dLike = np.nan,
    leverage: tp.FlexArray2dLike = 1.0,
    leverage_mode: tp.FlexArray2dLike = LeverageMode.Lazy,
    reject_prob: tp.FlexArray2dLike = 0.0,
    price_area_vio_mode: tp.FlexArray2dLike = PriceAreaVioMode.Ignore,
    allow_partial: tp.FlexArray2dLike = True,
    raise_reject: tp.FlexArray2dLike = False,
    log: tp.FlexArray2dLike = False,
    val_price: tp.FlexArray2dLike = np.inf,
    from_ago: tp.FlexArray2dLike = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    call_seq: tp.Optional[tp.Array2d] = None,
    auto_call_seq: bool = False,
    ffill_val_price: bool = True,
    update_value: bool = False,
    save_state: bool = False,
    save_value: bool = False,
    save_returns: bool = False,
    skip_empty: bool = True,
    max_order_records: tp.Optional[int] = None,
    max_log_records: tp.Optional[int] = 0,
) -> SimulationOutput:
    """Generate simulation orders for each element in a column-major order using flexible broadcasting.

    This function processes market data and portfolio inputs to create simulation orders and
    update simulation state arrays.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.

            !!! note
                Should be grouped only if cash sharing is enabled.
        open (FlexArray2dLike): Open price.

            Provided as a scalar, or per row, column, or element.
        high (FlexArray2dLike): High price.

            Provided as a scalar, or per row, column, or element.
        low (FlexArray2dLike): Low price.

            Provided as a scalar, or per row, column, or element.
        close (FlexArray2dLike): Close price.

            Provided as a scalar, or per row, column, or element.
        init_cash (FlexArray1dLike): Initial cash amount.

            Provided as a scalar or per group.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.
        cash_deposits (FlexArray2dLike): Cash deposits or withdrawals at the beginning of each bar.

            Provided as a scalar, or per row, group, or element.
        cash_earnings (FlexArray2dLike): Cash earnings or losses at the end of each bar.

            Provided as a scalar, or per row, column, or element.
        cash_dividends (FlexArray2dLike): Cash dividends or interest at the end of each bar.

            Provided as a scalar, or per row, column, or element.
        size (FlexArray2dLike): Order size.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.size`.
        price (FlexArray2dLike): Order price.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.price`.
        size_type (FlexArray2dLike): Order size type.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.size_type`.
        fees (FlexArray2dLike): Transaction fee rate.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.fees`.
        fixed_fees (FlexArray2dLike): Fixed fee amount.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.fixed_fees`.
        slippage (FlexArray2dLike): Slippage amount.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.slippage`.
        min_size (FlexArray2dLike): Minimum allowable order size.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.min_size`.
        max_size (FlexArray2dLike): Maximum allowable order size.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.max_size`.
        size_granularity (FlexArray2dLike): Granularity for order size.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.size_granularity`.
        leverage (FlexArray2dLike): Leverage factor.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.leverage`.
        leverage_mode (FlexArray2dLike): Mode for leverage calculation.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.leverage_mode`.
        reject_prob (FlexArray2dLike): Probability of order rejection.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.reject_prob`.
        price_area_vio_mode (FlexArray2dLike): Mode for handling price area violations.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.price_area_vio_mode`.
        allow_partial (FlexArray2dLike): Indicator whether partial orders are allowed.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.allow_partial`.

            Does not apply when the order size is `np.inf`.
        raise_reject (FlexArray2dLike): Indicator whether to raise errors on order rejection.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.raise_reject`.
        log (FlexArray2dLike): Logging flag.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.Order.log`.
        val_price (FlexArray2dLike): Valuation price.

            Provided as a scalar, or per row, column, or element.

            See `vectorbtpro.portfolio.enums.ValPriceType`.

            * Any `-np.inf` element is replaced by the latest valuation price
                (using `open` or a previously known value if `ffill_val_price` is True).
            * Any `np.inf` element is replaced by the current order price.
        from_ago (FlexArray2dLike): Lookback offset for price selection.

            Provided as a scalar, or per row, column, or element.

            Negative values are converted to positive to avoid look-ahead bias.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per group.
        call_seq (Optional[Array2d]): Sequence dictating the order in which columns are
            processed per row and group.
        auto_call_seq (bool): Flag to automatically sort the call sequence.
        ffill_val_price (bool): Flag to forward-fill valuation price.
        update_value (bool): Flag to update portfolio value with each order.
        save_state (bool): Flag to record the account state.

            See `vectorbtpro.portfolio.enums.AccountState`.
        save_value (bool): Flag to record the portfolio value.
        save_returns (bool): Flag to record the portfolio returns.
        skip_empty (bool): Flag indicating whether to skip processing when order data is empty.
        max_order_records (Optional[int]): Maximum number of order records expected per column.

            Defaults to the number of rows in the broadcasted shape. Set to 0 to disable,
            lower to reduce memory usage, or higher if multiple orders per timestamp are expected.
        max_log_records (Optional[int]): Maximum number of log records expected per column.

            Set to the number of rows in the broadcasted shape if logging is enabled. Set lower to
            reduce memory usage, or higher if multiple logs per timestamp are expected.

    Returns:
        SimulationOutput: Simulation output containing order and log records, cash deposits,
            earnings, and other aggregated metrics.

    !!! tip
        This function is parallelizable.

    Examples:
        Buy and hold using all cash and close price (default):

        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.records.nb import col_map_nb
        >>> from vectorbtpro.portfolio.nb import from_orders_nb, asset_flow_nb

        >>> close = np.array([1, 2, 3, 4, 5])[:, None]
        >>> sim_out = from_orders_nb(
        ...     target_shape=close.shape,
        ...     group_lens=np.array([1]),
        ...     call_seq=np.full(close.shape, 0),
        ...     close=close
        ... )
        >>> col_map = col_map_nb(sim_out.order_records['col'], close.shape[1])
        >>> asset_flow = asset_flow_nb(close.shape, sim_out.order_records, col_map)
        >>> asset_flow
        array([[100.],
               [  0.],
               [  0.],
               [  0.],
               [  0.]])
        ```
    """
    check_group_lens_nb(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)

    open_ = to_2d_array_nb(np.asarray(open))
    high_ = to_2d_array_nb(np.asarray(high))
    low_ = to_2d_array_nb(np.asarray(low))
    close_ = to_2d_array_nb(np.asarray(close))
    init_cash_ = to_1d_array_nb(np.asarray(init_cash))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))
    cash_dividends_ = to_2d_array_nb(np.asarray(cash_dividends))
    size_ = to_2d_array_nb(np.asarray(size))
    price_ = to_2d_array_nb(np.asarray(price))
    size_type_ = to_2d_array_nb(np.asarray(size_type))
    direction_ = to_2d_array_nb(np.asarray(direction))
    fees_ = to_2d_array_nb(np.asarray(fees))
    fixed_fees_ = to_2d_array_nb(np.asarray(fixed_fees))
    slippage_ = to_2d_array_nb(np.asarray(slippage))
    min_size_ = to_2d_array_nb(np.asarray(min_size))
    max_size_ = to_2d_array_nb(np.asarray(max_size))
    size_granularity_ = to_2d_array_nb(np.asarray(size_granularity))
    leverage_ = to_2d_array_nb(np.asarray(leverage))
    leverage_mode_ = to_2d_array_nb(np.asarray(leverage_mode))
    reject_prob_ = to_2d_array_nb(np.asarray(reject_prob))
    price_area_vio_mode_ = to_2d_array_nb(np.asarray(price_area_vio_mode))
    allow_partial_ = to_2d_array_nb(np.asarray(allow_partial))
    raise_reject_ = to_2d_array_nb(np.asarray(raise_reject))
    log_ = to_2d_array_nb(np.asarray(log))
    val_price_ = to_2d_array_nb(np.asarray(val_price))
    from_ago_ = to_2d_array_nb(np.asarray(from_ago))

    order_records, log_records = prepare_records_nb(
        target_shape=target_shape,
        max_order_records=max_order_records,
        max_log_records=max_log_records,
    )
    order_counts = np.full(target_shape[1], 0, dtype=int_)
    log_counts = np.full(target_shape[1], 0, dtype=int_)
    last_cash = prepare_last_cash_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        init_cash=init_cash_,
    )
    last_position = prepare_last_position_nb(
        target_shape=target_shape,
        init_position=init_position_,
    )
    last_value = prepare_last_value_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
    )
    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full(target_shape[1], 0.0, dtype=float_)
    last_locked_cash = np.full(target_shape[1], 0.0, dtype=float_)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    track_cash_deposits = (
        cash_deposits_.size == 1 and cash_deposits_[0, 0] != 0
    ) or cash_deposits_.size > 1
    if track_cash_deposits:
        cash_deposits_out = np.full((target_shape[0], len(group_lens)), 0.0, dtype=float_)
    else:
        cash_deposits_out = np.full((1, 1), 0.0, dtype=float_)
    track_cash_earnings = (
        cash_earnings_.size == 1 and cash_earnings_[0, 0] != 0
    ) or cash_earnings_.size > 1
    track_cash_dividends = (
        cash_dividends_.size == 1 and cash_dividends_[0, 0] != 0
    ) or cash_dividends_.size > 1
    track_cash_earnings = track_cash_earnings or track_cash_dividends
    if track_cash_earnings:
        cash_earnings_out = np.full(target_shape, 0.0, dtype=float_)
    else:
        cash_earnings_out = np.full((1, 1), 0.0, dtype=float_)

    if save_state:
        cash = np.full((target_shape[0], len(group_lens)), np.nan, dtype=float_)
        position = np.full(target_shape, np.nan, dtype=float_)
        debt = np.full(target_shape, np.nan, dtype=float_)
        locked_cash = np.full(target_shape, np.nan, dtype=float_)
        free_cash = np.full((target_shape[0], len(group_lens)), np.nan, dtype=float_)
    else:
        cash = np.full((0, 0), np.nan, dtype=float_)
        position = np.full((0, 0), np.nan, dtype=float_)
        debt = np.full((0, 0), np.nan, dtype=float_)
        locked_cash = np.full((0, 0), np.nan, dtype=float_)
        free_cash = np.full((0, 0), np.nan, dtype=float_)
    if save_value:
        value = np.full((target_shape[0], len(group_lens)), np.nan, dtype=float_)
    else:
        value = np.full((0, 0), np.nan, dtype=float_)
    if save_returns:
        returns = np.full((target_shape[0], len(group_lens)), np.nan, dtype=float_)
    else:
        returns = np.full((0, 0), np.nan, dtype=float_)
    in_outputs = FOInOutputs(
        cash=cash,
        position=position,
        debt=debt,
        locked_cash=locked_cash,
        free_cash=free_cash,
        value=value,
        returns=returns,
    )

    temp_call_seq = np.empty(target_shape[1], dtype=int_)
    temp_order_value = np.empty(target_shape[1], dtype=float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(target_shape[0], len(group_lens)),
        sim_start=sim_start,
        sim_end=sim_end,
    )

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col

        _sim_start = sim_start_[group]
        _sim_end = sim_end_[group]
        for i in range(_sim_start, _sim_end):
            # Add cash
            _cash_deposits = flex_select_nb(cash_deposits_, i, group)
            if _cash_deposits < 0:
                _cash_deposits = max(_cash_deposits, -last_cash[group])
            last_cash[group] += _cash_deposits
            last_free_cash[group] += _cash_deposits
            last_cash_deposits[group] = _cash_deposits
            if track_cash_deposits:
                cash_deposits_out[i, group] += _cash_deposits

            skip = skip_empty
            if skip:
                for ci in range(group_len):
                    col = from_col + ci
                    _i = i - abs(flex_select_nb(from_ago_, i, col))
                    if _i < 0:
                        continue
                    if flex_select_nb(log_, i, col):
                        skip = False
                        break
                    if not np.isnan(flex_select_nb(size_, _i, col)):
                        if not np.isnan(flex_select_nb(price_, _i, col)):
                            skip = False
                            break

            if not skip or ffill_val_price:
                for ci in range(group_len):
                    col = from_col + ci

                    # Update valuation price using current open
                    _open = flex_select_nb(open_, i, col)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                    # Resolve valuation price
                    _val_price = flex_select_nb(val_price_, i, col)
                    if np.isinf(_val_price):
                        if _val_price > 0:
                            _i = i - abs(flex_select_nb(from_ago_, i, col))
                            if _i < 0:
                                _price = np.nan
                            else:
                                _price = flex_select_nb(price_, _i, col)
                            if np.isinf(_price):
                                if _price > 0:
                                    _price = flex_select_nb(close_, i, col)
                                else:
                                    _price = _open
                            _val_price = _price
                        else:
                            _val_price = last_val_price[col]
                    if not np.isnan(_val_price) or not ffill_val_price:
                        last_val_price[col] = _val_price

            if not skip:
                # Update value and return
                group_value = last_cash[group]
                for col in range(from_col, to_col):
                    if last_position[col] != 0:
                        group_value += last_position[col] * last_val_price[col]
                last_value[group] = group_value
                last_return[group] = get_return_nb(
                    input_value=prev_close_value[group],
                    output_value=last_value[group] - _cash_deposits,
                )

                if cash_sharing:
                    # Dynamically sort by order value -> selling comes first to release funds early
                    if call_seq is None:
                        for ci in range(group_len):
                            temp_call_seq[ci] = ci
                        call_seq_now = temp_call_seq[:group_len]
                    else:
                        call_seq_now = call_seq[i, from_col:to_col]
                    if auto_call_seq:
                        # Same as sort_by_order_value_ctx_nb but with flexible indexing
                        for ci in range(group_len):
                            col = from_col + ci
                            exec_state = ExecState(
                                cash=last_cash[group] if cash_sharing else last_cash[col],
                                position=last_position[col],
                                debt=last_debt[col],
                                locked_cash=last_locked_cash[col],
                                free_cash=last_free_cash[group]
                                if cash_sharing
                                else last_free_cash[col],
                                val_price=last_val_price[col],
                                value=last_value[group] if cash_sharing else last_value[col],
                            )
                            _i = i - abs(flex_select_nb(from_ago_, i, col))
                            if _i < 0:
                                temp_order_value[ci] = 0.0
                            else:
                                temp_order_value[ci] = approx_order_value_nb(
                                    exec_state,
                                    flex_select_nb(size_, _i, col),
                                    flex_select_nb(size_type_, _i, col),
                                    flex_select_nb(direction_, _i, col),
                                )
                            if call_seq_now[ci] != ci:
                                raise ValueError("Call sequence must follow CallSeqType.Default")

                        # Sort by order value
                        insert_argsort_nb(temp_order_value[:group_len], call_seq_now)

                for k in range(group_len):
                    if cash_sharing:
                        ci = call_seq_now[k]
                        if ci >= group_len:
                            raise ValueError("Call index out of bounds of the group")
                    else:
                        ci = k
                    col = from_col + ci

                    # Get current values per column
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    locked_cash_now = last_locked_cash[col]
                    val_price_now = last_val_price[col]
                    cash_now = last_cash[group]
                    free_cash_now = last_free_cash[group]
                    value_now = last_value[group]
                    return_now = last_return[group]

                    # Generate the next order
                    _i = i - abs(flex_select_nb(from_ago_, i, col))
                    if _i < 0:
                        continue
                    order = order_nb(
                        size=flex_select_nb(size_, _i, col),
                        price=flex_select_nb(price_, _i, col),
                        size_type=flex_select_nb(size_type_, _i, col),
                        direction=flex_select_nb(direction_, _i, col),
                        fees=flex_select_nb(fees_, _i, col),
                        fixed_fees=flex_select_nb(fixed_fees_, _i, col),
                        slippage=flex_select_nb(slippage_, _i, col),
                        min_size=flex_select_nb(min_size_, _i, col),
                        max_size=flex_select_nb(max_size_, _i, col),
                        size_granularity=flex_select_nb(size_granularity_, _i, col),
                        leverage=flex_select_nb(leverage_, _i, col),
                        leverage_mode=flex_select_nb(leverage_mode_, _i, col),
                        reject_prob=flex_select_nb(reject_prob_, _i, col),
                        price_area_vio_mode=flex_select_nb(price_area_vio_mode_, _i, col),
                        allow_partial=flex_select_nb(allow_partial_, _i, col),
                        raise_reject=flex_select_nb(raise_reject_, _i, col),
                        log=flex_select_nb(log_, _i, col),
                    )

                    # Process the order
                    price_area = PriceArea(
                        open=flex_select_nb(open_, i, col),
                        high=flex_select_nb(high_, i, col),
                        low=flex_select_nb(low_, i, col),
                        close=flex_select_nb(close_, i, col),
                    )
                    exec_state = ExecState(
                        cash=cash_now,
                        position=position_now,
                        debt=debt_now,
                        locked_cash=locked_cash_now,
                        free_cash=free_cash_now,
                        val_price=val_price_now,
                        value=value_now,
                    )
                    order_result, new_exec_state = process_order_nb(
                        group=group,
                        col=col,
                        i=i,
                        exec_state=exec_state,
                        order=order,
                        price_area=price_area,
                        update_value=update_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                    )

                    # Update execution state
                    cash_now = new_exec_state.cash
                    position_now = new_exec_state.position
                    debt_now = new_exec_state.debt
                    locked_cash_now = new_exec_state.locked_cash
                    free_cash_now = new_exec_state.free_cash
                    val_price_now = new_exec_state.val_price
                    value_now = new_exec_state.value

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    last_locked_cash[col] = locked_cash_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    last_cash[group] = cash_now
                    last_free_cash[group] = free_cash_now
                    last_value[group] = value_now
                    last_return[group] = return_now

            for col in range(from_col, to_col):
                # Update valuation price using current close
                _close = flex_select_nb(close_, i, col)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

                _cash_earnings = flex_select_nb(cash_earnings_, i, col)
                _cash_dividends = flex_select_nb(cash_dividends_, i, col)
                _cash_earnings += _cash_dividends * last_position[col]
                last_cash[group] += _cash_earnings
                last_free_cash[group] += _cash_earnings
                if track_cash_earnings:
                    cash_earnings_out[i, col] += _cash_earnings
                if save_state:
                    position[i, col] = last_position[col]
                    debt[i, col] = last_debt[col]
                    locked_cash[i, col] = last_locked_cash[col]
                    cash[i, group] = last_cash[group]
                    free_cash[i, group] = last_free_cash[group]

            # Update value and return
            group_value = last_cash[group]
            for col in range(from_col, to_col):
                if last_position[col] != 0:
                    group_value += last_position[col] * last_val_price[col]
            last_value[group] = group_value
            last_return[group] = get_return_nb(
                input_value=prev_close_value[group],
                output_value=last_value[group] - _cash_deposits,
            )
            prev_close_value[group] = last_value[group]
            if save_value:
                in_outputs.value[i, group] = last_value[group]
            if save_returns:
                in_outputs.returns[i, group] = last_return[group]

    sim_start_out, sim_end_out = generic_nb.resolve_ungrouped_sim_range_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start_,
        sim_end=sim_end_,
        allow_none=True,
    )
    return prepare_sim_out_nb(
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        cash_deposits=cash_deposits_out,
        cash_earnings=cash_earnings_out,
        call_seq=call_seq,
        in_outputs=in_outputs,
        sim_start=sim_start_out,
        sim_end=sim_end_out,
    )
