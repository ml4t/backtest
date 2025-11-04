# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for portfolio simulation based on an order function."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.portfolio.nb.iter_ import *
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns import nb as returns_nb_
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb
from vectorbtpro.utils.template import RepFunc


@register_jitted(cache=True)
def calc_group_value_nb(
    from_col: int,
    to_col: int,
    cash_now: float,
    last_position: tp.Array1d,
    last_val_price: tp.Array1d,
) -> float:
    """Calculate the group value by summing available cash with the value of positions.

    Args:
        from_col (int): Starting column index of the group.
        to_col (int): Ending column index of the group.
        cash_now (float): Available cash at the current moment.
        last_position (Array1d): Array of last positions for each column.
        last_val_price (Array1d): Array of last valuation prices for each column.

    Returns:
        float: Total group value.
    """
    group_value = cash_now
    group_len = to_col - from_col
    for k in range(group_len):
        col = from_col + k
        if last_position[col] != 0:
            group_value += last_position[col] * last_val_price[col]
    return group_value


@register_jitted
def calc_ctx_group_value_nb(seg_ctx: SegmentContext) -> float:
    """Calculate the group value using the provided segment context.

    Best called from `pre_segment_func_nb`. Modify `last_val_price` in-place to update the valuation price.

    Args:
        seg_ctx (SegmentContext): Simulation segment context.

    Returns:
        float: Calculated group value.

    !!! note
        Cash sharing must be enabled.
    """
    if not seg_ctx.cash_sharing:
        raise ValueError("Cash sharing must be enabled")
    return calc_group_value_nb(
        seg_ctx.from_col,
        seg_ctx.to_col,
        seg_ctx.last_cash[seg_ctx.group],
        seg_ctx.last_position,
        seg_ctx.last_val_price,
    )


@register_jitted
def sort_call_seq_out_1d_nb(
    c: SegmentContext,
    size: tp.FlexArray1d,
    size_type: tp.FlexArray1d,
    direction: tp.FlexArray1d,
    order_value_out: tp.Array1d,
    call_seq_out: tp.Array1d,
) -> None:
    """Sort the call sequence array `call_seq_out` by computing order values for each potential order.

    Best called from `pre_segment_func_nb`.

    Args:
        c (SegmentContext): Segment context.
        size (FlexArray1d): 1D array of order sizes.
        size_type (FlexArray1d): 1D array of order size types.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (FlexArray1d): 1D array of order directions.

            See `vectorbtpro.portfolio.enums.Direction`.
        order_value_out (Array1d): Array to hold computed order values; must be empty initially.
        call_seq_out (Array1d): Array containing default call sequence indices, which will be sorted in place.

    Returns:
        None: Function modifies `call_seq_out` in place.

    !!! note
        Cash sharing must be enabled and `call_seq_out` must follow `CallSeqType.Default`.
    """
    if not c.cash_sharing:
        raise ValueError("Cash sharing must be enabled")

    group_value_now = calc_ctx_group_value_nb(c)
    group_len = c.to_col - c.from_col
    for ci in range(group_len):
        if call_seq_out[ci] != ci:
            raise ValueError("call_seq_out must follow CallSeqType.Default")
        col = c.from_col + ci
        _size = flex_select_1d_pc_nb(size, ci)
        _size_type = flex_select_1d_pc_nb(size_type, ci)
        _direction = flex_select_1d_pc_nb(direction, ci)
        if c.cash_sharing:
            cash_now = c.last_cash[c.group]
            free_cash_now = c.last_free_cash[c.group]
        else:
            cash_now = c.last_cash[col]
            free_cash_now = c.last_free_cash[col]
        exec_state = ExecState(
            cash=cash_now,
            position=c.last_position[col],
            debt=c.last_debt[col],
            locked_cash=c.last_locked_cash[col],
            free_cash=free_cash_now,
            val_price=c.last_val_price[col],
            value=group_value_now,
        )
        order_value_out[ci] = approx_order_value_nb(
            exec_state,
            _size,
            _size_type,
            _direction,
        )
    # Sort by order value
    insert_argsort_nb(order_value_out, call_seq_out)


@register_jitted
def sort_call_seq_1d_nb(
    c: SegmentContext,
    size: tp.FlexArray1d,
    size_type: tp.FlexArray1d,
    direction: tp.FlexArray1d,
    order_value_out: tp.Array1d,
) -> None:
    """Sort the call sequence associated with the segment context using 1D flexible arrays.

    Args:
        c (SegmentContext): Segment context.
        size (FlexArray1d): 1D array of order sizes.
        size_type (FlexArray1d): 1D array of order size types.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (FlexArray1d): 1D array of order directions.

            See `vectorbtpro.portfolio.enums.Direction`.
        order_value_out (Array1d): Array to hold computed order values; must be empty initially.

    Returns:
        None: Function modifies `call_seq_now` in place.

    See:
        `sort_call_seq_out_1d_nb`

    !!! note
        Can only be used in non-flexible simulation functions.
    """
    if c.call_seq_now is None:
        raise ValueError(
            "Call sequence array is None. Use sort_call_seq_out_1d_nb to sort a custom array."
        )
    sort_call_seq_out_1d_nb(c, size, size_type, direction, order_value_out, c.call_seq_now)


@register_jitted
def sort_call_seq_out_nb(
    c: SegmentContext,
    size: tp.FlexArray2d,
    size_type: tp.FlexArray2d,
    direction: tp.FlexArray2d,
    order_value_out: tp.Array1d,
    call_seq_out: tp.Array1d,
) -> None:
    """Sort the call sequence array `call_seq_out` by computing order values using 2D flexible arrays.

    Args:
        c (SegmentContext): Segment context.
        size (FlexArray2d): Array of order sizes.
        size_type (FlexArray2d): 2D array of order size types.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (FlexArray2d): Array indicating the order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        order_value_out (Array1d): Array to hold computed order values; must be empty initially.
        call_seq_out (Array1d): Array containing default call sequence indices, which will be sorted in place.

    Returns:
        None: Function modifies `call_seq_out` in place.

    !!! note
        Cash sharing must be enabled.
    """
    if not c.cash_sharing:
        raise ValueError("Cash sharing must be enabled")

    group_value_now = calc_ctx_group_value_nb(c)
    group_len = c.to_col - c.from_col
    for ci in range(group_len):
        if call_seq_out[ci] != ci:
            raise ValueError("call_seq_out must follow CallSeqType.Default")
        col = c.from_col + ci
        _size = select_from_col_nb(c, col, size)
        _size_type = select_from_col_nb(c, col, size_type)
        _direction = select_from_col_nb(c, col, direction)
        if c.cash_sharing:
            cash_now = c.last_cash[c.group]
            free_cash_now = c.last_free_cash[c.group]
        else:
            cash_now = c.last_cash[col]
            free_cash_now = c.last_free_cash[col]
        exec_state = ExecState(
            cash=cash_now,
            position=c.last_position[col],
            debt=c.last_debt[col],
            locked_cash=c.last_locked_cash[col],
            free_cash=free_cash_now,
            val_price=c.last_val_price[col],
            value=group_value_now,
        )
        order_value_out[ci] = approx_order_value_nb(
            exec_state,
            _size,
            _size_type,
            _direction,
        )
    # Sort by order value
    insert_argsort_nb(order_value_out, call_seq_out)


@register_jitted
def sort_call_seq_nb(
    c: SegmentContext,
    size: tp.FlexArray2d,
    size_type: tp.FlexArray2d,
    direction: tp.FlexArray2d,
    order_value_out: tp.Array1d,
) -> None:
    """Sort the call sequence associated with the segment context using 2D flexible arrays.

    Args:
        c (SegmentContext): Segment context.
        size (FlexArray2d): Array of order sizes.
        size_type (FlexArray2d): 2D array of order size types.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (FlexArray2d): Array indicating the order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        order_value_out (Array1d): Array to hold computed order values; must be empty initially.

    Returns:
        None: Function modifies `call_seq_now` in place.

    !!! note
        Can only be used in non-flexible simulation functions.
    """
    if c.call_seq_now is None:
        raise ValueError(
            "Call sequence array is None. Use sort_call_seq_out_1d_nb to sort a custom array."
        )
    sort_call_seq_out_nb(c, size, size_type, direction, order_value_out, c.call_seq_now)


@register_jitted
def try_order_nb(c: OrderContext, order: Order) -> tp.Tuple[OrderResult, ExecState]:
    """Execute an order without persistence.

    Args:
        c (OrderContext): Order context.
        order (Order): Order to execute.

            See `vectorbtpro.portfolio.enums.Order`.

    Returns:
        Tuple[OrderResult, ExecState]: Tuple containing the order execution result and
            the updated execution state.
    """
    exec_state = ExecState(
        cash=c.cash_now,
        position=c.position_now,
        debt=c.debt_now,
        locked_cash=c.locked_cash_now,
        free_cash=c.free_cash_now,
        val_price=c.val_price_now,
        value=c.value_now,
    )
    price_area = PriceArea(
        open=flex_select_nb(c.open, c.i, c.col),
        high=flex_select_nb(c.high, c.i, c.col),
        low=flex_select_nb(c.low, c.i, c.col),
        close=flex_select_nb(c.close, c.i, c.col),
    )
    return execute_order_nb(exec_state=exec_state, order=order, price_area=price_area)


@register_jitted
def no_pre_func_nb(c: tp.NamedTuple, *args) -> tp.Args:
    """Forward received arguments for preprocessing.

    Args:
        c (NamedTuple): Context.
        *args: Additional positional arguments.

    Returns:
        Args: Forwarded positional arguments.
    """
    return args


@register_jitted
def no_order_func_nb(c: OrderContext, *args) -> Order:
    """Return a placeholder order indicating no order is placed.

    Args:
        c (OrderContext): Order context.
        *args: Additional positional arguments.

    Returns:
        Order: Placeholder order, represented by `vectorbtpro.portfolio.enums.NoOrder`.
    """
    return NoOrder


@register_jitted
def no_post_func_nb(c: tp.NamedTuple, *args) -> None:
    """Perform placeholder postprocessing with no effect.

    Args:
        c (NamedTuple): Context.
        *args: Additional positional arguments.

    Returns:
        None
    """
    return None


# % <block pre_sim_func_nb>
# % <skip? skip_func(out_lines, "pre_sim_func_nb")>
# % <uncomment>
# @register_jitted
# def pre_sim_func_nb(
#     c: SimulationContext,
#     *args,
# ) -> tp.Args:
#     """Custom simulation pre-processing function.
#
#     Args:
#         c (SimulationContext): Simulation context.
#         *args: Additional positional arguments.
#
#     Returns:
#         Args: Forwarded positional arguments.
#     """
#     return args
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block post_sim_func_nb>
# % <skip? skip_func(out_lines, "post_sim_func_nb")>
# % <uncomment>
# @register_jitted
# def post_sim_func_nb(
#     c: SimulationContext,
#     *args,
# ) -> None:
#     """Custom simulation post-processing function.
#
#     Args:
#         c (SimulationContext): Simulation context.
#         *args: Additional positional arguments.
#
#     Returns:
#         None
#     """
#     return None
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block pre_group_func_nb>
# % <skip? skip_func(out_lines, "pre_group_func_nb")>
# % <uncomment>
# @register_jitted
# def pre_group_func_nb(
#     c: GroupContext,
#     *args,
# ) -> tp.Args:
#     """Custom group pre-processing function.
#
#     Args:
#         c (GroupContext): Group context.
#         *args: Additional positional arguments.
#
#     Returns:
#         Args: Forwarded positional arguments.
#     """
#     return args
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block post_group_func_nb>
# % <skip? skip_func(out_lines, "post_group_func_nb")>
# % <uncomment>
# @register_jitted
# def post_group_func_nb(
#     c: GroupContext,
#     *args,
# ) -> None:
#     """Custom group post-processing function.
#
#     Args:
#         c (GroupContext): Group context.
#         *args: Additional positional arguments.
#
#     Returns:
#         None
#     """
#     return None
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block pre_segment_func_nb>
# % <skip? skip_func(out_lines, "pre_segment_func_nb")>
# % <uncomment>
# @register_jitted
# def pre_segment_func_nb(
#     c: SegmentContext,
#     *args,
# ) -> tp.Args:
#     """Custom segment pre-processing function.
#
#     Args:
#         c (SegmentContext): Segment context.
#         *args: Additional positional arguments.
#
#     Returns:
#         Args: Forwarded positional arguments.
#     """
#     return args
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block post_segment_func_nb>
# % <skip? skip_func(out_lines, "post_segment_func_nb")>
# % <uncomment>
# @register_jitted
# def post_segment_func_nb(
#     c: SegmentContext,
#     *args,
# ) -> None:
#     """Custom segment post-processing function.
#
#     Args:
#         c (SegmentContext): Segment context.
#         *args: Additional positional arguments.
#
#     Returns:
#         None
#     """
#     return None
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block order_func_nb>
# % <skip? skip_func(out_lines, "order_func_nb")>
# % <uncomment>
# @register_jitted
# def order_func_nb(
#     c: OrderContext,
#     *args,
# ) -> Order:
#     """Custom order processing function.
#
#     Args:
#         c (OrderContext): Order context.
#         *args: Additional positional arguments.
#
#     Returns:
#         Order: Created order.
#
#           In this placeholder, it returns `vectorbtpro.portfolio.enums.NoOrder`.
#     """
#     return NoOrder
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block post_order_func_nb>
# % <skip? skip_func(out_lines, "post_order_func_nb")>
# % <uncomment>
# @register_jitted
# def post_order_func_nb(
#     c: PostOrderContext,
#     *args,
# ) -> None:
#     """Custom order post-processing function.
#
#     Args:
#         c (PostOrderContext): Post-order context.
#         *args: Additional positional arguments.
#
#     Returns:
#         None
#     """
#     return None
#
#
# % </uncomment>
# % </skip>
# % </block>


# % <section from_order_func_nb>
# % <uncomment>
# import vectorbtpro as vbt
# from vectorbtpro.portfolio.nb.from_order_func import *
# %? import_lines
#
#
# % </uncomment>
# %? blocks[pre_sim_func_nb_block]
# % blocks["pre_sim_func_nb"]
# %? blocks[post_sim_func_nb_block]
# % blocks["post_sim_func_nb"]
# %? blocks[pre_group_func_nb_block]
# % blocks["pre_group_func_nb"]
# %? blocks[post_group_func_nb_block]
# % blocks["post_group_func_nb"]
# %? blocks[pre_segment_func_nb_block]
# % blocks["pre_segment_func_nb"]
# %? blocks[post_segment_func_nb_block]
# % blocks["post_segment_func_nb"]
# %? blocks[order_func_nb_block]
# % blocks["order_func_nb"]
# %? blocks[post_order_func_nb_block]
# % blocks["post_order_func_nb"]
@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        call_seq=base_ch.array_gl_slicer,
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=base_ch.flex_1d_array_gl_slicer,
        init_price=base_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=base_ch.flex_array_gl_slicer,
        segment_mask=base_ch.FlexArraySlicer(axis=1),
        call_pre_segment=None,
        call_post_segment=None,
        pre_sim_func_nb=None,  # % None
        pre_sim_args=ch.ArgsTaker(),
        post_sim_func_nb=None,  # % None
        post_sim_args=ch.ArgsTaker(),
        pre_group_func_nb=None,  # % None
        pre_group_args=ch.ArgsTaker(),
        post_group_func_nb=None,  # % None
        post_group_args=ch.ArgsTaker(),
        pre_segment_func_nb=None,  # % None
        pre_segment_args=ch.ArgsTaker(),
        post_segment_func_nb=None,  # % None
        post_segment_args=ch.ArgsTaker(),
        order_func_nb=None,  # % None
        order_args=ch.ArgsTaker(),
        post_order_func_nb=None,  # % None
        post_order_args=ch.ArgsTaker(),
        index=None,
        freq=None,
        open=base_ch.flex_array_gl_slicer,
        high=base_ch.flex_array_gl_slicer,
        low=base_ch.flex_array_gl_slicer,
        close=base_ch.flex_array_gl_slicer,
        bm_close=base_ch.flex_array_gl_slicer,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        ffill_val_price=None,
        update_value=None,
        fill_pos_info=None,
        track_value=None,
        max_order_records=None,
        max_log_records=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config,
    setup_id=None,  # %? line.replace("None", task_id)
)
@register_jitted(
    tags={"can_parallel"},
    cache=False,  # % line.replace("False", "True")
    task_id_or_func=None,  # %? line.replace("None", task_id)
)
def from_order_func_nb(  # %? line.replace("from_order_func_nb", new_func_name)
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    call_seq: tp.Optional[tp.Array2d] = None,
    init_cash: tp.FlexArray1dLike = 100.0,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    segment_mask: tp.FlexArray2dLike = True,
    call_pre_segment: bool = False,
    call_post_segment: bool = False,
    pre_sim_func_nb: tp.PreSimFunc = no_pre_func_nb,  # % None
    pre_sim_args: tp.Args = (),
    post_sim_func_nb: tp.PostSimFunc = no_post_func_nb,  # % None
    post_sim_args: tp.Args = (),
    pre_group_func_nb: tp.PreGroupFunc = no_pre_func_nb,  # % None
    pre_group_args: tp.Args = (),
    post_group_func_nb: tp.PostGroupFunc = no_post_func_nb,  # % None
    post_group_args: tp.Args = (),
    pre_segment_func_nb: tp.PreSegmentFunc = no_pre_func_nb,  # % None
    pre_segment_args: tp.Args = (),
    post_segment_func_nb: tp.PostSegmentFunc = no_post_func_nb,  # % None
    post_segment_args: tp.Args = (),
    order_func_nb: tp.OrderFunc = no_order_func_nb,  # % None
    order_args: tp.Args = (),
    post_order_func_nb: tp.PostOrderFunc = no_post_func_nb,  # % None
    post_order_args: tp.Args = (),
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray2dLike = np.nan,
    high: tp.FlexArray2dLike = np.nan,
    low: tp.FlexArray2dLike = np.nan,
    close: tp.FlexArray2dLike = np.nan,
    bm_close: tp.FlexArray2dLike = np.nan,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_info: bool = True,
    track_value: bool = True,
    max_order_records: tp.Optional[int] = None,
    max_log_records: tp.Optional[int] = 0,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Fill order and log records by iterating over a target shape and executing a
    sequence of user-defined functions.

    Starting with an initial cash balance (`init_cash`), this function iterates over
    each group and each column in `target_shape`. For every data point, it generates an
    order using `order_func_nb` and attempts to process it. Upon a successful order execution,
    the simulation state (including cash, positions, and valuation) is updated. The simulation
    output is returned as a `SimulationOutput`.

    Unlike `from_order_func_rw_nb`, order processing is performed in column-major order
    (i.e. processing the entire column or group across all rows before moving to the next).
    See [Row- and column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).

    Args:
        target_shape (Shape): See `vectorbtpro.portfolio.enums.SimulationContext.target_shape`.
        group_lens (GroupLens): See `vectorbtpro.portfolio.enums.SimulationContext.group_lens`.
        cash_sharing (bool): See `vectorbtpro.portfolio.enums.SimulationContext.cash_sharing`.
        call_seq (Optional[Array2d]): See `vectorbtpro.portfolio.enums.SimulationContext.call_seq`.
        init_cash (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_cash`.

            Provided as a scalar or per column or group with cash sharing.
        init_position (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_position`.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_price`.

            Provided as a scalar or per column.
        cash_deposits (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.cash_deposits`.

            Provided as a scalar, or per row, column or group with cash sharing, or element.
        cash_earnings (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.cash_earnings`.

            Provided as a scalar, or per row, column, or element.
        segment_mask (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.segment_mask`.

            Provided as a scalar, or per row, group, or element.
        call_pre_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_pre_segment`.
        call_post_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_post_segment`.
        pre_sim_func_nb (PreSimFunc): Callback function to be called before the simulation.

            This function is used for creating global arrays and setting the seed.

            Accepts `vectorbtpro.portfolio.enums.SimulationContext` and `*pre_sim_args`,
            and returns a tuple that is passed to `pre_group_func_nb` and `post_group_func_nb`.
        pre_sim_args (Args): Positional arguments for `pre_sim_func_nb`.
        post_sim_func_nb (PostSimFunc): Callback function to be called after the simulation.

            Accepts `vectorbtpro.portfolio.enums.SimulationContext` and `*post_sim_args`,
            and returns nothing.
        post_sim_args (Args): Positional arguments for `post_sim_func_nb`.
        pre_group_func_nb (PreGroupFunc): Callback function to be called before processing a group.

            Accepts `vectorbtpro.portfolio.enums.GroupContext`, the unpacked output from
            `pre_sim_func_nb`, and `*pre_group_args`, and returns a tuple that is passed to
            `pre_segment_func_nb` and `post_segment_func_nb`.
        pre_group_args (Args): Positional arguments for `pre_group_func_nb`.
        post_group_func_nb (PostGroupFunc): Callback function to be called after processing a group.

            Accepts `vectorbtpro.portfolio.enums.GroupContext`, the unpacked output from
            `pre_sim_func_nb`, and `*post_group_args`, and returns nothing.
        post_group_args (Args): Positional arguments for `post_group_func_nb`.
        pre_segment_func_nb (PreSegmentFunc): Callback function to be called before processing a segment
            if `segment_mask` or `call_pre_segment` is True.

            Accepts `vectorbtpro.portfolio.enums.SegmentContext`, the unpacked output from
            `pre_group_func_nb`, and `*pre_segment_args`, and returns a tuple that is passed to
            `order_func_nb` and `post_order_func_nb`.

            This is the appropriate place to adjust the call sequence or set the valuation price.
            Group re-valuation and updates of open position stats occur immediately after this function
            executes, regardless of whether it is called.

            !!! note
                To change the call sequence of a segment, modify
                `vectorbtpro.portfolio.enums.SegmentContext.call_seq_now` in-place.
                Avoid creating new arrays to prevent performance degradation.
                Assigning a new context is not supported.

            !!! note
                You can override elements of `last_val_price` to influence group valuation.
                See `vectorbtpro.portfolio.enums.SimulationContext.last_val_price`.
        pre_segment_args (Args): Positional arguments for `pre_segment_func_nb`.
        post_segment_func_nb (PostSegmentFunc): Callback function to be called after processing a segment
            if `segment_mask` or `call_post_segment` is True.

            Handles the addition of `cash_earnings`, final group re-valuation, and the final update
            of open position stats.

            Accepts `vectorbtpro.portfolio.enums.SegmentContext`, the unpacked output from
            `pre_group_func_nb`, and `*post_segment_args`, and returns nothing.
        post_segment_args (Args): Positional arguments for `post_segment_func_nb`.
        order_func_nb (OrderFunc): Callback function to be called to generate an order.

            Used for generating or skipping an order.

            Accepts `vectorbtpro.portfolio.enums.OrderContext`, the unpacked output from
            `pre_segment_func_nb`, and `*order_args`, and returns `vectorbtpro.portfolio.enums.Order`.

            !!! note
                If the returned order is rejected, a new order cannot be issued. Ensure that the order
                passes (e.g., by using `try_order_nb`). For greater flexibility, use `from_flex_order_func_nb`.
        order_args (Args): Positional arguments for `order_func_nb`.
        post_order_func_nb (PostOrderFunc): Callback function to be called after processing an order.

            Accepts `vectorbtpro.portfolio.enums.PostOrderContext`, the unpacked output
            from `pre_segment_func_nb`, and `*post_order_args`, and returns nothing.
        post_order_args (Args): Positional arguments for `post_order_func_nb`.
        index (Optional[Array1d]): See `vectorbtpro.portfolio.enums.SimulationContext.index`.
        freq (Optional[int]): See `vectorbtpro.portfolio.enums.SimulationContext.freq`.
        open (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.open`.

            Provided as a scalar, or per row, column, or element.
        high (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.high`.

            Provided as a scalar, or per row, column, or element.
        low (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.low`.

            Provided as a scalar, or per row, column, or element.
        close (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.close`.

            Provided as a scalar, or per row, column, or element.
        bm_close (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.bm_close`.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): See `vectorbtpro.portfolio.enums.SimulationContext.sim_start`.

            Provided as a scalar or per group.
        sim_end (Optional[FlexArray1dLike]): See `vectorbtpro.portfolio.enums.SimulationContext.sim_end`.

            Provided as a scalar or per group.
        ffill_val_price (bool): See `vectorbtpro.portfolio.enums.SimulationContext.ffill_val_price`.
        update_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.update_value`.
        fill_pos_info (bool): See `vectorbtpro.portfolio.enums.SimulationContext.fill_pos_info`.
        track_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.track_value`.
        max_order_records (Optional[int]): Maximum number of order records expected per column.

            Defaults to the number of rows in the broadcasted shape. Set to 0 to disable,
            lower to reduce memory usage, or higher if multiple orders per timestamp are expected.
        max_log_records (Optional[int]): Maximum number of log records expected per column.

            Set to the number of rows in the broadcasted shape if logging is enabled. Set lower to
            reduce memory usage, or higher if multiple logs per timestamp are expected.
        in_outputs (Optional[NamedTuple]): See `vectorbtpro.portfolio.enums.SimulationContext.in_outputs`.

    Returns:
        SimulationOutput: Simulation output containing order records, log records, and
            other simulation results.

    !!! note
        Indexing of 2D arrays in vectorbtpro follows the Pandas convention: `a[i, col]`.

    !!! warning
        You can only safely access data from columns left of the current group and rows above the
        current row within the same group. Other data points may have not been processed yet.
        Accessing unprocessed data may not trigger any errors or warnings but yield arbitrary values
        (see [np.empty](https://numpy.org/doc/stable/reference/generated/numpy.empty.html)).

    !!! tip
        This function is parallelizable.

    Call hierarchy:
        Simulation is carried out by iterating over an imaginary frame with dimensions representing
        time (rows) and assets/features (columns). Each element of this frame is a potential order
        generated by an order function.

        There are two processing patterns:

        * Column-major (used by `from_order_func_nb`): Processes all rows in a column/group
            before moving to the next.
        * Row-major (used by `from_order_func_rw_nb`): Processes all columns in a row
            before moving to the next row.

        This choice affects the data available during order generation.

        The frame is subdivided into blocks (columns, groups, rows, segments, elements). Columns can be
        grouped into groups that may or may not share the same capital. Regardless of capital sharing,
        each collection of elements within a group and a bar is called a segment, which simply
        defines a single context (such as shared capital) for one or multiple orders. Each segment
        can also define a custom sequence (a so-called call sequence) in which orders are executed.

        Each block has its own context and pre/post-processing functions. Pre-processing functions
        return a tuple (possibly empty) that is passed down, while post-processing functions can be
        used to write custom arrays (e.g., returns).

        ```plaintext
        1. pre_sim_out = pre_sim_func_nb(SimulationContext, *pre_sim_args)
            2. pre_group_out = pre_group_func_nb(GroupContext, *pre_sim_out, *pre_group_args)
                3. if call_pre_segment or segment_mask:
                    pre_segment_out = pre_segment_func_nb(SegmentContext, *pre_group_out, *pre_segment_args)
                    4. if segment_mask:
                        order = order_func_nb(OrderContext, *pre_segment_out, *order_args)
                    5. if order exists:
                        post_order_func_nb(PostOrderContext, *pre_segment_out, *post_order_args)
                6. if call_post_segment or segment_mask:
                    post_segment_func_nb(SegmentContext, *pre_group_out, *post_segment_args)
            7. post_group_func_nb(GroupContext, *pre_sim_out, *post_group_args)
        8. post_sim_func_nb(SimulationContext, *post_sim_args)
        ```

        Let's demonstrate a frame with one group of two columns and one group of one column, and the
        following call sequence:

        ```plaintext
        array([[0, 1, 0],
               [1, 0, 0]])
        ```

        ![](/assets/images/api/from_order_func_nb.svg){: loading=lazy style="width:800px;" }

        And here is the context information available at each step:

        ![](/assets/images/api/context_info.svg){: loading=lazy style="width:700px;" }

    Examples:
        Example below demonstrates simulating a portfolio of three assets sharing $100,
        rebalanced every second tick, with all processing performed in Numba:

        ```pycon
        >>> from vectorbtpro import *

        >>> @njit
        ... def pre_sim_func_nb(c):
        ...     print('before simulation')
        ...     # Create a temporary array and pass it down the stack
        ...     order_value_out = np.empty(c.target_shape[1], dtype=float_)
        ...     return (order_value_out,)

        >>> @njit
        ... def pre_group_func_nb(c, order_value_out):
        ...     print('\\tbefore group', c.group)
        ...     # Forward down the stack (you can omit pre_group_func_nb entirely)
        ...     return (order_value_out,)

        >>> @njit
        ... def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
        ...     print('\\t\\tbefore segment', c.i)
        ...     for col in range(c.from_col, c.to_col):
        ...         # Here we use order price for group valuation
        ...         c.last_val_price[col] = vbt.pf_nb.select_from_col_nb(c, col, price)
        ...
        ...     # Reorder call sequence of this segment such that selling orders come first and buying last
        ...     # Rearranges c.call_seq_now based on order value (size, size_type, direction, and val_price)
        ...     # Utilizes flexible indexing using select_from_col_nb (as we did above)
        ...     vbt.pf_nb.sort_call_seq_nb(
        ...         c,
        ...         size,
        ...         size_type,
        ...         direction,
        ...         order_value_out[c.from_col:c.to_col]
        ...     )
        ...     # Forward nothing
        ...     return ()

        >>> @njit
        ... def order_func_nb(c, size, price, size_type, direction, fees, fixed_fees, slippage):
        ...     print('\\t\\t\\tcreating order', c.call_idx, 'at column', c.col)
        ...     # Create and returns an order
        ...     return vbt.pf_nb.order_nb(
        ...         size=vbt.pf_nb.select_nb(c, size),
        ...         price=vbt.pf_nb.select_nb(c, price),
        ...         size_type=vbt.pf_nb.select_nb(c, size_type),
        ...         direction=vbt.pf_nb.select_nb(c, direction),
        ...         fees=vbt.pf_nb.select_nb(c, fees),
        ...         fixed_fees=vbt.pf_nb.select_nb(c, fixed_fees),
        ...         slippage=vbt.pf_nb.select_nb(c, slippage)
        ...     )

        >>> @njit
        ... def post_order_func_nb(c):
        ...     print('\\t\\t\\t\\torder status:', c.order_result.status)
        ...     return None

        >>> @njit
        ... def post_segment_func_nb(c, order_value_out):
        ...     print('\\t\\tafter segment', c.i)
        ...     return None

        >>> @njit
        ... def post_group_func_nb(c, order_value_out):
        ...     print('\\tafter group', c.group)
        ...     return None

        >>> @njit
        ... def post_sim_func_nb(c):
        ...     print('after simulation')
        ...     return None

        >>> target_shape = (5, 3)
        >>> np.random.seed(42)
        >>> group_lens = np.array([3])  # one group of three columns
        >>> cash_sharing = True
        >>> segment_mask = np.array([True, False, True, False, True])[:, None]
        >>> price = close = np.random.uniform(1, 10, size=target_shape)
        >>> size = np.array([[1 / target_shape[1]]])  # custom flexible arrays must be 2-dim
        >>> size_type = np.array([[vbt.pf_enums.SizeType.TargetPercent]])
        >>> direction = np.array([[vbt.pf_enums.Direction.LongOnly]])
        >>> fees = np.array([[0.001]])
        >>> fixed_fees = np.array([[1.]])
        >>> slippage = np.array([[0.001]])

        >>> sim_out = vbt.pf_nb.from_order_func_nb(
        ...     target_shape,
        ...     group_lens,
        ...     cash_sharing,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_group_func_nb=pre_group_func_nb,
        ...     post_group_func_nb=post_group_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     order_func_nb=order_func_nb,
        ...     order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        before simulation
            before group 0
                before segment 0
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 1
                        order status: 0
                    creating order 2 at column 2
                        order status: 0
                after segment 0
                before segment 2
                    creating order 0 at column 1
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 0
                        order status: 0
                after segment 2
                before segment 4
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 1
                        order status: 0
                after segment 4
            after group 0
        after simulation

        >>> pd.DataFrame.from_records(sim_out.order_records)
           id  col  idx       size     price      fees  side
        0   0    0    0   7.626262  4.375232  1.033367     0
        1   1    0    2   5.210115  1.524275  1.007942     0
        2   2    0    4   7.899568  8.483492  1.067016     1
        3   0    1    0   3.488053  9.565985  1.033367     0
        4   1    1    2   0.920352  8.786790  1.008087     1
        5   2    1    4  10.713236  2.913963  1.031218     0
        6   0    2    0   3.972040  7.595533  1.030170     0
        7   1    2    2   0.448747  6.403625  1.002874     1
        8   2    2    4  12.378281  2.639061  1.032667     0

        >>> col_map = vbt.rec_nb.col_map_nb(sim_out.order_records['col'], target_shape[1])
        >>> asset_flow = vbt.pf_nb.asset_flow_nb(target_shape, sim_out.order_records, col_map)
        >>> assets = vbt.pf_nb.assets_nb(asset_flow)
        >>> asset_value = vbt.pf_nb.asset_value_nb(close, assets)
        >>> vbt.Scatter(data=asset_value).fig.show()
        ```

        ![](/assets/images/api/from_order_func_nb_example.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/from_order_func_nb_example.dark.svg#only-dark){: .iimg loading=lazy }

        Note that the last order in a group with cash sharing is always disadvantaged
        because it receives slightly fewer funds due to costs not accounted for during valuation.
    """
    check_group_lens_nb(group_lens, target_shape[1])

    init_cash_ = to_1d_array_nb(np.asarray(init_cash))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))
    segment_mask_ = to_2d_array_nb(np.asarray(segment_mask))
    open_ = to_2d_array_nb(np.asarray(open))
    high_ = to_2d_array_nb(np.asarray(high))
    low_ = to_2d_array_nb(np.asarray(low))
    close_ = to_2d_array_nb(np.asarray(close))
    bm_close_ = to_2d_array_nb(np.asarray(bm_close))

    order_records, log_records = prepare_records_nb(
        target_shape=target_shape,
        max_order_records=max_order_records,
        max_log_records=max_log_records,
    )
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
    last_pos_info = prepare_last_pos_info_nb(
        target_shape,
        init_position=init_position_,
        init_price=init_price_,
        fill_pos_info=fill_pos_info,
    )

    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full_like(last_position, 0.0)
    last_locked_cash = np.full_like(last_position, 0.0)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=int_)
    log_counts = np.full(target_shape[1], 0, dtype=int_)

    temp_call_seq = np.empty(target_shape[1], dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(target_shape[0], len(group_lens)),
        sim_start=sim_start,
        sim_end=sim_end,
    )

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        segment_mask=segment_mask_,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open_,
        high=high_,
        low=low_,
        close=close_,
        bm_close=bm_close_,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_info=fill_pos_info,
        track_value=track_value,
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_locked_cash=last_locked_cash,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        last_pos_info=last_pos_info,
        sim_start=sim_start_,
        sim_end=sim_end_,
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col

        # Call function before the group
        pre_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            init_cash=init_cash_,
            init_position=init_position_,
            init_price=init_price_,
            cash_deposits=cash_deposits_,
            cash_earnings=cash_earnings_,
            segment_mask=segment_mask_,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            bm_close=bm_close_,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_info=fill_pos_info,
            track_value=track_value,
            order_records=order_records,
            order_counts=order_counts,
            log_records=log_records,
            log_counts=log_counts,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_locked_cash=last_locked_cash,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            last_pos_info=last_pos_info,
            sim_start=sim_start_,
            sim_end=sim_end_,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col,
        )
        pre_group_out = pre_group_func_nb(pre_group_ctx, *pre_sim_out, *pre_group_args)

        _sim_start = sim_start_[group]
        _sim_end = sim_end_[group]
        for i in range(_sim_start, _sim_end):
            if call_seq is None:
                for ci in range(group_len):
                    temp_call_seq[ci] = ci
                call_seq_now = temp_call_seq[:group_len]
            else:
                call_seq_now = call_seq[i, from_col:to_col]

            if track_value:
                # Update valuation price using current open
                for col in range(from_col, to_col):
                    _open = flex_select_nb(open_, i, col)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group], last_value[group]
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col], last_value[col]
                        )

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            is_segment_active = flex_select_nb(segment_mask_, i, group)
            if call_pre_segment or is_segment_active:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    cash_deposits=cash_deposits_,
                    cash_earnings=cash_earnings_,
                    segment_mask=segment_mask_,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    bm_close=bm_close_,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_info=fill_pos_info,
                    track_value=track_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    sim_start=sim_start_,
                    sim_end=sim_end_,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now,
                )
                pre_segment_out = pre_segment_func_nb(
                    pre_seg_ctx, *pre_group_out, *pre_segment_args
                )

            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_nb(cash_deposits_, i, group)
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_nb(cash_deposits_, i, col)
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits

            if track_value:
                # Update value and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            if is_segment_active:
                for k in range(group_len):
                    if cash_sharing:
                        ci = call_seq_now[k]
                        if ci >= group_len:
                            raise ValueError("Call index out of bounds of the group")
                    else:
                        ci = k
                    col = from_col + ci

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    locked_cash_now = last_locked_cash[col]
                    val_price_now = last_val_price[col]
                    pos_info_now = last_pos_info[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                        cash_deposits_now = last_cash_deposits[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]
                        cash_deposits_now = last_cash_deposits[col]

                    # Generate the next order
                    order_ctx = OrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        init_cash=init_cash_,
                        init_position=init_position_,
                        init_price=init_price_,
                        cash_deposits=cash_deposits_,
                        cash_earnings=cash_earnings_,
                        segment_mask=segment_mask_,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open_,
                        high=high_,
                        low=low_,
                        close=close_,
                        bm_close=bm_close_,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_info=fill_pos_info,
                        track_value=track_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_locked_cash=last_locked_cash,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        last_pos_info=last_pos_info,
                        sim_start=sim_start_,
                        sim_end=sim_end_,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        locked_cash_now=locked_cash_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_info_now=pos_info_now,
                    )
                    order = order_func_nb(order_ctx, *pre_segment_out, *order_args)

                    if not track_value:
                        if (
                            order.size_type == SizeType.Value
                            or order.size_type == SizeType.TargetValue
                            or order.size_type == SizeType.TargetPercent
                        ):
                            raise ValueError(
                                "Cannot use size type that depends on not tracked value"
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

                    if track_value:
                        val_price_now = new_exec_state.val_price
                        value_now = new_exec_state.value
                        if cash_sharing:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[group],
                                value_now - cash_deposits_now,
                            )
                        else:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[col], value_now - cash_deposits_now
                            )

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    last_locked_cash[col] = locked_cash_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now

                    if track_value:
                        if not np.isnan(val_price_now) or not ffill_val_price:
                            last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                            last_return[group] = return_now
                        else:
                            last_value[col] = value_now
                            last_return[col] = return_now

                    # Update position record
                    if fill_pos_info:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_info_nb(
                                pos_info_now,
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    # Post-order function
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        init_cash=init_cash_,
                        init_position=init_position_,
                        init_price=init_price_,
                        cash_deposits=cash_deposits_,
                        cash_earnings=cash_earnings_,
                        segment_mask=segment_mask_,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open_,
                        high=high_,
                        low=low_,
                        close=close_,
                        bm_close=bm_close_,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_info=fill_pos_info,
                        track_value=track_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_locked_cash=last_locked_cash,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        last_pos_info=last_pos_info,
                        sim_start=sim_start_,
                        sim_end=sim_end_,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_before=exec_state.cash,
                        position_before=exec_state.position,
                        debt_before=exec_state.debt,
                        locked_cash_before=exec_state.locked_cash,
                        free_cash_before=exec_state.free_cash,
                        val_price_before=exec_state.val_price,
                        value_before=exec_state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        locked_cash_now=locked_cash_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_info_now=pos_info_now,
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats for future rows
            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_nb(cash_earnings_, i, col)
                if cash_sharing:
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings

            if track_value:
                # Update valuation price using current close
                for col in range(from_col, to_col):
                    _close = flex_select_nb(close_, i, col)
                    if not np.isnan(_close) or not ffill_val_price:
                        last_val_price[col] = _close

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                    prev_close_value[group] = last_value[group]
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )
                        prev_close_value[col] = last_value[col]

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            if call_post_segment or is_segment_active:
                # Call function after the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    cash_deposits=cash_deposits_,
                    cash_earnings=cash_earnings_,
                    segment_mask=segment_mask_,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    bm_close=bm_close_,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_info=fill_pos_info,
                    track_value=track_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    sim_start=sim_start_,
                    sim_end=sim_end_,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now,
                )
                post_segment_func_nb(post_seg_ctx, *pre_group_out, *post_segment_args)

            if i >= sim_end_[group] - 1:
                break

        # Call function after the group
        post_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            init_cash=init_cash_,
            init_position=init_position_,
            init_price=init_price_,
            cash_deposits=cash_deposits_,
            cash_earnings=cash_earnings_,
            segment_mask=segment_mask_,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            bm_close=bm_close_,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_info=fill_pos_info,
            track_value=track_value,
            order_records=order_records,
            order_counts=order_counts,
            log_records=log_records,
            log_counts=log_counts,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_locked_cash=last_locked_cash,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            last_pos_info=last_pos_info,
            sim_start=sim_start_,
            sim_end=sim_end_,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col,
        )
        post_group_func_nb(post_group_ctx, *pre_sim_out, *post_group_args)

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        segment_mask=segment_mask_,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open_,
        high=high_,
        low=low_,
        close=close_,
        bm_close=bm_close_,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_info=fill_pos_info,
        track_value=track_value,
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_locked_cash=last_locked_cash,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        last_pos_info=last_pos_info,
        sim_start=sim_start_,
        sim_end=sim_end_,
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

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
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        call_seq=call_seq,
        in_outputs=in_outputs,
        sim_start=sim_start_out,
        sim_end=sim_end_out,
    )


# % </section>


# % <block pre_row_func_nb>
# % <skip? skip_func(out_lines, "pre_row_func_nb")>
# % <uncomment>
# @register_jitted
# def pre_row_func_nb(
#     c: RowContext,
#     *args,
# ) -> tp.Args:
#     """Custom row pre-processing function.
#
#     Args:
#         c (RowContext): Row context.
#         *args: Additional positional arguments.
#
#     Returns:
#         Args: Forwarded positional arguments.
#     """
#     return args
#
#
# % </uncomment>
# % </skip>
# % </block>

# % <block post_row_func_nb>
# % <skip? skip_func(out_lines, "post_row_func_nb")>
# % <uncomment>
# @register_jitted
# def post_row_func_nb(
#     c: RowContext,
#     *args,
# ) -> None:
#     """Custom row post-processing function.
#
#     Args:
#         c (RowContext): Row context.
#         *args: Additional positional arguments.
#
#     Returns:
#         None
#     """
#     return None
#
#
# % </uncomment>
# % </skip>
# % </block>


# % <section from_order_func_rw_nb>
# % <uncomment>
# import vectorbtpro as vbt
# from vectorbtpro.portfolio.nb.from_order_func import *
# %? import_lines
#
#
# % </uncomment>
# %? blocks[pre_sim_func_nb_block]
# % blocks["pre_sim_func_nb"]
# %? blocks[post_sim_func_nb_block]
# % blocks["post_sim_func_nb"]
# %? blocks[pre_row_func_nb_block]
# % blocks["pre_row_func_nb"]
# %? blocks[post_row_func_nb_block]
# % blocks["post_row_func_nb"]
# %? blocks[pre_segment_func_nb_block]
# % blocks["pre_segment_func_nb"]
# %? blocks[post_segment_func_nb_block]
# % blocks["post_segment_func_nb"]
# %? blocks[order_func_nb_block]
# % blocks["order_func_nb"]
# %? blocks[post_order_func_nb_block]
# % blocks["post_order_func_nb"]
@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        call_seq=base_ch.array_gl_slicer,
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=base_ch.flex_1d_array_gl_slicer,
        init_price=base_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=base_ch.flex_array_gl_slicer,
        segment_mask=base_ch.FlexArraySlicer(axis=1),
        call_pre_segment=None,
        call_post_segment=None,
        pre_sim_func_nb=None,  # % None
        pre_sim_args=ch.ArgsTaker(),
        post_sim_func_nb=None,  # % None
        post_sim_args=ch.ArgsTaker(),
        pre_row_func_nb=None,  # % None
        pre_row_args=ch.ArgsTaker(),
        post_row_func_nb=None,  # % None
        post_row_args=ch.ArgsTaker(),
        pre_segment_func_nb=None,  # % None
        pre_segment_args=ch.ArgsTaker(),
        post_segment_func_nb=None,  # % None
        post_segment_args=ch.ArgsTaker(),
        order_func_nb=None,  # % None
        order_args=ch.ArgsTaker(),
        post_order_func_nb=None,  # % None
        post_order_args=ch.ArgsTaker(),
        index=None,
        freq=None,
        open=base_ch.flex_array_gl_slicer,
        high=base_ch.flex_array_gl_slicer,
        low=base_ch.flex_array_gl_slicer,
        close=base_ch.flex_array_gl_slicer,
        bm_close=base_ch.flex_array_gl_slicer,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        ffill_val_price=None,
        update_value=None,
        fill_pos_info=None,
        track_value=None,
        max_order_records=None,
        max_log_records=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config,
    setup_id=None,  # %? line.replace("None", task_id)
)
@register_jitted(
    tags={"can_parallel"},
    cache=False,  # % line.replace("False", "True")
    task_id_or_func=None,  # %? line.replace("None", task_id)
)
def from_order_func_rw_nb(  # %? line.replace("from_order_func_rw_nb", new_func_name)
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    call_seq: tp.Optional[tp.Array2d] = None,
    init_cash: tp.FlexArray1dLike = 100.0,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    segment_mask: tp.FlexArray2dLike = True,
    call_pre_segment: bool = False,
    call_post_segment: bool = False,
    pre_sim_func_nb: tp.PreSimFunc = no_pre_func_nb,  # % None
    pre_sim_args: tp.Args = (),
    post_sim_func_nb: tp.PostSimFunc = no_post_func_nb,  # % None
    post_sim_args: tp.Args = (),
    pre_row_func_nb: tp.PreRowFunc = no_pre_func_nb,  # % None
    pre_row_args: tp.Args = (),
    post_row_func_nb: tp.PostRowFunc = no_post_func_nb,  # % None
    post_row_args: tp.Args = (),
    pre_segment_func_nb: tp.PreSegmentFunc = no_pre_func_nb,  # % None
    pre_segment_args: tp.Args = (),
    post_segment_func_nb: tp.PostSegmentFunc = no_post_func_nb,  # % None
    post_segment_args: tp.Args = (),
    order_func_nb: tp.OrderFunc = no_order_func_nb,  # % None
    order_args: tp.Args = (),
    post_order_func_nb: tp.PostOrderFunc = no_post_func_nb,  # % None
    post_order_args: tp.Args = (),
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray2dLike = np.nan,
    high: tp.FlexArray2dLike = np.nan,
    low: tp.FlexArray2dLike = np.nan,
    close: tp.FlexArray2dLike = np.nan,
    bm_close: tp.FlexArray2dLike = np.nan,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_info: bool = True,
    track_value: bool = True,
    max_order_records: tp.Optional[int] = None,
    max_log_records: tp.Optional[int] = 0,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Same as `from_order_func_nb`, but iterates in row-major order.

    Row-major order processes each row (i.e., all groups/columns) entirely before moving to the next row.

    The primary difference from `from_order_func_nb` is that it exposes `pre_row_func_nb` instead
    of `pre_group_func_nb`. The `pre_row_func_nb` function is executed for each entire row and must
    accept a `vectorbtpro.portfolio.enums.RowContext`.

    Args:
        target_shape (Shape): See `vectorbtpro.portfolio.enums.SimulationContext.target_shape`.
        group_lens (GroupLens): See `vectorbtpro.portfolio.enums.SimulationContext.group_lens`.
        cash_sharing (bool): See `vectorbtpro.portfolio.enums.SimulationContext.cash_sharing`.
        call_seq (Optional[Array2d]): See `vectorbtpro.portfolio.enums.SimulationContext.call_seq`.
        init_cash (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_cash`.

            Provided as a scalar or per column or group with cash sharing.
        init_position (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_position`.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_price`.

            Provided as a scalar or per column.
        cash_deposits (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.cash_deposits`.

            Provided as a scalar, or per row, column or group with cash sharing, or element.
        cash_earnings (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.cash_earnings`.

            Provided as a scalar, or per row, column, or element.
        segment_mask (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.segment_mask`.

            Provided as a scalar, or per row, group, or element.
        call_pre_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_pre_segment`.
        call_post_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_post_segment`.
        pre_sim_func_nb (PreSimFunc): Callback function to be called before the simulation.

            This function is used for creating global arrays and setting the seed.

            Accepts `vectorbtpro.portfolio.enums.SimulationContext` and `*pre_sim_args`,
            and returns a tuple that is passed to `pre_row_func_nb` and `post_row_func_nb`.
        pre_sim_args (Args): Positional arguments for `pre_sim_func_nb`.
        post_sim_func_nb (PostSimFunc): Callback function to be called after the simulation.

            Accepts `vectorbtpro.portfolio.enums.SimulationContext` and `*post_sim_args`,
            and returns nothing.
        post_sim_args (Args): Positional arguments for `post_sim_func_nb`.
        pre_row_func_nb (PreRowFunc): Callback function to be called before processing a row.

            Accepts `vectorbtpro.portfolio.enums.RowContext`, the unpacked output from
            `pre_sim_func_nb`, and `*pre_row_args`, and returns a tuple that is passed to
            `pre_segment_func_nb` and `post_segment_func_nb`.
        pre_row_args (Args): Positional arguments for `pre_row_func_nb`.
        post_row_func_nb (PostRowFunc): Callback function to be called after processing a row.

            Accepts `vectorbtpro.portfolio.enums.RowContext`, the unpacked output from
            `pre_sim_func_nb`, and `*post_row_args`, and returns nothing.
        post_row_args (Args): Positional arguments for `post_row_func_nb`.
        pre_segment_func_nb (PreSegmentFunc): Callback function to be called before processing a segment
            if `segment_mask` or `call_pre_segment` is True.

            Accepts `vectorbtpro.portfolio.enums.SegmentContext`, the unpacked output from
            `pre_row_func_nb`, and `*pre_segment_args`, and returns a tuple that is passed to
            `order_func_nb` and `post_order_func_nb`.

            This is the appropriate place to adjust the call sequence or set the valuation price.
            Group re-valuation and updates of open position stats occur immediately after this function
            executes, regardless of whether it is called.

            !!! note
                To change the call sequence of a segment, modify
                `vectorbtpro.portfolio.enums.SegmentContext.call_seq_now` in-place.
                Avoid creating new arrays to prevent performance degradation.
                Assigning a new context is not supported.

            !!! note
                You can override elements of `last_val_price` to influence group valuation.
                See `vectorbtpro.portfolio.enums.SimulationContext.last_val_price`.
        pre_segment_args (Args): Positional arguments for `pre_segment_func_nb`.
        post_segment_func_nb (PostSegmentFunc): Callback function to be called after processing a segment
            if `segment_mask` or `call_post_segment` is True.

            Handles the addition of `cash_earnings`, final group re-valuation, and the final update
            of open position stats.

            Accepts `vectorbtpro.portfolio.enums.SegmentContext`, the unpacked output from
            `pre_row_func_nb`, and `*post_segment_args`, and returns nothing.
        post_segment_args (Args): Positional arguments for `post_segment_func_nb`.
        order_func_nb (OrderFunc): Callback function to be called to generate an order.

            Used for generating or skipping an order.

            Accepts `vectorbtpro.portfolio.enums.OrderContext`, the unpacked output from
            `pre_segment_func_nb`, and `*order_args`, and returns `vectorbtpro.portfolio.enums.Order`.

            !!! note
                If the returned order is rejected, a new order cannot be issued. Ensure that the order
                passes (e.g., by using `try_order_nb`). For greater flexibility, use `from_flex_order_func_nb`.
        order_args (Args): Positional arguments for `order_func_nb`.
        post_order_func_nb (PostOrderFunc): Callback function to be called after processing an order.

            Accepts `vectorbtpro.portfolio.enums.PostOrderContext`, the unpacked output
            from `pre_segment_func_nb`, and `*post_order_args`, and returns nothing.
        post_order_args (Args): Positional arguments for `post_order_func_nb`.
        index (Optional[Array1d]): See `vectorbtpro.portfolio.enums.SimulationContext.index`.
        freq (Optional[int]): See `vectorbtpro.portfolio.enums.SimulationContext.freq`.
        open (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.open`.

            Provided as a scalar, or per row, column, or element.
        high (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.high`.

            Provided as a scalar, or per row, column, or element.
        low (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.low`.

            Provided as a scalar, or per row, column, or element.
        close (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.close`.

            Provided as a scalar, or per row, column, or element.
        bm_close (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.bm_close`.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): See `vectorbtpro.portfolio.enums.SimulationContext.sim_start`.

            Provided as a scalar or per group.
        sim_end (Optional[FlexArray1dLike]): See `vectorbtpro.portfolio.enums.SimulationContext.sim_end`.

            Provided as a scalar or per group.
        ffill_val_price (bool): See `vectorbtpro.portfolio.enums.SimulationContext.ffill_val_price`.
        update_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.update_value`.
        fill_pos_info (bool): See `vectorbtpro.portfolio.enums.SimulationContext.fill_pos_info`.
        track_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.track_value`.
        max_order_records (Optional[int]): Maximum number of order records expected per column.

            Defaults to the number of rows in the broadcasted shape. Set to 0 to disable,
            lower to reduce memory usage, or higher if multiple orders per timestamp are expected.
        max_log_records (Optional[int]): Maximum number of log records expected per column.

            Set to the number of rows in the broadcasted shape if logging is enabled. Set lower to
            reduce memory usage, or higher if multiple logs per timestamp are expected.
        in_outputs (Optional[NamedTuple]): See `vectorbtpro.portfolio.enums.SimulationContext.in_outputs`.

    Returns:
        SimulationOutput: Simulation output containing order records, log records, and
            other simulation results.

    !!! note
        The `pre_row_func_nb` function is only invoked if there is at least one active segment in the row.
        Similarly, `pre_segment_func_nb` and `order_func_nb` are only called if their corresponding segment
        is active. To ensure `pre_row_func_nb` is invoked when its primary task is to manage segment
        activation, all segments should be activated by default.

    !!! warning
        You can only safely access data points that are to the left of the current group and
        rows that are above the current row.

    !!! tip
        This function is parallelizable.

    Call hierarchy:
        ```plaintext
        1. pre_sim_out = pre_sim_func_nb(SimulationContext, *pre_sim_args)
            2. pre_row_out = pre_row_func_nb(RowContext, *pre_sim_out, *pre_row_args)
                3. if call_pre_segment or segment_mask:
                    pre_segment_out = pre_segment_func_nb(SegmentContext, *pre_row_out, *pre_segment_args)
                    4. if segment_mask:
                        order = order_func_nb(OrderContext, *pre_segment_out, *order_args)
                    5. if order exists:
                        post_order_func_nb(PostOrderContext, *pre_segment_out, *post_order_args)
                6. if call_post_segment or segment_mask:
                    post_segment_func_nb(SegmentContext, *pre_row_out, *post_segment_args)
            7. post_row_func_nb(RowContext, *pre_sim_out, *post_row_args)
        8. post_sim_func_nb(SimulationContext, *post_sim_args)
        ```

        Let's illustrate the same example as in `from_order_func_nb` but adapted for this function:

        ![](/assets/images/api/from_order_func_rw_nb.svg){: loading=lazy style="width:800px;" }

    Examples:
        Running the same example as in `from_order_func_nb` but adapted for this function:

        ```pycon
        >>> @njit
        ... def pre_row_func_nb(c, order_value_out):
        ...     print('\\tbefore row', c.i)
        ...     # Forward down the stack
        ...     return (order_value_out,)

        >>> @njit
        ... def post_row_func_nb(c, order_value_out):
        ...     print('\\tafter row', c.i)
        ...     return None

        >>> sim_out = vbt.pf_nb.from_order_func_rw_nb(
        ...     target_shape,
        ...     group_lens,
        ...     cash_sharing,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_row_func_nb=pre_row_func_nb,
        ...     post_row_func_nb=post_row_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     order_func_nb=order_func_nb,
        ...     order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        before simulation
            before row 0
                before segment 0
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 1
                        order status: 0
                    creating order 2 at column 2
                        order status: 0
                after segment 0
            after row 0
            before row 1
            after row 1
            before row 2
                before segment 2
                    creating order 0 at column 1
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 0
                        order status: 0
                after segment 2
            after row 2
            before row 3
            after row 3
            before row 4
                before segment 4
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 1
                        order status: 0
                after segment 4
            after row 4
        after simulation
        ```
    """
    check_group_lens_nb(group_lens, target_shape[1])

    init_cash_ = to_1d_array_nb(np.asarray(init_cash))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))
    segment_mask_ = to_2d_array_nb(np.asarray(segment_mask))
    open_ = to_2d_array_nb(np.asarray(open))
    high_ = to_2d_array_nb(np.asarray(high))
    low_ = to_2d_array_nb(np.asarray(low))
    close_ = to_2d_array_nb(np.asarray(close))
    bm_close_ = to_2d_array_nb(np.asarray(bm_close))

    order_records, log_records = prepare_records_nb(
        target_shape=target_shape,
        max_order_records=max_order_records,
        max_log_records=max_log_records,
    )
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
    last_pos_info = prepare_last_pos_info_nb(
        target_shape,
        init_position=init_position_,
        init_price=init_price_,
        fill_pos_info=fill_pos_info,
    )

    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full_like(last_position, 0.0)
    last_locked_cash = np.full_like(last_position, 0.0)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=int_)
    log_counts = np.full(target_shape[1], 0, dtype=int_)

    temp_call_seq = np.empty(target_shape[1], dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(target_shape[0], len(group_lens)),
        sim_start=sim_start,
        sim_end=sim_end,
    )

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        segment_mask=segment_mask_,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open_,
        high=high_,
        low=low_,
        close=close_,
        bm_close=bm_close_,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_info=fill_pos_info,
        track_value=track_value,
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_locked_cash=last_locked_cash,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        last_pos_info=last_pos_info,
        sim_start=sim_start_,
        sim_end=sim_end_,
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    _sim_start = sim_start_.min()
    _sim_end = sim_end_.max()
    for i in range(_sim_start, _sim_end):
        # Call function before the row
        pre_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            init_cash=init_cash_,
            init_position=init_position_,
            init_price=init_price_,
            cash_deposits=cash_deposits_,
            cash_earnings=cash_earnings_,
            segment_mask=segment_mask_,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            bm_close=bm_close_,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_info=fill_pos_info,
            track_value=track_value,
            order_records=order_records,
            order_counts=order_counts,
            log_records=log_records,
            log_counts=log_counts,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_locked_cash=last_locked_cash,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            last_pos_info=last_pos_info,
            sim_start=sim_start_,
            sim_end=sim_end_,
            i=i,
        )
        pre_row_out = pre_row_func_nb(pre_row_ctx, *pre_sim_out, *pre_row_args)

        for group in range(len(group_lens)):
            if i < sim_start_[group] or i >= sim_end_[group]:
                continue

            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]
            group_len = to_col - from_col

            if call_seq is None:
                for ci in range(group_len):
                    temp_call_seq[ci] = ci
                call_seq_now = temp_call_seq[:group_len]
            else:
                call_seq_now = call_seq[i, from_col:to_col]

            if track_value:
                # Update valuation price using current open
                for col in range(from_col, to_col):
                    _open = flex_select_nb(open_, i, col)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group], last_value[group]
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col], last_value[col]
                        )

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            is_segment_active = flex_select_nb(segment_mask_, i, group)
            if call_pre_segment or is_segment_active:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    cash_deposits=cash_deposits_,
                    cash_earnings=cash_earnings_,
                    segment_mask=segment_mask_,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    bm_close=bm_close_,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_info=fill_pos_info,
                    track_value=track_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    sim_start=sim_start_,
                    sim_end=sim_end_,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now,
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_row_out, *pre_segment_args)

            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_nb(cash_deposits_, i, group)
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_nb(cash_deposits_, i, col)
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits

            if track_value:
                # Update value and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            if is_segment_active:
                for k in range(group_len):
                    if cash_sharing:
                        ci = call_seq_now[k]
                        if ci >= group_len:
                            raise ValueError("Call index out of bounds of the group")
                    else:
                        ci = k
                    col = from_col + ci

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    locked_cash_now = last_locked_cash[col]
                    val_price_now = last_val_price[col]
                    pos_info_now = last_pos_info[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                        cash_deposits_now = last_cash_deposits[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]
                        cash_deposits_now = last_cash_deposits[col]

                    # Generate the next order
                    order_ctx = OrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        init_cash=init_cash_,
                        init_position=init_position_,
                        init_price=init_price_,
                        cash_deposits=cash_deposits_,
                        cash_earnings=cash_earnings_,
                        segment_mask=segment_mask_,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open_,
                        high=high_,
                        low=low_,
                        close=close_,
                        bm_close=bm_close_,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_info=fill_pos_info,
                        track_value=track_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_locked_cash=last_locked_cash,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        last_pos_info=last_pos_info,
                        sim_start=sim_start_,
                        sim_end=sim_end_,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        locked_cash_now=locked_cash_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_info_now=pos_info_now,
                    )
                    order = order_func_nb(order_ctx, *pre_segment_out, *order_args)

                    if not track_value:
                        if (
                            order.size_type == SizeType.Value
                            or order.size_type == SizeType.TargetValue
                            or order.size_type == SizeType.TargetPercent
                        ):
                            raise ValueError(
                                "Cannot use size type that depends on not tracked value"
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

                    if track_value:
                        val_price_now = new_exec_state.val_price
                        value_now = new_exec_state.value
                        if cash_sharing:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[group],
                                value_now - cash_deposits_now,
                            )
                        else:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[col], value_now - cash_deposits_now
                            )

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    last_locked_cash[col] = locked_cash_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now

                    if track_value:
                        if not np.isnan(val_price_now) or not ffill_val_price:
                            last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                            last_return[group] = return_now
                        else:
                            last_value[col] = value_now
                            last_return[col] = return_now

                    # Update position record
                    if fill_pos_info:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_info_nb(
                                pos_info_now,
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    # Post-order function
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        init_cash=init_cash_,
                        init_position=init_position_,
                        init_price=init_price_,
                        cash_deposits=cash_deposits_,
                        cash_earnings=cash_earnings_,
                        segment_mask=segment_mask_,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open_,
                        high=high_,
                        low=low_,
                        close=close_,
                        bm_close=bm_close_,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_info=fill_pos_info,
                        track_value=track_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_locked_cash=last_locked_cash,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        last_pos_info=last_pos_info,
                        sim_start=sim_start_,
                        sim_end=sim_end_,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_before=exec_state.cash,
                        position_before=exec_state.position,
                        debt_before=exec_state.debt,
                        locked_cash_before=exec_state.locked_cash,
                        free_cash_before=exec_state.free_cash,
                        val_price_before=exec_state.val_price,
                        value_before=exec_state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        locked_cash_now=locked_cash_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_info_now=pos_info_now,
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats for future rows
            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_nb(cash_earnings_, i, col)
                if cash_sharing:
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings

            if track_value:
                # Update valuation price using current close
                for col in range(from_col, to_col):
                    _close = flex_select_nb(close_, i, col)
                    if not np.isnan(_close) or not ffill_val_price:
                        last_val_price[col] = _close

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                    prev_close_value[group] = last_value[group]
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )
                        prev_close_value[col] = last_value[col]

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            if call_post_segment or is_segment_active:
                # Call function after the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    cash_deposits=cash_deposits_,
                    cash_earnings=cash_earnings_,
                    segment_mask=segment_mask_,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    bm_close=bm_close_,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_info=fill_pos_info,
                    track_value=track_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    sim_start=sim_start_,
                    sim_end=sim_end_,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now,
                )
                post_segment_func_nb(post_seg_ctx, *pre_row_out, *post_segment_args)

        # Call function after the row
        post_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            init_cash=init_cash_,
            init_position=init_position_,
            init_price=init_price_,
            cash_deposits=cash_deposits_,
            cash_earnings=cash_earnings_,
            segment_mask=segment_mask_,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            bm_close=bm_close_,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_info=fill_pos_info,
            track_value=track_value,
            order_records=order_records,
            order_counts=order_counts,
            log_records=log_records,
            log_counts=log_counts,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_locked_cash=last_locked_cash,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            last_pos_info=last_pos_info,
            sim_start=sim_start_,
            sim_end=sim_end_,
            i=i,
        )
        post_row_func_nb(post_row_ctx, *pre_sim_out, *post_row_args)

        sim_end_reached = True
        for group in range(len(group_lens)):
            if i < sim_end_[group] - 1:
                sim_end_reached = False
                break
        if sim_end_reached:
            break

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        segment_mask=segment_mask_,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open_,
        high=high_,
        low=low_,
        close=close_,
        bm_close=bm_close_,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_info=fill_pos_info,
        track_value=track_value,
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_locked_cash=last_locked_cash,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        last_pos_info=last_pos_info,
        sim_start=sim_start_,
        sim_end=sim_end_,
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

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
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        call_seq=call_seq,
        in_outputs=in_outputs,
        sim_start=sim_start_out,
        sim_end=sim_end_out,
    )


# % </section>


@register_jitted
def no_flex_order_func_nb(c: FlexOrderContext, *args) -> tp.Tuple[int, Order]:
    """Placeholder flexible order processing function.

    This function acts as a dummy flexible order function by always returning a break indicator (-1) and
    `vectorbtpro.portfolio.enums.NoOrder`.

    Args:
        c (FlexOrderContext): Flexible order context.
        *args: Additional positional arguments.

    Returns:
        Tuple[int, Order]: Tuple containing the break column indicator (-1) and
            `vectorbtpro.portfolio.enums.NoOrder`.
    """
    return -1, NoOrder


# % <block flex_order_func_nb>
# % <skip? skip_func(out_lines, "flex_order_func_nb")>
# % <uncomment>
# @register_jitted
# def flex_order_func_nb(
#     c: FlexOrderContext,
#     *args,
# ) -> tp.Tuple[int, Order]:
#     """Custom flexible order processing function.
#
#     Implements custom flexible order processing by returning a break indicator (-1) and
#   `vectorbtpro.portfolio.enums.NoOrder`.
#
#     Args:
#         c (FlexOrderContext): Flexible order context.
#         *args: Additional positional arguments.
#
#     Returns:
#         Tuple[int, Order]: Tuple with the break column indicator (-1) and
#               `vectorbtpro.portfolio.enums.NoOrder`.
#     """
#     return -1, NoOrder
#
#
# % </uncomment>
# % </skip>
# % </block>


# % <section from_flex_order_func_nb>
# % <uncomment>
# import vectorbtpro as vbt
# from vectorbtpro.portfolio.nb.from_order_func import *
# %? import_lines
#
#
# % </uncomment>
# %? blocks[pre_sim_func_nb_block]
# % blocks["pre_sim_func_nb"]
# %? blocks[post_sim_func_nb_block]
# % blocks["post_sim_func_nb"]
# %? blocks[pre_group_func_nb_block]
# % blocks["pre_group_func_nb"]
# %? blocks[post_group_func_nb_block]
# % blocks["post_group_func_nb"]
# %? blocks[pre_segment_func_nb_block]
# % blocks["pre_segment_func_nb"]
# %? blocks[post_segment_func_nb_block]
# % blocks["post_segment_func_nb"]
# %? blocks[flex_order_func_nb_block]
# % blocks["flex_order_func_nb"]
# %? blocks[post_order_func_nb_block]
# % blocks["post_order_func_nb"]
@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=base_ch.flex_1d_array_gl_slicer,
        init_price=base_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=base_ch.flex_array_gl_slicer,
        segment_mask=base_ch.FlexArraySlicer(axis=1),
        call_pre_segment=None,
        call_post_segment=None,
        pre_sim_func_nb=None,  # % None
        pre_sim_args=ch.ArgsTaker(),
        post_sim_func_nb=None,  # % None
        post_sim_args=ch.ArgsTaker(),
        pre_group_func_nb=None,  # % None
        pre_group_args=ch.ArgsTaker(),
        post_group_func_nb=None,  # % None
        post_group_args=ch.ArgsTaker(),
        pre_segment_func_nb=None,  # % None
        pre_segment_args=ch.ArgsTaker(),
        post_segment_func_nb=None,  # % None
        post_segment_args=ch.ArgsTaker(),
        flex_order_func_nb=None,  # % None
        flex_order_args=ch.ArgsTaker(),
        post_order_func_nb=None,  # % None
        post_order_args=ch.ArgsTaker(),
        index=None,
        freq=None,
        open=base_ch.flex_array_gl_slicer,
        high=base_ch.flex_array_gl_slicer,
        low=base_ch.flex_array_gl_slicer,
        close=base_ch.flex_array_gl_slicer,
        bm_close=base_ch.flex_array_gl_slicer,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        ffill_val_price=None,
        update_value=None,
        fill_pos_info=None,
        track_value=None,
        max_order_records=None,
        max_log_records=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config,
    setup_id=None,  # %? line.replace("None", task_id)
)
@register_jitted(
    tags={"can_parallel"},
    cache=False,  # % line.replace("False", "True")
    task_id_or_func=None,  # %? line.replace("None", task_id)
)
def from_flex_order_func_nb(  # %? line.replace("from_flex_order_func_nb", new_func_name)
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    init_cash: tp.FlexArray1dLike = 100.0,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    segment_mask: tp.FlexArray2dLike = True,
    call_pre_segment: bool = False,
    call_post_segment: bool = False,
    pre_sim_func_nb: tp.PreSimFunc = no_pre_func_nb,  # % None
    pre_sim_args: tp.Args = (),
    post_sim_func_nb: tp.PostSimFunc = no_post_func_nb,  # % None
    post_sim_args: tp.Args = (),
    pre_group_func_nb: tp.PreGroupFunc = no_pre_func_nb,  # % None
    pre_group_args: tp.Args = (),
    post_group_func_nb: tp.PostGroupFunc = no_post_func_nb,  # % None
    post_group_args: tp.Args = (),
    pre_segment_func_nb: tp.PreSegmentFunc = no_pre_func_nb,  # % None
    pre_segment_args: tp.Args = (),
    post_segment_func_nb: tp.PostSegmentFunc = no_post_func_nb,  # % None
    post_segment_args: tp.Args = (),
    flex_order_func_nb: tp.FlexOrderFunc = no_flex_order_func_nb,  # % None
    flex_order_args: tp.Args = (),
    post_order_func_nb: tp.PostOrderFunc = no_post_func_nb,  # % None
    post_order_args: tp.Args = (),
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray2dLike = np.nan,
    high: tp.FlexArray2dLike = np.nan,
    low: tp.FlexArray2dLike = np.nan,
    close: tp.FlexArray2dLike = np.nan,
    bm_close: tp.FlexArray2dLike = np.nan,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_info: bool = True,
    track_value: bool = True,
    max_order_records: tp.Optional[int] = None,
    max_log_records: tp.Optional[int] = 0,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Same as `from_order_func_nb`, but with no predefined call sequence.

    In contrast to `from_order_func_nb`, the `post_order_func_nb` in this function is a segment-level
    order function that returns both a column index and an order, and is repeatedly called until a break
    condition is met. This design enables multiple orders to be issued within a single element in
    an arbitrary order.

    The order function must accept a `vectorbtpro.portfolio.enums.FlexOrderContext`, an unpacked
    tuple output from `pre_segment_func_nb`, and additional positional arguments from `flex_order_args`.
    It should return a tuple of (column, order), where returning a column index of -1 signals to
    exit the order loop.

    Args:
        target_shape (Shape): See `vectorbtpro.portfolio.enums.SimulationContext.target_shape`.
        group_lens (GroupLens): See `vectorbtpro.portfolio.enums.SimulationContext.group_lens`.
        cash_sharing (bool): See `vectorbtpro.portfolio.enums.SimulationContext.cash_sharing`.
        call_seq (Optional[Array2d]): See `vectorbtpro.portfolio.enums.SimulationContext.call_seq`.
        init_cash (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_cash`.

            Provided as a scalar or per column or group with cash sharing.
        init_position (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_position`.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_price`.

            Provided as a scalar or per column.
        cash_deposits (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.cash_deposits`.

            Provided as a scalar, or per row, column or group with cash sharing, or element.
        cash_earnings (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.cash_earnings`.

            Provided as a scalar, or per row, column, or element.
        segment_mask (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.segment_mask`.

            Provided as a scalar, or per row, group, or element.
        call_pre_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_pre_segment`.
        call_post_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_post_segment`.
        pre_sim_func_nb (PreSimFunc): Callback function to be called before the simulation.

            This function is used for creating global arrays and setting the seed.

            Accepts `vectorbtpro.portfolio.enums.SimulationContext` and `*pre_sim_args`,
            and returns a tuple that is passed to `pre_group_func_nb` and `post_group_func_nb`.
        pre_sim_args (Args): Positional arguments for `pre_sim_func_nb`.
        post_sim_func_nb (PostSimFunc): Callback function to be called after the simulation.

            Accepts `vectorbtpro.portfolio.enums.SimulationContext` and `*post_sim_args`,
            and returns nothing.
        post_sim_args (Args): Positional arguments for `post_sim_func_nb`.
        pre_group_func_nb (PreGroupFunc): Callback function to be called before processing a group.

            Accepts `vectorbtpro.portfolio.enums.GroupContext`, the unpacked output from
            `pre_sim_func_nb`, and `*pre_group_args`, and returns a tuple that is passed to
            `pre_segment_func_nb` and `post_segment_func_nb`.
        pre_group_args (Args): Positional arguments for `pre_group_func_nb`.
        post_group_func_nb (PostGroupFunc): Callback function to be called after processing a group.

            Accepts `vectorbtpro.portfolio.enums.GroupContext`, the unpacked output from
            `pre_sim_func_nb`, and `*post_group_args`, returning nothing.
        post_group_args (Args): Positional arguments for `post_group_func_nb`.
        pre_segment_func_nb (PreSegmentFunc): Callback function to be called before processing a segment
            if `segment_mask` or `call_pre_segment` is True.

            Accepts `vectorbtpro.portfolio.enums.SegmentContext`, the unpacked output from
            `pre_group_func_nb`, and `*pre_segment_args`, and returns a tuple that is passed to
            `flex_order_func_nb` and `post_order_func_nb`.

            This is the appropriate place to adjust the call sequence or set the valuation price.
            Group re-valuation and updates of open position stats occur immediately after this function
            executes, regardless of whether it is called.

            !!! note
                To change the call sequence of a segment, modify
                `vectorbtpro.portfolio.enums.SegmentContext.call_seq_now` in-place.
                Avoid creating new arrays to prevent performance degradation.
                Assigning a new context is not supported.

            !!! note
                You can override elements of `last_val_price` to influence group valuation.
                See `vectorbtpro.portfolio.enums.SimulationContext.last_val_price`.
        pre_segment_args (Args): Positional arguments for `pre_segment_func_nb`.
        post_segment_func_nb (PostSegmentFunc): Callback function to be called after processing a segment
            if `segment_mask` or `call_post_segment` is True.

            Handles the addition of `cash_earnings`, final group re-valuation, and the final update
            of open position stats.

            Accepts `vectorbtpro.portfolio.enums.SegmentContext`, the unpacked output from
            `pre_group_func_nb`, and `*post_segment_args`, and returns nothing.
        post_segment_args (Args): Positional arguments for `post_segment_func_nb`.
        flex_order_func_nb (FlexOrderFunc): Callback function to be called to generate a flexible order.

            Used for generating an order in a column.

            Accepts `vectorbtpro.portfolio.enums.FlexOrderContext`, the unpacked output from
            `pre_segment_func_nb`, and `*order_args`, and returns a tuple of
            (column, `vectorbtpro.portfolio.enums.Order`).
        flex_order_args (Args): Positional arguments for `flex_order_func_nb`.
        post_order_func_nb (PostOrderFunc): Callback function to be called after processing an order.

            Accepts `vectorbtpro.portfolio.enums.PostOrderContext`, the unpacked output
            from `pre_segment_func_nb`, and `*post_order_args`, and returns nothing.
        post_order_args (Args): Positional arguments for `post_order_func_nb`.
        index (Optional[Array1d]): See `vectorbtpro.portfolio.enums.SimulationContext.index`.
        freq (Optional[int]): See `vectorbtpro.portfolio.enums.SimulationContext.freq`.
        open (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.open`.

            Provided as a scalar, or per row, column, or element.
        high (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.high`.

            Provided as a scalar, or per row, column, or element.
        low (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.low`.

            Provided as a scalar, or per row, column, or element.
        close (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.close`.

            Provided as a scalar, or per row, column, or element.
        bm_close (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.bm_close`.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): See `vectorbtpro.portfolio.enums.SimulationContext.sim_start`.

            Provided as a scalar or per group.
        sim_end (Optional[FlexArray1dLike]): See `vectorbtpro.portfolio.enums.SimulationContext.sim_end`.

            Provided as a scalar or per group.
        ffill_val_price (bool): See `vectorbtpro.portfolio.enums.SimulationContext.ffill_val_price`.
        update_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.update_value`.
        fill_pos_info (bool): See `vectorbtpro.portfolio.enums.SimulationContext.fill_pos_info`.
        track_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.track_value`.
        max_order_records (Optional[int]): Maximum number of order records expected per column.

            Defaults to the number of rows in the broadcasted shape. Set to 0 to disable,
            lower to reduce memory usage, or higher if multiple orders per timestamp are expected.
        max_log_records (Optional[int]): Maximum number of log records expected per column.

            Set to the number of rows in the broadcasted shape if logging is enabled. Set lower to
            reduce memory usage, or higher if multiple logs per timestamp are expected.
        in_outputs (Optional[NamedTuple]): See `vectorbtpro.portfolio.enums.SimulationContext.in_outputs`.

    Returns:
        SimulationOutput: Simulation output containing order records, log records, and
            other simulation results.

    !!! note
        Since multiple orders can be generated per element, an "order_records index out of range"
        exception may occur. In such cases, increase `max_order_records` manually to avoid
        performance degradation.

    !!! tip
        This function is parallelizable.

    Call hierarchy:
        ```plaintext
        1. pre_sim_out = pre_sim_func_nb(SimulationContext, *pre_sim_args)
            2. pre_group_out = pre_group_func_nb(GroupContext, *pre_sim_out, *pre_group_args)
                3. if call_pre_segment or segment_mask:
                    pre_segment_out = pre_segment_func_nb(SegmentContext, *pre_group_out, *pre_segment_args)
                    while col != -1:
                        4. if segment_mask:
                            col, order = flex_order_func_nb(FlexOrderContext, *pre_segment_out, *flex_order_args)
                        5. if order exists:
                            post_order_func_nb(PostOrderContext, *pre_segment_out, *post_order_args)
                6. if call_post_segment or segment_mask:
                    post_segment_func_nb(SegmentContext, *pre_group_out, *post_segment_args)
            7. post_group_func_nb(GroupContext, *pre_sim_out, *post_group_args)
        8. post_sim_func_nb(SimulationContext, *post_sim_args)
        ```

        Let's illustrate a similar example as in `from_order_func_nb`, but adapted for this function:

        ![](/assets/images/api/from_flex_order_func_nb.svg){: loading=lazy style="width:800px;" }

    Examples:
        Same example as in `from_order_func_nb`:

        ```pycon
        >>> from vectorbtpro import *

        >>> @njit
        ... def pre_sim_func_nb(c):
        ...     # Create temporary arrays and pass them down the stack
        ...     print('before simulation')
        ...     order_value_out = np.empty(c.target_shape[1], dtype=float_)
        ...     call_seq_out = np.empty(c.target_shape[1], dtype=int_)
        ...     return (order_value_out, call_seq_out)

        >>> @njit
        ... def pre_group_func_nb(c, order_value_out, call_seq_out):
        ...     print('\\tbefore group', c.group)
        ...     return (order_value_out, call_seq_out)

        >>> @njit
        ... def pre_segment_func_nb(c, order_value_out, call_seq_out, size, price, size_type, direction):
        ...     print('\\t\\tbefore segment', c.i)
        ...     for col in range(c.from_col, c.to_col):
        ...         # Here we use order price for group valuation
        ...         c.last_val_price[col] = vbt.pf_nb.select_from_col_nb(c, col, price)
        ...
        ...     # Same as for from_order_func_nb, but since we don't have a predefined c.call_seq_now anymore,
        ...     # we need to store our new call sequence somewhere else
        ...     call_seq_out[:] = np.arange(c.group_len)
        ...     vbt.pf_nb.sort_call_seq_out_nb(
        ...         c,
        ...         size,
        ...         size_type,
        ...         direction,
        ...         order_value_out[c.from_col:c.to_col],
        ...         call_seq_out[c.from_col:c.to_col]
        ...     )
        ...
        ...     # Forward the sorted call sequence
        ...     return (call_seq_out,)

        >>> @njit
        ... def flex_order_func_nb(c, call_seq_out, size, price, size_type, direction, fees, fixed_fees, slippage):
        ...     if c.call_idx < c.group_len:
        ...         col = c.from_col + call_seq_out[c.call_idx]
        ...         print('\\t\\t\\tcreating order', c.call_idx, 'at column', col)
        ...         # # Create and returns an order
        ...         return col, vbt.pf_nb.order_nb(
        ...             size=vbt.pf_nb.select_from_col_nb(c, col, size),
        ...             price=vbt.pf_nb.select_from_col_nb(c, col, price),
        ...             size_type=vbt.pf_nb.select_from_col_nb(c, col, size_type),
        ...             direction=vbt.pf_nb.select_from_col_nb(c, col, direction),
        ...             fees=vbt.pf_nb.select_from_col_nb(c, col, fees),
        ...             fixed_fees=vbt.pf_nb.select_from_col_nb(c, col, fixed_fees),
        ...             slippage=vbt.pf_nb.select_from_col_nb(c, col, slippage)
        ...         )
        ...     # All columns already processed -> break the loop
        ...     print('\\t\\t\\tbreaking out of the loop')
        ...     return -1, vbt.pf_nb.order_nothing_nb()

        >>> @njit
        ... def post_order_func_nb(c, call_seq_out):
        ...     print('\\t\\t\\t\\torder status:', c.order_result.status)
        ...     return None

        >>> @njit
        ... def post_segment_func_nb(c, order_value_out, call_seq_out):
        ...     print('\\t\\tafter segment', c.i)
        ...     return None

        >>> @njit
        ... def post_group_func_nb(c, order_value_out, call_seq_out):
        ...     print('\\tafter group', c.group)
        ...     return None

        >>> @njit
        ... def post_sim_func_nb(c):
        ...     print('after simulation')
        ...     return None

        >>> target_shape = (5, 3)
        >>> np.random.seed(42)
        >>> group_lens = np.array([3])  # one group of three columns
        >>> cash_sharing = True
        >>> segment_mask = np.array([True, False, True, False, True])[:, None]
        >>> price = close = np.random.uniform(1, 10, size=target_shape)
        >>> size = np.array([[1 / target_shape[1]]])  # custom flexible arrays must be 2-dim
        >>> size_type = np.array([[vbt.pf_enums.SizeType.TargetPercent]])
        >>> direction = np.array([[vbt.pf_enums.Direction.LongOnly]])
        >>> fees = np.array([[0.001]])
        >>> fixed_fees = np.array([[1.]])
        >>> slippage = np.array([[0.001]])

        >>> sim_out = vbt.pf_nb.from_flex_order_func_nb(
        ...     target_shape,
        ...     group_lens,
        ...     cash_sharing,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_group_func_nb=pre_group_func_nb,
        ...     post_group_func_nb=post_group_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     flex_order_func_nb=flex_order_func_nb,
        ...     flex_order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        before simulation
            before group 0
                before segment 0
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 1
                        order status: 0
                    creating order 2 at column 2
                        order status: 0
                    breaking out of the loop
                after segment 0
                before segment 2
                    creating order 0 at column 1
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 0
                        order status: 0
                    breaking out of the loop
                after segment 2
                before segment 4
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 1
                        order status: 0
                    breaking out of the loop
                after segment 4
            after group 0
        after simulation
        ```
    """
    check_group_lens_nb(group_lens, target_shape[1])

    init_cash_ = to_1d_array_nb(np.asarray(init_cash))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))
    segment_mask_ = to_2d_array_nb(np.asarray(segment_mask))
    open_ = to_2d_array_nb(np.asarray(open))
    high_ = to_2d_array_nb(np.asarray(high))
    low_ = to_2d_array_nb(np.asarray(low))
    close_ = to_2d_array_nb(np.asarray(close))
    bm_close_ = to_2d_array_nb(np.asarray(bm_close))

    order_records, log_records = prepare_records_nb(
        target_shape=target_shape,
        max_order_records=max_order_records,
        max_log_records=max_log_records,
    )
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
    last_pos_info = prepare_last_pos_info_nb(
        target_shape,
        init_position=init_position_,
        init_price=init_price_,
        fill_pos_info=fill_pos_info,
    )

    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full_like(last_position, 0.0)
    last_locked_cash = np.full_like(last_position, 0.0)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=int_)
    log_counts = np.full(target_shape[1], 0, dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(target_shape[0], len(group_lens)),
        sim_start=sim_start,
        sim_end=sim_end,
    )

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=None,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        segment_mask=segment_mask_,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open_,
        high=high_,
        low=low_,
        close=close_,
        bm_close=bm_close_,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_info=fill_pos_info,
        track_value=track_value,
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_locked_cash=last_locked_cash,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        last_pos_info=last_pos_info,
        sim_start=sim_start_,
        sim_end=sim_end_,
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col

        # Call function before the group
        pre_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=None,
            init_cash=init_cash_,
            init_position=init_position_,
            init_price=init_price_,
            cash_deposits=cash_deposits_,
            cash_earnings=cash_earnings_,
            segment_mask=segment_mask_,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            bm_close=bm_close_,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_info=fill_pos_info,
            track_value=track_value,
            order_records=order_records,
            order_counts=order_counts,
            log_records=log_records,
            log_counts=log_counts,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_locked_cash=last_locked_cash,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            last_pos_info=last_pos_info,
            sim_start=sim_start_,
            sim_end=sim_end_,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col,
        )
        pre_group_out = pre_group_func_nb(pre_group_ctx, *pre_sim_out, *pre_group_args)

        _sim_start = sim_start_[group]
        _sim_end = sim_end_[group]
        for i in range(_sim_start, _sim_end):
            if track_value:
                # Update valuation price using current open
                for col in range(from_col, to_col):
                    _open = flex_select_nb(open_, i, col)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group], last_value[group]
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col], last_value[col]
                        )

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            is_segment_active = flex_select_nb(segment_mask_, i, group)
            if call_pre_segment or is_segment_active:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    cash_deposits=cash_deposits_,
                    cash_earnings=cash_earnings_,
                    segment_mask=segment_mask_,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    bm_close=bm_close_,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_info=fill_pos_info,
                    track_value=track_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    sim_start=sim_start_,
                    sim_end=sim_end_,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None,
                )
                pre_segment_out = pre_segment_func_nb(
                    pre_seg_ctx, *pre_group_out, *pre_segment_args
                )

            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_nb(cash_deposits_, i, group)
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_nb(cash_deposits_, i, col)
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits

            if track_value:
                # Update value and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            is_segment_active = flex_select_nb(segment_mask_, i, group)
            if is_segment_active:
                call_idx = -1
                while True:
                    call_idx += 1

                    # Generate the next order
                    flex_order_ctx = FlexOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        init_cash=init_cash_,
                        init_position=init_position_,
                        init_price=init_price_,
                        cash_deposits=cash_deposits_,
                        cash_earnings=cash_earnings_,
                        segment_mask=segment_mask_,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open_,
                        high=high_,
                        low=low_,
                        close=close_,
                        bm_close=bm_close_,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_info=fill_pos_info,
                        track_value=track_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_locked_cash=last_locked_cash,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        last_pos_info=last_pos_info,
                        sim_start=sim_start_,
                        sim_end=sim_end_,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        call_idx=call_idx,
                    )
                    col, order = flex_order_func_nb(
                        flex_order_ctx, *pre_segment_out, *flex_order_args
                    )

                    if col == -1:
                        break
                    if col < from_col or col >= to_col:
                        raise ValueError("Column out of bounds of the group")
                    if not track_value:
                        if (
                            order.size_type == SizeType.Value
                            or order.size_type == SizeType.TargetValue
                            or order.size_type == SizeType.TargetPercent
                        ):
                            raise ValueError(
                                "Cannot use size type that depends on not tracked value"
                            )

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    locked_cash_now = last_locked_cash[col]
                    val_price_now = last_val_price[col]
                    pos_info_now = last_pos_info[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                        cash_deposits_now = last_cash_deposits[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]
                        cash_deposits_now = last_cash_deposits[col]

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

                    if track_value:
                        val_price_now = new_exec_state.val_price
                        value_now = new_exec_state.value
                        if cash_sharing:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[group],
                                value_now - cash_deposits_now,
                            )
                        else:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[col], value_now - cash_deposits_now
                            )

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    last_locked_cash[col] = locked_cash_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now

                    if track_value:
                        if not np.isnan(val_price_now) or not ffill_val_price:
                            last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                            last_return[group] = return_now
                        else:
                            last_value[col] = value_now
                            last_return[col] = return_now

                    # Update position record
                    if fill_pos_info:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_info_nb(
                                pos_info_now,
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    # Post-order function
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        init_cash=init_cash_,
                        init_position=init_position_,
                        init_price=init_price_,
                        cash_deposits=cash_deposits_,
                        cash_earnings=cash_earnings_,
                        segment_mask=segment_mask_,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open_,
                        high=high_,
                        low=low_,
                        close=close_,
                        bm_close=bm_close_,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_info=fill_pos_info,
                        track_value=track_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_locked_cash=last_locked_cash,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        last_pos_info=last_pos_info,
                        sim_start=sim_start_,
                        sim_end=sim_end_,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        col=col,
                        call_idx=call_idx,
                        cash_before=exec_state.cash,
                        position_before=exec_state.position,
                        debt_before=exec_state.debt,
                        locked_cash_before=exec_state.locked_cash,
                        free_cash_before=exec_state.free_cash,
                        val_price_before=exec_state.val_price,
                        value_before=exec_state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        locked_cash_now=locked_cash_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_info_now=pos_info_now,
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats for future rows
            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_nb(cash_earnings_, i, col)
                if cash_sharing:
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings

            if track_value:
                # Update valuation price using current close
                for col in range(from_col, to_col):
                    _close = flex_select_nb(close_, i, col)
                    if not np.isnan(_close) or not ffill_val_price:
                        last_val_price[col] = _close

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                    prev_close_value[group] = last_value[group]
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )
                        prev_close_value[col] = last_value[col]

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            if call_post_segment or is_segment_active:
                # Call function after the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    cash_deposits=cash_deposits_,
                    cash_earnings=cash_earnings_,
                    segment_mask=segment_mask_,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    bm_close=bm_close_,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_info=fill_pos_info,
                    track_value=track_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    sim_start=sim_start_,
                    sim_end=sim_end_,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None,
                )
                post_segment_func_nb(post_seg_ctx, *pre_group_out, *post_segment_args)

            if i >= sim_end_[group] - 1:
                break

        # Call function after the group
        post_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=None,
            init_cash=init_cash_,
            init_position=init_position_,
            init_price=init_price_,
            cash_deposits=cash_deposits_,
            cash_earnings=cash_earnings_,
            segment_mask=segment_mask_,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            bm_close=bm_close_,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_info=fill_pos_info,
            track_value=track_value,
            order_records=order_records,
            order_counts=order_counts,
            log_records=log_records,
            log_counts=log_counts,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_locked_cash=last_locked_cash,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            last_pos_info=last_pos_info,
            sim_start=sim_start_,
            sim_end=sim_end_,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col,
        )
        post_group_func_nb(post_group_ctx, *pre_sim_out, *post_group_args)

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=None,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        segment_mask=segment_mask_,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open_,
        high=high_,
        low=low_,
        close=close_,
        bm_close=bm_close_,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_info=fill_pos_info,
        track_value=track_value,
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_locked_cash=last_locked_cash,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        last_pos_info=last_pos_info,
        sim_start=sim_start_,
        sim_end=sim_end_,
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

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
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        call_seq=None,
        in_outputs=in_outputs,
        sim_start=sim_start_out,
        sim_end=sim_end_out,
    )


# % </section>


# % <section from_flex_order_func_rw_nb>
# % <uncomment>
# import vectorbtpro as vbt
# from vectorbtpro.portfolio.nb.from_order_func import *
# %? import_lines
#
#
# % </uncomment>
# %? blocks[pre_sim_func_nb_block]
# % blocks["pre_sim_func_nb"]
# %? blocks[post_sim_func_nb_block]
# % blocks["post_sim_func_nb"]
# %? blocks[pre_row_func_nb_block]
# % blocks["pre_row_func_nb"]
# %? blocks[post_row_func_nb_block]
# % blocks["post_row_func_nb"]
# %? blocks[pre_segment_func_nb_block]
# % blocks["pre_segment_func_nb"]
# %? blocks[post_segment_func_nb_block]
# % blocks["post_segment_func_nb"]
# %? blocks[flex_order_func_nb_block]
# % blocks["flex_order_func_nb"]
# %? blocks[post_order_func_nb_block]
# % blocks["post_order_func_nb"]
@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=base_ch.flex_1d_array_gl_slicer,
        init_price=base_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=base_ch.flex_array_gl_slicer,
        segment_mask=base_ch.FlexArraySlicer(axis=1),
        call_pre_segment=None,
        call_post_segment=None,
        pre_sim_func_nb=None,  # % None
        pre_sim_args=ch.ArgsTaker(),
        post_sim_func_nb=None,  # % None
        post_sim_args=ch.ArgsTaker(),
        pre_row_func_nb=None,  # % None
        pre_row_args=ch.ArgsTaker(),
        post_row_func_nb=None,  # % None
        post_row_args=ch.ArgsTaker(),
        pre_segment_func_nb=None,  # % None
        pre_segment_args=ch.ArgsTaker(),
        post_segment_func_nb=None,  # % None
        post_segment_args=ch.ArgsTaker(),
        flex_order_func_nb=None,  # % None
        flex_order_args=ch.ArgsTaker(),
        post_order_func_nb=None,  # % None
        post_order_args=ch.ArgsTaker(),
        index=None,
        freq=None,
        open=base_ch.flex_array_gl_slicer,
        high=base_ch.flex_array_gl_slicer,
        low=base_ch.flex_array_gl_slicer,
        close=base_ch.flex_array_gl_slicer,
        bm_close=base_ch.flex_array_gl_slicer,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        ffill_val_price=None,
        update_value=None,
        fill_pos_info=None,
        track_value=None,
        max_order_records=None,
        max_log_records=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config,
    setup_id=None,  # %? line.replace("None", task_id)
)
@register_jitted(
    tags={"can_parallel"},
    cache=False,  # % line.replace("False", "True")
    task_id_or_func=None,  # %? line.replace("None", task_id)
)
def from_flex_order_func_rw_nb(  # %? line.replace("from_flex_order_func_rw_nb", new_func_name)
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    init_cash: tp.FlexArray1dLike = 100.0,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    segment_mask: tp.FlexArray2dLike = True,
    call_pre_segment: bool = False,
    call_post_segment: bool = False,
    pre_sim_func_nb: tp.PreSimFunc = no_pre_func_nb,  # % None
    pre_sim_args: tp.Args = (),
    post_sim_func_nb: tp.PostSimFunc = no_post_func_nb,  # % None
    post_sim_args: tp.Args = (),
    pre_row_func_nb: tp.PreRowFunc = no_pre_func_nb,  # % None
    pre_row_args: tp.Args = (),
    post_row_func_nb: tp.PostRowFunc = no_post_func_nb,  # % None
    post_row_args: tp.Args = (),
    pre_segment_func_nb: tp.PreSegmentFunc = no_pre_func_nb,  # % None
    pre_segment_args: tp.Args = (),
    post_segment_func_nb: tp.PostSegmentFunc = no_post_func_nb,  # % None
    post_segment_args: tp.Args = (),
    flex_order_func_nb: tp.FlexOrderFunc = no_flex_order_func_nb,  # % None
    flex_order_args: tp.Args = (),
    post_order_func_nb: tp.PostOrderFunc = no_post_func_nb,  # % None
    post_order_args: tp.Args = (),
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray2dLike = np.nan,
    high: tp.FlexArray2dLike = np.nan,
    low: tp.FlexArray2dLike = np.nan,
    close: tp.FlexArray2dLike = np.nan,
    bm_close: tp.FlexArray2dLike = np.nan,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_info: bool = True,
    track_value: bool = True,
    max_order_records: tp.Optional[int] = None,
    max_log_records: tp.Optional[int] = 0,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Same as `from_flex_order_func_nb`, but iterates in row-major order with rows changing fastest
    and columns/groups changing slowest.

    Args:
        target_shape (Shape): See `vectorbtpro.portfolio.enums.SimulationContext.target_shape`.
        group_lens (GroupLens): See `vectorbtpro.portfolio.enums.SimulationContext.group_lens`.
        cash_sharing (bool): See `vectorbtpro.portfolio.enums.SimulationContext.cash_sharing`.
        call_seq (Optional[Array2d]): See `vectorbtpro.portfolio.enums.SimulationContext.call_seq`.
        init_cash (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_cash`.

            Provided as a scalar or per column or group with cash sharing.
        init_position (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_position`.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): See `vectorbtpro.portfolio.enums.SimulationContext.init_price`.

            Provided as a scalar or per column.
        cash_deposits (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.cash_deposits`.

            Provided as a scalar, or per row, column or group with cash sharing, or element.
        cash_earnings (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.cash_earnings`.

            Provided as a scalar, or per row, column, or element.
        segment_mask (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.segment_mask`.

            Provided as a scalar, or per row, group, or element.
        call_pre_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_pre_segment`.
        call_post_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_post_segment`.
        pre_sim_func_nb (PreSimFunc): Callback function to be called before the simulation.

            This function is used for creating global arrays and setting the seed.

            Accepts `vectorbtpro.portfolio.enums.SimulationContext` and `*pre_sim_args`,
            and returns a tuple that is passed to `pre_row_func_nb` and `post_row_func_nb`.
        pre_sim_args (Args): Positional arguments for `pre_sim_func_nb`.
        post_sim_func_nb (PostSimFunc): Callback function to be called after the simulation.

            Accepts `vectorbtpro.portfolio.enums.SimulationContext` and `*post_sim_args`,
            and returns nothing.
        post_sim_args (Args): Positional arguments for `post_sim_func_nb`.
        pre_row_func_nb (PreRowFunc): Callback function to be called before processing a row.

            Accepts `vectorbtpro.portfolio.enums.RowContext`, the unpacked output from
            `pre_sim_func_nb`, and `*pre_row_args`, and returns a tuple that is passed to
            `pre_segment_func_nb` and `post_segment_func_nb`.
        pre_row_args (Args): Positional arguments for `pre_row_func_nb`.
        post_row_func_nb (PostRowFunc): Callback function to be called after processing a row.

            Accepts `vectorbtpro.portfolio.enums.RowContext`, the unpacked output from
            `pre_sim_func_nb`, and `*post_row_args`, and returns nothing.
        post_row_args (Args): Positional arguments for `post_row_func_nb`.
        pre_segment_func_nb (PreSegmentFunc): Callback function to be called before processing a segment
            if `segment_mask` or `call_pre_segment` is True.

            Accepts `vectorbtpro.portfolio.enums.SegmentContext`, the unpacked output from
            `pre_row_func_nb`, and `*pre_segment_args`, and returns a tuple that is passed to
            `flex_order_func_nb` and `post_order_func_nb`.

            This is the appropriate place to adjust the call sequence or set the valuation price.
            Group re-valuation and updates of open position stats occur immediately after this function
            executes, regardless of whether it is called.

            !!! note
                To change the call sequence of a segment, modify
                `vectorbtpro.portfolio.enums.SegmentContext.call_seq_now` in-place.
                Avoid creating new arrays to prevent performance degradation.
                Assigning a new context is not supported.

            !!! note
                You can override elements of `last_val_price` to influence group valuation.
                See `vectorbtpro.portfolio.enums.SimulationContext.last_val_price`.
        pre_segment_args (Args): Positional arguments for `pre_segment_func_nb`.
        post_segment_func_nb (PostSegmentFunc): Callback function to be called after processing a segment
            if `segment_mask` or `call_post_segment` is True.

            Handles the addition of `cash_earnings`, final group re-valuation, and the final update
            of open position stats.

            Accepts `vectorbtpro.portfolio.enums.SegmentContext`, the unpacked output from
            `pre_row_func_nb`, and `*post_segment_args`, and returns nothing.
        post_segment_args (Args): Positional arguments for `post_segment_func_nb`.
        flex_order_func_nb (FlexOrderFunc): Callback function to be called to generate a flexible order.

            Used for generating an order in a column.

            Accepts `vectorbtpro.portfolio.enums.FlexOrderContext`, the unpacked output from
            `pre_segment_func_nb`, and `*order_args`, and returns a tuple of
            (column, `vectorbtpro.portfolio.enums.Order`).
        flex_order_args (Args): Positional arguments for `flex_order_func_nb`.
        post_order_func_nb (PostOrderFunc): Callback function to be called after processing an order.

            Accepts `vectorbtpro.portfolio.enums.PostOrderContext`, the unpacked output
            from `pre_segment_func_nb`, and `*post_order_args`, and returns nothing.
        post_order_args (Args): Positional arguments for `post_order_func_nb`.
        index (Optional[Array1d]): See `vectorbtpro.portfolio.enums.SimulationContext.index`.
        freq (Optional[int]): See `vectorbtpro.portfolio.enums.SimulationContext.freq`.
        open (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.open`.

            Provided as a scalar, or per row, column, or element.
        high (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.high`.

            Provided as a scalar, or per row, column, or element.
        low (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.low`.

            Provided as a scalar, or per row, column, or element.
        close (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.close`.

            Provided as a scalar, or per row, column, or element.
        bm_close (FlexArray2dLike): See `vectorbtpro.portfolio.enums.SimulationContext.bm_close`.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): See `vectorbtpro.portfolio.enums.SimulationContext.sim_start`.

            Provided as a scalar or per group.
        sim_end (Optional[FlexArray1dLike]): See `vectorbtpro.portfolio.enums.SimulationContext.sim_end`.

            Provided as a scalar or per group.
        ffill_val_price (bool): See `vectorbtpro.portfolio.enums.SimulationContext.ffill_val_price`.
        update_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.update_value`.
        fill_pos_info (bool): See `vectorbtpro.portfolio.enums.SimulationContext.fill_pos_info`.
        track_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.track_value`.
        max_order_records (Optional[int]): Maximum number of order records expected per column.

            Defaults to the number of rows in the broadcasted shape. Set to 0 to disable,
            lower to reduce memory usage, or higher if multiple orders per timestamp are expected.
        max_log_records (Optional[int]): Maximum number of log records expected per column.

            Set to the number of rows in the broadcasted shape if logging is enabled. Set lower to
            reduce memory usage, or higher if multiple logs per timestamp are expected.
        in_outputs (Optional[NamedTuple]): See `vectorbtpro.portfolio.enums.SimulationContext.in_outputs`.

    Returns:
        SimulationOutput: Simulation output containing order records, log records, and
            other simulation results.

    !!! tip
        This function is parallelizable.

    Call hierarchy:
        ```plaintext
        1. pre_sim_out = pre_sim_func_nb(SimulationContext, *pre_sim_args)
            2. pre_row_out = pre_row_func_nb(RowContext, *pre_sim_out, *pre_row_args)
                3. if call_pre_segment or segment_mask:
                    pre_segment_out = pre_segment_func_nb(SegmentContext, *pre_row_out, *pre_segment_args)
                    while col != -1:
                        4. if segment_mask:
                            col, order = flex_order_func_nb(FlexOrderContext, *pre_segment_out, *flex_order_args)
                        5. if order:
                            post_order_func_nb(PostOrderContext, *pre_segment_out, *post_order_args)
                6. if call_post_segment or segment_mask:
                    post_segment_func_nb(SegmentContext, *pre_row_out, *post_segment_args)
            7. post_row_func_nb(RowContext, *pre_sim_out, *post_row_args)
        8. post_sim_func_nb(SimulationContext, *post_sim_args)
        ```

        Let's illustrate the same example as in `from_order_func_nb` but adapted for this function:

        ```pycon
        >>> @njit
        ... def pre_row_func_nb(c, order_value_out, call_seq_out):
        ...     print('\\tbefore row', c.i)
        ...     return (order_value_out, call_seq_out)

        >>> @njit
        ... def post_row_func_nb(c, order_value_out, call_seq_out):
        ...     print('\\tafter row', c.i)
        ...     return None

        >>> sim_out = vbt.pf_nb.from_flex_order_func_rw_nb(
        ...     target_shape,
        ...     group_lens,
        ...     cash_sharing,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_row_func_nb=pre_row_func_nb,
        ...     post_row_func_nb=post_row_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     flex_order_func_nb=flex_order_func_nb,
        ...     flex_order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        ```

        ![](/assets/images/api/from_flex_order_func_rw_nb.svg){: loading=lazy style="width:800px;" }
    """
    check_group_lens_nb(group_lens, target_shape[1])

    init_cash_ = to_1d_array_nb(np.asarray(init_cash))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))
    segment_mask_ = to_2d_array_nb(np.asarray(segment_mask))
    open_ = to_2d_array_nb(np.asarray(open))
    high_ = to_2d_array_nb(np.asarray(high))
    low_ = to_2d_array_nb(np.asarray(low))
    close_ = to_2d_array_nb(np.asarray(close))
    bm_close_ = to_2d_array_nb(np.asarray(bm_close))

    order_records, log_records = prepare_records_nb(
        target_shape=target_shape,
        max_order_records=max_order_records,
        max_log_records=max_log_records,
    )
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
    last_pos_info = prepare_last_pos_info_nb(
        target_shape,
        init_position=init_position_,
        init_price=init_price_,
        fill_pos_info=fill_pos_info,
    )

    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full_like(last_position, 0.0)
    last_locked_cash = np.full_like(last_position, 0.0)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=int_)
    log_counts = np.full(target_shape[1], 0, dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(target_shape[0], len(group_lens)),
        sim_start=sim_start,
        sim_end=sim_end,
    )

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=None,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        segment_mask=segment_mask_,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open_,
        high=high_,
        low=low_,
        close=close_,
        bm_close=bm_close_,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_info=fill_pos_info,
        track_value=track_value,
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_locked_cash=last_locked_cash,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        last_pos_info=last_pos_info,
        sim_start=sim_start_,
        sim_end=sim_end_,
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    _sim_start = sim_start_.min()
    _sim_end = sim_end_.max()
    for i in range(_sim_start, _sim_end):
        # Call function before the row
        pre_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=None,
            init_cash=init_cash_,
            init_position=init_position_,
            init_price=init_price_,
            cash_deposits=cash_deposits_,
            cash_earnings=cash_earnings_,
            segment_mask=segment_mask_,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            bm_close=bm_close_,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_info=fill_pos_info,
            track_value=track_value,
            order_records=order_records,
            order_counts=order_counts,
            log_records=log_records,
            log_counts=log_counts,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_locked_cash=last_locked_cash,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            last_pos_info=last_pos_info,
            sim_start=sim_start_,
            sim_end=sim_end_,
            i=i,
        )
        pre_row_out = pre_row_func_nb(pre_row_ctx, *pre_sim_out, *pre_row_args)

        for group in range(len(group_lens)):
            if i < sim_start_[group] or i >= sim_end_[group]:
                continue

            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]
            group_len = to_col - from_col

            if track_value:
                # Update valuation price using current open
                for col in range(from_col, to_col):
                    _open = flex_select_nb(open_, i, col)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group], last_value[group]
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col], last_value[col]
                        )

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            is_segment_active = flex_select_nb(segment_mask_, i, group)
            if call_pre_segment or is_segment_active:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    cash_deposits=cash_deposits_,
                    cash_earnings=cash_earnings_,
                    segment_mask=segment_mask_,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    bm_close=bm_close_,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_info=fill_pos_info,
                    track_value=track_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    sim_start=sim_start_,
                    sim_end=sim_end_,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None,
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_row_out, *pre_segment_args)

            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_nb(cash_deposits_, i, group)
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_nb(cash_deposits_, i, col)
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits

            if track_value:
                # Update value and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            is_segment_active = flex_select_nb(segment_mask_, i, group)
            if is_segment_active:
                call_idx = -1
                while True:
                    call_idx += 1

                    # Generate the next order
                    flex_order_ctx = FlexOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        init_cash=init_cash_,
                        init_position=init_position_,
                        init_price=init_price_,
                        cash_deposits=cash_deposits_,
                        cash_earnings=cash_earnings_,
                        segment_mask=segment_mask_,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open_,
                        high=high_,
                        low=low_,
                        close=close_,
                        bm_close=bm_close_,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_info=fill_pos_info,
                        track_value=track_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_locked_cash=last_locked_cash,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        last_pos_info=last_pos_info,
                        sim_start=sim_start_,
                        sim_end=sim_end_,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        call_idx=call_idx,
                    )
                    col, order = flex_order_func_nb(
                        flex_order_ctx, *pre_segment_out, *flex_order_args
                    )

                    if col == -1:
                        break
                    if col < from_col or col >= to_col:
                        raise ValueError("Column out of bounds of the group")
                    if not track_value:
                        if (
                            order.size_type == SizeType.Value
                            or order.size_type == SizeType.TargetValue
                            or order.size_type == SizeType.TargetPercent
                        ):
                            raise ValueError(
                                "Cannot use size type that depends on not tracked value"
                            )

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    locked_cash_now = last_locked_cash[col]
                    val_price_now = last_val_price[col]
                    pos_info_now = last_pos_info[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                        cash_deposits_now = last_cash_deposits[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]
                        cash_deposits_now = last_cash_deposits[col]

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

                    if track_value:
                        val_price_now = new_exec_state.val_price
                        value_now = new_exec_state.value
                        if cash_sharing:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[group],
                                value_now - cash_deposits_now,
                            )
                        else:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[col], value_now - cash_deposits_now
                            )

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    last_locked_cash[col] = locked_cash_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now

                    if track_value:
                        if not np.isnan(val_price_now) or not ffill_val_price:
                            last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                            last_return[group] = return_now
                        else:
                            last_value[col] = value_now
                            last_return[col] = return_now

                    # Update position record
                    if fill_pos_info:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_info_nb(
                                pos_info_now,
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    # Post-order function
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        init_cash=init_cash_,
                        init_position=init_position_,
                        init_price=init_price_,
                        cash_deposits=cash_deposits_,
                        cash_earnings=cash_earnings_,
                        segment_mask=segment_mask_,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open_,
                        high=high_,
                        low=low_,
                        close=close_,
                        bm_close=bm_close_,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_info=fill_pos_info,
                        track_value=track_value,
                        order_records=order_records,
                        order_counts=order_counts,
                        log_records=log_records,
                        log_counts=log_counts,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_locked_cash=last_locked_cash,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        last_pos_info=last_pos_info,
                        sim_start=sim_start_,
                        sim_end=sim_end_,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        col=col,
                        call_idx=call_idx,
                        cash_before=exec_state.cash,
                        position_before=exec_state.position,
                        debt_before=exec_state.debt,
                        locked_cash_before=exec_state.locked_cash,
                        free_cash_before=exec_state.free_cash,
                        val_price_before=exec_state.val_price,
                        value_before=exec_state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        locked_cash_now=locked_cash_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_info_now=pos_info_now,
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats for future rows
            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_nb(cash_earnings_, i, col)
                if cash_sharing:
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings

            if track_value:
                # Update valuation price using current close
                for col in range(from_col, to_col):
                    _close = flex_select_nb(close_, i, col)
                    if not np.isnan(_close) or not ffill_val_price:
                        last_val_price[col] = _close

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = calc_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                    prev_close_value[group] = last_value[group]
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = (
                                last_cash[col] + last_position[col] * last_val_price[col]
                            )
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )
                        prev_close_value[col] = last_value[col]

                # Update open position stats
                if fill_pos_info:
                    for col in range(from_col, to_col):
                        update_open_pos_info_stats_nb(
                            last_pos_info[col], last_position[col], last_val_price[col]
                        )

            # Is this segment active?
            if call_post_segment or is_segment_active:
                # Call function after the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    init_cash=init_cash_,
                    init_position=init_position_,
                    init_price=init_price_,
                    cash_deposits=cash_deposits_,
                    cash_earnings=cash_earnings_,
                    segment_mask=segment_mask_,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open_,
                    high=high_,
                    low=low_,
                    close=close_,
                    bm_close=bm_close_,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_info=fill_pos_info,
                    track_value=track_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_locked_cash=last_locked_cash,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_pos_info=last_pos_info,
                    sim_start=sim_start_,
                    sim_end=sim_end_,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None,
                )
                post_segment_func_nb(post_seg_ctx, *pre_row_out, *post_segment_args)

        # Call function after the row
        post_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=None,
            init_cash=init_cash_,
            init_position=init_position_,
            init_price=init_price_,
            cash_deposits=cash_deposits_,
            cash_earnings=cash_earnings_,
            segment_mask=segment_mask_,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open_,
            high=high_,
            low=low_,
            close=close_,
            bm_close=bm_close_,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_info=fill_pos_info,
            track_value=track_value,
            order_records=order_records,
            order_counts=order_counts,
            log_records=log_records,
            log_counts=log_counts,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_locked_cash=last_locked_cash,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            last_pos_info=last_pos_info,
            sim_start=sim_start_,
            sim_end=sim_end_,
            i=i,
        )
        post_row_func_nb(post_row_ctx, *pre_sim_out, *post_row_args)

        sim_end_reached = True
        for group in range(len(group_lens)):
            if i < sim_end_[group] - 1:
                sim_end_reached = False
                break
        if sim_end_reached:
            break

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=None,
        init_cash=init_cash_,
        init_position=init_position_,
        init_price=init_price_,
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        segment_mask=segment_mask_,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open_,
        high=high_,
        low=low_,
        close=close_,
        bm_close=bm_close_,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_info=fill_pos_info,
        track_value=track_value,
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_locked_cash=last_locked_cash,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        last_pos_info=last_pos_info,
        sim_start=sim_start_,
        sim_end=sim_end_,
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

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
        cash_deposits=cash_deposits_,
        cash_earnings=cash_earnings_,
        call_seq=None,
        in_outputs=in_outputs,
        sim_start=sim_start_out,
        sim_end=sim_end_out,
    )


# % </section>


@register_jitted
def set_val_price_nb(c: SegmentContext, val_price: tp.FlexArray2d, price: tp.FlexArray2d) -> None:
    """Override valuation price in a segment context.

    Updates the valuation price in the context using the provided valuation and price arrays.
    When a valuation price is positive infinity, the corresponding market price is used.
    If that price is also positive infinity, the function falls back to the close or open price.
    For a negative infinity valuation price, the last valuation price is retained.

    Args:
        c (SegmentContext): Segment context.
        val_price (FlexArray2d): Array of valuation prices with special handling for infinity.
        price (FlexArray2d): Array of current market prices.

    Returns:
        None: Function modifies the context in place.
    """
    for col in range(c.from_col, c.to_col):
        _val_price = select_from_col_nb(c, col, val_price)
        if np.isinf(_val_price):
            if _val_price > 0:
                _price = select_from_col_nb(c, col, price)
                if np.isinf(_price):
                    if _price > 0:
                        _price = select_from_col_nb(c, col, c.close)
                    else:
                        _price = select_from_col_nb(c, col, c.open)
                _val_price = _price
            else:
                _val_price = c.last_val_price[col]
        if not np.isnan(_val_price) or not c.ffill_val_price:
            c.last_val_price[col] = _val_price


# % <block def_pre_segment_func_nb>
@register_jitted
def def_pre_segment_func_nb(  # % line.replace("def_pre_segment_func_nb", "pre_segment_func_nb")
    c: SegmentContext,
    val_price: tp.FlexArray2d,
    price: tp.FlexArray2d,
    size: tp.FlexArray2d,
    size_type: tp.FlexArray2d,
    direction: tp.FlexArray2d,
    auto_call_seq: bool,
) -> tp.Args:
    """Custom segment pre-processing function that sets valuation price and sorts the call sequence.

    Adjusts the valuation price in the segment context using the provided valuation and price arrays.
    If `auto_call_seq` is True, it also sorts the call sequence based on trade size parameters.

    Args:
        c (SegmentContext): Segment context.
        val_price (FlexArray2d): Array of valuation prices with special handling for infinity.
        price (FlexArray2d): Array of market prices.
        size (FlexArray2d): Array of order sizes.
        size_type (FlexArray2d): Array denoting the type for each trade size.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (FlexArray2d): Array indicating the order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        auto_call_seq (bool): Flag to automatically sort the call sequence.

    Returns:
        Args: Empty tuple.
    """
    set_val_price_nb(c, val_price, price)
    if auto_call_seq:
        order_value_out = np.empty(c.group_len, dtype=float_)
        sort_call_seq_nb(c, size, size_type, direction, order_value_out)
    return ()


# % </block>


# % <block def_order_func_nb>
@register_jitted
def def_order_func_nb(  # % line.replace("def_order_func_nb", "order_func_nb")
    c: OrderContext,
    size: tp.FlexArray2d,
    price: tp.FlexArray2d,
    size_type: tp.FlexArray2d,
    direction: tp.FlexArray2d,
    fees: tp.FlexArray2d,
    fixed_fees: tp.FlexArray2d,
    slippage: tp.FlexArray2d,
    min_size: tp.FlexArray2d,
    max_size: tp.FlexArray2d,
    size_granularity: tp.FlexArray2d,
    leverage: tp.FlexArray2d,
    leverage_mode: tp.FlexArray2d,
    reject_prob: tp.FlexArray2d,
    price_area_vio_mode: tp.FlexArray2d,
    allow_partial: tp.FlexArray2d,
    raise_reject: tp.FlexArray2d,
    log: tp.FlexArray2d,
) -> tp.Tuple[int, Order]:
    """Custom order processing function that creates an order with default parameters.

    Constructs an order using the provided arrays for size, price, fees, and other order parameters
    by selecting the appropriate value for the current context.

    Args:
        c (OrderContext): Order context.
        size (FlexArray2d): Array of order sizes.
        price (FlexArray2d): Array of order prices.
        size_type (FlexArray2d): Array specifying the type of each order size.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (FlexArray2d): Array indicating the order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        fees (FlexArray2d): Array of fee values.
        fixed_fees (FlexArray2d): Array of fixed fee values.
        slippage (FlexArray2d): Array representing slippage.
        min_size (FlexArray2d): Array of minimum order sizes.
        max_size (FlexArray2d): Array of maximum order sizes.
        size_granularity (FlexArray2d): Array defining granularity for order sizes.
        leverage (FlexArray2d): Array of leverage amounts.
        leverage_mode (FlexArray2d): Array indicating leverage modes.

            See `vectorbtpro.portfolio.enums.LeverageMode`.
        reject_prob (FlexArray2d): Array of rejection probabilities.
        price_area_vio_mode (FlexArray2d): Array specifying handling of price area violations.

            See `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
        allow_partial (FlexArray2d): Array indicating whether partial orders are allowed.
        raise_reject (FlexArray2d): Array determining if rejections should raise errors.
        log (FlexArray2d): Array containing logging configurations.

    Returns:
        Tuple[int, Order]: Tuple where the first element is an indicator (typically a column index)
            and the second element is the created order.
    """
    return order_nb(
        size=select_nb(c, size),
        price=select_nb(c, price),
        size_type=select_nb(c, size_type),
        direction=select_nb(c, direction),
        fees=select_nb(c, fees),
        fixed_fees=select_nb(c, fixed_fees),
        slippage=select_nb(c, slippage),
        min_size=select_nb(c, min_size),
        max_size=select_nb(c, max_size),
        size_granularity=select_nb(c, size_granularity),
        leverage=select_nb(c, leverage),
        leverage_mode=select_nb(c, leverage_mode),
        reject_prob=select_nb(c, reject_prob),
        price_area_vio_mode=select_nb(c, price_area_vio_mode),
        allow_partial=select_nb(c, allow_partial),
        raise_reject=select_nb(c, raise_reject),
        log=select_nb(c, log),
    )


# % </block>


# % <block def_flex_pre_segment_func_nb>
@register_jitted
def def_flex_pre_segment_func_nb(  # % line.replace("def_flex_pre_segment_func_nb", "pre_segment_func_nb")
    c: SegmentContext,
    val_price: tp.FlexArray2d,
    price: tp.FlexArray2d,
    size: tp.FlexArray2d,
    size_type: tp.FlexArray2d,
    direction: tp.FlexArray2d,
    auto_call_seq: bool,
) -> tp.Args:
    """Custom flexible segment pre-processing function that sets valuation price and returns a call sequence.

    Sets the valuation price in the segment context using the provided arrays and computes
    a flexible call sequence array. If `auto_call_seq` is True, the function sorts the call
    sequence using trade size parameters.

    Args:
        c (SegmentContext): Segment context.
        val_price (FlexArray2d): Array of valuation prices with special handling for infinity.
        price (FlexArray2d): Array of market prices.
        size (FlexArray2d): Array of order sizes.
        size_type (FlexArray2d): Array indicating the type of trade sizes.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (FlexArray2d): Array indicating the order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        auto_call_seq (bool): Flag to automatically sort the call sequence.

    Returns:
        Args: Tuple containing a 1D array of indices representing the call sequence.
    """
    set_val_price_nb(c, val_price, price)
    call_seq_out = np.arange(c.group_len)
    if auto_call_seq:
        order_value_out = np.empty(c.group_len, dtype=float_)
        sort_call_seq_out_nb(c, size, size_type, direction, order_value_out, call_seq_out)
    return (call_seq_out,)


# % </block>


# % <block def_flex_order_func_nb>
@register_jitted
def def_flex_order_func_nb(  # % line.replace("def_flex_order_func_nb", "flex_order_func_nb")
    c: FlexOrderContext,
    call_seq_now: tp.Array1d,
    size: tp.FlexArray2d,
    price: tp.FlexArray2d,
    size_type: tp.FlexArray2d,
    direction: tp.FlexArray2d,
    fees: tp.FlexArray2d,
    fixed_fees: tp.FlexArray2d,
    slippage: tp.FlexArray2d,
    min_size: tp.FlexArray2d,
    max_size: tp.FlexArray2d,
    size_granularity: tp.FlexArray2d,
    leverage: tp.FlexArray2d,
    leverage_mode: tp.FlexArray2d,
    reject_prob: tp.FlexArray2d,
    price_area_vio_mode: tp.FlexArray2d,
    allow_partial: tp.FlexArray2d,
    raise_reject: tp.FlexArray2d,
    log: tp.FlexArray2d,
) -> tp.Tuple[int, Order]:
    """Custom flexible order processing function that creates an order with default parameters.

    Constructs an order for a flexible order context by selecting values from the provided arrays using
    a column determined by the current call sequence. If no valid call index exists, a no-op order is returned.

    Args:
        c (FlexOrderContext): Flexible order context.
        call_seq_now (Array1d): Array representing the current call sequence for column selection.
        size (FlexArray2d): Array of order sizes.
        price (FlexArray2d): Array of order prices.
        size_type (FlexArray2d): Array specifying the type of each order size.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (FlexArray2d): Array indicating the order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        fees (FlexArray2d): Array of fee values.
        fixed_fees (FlexArray2d): Array of fixed fee values.
        slippage (FlexArray2d): Array representing slippage.
        min_size (FlexArray2d): Array of minimum order sizes.
        max_size (FlexArray2d): Array of maximum order sizes.
        size_granularity (FlexArray2d): Array defining granularity for order sizes.
        leverage (FlexArray2d): Array of leverage amounts.
        leverage_mode (FlexArray2d): Array indicating leverage modes.

            See `vectorbtpro.portfolio.enums.LeverageMode`.
        reject_prob (FlexArray2d): Array of rejection probabilities.
        price_area_vio_mode (FlexArray2d): Array specifying handling of price area violations.

            See `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
        allow_partial (FlexArray2d): Array indicating whether partial orders are allowed.
        raise_reject (FlexArray2d): Array determining if rejections should raise errors.
        log (FlexArray2d): Array containing logging configurations.

    Returns:
        Tuple[int, Order]: Tuple where the first element is the column index used to generate the order
            (or -1 if no valid index exists) and the second element is the created order (or a no-op order).
    """
    if c.call_idx < c.group_len:
        col = c.from_col + call_seq_now[c.call_idx]
        order = order_nb(
            size=select_from_col_nb(c, col, size),
            price=select_from_col_nb(c, col, price),
            size_type=select_from_col_nb(c, col, size_type),
            direction=select_from_col_nb(c, col, direction),
            fees=select_from_col_nb(c, col, fees),
            fixed_fees=select_from_col_nb(c, col, fixed_fees),
            slippage=select_from_col_nb(c, col, slippage),
            min_size=select_from_col_nb(c, col, min_size),
            max_size=select_from_col_nb(c, col, max_size),
            size_granularity=select_from_col_nb(c, col, size_granularity),
            leverage=select_from_col_nb(c, col, leverage),
            leverage_mode=select_from_col_nb(c, col, leverage_mode),
            reject_prob=select_from_col_nb(c, col, reject_prob),
            price_area_vio_mode=select_from_col_nb(c, col, price_area_vio_mode),
            allow_partial=select_from_col_nb(c, col, allow_partial),
            raise_reject=select_from_col_nb(c, col, raise_reject),
            log=select_from_col_nb(c, col, log),
        )
        return col, order
    return -1, order_nothing_nb()


# % </block>
