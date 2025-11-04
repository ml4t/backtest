# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for portfolio analysis."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns import nb as returns_nb_
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import add_nb, is_close_nb
from vectorbtpro.utils.template import RepFunc

# ############# Assets ############# #


@register_jitted(cache=True)
def get_long_size_nb(position_before: float, position_now: float) -> float:
    """Compute the change in long position size.

    Calculates the change in long position based on the previous position (`position_before`)
    and the current position (`position_now`). Returns 0.0 if both positions are not long;
    otherwise, computes the adjustment for a transition into or out of a long position.

    Args:
        position_before (float): Asset position before the trade.
        position_now (float): Asset position after the trade.

    Returns:
        float: Computed change in long size.
    """
    if position_before <= 0 and position_now <= 0:
        return 0.0
    if position_before >= 0 and position_now < 0:
        return -position_before
    if position_before < 0 and position_now >= 0:
        return position_now
    return add_nb(position_now, -position_before)


@register_jitted(cache=True)
def get_short_size_nb(position_before: float, position_now: float) -> float:
    """Compute the change in short position size.

    Calculates the change in short position based on the previous position (`position_before`)
    and the current position (`position_now`). Returns 0.0 if both positions are not short;
    otherwise, computes the adjustment for a transition into or out of a short position.

    Args:
        position_before (float): Asset position before the trade.
        position_now (float): Asset position after the trade.

    Returns:
        float: Computed change in short size.
    """
    if position_before >= 0 and position_now >= 0:
        return 0.0
    if position_before >= 0 and position_now < 0:
        return -position_now
    if position_before < 0 and position_now >= 0:
        return position_before
    return add_nb(position_before, -position_now)


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        direction=None,
        init_position=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_flow_nb(
    target_shape: tp.Shape,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    direction: int = Direction.Both,
    init_position: tp.FlexArray1dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute asset flow series per column.

    Calculates the total transacted asset amount at each bar based on the order records.
    The asset flow is determined by computing changes in position from the orders and may be filtered
    by direction using the `direction` parameter.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        direction (int): Position direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array representing asset flow at each bar per column.

    !!! tip
        This function is parallelizable.
    """
    init_position_ = to_1d_array_nb(np.asarray(init_position))

    out = np.full(target_shape, np.nan, dtype=float_)

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        out[_sim_start:_sim_end, col] = 0.0
        if _sim_start >= _sim_end:
            continue
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = flex_select_1d_pc_nb(init_position_, col)

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            i = order_record["idx"]
            side = order_record["side"]
            size = order_record["size"]

            if side == OrderSide.Sell:
                size *= -1
            new_position_now = add_nb(position_now, size)
            if direction == Direction.LongOnly:
                asset_flow = get_long_size_nb(position_now, new_position_now)
            elif direction == Direction.ShortOnly:
                asset_flow = get_short_size_nb(position_now, new_position_now)
            else:
                asset_flow = size
            out[i, col] = add_nb(out[i, col], asset_flow)
            position_now = new_position_now
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_flow", axis=1),
    arg_take_spec=dict(
        asset_flow=ch.ArraySlicer(axis=1),
        direction=None,
        init_position=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def assets_nb(
    asset_flow: tp.Array2d,
    direction: int = Direction.Both,
    init_position: tp.FlexArray1dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute asset series per column.

    Calculates the current asset position at each bar by cumulatively summing the asset flow.
    Positions are adjusted based on the specified `direction`, which filters long or short exposures.

    Args:
        asset_flow (Array2d): Array of asset flow values.
        direction (int): Position direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array containing the updated asset position at each bar per column.

    !!! tip
        This function is parallelizable.
    """
    init_position_ = to_1d_array_nb(np.asarray(init_position))

    out = np.full(asset_flow.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_flow.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(asset_flow.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        position_now = flex_select_1d_pc_nb(init_position_, col)

        for i in range(_sim_start, _sim_end):
            flow_value = asset_flow[i, col]
            position_now = add_nb(position_now, flow_value)
            if direction == Direction.Both or direction == Direction.LongOnly and position_now > 0:
                out[i, col] = position_now
            elif direction == Direction.ShortOnly and position_now < 0:
                out[i, col] = -position_now
            else:
                out[i, col] = 0.0
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="assets", axis=1),
    arg_take_spec=dict(
        assets=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def position_mask_nb(
    assets: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute position mask per column.

    Generates a boolean mask that indicates whether a non-zero asset position exists at each bar.
    The mask is computed within the simulation range defined by `sim_start` and `sim_end`.

    Args:
        assets (Array2d): Array of asset positions per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Boolean array indicating the presence of an asset position per column at each bar.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(assets.shape, False, dtype=np.bool_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=assets.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(assets.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            if assets[i, col] != 0:
                out[i, col] = True
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        assets=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def position_mask_grouped_nb(
    assets: tp.Array2d,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Generate a boolean mask indicating active positions for each group based on asset data.

    Args:
        assets (Array2d): Array of asset positions per column.
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Boolean array of shape (number of bars, number of groups) where True indicates
            at least one active position in the corresponding group.

    !!! tip
        This function is parallelizable.
    """
    out = np.full((assets.shape[0], len(group_lens)), False, dtype=np.bool_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=assets.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue

            for i in range(_sim_start, _sim_end):
                if not np.isnan(assets[i, col]) and assets[i, col] != 0:
                    out[i, group] = True
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="assets", axis=1),
    arg_take_spec=dict(
        assets=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def position_coverage_nb(
    assets: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute the position coverage ratio for each asset column.

    Args:
        assets (Array2d): Array of asset positions per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array containing the coverage ratio (fraction of non-zero positions) for each asset column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(assets.shape[1], 0.0, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=assets.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(assets.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        hit_elements = 0

        for i in range(_sim_start, _sim_end):
            if assets[i, col] != 0:
                hit_elements += 1

        out[col] = hit_elements / (_sim_end - _sim_start)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        assets=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        granular_groups=None,
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def position_coverage_grouped_nb(
    assets: tp.Array2d,
    group_lens: tp.GroupLens,
    granular_groups: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute the position coverage ratio for each group of asset columns.

    Args:
        assets (Array2d): Array of asset positions per column.
        group_lens (GroupLens): Array defining the number of columns in each group.
        granular_groups (bool): Flag to determine if coverage is computed per individual column within a group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array of coverage ratios for each group.

            Each ratio is the fraction of bars with active positions within
            the aggregated simulation range, or NaN if no valid simulation range exists.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(len(group_lens), 0.0, dtype=float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=assets.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        n_elements = 0
        hit_elements = 0

        if granular_groups:
            for col in range(from_col, to_col):
                _sim_start = sim_start_[col]
                _sim_end = sim_end_[col]
                if _sim_start >= _sim_end:
                    continue
                n_elements += _sim_end - _sim_start

                for i in range(_sim_start, _sim_end):
                    if not np.isnan(assets[i, col]) and assets[i, col] != 0:
                        hit_elements += 1
        else:
            min_sim_start = assets.shape[0]
            max_sim_end = 0
            for col in range(from_col, to_col):
                _sim_start = sim_start_[col]
                _sim_end = sim_end_[col]
                if _sim_start >= _sim_end:
                    continue
                if _sim_start < min_sim_start:
                    min_sim_start = _sim_start
                if _sim_end > max_sim_end:
                    max_sim_end = _sim_end
            if min_sim_start >= max_sim_end:
                continue
            n_elements = max_sim_end - min_sim_start

            for i in range(min_sim_start, max_sim_end):
                for col in range(from_col, to_col):
                    _sim_start = sim_start_[col]
                    _sim_end = sim_end_[col]
                    if _sim_start >= _sim_end:
                        continue
                    if not np.isnan(assets[i, col]) and assets[i, col] != 0:
                        hit_elements += 1
                        break

        if n_elements == 0:
            out[group] = np.nan
        else:
            out[group] = hit_elements / n_elements
    return out


# ############# Cash ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        cash_deposits_raw=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        split_shared=None,
        weights=base_ch.flex_1d_array_gl_slicer,
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_deposits_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    cash_deposits_raw: tp.FlexArray2dLike = 0.0,
    split_shared: bool = False,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate cash deposit series per column.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        cash_sharing (bool): Flag indicating whether cash is shared among assets of the same group.
        cash_deposits_raw (FlexArray2dLike): Raw cash deposits.

            Provided as a scalar, or per row, column, or element.
        split_shared (bool): Whether to split shared cash equally among columns in a group.
        weights (Optional[FlexArray1dLike]): Weights applied to cash deposits.

            Provided as a scalar or per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array containing the cash deposit series per column.

    !!! tip
        This function is parallelizable.
    """
    cash_deposits_raw_ = to_2d_array_nb(np.asarray(cash_deposits_raw))
    if weights is None:
        weights_ = np.full(target_shape[1], np.nan, dtype=float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(float_))

    out = np.full(target_shape, np.nan, dtype=float_)

    if not cash_sharing:
        sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
            sim_shape=target_shape,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        for col in prange(target_shape[1]):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue
            _weights = flex_select_1d_pc_nb(weights_, col)

            for i in range(_sim_start, _sim_end):
                _cash_deposits = flex_select_nb(cash_deposits_raw_, i, col)
                if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                    out[i, col] = _weights * _cash_deposits
                else:
                    out[i, col] = _cash_deposits
    else:
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
            sim_shape=target_shape,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        for group in prange(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]

            for col in range(from_col, to_col):
                _sim_start = sim_start_[col]
                _sim_end = sim_end_[col]
                if _sim_start >= _sim_end:
                    continue
                _weights = flex_select_1d_pc_nb(weights_, col)

                for i in range(_sim_start, _sim_end):
                    _cash_deposits = flex_select_nb(cash_deposits_raw_, i, group)
                    if split_shared:
                        if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                            out[i, col] = _weights * _cash_deposits / (to_col - from_col)
                        else:
                            out[i, col] = _cash_deposits / (to_col - from_col)
                    else:
                        if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                            out[i, col] = _weights * _cash_deposits
                        else:
                            out[i, col] = _cash_deposits
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        cash_deposits_raw=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        weights=base_ch.flex_1d_array_gl_slicer,
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_deposits_grouped_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    cash_deposits_raw: tp.FlexArray2dLike = 0.0,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate cash deposit series aggregated by group.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        cash_sharing (bool): Flag indicating whether cash is shared among assets of the same group.
        cash_deposits_raw (FlexArray2dLike): Raw cash deposits.

            Provided as a scalar, or per row, column, or element.
        weights (Optional[FlexArray1dLike]): Weights applied to cash deposits.

            Provided as a scalar or per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group with cash sharing.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group with cash sharing.

    Returns:
        Array2d: 2D array containing the grouped cash deposit series.

    !!! tip
        This function is parallelizable.
    """
    cash_deposits_raw_ = to_2d_array_nb(np.asarray(cash_deposits_raw))
    if weights is None:
        weights_ = np.full(target_shape[1], np.nan, dtype=float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(float_))

    out = np.full((target_shape[0], len(group_lens)), np.nan, dtype=float_)

    if cash_sharing:
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        sim_start_, sim_end_ = generic_nb.prepare_grouped_sim_range_nb(
            target_shape=target_shape,
            group_lens=group_lens,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        for group in prange(len(group_lens)):
            _sim_start = sim_start_[group]
            _sim_end = sim_end_[group]
            if _sim_start >= _sim_end:
                continue
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]

            for i in range(_sim_start, _sim_end):
                _cash_deposits = flex_select_nb(cash_deposits_raw_, i, group)
                if np.isnan(_cash_deposits) or _cash_deposits == 0:
                    out[i, group] = _cash_deposits
                    continue
                group_weight = 0.0
                for col in range(from_col, to_col):
                    _weights = flex_select_1d_pc_nb(weights_, col)
                    if not np.isnan(group_weight) and not np.isnan(_weights):
                        group_weight += _weights
                    else:
                        group_weight = np.nan
                        break
                if not np.isnan(group_weight):
                    group_weight /= group_lens[group]
                if not np.isnan(group_weight) and not is_close_nb(group_weight, 1.0):
                    out[i, group] = group_weight * _cash_deposits
                else:
                    out[i, group] = _cash_deposits
    else:
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
            sim_shape=target_shape,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        for group in prange(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]

            for col in range(from_col, to_col):
                _sim_start = sim_start_[col]
                _sim_end = sim_end_[col]
                if _sim_start >= _sim_end:
                    continue
                _weights = flex_select_1d_pc_nb(weights_, col)

                for i in range(_sim_start, _sim_end):
                    _cash_deposits = flex_select_nb(cash_deposits_raw_, i, col)
                    if np.isnan(out[i, group]):
                        out[i, group] = 0.0
                    if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                        out[i, group] += _weights * _cash_deposits
                    else:
                        out[i, group] += _cash_deposits
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        cash_earnings_raw=base_ch.FlexArraySlicer(axis=1),
        weights=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_earnings_nb(
    target_shape: tp.Shape,
    cash_earnings_raw: tp.FlexArray2dLike = 0.0,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate cash earning series per column by applying weight adjustments and simulation range filters.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        cash_earnings_raw (FlexArray2dLike): Raw cash earnings.

            Provided as a scalar, or per row, column, or element.
        weights (FlexArray1dLike): Weight factors for scaling cash earnings.

            Provided as a scalar or per column.
        sim_start (FlexArray1dLike): Start indices of the simulation range for each column.

            Provided as a scalar or per column.
        sim_end (FlexArray1dLike): End indices of the simulation range for each column.

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array with calculated cash earnings per column.

    !!! tip
        This function is parallelizable.
    """
    cash_earnings_raw_ = to_2d_array_nb(np.asarray(cash_earnings_raw))
    if weights is None:
        weights_ = np.full(target_shape[1], np.nan, dtype=float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(float_))

    out = np.full(target_shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(target_shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _weights = flex_select_1d_pc_nb(weights_, col)

        for i in range(_sim_start, _sim_end):
            _cash_earnings = flex_select_nb(cash_earnings_raw_, i, col)
            if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                out[i, col] = _weights * _cash_earnings
            else:
                out[i, col] = _cash_earnings
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_earnings_raw=base_ch.flex_array_gl_slicer,
        weights=base_ch.flex_1d_array_gl_slicer,
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_earnings_grouped_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_earnings_raw: tp.FlexArray2dLike = 0.0,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate aggregated cash earning series per group by summing weighted cash earnings of grouped columns.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        cash_earnings_raw (FlexArray2dLike): Raw cash earnings.

            Provided as a scalar, or per row, column, or element.
        weights (FlexArray1dLike): Weight factors for scaling cash earnings.

            Provided as a scalar or per column.
        sim_start (FlexArray1dLike): Start indices of the simulation range for each column.

            Provided as a scalar or per column.
        sim_end (FlexArray1dLike): End indices of the simulation range for each column.

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array with aggregated cash earnings per group.

    !!! tip
        This function is parallelizable.
    """
    cash_earnings_raw_ = to_2d_array_nb(np.asarray(cash_earnings_raw))
    if weights is None:
        weights_ = np.full(target_shape[1], np.nan, dtype=float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(float_))

    out = np.full((target_shape[0], len(group_lens)), np.nan, dtype=float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue
            _weights = flex_select_1d_pc_nb(weights_, col)

            for i in range(_sim_start, _sim_end):
                _cash_earnings = flex_select_nb(cash_earnings_raw_, i, col)
                if np.isnan(out[i, group]):
                    out[i, group] = 0.0
                if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                    out[i, group] += _weights * _cash_earnings
                else:
                    out[i, group] += _cash_earnings
    return out


@register_jitted(cache=True)
def get_free_cash_diff_nb(
    position_before: float,
    position_now: float,
    debt_now: float,
    price: float,
    fees: float,
) -> tp.Tuple[float, float]:
    """Calculate updated debt and free cash difference after a position change.

    Args:
        position_before (float): Position amount before the transaction.
        position_now (float): Position amount after the transaction.
        debt_now (float): Debt amount after the transaction.
        price (float): Asset price used for computing the transaction value.
        fees (float): Fraction of the order value charged as fee.

    Returns:
        Tuple[float, float]: Tuple containing the updated debt and the free cash difference.
    """
    size = add_nb(position_now, -position_before)
    final_cash = -size * price - fees
    if is_close_nb(size, 0):
        new_debt = debt_now
        free_cash_diff = 0.0
    elif size > 0:
        if position_before < 0:
            if position_now < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_before)
            avg_entry_price = debt_now / abs(position_before)
            debt_diff = short_size * avg_entry_price
            new_debt = add_nb(debt_now, -debt_diff)
            free_cash_diff = add_nb(2 * debt_diff, final_cash)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    else:
        if position_now < 0:
            if position_before < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_now)
            short_value = short_size * price
            new_debt = debt_now + short_value
            free_cash_diff = add_nb(final_cash, -2 * short_value)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    return new_debt, free_cash_diff


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        free=None,
        cash_earnings=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_flow_nb(
    target_shape: tp.Shape,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    free: bool = False,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute cash flow series per column.

    Calculates the cash flow for each column based on order records, cash earnings,
    and simulation ranges. If `free` is True, computes free cash flow differences;
    otherwise, computes standard cash flow.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        free (bool): Flag indicating whether to use free cash flow.
        cash_earnings (FlexArray2dLike): Cash earnings or losses at the end of each bar.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array representing the cash flow series.

    !!! tip
        This function is parallelizable.
    """
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))

    out = np.full(target_shape, np.nan, dtype=float_)
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in range(target_shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            out[i, col] = flex_select_nb(cash_earnings_, i, col)

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = 0.0
        debt_now = 0.0

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            i = order_record["idx"]
            side = order_record["side"]
            size = order_record["size"]
            price = order_record["price"]
            fees = order_record["fees"]

            if side == OrderSide.Sell:
                size *= -1
            position_before = position_now
            position_now = add_nb(position_now, size)
            if free:
                debt_now, cash_flow = get_free_cash_diff_nb(
                    position_before=position_before,
                    position_now=position_now,
                    debt_now=debt_now,
                    price=price,
                    fees=fees,
                )
            else:
                cash_flow = -size * price - fees
            out[i, col] = add_nb(out[i, col], cash_flow)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        cash_flow=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_flow_grouped_nb(
    cash_flow: tp.Array2d,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute aggregated cash flow series per group.

    Aggregates cash flow data from the input array into groups determined by the provided
    group lengths. The function sums cash flow values across columns within each group over
    the simulation range.

    Args:
        cash_flow (Array2d): Array of cash flow values.
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Aggregated cash flow series for each group.

    !!! tip
        This function is parallelizable.
    """
    out = np.full((cash_flow.shape[0], len(group_lens)), np.nan, dtype=float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=cash_flow.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue

            for i in range(_sim_start, _sim_end):
                if np.isnan(out[i, group]):
                    out[i, group] = 0.0
                out[i, group] += cash_flow[i, col]

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="free_cash_flow", axis=1),
    arg_take_spec=dict(
        init_cash_raw=None,
        free_cash_flow=ch.ArraySlicer(axis=1),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def align_init_cash_nb(
    init_cash_raw: int,
    free_cash_flow: tp.Array2d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Align initial cash based on the maximum negative free cash flow per column or group.

    Evaluates the cumulative free cash flow by summing free cash flow and cash deposits
    over the simulation range for each column. The aligned initial cash is set to cover the
    maximum negative cash requirement. If `init_cash_raw` equals auto-align mode, all columns
    are adjusted to the maximum required cash across columns.

    Args:
        init_cash_raw (int): Raw initial cash value or mode indicator for auto alignment.
        free_cash_flow (Array2d): Array of free cash flow values.
        cash_deposits (FlexArray2dLike): Cash deposits or withdrawals at the beginning of each bar.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Aligned initial cash values for each column.

    !!! tip
        This function is parallelizable.
    """
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full(free_cash_flow.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=free_cash_flow.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(free_cash_flow.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        free_cash = 0.0
        min_req_cash = np.inf

        for i in range(_sim_start, _sim_end):
            free_cash = add_nb(free_cash, free_cash_flow[i, col])
            free_cash = add_nb(free_cash, flex_select_nb(cash_deposits_, i, col))
            if free_cash < min_req_cash:
                min_req_cash = free_cash

        if min_req_cash < 0:
            out[col] = np.abs(min_req_cash)
        else:
            out[col] = 1.0

    if init_cash_raw == InitCashMode.AutoAlign:
        out = np.full(out.shape, np.max(out))
    return out


@register_jitted(cache=True)
def init_cash_nb(
    init_cash_raw: tp.FlexArray1d,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    split_shared: bool = False,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute initial cash for each column based on initial cash values, group lengths, cash sharing, and weights.

    Args:
        init_cash_raw (FlexArray1d): Raw initial cash values.
        group_lens (GroupLens): Array defining the number of columns in each group.
        cash_sharing (bool): Flag indicating whether cash is shared among assets of the same group.
        split_shared (bool): Whether to split shared cash equally among columns in a group.
        weights (Optional[FlexArray1dLike]): Optional weights to adjust the initial cash.

            Provided as a scalar or per column.

    Returns:
        Array1d: Computed initial cash values per column.
    """
    out = np.empty(np.sum(group_lens), dtype=float_)
    if weights is None:
        weights_ = np.full(group_lens.sum(), np.nan, dtype=float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(float_))

    if not cash_sharing:
        for col in range(out.shape[0]):
            _init_cash = flex_select_1d_pc_nb(init_cash_raw, col)
            _weights = flex_select_1d_pc_nb(weights_, col)
            if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                out[col] = _weights * _init_cash
            else:
                out[col] = _init_cash
    else:
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            group_len = to_col - from_col
            _init_cash = flex_select_1d_pc_nb(init_cash_raw, group)
            for col in range(from_col, to_col):
                _weights = flex_select_1d_pc_nb(weights_, col)
                if split_shared:
                    if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                        out[col] = _weights * _init_cash / group_len
                    else:
                        out[col] = _init_cash / group_len
                else:
                    if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                        out[col] = _weights * _init_cash
                    else:
                        out[col] = _init_cash
            from_col = to_col
    return out


@register_jitted(cache=True)
def init_cash_grouped_nb(
    init_cash_raw: tp.FlexArray1d,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute initial cash for each group based on raw initial cash values, group lengths, cash sharing, and weights.

    Args:
        init_cash_raw (FlexArray1d): Raw initial cash values.
        group_lens (GroupLens): Array defining the number of columns in each group.
        cash_sharing (bool): Flag indicating whether cash is shared among assets of the same group.
        weights (Optional[FlexArray1dLike]): Optional weights to adjust the initial cash.

            Provided as a scalar or per column.

    Returns:
        Array1d: Computed initial cash values per group.
    """
    out = np.empty(group_lens.shape, dtype=float_)
    if weights is None:
        weights_ = np.full(group_lens.sum(), np.nan, dtype=float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(float_))

    if cash_sharing:
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            _init_cash = flex_select_1d_pc_nb(init_cash_raw, group)
            group_weight = 0.0
            for col in range(from_col, to_col):
                _weights = flex_select_1d_pc_nb(weights_, col)
                if not np.isnan(group_weight) and not np.isnan(_weights):
                    group_weight += _weights
                else:
                    group_weight = np.nan
                    break
            if not np.isnan(group_weight):
                group_weight /= group_lens[group]
            if not np.isnan(group_weight) and not is_close_nb(group_weight, 1.0):
                out[group] = group_weight * _init_cash
            else:
                out[group] = _init_cash
            from_col = to_col
    else:
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            cash_sum = 0.0
            for col in range(from_col, to_col):
                _init_cash = flex_select_1d_pc_nb(init_cash_raw, col)
                _weights = flex_select_1d_pc_nb(weights_, col)
                if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                    cash_sum += _weights * _init_cash
                else:
                    cash_sum += _init_cash
            out[group] = cash_sum
            from_col = to_col
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="cash_flow", axis=1),
    arg_take_spec=dict(
        cash_flow=ch.ArraySlicer(axis=1),
        init_cash=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_nb(
    cash_flow: tp.Array2d,
    init_cash: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute cash series per column or group using cash flow, initial cash, cash deposits, and simulation range.

    Args:
        cash_flow (Array2d): 2D array of cash flow values.
        init_cash (FlexArray1d): Initial cash amounts per column.
        cash_deposits (FlexArray2dLike): Cash deposits or withdrawals at the beginning of each bar.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array of computed cash series.

    !!! tip
        This function is parallelizable.
    """
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full(cash_flow.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=cash_flow.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(cash_flow.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        cash_now = flex_select_1d_pc_nb(init_cash, col)

        for i in range(_sim_start, _sim_end):
            cash_now = add_nb(cash_now, flex_select_nb(cash_deposits_, i, col))
            cash_now = add_nb(cash_now, cash_flow[i, col])
            out[i, col] = cash_now
    return out


# ############# Value ############# #


@register_jitted(cache=True)
def init_position_value_nb(
    n_cols: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Compute the initial position value for each column as the product of initial position and initial price.

    Args:
        n_cols (int): Number of columns.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

    Returns:
        Array1d: Computed initial position values per column.
    """
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    out = np.empty(n_cols, dtype=float_)

    for col in range(n_cols):
        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))
        if _init_position == 0:
            out[col] = 0.0
        else:
            out[col] = _init_position * _init_price
    return out


@register_jitted(cache=True)
def init_position_value_grouped_nb(
    group_lens: tp.GroupLens,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Compute the aggregated initial position value per group as the sum of the product of initial position and price for columns within the group.

    Args:
        group_lens (GroupLens): Array defining the number of columns in each group.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

    Returns:
        Array1d: Computed initial position values per group.
    """
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    out = np.full(len(group_lens), 0.0, dtype=float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _init_position = float(flex_select_1d_pc_nb(init_position_, col))
            _init_price = float(flex_select_1d_pc_nb(init_price_, col))
            if _init_position != 0:
                out[group] += _init_position * _init_price
    return out


@register_jitted(cache=True)
def init_value_nb(init_position_value: tp.Array1d, init_cash: tp.FlexArray1d) -> tp.Array1d:
    """Compute the total initial value per column or group by summing initial cash and initial position value.

    Args:
        init_position_value (Array1d): Initial position per column.
        init_cash (FlexArray1d): Initial cash amounts per column.

    Returns:
        Array1d: Computed total initial values per column or group.
    """
    out = np.empty(len(init_position_value), dtype=float_)

    for col in range(len(init_position_value)):
        _init_cash = flex_select_1d_pc_nb(init_cash, col)
        out[col] = _init_cash + init_position_value[col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        assets=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_value_nb(
    close: tp.Array2d,
    assets: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute asset value series per column.

    Args:
        close (Array2d): Price series per column.
        assets (Array2d): Array of asset positions per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Asset value series computed as the product of price and asset quantity
            when assets are non-zero; otherwise, zero.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(close.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=close.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(close.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            if assets[i, col] == 0:
                out[i, col] = 0.0
            else:
                out[i, col] = close[i, col] * assets[i, col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        asset_value=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_value_grouped_nb(
    asset_value: tp.Array2d,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute grouped asset value series.

    Args:
        asset_value (Array2d): Asset value series per column.
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Grouped asset value series computed by summing column values within each group.

    !!! tip
        This function is parallelizable.
    """
    out = np.full((asset_value.shape[0], len(group_lens)), np.nan, dtype=float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue

            for i in range(_sim_start, _sim_end):
                if np.isnan(out[i, group]):
                    out[i, group] = 0.0
                out[i, group] += asset_value[i, col]

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="cash", axis=1),
    arg_take_spec=dict(
        cash=ch.ArraySlicer(axis=1),
        asset_value=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_nb(
    cash: tp.Array2d,
    asset_value: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute value series per column or group.

    Args:
        cash (Array2d): Cash series per column.
        asset_value (Array2d): Asset value series per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Value series computed as the sum of cash and asset value for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(cash.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=cash.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(cash.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            out[i, col] = cash[i, col] + asset_value[i, col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_value", axis=1),
    arg_take_spec=dict(
        asset_value=ch.ArraySlicer(axis=1),
        value=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def gross_exposure_nb(
    asset_value: tp.Array2d,
    value: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute gross exposure series per column.

    Args:
        asset_value (Array2d): Asset value series per column.
        value (Array2d): Total value series per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Gross exposure series calculated as the absolute ratio of
            asset value to total value for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(asset_value.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(asset_value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            if value[i, col] == 0:
                out[i, col] = np.nan
            else:
                out[i, col] = abs(asset_value[i, col] / value[i, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="long_exposure", axis=1),
    arg_take_spec=dict(
        long_exposure=ch.ArraySlicer(axis=1),
        short_exposure=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def net_exposure_nb(
    long_exposure: tp.Array2d,
    short_exposure: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute net exposure series per column.

    Args:
        long_exposure (Array2d): Long exposure series per column.
        short_exposure (Array2d): Short exposure series per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Net exposure series calculated as the difference between
            long and short exposures for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(long_exposure.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=long_exposure.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(long_exposure.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            out[i, col] = long_exposure[i, col] - short_exposure[i, col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        asset_value=base_ch.array_gl_slicer,
        value=ch.ArraySlicer(axis=1),
        group_lens=ch.ArraySlicer(axis=0),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def allocations_nb(
    asset_value: tp.Array2d,
    value: tp.Array2d,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate allocations for each column by computing the ratio of asset values to
    group totals over the simulation period.

    Args:
        asset_value (Array2d): Matrix of asset values.
        value (Array2d): Matrix of group total values used for normalization.
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Matrix of calculated allocations per column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(asset_value.shape, np.nan, dtype=float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue

            for i in range(_sim_start, _sim_end):
                if value[i, group] == 0:
                    out[i, col] = np.nan
                else:
                    out[i, col] = asset_value[i, col] / value[i, group]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        init_position=base_ch.FlexArraySlicer(),
        init_price=base_ch.FlexArraySlicer(),
        cash_earnings=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def total_profit_nb(
    target_shape: tp.Shape,
    close: tp.Array2d,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute total profit for each column by aggregating asset values, order records, and
    cash flows over the simulation period.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        close (Array2d): Matrix of close prices.
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.
        cash_earnings (FlexArray2dLike): Cash earnings or losses at the end of each bar.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array containing the computed total profit for each column.

    !!! tip
        This function is parallelizable.
    """
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))

    assets = np.full(target_shape[1], 0.0, dtype=float_)
    cash = np.full(target_shape[1], 0.0, dtype=float_)
    total_profit = np.full(target_shape[1], np.nan, dtype=float_)

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(target_shape[1]):
        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))
        if _init_position != 0:
            assets[col] = _init_position
            cash[col] = -_init_position * _init_price
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        for i in range(_sim_start, _sim_end):
            cash[col] += flex_select_nb(cash_earnings_, i, col)

    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        col_len = col_lens[col]
        if col_len == 0:
            if assets[col] == 0 and cash[col] == 0:
                total_profit[col] = 0.0
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            # Fill assets
            if order_record["side"] == OrderSide.Buy:
                order_size = order_record["size"]
                assets[col] = add_nb(assets[col], order_size)
            else:
                order_size = order_record["size"]
                assets[col] = add_nb(assets[col], -order_size)

            # Fill cash balance
            if order_record["side"] == OrderSide.Buy:
                order_cash = order_record["size"] * order_record["price"] + order_record["fees"]
                cash[col] = add_nb(cash[col], -order_cash)
            else:
                order_cash = order_record["size"] * order_record["price"] - order_record["fees"]
                cash[col] = add_nb(cash[col], order_cash)

        total_profit[col] = cash[col] + assets[col] * close[_sim_end - 1, col]
    return total_profit


@register_jitted(cache=True)
def total_profit_grouped_nb(total_profit: tp.Array1d, group_lens: tp.GroupLens) -> tp.Array1d:
    """Aggregate total profit over groups by summing the column profits within each group.

    Args:
        total_profit (Array1d): Array of total profits per column.
        group_lens (GroupLens): Array defining the number of columns in each group.

    Returns:
        Array1d: Array containing the aggregated total profit for each group.
    """
    out = np.empty(len(group_lens), dtype=float_)

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[group] = np.sum(total_profit[from_col:to_col])
        from_col = to_col
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="value", axis=1),
    arg_take_spec=dict(
        value=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        cash_deposits_as_input=None,
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def returns_nb(
    value: tp.Array2d,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_deposits_as_input: bool = False,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute return series per column or group.

    Args:
        value (Array2d): Array of asset values.
        init_value (FlexArray1d): Initial asset value per column.
        cash_deposits (FlexArray2dLike): Cash deposits or withdrawals at the beginning of each bar.

            Provided as a scalar, or per row, column, or element.
        cash_deposits_as_input (bool): Whether to add cash deposits to the input value.
        log_returns (bool): Flag to compute logarithmic returns.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Computed return series for each column.

    !!! tip
        This function is parallelizable.
    """
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full(value.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        input_value = flex_select_1d_pc_nb(init_value, col)

        for i in range(_sim_start, _sim_end):
            _cash_deposits = flex_select_nb(cash_deposits_, i, col)
            output_value = value[i, col]
            if cash_deposits_as_input:
                adj_input_value = input_value + _cash_deposits
                out[i, col] = returns_nb_.get_return_nb(
                    adj_input_value, output_value, log_returns=log_returns
                )
            else:
                adj_output_value = output_value - _cash_deposits
                out[i, col] = returns_nb_.get_return_nb(
                    input_value, adj_output_value, log_returns=log_returns
                )
            input_value = output_value
    return out


@register_jitted(cache=True)
def get_asset_pnl_nb(
    input_asset_value: float,
    output_asset_value: float,
    cash_flow: float,
) -> float:
    """Compute asset profit and loss from input and output asset values and cash flow.

    Args:
        input_asset_value (float): Asset value at the beginning.
        output_asset_value (float): Asset value at the end.
        cash_flow (float): Cash flow during the period.

    Returns:
        float: Calculated asset profit and loss.
    """
    return output_asset_value + cash_flow - input_asset_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_value", axis=1),
    arg_take_spec=dict(
        asset_value=ch.ArraySlicer(axis=1),
        cash_flow=ch.ArraySlicer(axis=1),
        init_position_value=base_ch.FlexArraySlicer(axis=0),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_pnl_nb(
    asset_value: tp.Array2d,
    cash_flow: tp.Array2d,
    init_position_value: tp.FlexArray1dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute asset (realized and unrealized) profit and loss series per column or group.

    Args:
        asset_value (Array2d): Array of asset values.
        cash_flow (Array2d): Array of cash flows.
        init_position_value (FlexArray1dLike): Initial position value.

            Provided as a scalar or per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Calculated asset profit and loss series for each column.

    !!! tip
        This function is parallelizable.
    """
    init_position_value_ = to_1d_array_nb(np.asarray(init_position_value))

    out = np.full(asset_value.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(asset_value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _init_position_value = flex_select_1d_pc_nb(init_position_value_, col)

        for i in range(_sim_start, _sim_end):
            if i == _sim_start:
                input_asset_value = _init_position_value
            else:
                input_asset_value = asset_value[i - 1, col]
            out[i, col] = get_asset_pnl_nb(
                input_asset_value,
                asset_value[i, col],
                cash_flow[i, col],
            )
    return out


@register_jitted(cache=True)
def get_asset_return_nb(
    input_asset_value: float,
    output_asset_value: float,
    cash_flow: float,
    log_returns: bool = False,
) -> float:
    """Compute asset return from input and output asset values and cash flow.

    Args:
        input_asset_value (float): Asset value at the beginning.
        output_asset_value (float): Asset value at the end.
        cash_flow (float): Cash flow during the period.
        log_returns (bool): Flag to compute logarithmic returns.

    Returns:
        float: Computed asset return.
    """
    if is_close_nb(input_asset_value, 0):
        input_value = -output_asset_value
        output_value = cash_flow
    else:
        input_value = input_asset_value
        output_value = output_asset_value + cash_flow
    if input_value < 0 and output_value < 0:
        return_value = -returns_nb_.get_return_nb(-input_value, -output_value, log_returns=False)
    else:
        return_value = returns_nb_.get_return_nb(input_value, output_value, log_returns=False)
    if log_returns:
        return np.log1p(return_value)
    return return_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_value", axis=1),
    arg_take_spec=dict(
        asset_value=ch.ArraySlicer(axis=1),
        cash_flow=ch.ArraySlicer(axis=1),
        init_position_value=base_ch.FlexArraySlicer(axis=0),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_returns_nb(
    asset_value: tp.Array2d,
    cash_flow: tp.Array2d,
    init_position_value: tp.FlexArray1dLike = 0.0,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute asset return series per column or group.

    Args:
        asset_value (Array2d): Array of asset values.
        cash_flow (Array2d): Array of cash flows corresponding to asset values.
        init_position_value (FlexArray1dLike): Initial position value.

            Provided as a scalar or per column.
        log_returns (bool): Flag to compute logarithmic returns.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Computed asset return series for each column.

    !!! tip
        This function is parallelizable.
    """
    init_position_value_ = to_1d_array_nb(np.asarray(init_position_value))

    out = np.full(asset_value.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(asset_value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _init_position_value = flex_select_1d_pc_nb(init_position_value_, col)

        for i in range(_sim_start, _sim_end):
            if i == _sim_start:
                input_asset_value = _init_position_value
            else:
                input_asset_value = asset_value[i - 1, col]
            out[i, col] = get_asset_return_nb(
                input_asset_value,
                asset_value[i, col],
                cash_flow[i, col],
                log_returns=log_returns,
            )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def market_value_nb(
    close: tp.Array2d,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute market value for each column based on asset prices and cash deposits.

    Args:
        close (Array2d): Asset prices with rows as bars and columns as assets.
        init_value (FlexArray1d): Initial market values for each asset.
        cash_deposits (FlexArray2dLike): Cash deposits or withdrawals at the beginning of each bar.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Computed market values for each asset over the simulation period.

    !!! tip
        This function is parallelizable.
    """
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full(close.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=close.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(close.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        curr_value = flex_select_1d_pc_nb(init_value, col)

        for i in range(_sim_start, _sim_end):
            if i > _sim_start:
                curr_value *= close[i, col] / close[i - 1, col]
            curr_value += flex_select_nb(cash_deposits_, i, col)
            out[i, col] = curr_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        close=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        init_value=base_ch.FlexArraySlicer(mapper=base_ch.group_lens_mapper),
        cash_deposits=base_ch.FlexArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def market_value_grouped_nb(
    close: tp.Array2d,
    group_lens: tp.GroupLens,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute market value for each group by aggregating asset market values based on group lengths,
    asset prices, and cash deposits.

    Args:
        close (Array2d): Asset prices with rows as bars and columns as individual assets.
        group_lens (GroupLens): Array defining the number of columns in each group.
        init_value (FlexArray1d): Initial market values for each asset.
        cash_deposits (FlexArray2dLike): Cash deposits or withdrawals at the beginning of each bar.

            Provided as a scalar, or per row, column, or element.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Aggregated market values per group over time.

    !!! tip
        This function is parallelizable.
    """
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full((close.shape[0], len(group_lens)), np.nan, dtype=float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=close.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue
            curr_value = prev_value = flex_select_1d_pc_nb(init_value, col)

            for i in range(_sim_start, _sim_end):
                if i > _sim_start:
                    if not np.isnan(close[i - 1, col]):
                        prev_close = close[i - 1, col]
                        prev_value = prev_close
                    else:
                        prev_close = prev_value
                    if not np.isnan(close[i, col]):
                        curr_close = close[i, col]
                        prev_value = curr_close
                    else:
                        curr_close = prev_value
                    curr_value *= curr_close / prev_close
                curr_value += flex_select_nb(cash_deposits_, i, col)
                if np.isnan(out[i, group]):
                    out[i, group] = 0.0
                out[i, group] += curr_value
    return out


@register_jitted(cache=True)
def total_market_return_nb(
    market_value: tp.Array2d,
    input_value: tp.FlexArray1d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute total market return per column or group using final market value and initial input value.

    Args:
        market_value (Array2d): Market values over time per column or group.
        input_value (FlexArray1d): Initial input values for each column or group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Total market return for each column or group.
    """
    out = np.full(market_value.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=market_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in range(market_value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        _input_value = flex_select_1d_pc_nb(input_value, col)
        if _input_value != 0:
            out[col] = (market_value[_sim_end - 1, col] - _input_value) / _input_value
    return out
