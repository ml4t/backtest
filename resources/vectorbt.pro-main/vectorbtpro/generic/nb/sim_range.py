# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing generic Numba-compiled functions for simulation ranges.

!!! warning
    Resolution is more flexible and may return None while preparation always returns NumPy arrays.
    Thus, use preparation, not resolution, in Numba-parallel workflows.
"""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.registries.jit_registry import register_jitted


@register_jitted(cache=True)
def resolve_sim_start_nb(
    sim_shape: tp.Shape,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve simulation start positions for simulation.

    Args:
        sim_shape (Shape): Tuple representing the dimensions of the simulation.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Optional[Array1d]: Array of simulation start positions or None.
    """
    if sim_start is None:
        if allow_none:
            return None
        return np.full(sim_shape[1], 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if not check_bounds and len(sim_start_) == sim_shape[1]:
        return sim_start_

    sim_start_out = np.empty(sim_shape[1], dtype=int_)
    can_be_none = True

    for i in range(sim_shape[1]):
        _sim_start = flex_select_1d_pc_nb(sim_start_, i)
        if _sim_start < 0:
            _sim_start = sim_shape[0] + _sim_start
        elif _sim_start > sim_shape[0]:
            _sim_start = sim_shape[0]
        sim_start_out[i] = _sim_start
        if _sim_start != 0:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return sim_start_out


@register_jitted(cache=True)
def resolve_sim_end_nb(
    sim_shape: tp.Shape,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve simulation end positions for simulation.

    Args:
        sim_shape (Shape): Tuple representing the dimensions of the simulation.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Optional[Array1d]: Array of simulation end positions or None.
    """
    if sim_end is None:
        if allow_none:
            return None
        return np.full(sim_shape[1], sim_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if not check_bounds and len(sim_end_) == sim_shape[1]:
        return sim_end_

    new_sim_end = np.empty(sim_shape[1], dtype=int_)
    can_be_none = True

    for i in range(sim_shape[1]):
        _sim_end = flex_select_1d_pc_nb(sim_end_, i)
        if _sim_end < 0:
            _sim_end = sim_shape[0] + _sim_end
        elif _sim_end > sim_shape[0]:
            _sim_end = sim_shape[0]
        new_sim_end[i] = _sim_end
        if _sim_end != sim_shape[0]:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_end


@register_jitted(cache=True)
def resolve_sim_range_nb(
    sim_shape: tp.Shape,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Optional[tp.Array1d], tp.Optional[tp.Array1d]]:
    """Resolve simulation start and end positions for simulation.

    Args:
        sim_shape (Shape): Tuple representing the dimensions of the simulation.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Tuple[Optional[Array1d], Optional[Array1d]]: Tuple containing the simulation start and end positions.
    """
    new_sim_start = resolve_sim_start_nb(
        sim_shape=sim_shape,
        sim_start=sim_start,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    new_sim_end = resolve_sim_end_nb(
        sim_shape=sim_shape,
        sim_end=sim_end,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def resolve_grouped_sim_start_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve grouped simulation start positions for simulation groups.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Optional[Array1d]: Array of simulation start positions for each group or None.
    """
    if sim_start is None:
        if allow_none:
            return None
        return np.full(len(group_lens), 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if len(sim_start_) == len(group_lens):
        if not check_bounds:
            return sim_start_
        return resolve_sim_start_nb(
            (target_shape[0], len(group_lens)),
            sim_start=sim_start_,
            allow_none=allow_none,
            check_bounds=check_bounds,
        )

    new_sim_start = np.empty(len(group_lens), dtype=int_)
    can_be_none = True

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        min_sim_start = target_shape[0]
        for col in range(from_col, to_col):
            _sim_start = flex_select_1d_pc_nb(sim_start_, col)
            if _sim_start < 0:
                _sim_start = target_shape[0] + _sim_start
            elif _sim_start > target_shape[0]:
                _sim_start = target_shape[0]
            if _sim_start < min_sim_start:
                min_sim_start = _sim_start
        new_sim_start[group] = min_sim_start
        if min_sim_start != 0:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_start


@register_jitted(cache=True)
def resolve_grouped_sim_end_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve grouped simulation end positions for simulation groups.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Optional[Array1d]: Array of simulation end positions for each group or None.
    """
    if sim_end is None:
        if allow_none:
            return None
        return np.full(len(group_lens), target_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if len(sim_end_) == len(group_lens):
        if not check_bounds:
            return sim_end_
        return resolve_sim_end_nb(
            (target_shape[0], len(group_lens)),
            sim_end=sim_end_,
            allow_none=allow_none,
            check_bounds=check_bounds,
        )

    new_sim_end = np.empty(len(group_lens), dtype=int_)
    can_be_none = True

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        max_sim_end = 0
        for col in range(from_col, to_col):
            _sim_end = flex_select_1d_pc_nb(sim_end_, col)
            if _sim_end < 0:
                _sim_end = target_shape[0] + _sim_end
            elif _sim_end > target_shape[0]:
                _sim_end = target_shape[0]
            if _sim_end > max_sim_end:
                max_sim_end = _sim_end
        new_sim_end[group] = max_sim_end
        if max_sim_end != target_shape[0]:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_end


@register_jitted(cache=True)
def resolve_grouped_sim_range_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Optional[tp.Array1d], tp.Optional[tp.Array1d]]:
    """Resolve simulation start and end for grouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Tuple[Optional[Array1d], Optional[Array1d]]: Tuple containing the resolved
            simulation start and end positions.
    """
    new_sim_start = resolve_grouped_sim_start_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    new_sim_end = resolve_grouped_sim_end_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_end=sim_end,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def resolve_ungrouped_sim_start_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve simulation start for ungrouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Optional[Array1d]: Resolved simulation start array or None.
    """
    if sim_start is None:
        if allow_none:
            return None
        return np.full(target_shape[1], 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if len(sim_start_) == target_shape[1]:
        if not check_bounds:
            return sim_start_
        return resolve_sim_start_nb(
            target_shape,
            sim_start=sim_start_,
            allow_none=allow_none,
            check_bounds=check_bounds,
        )

    new_sim_start = np.empty(target_shape[1], dtype=int_)
    can_be_none = True

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        _sim_start = flex_select_1d_pc_nb(sim_start_, group)
        if _sim_start < 0:
            _sim_start = target_shape[0] + _sim_start
        elif _sim_start > target_shape[0]:
            _sim_start = target_shape[0]
        for col in range(from_col, to_col):
            new_sim_start[col] = _sim_start
        if _sim_start != 0:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_start


@register_jitted(cache=True)
def resolve_ungrouped_sim_end_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve simulation end for ungrouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Optional[Array1d]: Resolved simulation end array or None.
    """
    if sim_end is None:
        if allow_none:
            return None
        return np.full(target_shape[1], target_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if len(sim_end_) == target_shape[1]:
        if not check_bounds:
            return sim_end_
        return resolve_sim_end_nb(
            target_shape,
            sim_end=sim_end_,
            allow_none=allow_none,
            check_bounds=check_bounds,
        )

    new_sim_end = np.empty(target_shape[1], dtype=int_)
    can_be_none = True

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        _sim_end = flex_select_1d_pc_nb(sim_end_, group)
        if _sim_end < 0:
            _sim_end = target_shape[0] + _sim_end
        elif _sim_end > target_shape[0]:
            _sim_end = target_shape[0]
        for col in range(from_col, to_col):
            new_sim_end[col] = _sim_end
        if _sim_end != target_shape[0]:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_end


@register_jitted(cache=True)
def resolve_ungrouped_sim_range_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Optional[tp.Array1d], tp.Optional[tp.Array1d]]:
    """Resolve simulation start and end for ungrouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Tuple[Optional[Array1d], Optional[Array1d]]: Tuple containing the resolved
            simulation start and end positions.
    """
    new_sim_start = resolve_ungrouped_sim_start_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    new_sim_end = resolve_ungrouped_sim_end_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_end=sim_end,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def prepare_sim_start_nb(
    sim_shape: tp.Shape,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare the simulation start array based on provided parameters.

    Args:
        sim_shape (Shape): Tuple representing the dimensions of the simulation.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Array1d: Prepared simulation start array.
    """
    if sim_start is None:
        return np.full(sim_shape[1], 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if not check_bounds and len(sim_start_) == sim_shape[1]:
        return sim_start_

    sim_start_out = np.empty(sim_shape[1], dtype=int_)

    for i in range(sim_shape[1]):
        _sim_start = flex_select_1d_pc_nb(sim_start_, i)
        if _sim_start < 0:
            _sim_start = sim_shape[0] + _sim_start
        elif _sim_start > sim_shape[0]:
            _sim_start = sim_shape[0]
        sim_start_out[i] = _sim_start

    return sim_start_out


@register_jitted(cache=True)
def prepare_sim_end_nb(
    sim_shape: tp.Shape,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare the simulation end array based on provided parameters.

    Args:
        sim_shape (Shape): Tuple representing the dimensions of the simulation.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Array1d: Prepared simulation end array.
    """
    if sim_end is None:
        return np.full(sim_shape[1], sim_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if not check_bounds and len(sim_end_) == sim_shape[1]:
        return sim_end_

    new_sim_end = np.empty(sim_shape[1], dtype=int_)

    for i in range(sim_shape[1]):
        _sim_end = flex_select_1d_pc_nb(sim_end_, i)
        if _sim_end < 0:
            _sim_end = sim_shape[0] + _sim_end
        elif _sim_end > sim_shape[0]:
            _sim_end = sim_shape[0]
        new_sim_end[i] = _sim_end

    return new_sim_end


@register_jitted(cache=True)
def prepare_sim_range_nb(
    sim_shape: tp.Shape,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Prepare both simulation start and end positions based on provided parameters.

    Args:
        sim_shape (Shape): Tuple representing the dimensions of the simulation.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing the prepared simulation start and end positions.
    """
    new_sim_start = prepare_sim_start_nb(
        sim_shape=sim_shape,
        sim_start=sim_start,
        check_bounds=check_bounds,
    )
    new_sim_end = prepare_sim_end_nb(
        sim_shape=sim_shape,
        sim_end=sim_end,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def prepare_grouped_sim_start_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare grouped simulation start positions for grouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Array1d: Array containing the prepared simulation start positions for each group.
    """
    if sim_start is None:
        return np.full(len(group_lens), 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if len(sim_start_) == len(group_lens):
        if not check_bounds:
            return sim_start_
        return prepare_sim_start_nb(
            (target_shape[0], len(group_lens)),
            sim_start=sim_start_,
            check_bounds=check_bounds,
        )

    new_sim_start = np.empty(len(group_lens), dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        min_sim_start = target_shape[0]
        for col in range(from_col, to_col):
            _sim_start = flex_select_1d_pc_nb(sim_start_, col)
            if _sim_start < 0:
                _sim_start = target_shape[0] + _sim_start
            elif _sim_start > target_shape[0]:
                _sim_start = target_shape[0]
            if _sim_start < min_sim_start:
                min_sim_start = _sim_start
        new_sim_start[group] = min_sim_start

    return new_sim_start


@register_jitted(cache=True)
def prepare_grouped_sim_end_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare grouped simulation end positions for grouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Array1d: Array containing the prepared simulation end positions for each group.
    """
    if sim_end is None:
        return np.full(len(group_lens), target_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if len(sim_end_) == len(group_lens):
        if not check_bounds:
            return sim_end_
        return prepare_sim_end_nb(
            (target_shape[0], len(group_lens)),
            sim_end=sim_end_,
            check_bounds=check_bounds,
        )

    new_sim_end = np.empty(len(group_lens), dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        max_sim_end = 0
        for col in range(from_col, to_col):
            _sim_end = flex_select_1d_pc_nb(sim_end_, col)
            if _sim_end < 0:
                _sim_end = target_shape[0] + _sim_end
            elif _sim_end > target_shape[0]:
                _sim_end = target_shape[0]
            if _sim_end > max_sim_end:
                max_sim_end = _sim_end
        new_sim_end[group] = max_sim_end

    return new_sim_end


@register_jitted(cache=True)
def prepare_grouped_sim_range_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Prepare grouped simulation start and end positions for grouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing the prepared simulation start and end positions.
    """
    new_sim_start = prepare_grouped_sim_start_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start,
        check_bounds=check_bounds,
    )
    new_sim_end = prepare_grouped_sim_end_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_end=sim_end,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def prepare_ungrouped_sim_start_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare ungrouped simulation start positions for ungrouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Array1d: Array containing the prepared simulation start positions for ungrouped data.
    """
    if sim_start is None:
        return np.full(target_shape[1], 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if len(sim_start_) == target_shape[1]:
        if not check_bounds:
            return sim_start_
        return prepare_sim_start_nb(
            target_shape,
            sim_start=sim_start_,
            check_bounds=check_bounds,
        )

    new_sim_start = np.empty(target_shape[1], dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        _sim_start = flex_select_1d_pc_nb(sim_start_, group)
        if _sim_start < 0:
            _sim_start = target_shape[0] + _sim_start
        elif _sim_start > target_shape[0]:
            _sim_start = target_shape[0]
        for col in range(from_col, to_col):
            new_sim_start[col] = _sim_start

    return new_sim_start


@register_jitted(cache=True)
def prepare_ungrouped_sim_end_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare ungrouped simulation end positions for ungrouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Array1d: Array containing the prepared simulation end positions for ungrouped data.
    """
    if sim_end is None:
        return np.full(target_shape[1], target_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if len(sim_end_) == target_shape[1]:
        if not check_bounds:
            return sim_end_
        return prepare_sim_end_nb(
            target_shape,
            sim_end=sim_end_,
            check_bounds=check_bounds,
        )

    new_sim_end = np.empty(target_shape[1], dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        _sim_end = flex_select_1d_pc_nb(sim_end_, group)
        if _sim_end < 0:
            _sim_end = target_shape[0] + _sim_end
        elif _sim_end > target_shape[0]:
            _sim_end = target_shape[0]
        for col in range(from_col, to_col):
            new_sim_end[col] = _sim_end

    return new_sim_end


@register_jitted(cache=True)
def prepare_ungrouped_sim_range_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Prepare ungrouped simulation start and end positions for ungrouped data.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column or group.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column or group.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing the prepared simulation start and
            end positions for ungrouped data.
    """
    new_sim_start = prepare_ungrouped_sim_start_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start,
        check_bounds=check_bounds,
    )
    new_sim_end = prepare_ungrouped_sim_end_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_end=sim_end,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end
