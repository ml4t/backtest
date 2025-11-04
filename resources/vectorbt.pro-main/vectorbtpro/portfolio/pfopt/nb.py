# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for portfolio optimization."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.portfolio.enums import Direction, alloc_point_dt, alloc_range_dt
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import rescale_nb

__all__ = []


@register_jitted(cache=True)
def get_alloc_points_nb(
    filled_allocations: tp.Array2d,
    valid_only: bool = True,
    nonzero_only: bool = True,
    unique_only: bool = True,
) -> tp.Array1d:
    """Get allocation points from filled allocations.

    Args:
        filled_allocations (Array2d): 2D array of allocation values.
        valid_only (bool): Skip rows where all values are NaN.
        nonzero_only (bool): Skip rows where all values are zero.
        unique_only (bool): Skip rows that are identical to the previous valid row.

    Returns:
        Array1d: 1D array of indices representing allocation points.
    """
    out = np.empty(len(filled_allocations), dtype=int_)
    k = 0
    for i in range(filled_allocations.shape[0]):
        all_nan = True
        all_zeros = True
        all_same = True
        for col in range(filled_allocations.shape[1]):
            if not np.isnan(filled_allocations[i, col]):
                all_nan = False
            if abs(filled_allocations[i, col]) > 0:
                all_zeros = False
            if k == 0 or (
                k > 0 and filled_allocations[i, col] != filled_allocations[out[k - 1], col]
            ):
                all_same = False
        if valid_only and all_nan:
            continue
        if nonzero_only and all_zeros:
            continue
        if unique_only and all_same:
            continue
        out[k] = i
        k += 1
    return out[:k]


@register_chunkable(
    size=ch.ArraySizer(arg_query="range_starts", axis=0),
    arg_take_spec=dict(
        n_cols=None,
        range_starts=ch.ArraySlicer(axis=0),
        range_ends=ch.ArraySlicer(axis=0),
        optimize_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="row_stack",
)
@register_jitted(tags={"can_parallel"})
def optimize_meta_nb(
    n_cols: int,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    optimize_func_nb: tp.OptimizeFunc,
    *args,
) -> tp.Array2d:
    """Optimize over each index range.

    Apply the provided optimization function to each range defined by `range_starts` and `range_ends`.

    Args:
        n_cols (int): Number of columns for the output allocation.
        range_starts (Array1d): Array of starting indices for each range.
        range_ends (Array1d): Array of ending indices for each range.
        optimize_func_nb (OptimizeFunc): Callback function that accepts an index,
            start row index, end row index, and additional arguments, and returns the optimized
            allocation as a single value or an array that broadcasts to the number of columns.
        *args: Positional arguments for `optimize_func_nb`.

    Returns:
        Array2d: 2D array where each row contains the optimized allocation for a range.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((range_starts.shape[0], n_cols), dtype=float_)
    for i in prange(len(range_starts)):
        out[i, :] = optimize_func_nb(i, range_starts[i], range_ends[i], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="index_points", axis=0),
    arg_take_spec=dict(
        n_cols=None,
        index_points=ch.ArraySlicer(axis=0),
        allocate_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="row_stack",
)
@register_jitted(tags={"can_parallel"})
def allocate_meta_nb(
    n_cols: int,
    index_points: tp.Array1d,
    allocate_func_nb: tp.AllocateFunc,
    *args,
) -> tp.Array2d:
    """Allocate by mapping each index point.

    Apply the provided allocation function to each index point in `index_points`.

    Args:
        n_cols (int): Number of columns for the allocation.
        index_points (Array1d): Array of index points for allocation.
        allocate_func_nb (AllocateFunc): Callback function that accepts an index,
            index point, and additional arguments, and returns the allocation as a single value
            or an array that broadcasts to the number of columns.
        *args: Positional arguments for `allocate_func_nb`.

    Returns:
        Array2d: 2D array where each row represents the allocation for an index point.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((index_points.shape[0], n_cols), dtype=float_)
    for i in prange(len(index_points)):
        out[i, :] = allocate_func_nb(i, index_points[i], *args)
    return out


@register_jitted(cache=True)
def pick_idx_allocate_func_nb(i: int, index_point: int, allocations: tp.Array2d) -> tp.Array1d:
    """Pick the allocation at an absolute index.

    Args:
        i (int): Absolute index in the allocation array.
        index_point (int): Index from which to pick the allocation.
        allocations (Array2d): 2D array of allocation values.

    Returns:
        Array1d: Allocation corresponding to the specified absolute index.
    """
    return allocations[i]


@register_jitted(cache=True)
def pick_point_allocate_func_nb(i: int, index_point: int, allocations: tp.Array2d) -> tp.Array1d:
    """Pick the allocation at the given index point.

    Args:
        i (int): Allocation counter (not used in the current implementation).
        index_point (int): Index from which to pick the allocation.
        allocations (Array2d): 2D array of allocation values.

    Returns:
        Array1d: Allocation corresponding to the specified index point.
    """
    return allocations[index_point]


@register_jitted(cache=True)
def random_allocate_func_nb(
    i: int,
    index_point: int,
    n_cols: int,
    direction: int = Direction.LongOnly,
    n: tp.Optional[int] = None,
) -> tp.Array1d:
    """Generate a random normalized allocation.

    The function creates an allocation array with random weights for each column.
    Weights are generated using a uniform distribution and are assigned signs based
    on the specified `direction`. If `n` is provided, a random subset of columns is
    selected for weight assignment; otherwise, every column is assigned a weight.
    The weights for positive and negative values are normalized separately to sum up to one.
    Any unassigned weights are set to 0.

    Args:
        i (int): Iteration index.
        index_point (int): Index from which to pick the allocation.
        n_cols (int): Number of columns for the allocation.
        direction (int): Direction for weight assignment.

            See `vectorbtpro.portfolio.enums.Direction`.
        n (Optional[int]): Number of columns to assign random weights.

            If None, assign weights to all columns.

    Returns:
        Array1d: 1D array of normalized random allocation weights.
    """
    weights = np.full(n_cols, np.nan, dtype=float_)
    pos_sum = 0
    neg_sum = 0
    if n is None:
        for c in range(n_cols):
            w = np.random.uniform(0, 1)
            if direction == Direction.ShortOnly:
                w = -w
            elif direction == Direction.Both:
                if np.random.randint(0, 2) == 0:
                    w = -w
            if w >= 0:
                pos_sum += w
            else:
                neg_sum += abs(w)
            weights[c] = w
    else:
        rand_indices = np.random.choice(n_cols, size=n, replace=False)
        for k in range(len(rand_indices)):
            w = np.random.uniform(0, 1)
            if direction == Direction.ShortOnly:
                w = -w
            elif direction == Direction.Both:
                if np.random.randint(0, 2) == 0:
                    w = -w
            if w >= 0:
                pos_sum += w
            else:
                neg_sum += abs(w)
            weights[rand_indices[k]] = w
    for c in range(n_cols):
        if not np.isnan(weights[c]):
            if weights[c] >= 0:
                if pos_sum > 0:
                    weights[c] = weights[c] / pos_sum
            else:
                if neg_sum > 0:
                    weights[c] = weights[c] / neg_sum
        else:
            weights[c] = 0.0
    return weights


@register_jitted(cache=True)
def prepare_alloc_points_nb(
    index_points: tp.Array1d,
    allocations: tp.Array2d,
    group: int,
) -> tp.Tuple[tp.RecordArray, tp.Array2d]:
    """Prepare allocation points from index points and allocation values.

    Iterate over allocation rows, skipping rows where all values are NaN.
    If consecutive rows share the same index point, update the allocation at
    the previous record with the new values; otherwise, register a new allocation point.

    Args:
        index_points (Array1d): Array of index points for allocation.
        allocations (Array2d): 2D array of allocation values.
        group (int): Identifier for the allocation group.

    Returns:
        Tuple[RecordArray, Array2d]: Tuple containing:

            * Record array of prepared allocation points (dtype `vectorbtpro.portfolio.enums.alloc_point_dt`).
            * 2D array of updated allocation values.
    """
    alloc_points = np.empty_like(index_points, dtype=alloc_point_dt)
    new_allocations = np.empty_like(allocations)
    k = 0
    for i in range(allocations.shape[0]):
        all_nan = True
        for col in range(allocations.shape[1]):
            if not np.isnan(allocations[i, col]):
                all_nan = False
                break
        if all_nan:
            continue
        if k > 0 and alloc_points["alloc_idx"][k - 1] == index_points[i]:
            new_allocations[k - 1] = allocations[i]
        else:
            alloc_points["id"][k] = k
            alloc_points["col"][k] = group
            alloc_points["alloc_idx"][k] = index_points[i]
            new_allocations[k] = allocations[i]
            k += 1
    return alloc_points[:k], new_allocations[:k]


@register_jitted(cache=True)
def prepare_alloc_ranges_nb(
    start_idx: tp.Array1d,
    end_idx: tp.Array1d,
    alloc_idx: tp.Array1d,
    status: tp.Array1d,
    allocations: tp.Array2d,
    group: int,
) -> tp.Tuple[tp.RecordArray, tp.Array2d]:
    """Prepare allocation ranges.

    Process allocation arrays to consolidate contiguous ranges and filter duplicate allocation indices.

    Args:
        start_idx (Array1d): Array of starting indices for each allocation.
        end_idx (Array1d): Array of ending indices for each allocation.
        alloc_idx (Array1d): Array of allocation indices.
        status (Array1d): Array of status values corresponding to each allocation.
        allocations (Array2d): 2D array containing allocation values.
        group (int): Group identifier used to assign the corresponding column.

    Returns:
        Tuple[RecordArray, Array2d]: Tuple containing:

            * Record array of allocation ranges (dtype `vectorbtpro.portfolio.enums.alloc_point_dt`).
            * 2D array of processed allocation values.
    """
    alloc_ranges = np.empty_like(alloc_idx, dtype=alloc_range_dt)
    new_allocations = np.empty_like(allocations)
    k = 0
    for i in range(allocations.shape[0]):
        all_nan = True
        for col in range(allocations.shape[1]):
            if not np.isnan(allocations[i, col]):
                all_nan = False
                break
        if all_nan:
            continue
        if k > 0 and alloc_ranges["alloc_idx"][k - 1] == alloc_idx[i]:
            new_allocations[k - 1] = allocations[i]
        else:
            alloc_ranges["id"][k] = k
            alloc_ranges["col"][k] = group
            alloc_ranges["start_idx"][k] = start_idx[i]
            alloc_ranges["end_idx"][k] = end_idx[i]
            alloc_ranges["alloc_idx"][k] = alloc_idx[i]
            alloc_ranges["status"][k] = status[i]
            new_allocations[k] = allocations[i]
            k += 1
    return alloc_ranges[:k], new_allocations[:k]


@register_jitted(cache=True)
def rescale_allocations_nb(allocations: tp.Array2d, to_range: tp.Tuple[float, float]) -> tp.Array2d:
    """Rescale allocations to a new scale.

    Positive and negative weights are rescaled separately.

    Args:
        allocations (Array2d): 2D array containing allocation values.
        to_range (Tuple[float, float]): Target value range.

    Returns:
        Array2d: 2D array of allocations rescaled to the specified range.
    """
    new_min, new_max = to_range
    if np.isnan(new_min) or np.isinf(new_min):
        raise ValueError("Minimum of the new scale must be finite")
    if np.isnan(new_max) or np.isinf(new_max):
        raise ValueError("Maximum of the new scale must be finite")
    if new_min >= new_max:
        raise ValueError("Minimum cannot be equal to or higher than maximum")
    out = np.empty_like(allocations, dtype=float_)

    for i in range(allocations.shape[0]):
        all_nan = True
        all_zero = True
        pos_sum = 0.0
        neg_sum = 0.0
        for col in range(allocations.shape[1]):
            if np.isnan(allocations[i, col]):
                continue
            all_nan = False
            if allocations[i, col] > 0:
                all_zero = False
                pos_sum += allocations[i, col]
            elif allocations[i, col] < 0:
                all_zero = False
                neg_sum += abs(allocations[i, col])
        if all_nan:
            out[i] = np.nan
            continue
        if all_zero:
            out[i] = 0.0
            continue
        if new_max <= 0 and pos_sum > 0:
            raise ValueError("Cannot rescale positive weights to a negative scale")
        if new_min >= 0 and neg_sum > 0:
            raise ValueError("Cannot rescale negative weights to a positive scale")

        for col in range(allocations.shape[1]):
            if np.isnan(allocations[i, col]):
                out[i, col] = np.nan
                continue
            if allocations[i, col] > 0:
                out[i, col] = rescale_nb(
                    allocations[i, col] / pos_sum, (0.0, 1.0), (max(0.0, new_min), new_max)
                )
            elif allocations[i, col] < 0:
                out[i, col] = rescale_nb(
                    abs(allocations[i, col]) / neg_sum, (0.0, 1.0), (min(new_max, 0.0), new_min)
                )
            else:
                out[i, col] = 0.0

    return out
