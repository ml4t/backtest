# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for splitting."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []


@register_jitted(cache=True, tags={"can_parallel"})
def split_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the overlap matrix for splits.

    Args:
        mask_arr (Array3d): 3-dimensional boolean array representing split masks.

            Each split is defined by a set of columns that are marked active if any
            element in the corresponding sub-array (across the second axis) is True.

    Returns:
        Array2d: 2-dimensional integer array where each element `[i, j]` indicates the count
            of overlapping active columns between split `i` and split `j`.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((mask_arr.shape[0], mask_arr.shape[0]), dtype=int_)
    temp_mask = np.empty((mask_arr.shape[0], mask_arr.shape[2]), dtype=np.bool_)
    for i in range(mask_arr.shape[0]):
        for m in range(mask_arr.shape[2]):
            if mask_arr[i, :, m].any():
                temp_mask[i, m] = True
            else:
                temp_mask[i, m] = False
    for i1 in prange(mask_arr.shape[0]):
        for i2 in range(mask_arr.shape[0]):
            intersection = (temp_mask[i1] & temp_mask[i2]).sum()
            out[i1, i2] = intersection
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def norm_split_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the normalized overlap matrix for splits.

    Args:
        mask_arr (Array3d): 3-dimensional boolean array representing split masks.

            Each split is identified by active columns determined by any True value
            along the second axis.

    Returns:
        Array2d: 2-dimensional float array where each element `[i, j]` is the ratio of
            the count of overlapping active columns to the total number of active columns
            (union) between split `i` and split `j`.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((mask_arr.shape[0], mask_arr.shape[0]), dtype=float_)
    temp_mask = np.empty((mask_arr.shape[0], mask_arr.shape[2]), dtype=np.bool_)
    for i in range(mask_arr.shape[0]):
        for m in range(mask_arr.shape[2]):
            if mask_arr[i, :, m].any():
                temp_mask[i, m] = True
            else:
                temp_mask[i, m] = False
    for i1 in prange(mask_arr.shape[0]):
        for i2 in range(mask_arr.shape[0]):
            intersection = (temp_mask[i1] & temp_mask[i2]).sum()
            union = (temp_mask[i1] | temp_mask[i2]).sum()
            out[i1, i2] = intersection / union
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def set_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the overlap matrix for sets.

    Args:
        mask_arr (Array3d): 3-dimensional boolean array where each set is defined
            along the second dimension.

            A set is marked active for a column if any element (across the first axis) is True.

    Returns:
        Array2d: 2-dimensional integer array where each element `[j, k]` represents the count
            of common active columns between set `j` and set `k`.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((mask_arr.shape[1], mask_arr.shape[1]), dtype=int_)
    temp_mask = np.empty((mask_arr.shape[1], mask_arr.shape[2]), dtype=np.bool_)
    for j in range(mask_arr.shape[1]):
        for m in range(mask_arr.shape[2]):
            if mask_arr[:, j, m].any():
                temp_mask[j, m] = True
            else:
                temp_mask[j, m] = False
    for j1 in prange(mask_arr.shape[1]):
        for j2 in range(mask_arr.shape[1]):
            intersection = (temp_mask[j1] & temp_mask[j2]).sum()
            out[j1, j2] = intersection
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def norm_set_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the normalized overlap matrix for sets.

    Args:
        mask_arr (Array3d): 3-dimensional boolean array where each set is defined
            along the second dimension.

            A set is marked active for a column if any element (across the first axis) is True.

    Returns:
        Array2d: 2-dimensional float array where each element `[j, k]` is the ratio of the count
            of overlapping active columns to the total active columns (union) between set `j` and set `k`.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((mask_arr.shape[1], mask_arr.shape[1]), dtype=float_)
    temp_mask = np.empty((mask_arr.shape[1], mask_arr.shape[2]), dtype=np.bool_)
    for j in range(mask_arr.shape[1]):
        for m in range(mask_arr.shape[2]):
            if mask_arr[:, j, m].any():
                temp_mask[j, m] = True
            else:
                temp_mask[j, m] = False
    for j1 in prange(mask_arr.shape[1]):
        for j2 in range(mask_arr.shape[1]):
            intersection = (temp_mask[j1] & temp_mask[j2]).sum()
            union = (temp_mask[j1] | temp_mask[j2]).sum()
            out[j1, j2] = intersection / union
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def range_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the overlap matrix for ranges.

    Args:
        mask_arr (Array3d): 3-dimensional boolean array where each range mask is defined
            by a pair of indices corresponding to the first and second dimensions.

    Returns:
        Array2d: 2-dimensional integer array with shape `(n*m, n*m)`, where each element
            represents the count of overlapping True values between two range masks.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(
        (mask_arr.shape[0] * mask_arr.shape[1], mask_arr.shape[0] * mask_arr.shape[1]), dtype=int_
    )
    for k in prange(mask_arr.shape[0] * mask_arr.shape[1]):
        i1 = k // mask_arr.shape[1]
        j1 = k % mask_arr.shape[1]
        for l in range(mask_arr.shape[0] * mask_arr.shape[1]):
            i2 = l // mask_arr.shape[1]
            j2 = l % mask_arr.shape[1]
            intersection = (mask_arr[i1, j1] & mask_arr[i2, j2]).sum()
            out[k, l] = intersection
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def norm_range_overlap_matrix_nb(mask_arr: tp.Array3d) -> tp.Array2d:
    """Compute the normalized overlap matrix for ranges.

    Args:
        mask_arr (Array3d): 3-dimensional boolean array where each range mask is defined
            by a pair of indices from the first and second dimensions.

    Returns:
        Array2d: 2-dimensional float array with shape `(n*m, n*m)` where each element is the ratio
            of the count of overlapping True values to the union of True values between two range masks.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(
        (mask_arr.shape[0] * mask_arr.shape[1], mask_arr.shape[0] * mask_arr.shape[1]), dtype=float_
    )
    for k in prange(mask_arr.shape[0] * mask_arr.shape[1]):
        i1 = k // mask_arr.shape[1]
        j1 = k % mask_arr.shape[1]
        for l in range(mask_arr.shape[0] * mask_arr.shape[1]):
            i2 = l // mask_arr.shape[1]
            j2 = l % mask_arr.shape[1]
            intersection = (mask_arr[i1, j1] & mask_arr[i2, j2]).sum()
            union = (mask_arr[i1, j1] | mask_arr[i2, j2]).sum()
            out[k, l] = intersection / union
    return out


@register_jitted(cache=True)
def split_range_by_gap_nb(range_: tp.Array1d) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Split a sequence of indices into contiguous segments.

    Args:
        range_ (Array1d): 1-dimensional array of indices.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing two arrays:

            * First array contains the starting indices of each contiguous segment.
            * Second array contains the ending indices (exclusive) of each contiguous segment.
    """
    if len(range_) == 0:
        raise ValueError("Range is empty")
    start_idxs_out = np.empty(len(range_), dtype=int_)
    stop_idxs_out = np.empty(len(range_), dtype=int_)
    start_idxs_out[0] = 0
    k = 0
    for i in range(1, len(range_)):
        if range_[i] - range_[i - 1] != 1:
            stop_idxs_out[k] = i
            k += 1
            start_idxs_out[k] = i
    stop_idxs_out[k] = len(range_)
    return start_idxs_out[: k + 1], stop_idxs_out[: k + 1]
