# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with arrays."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []


def is_sorted(arr: tp.Array1d) -> bool:
    """Return whether the given array is sorted in non-decreasing order.

    Args:
        arr (Array1d): Input array.

    Returns:
        bool: True if the array is sorted, False otherwise.
    """
    return np.all(arr[:-1] <= arr[1:])


@register_jitted(cache=True)
def is_sorted_nb(arr: tp.Array1d) -> bool:
    """Return whether the given array is sorted in non-decreasing order using a Numba-compiled implementation.

    Args:
        arr (Array1d): Input array.

    Returns:
        bool: True if the array is sorted, False otherwise.
    """
    for i in range(arr.size - 1):
        if arr[i + 1] < arr[i]:
            return False
    return True


def is_range(arr: tp.Array1d) -> bool:
    """Return whether the given array represents a consecutive integer sequence.

    Args:
        arr (Array1d): Input array.

    Returns:
        bool: True if the array is a consecutive integer sequence, False otherwise.
    """
    return np.all(np.diff(arr) == 1)


@register_jitted(cache=True)
def is_range_nb(arr: tp.Array1d) -> bool:
    """Return whether the given array represents a consecutive integer sequence using a Numba-compiled implementation.

    Args:
        arr (Array1d): Input array.

    Returns:
        bool: True if the array is a consecutive integer sequence, False otherwise.
    """
    for i in range(arr.size):
        if arr[i] != arr[0] + i:
            return False
    return True


@register_jitted(cache=True)
def insert_argsort_nb(A: tp.Array1d, I: tp.Array1d) -> None:
    """Sort the array and update its index array using an insertion sort algorithm.

    This in-memory, non-recursive approach is optimized for small arrays.

    Args:
        A (Array1d): Array of values to sort.
        I (Array1d): Array of indices to reorder alongside A.

    Returns:
        None: Function modifies the input arrays in place.
    """
    for j in range(1, len(A)):
        A_j = A[j]
        I_j = I[j]
        i = j - 1
        while i >= 0 and (A[i] > A_j or np.isnan(A[i])):
            A[i + 1] = A[i]
            I[i + 1] = I[i]
            i = i - 1
        A[i + 1] = A_j
        I[i + 1] = I_j


def get_ranges_arr(starts: tp.ArrayLike, ends: tp.ArrayLike) -> tp.Array1d:
    """Construct an array of cumulative indices based on given start and end indices.

    Based on https://stackoverflow.com/a/37626057

    Args:
        starts (ArrayLike): Starting indices.
        ends (ArrayLike): Ending indices.

    Returns:
        Array1d: 1-dimensional array of cumulative indices.
    """
    starts_arr = np.asarray(starts)
    if starts_arr.ndim == 0:
        starts_arr = np.array([starts_arr])
    ends_arr = np.asarray(ends)
    if ends_arr.ndim == 0:
        ends_arr = np.array([ends_arr])
    starts_arr, end = np.broadcast_arrays(starts_arr, ends_arr)
    counts = ends_arr - starts_arr
    counts_csum = counts.cumsum()
    id_arr = np.ones(counts_csum[-1], dtype=int)
    id_arr[0] = starts_arr[0]
    id_arr[counts_csum[:-1]] = starts_arr[1:] - ends_arr[:-1] + 1
    return id_arr.cumsum()


@register_jitted(cache=True)
def uniform_summing_to_one_nb(n: int) -> tp.Array1d:
    """Generate an array of random floats that partition the unit interval.

    See https://stackoverflow.com/a/2640067/8141780

    Args:
        n (int): Number of random segments to generate.

    Returns:
        Array1d: Array of floats summing to one.
    """
    rand_floats = np.empty(n + 1, dtype=float_)
    rand_floats[0] = 0.0
    rand_floats[1] = 1.0
    rand_floats[2:] = np.random.uniform(0, 1, n - 1)
    rand_floats = np.sort(rand_floats)
    rand_floats = rand_floats[1:] - rand_floats[:-1]
    return rand_floats


def rescale(
    arr: tp.MaybeArray,
    from_range: tp.Tuple[float, float],
    to_range: tp.Tuple[float, float],
) -> tp.MaybeArray:
    """Return an array with values linearly rescaled from the original range to the target range.

    Args:
        arr (MaybeArray): Array to rescale.
        from_range (Tuple[float, float]): Original value range.
        to_range (Tuple[float, float]): Target value range.

    Returns:
        MaybeArray: Rescaled array.
    """
    from_min, from_max = from_range
    to_min, to_max = to_range
    from_delta = from_max - from_min
    to_delta = to_max - to_min
    return (to_delta * (arr - from_min) / from_delta) + to_min


@register_jitted(cache=True)
def rescale_nb(
    arr: tp.MaybeArray,
    from_range: tp.Tuple[float, float],
    to_range: tp.Tuple[float, float],
) -> tp.MaybeArray:
    """Return an array with values linearly rescaled from the original range to the target
    range using a Numba-compatible implementation.

    Args:
        arr (MaybeArray): Array to rescale.
        from_range (Tuple[float, float]): Original value range.
        to_range (Tuple[float, float]): Target value range.

    Returns:
        MaybeArray: Rescaled array.
    """
    from_min, from_max = from_range
    to_min, to_max = to_range
    from_delta = from_max - from_min
    to_delta = to_max - to_min
    return (to_delta * (arr - from_min) / from_delta) + to_min


def min_rel_rescale(arr: tp.Array, to_range: tp.Tuple[float, float]) -> tp.Array:
    """Return an array with values rescaled relative to the minimum value in the array.

    If all elements in the array are equal, returns an array filled with the lower bound of the target range.

    Args:
        arr (Array): Input array.
        to_range (Tuple[float, float]): Target value range.

    Returns:
        Array: Rescaled array with values adjusted relative to the minimum.
    """
    a_min = np.min(arr)
    a_max = np.max(arr)
    if a_max - a_min == 0:
        return np.full(arr.shape, to_range[0])
    from_range = (a_min, a_max)

    from_range_ratio = np.inf
    if a_min != 0:
        from_range_ratio = a_max / a_min

    to_range_ratio = to_range[1] / to_range[0]
    if from_range_ratio < to_range_ratio:
        to_range = (to_range[0], to_range[0] * from_range_ratio)
    return rescale(arr, from_range, to_range)


def max_rel_rescale(arr: tp.Array, to_range: tp.Tuple[float, float]) -> tp.Array:
    """Return an array with values rescaled relative to the maximum value in the array.

    If all elements in the array are equal, returns an array filled with the upper bound of the target range.

    Args:
        arr (Array): Input array.
        to_range (Tuple[float, float]): Target value range.

    Returns:
        Array: Rescaled array with values adjusted relative to the maximum.
    """
    a_min = np.min(arr)
    a_max = np.max(arr)
    if a_max - a_min == 0:
        return np.full(arr.shape, to_range[1])
    from_range = (a_min, a_max)

    from_range_ratio = np.inf
    if a_min != 0:
        from_range_ratio = a_max / a_min

    to_range_ratio = to_range[1] / to_range[0]
    if from_range_ratio < to_range_ratio:
        to_range = (to_range[1] / from_range_ratio, to_range[1])
    return rescale(arr, from_range, to_range)


@register_jitted(cache=True)
def rescale_float_to_int_nb(
    floats: tp.Array, int_range: tp.Tuple[float, float], total: float
) -> tp.Array:
    """Return an integer array obtained by rescaling a float array into a specified integer range.

    The function floors the rescaled values and randomly distributes any remaining difference
    to ensure the total sum matches the provided total.

    Args:
        floats (Array): Input array of floats.
        int_range (Tuple[float, float]): Target integer range.
        total (float): Desired total sum for the integer array.

    Returns:
        Array: Integer array after rescaling.
    """
    ints = np.floor(rescale_nb(floats, (0.0, 1.0), int_range))
    leftover = int(total - ints.sum())
    for i in range(leftover):
        ints[np.random.choice(len(ints))] += 1
    return ints


@register_jitted(cache=True)
def int_digit_count_nb(number: int) -> int:
    """Return the number of digits in the given integer.

    Args:
        number (int): Integer to evaluate.

    Returns:
        int: Count of digits in the integer.
    """
    out = 0
    while number != 0:
        number //= 10
        out += 1
    return out


@register_jitted(cache=True)
def hash_int_rows_nb(arr: tp.Array2d) -> tp.Array1d:
    """Return an integer hash for each row in a 2-dimensional array.

    Each hash is constructed so that digits corresponding to columns appear from left to right
    in the resulting number, making the hash unsuitable for numerical sorting.

    Args:
        arr (Array2d): 2-dimensional input array.

    Returns:
        Array1d: Array of integer hashes for each row.
    """
    out = np.full(arr.shape[0], 0, dtype=int_)
    prefix = 1
    for col in range(arr.shape[1]):
        vmax = np.nan
        for i in range(arr.shape[0]):
            if np.isnan(vmax) or arr[i, col] > vmax:
                vmax = arr[i, col]
            out[i] += arr[i, col] * prefix
        prefix *= 10 ** int_digit_count_nb(vmax)
    return out


@register_jitted(cache=True)
def index_repeating_rows_nb(arr: tp.Array2d) -> tp.Array1d:
    """Return an index array that assigns monotonically increasing numbers to repeating rows.

    Increments the index each time a row differs from the preceding unique row.

    Args:
        arr (Array2d): 2-dimensional input array.

    Returns:
        Array1d: Array of row indices.
    """
    out = np.empty(arr.shape[0], dtype=int_)
    temp = np.copy(arr[0])

    k = 0
    for i in range(arr.shape[0]):
        new_unique = False
        for col in range(arr.shape[1]):
            if arr[i, col] != temp[col]:
                if not new_unique:
                    k += 1
                    new_unique = True
                temp[col] = arr[i, col]
        out[i] = k
    return out


def build_nan_mask(*arrs: tp.Array) -> tp.Optional[tp.Array]:
    """Build a boolean mask for NaN values from one or more input arrays using an OR rule.

    Args:
        *arrs (Array): Input array(s) to check for NaN values.

    Returns:
        Optional[Array]: Boolean mask where each element is True if any corresponding element
            in the input arrays is NaN.
    """
    nan_mask = None
    for arr in arrs:
        if nan_mask is None:
            nan_mask = np.isnan(arr)
        else:
            nan_mask |= np.isnan(arr)
    if nan_mask.ndim == 2:
        nan_mask = nan_mask.any(axis=1)
    return nan_mask


def squeeze_nan(
    *arrs: tp.Array, nan_mask: tp.Optional[tp.Array1d] = None
) -> tp.Tuple[tp.Array, ...]:
    """Remove entries corresponding to True values in the provided NaN mask from input arrays.

    Args:
        *arrs (Array): Input array(s) from which to remove elements.
        nan_mask (Optional[Array1d]): Boolean mask indicating positions of NaN values.

            If None or if no True values are detected, the original arrays are returned.

    Returns:
        Tuple[Array, ...]: Tuple of arrays with entries removed where `nan_mask` is True.
    """
    if nan_mask is None or not np.any(nan_mask):
        return arrs

    new_arrs = ()
    for arr in arrs:
        new_arrs += (arr[~nan_mask],)
    return new_arrs


def unsqueeze_nan(
    *arrs: tp.Array, nan_mask: tp.Optional[tp.Array1d] = None
) -> tp.Tuple[tp.Array, ...]:
    """Insert NaN values into input arrays at positions indicated by the provided mask.

    Args:
        *arrs (Array): Input array(s) where NaN values will be inserted.
        nan_mask (Optional[Array1d]): Boolean mask indicating positions of NaN values.

            If None or if no True values are detected, the original arrays are returned.

    Returns:
        Tuple[Array, ...]: Tuple of arrays with NaN values inserted according to the mask.
    """
    if nan_mask is None or not np.any(nan_mask):
        return arrs

    new_arrs = ()
    for arr in arrs:
        if arr.ndim == 2:
            new_arr = np.full((len(nan_mask), arr.shape[1]), np.nan, dtype=float_)
        else:
            new_arr = np.full(len(nan_mask), np.nan, dtype=float_)
        new_arr[~nan_mask] = arr
        new_arrs += (new_arr,)
    return new_arrs


def cast_to_min_precision(
    arr: tp.Array,
    min_precision: tp.Union[int, str],
    float_only: bool = True,
) -> tp.Array:
    """Cast an array to at least the specified integer or floating-point precision.

    Args:
        arr (Array): Input array to cast.
        min_precision (Union[int, str]): Minimum precision specified as a number of bits or
            one of `half`, `single`, or `double`.
        float_only (bool): If True, applies casting only to floating-point types.

    Returns:
        Array: Input array cast to at least the specified precision if applicable.
    """
    if min_precision is None:
        return arr
    if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64):
        return arr
    if float_only and np.issubdtype(arr.dtype, np.integer):
        return arr
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        if isinstance(min_precision, str):
            if min_precision == "half":
                min_precision = 16
            elif min_precision == "single":
                min_precision = 32
            elif min_precision == "double":
                min_precision = 64
            else:
                raise ValueError(
                    "Only 'half', 'single', and 'double' max precisions are supported"
                )
        if isinstance(min_precision, int):
            if np.issubdtype(arr.dtype, np.integer):
                prefix = "int"
            else:
                prefix = "float"
            target_dtype = np.dtype(prefix + str(min_precision))
        else:
            raise TypeError("Minimum precision must be either integer or string")
        if arr.dtype < target_dtype:
            return arr.astype(target_dtype)
    return arr


def cast_to_max_precision(
    arr: tp.Array,
    max_precision: tp.Union[int, str],
    float_only: bool = True,
    check_bounds: bool = True,
    strict: bool = True,
) -> tp.Array:
    """Cast an array to at most the specified integer or floating-point precision.

    Args:
        arr (Array): Input array to cast.
        max_precision (Union[int, str]): Maximum precision specified as a number of bits or
            one of `half`, `single`, or `double`.
        float_only (bool): If True, applies casting only to floating-point types.
        check_bounds (bool): Flag to validate that simulation positions are within bounds.
        strict (bool): If True, raises an error when values exceed the bounds of the target precision.

    Returns:
        Array: Input array cast to at most the specified precision if applicable.
    """
    if max_precision is None:
        return arr
    if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64):
        return arr
    if float_only and np.issubdtype(arr.dtype, np.integer):
        return arr
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
        if isinstance(max_precision, str):
            if max_precision == "half":
                max_precision = 16
            elif max_precision == "single":
                max_precision = 32
            elif max_precision == "double":
                max_precision = 64
            else:
                raise ValueError(
                    "Only 'half', 'single', and 'double' max precisions are supported"
                )
        if isinstance(max_precision, int):
            if np.issubdtype(arr.dtype, np.integer):
                prefix = "int"
                dtype_info = np.iinfo
            else:
                prefix = "float"
                dtype_info = np.finfo
            target_dtype = np.dtype(prefix + str(max_precision))
        else:
            raise TypeError("Maximum precision must be either integer or string")
        if arr.dtype > target_dtype:
            if check_bounds:
                min_overflow = np.min(arr) < dtype_info(target_dtype).min
                max_overflow = np.max(arr) > dtype_info(target_dtype).max
                if min_overflow or max_overflow:
                    if strict:
                        raise ValueError(
                            f"Cannot lower dtype to {target_dtype}: values out of bounds"
                        )
                    return arr
            return arr.astype(target_dtype)
    return arr


@register_jitted(cache=True)
def min_count_nb(arr: tp.Array1d) -> tp.Tuple[int, float, int]:
    """Return the index of the first occurrence, the minimum value, and the count of
    occurrences of the minimum in a 1-D array.

    Args:
        arr (Array1d): One-dimensional input array.

    Returns:
        Tuple[int, float, int]: Tuple containing the index of the first minimum,
            the minimum value, and its count.
    """
    mini = 0
    minv = arr[0]
    minc = 1
    for i in range(1, len(arr)):
        if arr[i] == minv:
            minc += 1
        elif arr[i] < minv:
            mini = i
            minv = arr[i]
            minc = 1
    return mini, minv, minc


@register_jitted(cache=True)
def max_count_nb(arr: tp.Array1d) -> tp.Tuple[int, float, int]:
    """Return the index of the first occurrence, the maximum value, and the count of
        occurrences of the maximum in a 1-D array.

    Args:
        arr (Array1d): One-dimensional input array.

    Returns:
        Tuple[int, float, int]: Tuple containing the index of the first maximum,
            the maximum value, and its count.
    """
    maxi = 0
    maxv = arr[0]
    maxc = 1
    for i in range(1, len(arr)):
        if arr[i] == maxv:
            maxc += 1
        elif arr[i] > maxv:
            maxi = i
            maxv = arr[i]
            maxc = 1
    return maxi, maxv, maxc
