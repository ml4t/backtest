# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for resampling."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.datetime_nb import d_td

__all__ = []


@register_jitted(cache=True)
def date_range_nb(
    start: np.datetime64,
    end: np.datetime64,
    freq: np.timedelta64 = d_td,
    incl_left: bool = True,
    incl_right: bool = True,
) -> tp.Array1d:
    """Generate a datetime index with nanosecond precision for a given date range.

    Inspired by `pd.date_range`.

    Args:
        start (np.datetime64): Start datetime.
        end (np.datetime64): End datetime.
        freq (np.timedelta64): Frequency interval for datetime increments.
        incl_left (bool): Include the start datetime if True.
        incl_right (bool): Include the end datetime if True.

    Returns:
        Array1d: Array of datetime64[ns] values representing the index.
    """
    values_len = int(np.floor((end - start) / freq)) + 1
    values = np.empty(values_len, dtype="datetime64[ns]")
    for i in range(values_len):
        values[i] = start + i * freq
    if start == end:
        if not incl_left and not incl_right:
            values = values[1:-1]
    else:
        if not incl_left or not incl_right:
            if not incl_left and len(values) and values[0] == start:
                values = values[1:]
            if not incl_right and len(values) and values[-1] == end:
                values = values[:-1]
    return values


@register_jitted(cache=True)
def map_to_target_index_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    target_freq: tp.Optional[tp.Scalar] = None,
    before: bool = False,
    raise_missing: bool = True,
) -> tp.Array1d:
    """Map each element of `source_index` to its corresponding index in `target_index`.

    The mapping is determined by the provided `target_freq` and `before` flag. When
    `target_freq` is specified, the function restricts the mapping to a given frequency interval.
    If no valid mapping is found and `raise_missing` is True, a `ValueError` is raised; otherwise,
    the source element is mapped to -1.

    Args:
        source_index (Array1d): Array of source indices.
        target_index (Array1d): Array of target indices, which must be strictly increasing.
        target_freq (Optional[Scalar]): Frequency offset for the target index.
        before (bool): If True, include source indices preceding or equal to the target;
            otherwise, include those following or equal.
        raise_missing (bool): If True, raise an error when a source index cannot be mapped; otherwise, assign -1.

    Returns:
        Array1d: Array of integer indices from `target_index` corresponding to each element of `source_index`.
    """
    out = np.empty(len(source_index), dtype=int_)
    from_j = 0
    for i in range(len(source_index)):
        if i > 0 and source_index[i] < source_index[i - 1]:
            raise ValueError("Source index must be increasing")
        if i > 0 and source_index[i] == source_index[i - 1]:
            out[i] = out[i - 1]

        found = False
        for j in range(from_j, len(target_index)):
            if j > 0 and target_index[j] <= target_index[j - 1]:
                raise ValueError("Target index must be strictly increasing")
            if target_freq is None:
                if before and source_index[i] <= target_index[j]:
                    if j == 0 or target_index[j - 1] < source_index[i]:
                        out[i] = from_j = j
                        found = True
                        break
                if not before and target_index[j] <= source_index[i]:
                    if j == len(target_index) - 1 or source_index[i] < target_index[j + 1]:
                        out[i] = from_j = j
                        found = True
                        break
            else:
                if before and target_index[j] - target_freq < source_index[i] <= target_index[j]:
                    out[i] = from_j = j
                    found = True
                    break
                if (
                    not before
                    and target_index[j] <= source_index[i] < target_index[j] + target_freq
                ):
                    out[i] = from_j = j
                    found = True
                    break

        if not found:
            if raise_missing:
                raise ValueError("Resampling failed: cannot map some source indices")
            out[i] = -1
    return out


@register_jitted(cache=True)
def index_difference_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
) -> tp.Array1d:
    """Return the positions in `source_index` whose values are not present in `target_index`.

    The function iterates over `source_index` and identifies indices that are absent from
    `target_index`. Both arrays must be strictly increasing.

    Args:
        source_index (Array1d): Array of source indices, expected to be strictly increasing.
        target_index (Array1d): Array of target indices, which must be strictly increasing.

    Returns:
        Array1d: Array of integer positions from `source_index` that do not appear in `target_index`.
    """
    out = np.empty(len(source_index), dtype=int_)
    from_j = 0
    k = 0
    for i in range(len(source_index)):
        if i > 0 and source_index[i] <= source_index[i - 1]:
            raise ValueError("Array index must be strictly increasing")
        found = False
        for j in range(from_j, len(target_index)):
            if j > 0 and target_index[j] <= target_index[j - 1]:
                raise ValueError("Target index must be strictly increasing")
            if source_index[i] < target_index[j]:
                break
            if source_index[i] == target_index[j]:
                from_j = j
                found = True
                break
            from_j = j
        if not found:
            out[k] = i
            k += 1
    return out[:k]


@register_jitted(cache=True)
def map_index_to_source_ranges_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    target_freq: tp.Optional[tp.Scalar] = None,
    before: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Map each element of `target_index` to a range of indices in `source_index`.

    For each target index, determine the corresponding range in `source_index`. The start index
    is inclusive and the end index is exclusive. If `target_freq` is provided, the right bound is limited
    by the frequency interval; otherwise, it extends to the next target index. If no valid mapping is
    found, both start and end are set to -1.

    Args:
        source_index (Array1d): Array of source indices in increasing order.
        target_index (Array1d): Array of target indices, which must be strictly increasing.
        target_freq (Optional[Scalar]): Frequency offset for the target index.
        before (bool): If True, include source indices preceding or equal to the target;
            otherwise, include those following or equal.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing:

            * Inclusive start indices in `source_index`.
            * Exclusive end indices in `source_index`.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed.
    """
    range_starts_out = np.empty(len(target_index), dtype=int_)
    range_ends_out = np.empty(len(target_index), dtype=int_)

    to_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be increasing")

        from_j = -1
        for j in range(to_j, len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Array index must be increasing")
            found = False
            if target_freq is None:
                if before:
                    if i == 0 and source_index[j] <= target_index[i] or i > 0 and target_index[i - 1] < source_index[j] <= target_index[i]:
                        found = True
                    elif source_index[j] > target_index[i]:
                        break
                else:
                    if i == len(target_index) - 1 and target_index[i] <= source_index[j] or (
                        i < len(target_index) - 1
                        and target_index[i] <= source_index[j] < target_index[i + 1]
                    ):
                        found = True
                    elif i < len(target_index) - 1 and source_index[j] >= target_index[i + 1]:
                        break
            else:
                if before:
                    if target_index[i] - target_freq < source_index[j] <= target_index[i]:
                        found = True
                    elif source_index[j] > target_index[i]:
                        break
                else:
                    if target_index[i] <= source_index[j] < target_index[i] + target_freq:
                        found = True
                    elif source_index[j] >= target_index[i] + target_freq:
                        break
            if found:
                if from_j == -1:
                    from_j = j
                to_j = j + 1

        if from_j == -1:
            range_starts_out[i] = -1
            range_ends_out[i] = -1
        else:
            range_starts_out[i] = from_j
            range_ends_out[i] = to_j

    return range_starts_out, range_ends_out


@register_jitted(cache=True)
def map_bounds_to_source_ranges_nb(
    source_index: tp.Array1d,
    target_lbound_index: tp.Array1d,
    target_rbound_index: tp.Array1d,
    closed_lbound: bool = True,
    closed_rbound: bool = False,
    skip_not_found: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Return the source bounds corresponding to the target bound indices.

    Each target range defined by `target_lbound_index` and `target_rbound_index` is mapped
    to a contiguous range in `source_index`. If no valid mapping is found for a target range,
    both the start and end indices are set to -1.

    Args:
        source_index (Array1d): Array of source indices, sorted in increasing order.
        target_lbound_index (Array1d): Array of target left-bound indices, sorted in increasing order.
        target_rbound_index (Array1d): Array of target right-bound indices, sorted in increasing order.
        closed_lbound (bool): Indicates if the left bound is inclusive.
        closed_rbound (bool): Indicates if the right bound is inclusive.
        skip_not_found (bool): Whether to drop indices that are -1 (not found).

    Returns:
        Tuple[Array1d, Array1d]: Tuple where the first array contains inclusive start indices and the
        second array contains exclusive end indices corresponding to the source ranges.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed.
    """
    range_starts_out = np.empty(len(target_lbound_index), dtype=int_)
    range_ends_out = np.empty(len(target_lbound_index), dtype=int_)
    k = 0

    to_j = 0
    for i in range(len(target_lbound_index)):
        if i > 0 and target_lbound_index[i] < target_lbound_index[i - 1]:
            raise ValueError("Target left-bound index must be increasing")
        if i > 0 and target_rbound_index[i] < target_rbound_index[i - 1]:
            raise ValueError("Target right-bound index must be increasing")

        from_j = -1
        for j in range(len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Array index must be increasing")
            found = False
            if closed_lbound and closed_rbound:
                if target_lbound_index[i] <= source_index[j] <= target_rbound_index[i]:
                    found = True
                elif source_index[j] > target_rbound_index[i]:
                    break
            elif closed_lbound:
                if target_lbound_index[i] <= source_index[j] < target_rbound_index[i]:
                    found = True
                elif source_index[j] >= target_rbound_index[i]:
                    break
            elif closed_rbound:
                if target_lbound_index[i] < source_index[j] <= target_rbound_index[i]:
                    found = True
                elif source_index[j] > target_rbound_index[i]:
                    break
            else:
                if target_lbound_index[i] < source_index[j] < target_rbound_index[i]:
                    found = True
                elif source_index[j] >= target_rbound_index[i]:
                    break
            if found:
                if from_j == -1:
                    from_j = j
                to_j = j + 1

        if skip_not_found:
            if from_j != -1:
                range_starts_out[k] = from_j
                range_ends_out[k] = to_j
                k += 1
        else:
            if from_j == -1:
                range_starts_out[i] = -1
                range_ends_out[i] = -1
            else:
                range_starts_out[i] = from_j
                range_ends_out[i] = to_j

    if skip_not_found:
        return range_starts_out[:k], range_ends_out[:k]
    return range_starts_out, range_ends_out


@register_jitted(cache=True)
def resample_source_mask_nb(
    source_mask: tp.Array1d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    source_freq: tp.Optional[tp.Scalar] = None,
    target_freq: tp.Optional[tp.Scalar] = None,
) -> tp.Array1d:
    """Return a resampled source mask aligned with the target index.

    Each element in the output becomes True only if the target bar, defined by `target_index`
    (and optionally adjusted with `target_freq`), is fully contained within a source bar.
    A source bar is represented by a contiguous sequence of True values in `source_mask` derived
    from `source_index` (and optionally adjusted with `source_freq`).

    Args:
        source_mask (Array1d): Boolean array representing the source mask.
        source_index (Array1d): Array of source indices, sorted in increasing order.
        target_index (Array1d): Array of target indices, which must be strictly increasing.
        source_freq (Optional[Scalar]): Frequency offset for the source index.
        target_freq (Optional[Scalar]): Frequency offset for the target index.

    Returns:
        Array1d: Boolean array indicating whether each target bar is fully covered by a source bar.
    """
    out = np.full(len(target_index), False, dtype=np.bool_)

    from_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be increasing")
        target_lbound = target_index[i]
        if target_freq is None:
            if i + 1 < len(target_index):
                target_rbound = target_index[i + 1]
            else:
                target_rbound = None
        else:
            target_rbound = target_index[i] + target_freq

        found_start = False
        for j in range(from_j, len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Source index must be increasing")
            source_lbound = source_index[j]
            if source_freq is None:
                if j + 1 < len(source_index):
                    source_rbound = source_index[j + 1]
                else:
                    source_rbound = None
            else:
                source_rbound = source_index[j] + source_freq

            if target_rbound is not None and target_rbound <= source_lbound:
                break
            if found_start or (
                target_lbound >= source_lbound
                and (source_rbound is None or target_lbound < source_rbound)
            ):
                if not found_start:
                    from_j = j
                    found_start = True
                if not source_mask[j]:
                    break
                if source_rbound is None or (
                    target_rbound is not None and target_rbound <= source_rbound
                ):
                    out[i] = True
                    break

    return out


@register_jitted(cache=True)
def last_before_target_index_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    incl_source: bool = True,
    incl_target: bool = False,
) -> tp.Array1d:
    """Return an array of positions representing the last valid source index within each range
    defined by `source_index` and `target_index`.

    For each element in `source_index`, the function finds the last index in `source_index`
    that lies between the original value and the corresponding value in `target_index`. The
    inclusivity of the endpoints is controlled by `incl_source` and `incl_target`.

    Args:
        source_index (Array1d): Array of source indices, sorted in increasing order.
        target_index (Array1d): Array of target indices, which must be strictly increasing.
        incl_source (bool): Whether to include the original source index in the result.
        incl_target (bool): Whether to include the target index if it matches a source index.

    Returns:
        Array1d: Array containing the position of the last valid source index for each element.
    """
    out = np.empty(len(source_index), dtype=int_)

    last_j = -1
    for i in range(len(source_index)):
        if i > 0 and source_index[i] < source_index[i - 1]:
            raise ValueError("Source index must be increasing")
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be increasing")
        if source_index[i] > target_index[i]:
            raise ValueError("Target index must be equal to or greater than source index")
        if last_j == -1:
            from_i = i + 1
        else:
            from_i = last_j
        if incl_source:
            last_j = i
        else:
            last_j = -1
        for j in range(from_i, len(source_index)):
            if source_index[j] < target_index[i] or incl_target and source_index[j] == target_index[i]:
                last_j = j
            else:
                break
        out[i] = last_j

    return out
