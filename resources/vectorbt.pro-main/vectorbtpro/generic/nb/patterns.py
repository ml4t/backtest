# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing generic Numba-compiled functions for working with patterns."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.flex_indexing import flex_select_1d_nb
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.generic.enums import DistanceMeasure, ErrorType, InterpMode, RescaleMode
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.array_ import rescale_nb


@register_jitted(cache=True)
def linear_interp_nb(arr: tp.FlexArray1d, i: int, source_size: int, target_size: int) -> float:
    """Return the interpolated value at a specified index using linear interpolation.

    Args:
        arr (FlexArray1d): Array to interpolate.
        i (int): Target index position.
        source_size (int): Size of the source array.
        target_size (int): Desired size for interpolation.

    Returns:
        float: Interpolated value.
    """
    if i == 0 or source_size == 1 or target_size == 1:
        return float(flex_select_1d_nb(arr, 0))
    if source_size == target_size:
        return float(flex_select_1d_nb(arr, i))
    if i == target_size - 1:
        return float(flex_select_1d_nb(arr, source_size - 1))
    mapped_i = i / (target_size - 1) * (source_size - 1)
    left_i = int(np.floor(mapped_i))
    right_i = int(np.ceil(mapped_i))
    norm_mapped_i = mapped_i - left_i
    left_elem = float(flex_select_1d_nb(arr, left_i))
    right_elem = float(flex_select_1d_nb(arr, right_i))
    return left_elem + norm_mapped_i * (right_elem - left_elem)


@register_jitted(cache=True)
def nearest_interp_nb(arr: tp.FlexArray1d, i: int, source_size: int, target_size: int) -> float:
    """Return the interpolated value at a specified index using nearest-neighbor interpolation.

    Args:
        arr (FlexArray1d): Array to interpolate.
        i (int): Target index position.
        source_size (int): Size of the source array.
        target_size (int): Desired size for interpolation.

    Returns:
        float: Interpolated value.
    """
    if i == 0 or source_size == 1 or target_size == 1:
        return float(flex_select_1d_nb(arr, 0))
    if source_size == target_size:
        return float(flex_select_1d_nb(arr, i))
    if i == target_size - 1:
        return float(flex_select_1d_nb(arr, source_size - 1))
    mapped_i = i / (target_size - 1) * (source_size - 1)
    return float(flex_select_1d_nb(arr, round(mapped_i)))


@register_jitted(cache=True)
def discrete_interp_nb(arr: tp.FlexArray1d, i: int, source_size: int, target_size: int) -> float:
    """Return the interpolated value at a specified index using discrete interpolation.

    This function returns the nearest array value when the mapping aligns with a source index.
    When ambiguity arises, it returns NaN.

    Args:
        arr (FlexArray1d): Array to interpolate.
        i (int): Target index position.
        source_size (int): Size of the source array.
        target_size (int): Desired size for interpolation.

    Returns:
        float: Interpolated value or NaN if ambiguous.
    """
    if source_size >= target_size:
        return nearest_interp_nb(arr, i, source_size, target_size)
    if i == 0 or source_size == 1 or target_size == 1:
        return float(flex_select_1d_nb(arr, 0))
    if i == target_size - 1:
        return float(flex_select_1d_nb(arr, source_size - 1))
    curr_float_mapped_i = i / (target_size - 1) * (source_size - 1)
    curr_remainder = curr_float_mapped_i % 1
    if curr_remainder == 0:
        return float(flex_select_1d_nb(arr, int(curr_float_mapped_i)))
    if curr_remainder <= 0.5:
        prev_float_mapped_i = (i - 1) / (target_size - 1) * (source_size - 1)
        if int(curr_float_mapped_i) != int(prev_float_mapped_i):
            prev_remainder = prev_float_mapped_i % 1
            if curr_remainder < 1 - prev_remainder:
                return float(flex_select_1d_nb(arr, int(np.floor(curr_float_mapped_i))))
        return np.nan
    next_float_mapped_i = (i + 1) / (target_size - 1) * (source_size - 1)
    if int(curr_float_mapped_i) != int(next_float_mapped_i):
        next_remainder = next_float_mapped_i % 1
        if 1 - curr_remainder <= next_remainder:
            return float(flex_select_1d_nb(arr, int(np.ceil(curr_float_mapped_i))))
    return np.nan


@register_jitted(cache=True)
def mixed_interp_nb(arr: tp.FlexArray1d, i: int, source_size: int, target_size: int) -> float:
    """Return the interpolated value at a specified index using mixed interpolation.

    This method uses discrete interpolation as the primary approach and applies linear interpolation
    as a fallback for NaN cases, thereby preserving the vertical scale of the pattern.

    Args:
        arr (FlexArray1d): Array to interpolate.
        i (int): Target index position.
        source_size (int): Size of the source array.
        target_size (int): Desired size for interpolation.

    Returns:
        float: Interpolated value.
    """
    value = discrete_interp_nb(arr, i, source_size, target_size)
    if np.isnan(value):
        value = linear_interp_nb(arr, i, source_size, target_size)
    return value


@register_jitted(cache=True)
def interp_nb(
    arr: tp.FlexArray1d, i: int, source_size: int, target_size: int, interp_mode: int
) -> float:
    """Return the interpolated value at a specified index based on the given interpolation mode.

    The interpolation mode is defined in `vectorbtpro.generic.enums.InterpMode`.

    Args:
        arr (FlexArray1d): Array to interpolate.
        i (int): Target index position.
        source_size (int): Size of the source array.
        target_size (int): Desired size for interpolation.
        interp_mode (int): Interpolation mode.

            See `vectorbtpro.generic.enums.InterpMode`.

    Returns:
        float: Interpolated value.
    """
    if interp_mode == InterpMode.Linear:
        return linear_interp_nb(arr, i, source_size, target_size)
    if interp_mode == InterpMode.Nearest:
        return nearest_interp_nb(arr, i, source_size, target_size)
    if interp_mode == InterpMode.Discrete:
        return discrete_interp_nb(arr, i, source_size, target_size)
    if interp_mode == InterpMode.Mixed:
        return mixed_interp_nb(arr, i, source_size, target_size)
    raise ValueError("Invalid interpolation mode")


@register_jitted(cache=True)
def interp_resize_1d_nb(arr: tp.FlexArray1d, target_size: int, interp_mode: int) -> tp.Array1d:
    """Return a resized array with interpolated values using `interp_nb`.

    Args:
        arr (FlexArray1d): Array to be resized.
        target_size (int): Desired size for interpolation.
        interp_mode (int): Interpolation mode.

            See `vectorbtpro.generic.enums.InterpMode`.

    Returns:
        Array1d: Resized array with interpolated values.
    """
    out = np.empty(target_size, dtype=float_)
    for i in range(target_size):
        out[i] = interp_nb(arr, i, arr.size, target_size, interp_mode)
    return out


@register_jitted(cache=True)
def fit_pattern_nb(
    arr: tp.Array1d,
    pattern: tp.Array1d,
    interp_mode: int = InterpMode.Mixed,
    rescale_mode: int = RescaleMode.MinMax,
    vmin: float = np.nan,
    vmax: float = np.nan,
    pmin: float = np.nan,
    pmax: float = np.nan,
    invert: bool = False,
    error_type: int = ErrorType.Absolute,
    max_error: tp.FlexArray1dLike = np.nan,
    max_error_interp_mode: tp.Optional[int] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Fit pattern.

    Resize and rescale the specified `pattern` to match the length and scale of the input array `arr`.

    Args:
        arr (Array1d): Input array determining the target length and rescaling range.
        pattern (Array1d): Pattern array to be resized and rescaled.
        interp_mode (int): Interpolation mode.

            See `vectorbtpro.generic.enums.InterpMode`.
        rescale_mode (int): Rescaling mode for adjusting the ranges of `arr` and `pattern`.

            See `vectorbtpro.generic.enums.RescaleMode`.
        vmin (float): Minimum threshold for clipping the input array.

            If NaN, the minimum of `arr` is used.
        vmax (float): Maximum threshold for clipping the input array.

            If NaN, the maximum of `arr` is used.
        pmin (float): Minimum threshold for clipping the resized pattern.

            If NaN, the minimum of the resized pattern is used.
        pmax (float): Maximum threshold for clipping the resized pattern.

            If NaN, the maximum of the resized pattern is used.
        invert (bool): Invert the pattern by reflecting its values.
        error_type (int): Error computation mode.

            See `vectorbtpro.generic.enums.ErrorType`.
        max_error (FlexArray1dLike): Maximum error threshold for normalization.

            Provided as a scalar or per element in the pattern.
        max_error_interp_mode (Optional[int]): Interpolation mode for `max_error`.

            See `vectorbtpro.generic.enums.InterpMode`.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing the resized and rescaled pattern and
            the corresponding maximum error.
    """
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if max_error_interp_mode is None:
        max_error_interp_mode = interp_mode
    fit_pattern = interp_resize_1d_nb(
        pattern,
        len(arr),
        interp_mode,
    )
    fit_max_error = interp_resize_1d_nb(
        max_error_,
        len(arr),
        max_error_interp_mode,
    )
    if np.isnan(vmin):
        vmin = np.nanmin(arr)
    else:
        vmin = vmin
    if np.isnan(vmax):
        vmax = np.nanmax(arr)
    else:
        vmax = vmax
    if np.isnan(pmin):
        pmin = np.nanmin(fit_pattern)
    else:
        pmin = pmin
    if np.isnan(pmax):
        pmax = np.nanmax(fit_pattern)
    else:
        pmax = pmax
    if invert:
        fit_pattern = pmax + pmin - fit_pattern
    if rescale_mode == RescaleMode.Rebase:
        if not np.isnan(pmin):
            if fit_pattern[0] == 0:
                pmin = np.nan
            else:
                pmin = pmin / fit_pattern[0] * arr[0]
        if not np.isnan(pmax):
            if fit_pattern[0] == 0:
                pmax = np.nan
            else:
                pmax = pmax / fit_pattern[0] * arr[0]
    if rescale_mode == RescaleMode.Rebase:
        if fit_pattern[0] == 0:
            fit_pattern = np.full(fit_pattern.shape, np.nan)
        else:
            fit_pattern = fit_pattern / fit_pattern[0] * arr[0]
        fit_max_error = fit_max_error * fit_pattern
    fit_pattern = np.clip(fit_pattern, pmin, pmax)
    if rescale_mode == RescaleMode.MinMax:
        fit_pattern = rescale_nb(fit_pattern, (pmin, pmax), (vmin, vmax))
        if error_type == ErrorType.Absolute:
            if pmax - pmin == 0:
                fit_max_error = np.full(fit_max_error.shape, np.nan)
            else:
                fit_max_error = fit_max_error / (pmax - pmin) * (vmax - vmin)
        else:
            fit_max_error = fit_max_error * fit_pattern
    return fit_pattern, fit_max_error


@register_jitted(cache=True)
def pattern_similarity_nb(
    arr: tp.Array1d,
    pattern: tp.Array1d,
    interp_mode: int = InterpMode.Mixed,
    rescale_mode: int = RescaleMode.MinMax,
    vmin: float = np.nan,
    vmax: float = np.nan,
    pmin: float = np.nan,
    pmax: float = np.nan,
    invert: bool = False,
    error_type: int = ErrorType.Absolute,
    distance_measure: int = DistanceMeasure.MAE,
    max_error: tp.FlexArray1dLike = np.nan,
    max_error_interp_mode: tp.Optional[int] = None,
    max_error_as_maxdist: bool = False,
    max_error_strict: bool = False,
    min_pct_change: float = np.nan,
    max_pct_change: float = np.nan,
    min_similarity: float = np.nan,
    minp: tp.Optional[int] = None,
) -> float:
    """Return the similarity between an input array and a pattern array.

    This function maps each element of `arr` into the range of `pattern` and computes
    the absolute difference (error) between the actual and mapped values. It then normalizes
    the error by the corresponding maximum error at that position to obtain a relative error
    between 0 and 1, and aggregates these normalized errors to derive the overall similarity measure.

    Args:
        arr (Array1d): 1-dimensional array to compare with the pattern.
        pattern (Array1d): 1-dimensional array representing the reference pattern.
        interp_mode (int): Interpolation mode.

            See `vectorbtpro.generic.enums.InterpMode`.
        rescale_mode (int): Rescaling mode for adjusting the ranges of `arr` and `pattern`.

            See `vectorbtpro.generic.enums.RescaleMode`.
        vmin (float): Minimum value used for rescaling `arr`.
        vmax (float): Maximum value used for rescaling `arr`.
        pmin (float): Minimum value used for rescaling `pattern`.
        pmax (float): Maximum value used for rescaling `pattern`.
        invert (bool): Invert the pattern by reflecting its values.
        error_type (int): Error computation mode.

            See `vectorbtpro.generic.enums.ErrorType`.
        distance_measure (int): Method for measuring distance (e.g., MAE, MSE, RMSE).

            See `vectorbtpro.generic.enums.DistanceMeasure`.
        max_error (FlexArray1dLike): Maximum error threshold for normalization.

            Provided as a scalar or per element in the pattern.
        max_error_interp_mode (Optional[int]): Interpolation mode for `max_error`.

            See `vectorbtpro.generic.enums.InterpMode`.
        max_error_as_maxdist (bool): Indicates whether `max_error` represents the maximum distance at each point.
        max_error_strict (bool): If True, any instance of exceeding `max_error` results in a similarity of NaN.
        min_pct_change (float): Minimum percent change applied during rescaling.
        max_pct_change (float): Maximum percentage change allowed for a window to remain a search candidate.
        min_similarity (float): Minimum similarity threshold.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        float: Similarity measure between `arr` and `pattern`,
            or NaN if the computation conditions are not met.
    """
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if len(arr) == 0:
        return np.nan
    if len(pattern) == 0:
        return np.nan
    if rescale_mode == RescaleMode.Rebase:
        if np.isnan(pattern[0]):
            return np.nan
        if np.isnan(arr[0]):
            return np.nan
    if max_error_interp_mode is None or max_error_interp_mode == -1:
        _max_error_interp_mode = interp_mode
    else:
        _max_error_interp_mode = max_error_interp_mode
    max_size = max(arr.shape[0], pattern.shape[0])
    if error_type != ErrorType.Absolute and error_type != ErrorType.Relative:
        raise ValueError("Invalid error type")
    if (
        distance_measure != DistanceMeasure.MAE
        and distance_measure != DistanceMeasure.MSE
        and distance_measure != DistanceMeasure.RMSE
    ):
        raise ValueError("Invalid distance mode")
    if minp is None:
        minp = arr.shape[0]

    min_max_required = False
    if rescale_mode == RescaleMode.MinMax:
        min_max_required = True
    if not np.isnan(min_pct_change):
        min_max_required = True
    if not np.isnan(max_pct_change):
        min_max_required = True
    if not max_error_as_maxdist:
        min_max_required = True
    if invert:
        min_max_required = True
    if min_max_required:
        vmin_set = not np.isnan(vmin)
        vmax_set = not np.isnan(vmax)
        pmin_set = not np.isnan(pmin)
        pmax_set = not np.isnan(pmax)
        if not vmin_set or not vmax_set or not pmin_set or not pmax_set:
            for i in range(max_size):
                if arr.shape[0] >= pattern.shape[0]:
                    arr_elem = arr[i]
                else:
                    arr_elem = interp_nb(arr, i, arr.shape[0], pattern.shape[0], interp_mode)
                if pattern.shape[0] >= arr.shape[0]:
                    pattern_elem = pattern[i]
                else:
                    pattern_elem = interp_nb(
                        pattern, i, pattern.shape[0], arr.shape[0], interp_mode
                    )
                if not np.isnan(arr_elem):
                    if not vmin_set and (np.isnan(vmin) or arr_elem < vmin):
                        vmin = arr_elem
                    if not vmax_set and (np.isnan(vmax) or arr_elem > vmax):
                        vmax = arr_elem
                if not np.isnan(pattern_elem):
                    if not pmin_set and (np.isnan(pmin) or pattern_elem < pmin):
                        pmin = pattern_elem
                    if not pmax_set and (np.isnan(pmax) or pattern_elem > pmax):
                        pmax = pattern_elem
        if np.isnan(vmin) or np.isnan(vmax):
            return np.nan
        if np.isnan(pmin) or np.isnan(pmax):
            return np.nan
        if vmin == vmax and rescale_mode == RescaleMode.MinMax:
            return np.nan
        if pmin == pmax and rescale_mode == RescaleMode.MinMax:
            return np.nan
        if not np.isnan(min_pct_change) and vmin != 0 and (vmax - vmin) / vmin < min_pct_change:
            return np.nan
        if not np.isnan(max_pct_change) and vmin != 0 and (vmax - vmin) / vmin > max_pct_change:
            return np.nan

    first_pattern_elem = pattern[0]
    if invert:
        first_pattern_elem = pmax + pmin - first_pattern_elem
    if rescale_mode == RescaleMode.Rebase:
        if not np.isnan(pmin):
            if first_pattern_elem == 0:
                pmin = np.nan
            else:
                pmin = pmin / first_pattern_elem * arr[0]
        if not np.isnan(pmax):
            if first_pattern_elem == 0:
                pmax = np.nan
            else:
                pmax = pmax / first_pattern_elem * arr[0]
    if rescale_mode == RescaleMode.Rebase or rescale_mode == RescaleMode.Disable:
        if not np.isnan(pmin) and not np.isnan(vmin):
            _min = min(pmin, vmin)
        else:
            _min = vmin
        if not np.isnan(pmax) and not np.isnan(vmax):
            _max = max(pmax, vmax)
        else:
            _max = vmax
    else:
        _min = vmin
        _max = vmax

    distance_sum = 0.0
    maxdistance_sum = 0.0
    nan_count = 0
    for i in range(max_size):
        if i < arr.shape[0]:
            if np.isnan(arr[i]):
                nan_count += 1
            if max_size - nan_count < minp:
                return np.nan

        if arr.shape[0] == pattern.shape[0]:
            arr_elem = arr[i]
            pattern_elem = pattern[i]
            _max_error = flex_select_1d_nb(max_error_, i)
        elif arr.shape[0] > pattern.shape[0]:
            arr_elem = arr[i]
            pattern_elem = interp_nb(pattern, i, pattern.shape[0], arr.shape[0], interp_mode)
            _max_error = interp_nb(
                max_error_, i, pattern.shape[0], arr.shape[0], _max_error_interp_mode
            )
        else:
            arr_elem = interp_nb(arr, i, arr.shape[0], pattern.shape[0], interp_mode)
            pattern_elem = pattern[i]
            _max_error = flex_select_1d_nb(max_error_, i)

        if not np.isnan(arr_elem) and not np.isnan(pattern_elem):
            if invert:
                pattern_elem = pmax + pmin - pattern_elem
            if rescale_mode == RescaleMode.Rebase:
                if first_pattern_elem == 0:
                    pattern_elem = np.nan
                else:
                    pattern_elem = pattern_elem / first_pattern_elem * arr[0]
                if error_type == ErrorType.Absolute:
                    _max_error = _max_error * pattern_elem
            if not np.isnan(vmin) and arr_elem < vmin:
                arr_elem = vmin
            if not np.isnan(vmax) and arr_elem > vmax:
                arr_elem = vmax
            if not np.isnan(pmin) and pattern_elem < pmin:
                pattern_elem = pmin
            if not np.isnan(pmax) and pattern_elem > pmax:
                pattern_elem = pmax
            if rescale_mode == RescaleMode.MinMax:
                if pmax - pmin == 0:
                    pattern_elem = np.nan
                else:
                    pattern_elem = (pattern_elem - pmin) / (pmax - pmin) * (vmax - vmin) + vmin
                if error_type == ErrorType.Absolute:
                    if pmax - pmin == 0:
                        _max_error = np.nan
                    else:
                        _max_error = _max_error / (pmax - pmin) * (vmax - vmin)

            if distance_measure == DistanceMeasure.MAE:
                if error_type == ErrorType.Absolute:
                    dist = abs(arr_elem - pattern_elem)
                else:
                    if pattern_elem == 0:
                        dist = np.nan
                    else:
                        dist = abs(arr_elem - pattern_elem) / pattern_elem
            else:
                if error_type == ErrorType.Absolute:
                    dist = (arr_elem - pattern_elem) ** 2
                else:
                    if pattern_elem == 0:
                        dist = np.nan
                    else:
                        dist = ((arr_elem - pattern_elem) / pattern_elem) ** 2
            if max_error_as_maxdist:
                if np.isnan(_max_error):
                    continue
                maxdist = _max_error
            else:
                if distance_measure == DistanceMeasure.MAE:
                    if error_type == ErrorType.Absolute:
                        maxdist = max(pattern_elem - _min, _max - pattern_elem)
                    else:
                        if pattern_elem == 0:
                            maxdist = np.nan
                        else:
                            maxdist = max(pattern_elem - _min, _max - pattern_elem) / pattern_elem
                else:
                    if error_type == ErrorType.Absolute:
                        maxdist = max(pattern_elem - _min, _max - pattern_elem) ** 2
                    else:
                        if pattern_elem == 0:
                            maxdist = np.nan
                        else:
                            maxdist = (
                                max(pattern_elem - _min, _max - pattern_elem) / pattern_elem
                            ) ** 2
            if dist > 0 and maxdist == 0:
                return np.nan
            if not np.isnan(_max_error) and dist > _max_error:
                if max_error_strict:
                    return np.nan
                dist = maxdist
            if dist > maxdist:
                dist = maxdist
            distance_sum = distance_sum + dist
            maxdistance_sum = maxdistance_sum + maxdist

            if not np.isnan(min_similarity):
                if not max_error_as_maxdist or max_error_.size == 1:
                    if max_error_as_maxdist:
                        if np.isnan(_max_error):
                            return np.nan
                        worst_maxdist = _max_error
                    else:
                        if distance_measure == DistanceMeasure.MAE:
                            if error_type == ErrorType.Absolute:
                                worst_maxdist = _max - _min
                            else:
                                if _min == 0:
                                    worst_maxdist = np.nan
                                else:
                                    worst_maxdist = (_max - _min) / _min
                        else:
                            if error_type == ErrorType.Absolute:
                                worst_maxdist = (_max - _min) ** 2
                            else:
                                if _min == 0:
                                    worst_maxdist = np.nan
                                else:
                                    worst_maxdist = ((_max - _min) / _min) ** 2
                    worst_maxdistance_sum = maxdistance_sum + worst_maxdist * (max_size - i - 1)
                    if worst_maxdistance_sum == 0:
                        return np.nan
                    if distance_measure == DistanceMeasure.RMSE:
                        if worst_maxdistance_sum == 0:
                            best_similarity = np.nan
                        else:
                            best_similarity = 1 - np.sqrt(distance_sum) / np.sqrt(
                                worst_maxdistance_sum
                            )
                    else:
                        if worst_maxdistance_sum == 0:
                            best_similarity = np.nan
                        else:
                            best_similarity = 1 - distance_sum / worst_maxdistance_sum
                    if best_similarity < min_similarity:
                        return np.nan

    if distance_sum == 0:
        return 1.0
    if maxdistance_sum == 0:
        return np.nan
    if distance_measure == DistanceMeasure.RMSE:
        if maxdistance_sum == 0:
            similarity = np.nan
        else:
            similarity = 1 - np.sqrt(distance_sum) / np.sqrt(maxdistance_sum)
    else:
        if maxdistance_sum == 0:
            similarity = np.nan
        else:
            similarity = 1 - distance_sum / maxdistance_sum
    if not np.isnan(min_similarity):
        if similarity < min_similarity:
            return np.nan
    return similarity
