# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for grouping."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []

GroupByT = tp.Union[None, bool, tp.Index]


@register_jitted(cache=True)
def get_group_lens_nb(groups: tp.Array1d) -> tp.GroupLens:
    """Return the count of elements per group in a sorted array.

    Args:
        groups (Array1d): Array of group identifiers sorted in non-decreasing order.

    Returns:
        GroupLens: Array of counts for each group.

    !!! note
        Columns must form monolithic, sorted groups. For unsorted groups, use `get_group_map_nb`.
    """
    result = np.empty(groups.shape[0], dtype=int_)
    j = 0
    last_group = -1
    group_len = 0
    for i in range(groups.shape[0]):
        cur_group = groups[i]
        if cur_group < last_group:
            raise ValueError("Columns must form monolithic, sorted groups")
        if cur_group != last_group:
            if last_group != -1:
                # Process previous group
                result[j] = group_len
                j += 1
                group_len = 0
            last_group = cur_group
        group_len += 1
        if i == groups.shape[0] - 1:
            # Process last group
            result[j] = group_len
            j += 1
            group_len = 0
    return result[:j]


@register_jitted(cache=True)
def get_group_map_nb(groups: tp.Array1d, n_groups: int) -> tp.GroupMap:
    """Build mapping between groups and array indices.

    Args:
        groups (Array1d): Array of group identifiers.
        n_groups (int): Total number of groups.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing:

            * Array of indices segmented by group.
            * Array of group lengths.
    """
    group_lens_out = np.full(n_groups, 0, dtype=int_)
    for g in range(groups.shape[0]):
        group = groups[g]
        group_lens_out[group] += 1

    group_start_idxs = np.cumsum(group_lens_out) - group_lens_out
    group_idxs_out = np.empty((groups.shape[0],), dtype=int_)
    group_i = np.full(n_groups, 0, dtype=int_)
    for g in range(groups.shape[0]):
        group = groups[g]
        group_idxs_out[group_start_idxs[group] + group_i[group]] = g
        group_i[group] += 1

    return group_idxs_out, group_lens_out


@register_jitted(cache=True)
def group_lens_select_nb(
    group_lens: tp.GroupLens, new_groups: tp.Array1d
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Perform selection indexing on a sorted group array using group lengths.

    Args:
        group_lens (GroupLens): Array defining the number of columns in each group.
        new_groups (Array1d): Array of group identifiers to select.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing:

            * Array of indices for the selected groups.
            * Array mapping each index to its new group position.
    """
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    n_values = np.sum(group_lens[new_groups])
    indices_out = np.empty(n_values, dtype=int_)
    group_arr_out = np.empty(n_values, dtype=int_)
    j = 0

    for c in range(new_groups.shape[0]):
        from_r = group_start_idxs[new_groups[c]]
        to_r = group_end_idxs[new_groups[c]]
        if from_r == to_r:
            continue
        rang = np.arange(from_r, to_r)
        indices_out[j : j + rang.shape[0]] = rang
        group_arr_out[j : j + rang.shape[0]] = c
        j += rang.shape[0]
    return indices_out, group_arr_out


@register_jitted(cache=True)
def group_map_select_nb(
    group_map: tp.GroupMap, new_groups: tp.Array1d
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Perform selection indexing using a provided group map.

    Args:
        group_map (GroupMap): Tuple of indices and lengths for each group.
        new_groups (Array1d): Array of original group identifiers to select.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing:

            * Array of indices from the group map corresponding to the selected groups.
            * Array mapping each index to its new group position.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    total_count = np.sum(group_lens[new_groups])
    indices_out = np.empty(total_count, dtype=int_)
    group_arr_out = np.empty(total_count, dtype=int_)
    j = 0

    for new_group_i in range(len(new_groups)):
        new_group = new_groups[new_group_i]
        group_len = group_lens[new_group]
        if group_len == 0:
            continue
        group_start_idx = group_start_idxs[new_group]
        idxs = group_idxs[group_start_idx : group_start_idx + group_len]
        indices_out[j : j + group_len] = idxs
        group_arr_out[j : j + group_len] = new_group_i
        j += group_len
    return indices_out, group_arr_out


@register_jitted(cache=True)
def group_by_evenly_nb(n: int, n_splits: int) -> tp.Array1d:
    """Compute group assignment by evenly splitting a space of values.

    Args:
        n (int): Total number of values.
        n_splits (int): Number of groups to split the values into.

    Returns:
        Array1d: Array where each element's value indicates its group assignment.
    """
    out = np.empty(n, dtype=int_)
    for i in range(n):
        out[i] = i * n_splits // n + n_splits // (2 * n)
    return out
