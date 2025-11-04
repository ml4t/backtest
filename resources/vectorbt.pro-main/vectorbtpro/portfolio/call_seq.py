# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing functions for working with call sequence arrays."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.portfolio.enums import CallSeqType
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []


@register_jitted(cache=True)
def shuffle_call_seq_nb(call_seq: tp.Array2d, group_lens: tp.GroupLens) -> None:
    """Shuffle segments of the call sequence array in place.

    Args:
        call_seq (Array2d): Array representing the call sequence.
        group_lens (GroupLens): Array defining the number of columns in each group.

    Returns:
        None: Function modifies `call_seq` in place.
    """
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        for i in range(call_seq.shape[0]):
            np.random.shuffle(call_seq[i, from_col:to_col])
        from_col = to_col


@register_jitted(cache=True)
def build_call_seq_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    call_seq_type: int = CallSeqType.Default,
) -> tp.Array2d:
    """Build a call sequence array with specified structure.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        call_seq_type (int): Identifier for the type of call sequence construction.

            See `vectorbtpro.portfolio.enums.CallSeqType`.

    Returns:
        Array2d: Call sequence array constructed based on the provided parameters.
    """
    if call_seq_type == CallSeqType.Reversed:
        out = np.full(target_shape[1], 1, dtype=int_)
        out[np.cumsum(group_lens)[1:] - group_lens[1:] - 1] -= group_lens[1:]
        out = np.cumsum(out[::-1])[::-1] - 1
        out = out * np.ones((target_shape[0], 1), dtype=int_)
        return out
    out = np.full(target_shape[1], 1, dtype=int_)
    out[np.cumsum(group_lens)[:-1]] -= group_lens[:-1]
    out = np.cumsum(out) - 1
    out = out * np.ones((target_shape[0], 1), dtype=int_)
    if call_seq_type == CallSeqType.Random:
        shuffle_call_seq_nb(out, group_lens)
    return out


def require_call_seq(call_seq: tp.Array2d) -> tp.Array2d:
    """Ensure the call sequence array meets required conditions.

    Args:
        call_seq (Array2d): Call sequence array to validate.

    Returns:
        Array2d: Validated call sequence array.
    """
    return np.require(call_seq, dtype=int_, requirements=["A", "O", "W", "F"])


def build_call_seq(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    call_seq_type: int = CallSeqType.Default,
) -> tp.Array2d:
    """Build a call sequence array using a faster, non-jitted implementation.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_lens (GroupLens): Array defining the number of columns in each group.
        call_seq_type (int): Identifier for the type of call sequence construction.

            See `vectorbtpro.portfolio.enums.CallSeqType`.

    Returns:
        Array2d: Call sequence array constructed based on the provided parameters.
    """
    call_seq = np.full(target_shape[1], 1, dtype=int_)
    if call_seq_type == CallSeqType.Reversed:
        call_seq[np.cumsum(group_lens)[1:] - group_lens[1:] - 1] -= group_lens[1:]
        call_seq = np.cumsum(call_seq[::-1])[::-1] - 1
    else:
        call_seq[np.cumsum(group_lens[:-1])] -= group_lens[:-1]
        call_seq = np.cumsum(call_seq) - 1
    call_seq = np.broadcast_to(call_seq, target_shape)
    if call_seq_type == CallSeqType.Random:
        call_seq = require_call_seq(call_seq)
        shuffle_call_seq_nb(call_seq, group_lens)
    return require_call_seq(call_seq)
