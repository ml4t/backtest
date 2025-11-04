# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for signals.

Provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of the backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbtpro treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled.
"""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb, flex_select_nb
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.enums import RangeStatus, range_dt
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.signals.enums import *
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import rescale_float_to_int_nb, rescale_nb, uniform_summing_to_one_nb
from vectorbtpro.utils.template import Rep

__all__ = []


# ############# Generation ############# #


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        place_func_nb=None,
        place_args=ch.ArgsTaker(),
        only_once=None,
        wait=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def generate_nb(
    target_shape: tp.Shape,
    place_func_nb: tp.PlaceFunc,
    place_args: tp.Args = (),
    only_once: bool = True,
    wait: int = 1,
) -> tp.Array2d:
    """Create a boolean matrix of `target_shape` and place signals using `place_func_nb`.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        place_func_nb (PlaceFunc): Callback function that accepts `vectorbtpro.signals.enums.GenEnContext`
            and additional arguments, and returns the index of the last signal (-1 to break the loop).

            To place multiple signals, modify `c.out` in the context and return the last index.
        place_args (Args): Positional arguments for `place_func_nb`.
        only_once (bool): Whether to run the placement function only once.
        wait (int): Number of ticks to wait before placing the next entry.

    Returns:
        Array2d: 2-dimensional boolean array with placed signals.

    !!! note
        The first argument is always a 1-dimensional boolean array that contains only those
        elements where signals can be placed. The range and column indices only describe
        which range this array maps to.

    !!! tip
        This function is parallelizable.
    """
    if wait < 0:
        raise ValueError("wait must be zero or greater")
    out = np.full(target_shape, False, dtype=np.bool_)

    for col in prange(target_shape[1]):
        from_i = 0
        while from_i <= target_shape[0] - 1:
            c = GenEnContext(
                target_shape=target_shape,
                only_once=only_once,
                wait=wait,
                entries_out=out,
                out=out[from_i:, col],
                from_i=from_i,
                to_i=target_shape[0],
                col=col,
            )
            _last_i = place_func_nb(c, *place_args)
            if _last_i == -1:
                break
            last_i = from_i + _last_i
            if last_i < from_i or last_i >= target_shape[0]:
                raise ValueError("Last index is out of bounds")
            if not out[last_i, col]:
                out[last_i, col] = True
            if only_once:
                break
            from_i = last_i + wait
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="entries", axis=1),
    arg_take_spec=dict(
        entries=ch.ArraySlicer(axis=1),
        exit_place_func_nb=None,
        exit_place_args=ch.ArgsTaker(),
        wait=None,
        until_next=None,
        skip_until_exit=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def generate_ex_nb(
    entries: tp.Array2d,
    exit_place_func_nb: tp.PlaceFunc,
    exit_place_args: tp.Args = (),
    wait: int = 1,
    until_next: bool = True,
    skip_until_exit: bool = False,
) -> tp.Array2d:
    """Place exit signals using `exit_place_func_nb` after each signal in `entries`.

    Args:
        entries (Array2d): Boolean array with entry signals.
        exit_place_func_nb (PlaceFunc): Callback function that accepts `vectorbtpro.signals.enums.GenExContext`
            and additional arguments, and returns the index of the last exit signal (-1 to break the loop).

            To place multiple signals, modify `c.out` in the context and return the last index.
        exit_place_args (Args): Positional arguments for `exit_place_func_nb`.
        wait (int): Number of ticks to wait before placing exits.

            !!! note
                Setting `wait` to 0 or False may result in two signals at one bar.
        until_next (bool): Whether to place signals up to the next entry signal.

            !!! note
                Setting it to False makes it difficult to tell which exit belongs to which entry.
        skip_until_exit (bool): Whether to skip processing entry signals until the next exit.

            Has only effect when `until_next` is disabled.

            !!! note
                Setting it to True makes it impossible to tell which exit belongs to which entry.

    Returns:
        Array2d: 2-dimensional boolean array with placed exit signals.

    !!! tip
        This function is parallelizable.
    """
    if wait < 0:
        raise ValueError("wait must be zero or greater")
    out = np.full_like(entries, False)

    def _place_exits(from_i, to_i, col, last_exit_i):
        if from_i > -1:
            if skip_until_exit and from_i <= last_exit_i:
                return last_exit_i
            from_i += wait
            if not until_next:
                to_i = entries.shape[0]
            if to_i > from_i:
                c = GenExContext(
                    entries=out,
                    until_next=until_next,
                    skip_until_exit=skip_until_exit,
                    exits_out=out,
                    out=out[from_i:to_i, col],
                    wait=wait,
                    from_i=from_i,
                    to_i=to_i,
                    col=col,
                )
                _last_exit_i = exit_place_func_nb(c, *exit_place_args)
                if _last_exit_i != -1:
                    last_exit_i = from_i + _last_exit_i
                    if last_exit_i < from_i or last_exit_i >= entries.shape[0]:
                        raise ValueError("Last index is out of bounds")
                    if not out[last_exit_i, col]:
                        out[last_exit_i, col] = True
                elif skip_until_exit:
                    last_exit_i = -1
        return last_exit_i

    for col in prange(entries.shape[1]):
        from_i = -1
        last_exit_i = -1
        should_stop = False
        for i in range(entries.shape[0]):
            if entries[i, col]:
                last_exit_i = _place_exits(from_i, i, col, last_exit_i)
                if skip_until_exit and last_exit_i == -1 and from_i != -1:
                    should_stop = True
                    break
                from_i = i
        if should_stop:
            continue
        last_exit_i = _place_exits(from_i, entries.shape[0], col, last_exit_i)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        entry_place_func_nb=None,
        entry_place_args=ch.ArgsTaker(),
        exit_place_func_nb=None,
        exit_place_args=ch.ArgsTaker(),
        entry_wait=None,
        exit_wait=None,
    ),
    merge_func="column_stack",
)
@register_jitted
def generate_enex_nb(
    target_shape: tp.Shape,
    entry_place_func_nb: tp.PlaceFunc,
    entry_place_args: tp.Args,
    exit_place_func_nb: tp.PlaceFunc,
    exit_place_args: tp.Args,
    entry_wait: int = 1,
    exit_wait: int = 1,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Place entry and exit signals sequentially using provided placement functions.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        entry_place_func_nb (PlaceFunc): Callback function that accepts `vectorbtpro.signals.enums.GenEnExContext`
            and additional arguments, and returns the index of the last entry signal (-1 to break the loop).

            To place multiple signals, modify `c.out` in the context and return the last index.
        entry_place_args (Args): Positional arguments for `entry_place_func_nb`.
        exit_place_func_nb (PlaceFunc): Callback function that accepts `vectorbtpro.signals.enums.GenEnExContext`
            and additional arguments, and returns the index of the last exit signal (-1 to break the loop).

            To place multiple signals, modify `c.out` in the context and return the last index.
        exit_place_args (Args): Positional arguments for `exit_place_func_nb`.
        entry_wait (int): Number of periods to wait before an entry signal is triggered.

            !!! note
                Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and exit can be processed before entry.
        exit_wait (int): Number of periods to wait before an exit signal is triggered.

            !!! note
                Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and entry can be processed before exit.

    Returns:
        Tuple[Array2d, Array2d]: Tuple containing the entry signals array and the exit signals array.

    !!! tip
        This function is parallelizable.
    """
    if entry_wait < 0:
        raise ValueError("entry_wait must be zero or greater")
    if exit_wait < 0:
        raise ValueError("exit_wait must be zero or greater")
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait and exit_wait cannot be both 0")
    entries = np.full(target_shape, False)
    exits = np.full(target_shape, False)

    def _place_signals(entries_turn, out, from_i, col, wait, place_func_nb, args):
        to_i = target_shape[0]
        if to_i > from_i:
            c = GenEnExContext(
                target_shape=target_shape,
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                entries_out=entries,
                exits_out=exits,
                entries_turn=entries_turn,
                out=out[from_i:to_i, col],
                wait=wait if from_i > 0 else 0,
                from_i=from_i,
                to_i=to_i,
                col=col,
            )
            _last_i = place_func_nb(c, *args)
            if _last_i == -1:
                return -1
            last_i = from_i + _last_i
            if last_i < from_i or last_i >= target_shape[0]:
                raise ValueError("Last index is out of bounds")
            if not out[last_i, col]:
                out[last_i, col] = True
            return last_i
        return -1

    for col in range(target_shape[1]):
        from_i = 0
        entries_turn = True
        first_signal = True
        while from_i != -1:
            if entries_turn:
                if not first_signal:
                    from_i += entry_wait
                from_i = _place_signals(
                    entries_turn,
                    entries,
                    from_i,
                    col,
                    entry_wait,
                    entry_place_func_nb,
                    entry_place_args,
                )
                entries_turn = False
            else:
                from_i += exit_wait
                from_i = _place_signals(
                    entries_turn,
                    exits,
                    from_i,
                    col,
                    exit_wait,
                    exit_place_func_nb,
                    exit_place_args,
                )
                entries_turn = True
            first_signal = False

    return entries, exits


# ############# Random signals ############# #


@register_jitted
def rand_place_nb(
    c: tp.Union[GenEnContext, GenExContext, GenEnExContext], n: tp.FlexArray1d
) -> int:
    """Randomly pick signals by marking eligible positions in the context's output array.

    Args:
        c (Union[GenEnContext, GenExContext, GenEnExContext]): Signal generation context.
        n (FlexArray1d): Flexible array indicating the number of signals to place.

    Returns:
        int: Index of the last placed signal.
    """
    size = min(c.to_i - c.from_i, flex_select_1d_pc_nb(n, c.col))
    k = 0
    last_i = -1
    while k < size:
        i = np.random.choice(len(c.out))
        if not c.out[i]:
            c.out[i] = True
            k += 1
        if i > last_i:
            last_i = i
    return last_i


@register_jitted
def rand_by_prob_place_nb(
    c: tp.Union[GenEnContext, GenExContext, GenEnExContext],
    prob: tp.FlexArray2d,
    pick_first: bool = False,
) -> int:
    """Randomly place signals based on probabilities by marking positions in the context's output array.

    Args:
        c (Union[GenEnContext, GenExContext, GenEnExContext]): Signal generation context.
        prob (FlexArray2d): Flexible 2D array of probabilities used for signal placement.
        pick_first (bool): If True, stop after placing the first signal.

    Returns:
        int: Index (relative to the start of the context's array) of the last placed signal.
    """
    last_i = -1
    for i in range(c.from_i, c.to_i):
        if np.random.uniform(0, 1) < flex_select_nb(prob, i, c.col):
            c.out[i - c.from_i] = True
            last_i = i - c.from_i
            if pick_first:
                break
    return last_i


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        n=base_ch.FlexArraySlicer(),
        entry_wait=None,
        exit_wait=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def generate_rand_enex_nb(
    target_shape: tp.Shape,
    n: tp.FlexArray1d,
    entry_wait: int = 1,
    exit_wait: int = 1,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Pick a number of entries and the same number of exits one after another.

    Respects `entry_wait` and `exit_wait` constraints through a number of tricks.
    Tries to mimic a uniform distribution as much as possible.

    The idea is the following: with constraints, there is some fixed amount of total
    space required between the first entry and the last exit. Upscale this space so that
    the distribution of entries and exits mimics a uniform distribution. This involves
    randomizing the position of the first entry, the last exit, and all signals in between.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        n (FlexArray1d): Flexible array specifying the number of entry-exit pairs per column.
        entry_wait (int): Number of periods to wait before an entry signal is triggered.

            !!! note
                Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and exit can be processed before entry.
        exit_wait (int): Number of periods to wait before an exit signal is triggered.

            !!! note
                Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and entry can be processed before exit.

    Returns:
        Tuple[Array2d, Array2d]: Tuple containing a boolean array for entries and a boolean array for exits.

    !!! tip
        This function is parallelizable.
    """
    entries = np.full(target_shape, False)
    exits = np.full(target_shape, False)
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait and exit_wait cannot be both 0")

    if entry_wait == 1 and exit_wait == 1:
        # Basic case
        both = generate_nb(target_shape, rand_place_nb, (n * 2,), only_once=True, wait=1)
        for col in prange(both.shape[1]):
            both_idxs = np.flatnonzero(both[:, col])
            entries[both_idxs[0::2], col] = True
            exits[both_idxs[1::2], col] = True
    else:
        for col in prange(target_shape[1]):
            _n = flex_select_1d_pc_nb(n, col)
            if _n == 1:
                entry_idx = np.random.randint(0, target_shape[0] - exit_wait)
                entries[entry_idx, col] = True
            else:
                min_range = entry_wait + exit_wait
                min_total_range = min_range * (_n - 1)
                if target_shape[0] < min_total_range + exit_wait + 1:
                    raise ValueError("Cannot take a larger sample than population")

                max_free_space = target_shape[0] - min_total_range - 1
                free_space = min(max_free_space, 3 * target_shape[0] // (_n + 1))
                free_space -= exit_wait

                rand_floats = uniform_summing_to_one_nb(6)
                chosen_spaces = rescale_float_to_int_nb(rand_floats, (0, free_space), free_space)
                first_idx = chosen_spaces[0]
                last_idx = target_shape[0] - np.sum(chosen_spaces[-2:]) - exit_wait - 1
                total_range = last_idx - first_idx
                max_range = total_range - (_n - 2) * min_range

                rand_floats = uniform_summing_to_one_nb(_n - 1)
                chosen_ranges = rescale_float_to_int_nb(
                    rand_floats, (min_range, max_range), total_range
                )
                entry_idxs = np.empty(_n, dtype=int_)
                entry_idxs[0] = first_idx
                entry_idxs[1:] = chosen_ranges
                entry_idxs = np.cumsum(entry_idxs)
                entries[entry_idxs, col] = True

        for col in range(target_shape[1]):
            entry_idxs = np.flatnonzero(entries[:, col])
            for j in range(len(entry_idxs)):
                entry_i = entry_idxs[j] + exit_wait
                if j < len(entry_idxs) - 1:
                    exit_i = entry_idxs[j + 1] - entry_wait
                else:
                    exit_i = entries.shape[0] - 1
                i = np.random.randint(exit_i - entry_i + 1)
                exits[entry_i + i, col] = True
    return entries, exits


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        n=base_ch.FlexArraySlicer(),
        entry_wait=None,
        exit_wait=None,
    ),
    merge_func="column_stack",
)
@register_jitted
def rand_enex_apply_nb(
    target_shape: tp.Shape,
    n: tp.FlexArray1d,
    entry_wait: int = 1,
    exit_wait: int = 1,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Call `generate_rand_enex_nb` using `apply_func_nb`.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        n (FlexArray1d): Flexible array specifying the number of entry-exit pairs per column.
        entry_wait (int): Number of periods to wait before an entry signal is triggered.

            !!! note
                Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and exit can be processed before entry.
        exit_wait (int): Number of periods to wait before an exit signal is triggered.

            !!! note
                Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and entry can be processed before exit.

    Returns:
        Tuple[Array2d, Array2d]: Tuple containing boolean arrays for entries and exits.
    """
    return generate_rand_enex_nb(target_shape, n, entry_wait=entry_wait, exit_wait=exit_wait)


# ############# Stop signals ############# #


@register_jitted
def first_place_nb(
    c: tp.Union[GenEnContext, GenExContext, GenEnExContext], mask: tp.Array2d
) -> int:
    """Keep only the first signal in `mask`.

    Iterate over indices from `c.from_i` to `c.to_i` in column `c.col`. Set the first encountered
    signal in `mask` to True in `c.out` and return its relative index. If no signal is found, return -1.

    Args:
        c (Union[GenEnContext, GenExContext, GenEnExContext]): Signal generation context.
        mask (Array2d): Boolean array indicating candidate signal positions.

    Returns:
        int: Relative index of the first signal in the mask, or -1 if no signal is found.
    """
    last_i = -1
    for i in range(c.from_i, c.to_i):
        if mask[i, c.col]:
            c.out[i - c.from_i] = True
            last_i = i - c.from_i
            break
    return last_i


@register_jitted
def stop_place_nb(
    c: tp.Union[GenExContext, GenEnExContext],
    entry_ts: tp.FlexArray2d,
    ts: tp.FlexArray2d,
    follow_ts: tp.FlexArray2d,
    stop_ts_out: tp.Array2d,
    stop: tp.FlexArray2d,
    trailing: tp.FlexArray2d,
) -> int:
    """Place an exit signal when a threshold is reached, implementing `place_func_nb`.

    Args:
        c (Union[GenExContext, GenEnExContext]): Signal generation context.
        entry_ts (FlexArray2d): Entry price array.

            Uses flexible indexing.
        ts (FlexArray2d): Price array for comparing the stop value.

            Uses flexible indexing. If NaN, defaults to the corresponding entry price.
        follow_ts (FlexArray2d): Array of following prices.

            Uses flexible indexing. If NaN, defaults to the corresponding value from `ts`.
            Applied only when the stop is trailing.
        stop_ts_out (Array2d): Array to store the hit price of each exit.

            Must have the complete shape.
        stop (FlexArray2d): Stop value array.

            Uses flexible indexing. An element set to `np.nan` disables it.
        trailing (FlexArray2d): Array indicating whether the stop is trailing.

            Uses flexible indexing. An element set to False disables trailing.

    Returns:
        int: Relative index at which the exit signal was placed, or -1 if no exit signal occurred.

    !!! note
        Waiting time cannot exceed 1.

        * If waiting time is 0, `entry_ts` corresponds to the first value in the bar.
        * If waiting time is 1, `entry_ts` corresponds to the last value in the bar.
    """
    if c.wait > 1:
        raise ValueError("Wait must be either 0 or 1")
    init_i = c.from_i - c.wait
    init_entry_ts = flex_select_nb(entry_ts, init_i, c.col)
    init_stop = flex_select_nb(stop, init_i, c.col)
    if init_stop == 0:
        init_stop = np.nan
    init_trailing = flex_select_nb(trailing, init_i, c.col)
    max_high = min_low = init_entry_ts

    last_i = -1
    for i in range(c.from_i, c.to_i):
        curr_entry_ts = flex_select_nb(entry_ts, i, c.col)
        curr_ts = flex_select_nb(ts, i, c.col)
        curr_follow_ts = flex_select_nb(follow_ts, i, c.col)
        if np.isnan(curr_ts):
            curr_ts = curr_entry_ts
        if np.isnan(curr_follow_ts):
            if not np.isnan(curr_entry_ts):
                if init_stop >= 0:
                    curr_follow_ts = min(curr_entry_ts, curr_ts)
                else:
                    curr_follow_ts = max(curr_entry_ts, curr_ts)
            else:
                curr_follow_ts = curr_ts

        if not np.isnan(init_stop):
            if init_trailing:
                if init_stop >= 0:
                    # Trailing stop buy
                    curr_stop_price = min_low * (1 + abs(init_stop))
                else:
                    # Trailing stop sell
                    curr_stop_price = max_high * (1 - abs(init_stop))
            else:
                curr_stop_price = init_entry_ts * (1 + init_stop)

        # Check if stop price is within bar
        if not np.isnan(init_stop):
            if init_stop >= 0:
                exit_signal = curr_ts >= curr_stop_price
            else:
                exit_signal = curr_ts <= curr_stop_price
            if exit_signal:
                stop_ts_out[i, c.col] = curr_stop_price
                c.out[i - c.from_i] = True
                last_i = i - c.from_i
                break

        # Keep track of lowest low and highest high if trailing
        if init_trailing:
            if curr_follow_ts < min_low:
                min_low = curr_follow_ts
            elif curr_follow_ts > max_high:
                max_high = curr_follow_ts

    return last_i


@register_jitted
def ohlc_stop_place_nb(
    c: tp.Union[GenExContext, GenEnExContext],
    entry_price: tp.FlexArray2d,
    open: tp.FlexArray2d,
    high: tp.FlexArray2d,
    low: tp.FlexArray2d,
    close: tp.FlexArray2d,
    stop_price_out: tp.Array2d,
    stop_type_out: tp.Array2d,
    sl_stop: tp.FlexArray2d,
    tsl_th: tp.FlexArray2d,
    tsl_stop: tp.FlexArray2d,
    tp_stop: tp.FlexArray2d,
    reverse: tp.FlexArray2d,
    is_entry_open: bool = False,
) -> int:
    """Place an exit signal when a threshold is hit using OHLC data.

    This function extends `stop_place_nb` by considering the entire bar.
    It simultaneously checks for trailing stop loss and take profit thresholds,
    tracking the hit price and corresponding stop type.

    Args:
        c (Union[GenExContext, GenEnExContext]): Signal generation context.
        entry_price (FlexArray2d): Entry price array.

            Utilizes flexible indexing.
        open (FlexArray2d): Open price array.

            Utilizes flexible indexing. If `np.nan` and `is_entry_open` is True, defaults to `entry_price`.
        high (FlexArray2d): High price array.

            Utilizes flexible indexing. If `np.nan`, it is computed from `open` and `close`.
        low (FlexArray2d): Low price array.

            Utilizes flexible indexing. If `np.nan`, it is computed from `open` and `close`.
        close (FlexArray2d): Close price array.

            Utilizes flexible indexing. If `np.nan` and `is_entry_open` is False, defaults to `entry_price`.
        stop_price_out (Array2d): Array to store the hit price of each exit; must have full shape.
        stop_type_out (Array2d): Array to store the exit signal stop type; must have full shape.

            Typically, uses `StopType.SL` for stop loss, `StopType.TP` for take profit,
            and trailing stop types (`StopType.TSL` or `StopType.TTP`).
            See `vectorbtpro.signals.enums.StopType`.
        sl_stop (FlexArray2d): Stop loss percentage array.

            Utilizes flexible indexing. Set an element to `np.nan` to disable.
        tsl_th (FlexArray2d): Take profit threshold percentage for trailing stop loss.

            Utilizes flexible indexing. Set an element to `np.nan` to disable.
        tsl_stop (FlexArray2d): Trailing stop loss percentage array.

            Utilizes flexible indexing. Set an element to `np.nan` to disable.
        tp_stop (FlexArray2d): Take profit percentage array.

            Utilizes flexible indexing. Set an element to `np.nan` to disable.
        reverse (FlexArray2d): Array indicating reversal (prices are followed downwards).

            Utilizes flexible indexing.
        is_entry_open (bool): Indicates if the entry price is available at or before open.

            If True, uses the bar's high and low; otherwise, uses only close.

    Returns:
        int: Index offset (relative to `c.from_i`) of the bar at which an exit signal was triggered,
            or -1 if no exit signal was generated.

    !!! note
        The waiting time must not exceed 1.
    """
    if c.wait > 1:
        raise ValueError("Wait must be either 0 or 1")
    init_i = c.from_i - c.wait
    init_entry_price = flex_select_nb(entry_price, init_i, c.col)
    init_sl_stop = abs(flex_select_nb(sl_stop, init_i, c.col))
    init_tp_stop = abs(flex_select_nb(tp_stop, init_i, c.col))
    init_tsl_th = abs(flex_select_nb(tsl_th, init_i, c.col))
    init_tsl_stop = abs(flex_select_nb(tsl_stop, init_i, c.col))
    init_reverse = flex_select_nb(reverse, init_i, c.col)
    last_high = last_low = init_entry_price

    last_i = -1
    for i in range(c.from_i - c.wait, c.to_i):
        # Resolve current bar
        _entry_price = flex_select_nb(entry_price, i, c.col)
        _open = flex_select_nb(open, i, c.col)
        _high = flex_select_nb(high, i, c.col)
        _low = flex_select_nb(low, i, c.col)
        _close = flex_select_nb(close, i, c.col)
        if np.isnan(_open) and not np.isnan(_entry_price) and is_entry_open:
            _open = _entry_price
        if np.isnan(_close) and not np.isnan(_entry_price) and not is_entry_open:
            _close = _entry_price
        if np.isnan(_high):
            if np.isnan(_open):
                _high = _close
            elif np.isnan(_close):
                _high = _open
            else:
                _high = max(_open, _close)
        if np.isnan(_low):
            if np.isnan(_open):
                _low = _close
            elif np.isnan(_close):
                _low = _open
            else:
                _low = min(_open, _close)
        if i > init_i or is_entry_open:
            curr_high = _high
            curr_low = _low
        else:
            curr_high = curr_low = _close

        if i >= c.from_i:
            # Calculate stop prices
            if not np.isnan(init_sl_stop):
                if init_reverse:
                    curr_sl_stop_price = init_entry_price * (1 + init_sl_stop)
                else:
                    curr_sl_stop_price = init_entry_price * (1 - init_sl_stop)
            if not np.isnan(init_tsl_stop):
                if np.isnan(init_tsl_th):
                    if init_reverse:
                        curr_tsl_stop_price = last_low * (1 + init_tsl_stop)
                    else:
                        curr_tsl_stop_price = last_high * (1 - init_tsl_stop)
                else:
                    if init_reverse:
                        if last_low <= init_entry_price * (1 - init_tsl_th):
                            curr_tsl_stop_price = last_low * (1 + init_tsl_stop)
                        else:
                            curr_tsl_stop_price = np.nan
                    else:
                        if last_high >= init_entry_price * (1 + init_tsl_th):
                            curr_tsl_stop_price = last_high * (1 - init_tsl_stop)
                        else:
                            curr_tsl_stop_price = np.nan
            if not np.isnan(init_tp_stop):
                if init_reverse:
                    curr_tp_stop_price = init_entry_price * (1 - init_tp_stop)
                else:
                    curr_tp_stop_price = init_entry_price * (1 + init_tp_stop)

            # Check if stop price is within bar
            exit_signal = False
            if not np.isnan(init_sl_stop):
                # SL hit?
                stop_price = np.nan
                if not init_reverse:
                    if _open <= curr_sl_stop_price:
                        stop_price = _open
                    if curr_low <= curr_sl_stop_price:
                        stop_price = curr_sl_stop_price
                else:
                    if _open >= curr_sl_stop_price:
                        stop_price = _open
                    if curr_high >= curr_sl_stop_price:
                        stop_price = curr_sl_stop_price
                if not np.isnan(stop_price):
                    stop_price_out[i, c.col] = stop_price
                    stop_type_out[i, c.col] = StopType.SL
                    exit_signal = True

            if not exit_signal and not np.isnan(init_tsl_stop):
                # TSL/TTP hit?
                stop_price = np.nan
                if not init_reverse:
                    if _open <= curr_tsl_stop_price:
                        stop_price = _open
                    if curr_low <= curr_tsl_stop_price:
                        stop_price = curr_tsl_stop_price
                else:
                    if _open >= curr_tsl_stop_price:
                        stop_price = _open
                    if curr_high >= curr_tsl_stop_price:
                        stop_price = curr_tsl_stop_price
                if not np.isnan(stop_price):
                    stop_price_out[i, c.col] = stop_price
                    if np.isnan(init_tsl_th):
                        stop_type_out[i, c.col] = StopType.TSL
                    else:
                        stop_type_out[i, c.col] = StopType.TTP
                    exit_signal = True

            if not exit_signal and not np.isnan(init_tp_stop):
                # TP hit?
                stop_price = np.nan
                if not init_reverse:
                    if _open >= curr_tp_stop_price:
                        stop_price = _open
                    if curr_high >= curr_tp_stop_price:
                        stop_price = curr_tp_stop_price
                else:
                    if _open <= curr_tp_stop_price:
                        stop_price = _open
                    if curr_low <= curr_tp_stop_price:
                        stop_price = curr_tp_stop_price
                if not np.isnan(stop_price):
                    stop_price_out[i, c.col] = stop_price
                    stop_type_out[i, c.col] = StopType.TP
                    exit_signal = True

            if exit_signal:
                c.out[i - c.from_i] = True
                last_i = i - c.from_i
                break

        if i > init_i or is_entry_open:
            # Keep track of the lowest low and the highest high
            if curr_low < last_low:
                last_low = curr_low
            if curr_high > last_high:
                last_high = curr_high

    return last_i


# ############# Ranking ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1),
        rank_func_nb=None,
        rank_args=ch.ArgsTaker(),
        reset_by=None,
        after_false=None,
        after_reset=None,
        reset_wait=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rank_nb(
    mask: tp.Array2d,
    rank_func_nb: tp.RankFunc,
    rank_args: tp.Args = (),
    reset_by: tp.Optional[tp.Array2d] = None,
    after_false: bool = False,
    after_reset: bool = False,
    reset_wait: int = 1,
) -> tp.Array2d:
    """Rank each signal using `rank_func_nb`.

    Applies `rank_func_nb` on each True value in the input `mask`.

    Args:
        mask (Array2d): Boolean mask array indicating positions of signals.
        rank_func_nb (RankFunc): Callback function that accepts `vectorbtpro.signals.enums.RankContext`
            and additional arguments, and returns -1 for no rank or a non-negative integer for a valid rank.
        rank_args (Args): Positional arguments for `rank_func_nb`.
        reset_by (Optional[Array2d]): Boolean array indicating reset positions.
        after_false (bool): If True, disregards the first True partition with no preceding False.
        after_reset (bool): If True, disregards the first True partition before a reset signal.
        reset_wait (int): Offset to treat reset signals.

            * 0 treats the signal at reset as the first in the next partition.
            * 1 treats it as the last in the previous partition.

    Returns:
        Array2d: Array containing computed signal ranks.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(mask.shape, -1, dtype=int_)

    for col in prange(mask.shape[1]):
        in_partition = False
        false_seen = not after_false
        reset_seen = reset_by is None
        last_false_i = -1
        last_reset_i = -1
        all_sig_cnt = 0
        all_part_cnt = 0
        all_sig_in_part_cnt = 0
        nonres_sig_cnt = 0
        nonres_part_cnt = 0
        nonres_sig_in_part_cnt = 0
        sig_cnt = 0
        part_cnt = 0
        sig_in_part_cnt = 0

        for i in range(mask.shape[0]):
            if reset_by is not None and reset_by[i, col]:
                last_reset_i = i
            if last_reset_i > -1 and i - last_reset_i == reset_wait:
                reset_seen = True
                sig_cnt = 0
                part_cnt = 0
                sig_in_part_cnt = 0
            if mask[i, col]:
                all_sig_cnt += 1
                if i == 0 or not mask[i - 1, col]:
                    all_part_cnt += 1
                all_sig_in_part_cnt += 1
                if not (after_false and not false_seen) and not (after_reset and not reset_seen):
                    nonres_sig_cnt += 1
                    sig_cnt += 1
                    if not in_partition:
                        nonres_part_cnt += 1
                        part_cnt += 1
                    elif last_reset_i > -1 and i - last_reset_i == reset_wait:
                        part_cnt += 1
                    nonres_sig_in_part_cnt += 1
                    sig_in_part_cnt += 1
                    in_partition = True
                    c = RankContext(
                        mask=mask,
                        reset_by=reset_by,
                        after_false=after_false,
                        after_reset=after_reset,
                        reset_wait=reset_wait,
                        col=col,
                        i=i,
                        last_false_i=last_false_i,
                        last_reset_i=last_reset_i,
                        all_sig_cnt=all_sig_cnt,
                        all_part_cnt=all_part_cnt,
                        all_sig_in_part_cnt=all_sig_in_part_cnt,
                        nonres_sig_cnt=nonres_sig_cnt,
                        nonres_part_cnt=nonres_part_cnt,
                        nonres_sig_in_part_cnt=nonres_sig_in_part_cnt,
                        sig_cnt=sig_cnt,
                        part_cnt=part_cnt,
                        sig_in_part_cnt=sig_in_part_cnt,
                    )
                    out[i, col] = rank_func_nb(c, *rank_args)
            else:
                all_sig_in_part_cnt = 0
                nonres_sig_in_part_cnt = 0
                sig_in_part_cnt = 0
                last_false_i = i
                in_partition = False
                false_seen = True

    return out


@register_jitted
def sig_pos_rank_nb(c: RankContext, allow_gaps: bool) -> int:
    """Return the signal rank based on its position.

    Computes the rank for a signal using its order in the partition. If `allow_gaps` is True,
    the function returns a global rank ignoring partition gaps; otherwise, it uses the rank within
    the current partition. Resets occur at each reset signal.

    Args:
        c (RankContext): Ranking context.
        allow_gaps (bool): Flag to determine whether to allow gaps in ranking.

    Returns:
        int: Computed rank.
    """
    if allow_gaps:
        return c.sig_cnt - 1
    return c.sig_in_part_cnt - 1


@register_jitted
def part_pos_rank_nb(c: RankContext) -> int:
    """Return the partition rank based on its position.

    Computes the rank for a partition based on its order in the series. Resets occur at each reset signal.

    Args:
        c (RankContext): Ranking context.

    Returns:
        int: Computed partition rank.
    """
    return c.part_cnt - 1


# ############# Distance ############# #


@register_jitted(cache=True)
def distance_from_last_1d_nb(mask: tp.Array1d, nth: int = 1) -> tp.Array1d:
    """Compute the distance from the n-th last True value to the current index.

    Unless `nth` is zero, the current True value is not considered among the last True values.

    Args:
        mask (Array1d): 1D boolean array indicating signal positions.
        nth (int): Index of the last True value to measure the distance from.

    Returns:
        Array1d: Array of distances.
    """
    if nth < 0:
        raise ValueError("nth must be at least 0")
    out = np.empty(mask.shape, dtype=int_)
    last_indices = np.empty(mask.shape, dtype=int_)
    k = 0
    for i in range(mask.shape[0]):
        if nth == 0:
            if mask[i]:
                last_indices[k] = i
                k += 1
            if k - 1 < 0:
                out[i] = -1
            else:
                out[i] = i - last_indices[k - 1]
        elif nth == 1:
            if k - nth < 0:
                out[i] = -1
            else:
                out[i] = i - last_indices[k - nth]
            if mask[i]:
                last_indices[k] = i
                k += 1
        else:
            if mask[i]:
                last_indices[k] = i
                k += 1
            if k - nth < 0:
                out[i] = -1
            else:
                out[i] = i - last_indices[k - nth]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1), nth=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def distance_from_last_nb(mask: tp.Array2d, nth: int = 1) -> tp.Array2d:
    """Compute the distance from the n-th last True value to each element in a 2D array.

    Applies `distance_from_last_1d_nb` for each column in the 2D boolean array.

    Args:
        mask (Array2d): 2D boolean array indicating signal positions.
        nth (int): Index of the last True value to measure the distance from.

    Returns:
        Array2d: 2D array with computed distances.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(mask.shape, dtype=int_)
    for col in prange(mask.shape[1]):
        out[:, col] = distance_from_last_1d_nb(mask[:, col], nth=nth)
    return out


# ############# Cleaning ############# #


@register_jitted(cache=True)
def clean_enex_1d_nb(
    entries: tp.Array1d,
    exits: tp.Array1d,
    force_first: bool = True,
    keep_conflicts: bool = False,
    reverse_order: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Clean entry and exit arrays by selecting the first valid signal in each partition.

    Parameters determine the ordering and conflict handling:

    * `force_first`: If True, the first signal (entry or exit) is forced before its counterpart.
    * `keep_conflicts`: If True, simultaneous signals are processed sequentially without removal.
    * `reverse_order`: If True, the order of signals is reversed.

    Args:
        entries (Array1d): 1D boolean array indicating entry signals.
        exits (Array1d): 1D boolean array indicating exit signals.
        force_first (bool): Determines whether the first signal is forced to precede its counterpart.
        keep_conflicts (bool): Determines if simultaneous signals are processed sequentially.
        reverse_order (bool): Determines whether to reverse the order of signals.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing the cleaned entry and exit arrays.
    """
    entries_out = np.full(entries.shape, False, dtype=np.bool_)
    exits_out = np.full(exits.shape, False, dtype=np.bool_)

    def _process_entry(i, phase):
        if ((not force_first or not reverse_order) and phase == -1) or phase == 1:
            phase = 0
            entries_out[i] = True
        return phase

    def _process_exit(i, phase):
        if ((not force_first or reverse_order) and phase == -1) or phase == 0:
            phase = 1
            exits_out[i] = True
        return phase

    phase = -1
    for i in range(entries.shape[0]):
        if entries[i] and exits[i]:
            if keep_conflicts:
                if not reverse_order:
                    phase = _process_entry(i, phase)
                    phase = _process_exit(i, phase)
                else:
                    phase = _process_exit(i, phase)
                    phase = _process_entry(i, phase)
        elif entries[i]:
            phase = _process_entry(i, phase)
        elif exits[i]:
            phase = _process_exit(i, phase)

    return entries_out, exits_out


@register_chunkable(
    size=ch.ArraySizer(arg_query="entries", axis=1),
    arg_take_spec=dict(
        entries=ch.ArraySlicer(axis=1),
        exits=ch.ArraySlicer(axis=1),
        force_first=None,
        keep_conflicts=None,
        reverse_order=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def clean_enex_nb(
    entries: tp.Array2d,
    exits: tp.Array2d,
    force_first: bool = True,
    keep_conflicts: bool = False,
    reverse_order: bool = False,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Clean 2D entry and exit signals by applying `clean_enex_1d_nb` column-wise.

    Args:
        entries (Array2d): 2D boolean array of entry signals.

            Each column represents signals for a distinct series.
        exits (Array2d): 2D boolean array of exit signals.

            Each column represents signals for a distinct series.
        force_first (bool): Determines whether the first signal is forced to precede its counterpart.
        keep_conflicts (bool): Determines if simultaneous signals are processed sequentially.
        reverse_order (bool): Determines whether to reverse the order of signals.

    Returns:
        Tuple[Array2d, Array2d]: Tuple containing the cleaned entries and exits arrays.

    !!! tip
        This function is parallelizable.
    """
    entries_out = np.empty(entries.shape, dtype=np.bool_)
    exits_out = np.empty(exits.shape, dtype=np.bool_)

    for col in prange(entries.shape[1]):
        entries_out[:, col], exits_out[:, col] = clean_enex_1d_nb(
            entries[:, col],
            exits[:, col],
            force_first=force_first,
            keep_conflicts=keep_conflicts,
            reverse_order=reverse_order,
        )
    return entries_out, exits_out


# ############# Relation ############# #


@register_jitted(cache=True)
def relation_idxs_1d_nb(
    source_mask: tp.Array1d,
    target_mask: tp.Array1d,
    relation: int = SignalRelation.OneMany,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
    """Get index pairs between a source and target mask based on a signal relation.

    This function computes index pairs and order positions for True entries in
    `source_mask` and `target_mask` according to the specified signal relation.
    For available relations, see `vectorbtpro.signals.enums.SignalRelation`.

    Args:
        source_mask (Array1d): Boolean array indicating source signals.
        target_mask (Array1d): Boolean array indicating target signals.
        relation (int): Relation mode for pairing signals.

            See `vectorbtpro.signals.enums.SignalRelation`.

    Returns:
        Tuple[Array1d, Array1d, Array1d, Array1d]: Tuple containing:

            * `source_range`: Order positions for source signals.
            * `target_range`: Order positions for target signals.
            * `source_idxs`: Indices of True entries in the source mask.
            * `target_idxs`: Indices of True entries in the target mask.

    !!! note
        If both True values occur simultaneously, the source signal is prioritized.
    """
    if relation == SignalRelation.Chain or relation == SignalRelation.AnyChain:
        max_signals = source_mask.shape[0] * 2
    else:
        max_signals = source_mask.shape[0]
    source_range_out = np.full(max_signals, -1, dtype=int_)
    target_range_out = np.full(max_signals, -1, dtype=int_)
    source_idxs_out = np.full(max_signals, -1, dtype=int_)
    target_idxs_out = np.full(max_signals, -1, dtype=int_)
    source_j = -1
    target_j = -1
    k = 0

    if relation == SignalRelation.OneOne:
        fill_k = -1
        for i in range(source_mask.shape[0]):
            if source_mask[i]:
                source_j += 1
            if target_mask[i]:
                target_j += 1
            if source_mask[i]:
                source_range_out[k] = source_j
                source_idxs_out[k] = i
                if fill_k == -1:
                    fill_k = k
                k += 1
            if target_mask[i]:
                if fill_k == -1:
                    target_range_out[k] = target_j
                    target_idxs_out[k] = i
                    k += 1
                else:
                    target_range_out[fill_k] = target_j
                    target_idxs_out[fill_k] = i
                    fill_k += 1
                    if fill_k == k:
                        fill_k = -1
    elif relation == SignalRelation.OneMany:
        source_idx = -1
        source_placed = True
        for i in range(source_mask.shape[0]):
            if source_mask[i]:
                source_j += 1
            if target_mask[i]:
                target_j += 1
            if source_mask[i]:
                if not source_placed:
                    source_range_out[k] = source_j
                    source_idxs_out[k] = source_idx
                    k += 1
                source_idx = i
                source_placed = False
            if target_mask[i]:
                if source_idx == -1:
                    target_range_out[k] = target_j
                    target_idxs_out[k] = i
                    k += 1
                else:
                    source_range_out[k] = source_j
                    target_range_out[k] = target_j
                    source_idxs_out[k] = source_idx
                    target_idxs_out[k] = i
                    k += 1
                    source_placed = True
        if not source_placed:
            source_range_out[k] = source_j
            source_idxs_out[k] = source_idx
            k += 1
    elif relation == SignalRelation.ManyOne:
        target_idx = -1
        target_placed = True
        for i in range(source_mask.shape[0] - 1, -1, -1):
            if source_mask[i]:
                source_j += 1
            if target_mask[i]:
                target_j += 1
            if target_mask[i]:
                if not target_placed:
                    target_range_out[k] = target_j
                    target_idxs_out[k] = target_idx
                    k += 1
                target_idx = i
                target_placed = False
            if source_mask[i]:
                if target_idx == -1:
                    source_range_out[k] = source_j
                    source_idxs_out[k] = i
                    k += 1
                else:
                    source_range_out[k] = source_j
                    target_range_out[k] = target_j
                    source_idxs_out[k] = i
                    target_idxs_out[k] = target_idx
                    k += 1
                    target_placed = True
        if not target_placed:
            target_range_out[k] = target_j
            target_idxs_out[k] = target_idx
            k += 1
        source_range_out[:k] = source_range_out[:k][::-1]
        target_range_out[:k] = target_range_out[:k][::-1]
        for _k in range(k):
            source_range_out[_k] = source_j - source_range_out[_k]
            target_range_out[_k] = target_j - target_range_out[_k]
        source_idxs_out[:k] = source_idxs_out[:k][::-1]
        target_idxs_out[:k] = target_idxs_out[:k][::-1]
    elif relation == SignalRelation.ManyMany:
        source_idx = -1
        from_k = -1
        for i in range(source_mask.shape[0]):
            if source_mask[i]:
                source_j += 1
            if target_mask[i]:
                target_j += 1
            if source_mask[i]:
                source_idx = i
                source_range_out[k] = source_j
                source_idxs_out[k] = source_idx
                if from_k == -1:
                    from_k = k
                k += 1
            if target_mask[i]:
                if from_k == -1:
                    if source_idx != -1:
                        source_range_out[k] = source_j
                        source_idxs_out[k] = source_idx
                    target_range_out[k] = target_j
                    target_idxs_out[k] = i
                    k += 1
                else:
                    for _k in range(from_k, k):
                        target_range_out[_k] = target_j
                        target_idxs_out[_k] = i
                    from_k = -1
    elif relation == SignalRelation.Chain:
        source_idx = -1
        target_idx = -1
        any_placed = False
        for i in range(source_mask.shape[0]):
            if source_mask[i]:
                source_j += 1
            if target_mask[i]:
                target_j += 1
            if source_mask[i]:
                if source_idx == -1:
                    source_idx = i
                    source_range_out[k] = source_j
                    source_idxs_out[k] = source_idx
                    any_placed = True
                    if target_idx != -1:
                        target_idx = -1
                        k += 1
                        source_range_out[k] = source_j
                        source_idxs_out[k] = source_idx
            if target_mask[i]:
                if target_idx == -1 and source_idx != -1:
                    target_idx = i
                    target_range_out[k] = target_j
                    target_idxs_out[k] = target_idx
                    source_idx = -1
                    k += 1
                    target_range_out[k] = target_j
                    target_idxs_out[k] = target_idx
                    any_placed = True
        if any_placed:
            k += 1
    elif relation == SignalRelation.AnyChain:
        source_idx = -1
        target_idx = -1
        any_placed = False
        for i in range(source_mask.shape[0]):
            if source_mask[i]:
                source_j += 1
            if target_mask[i]:
                target_j += 1
            if source_mask[i]:
                if source_idx == -1:
                    source_idx = i
                    source_range_out[k] = source_j
                    source_idxs_out[k] = source_idx
                    any_placed = True
                    if target_idx != -1:
                        target_idx = -1
                        k += 1
                        source_range_out[k] = source_j
                        source_idxs_out[k] = source_idx
            if target_mask[i]:
                if target_idx == -1:
                    target_idx = i
                    target_range_out[k] = target_j
                    target_idxs_out[k] = target_idx
                    any_placed = True
                    if source_idx != -1:
                        source_idx = -1
                        k += 1
                        target_range_out[k] = target_j
                        target_idxs_out[k] = target_idx
        if any_placed:
            k += 1
    else:
        raise ValueError("Invalid SignalRelation option")
    return source_range_out[:k], target_range_out[:k], source_idxs_out[:k], target_idxs_out[:k]


# ############# Ranges ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1), incl_open=None),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def between_ranges_nb(mask: tp.Array2d, incl_open: bool = False) -> tp.RecordArray:
    """Create a record array representing ranges between signals in `mask`.

    The function iterates over each column of the boolean array `mask` to identify consecutive signals.
    When a pair of signals is found, a record is created with start and end indices and a closed status.
    If `incl_open` is True and the last signal does not have a corresponding closing signal,
    an open range is recorded.

    Args:
        mask (Array2d): Boolean array indicating signal positions.
        incl_open (bool): Include an open range if no closing signal is found.

    Returns:
        RecordArray: Array of records containing range metadata.

            Has the `vectorbtpro.generic.enums.range_dt` dtype.

    !!! tip
        This function is parallelizable.
    """
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=int_)

    for col in prange(mask.shape[1]):
        from_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if from_i > -1:
                    r = counts[col]
                    new_records["id"][r, col] = r
                    new_records["col"][r, col] = col
                    new_records["start_idx"][r, col] = from_i
                    new_records["end_idx"][r, col] = i
                    new_records["status"][r, col] = RangeStatus.Closed
                    counts[col] += 1
                from_i = i
        if incl_open and from_i < mask.shape[0] - 1:
            r = counts[col]
            new_records["id"][r, col] = r
            new_records["col"][r, col] = col
            new_records["start_idx"][r, col] = from_i
            new_records["end_idx"][r, col] = mask.shape[0] - 1
            new_records["status"][r, col] = RangeStatus.Open
            counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="source_mask", axis=1),
    arg_take_spec=dict(
        source_mask=ch.ArraySlicer(axis=1),
        target_mask=ch.ArraySlicer(axis=1),
        relation=None,
        incl_open=None,
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def between_two_ranges_nb(
    source_mask: tp.Array2d,
    target_mask: tp.Array2d,
    relation: int = SignalRelation.OneMany,
    incl_open: bool = False,
) -> tp.RecordArray:
    """Create a record array representing ranges between source and target masks.

    The function uses `relation_idxs_1d_nb` to determine paired indices from the `source_mask`
    and `target_mask` for each column. When valid index pairs are found, a closed range record is created.
    If only the source index is valid and `incl_open` is True, an open range record is recorded.

    Args:
        source_mask (Array2d): Boolean array indicating positions of source signals.
        target_mask (Array2d): Boolean array indicating positions of target signals.
        relation (int): Relation mode for pairing signals.

            See `vectorbtpro.signals.enums.SignalRelation`.
        incl_open (bool): Include an open range if no closing signal is found.

    Returns:
        RecordArray: Array of records containing range data.

            Has the `vectorbtpro.generic.enums.range_dt` dtype.

    !!! tip
        This function is parallelizable.
    """
    new_records = np.empty(source_mask.shape, dtype=range_dt)
    counts = np.full(source_mask.shape[1], 0, dtype=int_)

    for col in prange(source_mask.shape[1]):
        _, _, source_idxs, target_idsx = relation_idxs_1d_nb(
            source_mask[:, col],
            target_mask[:, col],
            relation=relation,
        )
        for i in range(len(source_idxs)):
            if source_idxs[i] != -1 and target_idsx[i] != -1:
                r = counts[col]
                new_records["id"][r, col] = r
                new_records["col"][r, col] = col
                new_records["start_idx"][r, col] = source_idxs[i]
                new_records["end_idx"][r, col] = target_idsx[i]
                new_records["status"][r, col] = RangeStatus.Closed
                counts[col] += 1
            elif source_idxs[i] != -1 and target_idsx[i] == -1 and incl_open:
                r = counts[col]
                new_records["id"][r, col] = r
                new_records["col"][r, col] = col
                new_records["start_idx"][r, col] = source_idxs[i]
                new_records["end_idx"][r, col] = source_mask.shape[0] - 1
                new_records["status"][r, col] = RangeStatus.Open
                counts[col] += 1
    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1)),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def partition_ranges_nb(mask: tp.Array2d) -> tp.RecordArray:
    """Create a record array representing partitions of signals in `mask`.

    The function scans each column of the boolean array `mask` to identify contiguous
    sequences of signals. Each identified partition is recorded with its start and end indices;
    partitions ending before the last index are marked as closed, while those extending to the
    end are marked as open.

    Args:
        mask (Array2d): Boolean array where True indicates the presence of a signal.

    Returns:
        RecordArray: Array of records containing partition range data.

            Has the `vectorbtpro.generic.enums.range_dt` dtype.

    !!! tip
        This function is parallelizable.
    """
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=int_)

    for col in prange(mask.shape[1]):
        is_partition = False
        from_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if not is_partition:
                    from_i = i
                is_partition = True
            elif is_partition:
                to_i = i
                r = counts[col]
                new_records["id"][r, col] = r
                new_records["col"][r, col] = col
                new_records["start_idx"][r, col] = from_i
                new_records["end_idx"][r, col] = to_i
                new_records["status"][r, col] = RangeStatus.Closed
                counts[col] += 1
                is_partition = False
            if i == mask.shape[0] - 1:
                if is_partition:
                    to_i = mask.shape[0] - 1
                    r = counts[col]
                    new_records["id"][r, col] = r
                    new_records["col"][r, col] = col
                    new_records["start_idx"][r, col] = from_i
                    new_records["end_idx"][r, col] = to_i
                    new_records["status"][r, col] = RangeStatus.Open
                    counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1)),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def between_partition_ranges_nb(mask: tp.Array2d) -> tp.RecordArray:
    """Create a record array representing ranges between partitions in `mask`.

    The function identifies the gaps between consecutive partitions in the boolean array `mask`.
    When a new partition starts after an existing one, a range is recorded from the previous
    partition start to the current index.

    Args:
        mask (Array2d): Boolean array indicating partitions with signal presence.

    Returns:
        RecordArray: Array of records containing range data between partitions.

            Has the `vectorbtpro.generic.enums.range_dt` dtype.

    !!! tip
        This function is parallelizable.
    """
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=int_)

    for col in prange(mask.shape[1]):
        is_partition = False
        from_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if not is_partition and from_i != -1:
                    to_i = i
                    r = counts[col]
                    new_records["id"][r, col] = r
                    new_records["col"][r, col] = col
                    new_records["start_idx"][r, col] = from_i
                    new_records["end_idx"][r, col] = to_i
                    new_records["status"][r, col] = RangeStatus.Closed
                    counts[col] += 1
                is_partition = True
                from_i = i
            else:
                is_partition = False

    return generic_nb.repartition_nb(new_records, counts)


# ############# Raveling ############# #


@register_jitted(cache=True)
def unravel_nb(
    mask: tp.Array2d,
    incl_empty_cols: bool = True,
) -> tp.Tuple[
    tp.Array2d,
    tp.Array1d,
    tp.Array1d,
    tp.Array1d,
]:
    """Unravel each True value in a mask into a separate column.

    Args:
        mask (Array2d): Input 2D boolean mask.
        incl_empty_cols (bool): Whether to include columns that contain no resolved pairs.

    Returns:
        Tuple[Array2d, Array1d, Array1d, Array1d]: Tuple containing:

            * New mask with each True value unwrapped into a separate column.
            * Index position of each True value within its new column.
            * Row indices of each True value in the original mask.
            * Original column indices corresponding to each True value.
    """
    true_idxs = np.flatnonzero(mask.transpose())

    start_idxs = np.full(mask.shape[1], -1, dtype=int_)
    end_idxs = np.full(mask.shape[1], 0, dtype=int_)
    for i in range(len(true_idxs)):
        col = true_idxs[i] // mask.shape[0]
        if i == 0:
            prev_col = -1
        else:
            prev_col = true_idxs[i - 1] // mask.shape[0]
        if col != prev_col:
            start_idxs[col] = i
        end_idxs[col] = i + 1

    n_cols = (end_idxs - start_idxs).sum()
    new_mask = np.full((mask.shape[0], n_cols), False, dtype=np.bool_)
    range_ = np.full(n_cols, -1, dtype=int_)
    row_idxs = np.full(n_cols, -1, dtype=int_)
    col_idxs = np.empty(n_cols, dtype=int_)
    k = 0
    for i in range(len(start_idxs)):
        start_idx = start_idxs[i]
        end_idx = end_idxs[i]
        col_filled = False
        if start_idx != -1:
            for j in range(start_idx, end_idx):
                new_mask[true_idxs[j] % mask.shape[0], k] = True
                range_[k] = j - start_idx
                row_idxs[k] = true_idxs[j] % mask.shape[0]
                col_idxs[k] = true_idxs[j] // mask.shape[0]
                k += 1
                col_filled = True
        if not col_filled and incl_empty_cols:
            if k == 0:
                col_idxs[k] = 0
            else:
                col_idxs[k] = col_idxs[k - 1] + 1
            k += 1
    return new_mask[:, :k], range_[:k], row_idxs[:k], col_idxs[:k]


@register_jitted(cache=True)
def unravel_between_nb(
    mask: tp.Array2d,
    incl_open_source: bool = False,
    incl_empty_cols: bool = True,
) -> tp.Tuple[
    tp.Array2d,
    tp.Array1d,
    tp.Array1d,
    tp.Array1d,
    tp.Array1d,
    tp.Array1d,
]:
    """Unravel each pair of successive True values in a mask into a separate column.

    Args:
        mask (Array2d): Input 2D boolean mask.
        incl_open_source (bool): Flag to include the source True value even if a valid target is absent.
        incl_empty_cols (bool): Whether to include columns that contain no resolved pairs.

    Returns:
        Tuple[Array2d, Array1d, Array1d, Array1d, Array1d, Array1d]: Tuple containing:

            * New mask with each pair of successive True values unwrapped into a separate column.
            * Index position of the source True value within its new column.
            * Index position of the target True value within its new column.
            * Row indices of the source True values in the original mask.
            * Row indices of the target True values in the original mask.
            * Original column indices corresponding to each True value.
    """
    true_idxs = np.flatnonzero(mask.transpose())

    start_idxs = np.full(mask.shape[1], -1, dtype=int_)
    end_idxs = np.full(mask.shape[1], 0, dtype=int_)
    for i in range(len(true_idxs)):
        col = true_idxs[i] // mask.shape[0]
        if i == 0:
            prev_col = -1
        else:
            prev_col = true_idxs[i - 1] // mask.shape[0]
        if col != prev_col:
            start_idxs[col] = i
        end_idxs[col] = i + 1

    n_cols = (end_idxs - start_idxs).sum()
    new_mask = np.full((mask.shape[0], n_cols), False, dtype=np.bool_)
    source_range = np.full(n_cols, -1, dtype=int_)
    target_range = np.full(n_cols, -1, dtype=int_)
    source_idxs = np.full(n_cols, -1, dtype=int_)
    target_idxs = np.full(n_cols, -1, dtype=int_)
    col_idxs = np.empty(n_cols, dtype=int_)
    k = 0
    for i in range(len(start_idxs)):
        start_idx = start_idxs[i]
        end_idx = end_idxs[i]
        col_filled = False
        if start_idx != -1:
            for j in range(start_idx, end_idx):
                if j == end_idx - 1 and not incl_open_source:
                    continue
                new_mask[true_idxs[j] % mask.shape[0], k] = True
                source_range[k] = j - start_idx
                source_idxs[k] = true_idxs[j] % mask.shape[0]
                if j < end_idx - 1:
                    new_mask[true_idxs[j + 1] % mask.shape[0], k] = True
                    target_range[k] = j + 1 - start_idx
                    target_idxs[k] = true_idxs[j + 1] % mask.shape[0]
                col_idxs[k] = true_idxs[j] // mask.shape[0]
                k += 1
                col_filled = True
        if not col_filled and incl_empty_cols:
            if k == 0:
                col_idxs[k] = 0
            else:
                col_idxs[k] = col_idxs[k - 1] + 1
            k += 1
    return (
        new_mask[:, :k],
        source_range[:k],
        target_range[:k],
        source_idxs[:k],
        target_idxs[:k],
        col_idxs[:k],
    )


@register_jitted(cache=True)
def unravel_between_two_nb(
    source_mask: tp.Array2d,
    target_mask: tp.Array2d,
    relation: int = SignalRelation.OneMany,
    incl_open_source: bool = False,
    incl_open_target: bool = False,
    incl_empty_cols: bool = True,
) -> tp.Tuple[
    tp.Array2d,
    tp.Array2d,
    tp.Array1d,
    tp.Array1d,
    tp.Array1d,
    tp.Array1d,
    tp.Array1d,
]:
    """Unravel each pair of successive True values between a source and target mask into separate columns.

    This function resolves index pairs with `relation_idxs_1d_nb` and returns
    new masks along with relevant indices.

    Args:
        source_mask (Array2d): 2D boolean mask for source signals.
        target_mask (Array2d): 2D boolean mask for target signals.
        relation (int): Relation mode for pairing signals.

            See `vectorbtpro.signals.enums.SignalRelation`.
        incl_open_source (bool): Flag to include the source True value even if a valid target is absent.
        incl_open_target (bool): Include open target signals when a matching source is not found.
        incl_empty_cols (bool): Whether to include columns that contain no resolved pairs.

    Returns:
        Tuple[Array2d, Array2d, Array1d, Array1d, Array1d, Array1d, Array1d]: Tuple containing:

            * New source mask.
            * New target mask.
            * Array of source range values.
            * Array of target range values.
            * Array of source indices.
            * Array of target indices.
            * Array of original column indices.
    """
    if relation == SignalRelation.Chain or relation == SignalRelation.AnyChain:
        max_signals = source_mask.shape[0] * 2
    else:
        max_signals = source_mask.shape[0]
    source_range_2d = np.empty((max_signals, source_mask.shape[1]), dtype=int_)
    target_range_2d = np.empty((max_signals, source_mask.shape[1]), dtype=int_)
    source_idxs_2d = np.empty((max_signals, source_mask.shape[1]), dtype=int_)
    target_idxs_2d = np.empty((max_signals, source_mask.shape[1]), dtype=int_)
    counts = np.empty(source_mask.shape[1], dtype=int_)
    n_cols = 0

    for col in range(source_mask.shape[1]):
        source_range_col, target_range_col, source_idxs_col, target_idxs_col = relation_idxs_1d_nb(
            source_mask[:, col],
            target_mask[:, col],
            relation=relation,
        )
        n_idxs = len(source_idxs_col)
        source_range_2d[:n_idxs, col] = source_range_col
        target_range_2d[:n_idxs, col] = target_range_col
        source_idxs_2d[:n_idxs, col] = source_idxs_col
        target_idxs_2d[:n_idxs, col] = target_idxs_col
        counts[col] = n_idxs
        if n_idxs == 0:
            n_cols += 1
        else:
            n_cols += n_idxs

    new_source_mask = np.full((source_mask.shape[0], n_cols), False, dtype=np.bool_)
    new_target_mask = np.full((source_mask.shape[0], n_cols), False, dtype=np.bool_)
    source_range = np.full(n_cols, -1, dtype=int_)
    target_range = np.full(n_cols, -1, dtype=int_)
    source_idxs = np.full(n_cols, -1, dtype=int_)
    target_idxs = np.full(n_cols, -1, dtype=int_)
    col_idxs = np.empty(n_cols, dtype=int_)
    k = 0
    for c in range(len(counts)):
        col_filled = False
        if counts[c] > 0:
            for j in range(counts[c]):
                source_idx = source_idxs_2d[j, c]
                target_idx = target_idxs_2d[j, c]
                if source_idx != -1 and target_idx != -1:
                    new_source_mask[source_idx, k] = True
                    new_target_mask[target_idx, k] = True
                    source_range[k] = source_range_2d[j, c]
                    target_range[k] = target_range_2d[j, c]
                    source_idxs[k] = source_idx
                    target_idxs[k] = target_idx
                    col_idxs[k] = c
                    k += 1
                    col_filled = True
                elif source_idx != -1 and incl_open_source:
                    new_source_mask[source_idx, k] = True
                    source_range[k] = source_range_2d[j, c]
                    source_idxs[k] = source_idx
                    col_idxs[k] = c
                    k += 1
                    col_filled = True
                elif target_idx != -1 and incl_open_target:
                    new_target_mask[target_idx, k] = True
                    target_range[k] = target_range_2d[j, c]
                    target_idxs[k] = target_idx
                    col_idxs[k] = c
                    k += 1
                    col_filled = True
        if not col_filled and incl_empty_cols:
            col_idxs[k] = c
            k += 1
    return (
        new_source_mask[:, :k],
        new_target_mask[:, :k],
        source_range[:k],
        target_range[:k],
        source_idxs[:k],
        target_idxs[:k],
        col_idxs[:k],
    )


@register_jitted(cache=True, tags={"can_parallel"})
def ravel_nb(mask: tp.Array2d, group_map: tp.GroupMap) -> tp.Array2d:
    """Ravel True values into separate columns for each group.

    Args:
        mask (Array2d): 2D boolean mask to process.
        group_map (GroupMap): Tuple of indices and lengths for each group.

    Returns:
        Array2d: 2D boolean mask with True values raveled into individual columns per group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.full((mask.shape[0], len(group_lens)), False, dtype=np.bool_)

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for col in col_idxs:
            for i in range(mask.shape[0]):
                if mask[i, col]:
                    out[i, group] = True
    return out


# ############# Index ############# #


@register_jitted(cache=True)
def nth_index_1d_nb(mask: tp.Array1d, n: int) -> int:
    """Return the index of the n-th True value in a 1D boolean mask.

    Args:
        mask (Array1d): 1D boolean array.
        n (int): Zero-based index of the True value to locate.

            Negative values count from the end.

    Returns:
        int: Index of the n-th True value, or -1 if it does not exist.

    !!! note
        `n` starts with 0 and can be negative.
    """
    if n >= 0:
        found = -1
        for i in range(mask.shape[0]):
            if mask[i]:
                found += 1
                if found == n:
                    return i
    else:
        found = 0
        for i in range(mask.shape[0] - 1, -1, -1):
            if mask[i]:
                found -= 1
                if found == n:
                    return i
    return -1


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1), n=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nth_index_nb(mask: tp.Array2d, n: int) -> tp.Array1d:
    """Return the index of the n-th True value for each column in a 2D boolean mask.

    Args:
        mask (Array2d): 2D boolean array to evaluate column-wise.
        n (int): Zero-based index of the True value to locate in each column.

    Returns:
        Array1d: Array of indices corresponding to the n-th True value in each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(mask.shape[1], dtype=int_)
    for col in prange(mask.shape[1]):
        out[col] = nth_index_1d_nb(mask[:, col], n)
    return out


@register_jitted(cache=True)
def norm_avg_index_1d_nb(mask: tp.Array1d) -> float:
    """Return the normalized mean index of True values in a 1D mask.

    Args:
        mask (Array1d): 1D boolean array.

    Returns:
        float: Normalized average index scaled to the range (-1, 1).
    """
    mean_index = np.mean(np.flatnonzero(mask))
    return rescale_nb(mean_index, (0, len(mask) - 1), (-1, 1))


@register_chunkable(
    size=ch.ArraySizer(arg_query="mask", axis=1),
    arg_take_spec=dict(mask=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def norm_avg_index_nb(mask: tp.Array2d) -> tp.Array1d:
    """Return normalized average indices for each column in a 2D boolean mask.

    Args:
        mask (Array2d): 2D boolean mask.

    Returns:
        Array1d: Array of normalized mean indices for each column scaled to (-1, 1).

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(mask.shape[1], dtype=float_)
    for col in prange(mask.shape[1]):
        out[col] = norm_avg_index_1d_nb(mask[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        mask=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def norm_avg_index_grouped_nb(mask: tp.Array2d, group_lens: tp.Array1d) -> tp.Array1d:
    """Return normalized average indices for each group in a 2D boolean mask.

    Args:
        mask (Array2d): 2D boolean mask.
        group_lens (ArrayLike): 1D array of group lengths indicating the number of columns per group.

    Returns:
        Array1d: Array of normalized average indices for each group scaled to (-1, 1).

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(len(group_lens), dtype=float_)
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        temp_sum = 0
        temp_cnt = 0
        for col in range(from_col, to_col):
            for i in range(mask.shape[0]):
                if mask[i, col]:
                    temp_sum += i
                    temp_cnt += 1
        out[group] = rescale_nb(temp_sum / temp_cnt, (0, mask.shape[0] - 1), (-1, 1))
    return out
