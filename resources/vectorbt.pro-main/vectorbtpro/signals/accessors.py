# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing custom Pandas accessors for signals.

This module provides custom Pandas accessors that extend `vectorbtpro.generic.accessors`.
They allow access to signal-related methods on Pandas Series and DataFrames as follows:

* `SignalsSRAccessor` -> `pd.Series.vbt.signals.*`
* `SignalsDFAccessor` -> `pd.DataFrame.vbt.signals.*`

```pycon
>>> from vectorbtpro import *

>>> # vectorbtpro.signals.accessors.SignalsAccessor.pos_rank
>>> pd.Series([False, True, True, True, False]).vbt.signals.pos_rank()
0   -1
1    0
2    1
3    2
4   -1
dtype: int64
```

!!! note
    The underlying Series/DataFrame must already represent a signal series with boolean data type.

    Grouping is only supported by methods that accept the `group_by` argument. Accessors do not utilize caching.

Run for the examples below:

```pycon
>>> mask = pd.DataFrame({
...     'a': [True, False, False, False, False],
...     'b': [True, False, True, False, True],
...     'c': [True, True, True, False, False]
... }, index=pd.date_range("2020", periods=5))
>>> mask
                a      b      c
2020-01-01   True   True   True
2020-01-02  False  False   True
2020-01-03  False   True   True
2020-01-04  False  False  False
2020-01-05  False   True  False
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `SignalsAccessor.metrics`.

```pycon
>>> mask.vbt.signals.stats(column='a')
Start                         2020-01-01 00:00:00
End                           2020-01-05 00:00:00
Period                            5 days 00:00:00
Total                                           1
Rate [%]                                     20.0
First Index                   2020-01-01 00:00:00
Last Index                    2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1.0
Distance: Min                                 NaT
Distance: Median                              NaT
Distance: Max                                 NaT
Total Partitions                                1
Partition Rate [%]                          100.0
Partition Length: Min             1 days 00:00:00
Partition Length: Median          1 days 00:00:00
Partition Length: Max             1 days 00:00:00
Partition Distance: Min                       NaT
Partition Distance: Median                    NaT
Partition Distance: Max                       NaT
Name: a, dtype: object
```

Compare with a target signal array:

```pycon
>>> mask.vbt.signals.stats(column='a', settings=dict(target=mask['b']))
Start                         2020-01-01 00:00:00
End                           2020-01-05 00:00:00
Period                            5 days 00:00:00
Total                                           1
Rate [%]                                     20.0
Total Overlapping                               1
Overlapping Rate [%]                    33.333333
First Index                   2020-01-01 00:00:00
Last Index                    2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1.0
Distance -> Target: Min           0 days 00:00:00
Distance -> Target: Median        2 days 00:00:00
Distance -> Target: Max           4 days 00:00:00
Total Partitions                                1
Partition Rate [%]                          100.0
Partition Length: Min             1 days 00:00:00
Partition Length: Median          1 days 00:00:00
Partition Length: Max             1 days 00:00:00
Partition Distance: Min                       NaT
Partition Distance: Median                    NaT
Partition Distance: Max                       NaT
Name: a, dtype: object
```

Return duration as a floating point value instead of a timedelta:

```pycon
>>> mask.vbt.signals.stats(column='a', settings=dict(to_timedelta=False))
Start                         2020-01-01 00:00:00
End                           2020-01-05 00:00:00
Period                                          5
Total                                           1
Rate [%]                                     20.0
First Index                   2020-01-01 00:00:00
Last Index                    2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1.0
Distance: Min                                 NaN
Distance: Median                              NaN
Distance: Max                                 NaN
Total Partitions                                1
Partition Rate [%]                          100.0
Partition Length: Min                         1.0
Partition Length: Median                      1.0
Partition Length: Max                         1.0
Partition Distance: Min                       NaN
Partition Distance: Median                    NaN
Partition Distance: Max                       NaN
Name: a, dtype: object
```

`SignalsAccessor.stats` also supports regrouping:

```pycon
>>> mask.vbt.signals.stats(column=0, group_by=[0, 0, 1])
Start                         2020-01-01 00:00:00
End                           2020-01-05 00:00:00
Period                            5 days 00:00:00
Total                                           4
Rate [%]                                     40.0
First Index                   2020-01-01 00:00:00
Last Index                    2020-01-05 00:00:00
Norm Avg Index [-1, 1]                      -0.25
Distance: Min                     2 days 00:00:00
Distance: Median                  2 days 00:00:00
Distance: Max                     2 days 00:00:00
Total Partitions                                4
Partition Rate [%]                          100.0
Partition Length: Min             1 days 00:00:00
Partition Length: Median          1 days 00:00:00
Partition Length: Max             1 days 00:00:00
Partition Distance: Min           2 days 00:00:00
Partition Distance: Median        2 days 00:00:00
Partition Distance: Max           2 days 00:00:00
Name: 0, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `SignalsAccessor.subplots`.

Subplots functionality is inherited from `vectorbtpro.generic.accessors.GenericAccessor`.
"""

from functools import partialmethod

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.accessors import (
    register_df_vbt_accessor,
    register_sr_vbt_accessor,
    register_vbt_accessor,
)
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base import indexes, reshaping
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import GenericAccessor, GenericDFAccessor, GenericSRAccessor
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.signals import enums, nb
from vectorbtpro.utils import checks
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import hybrid_method, hybrid_property
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.random_ import set_seed_nb
from vectorbtpro.utils.template import RepEval, substitute_templates
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "SignalsAccessor",
    "SignalsSRAccessor",
    "SignalsDFAccessor",
]

__pdoc__ = {}


@register_vbt_accessor("signals")
class SignalsAccessor(GenericAccessor):
    """Class representing an accessor on top of signal series for both Series and DataFrames.

    Accessible via `pd.Series.vbt.signals` and `pd.DataFrame.vbt.signals`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Underlying object.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor`.

    !!! info
        For default settings, see `vectorbtpro._settings.signals`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        GenericAccessor.__init__(self, wrapper, obj=obj, **kwargs)

        checks.assert_dtype(self._obj, np.bool_)

    @hybrid_property
    def sr_accessor_cls(cls_or_self) -> tp.Type["SignalsSRAccessor"]:
        return SignalsSRAccessor

    @hybrid_property
    def df_accessor_cls(cls_or_self) -> tp.Type["SignalsDFAccessor"]:
        return SignalsDFAccessor

    # ############# Overriding ############# #

    @classmethod
    def empty(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """Return an empty Series or DataFrame with bool dtype.

        Args:
            *args: Positional arguments for `vectorbtpro.generic.accessors.GenericAccessor.empty`.
            fill_value (bool): Fill value indicator.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.empty`.

        Returns:
            SeriesFrame: Empty Series or DataFrame with bool dtype.
        """
        return GenericAccessor.empty(*args, fill_value=fill_value, dtype=np.bool_, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """Return an empty-like Series or DataFrame with bool dtype.

        Args:
            *args: Positional arguments for `vectorbtpro.generic.accessors.GenericAccessor.empty_like`.
            fill_value (bool): Fill value indicator.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.empty_like`.

        Returns:
            SeriesFrame: Series or DataFrame with the same shape and bool dtype.
        """
        return GenericAccessor.empty_like(*args, fill_value=fill_value, dtype=np.bool_, **kwargs)

    bshift = partialmethod(GenericAccessor.bshift, fill_value=False)
    fshift = partialmethod(GenericAccessor.fshift, fill_value=False)
    ago = partialmethod(GenericAccessor.ago, fill_value=False)
    realign = partialmethod(GenericAccessor.realign, nan_value=False)

    # ############# Generation ############# #

    @classmethod
    def generate(
        cls,
        shape: tp.Union[tp.ShapeLike, ArrayWrapper],
        place_func_nb: tp.PlaceFunc,
        *args,
        place_args: tp.ArgsLike = None,
        only_once: bool = True,
        wait: int = 1,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Generate a signal Series or DataFrame using a Numba placement function.

        `shape` can be a shape-like tuple or an instance of `vectorbtpro.base.wrapping.ArrayWrapper`
        (which is used as `wrapper`).

        Arguments to `place_func_nb` can be passed either as `*args` or `place_args` (but not both).

        Args:
            shape (Union[ShapeLike, ArrayWrapper]): Desired shape as a tuple or
                an `vectorbtpro.base.wrapping.ArrayWrapper` instance.
            place_func_nb (PlaceFunc): Callback function for placing signals.
            *args: Alias for `place_args`.
            place_args (ArgsLike): Positional arguments for `place_func_nb`.
            only_once (bool): Whether to run the placement function only once.
            wait (int): Waiting period before signal placement.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Wrapped output containing generated signals with bool dtype.

        See:
            `vectorbtpro.signals.nb.generate_nb`

        Examples:
            Generate random signals manually:

            ```pycon
            >>> @njit
            ... def place_func_nb(c):
            ...     i = np.random.choice(len(c.out))
            ...     c.out[i] = True
            ...     return i

            >>> vbt.pd_acc.signals.generate(
            ...     (5, 3),
            ...     place_func_nb,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
                            a      b      c
            2020-01-01   True  False  False
            2020-01-02  False   True  False
            2020-01-03  False  False   True
            2020-01-04  False  False  False
            2020-01-05  False  False  False
            ```
        """
        if isinstance(shape, ArrayWrapper):
            wrapper = shape
            shape = wrapper.shape
        if len(args) > 0 and place_args is not None:
            raise ValueError("Must provide either *args or place_args, not both")
        if place_args is None:
            place_args = args
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        shape_2d = cls.resolve_shape(shape)
        if len(broadcast_named_args) > 0:
            broadcast_named_args = reshaping.broadcast(
                broadcast_named_args, to_shape=shape_2d, **broadcast_kwargs
            )
        template_context = merge_dicts(
            broadcast_named_args,
            dict(shape=shape, shape_2d=shape_2d, wait=wait),
            template_context,
        )
        place_args = substitute_templates(place_args, template_context, eval_id="place_args")
        func = jit_reg.resolve_option(nb.generate_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        result = func(
            target_shape=shape_2d,
            place_func_nb=place_func_nb,
            place_args=place_args,
            only_once=only_once,
            wait=wait,
        )

        if wrapper is None:
            wrapper = ArrayWrapper.from_shape(shape, ndim=cls.ndim)
        if wrap_kwargs is None:
            wrap_kwargs = resolve_dict(wrap_kwargs)
        return wrapper.wrap(result, **wrap_kwargs)

    @classmethod
    def generate_both(
        cls,
        shape: tp.Union[tp.ShapeLike, ArrayWrapper],
        entry_place_func_nb: tp.PlaceFunc,
        exit_place_func_nb: tp.PlaceFunc,
        *args,
        entry_place_args: tp.ArgsLike = None,
        exit_place_args: tp.ArgsLike = None,
        entry_wait: int = 1,
        exit_wait: int = 1,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """Generate entry and exit signals using Numba-compiled entry and exit placement functions.

        `shape` can be provided as either a shape-like tuple or an
        `vectorbtpro.base.wrapping.ArrayWrapper` instance; if an instance is provided,
        it is used as the output wrapper.

        Arguments to `entry_place_func_nb` can be passed either as `*args` or `entry_place_args` (but not both).
        The same applies to `exit_place_func_nb` with `exit_place_args`.

        Args:
            shape (Union[ShapeLike, ArrayWrapper]): Desired shape as a tuple or
                an `vectorbtpro.base.wrapping.ArrayWrapper` instance.
            entry_place_func_nb (PlaceFunc): Callback function for placing entry signals.
            exit_place_func_nb (PlaceFunc): Callback function for placing exit signals.
            args: Positional arguments forwarded to both entry and exit functions if
                explicit arguments are not provided.
            entry_place_args (ArgsLike): Positional arguments for `entry_place_func_nb`.
            exit_place_args (ArgsLike): Positional arguments for `exit_place_func_nb`.
            entry_wait (int): Number of periods to wait before an entry signal is triggered.

                !!! note
                    Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                    within the same bar, and exit can be processed before entry.
            exit_wait (int): Number of periods to wait before an exit signal is triggered.

                !!! note
                    Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                    within the same bar, and entry can be processed before exit.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            Tuple[SeriesFrame, SeriesFrame]: Tuple containing the entry signals and
                exit signals as wrapped arrays.

        See:
            `vectorbtpro.signals.nb.generate_enex_nb`

        Examples:
            Generate entry and exit signals one after another:

            ```pycon
            >>> @njit
            ... def place_func_nb(c):
            ...     c.out[0] = True
            ...     return 0

            >>> en, ex = vbt.pd_acc.signals.generate_both(
            ...     (5, 3),
            ...     entry_place_func_nb=place_func_nb,
            ...     exit_place_func_nb=place_func_nb,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
            >>> en
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False  False  False
            2020-01-03   True   True   True
            2020-01-04  False  False  False
            2020-01-05   True   True   True
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            ```

            Generate three entries and one exit one after another:

            ```pycon
            >>> @njit
            ... def entry_place_func_nb(c, n):
            ...     c.out[:n] = True
            ...     return n - 1

            >>> @njit
            ... def exit_place_func_nb(c, n):
            ...     c.out[:n] = True
            ...     return n - 1

            >>> en, ex = vbt.pd_acc.signals.generate_both(
            ...     (5, 3),
            ...     entry_place_func_nb=entry_place_func_nb,
            ...     entry_place_args=(3,),
            ...     exit_place_func_nb=exit_place_func_nb,
            ...     exit_place_args=(1,),
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
            >>> en
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02   True   True   True
            2020-01-03   True   True   True
            2020-01-04  False  False  False
            2020-01-05   True   True   True
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            ```
        """
        if isinstance(shape, ArrayWrapper):
            wrapper = shape
            shape = wrapper.shape
        if len(args) > 0 and entry_place_args is not None:
            raise ValueError("Must provide either *args or entry_place_args, not both")
        if len(args) > 0 and exit_place_args is not None:
            raise ValueError("Must provide either *args or exit_place_args, not both")
        if entry_place_args is None:
            entry_place_args = args
        if exit_place_args is None:
            exit_place_args = args
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        shape_2d = cls.resolve_shape(shape)
        if len(broadcast_named_args) > 0:
            broadcast_named_args = reshaping.broadcast(
                broadcast_named_args,
                to_shape=shape_2d,
                **broadcast_kwargs,
            )
        template_context = merge_dicts(
            broadcast_named_args,
            dict(
                shape=shape,
                shape_2d=shape_2d,
                entry_wait=entry_wait,
                exit_wait=exit_wait,
            ),
            template_context,
        )
        entry_place_args = substitute_templates(
            entry_place_args, template_context, eval_id="entry_place_args"
        )
        exit_place_args = substitute_templates(
            exit_place_args, template_context, eval_id="exit_place_args"
        )
        func = jit_reg.resolve_option(nb.generate_enex_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        result1, result2 = func(
            target_shape=shape_2d,
            entry_place_func_nb=entry_place_func_nb,
            entry_place_args=entry_place_args,
            exit_place_func_nb=exit_place_func_nb,
            exit_place_args=exit_place_args,
            entry_wait=entry_wait,
            exit_wait=exit_wait,
        )
        if wrapper is None:
            wrapper = ArrayWrapper.from_shape(shape, ndim=cls.ndim)
        if wrap_kwargs is None:
            wrap_kwargs = resolve_dict(wrap_kwargs)
        return wrapper.wrap(result1, **wrap_kwargs), wrapper.wrap(result2, **wrap_kwargs)

    def generate_exits(
        self,
        exit_place_func_nb: tp.PlaceFunc,
        *args,
        exit_place_args: tp.ArgsLike = None,
        wait: int = 1,
        until_next: bool = True,
        skip_until_exit: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Generate exit signals using a Numba-compiled exit placement function.

        Arguments to `exit_place_func_nb` can be passed either as `*args` or `exit_place_args` (but not both).

        Args:
            exit_place_func_nb (PlaceFunc): Callback function for placing exit signals.
            *args: Alias for `exit_place_args`.
            exit_place_args (ArgsLike): Positional arguments for `exit_place_func_nb`.
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
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Boolean DataFrame or Series indicating the generated exit signals.

        See:
            `vectorbtpro.signals.nb.generate_ex_nb`

        Examples:
            Generate an exit just before the next entry:

            ```pycon
            >>> @njit
            ... def exit_place_func_nb(c):
            ...     c.out[-1] = True
            ...     return len(c.out) - 1

            >>> mask.vbt.signals.generate_exits(exit_place_func_nb)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False   True  False
            2020-01-03  False  False  False
            2020-01-04  False   True  False
            2020-01-05   True  False   True
            ```
        """
        if len(args) > 0 and exit_place_args is not None:
            raise ValueError("Must provide either *args or exit_place_args, not both")
        if exit_place_args is None:
            exit_place_args = args
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        obj = self.obj
        if len(broadcast_named_args) > 0:
            broadcast_named_args = {"obj": obj, **broadcast_named_args}
            broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
            obj = broadcast_named_args["obj"]
        else:
            wrapper = self.wrapper
            obj = reshaping.to_2d_array(obj)
        template_context = merge_dicts(
            broadcast_named_args,
            dict(wait=wait, until_next=until_next, skip_until_exit=skip_until_exit),
            template_context,
        )
        exit_place_args = substitute_templates(
            exit_place_args, template_context, eval_id="exit_place_args"
        )
        func = jit_reg.resolve_option(nb.generate_ex_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        exits = func(
            entries=obj,
            exit_place_func_nb=exit_place_func_nb,
            exit_place_args=exit_place_args,
            wait=wait,
            until_next=until_next,
            skip_until_exit=skip_until_exit,
        )
        return wrapper.wrap(exits, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Cleaning ############# #

    @hybrid_method
    def clean(
        cls_or_self,
        *objs: tp.ArrayLike,
        force_first: bool = True,
        keep_conflicts: bool = False,
        reverse_order: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Clean signal arrays.

        Depending on the number of signal arrays provided, this function cleans the signals.
        If one array is provided, it processes it using `SignalsAccessor.first`.
        If two arrays (entries and exits) are provided, it cleans them using `vectorbtpro.signals.nb.clean_enex_nb`.

        Args:
            *objs (ArrayLike): One or two array-like objects representing signal data.

                When one array is provided, it is treated as the primary signal array;
                when two arrays are provided, they are treated as entry and exit signals respectively.
            force_first (bool): Determines whether the first signal is forced to precede its counterpart.
            keep_conflicts (bool): Determines if simultaneous signals are processed sequentially.
            reverse_order (bool): Determines whether to reverse the order of signals.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeTuple[SeriesFrame]: Cleaned signal array if one is provided,
                or a tuple of cleaned entries and exits if two arrays are provided.

        See:
            `vectorbtpro.signals.nb.clean_enex_nb`
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self.obj, *objs)

        if len(objs) == 1:
            obj = objs[0]
            if not isinstance(obj, (pd.Series, pd.DataFrame)):
                obj = ArrayWrapper.from_obj(obj).wrap(obj)
            return obj.vbt.signals.first(wrap_kwargs=wrap_kwargs, jitted=jitted, chunked=chunked)
        if len(objs) == 2:
            broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
            broadcasted_args, wrapper = reshaping.broadcast(
                dict(entries=objs[0], exits=objs[1]),
                return_wrapper=True,
                **broadcast_kwargs,
            )
            func = jit_reg.resolve_option(nb.clean_enex_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            entries_out, exits_out = func(
                entries=broadcasted_args["entries"],
                exits=broadcasted_args["exits"],
                force_first=force_first,
                keep_conflicts=keep_conflicts,
                reverse_order=reverse_order,
            )
            return (
                wrapper.wrap(entries_out, group_by=False, **wrap_kwargs),
                wrapper.wrap(exits_out, group_by=False, **wrap_kwargs),
            )
        raise ValueError("This method accepts either one or two arrays")

    # ############# Random signals ############# #

    @classmethod
    def generate_random(
        cls,
        shape: tp.Union[tp.ShapeLike, ArrayWrapper],
        n: tp.Optional[tp.ArrayLike] = None,
        prob: tp.Optional[tp.ArrayLike] = None,
        pick_first: bool = False,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Generate signals randomly.

        Generates random signals based on either a fixed number of signals per column or a probability.

        Args:
            shape (Union[ShapeLike, ArrayWrapper]): Desired shape as a tuple or
                an `vectorbtpro.base.wrapping.ArrayWrapper` instance.
            n (Optional[ArrayLike]): Number of signals to generate.

                Must broadcast to the number of columns.
            prob (Optional[ArrayLike]): Probability of generating a signal.

                Must broadcast to match the provided shape.
            pick_first (bool): If True, stop after placing the first signal.
            seed (Optional[int]): Random seed for deterministic output.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `SignalsAccessor.generate`.

        Returns:
            SeriesFrame: Series or DataFrame containing the generated signals.

        See:
            * `vectorbtpro.signals.nb.rand_place_nb` if `n` is provided.
            * `vectorbtpro.signals.nb.rand_by_prob_place_nb` if `prob` is provided.

        Examples:
            For each column, generate a variable number of signals:

            ```pycon
            >>> vbt.pd_acc.signals.generate_random(
            ...     (5, 3),
            ...     n=[0, 1, 2],
            ...     seed=42,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False   True
            2020-01-04  False   True  False
            2020-01-05  False  False   True
            ```

            For each row and column, pick a signal with 50% probability:

            ```pycon
            >>> vbt.pd_acc.signals.generate_random(
            ...     (5, 3),
            ...     prob=0.5,
            ...     seed=42,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False   True  False
            2020-01-03  False  False  False
            2020-01-04  False  False   True
            2020-01-05   True  False   True
            ```
        """
        if isinstance(shape, ArrayWrapper):
            if "wrapper" in kwargs:
                raise ValueError("Must provide wrapper either via shape or wrapper, not both")
            kwargs["wrapper"] = shape
            shape = kwargs["wrapper"].shape
        shape_2d = cls.resolve_shape(shape)
        if n is not None and prob is not None:
            raise ValueError("Must provide either n or prob, not both")

        if seed is not None:
            set_seed_nb(seed)
        if n is not None:
            n = reshaping.broadcast_array_to(n, shape_2d[1])
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(),
                    ),
                ),
            )
            return cls.generate(
                shape,
                jit_reg.resolve_option(nb.rand_place_nb, jitted),
                n,
                jitted=jitted,
                chunked=chunked,
                **kwargs,
            )
        if prob is not None:
            prob = reshaping.to_2d_array(reshaping.broadcast_array_to(prob, shape))
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                        None,
                    ),
                ),
            )
            return cls.generate(
                shape,
                jit_reg.resolve_option(nb.rand_by_prob_place_nb, jitted),
                prob,
                pick_first,
                jitted=jitted,
                chunked=chunked,
                **kwargs,
            )
        raise ValueError("Must provide at least n or prob")

    @classmethod
    def generate_random_both(
        cls,
        shape: tp.Union[tp.ShapeLike, ArrayWrapper],
        n: tp.Optional[tp.ArrayLike] = None,
        entry_prob: tp.Optional[tp.ArrayLike] = None,
        exit_prob: tp.Optional[tp.ArrayLike] = None,
        seed: tp.Optional[int] = None,
        entry_wait: int = 1,
        exit_wait: int = 1,
        entry_pick_first: bool = True,
        exit_pick_first: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """Generate chain of entry and exit signals randomly.

        Generates random entry and exit signals based on either a specified
        number of signals per column or given entry and exit probabilities. If `shape` is an
        `vectorbtpro.base.wrapping.ArrayWrapper`, it will be used as the wrapper; otherwise,
        the provided shape-like tuple is used to determine signal dimensions.

        Args:
            shape (Union[ShapeLike, ArrayWrapper]): Desired shape as a tuple or
                an `vectorbtpro.base.wrapping.ArrayWrapper` instance.
            n (Optional[ArrayLike]): Number of signals to generate.

                When provided, signals are generated using `vectorbtpro.signals.nb.generate_rand_enex_nb`.
            entry_prob (Optional[ArrayLike]): Probability of generating an entry signal.

                Must be provided with `exit_prob` to generate signals using
                `vectorbtpro.signals.nb.rand_by_prob_place_nb`.
            exit_prob (Optional[ArrayLike]): Probability of generating an exit signal.

                Must be provided with `entry_prob` to generate signals using
                `vectorbtpro.signals.nb.rand_by_prob_place_nb`.
            seed (Optional[int]): Random seed for deterministic output.
            entry_wait (int): Number of periods to wait before an entry signal is triggered.

                !!! note
                    Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                    within the same bar, and exit can be processed before entry.
            exit_wait (int): Number of periods to wait before an exit signal is triggered.

                !!! note
                    Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                    within the same bar, and entry can be processed before exit.
            entry_pick_first (bool): Whether to stop after generating the first entry signal.
            exit_pick_first (bool): Whether to stop after generating the first exit signal.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            Tuple[SeriesFrame, SeriesFrame]: Tuple containing the wrapped entry and exit signal arrays.

        See:
            * `vectorbtpro.signals.nb.generate_rand_enex_nb` if `n` is provided.
            * `vectorbtpro.signals.nb.rand_by_prob_place_nb` if `entry_prob` and `exit_prob` are provided.

        Examples:
            For each column, generate two entries and exits randomly:

            ```pycon
            >>> en, ex = vbt.pd_acc.signals.generate_random_both(
            ...     (5, 3),
            ...     n=2,
            ...     seed=42,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
            >>> en
                            a      b      c
            2020-01-01  False  False   True
            2020-01-02   True   True  False
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False   True
            2020-01-03   True   True  False
            2020-01-04  False  False  False
            2020-01-05   True   True   True
            ```

            For each row and column, pick entry with 50% probability and exit right after:

            ```pycon
            >>> en, ex = vbt.pd_acc.signals.generate_random_both(
            ...     (5, 3),
            ...     entry_prob=0.5,
            ...     exit_prob=1.,
            ...     seed=42,
            ...     wrap_kwargs=dict(
            ...         index=mask.index,
            ...         columns=mask.columns
            ...     )
            ... )
            >>> en
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04  False  False   True
            2020-01-05   True  False  False
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04  False  False  False
            2020-01-05  False  False   True
            ```
        """
        if isinstance(shape, ArrayWrapper):
            wrapper = shape
            shape = wrapper.shape
        shape_2d = cls.resolve_shape(shape)
        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError(
                "Must provide either n or any of the entry_prob and exit_prob, not both"
            )

        if seed is not None:
            set_seed_nb(seed)
        if n is not None:
            n = reshaping.broadcast_array_to(n, shape_2d[1])
            func = jit_reg.resolve_option(nb.generate_rand_enex_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            entries, exits = func(shape_2d, n, entry_wait, exit_wait)
            if wrapper is None:
                wrapper = ArrayWrapper.from_shape(shape, ndim=cls.ndim)
            if wrap_kwargs is None:
                wrap_kwargs = resolve_dict(wrap_kwargs)
            return wrapper.wrap(entries, **wrap_kwargs), wrapper.wrap(exits, **wrap_kwargs)
        elif entry_prob is not None and exit_prob is not None:
            entry_prob = reshaping.to_2d_array(reshaping.broadcast_array_to(entry_prob, shape))
            exit_prob = reshaping.to_2d_array(reshaping.broadcast_array_to(exit_prob, shape))
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                    ),
                    exit_place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                    ),
                ),
            )
            return cls.generate_both(
                shape,
                entry_place_func_nb=jit_reg.resolve_option(nb.rand_by_prob_place_nb, jitted),
                entry_place_args=(entry_prob, entry_pick_first),
                exit_place_func_nb=jit_reg.resolve_option(nb.rand_by_prob_place_nb, jitted),
                exit_place_args=(exit_prob, exit_pick_first),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                jitted=jitted,
                chunked=chunked,
                wrapper=wrapper,
                wrap_kwargs=wrap_kwargs,
            )
        raise ValueError("Must provide at least n, or entry_prob and exit_prob")

    def generate_random_exits(
        self,
        prob: tp.Optional[tp.ArrayLike] = None,
        seed: tp.Optional[int] = None,
        wait: int = 1,
        until_next: bool = True,
        skip_until_exit: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Generate exit signals randomly.

        Randomly generate exit signals for the input data. If `prob` is None, exit signals are generated
        using `vectorbtpro.signals.nb.rand_place_nb`; otherwise, exit signals are generated based on the
        provided probability using `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

        Uses `SignalsAccessor.generate_exits`.

        Specify `seed` to ensure deterministic output.

        Args:
            prob (Optional[ArrayLike]): Probability of generating a signal.
            seed (Optional[int]): Random seed for deterministic output.
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
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.generate_exits`.

        Returns:
            SeriesFrame: Generated exit signals.

        See:
            * `vectorbtpro.signals.nb.rand_place_nb` if `prob` is not provided.
            * `vectorbtpro.signals.nb.rand_by_prob_place_nb` if `prob` is provided.

        Examples:
            After each entry in `mask`, generate exactly one exit:

            ```pycon
            >>> mask.vbt.signals.generate_random_exits(seed=42)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False   True  False
            2020-01-03  False  False  False
            2020-01-04   True   True  False
            2020-01-05  False  False   True
            ```

            After each entry in `mask` and at each row, generate exit with 50% probability:

            ```pycon
            >>> mask.vbt.signals.generate_random_exits(prob=0.5, seed=42)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True  False  False
            2020-01-03  False  False  False
            2020-01-04  False  False  False
            2020-01-05  False  False   True
            ```
        """
        if seed is not None:
            set_seed_nb(seed)
        if prob is not None:
            broadcast_kwargs = merge_dicts(
                dict(keep_flex=dict(obj=False, prob=True)),
                broadcast_kwargs,
            )
            broadcasted_args = reshaping.broadcast(
                dict(obj=self.obj, prob=prob),
                **broadcast_kwargs,
            )
            obj = broadcasted_args["obj"]
            prob = broadcasted_args["prob"]
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    exit_place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                    )
                ),
            )
            return obj.vbt.signals.generate_exits(
                jit_reg.resolve_option(nb.rand_by_prob_place_nb, jitted),
                prob,
                True,
                wait=wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
        n = reshaping.broadcast_array_to(1, self.wrapper.shape_2d[1])
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(
                exit_place_args=ch.ArgsTaker(
                    base_ch.FlexArraySlicer(),
                )
            ),
        )
        return self.generate_exits(
            jit_reg.resolve_option(nb.rand_place_nb, jitted),
            n,
            wait=wait,
            until_next=until_next,
            skip_until_exit=skip_until_exit,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    # ############# Stop signals ############# #

    def generate_stop_exits(
        self,
        entry_ts: tp.ArrayLike,
        ts: tp.ArrayLike = np.nan,
        follow_ts: tp.ArrayLike = np.nan,
        stop: tp.ArrayLike = np.nan,
        trailing: tp.ArrayLike = False,
        out_dict: tp.Optional[tp.Dict[str, tp.ArrayLike]] = None,
        entry_wait: int = 1,
        exit_wait: int = 1,
        until_next: bool = True,
        skip_until_exit: bool = False,
        chain: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Generate exits based on when `ts` hits the stop.

        If `chain` is True, uses `SignalsAccessor.generate_both`; otherwise, uses `SignalsAccessor.generate_exits`.

        Use `out_dict` as a dictionary to pass the computed `stop_ts` array.
        You can also set `out_dict` to `{}` to produce this array automatically and still have access to it.

        All array-like arguments, including stops and `out_dict`, will broadcast using
        `vectorbtpro.base.reshaping.broadcast` and `broadcast_kwargs`.

        Args:
            entry_ts (ArrayLike): Entry time series.
            ts (ArrayLike): Time series used for evaluating stop signals.
            follow_ts (ArrayLike): Follow-up time series.
            stop (ArrayLike): Level(s) at which to trigger exit signals.
            trailing (ArrayLike): Flag or array indicating whether the stop is trailing.
            out_dict (Optional[Dict[str, ArrayLike]]): Dictionary to store additional output arrays,
                specifically the computed `stop_ts` array.

                You can pass an empty dictionary to automatically generate and access the `stop_ts` array.
            entry_wait (int): Number of periods to wait before an entry signal is triggered.

                !!! note
                    Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                    within the same bar, and exit can be processed before entry.
            exit_wait (int): Number of periods to wait before an exit signal is triggered.

                !!! note
                    Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                    within the same bar, and entry can be processed before exit.
            until_next (bool): Whether to place signals up to the next entry signal.

                !!! note
                    Setting it to False makes it difficult to tell which exit belongs to which entry.
            skip_until_exit (bool): Whether to skip processing entry signals until the next exit.

                Has only effect when `until_next` is disabled.

                !!! note
                    Setting it to True makes it impossible to tell which exit belongs to which entry.
            chain (bool): If True, chains signals by returning both new entries and exit signals
                using `SignalsAccessor.generate_both`.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.generate_both` or
                `SignalsAccessor.generate_exits`.

        Returns:
            MaybeTuple[SeriesFrame]: Exit signals array if `chain` is False, or
                a tuple containing new entries and exit signals if `chain` is True.

        See:
            * `vectorbtpro.signals.nb.first_place_nb` as entry placement function.
            * `vectorbtpro.signals.nb.stop_place_nb` as exit placement function.

        !!! hint
            Default arguments will generate an exit signal strictly between two entry signals.
            If both entry signals are too close to each other, no exit will be generated.

            To ignore all entries that come between an entry and its exit, set `until_next` to False
            and `skip_until_exit` to True.

            To remove all entries that come between an entry and its exit, set `chain` to True.
            This will return two arrays: new entries and exits.

        Examples:
            Regular stop loss:

            ```pycon
            >>> ts = pd.Series([1, 2, 3, 2, 1])

            >>> mask.vbt.signals.generate_stop_exits(ts, stop=-0.1)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04  False   True   True
            2020-01-05  False  False  False
            ```

            Trailing stop loss:

            ```pycon
            >>> mask.vbt.signals.generate_stop_exits(ts, stop=-0.1, trailing=True)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            ```

            Testing multiple take profit stops:

            ```pycon
            >>> mask.vbt.signals.generate_stop_exits(ts, stop=vbt.Param([1.0, 1.5]))
            stop                        1.0                  1.5
                            a      b      c      a      b      c
            2020-01-01  False  False  False  False  False  False
            2020-01-02   True   True  False  False  False  False
            2020-01-03  False  False  False   True  False  False
            2020-01-04  False  False  False  False  False  False
            2020-01-05  False  False  False  False  False  False
            ```
        """
        if wrap_kwargs is None:
            wrap_kwargs = {}
        entries = self.obj
        if out_dict is None:
            out_dict_passed = False
            out_dict = {}
        else:
            out_dict_passed = True
        stop_ts = out_dict.get("stop_ts", np.nan if out_dict_passed else None)

        broadcastable_args = dict(
            entries=entries,
            entry_ts=entry_ts,
            ts=ts,
            follow_ts=follow_ts,
            stop=stop,
            trailing=trailing,
            stop_ts=stop_ts,
        )
        broadcast_kwargs = merge_dicts(
            dict(
                keep_flex=dict(entries=False, stop_ts=False, _def=True),
                require_kwargs=dict(requirements="W"),
            ),
            broadcast_kwargs,
        )
        broadcasted_args = reshaping.broadcast(broadcastable_args, **broadcast_kwargs)
        entries = broadcasted_args["entries"]
        stop_ts = broadcasted_args["stop_ts"]
        if stop_ts is None:
            stop_ts = np.empty_like(entries, dtype=float_)
        stop_ts = reshaping.to_2d_array(stop_ts)

        entries_arr = reshaping.to_2d_array(entries)
        wrapper = ArrayWrapper.from_obj(entries)
        if chain:
            if checks.is_series(entries):
                cls = self.sr_accessor_cls
            else:
                cls = self.df_accessor_cls
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_place_args=ch.ArgsTaker(
                        ch.ArraySlicer(axis=1),
                    ),
                    exit_place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                    ),
                ),
            )
            out_dict["stop_ts"] = wrapper.wrap(stop_ts, group_by=False, **wrap_kwargs)
            return cls.generate_both(
                entries.shape,
                entry_place_func_nb=jit_reg.resolve_option(nb.first_place_nb, jitted),
                entry_place_args=(entries_arr,),
                exit_place_func_nb=jit_reg.resolve_option(nb.stop_place_nb, jitted),
                exit_place_args=(
                    broadcasted_args["entry_ts"],
                    broadcasted_args["ts"],
                    broadcasted_args["follow_ts"],
                    stop_ts,
                    broadcasted_args["stop"],
                    broadcasted_args["trailing"],
                ),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                wrapper=wrapper,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
        else:
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    exit_place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                    )
                ),
            )
            if skip_until_exit and until_next:
                warn("skip_until_exit=True has only effect when until_next=False")
            out_dict["stop_ts"] = wrapper.wrap(stop_ts, group_by=False, **wrap_kwargs)
            return entries.vbt.signals.generate_exits(
                jit_reg.resolve_option(nb.stop_place_nb, jitted),
                broadcasted_args["entry_ts"],
                broadcasted_args["ts"],
                broadcasted_args["follow_ts"],
                stop_ts,
                broadcasted_args["stop"],
                broadcasted_args["trailing"],
                wait=exit_wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )

    def generate_ohlc_stop_exits(
        self,
        entry_price: tp.ArrayLike,
        open: tp.ArrayLike = np.nan,
        high: tp.ArrayLike = np.nan,
        low: tp.ArrayLike = np.nan,
        close: tp.ArrayLike = np.nan,
        sl_stop: tp.ArrayLike = np.nan,
        tsl_th: tp.ArrayLike = np.nan,
        tsl_stop: tp.ArrayLike = np.nan,
        tp_stop: tp.ArrayLike = np.nan,
        reverse: tp.ArrayLike = False,
        is_entry_open: bool = False,
        out_dict: tp.Optional[tp.Dict[str, tp.ArrayLike]] = None,
        entry_wait: int = 1,
        exit_wait: int = 1,
        until_next: bool = True,
        skip_until_exit: bool = False,
        chain: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Generate exits based on when the price hits (trailing) stop loss or take profit.

        Generate exit signals when the price reaches stop loss, trailing stop, or take profit levels.
        If `out_dict` is provided, use it to pass the computed `stop_price` and `stop_type` arrays.
        Providing an empty dictionary for `out_dict` will automatically generate these arrays.

        If `chain` is True, this method utilizes `SignalsAccessor.generate_both`; otherwise,
        it calls `SignalsAccessor.generate_exits`. All array-like arguments, including stops and
        `out_dict`, are broadcasted using `vectorbtpro.base.reshaping.broadcast` with `broadcast_kwargs`.

        Args:
            entry_price (ArrayLike): Entry price array.
            open (ArrayLike): Array of open prices.
            high (ArrayLike): Array of high prices.
            low (ArrayLike): Array of low prices.
            close (ArrayLike): Array of close prices.
            sl_stop (ArrayLike): Stop loss level(s).
            tsl_th (ArrayLike): Trailing stop loss threshold(s).
            tsl_stop (ArrayLike): Trailing stop loss level(s).
            tp_stop (ArrayLike): Take profit level(s).
            reverse (ArrayLike): Flar or array indicating whether to reverse exit signals.
            is_entry_open (bool): Indicates if the entry price is available at or before open.

                If True, uses the bar's high and low; otherwise, uses only close.
            out_dict (Optional[Dict[str, ArrayLike]]): Dictionary to store `stop_price` and `stop_type` arrays.

                Providing an empty dictionary will result in these arrays being generated automatically.
            entry_wait (int): Number of periods to wait before an entry signal is triggered.

                !!! note
                    Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                    within the same bar, and exit can be processed before entry.
            exit_wait (int): Number of periods to wait before an exit signal is triggered.

                !!! note
                    Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                    within the same bar, and entry can be processed before exit.
            until_next (bool): Whether to place signals up to the next entry signal.

                !!! note
                    Setting it to False makes it difficult to tell which exit belongs to which entry.
            skip_until_exit (bool): Whether to skip processing entry signals until the next exit.

                Has only effect when `until_next` is disabled.

                !!! note
                    Setting it to True makes it impossible to tell which exit belongs to which entry.
            chain (bool): If True, chains signals by returning both new entries and exit signals
                using `SignalsAccessor.generate_both`.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.generate_both` or
                `SignalsAccessor.generate_exits`.

        Returns:
            MaybeTuple[SeriesFrame]: If `chain` is True, returns a tuple (`new_entries`, `exits`);
                otherwise, returns an array of exit signals.

        See:
            * `vectorbtpro.signals.nb.first_place_nb` as entry placement function.
            * `vectorbtpro.signals.nb.ohlc_stop_place_nb` as exit placement function.

        !!! hint
            Default arguments will generate an exit signal strictly between two entry signals.
            If both entry signals are too close to each other, no exit will be generated.

            To ignore all entries that occur between an entry and its exit,
            set `until_next` to False and `skip_until_exit` to True.

            To remove all intermediate entries between an entry and its exit,
            set `chain` to True. This will return two arrays: new entries and exits.

        !!! warning
            The last examples make entries dependent upon exits, which only makes sense
            if no other exit arrays are combined with this stop exit array.

        Examples:
            Generate exits for TSL and TP of 10%:

            ```pycon
            >>> price = pd.DataFrame({
            ...     'open': [10, 11, 12, 11, 10],
            ...     'high': [11, 12, 13, 12, 11],
            ...     'low': [9, 10, 11, 10, 9],
            ...     'close': [10, 11, 12, 11, 10]
            ... })
            >>> out_dict = {}
            >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
            ...     price["open"],
            ...     price['open'],
            ...     price['high'],
            ...     price['low'],
            ...     price['close'],
            ...     tsl_stop=0.1,
            ...     tp_stop=0.1,
            ...     is_entry_open=True,
            ...     out_dict=out_dict,
            ... )
            >>> exits
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True  False
            2020-01-03  False  False  False
            2020-01-04  False   True   True
            2020-01-05  False  False  False

            >>> out_dict['stop_price']
                           a     b     c
            2020-01-01   NaN   NaN   NaN
            2020-01-02  11.0  11.0   NaN
            2020-01-03   NaN   NaN   NaN
            2020-01-04   NaN  10.8  10.8
            2020-01-05   NaN   NaN   NaN

            >>> out_dict['stop_type'].vbt(mapping=vbt.sig_enums.StopType).apply_mapping()
                           a     b     c
            2020-01-01  None  None  None
            2020-01-02    TP    TP  None
            2020-01-03  None  None  None
            2020-01-04  None   TSL   TSL
            2020-01-05  None  None  None
            ```

            Notice how the first two entry signals in the third column have no exit signal -
            there is no room between them for an exit signal.

            To find an exit for the first entry and ignore all intermediate entries,
            pass `until_next=False` and `skip_until_exit=True`:

            ```pycon
            >>> out_dict = {}
            >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
            ...     price['open'],
            ...     price['open'],
            ...     price['high'],
            ...     price['low'],
            ...     price['close'],
            ...     tsl_stop=0.1,
            ...     tp_stop=0.1,
            ...     is_entry_open=True,
            ...     out_dict=out_dict,
            ...     until_next=False,
            ...     skip_until_exit=True
            ... )
            >>> exits
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04  False   True   True
            2020-01-05  False  False  False

            >>> out_dict['stop_price']
                           a     b     c
            2020-01-01   NaN   NaN   NaN
            2020-01-02  11.0  11.0  11.0
            2020-01-03   NaN   NaN   NaN
            2020-01-04   NaN  10.8  10.8
            2020-01-05   NaN   NaN   NaN

            >>> out_dict['stop_type'].vbt(mapping=vbt.sig_enums.StopType).apply_mapping()
                           a     b     c
            2020-01-01  None  None  None
            2020-01-02    TP    TP    TP
            2020-01-03  None  None  None
            2020-01-04  None   TSL   TSL
            2020-01-05  None  None  None
            ```

            Now, the first signal in the third column is executed regardless of subsequent entries,
            similar to the logic in `vectorbtpro.portfolio.base.Portfolio.from_signals`.

            To automatically remove intermediate entry signals, pass `chain=True`.
            This returns a tuple of new entries and exits:

            ```pycon
            >>> out_dict = {}
            >>> new_entries, exits = mask.vbt.signals.generate_ohlc_stop_exits(
            ...     price['open'],
            ...     price['open'],
            ...     price['high'],
            ...     price['low'],
            ...     price['close'],
            ...     tsl_stop=0.1,
            ...     tp_stop=0.1,
            ...     is_entry_open=True,
            ...     out_dict=out_dict,
            ...     chain=True
            ... )
            >>> new_entries
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False  False  False  << removed entry in the third column
            2020-01-03  False   True   True
            2020-01-04  False  False  False
            2020-01-05  False   True  False
            >>> exits
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04  False   True   True
            2020-01-05  False  False  False
            ```

            !!! warning
                The last two examples above make entries dependent upon exitsthis makes only sense
                if you have no other exit arrays to combine this stop exit array with.

            Test multiple parameter combinations:

            ```pycon
            >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
            ...     price['open'],
            ...     price['open'],
            ...     price['high'],
            ...     price['low'],
            ...     price['close'],
            ...     sl_stop=vbt.Param([False, 0.1]),
            ...     tsl_stop=vbt.Param([False, 0.1]),
            ...     is_entry_open=True
            ... )
            >>> exits
            sl_stop     False                                       0.1                \\
            tsl_stop    False                  0.1                False
                            a      b      c      a      b      c      a      b      c
            2020-01-01  False  False  False  False  False  False  False  False  False
            2020-01-02  False  False  False  False  False  False  False  False  False
            2020-01-03  False  False  False  False  False  False  False  False  False
            2020-01-04  False  False  False   True   True   True  False   True   True
            2020-01-05  False  False  False  False  False  False   True  False  False

            sl_stop
            tsl_stop      0.1
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04   True   True   True
            2020-01-05  False  False  False
            ```
        """
        if wrap_kwargs is None:
            wrap_kwargs = {}
        entries = self.obj
        if out_dict is None:
            out_dict_passed = False
            out_dict = {}
        else:
            out_dict_passed = True
        stop_price = out_dict.get("stop_price", np.nan if out_dict_passed else None)
        stop_type = out_dict.get("stop_type", -1 if out_dict_passed else None)

        broadcastable_args = dict(
            entries=entries,
            entry_price=entry_price,
            open=open,
            high=high,
            low=low,
            close=close,
            sl_stop=sl_stop,
            tsl_th=tsl_th,
            tsl_stop=tsl_stop,
            tp_stop=tp_stop,
            reverse=reverse,
            stop_price=stop_price,
            stop_type=stop_type,
        )
        broadcast_kwargs = merge_dicts(
            dict(
                keep_flex=dict(entries=False, stop_price=False, stop_type=False, _def=True),
                require_kwargs=dict(requirements="W"),
            ),
            broadcast_kwargs,
        )
        broadcasted_args = reshaping.broadcast(broadcastable_args, **broadcast_kwargs)
        entries = broadcasted_args["entries"]
        stop_price = broadcasted_args["stop_price"]
        if stop_price is None:
            stop_price = np.empty_like(entries, dtype=float_)
        stop_price = reshaping.to_2d_array(stop_price)
        stop_type = broadcasted_args["stop_type"]
        if stop_type is None:
            stop_type = np.empty_like(entries, dtype=int_)
        stop_type = reshaping.to_2d_array(stop_type)

        entries_arr = reshaping.to_2d_array(entries)
        wrapper = ArrayWrapper.from_obj(entries)
        if chain:
            if checks.is_series(entries):
                cls = self.sr_accessor_cls
            else:
                cls = self.df_accessor_cls
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_place_args=ch.ArgsTaker(
                        ch.ArraySlicer(axis=1),
                    ),
                    exit_place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                    ),
                ),
            )
            new_entries, exits = cls.generate_both(
                entries.shape,
                entry_place_func_nb=jit_reg.resolve_option(nb.first_place_nb, jitted),
                entry_place_args=(entries_arr,),
                exit_place_func_nb=jit_reg.resolve_option(nb.ohlc_stop_place_nb, jitted),
                exit_place_args=(
                    broadcasted_args["entry_price"],
                    broadcasted_args["open"],
                    broadcasted_args["high"],
                    broadcasted_args["low"],
                    broadcasted_args["close"],
                    stop_price,
                    stop_type,
                    broadcasted_args["sl_stop"],
                    broadcasted_args["tsl_th"],
                    broadcasted_args["tsl_stop"],
                    broadcasted_args["tp_stop"],
                    broadcasted_args["reverse"],
                    is_entry_open,
                ),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                wrapper=wrapper,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
            out_dict["stop_price"] = wrapper.wrap(stop_price, group_by=False, **wrap_kwargs)
            out_dict["stop_type"] = wrapper.wrap(stop_type, group_by=False, **wrap_kwargs)
            return new_entries, exits
        else:
            if skip_until_exit and until_next:
                warn("skip_until_exit=True has only effect when until_next=False")
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    exit_place_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1),
                        None,
                    )
                ),
            )
            exits = entries.vbt.signals.generate_exits(
                jit_reg.resolve_option(nb.ohlc_stop_place_nb, jitted),
                broadcasted_args["entry_price"],
                broadcasted_args["open"],
                broadcasted_args["high"],
                broadcasted_args["low"],
                broadcasted_args["close"],
                stop_price,
                stop_type,
                broadcasted_args["sl_stop"],
                broadcasted_args["tsl_th"],
                broadcasted_args["tsl_stop"],
                broadcasted_args["tp_stop"],
                broadcasted_args["reverse"],
                is_entry_open,
                wait=exit_wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
            out_dict["stop_price"] = wrapper.wrap(stop_price, group_by=False, **wrap_kwargs)
            out_dict["stop_type"] = wrapper.wrap(stop_type, group_by=False, **wrap_kwargs)
            return exits

    # ############# Ranking ############# #

    def rank(
        self,
        rank_func_nb: tp.RankFunc,
        *args,
        rank_args: tp.ArgsLike = None,
        reset_by: tp.Optional[tp.ArrayLike] = None,
        after_false: bool = False,
        after_reset: bool = False,
        reset_wait: int = 1,
        as_mapped: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Compute signal ranks.

        Arguments to `rank_func_nb` can be passed either as `*args` or `rank_args` (but not both).

        The input is broadcast using `vectorbtpro.base.reshaping.broadcast` if `reset_by` is specified.

        Optionally, the returned result can be converted to a
        `vectorbtpro.records.mapped_array.MappedArray` when `as_mapped` is True.

        Args:
            rank_func_nb (RankFunc): Compiled function for ranking.
            *args: Alias for `rank_args`.
            rank_args (ArgsLike): Positional arguments for `rank_func_nb`.
            reset_by (Optional[ArrayLike]): Boolean array indicating reset positions.
            after_false (bool): If True, disregards the first True partition with no preceding False.
            after_reset (bool): If True, disregards the first True partition before a reset signal.
            reset_wait (int): Offset to treat reset signals.

                * 0 treats the signal at reset as the first in the next partition.
                * 1 treats it as the last in the previous partition.
            as_mapped (bool): If True, return the result as a
                `vectorbtpro.records.mapped_array.MappedArray` instance.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.to_mapped`.

        Returns:
            Union[SeriesFrame, MappedArray]: Ranked positions as an array or a mapped array.

        See:
            `vectorbtpro.signals.nb.rank_nb`
        """
        if len(args) > 0 and rank_args is not None:
            raise ValueError("Must provide either *args or rank_args, not both")
        if rank_args is None:
            rank_args = args
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        if reset_by is not None:
            broadcast_named_args = {"obj": self.obj, "reset_by": reset_by, **broadcast_named_args}
        else:
            broadcast_named_args = {"obj": self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
        else:
            wrapper = self.wrapper
        obj = reshaping.to_2d_array(broadcast_named_args["obj"])
        if reset_by is not None:
            reset_by = reshaping.to_2d_array(broadcast_named_args["reset_by"])
        template_context = merge_dicts(
            dict(
                obj=obj,
                reset_by=reset_by,
                after_false=after_false,
                after_reset=after_reset,
                reset_wait=reset_wait,
            ),
            template_context,
        )
        rank_args = substitute_templates(rank_args, template_context, eval_id="rank_args")
        func = jit_reg.resolve_option(nb.rank_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        rank = func(
            mask=obj,
            rank_func_nb=rank_func_nb,
            rank_args=rank_args,
            reset_by=reset_by,
            after_false=after_false,
            after_reset=after_reset,
            reset_wait=reset_wait,
        )
        rank_wrapped = wrapper.wrap(rank, group_by=False, **wrap_kwargs)
        if as_mapped:
            rank_wrapped = rank_wrapped.replace(-1, np.nan)
            return rank_wrapped.vbt.to_mapped(dropna=True, dtype=int_, **kwargs)
        return rank_wrapped

    def pos_rank(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        allow_gaps: bool = False,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Compute signal position ranks.

        The ranking is performed partition-wise on True values.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            allow_gaps (bool): Flag to determine whether to allow gaps in ranking.
            **kwargs: Keyword arguments for `SignalsAccessor.rank`.

        Returns:
            Union[SeriesFrame, MappedArray]: Signal position ranks.

        See:
            `vectorbtpro.signals.nb.sig_pos_rank_nb`

        Examples:
            Rank each True value in each partition in `mask`:

            ```pycon
            >>> mask.vbt.signals.pos_rank()
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  1
            2020-01-03 -1  0  2
            2020-01-04 -1 -1 -1
            2020-01-05 -1  0 -1

            >>> mask.vbt.signals.pos_rank(after_false=True)
                        a  b  c
            2020-01-01 -1 -1 -1
            2020-01-02 -1 -1 -1
            2020-01-03 -1  0 -1
            2020-01-04 -1 -1 -1
            2020-01-05 -1  0 -1

            >>> mask.vbt.signals.pos_rank(allow_gaps=True)
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  1
            2020-01-03 -1  1  2
            2020-01-04 -1 -1 -1
            2020-01-05 -1  2 -1

            >>> mask.vbt.signals.pos_rank(reset_by=~mask, allow_gaps=True)
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  1
            2020-01-03 -1  0  2
            2020-01-04 -1 -1 -1
            2020-01-05 -1  0 -1
            ```
        """
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(
                rank_args=ch.ArgsTaker(
                    None,
                )
            ),
        )
        return self.rank(
            rank_func_nb=jit_reg.resolve_option(nb.sig_pos_rank_nb, jitted),
            rank_args=(allow_gaps,),
            jitted=jitted,
            chunked=chunked,
            **kwargs,
        )

    def pos_rank_after(
        self,
        reset_by: tp.ArrayLike,
        after_reset: bool = True,
        allow_gaps: bool = True,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Return signal position ranks computed after each signal specified in `reset_by`.

        Args:
            reset_by (ArrayLike): Array used to reset the ranking.
            after_reset (bool): If True, disregards the first True partition before a reset signal.
            allow_gaps (bool): Flag to determine whether to allow gaps in ranking.
            **kwargs: Keyword arguments for `pos_rank`.

        Returns:
            Union[SeriesFrame, MappedArray]: Signal position ranks after the reset.

        !!! note
            `allow_gaps` is enabled by default.
        """
        return self.pos_rank(
            reset_by=reset_by, after_reset=after_reset, allow_gaps=allow_gaps, **kwargs
        )

    def partition_pos_rank(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Compute partition position ranks.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `SignalsAccessor.rank`.

        Returns:
            Union[SeriesFrame, MappedArray]: Partition position ranks.

        See:
            `vectorbtpro.signals.nb.part_pos_rank_nb`

        Examples:
            Rank each partition of True values in `mask`:

            ```pycon
            >>> mask.vbt.signals.partition_pos_rank()
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  0
            2020-01-03 -1  1  0
            2020-01-04 -1 -1 -1
            2020-01-05 -1  2 -1

            >>> mask.vbt.signals.partition_pos_rank(after_false=True)
                        a  b  c
            2020-01-01 -1 -1 -1
            2020-01-02 -1 -1 -1
            2020-01-03 -1  0 -1
            2020-01-04 -1 -1 -1
            2020-01-05 -1  1 -1

            >>> mask.vbt.signals.partition_pos_rank(reset_by=mask)
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02 -1 -1  0
            2020-01-03 -1  0  0
            2020-01-04 -1 -1 -1
            2020-01-05 -1  0 -1
            ```
        """
        return self.rank(
            jit_reg.resolve_option(nb.part_pos_rank_nb, jitted),
            jitted=jitted,
            chunked=chunked,
            **kwargs,
        )

    def partition_pos_rank_after(
        self, reset_by: tp.ArrayLike, **kwargs
    ) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Return partition position ranks computed after each signal specified in `reset_by`.

        Args:
            reset_by (ArrayLike): Array used to reset the ranking.
            **kwargs: Keyword arguments for `SignalsAccessor.partition_pos_rank`.

        Returns:
            Union[SeriesFrame, MappedArray]: Partition position ranks after reset.
        """
        return self.partition_pos_rank(reset_by=reset_by, after_reset=True, **kwargs)

    def first(
        self,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == 0`.

        Args:
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank`.

        Returns:
            SeriesFrame: Wrapped array of signals with `pos_rank == 0`.
        """
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank == 0, group_by=False, **resolve_dict(wrap_kwargs))

    def first_after(
        self,
        reset_by: tp.ArrayLike,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == 0`.

        Args:
            reset_by (ArrayLike): Array used to reset the ranking.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank_after`.

        Returns:
            SeriesFrame: Wrapped array of signals with `pos_rank == 0`.
        """
        pos_rank = self.pos_rank_after(reset_by, **kwargs).values
        return self.wrapper.wrap(pos_rank == 0, group_by=False, **resolve_dict(wrap_kwargs))

    def nth(
        self,
        n: int,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == n`.

        Args:
            n (int): Specific position rank that signals must equal.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank`.

        Returns:
            SeriesFrame: Wrapped array of signals with `pos_rank == n`.
        """
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank == n, group_by=False, **resolve_dict(wrap_kwargs))

    def nth_after(
        self,
        n: int,
        reset_by: tp.ArrayLike,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == n`.

        Args:
            n (int): Specific position rank value.
            reset_by (ArrayLike): Array used to reset the ranking.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank_after`.

        Returns:
            SeriesFrame: Wrapped array of signals with `pos_rank == n`.
        """
        pos_rank = self.pos_rank_after(reset_by, **kwargs).values
        return self.wrapper.wrap(pos_rank == n, group_by=False, **resolve_dict(wrap_kwargs))

    def from_nth(
        self,
        n: int,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank >= n`.

        Args:
            n (int): Lower bound for the position rank.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank`.

        Returns:
            SeriesFrame: Wrapped array of signals with `pos_rank >= n`.
        """
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank >= n, group_by=False, **resolve_dict(wrap_kwargs))

    def from_nth_after(
        self,
        n: int,
        reset_by: tp.ArrayLike,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank >= n`.

        Args:
            n (int): Lower bound for the position rank.
            reset_by (ArrayLike): Array used to reset the ranking.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank_after`.

        Returns:
            SeriesFrame: Wrapped array of signals with `pos_rank >= n`.
        """
        pos_rank = self.pos_rank_after(reset_by, **kwargs).values
        return self.wrapper.wrap(pos_rank >= n, group_by=False, **resolve_dict(wrap_kwargs))

    def to_nth(
        self,
        n: int,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank < n`.

        Args:
            n (int): Upper bound for the position rank (exclusive).
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank`.

        Returns:
            SeriesFrame: Wrapped array of signals with `pos_rank < n`.
        """
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank < n, group_by=False, **resolve_dict(wrap_kwargs))

    def to_nth_after(
        self,
        n: int,
        reset_by: tp.ArrayLike,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank < n`.

        Args:
            n (int): Upper bound for the position rank (exclusive).
            reset_by (ArrayLike): Array used to reset the ranking.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank_after`.

        Returns:
            SeriesFrame: Wrapped array of signals with `pos_rank < n`.
        """
        pos_rank = self.pos_rank_after(reset_by, **kwargs).values
        return self.wrapper.wrap(pos_rank < n, group_by=False, **resolve_dict(wrap_kwargs))

    def pos_rank_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of signal position ranks.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `SignalsAccessor.pos_rank`.

        Returns:
            MappedArray: Mapped array of signal position ranks.
        """
        return self.pos_rank(as_mapped=True, group_by=group_by, **kwargs)

    def partition_pos_rank_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of partition position ranks.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `SignalsAccessor.partition_pos_rank`.

        Returns:
            MappedArray: Mapped array of partition position ranks.
        """
        return self.partition_pos_rank(as_mapped=True, group_by=group_by, **kwargs)

    # ############# Distance ############# #

    def distance_from_last(
        self,
        nth: int = 1,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Calculate the distance from the last signal occurrence.

        Args:
            nth (int): Index of the last True value to measure the distance from.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Wrapped array representing the distance from the nth last signal.

        See:
            `vectorbtpro.signals.nb.distance_from_last_nb`

        Examples:
            Get the distance to the last signal:

            ```pycon
            >>> mask.vbt.signals.distance_from_last()
                        a  b  c
            2020-01-01 -1 -1 -1
            2020-01-02  1  1  1
            2020-01-03  2  2  1
            2020-01-04  3  1  1
            2020-01-05  4  2  2
            ```

            Get the distance to the second last signal:

            ```pycon
            >>> mask.vbt.signals.distance_from_last(nth=2)
                        a  b  c
            2020-01-01 -1 -1 -1
            2020-01-02 -1 -1  1
            2020-01-03 -1  2  1
            2020-01-04 -1  3  2
            2020-01-05 -1  2  3
            ```
        """
        func = jit_reg.resolve_option(nb.distance_from_last_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        distance_from_last = func(self.to_2d_array(), nth=nth)
        return self.wrapper.wrap(distance_from_last, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Conversion ############# #

    def to_mapped(
        self,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> MappedArray:
        """Convert the signals into a mapped array.

        Convert this object into an instance of `vectorbtpro.records.mapped_array.MappedArray`
        by flattening the signal array and generating corresponding index and column arrays.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray`.

        Returns:
            MappedArray: Regrouped mapped array of signals.
        """
        mapped_arr = self.to_2d_array().flatten(order="F")
        col_arr = np.repeat(np.arange(self.wrapper.shape_2d[1]), self.wrapper.shape_2d[0])
        idx_arr = np.tile(np.arange(self.wrapper.shape_2d[0]), self.wrapper.shape_2d[1])
        new_mapped_arr = mapped_arr[mapped_arr]
        new_col_arr = col_arr[mapped_arr]
        new_idx_arr = idx_arr[mapped_arr]
        return MappedArray(
            wrapper=self.wrapper,
            mapped_arr=new_mapped_arr,
            col_arr=new_col_arr,
            idx_arr=new_idx_arr,
            **kwargs,
        ).regroup(group_by)

    # ############# Relation ############# #

    def get_relation_str(self, relation: tp.Union[int, str]) -> str:
        """Get the direction string corresponding to a signal relation.

        Args:
            relation (Union[int, str]): Relation mode for pairing signals.

                Mapped using `vectorbtpro.signals.enums.SignalRelation` if provided as a string.

        Returns:
            str: String indicating the relation direction.
        """
        if isinstance(relation, str):
            relation = map_enum_fields(relation, enums.SignalRelation)
        if relation == enums.SignalRelation.OneOne:
            return ">-<"
        if relation == enums.SignalRelation.OneMany:
            return "->"
        if relation == enums.SignalRelation.ManyOne:
            return "<-"
        if relation == enums.SignalRelation.ManyMany:
            return "<->"
        raise ValueError(f"Invalid relation: {relation}")

    # ############# Ranges ############# #

    def delta_ranges(
        self,
        delta: tp.Union[str, int, tp.FrequencyLike],
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Ranges:
        """Build a record array of ranges from a delta applied after each signal.

        Constructs a `vectorbtpro.generic.ranges.Ranges` object from a delta value that is
        applied after each signal (or before if the delta is negative).

        Args:
            delta (Union[str, int, FrequencyLike]): Delta value applied relative to each signal.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_delta`.

        Returns:
            Ranges: Regrouped record array representing the ranges.
        """
        return Ranges.from_delta(self.to_mapped(), delta=delta, **kwargs).regroup(group_by)

    def between_ranges(
        self,
        target: tp.Optional[tp.ArrayLike] = None,
        relation: tp.Union[int, str] = "onemany",
        incl_open: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        attach_target: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> Ranges:
        """Build a record array of ranges between signals.

        Args:
            target (Optional[ArrayLike]): Array-like target used for a two-range operation.
            relation (Union[int, str]): Relation mode for pairing signals.

                Mapped using `vectorbtpro.signals.enums.SignalRelation` if provided as a string.
            incl_open (bool): Include an open range if no closing signal is found.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            attach_target (bool): If True, the target array is attached to the result.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_records`.

        Returns:
            Ranges: `vectorbtpro.generic.ranges.Ranges` instance representing the computed ranges.

        See:
            * `vectorbtpro.signals.nb.between_ranges_nb` if `target` is not provided.
            * `vectorbtpro.signals.nb.between_two_ranges_nb` if `target` is provided.

        Examples:
            One array:

            ```pycon
            >>> mask_sr = pd.Series([True, False, False, True, False, True, True])
            >>> ranges = mask_sr.vbt.signals.between_ranges()
            >>> ranges
            <vectorbtpro.generic.ranges.Ranges at 0x7ff29ea7c7b8>

            >>> ranges.readable
               Range Id  Column  Start Index  End Index  Status
            0         0       0            0          3  Closed
            1         1       0            3          5  Closed
            2         2       0            5          6  Closed

            >>> ranges.duration.values
            array([3, 2, 1])
            ```

            Two arrays, traversing the signals of the first array:

            ```pycon
            >>> mask_sr1 = pd.Series([True, True, True, False, False])
            >>> mask_sr2 = pd.Series([False, False, True, False, True])
            >>> ranges = mask_sr1.vbt.signals.between_ranges(target=mask_sr2)
            >>> ranges
            <vectorbtpro.generic.ranges.Ranges at 0x7ff29e3b80f0>

            >>> ranges.readable
               Range Id  Column  Start Index  End Index  Status
            0         0       0            2          2  Closed
            1         1       0            2          4  Closed

            >>> ranges.duration.values
            array([0, 2])
            ```

            Two arrays, traversing the signals of the second array:

            ```pycon
            >>> ranges = mask_sr1.vbt.signals.between_ranges(target=mask_sr2, relation="manyone")
            >>> ranges
            <vectorbtpro.generic.ranges.Ranges at 0x7ff29eccbd68>

            >>> ranges.readable
               Range Id  Column  Start Index  End Index  Status
            0         0       0            0          2  Closed
            1         1       0            1          2  Closed
            2         2       0            2          2  Closed

            >>> ranges.duration.values
            array([0, 2])
            ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if isinstance(relation, str):
            relation = map_enum_fields(relation, enums.SignalRelation)

        if target is None:
            func = jit_reg.resolve_option(nb.between_ranges_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            range_records = func(self.to_2d_array(), incl_open=incl_open)
            wrapper = self.wrapper
            to_attach = self.obj
        else:
            broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
            broadcasted_args, wrapper = reshaping.broadcast(
                dict(obj=self.obj, target=target),
                return_wrapper=True,
                **broadcast_kwargs,
            )
            func = jit_reg.resolve_option(nb.between_two_ranges_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            range_records = func(
                broadcasted_args["obj"],
                broadcasted_args["target"],
                relation=relation,
                incl_open=incl_open,
            )
            to_attach = broadcasted_args["target"] if attach_target else broadcasted_args["obj"]
        kwargs = merge_dicts(dict(close=to_attach), kwargs)
        return Ranges.from_records(wrapper, range_records, **kwargs).regroup(group_by)

    def partition_ranges(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> Ranges:
        """Build a record array of ranges from signal partitions.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_records`.

        Returns:
            Ranges: `vectorbtpro.generic.ranges.Ranges` instance representing the partitioned ranges.

        See:
            `vectorbtpro.signals.nb.partition_ranges_nb`

        Examples:
            ```pycon
            >>> mask_sr = pd.Series([True, True, True, False, True, True])
            >>> mask_sr.vbt.signals.partition_ranges().readable
               Range Id  Column  Start Timestamp  End Timestamp  Status
            0         0       0                0              3  Closed
            1         1       0                4              5    Open
            ```
        """
        func = jit_reg.resolve_option(nb.partition_ranges_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        range_records = func(self.to_2d_array())
        kwargs = merge_dicts(dict(close=self.obj), kwargs)
        return Ranges.from_records(self.wrapper, range_records, **kwargs).regroup(group_by)

    def between_partition_ranges(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> Ranges:
        """Build a record array of ranges between partitions.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_records`.

        Returns:
            Ranges: `vectorbtpro.generic.ranges.Ranges` instance representing the computed ranges.

        See:
            `vectorbtpro.signals.nb.between_partition_ranges_nb`

        Examples:
            ```pycon
            >>> mask_sr = pd.Series([True, False, False, True, False, True, True])
            >>> mask_sr.vbt.signals.between_partition_ranges().readable
               Range Id  Column  Start Timestamp  End Timestamp  Status
            0         0       0                0              3  Closed
            1         1       0                3              5  Closed
            ```
        """
        func = jit_reg.resolve_option(nb.between_partition_ranges_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        range_records = func(self.to_2d_array())
        kwargs = merge_dicts(dict(close=self.obj), kwargs)
        return Ranges.from_records(self.wrapper, range_records, **kwargs).regroup(group_by)

    # ############# Raveling ############# #

    @classmethod
    def index_from_unravel(
        cls,
        range_: tp.Array1d,
        row_idxs: tp.Array1d,
        index: tp.Index,
        signal_index_type: str = "range",
        signal_index_name: str = "signal",
    ):
        """Get index from an unraveling operation.

        Args:
            range_ (Array1d): Array of range values for constructing the index.
            row_idxs (Array1d): Array of row indices used for selecting index labels when needed.
            index (Index): Reference Pandas index for deriving labels.
            signal_index_type (str): Type of signal index to generate.

                Allowed values:

                * "range": Basic signal counter in a column.
                * "position(s)": Row index of the signal in a column.
                * "label(s)": Label identifying the signal in a column.
            signal_index_name (str): Name to assign to the signal index.

        Returns:
            Index: Pandas Index constructed based on the specified signal index type.
        """
        if signal_index_type.lower() == "range":
            return pd.Index(range_, name=signal_index_name)
        if signal_index_type.lower() in ("position", "positions"):
            return pd.Index(row_idxs, name=signal_index_name)
        if signal_index_type.lower() in ("label", "labels"):
            if -1 in row_idxs:
                raise ValueError("Some columns have no signals. Use other signal index types.")
            return pd.Index(index[row_idxs], name=signal_index_name)
        raise ValueError(f"Invalid signal_index_type: '{signal_index_type}'")

    def unravel(
        self,
        incl_empty_cols: bool = True,
        force_signal_index: bool = False,
        signal_index_type: str = "range",
        signal_index_name: str = "signal",
        jitted: tp.JittedOption = None,
        clean_index_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Unravel signals.

        Args:
            incl_empty_cols (bool): Whether to include columns that contain no resolved pairs.
            force_signal_index (bool): Force creation of a new signal index even
                if the unraveled mask has the same shape as the original.
            signal_index_type (str): Type of signal index to generate.

                Allowed values:

                * "range": Basic signal counter in a column.
                * "position(s)": Row index of the signal in a column.
                * "label(s)": Label identifying the signal in a column.
            signal_index_name (str): Name to assign to the signal index.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeTuple[SeriesFrame]: Wrapped result containing the unraveled signals.

        See:
            `vectorbtpro.signals.nb.unravel_nb`

        Examples:
            ```pycon
            >>> mask.vbt.signals.unravel()
            signal          0      0      1      2      0      1      2
                            a      b      b      b      c      c      c
            2020-01-01   True   True  False  False   True  False  False
            2020-01-02  False  False  False  False  False   True  False
            2020-01-03  False  False   True  False  False  False   True
            2020-01-04  False  False  False  False  False  False  False
            2020-01-05  False  False  False   True  False  False  False
            ```
        """
        if clean_index_kwargs is None:
            clean_index_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        func = jit_reg.resolve_option(nb.unravel_nb, jitted)
        new_mask, range_, row_idxs, col_idxs = func(
            self.to_2d_array(), incl_empty_cols=incl_empty_cols
        )
        if new_mask.shape == self.wrapper.shape_2d and incl_empty_cols and not force_signal_index:
            return self.wrapper.wrap(new_mask)
        if not incl_empty_cols and (row_idxs == -1).all():
            raise ValueError("No columns left")
        signal_index = self.index_from_unravel(
            range_,
            row_idxs,
            self.wrapper.index,
            signal_index_type=signal_index_type,
            signal_index_name=signal_index_name,
        )
        new_columns = indexes.stack_indexes(
            (signal_index, self.wrapper.columns[col_idxs]), **clean_index_kwargs
        )
        return self.wrapper.wrap(new_mask, columns=new_columns, group_by=False, **wrap_kwargs)

    @hybrid_method
    def unravel_between(
        cls_or_self,
        *objs: tp.ArrayLike,
        relation: tp.Union[int, str] = "onemany",
        incl_open_source: bool = False,
        incl_open_target: bool = False,
        incl_empty_cols: bool = True,
        broadcast_kwargs: tp.KwargsLike = None,
        force_signal_index: bool = False,
        signal_index_type: str = "pair_range",
        signal_index_name: str = "signal",
        jitted: tp.JittedOption = None,
        clean_index_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Unravel signal pairs.

        Selects the appropriate unraveling method based on the number of input arrays.

        Args:
            *objs (ArrayLike): One or two array-like objects representing signal data.

                When one array is provided, it is treated as the primary signal array;
                when two arrays are provided, they are treated as source and target signals respectively.
            relation (Union[int, str]): Relation mode for pairing signals.

                Mapped using `vectorbtpro.signals.enums.SignalRelation` if provided as a string.
            incl_open_source (bool): Flag to include the source True value even if a valid target is absent.
            incl_open_target (bool): Include open target signals when a matching source is not found.
            incl_empty_cols (bool): Whether to include columns that contain no resolved pairs.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            force_signal_index (bool): Force creation of a new signal index even
                if the unraveled mask has the same shape as the original.
            signal_index_type (str): Type of signal index to generate.

                Allowed values:

                * "pair_range": Basic pair counter in a column.
                * "range": Basic signal counter in a column.
                * "source_range": Basic signal counter in a source column.
                * "target_range": Basic signal counter in a target column.
                * "position(s)": Integer position (row) of signal in a column.
                * "source_position(s)": Integer position (row) of signal in a source column.
                * "target_position(s)": Integer position (row) of signal in a target column.
                * "label(s)": Label of signal in a column.
                * "source_label(s)": Label of signal in a source column.
                * "target_label(s)": Label of signal in a target column.
            signal_index_name (str): Name to assign to the signal index.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame or Tuple[SeriesFrame, SeriesFrame]:
                * If one array is provided, returns a wrapped signal mask with an updated index.
                * If two arrays are provided, returns a tuple of wrapped source and target
                    signal masks with updated indices.

        See:
            * `vectorbtpro.signals.nb.unravel_between_nb` for one array.
            * `vectorbtpro.signals.nb.unravel_between_two_nb` for two arrays.

        Examples:
            One mask:

            ```pycon
            >>> mask.vbt.signals.unravel_between()
            signal         -1      0      1      0      1
                            a      b      b      c      c
            2020-01-01  False   True  False   True  False
            2020-01-02  False  False  False   True   True
            2020-01-03  False   True   True  False   True
            2020-01-04  False  False  False  False  False
            2020-01-05  False  False   True  False  False

            >>> mask.vbt.signals.unravel_between(signal_index_type="position")
            source_signal     -1      0      2      0      1
            target_signal     -1      2      4      1      2
                               a      b      b      c      c
            2020-01-01     False   True  False   True  False
            2020-01-02     False  False  False   True   True
            2020-01-03     False   True   True  False   True
            2020-01-04     False  False  False  False  False
            2020-01-05     False  False   True  False  False
            ```

            Two masks:

            ```pycon
            >>> source_mask = pd.Series([True, True, False, False, True, True])
            >>> target_mask = pd.Series([False, False, True, True, False, False])
            >>> new_source_mask, new_target_mask = vbt.pd_acc.signals.unravel_between(
            ...     source_mask,
            ...     target_mask
            ... )
            >>> new_source_mask
            signal      0      1
            0       False  False
            1        True   True
            2       False  False
            3       False  False
            4       False  False
            5       False  False
            >>> new_target_mask
            signal      0      1
            0       False  False
            1       False  False
            2        True  False
            3       False   True
            4       False  False
            5       False  False

            >>> new_source_mask, new_target_mask = vbt.pd_acc.signals.unravel_between(
            ...     source_mask,
            ...     target_mask,
            ...     relation="chain"
            ... )
            >>> new_source_mask
            signal      0      1
            0        True  False
            1       False  False
            2       False  False
            3       False  False
            4       False   True
            5       False  False
            >>> new_target_mask
            signal      0      1
            0       False  False
            1       False  False
            2        True   True
            3       False  False
            4       False  False
            5       False  False
            ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if clean_index_kwargs is None:
            clean_index_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}
        if isinstance(relation, str):
            relation = map_enum_fields(relation, enums.SignalRelation)
        signal_index_type = signal_index_type.lower()
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self.obj, *objs)

        def _build_new_columns(
            source_range,
            target_range,
            source_idxs,
            target_idxs,
            col_idxs,
        ):
            indexes_to_stack = []
            if signal_index_type == "pair_range":
                one_points = np.concatenate((np.array([0]), col_idxs[1:] - col_idxs[:-1]))
                basic_range = np.arange(len(col_idxs))
                range_points = np.where(one_points == 1, basic_range, one_points)
                signal_range = basic_range - np.maximum.accumulate(range_points)
                signal_range[(source_range == -1) & (target_range == -1)] = -1
                indexes_to_stack.append(pd.Index(signal_range, name=signal_index_name))
            else:
                if not signal_index_type.startswith("target_"):
                    indexes_to_stack.append(
                        cls_or_self.index_from_unravel(
                            source_range,
                            source_idxs,
                            wrapper.index,
                            signal_index_type=signal_index_type.replace("source_", ""),
                            signal_index_name="source_" + signal_index_name,
                        )
                    )
                if not signal_index_type.startswith("source_"):
                    indexes_to_stack.append(
                        cls_or_self.index_from_unravel(
                            target_range,
                            target_idxs,
                            wrapper.index,
                            signal_index_type=signal_index_type.replace("target_", ""),
                            signal_index_name="target_" + signal_index_name,
                        )
                    )
            if len(indexes_to_stack) == 1:
                indexes_to_stack[0] = indexes_to_stack[0].rename(signal_index_name)
            return indexes.stack_indexes(
                (*indexes_to_stack, wrapper.columns[col_idxs]), **clean_index_kwargs
            )

        if len(objs) == 1:
            obj = objs[0]
            wrapper = ArrayWrapper.from_obj(obj)
            if not isinstance(obj, (pd.Series, pd.DataFrame)):
                obj = wrapper.wrap(obj)
            func = jit_reg.resolve_option(nb.unravel_between_nb, jitted)
            new_mask, source_range, target_range, source_idxs, target_idxs, col_idxs = func(
                reshaping.to_2d_array(obj),
                incl_open_source=incl_open_source,
                incl_empty_cols=incl_empty_cols,
            )
            if new_mask.shape == wrapper.shape_2d and incl_empty_cols and not force_signal_index:
                return wrapper.wrap(new_mask)
            if not incl_empty_cols and (source_idxs == -1).all():
                raise ValueError("No columns left")
            new_columns = _build_new_columns(
                source_range,
                target_range,
                source_idxs,
                target_idxs,
                col_idxs,
            )
            return wrapper.wrap(new_mask, columns=new_columns, group_by=False, **wrap_kwargs)
        if len(objs) == 2:
            source = objs[0]
            target = objs[1]
            broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
            broadcasted_args, wrapper = reshaping.broadcast(
                dict(source=source, target=target),
                return_wrapper=True,
                **broadcast_kwargs,
            )
            func = jit_reg.resolve_option(nb.unravel_between_two_nb, jitted)
            (
                new_source_mask,
                new_target_mask,
                source_range,
                target_range,
                source_idxs,
                target_idxs,
                col_idxs,
            ) = func(
                broadcasted_args["source"],
                broadcasted_args["target"],
                relation=relation,
                incl_open_source=incl_open_source,
                incl_open_target=incl_open_target,
                incl_empty_cols=incl_empty_cols,
            )
            if (
                new_source_mask.shape == wrapper.shape_2d
                and incl_empty_cols
                and not force_signal_index
            ):
                return wrapper.wrap(new_source_mask), wrapper.wrap(new_target_mask)
            if not incl_empty_cols and (source_idxs == -1).all() and (target_idxs == -1).all():
                raise ValueError("No columns left")
            new_columns = _build_new_columns(
                source_range,
                target_range,
                source_idxs,
                target_idxs,
                col_idxs,
            )
            new_source_mask = wrapper.wrap(
                new_source_mask, columns=new_columns, group_by=False, **wrap_kwargs
            )
            new_target_mask = wrapper.wrap(
                new_target_mask, columns=new_columns, group_by=False, **wrap_kwargs
            )
            return new_source_mask, new_target_mask
        raise ValueError("This method accepts either one or two arrays")

    def ravel(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Ravel signals.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Wrapped array of raveled signals.

        See:
            `vectorbtpro.signals.nb.ravel_nb`

        Examples:
            ```pycon
            >>> unravel_mask = mask.vbt.signals.unravel()
            >>> original_mask = unravel_mask.vbt.signals.ravel(group_by=vbt.ExceptLevel("signal"))
            >>> original_mask
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False  False   True
            2020-01-03  False   True   True
            2020-01-04  False  False  False
            2020-01-05  False   True  False
            ```
        """
        if wrap_kwargs is None:
            wrap_kwargs = {}

        group_map = self.wrapper.grouper.get_group_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.ravel_nb, jitted)
        new_mask = func(self.to_2d_array(), group_map)
        return self.wrapper.wrap(new_mask, group_by=group_by, **wrap_kwargs)

    # ############# Index ############# #

    def nth_index(
        self,
        n: int,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the nth index of signals for each column or group.

        Calls `vectorbtpro.signals.nb.nth_index_nb` to compute the desired index.

        Args:
            n (int): Index offset to select the nth signal.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Series with the nth index for each column or group.

        See:
            * `vectorbtpro.signals.nb.nth_index_nb` regardless of grouping.
            * `vectorbtpro.generic.nb.apply_reduce.any_reduce_nb` if grouping is enabled.

        Examples:
            ```pycon
            >>> mask.vbt.signals.nth_index(0)
            a   2020-01-01
            b   2020-01-01
            c   2020-01-01
            Name: nth_index, dtype: datetime64[ns]

            >>> mask.vbt.signals.nth_index(2)
            a          NaT
            b   2020-01-05
            c   2020-01-03
            Name: nth_index, dtype: datetime64[ns]

            >>> mask.vbt.signals.nth_index(-1)
            a   2020-01-01
            b   2020-01-05
            c   2020-01-03
            Name: nth_index, dtype: datetime64[ns]

            >>> mask.vbt.signals.nth_index(-1, group_by=True)
            Timestamp('2020-01-05 00:00:00')
            ```
        """
        if self.is_frame() and self.wrapper.grouper.is_grouped(group_by=group_by):
            squeezed = self.squeeze_grouped(
                jit_reg.resolve_option(generic_nb.any_reduce_nb, jitted),
                group_by=group_by,
                jitted=jitted,
                chunked=chunked,
            )
            arr = reshaping.to_2d_array(squeezed)
        else:
            arr = self.to_2d_array()
        func = jit_reg.resolve_option(nb.nth_index_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        nth_index = func(arr, n)
        wrap_kwargs = merge_dicts(dict(name_or_index="nth_index", to_index=True), wrap_kwargs)
        return self.wrapper.wrap_reduced(nth_index, group_by=group_by, **wrap_kwargs)

    def norm_avg_index(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the normalized average index of signals for each column or group.

        This function computes the average signal position relative to the middle of the column.
        The result indicates the signal distribution:

        * `-1.0`: Only the first signal is set.
        * `1.0`: Only the last signal is set.
        * `0.0`: Symmetric distribution around the middle.
        * `[-1.0, 0.0)`: Average signal is on the left.
        * `(0.0, 1.0]`: Average signal is on the right.

        Calls `vectorbtpro.signals.nb.norm_avg_index_nb` (or its grouped variant) to perform the computation.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Series with the normalized average index for each column or group.

        See:
            * `vectorbtpro.signals.nb.norm_avg_index_grouped_nb` if grouping is enabled.
            * `vectorbtpro.signals.nb.norm_avg_index_nb` if grouping is disabled.

        Examples:
            ```pycon
            >>> pd.Series([True, False, False, False]).vbt.signals.norm_avg_index()
            -1.0

            >>> pd.Series([False, False, False, True]).vbt.signals.norm_avg_index()
            1.0

            >>> pd.Series([True, False, False, True]).vbt.signals.norm_avg_index()
            0.0
            ```
        """
        if self.is_frame() and self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.norm_avg_index_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            norm_avg_index = func(self.to_2d_array(), group_lens)
        else:
            func = jit_reg.resolve_option(nb.norm_avg_index_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            norm_avg_index = func(self.to_2d_array())
        wrap_kwargs = merge_dicts(dict(name_or_index="norm_avg_index"), wrap_kwargs)
        return self.wrapper.wrap_reduced(norm_avg_index, group_by=group_by, **wrap_kwargs)

    def index_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of indices based on the current signals.

        Considers only True values.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.to_mapped`.

        Returns:
            MappedArray: Mapped array of indices where only True values are considered.
        """
        indices = np.arange(len(self.wrapper.index), dtype=float_)[:, None]
        indices = np.tile(indices, (1, len(self.wrapper.columns)))
        indices = reshaping.soft_to_ndim(indices, self.wrapper.ndim)
        indices[~self.obj.values] = np.nan
        return self.wrapper.wrap(indices).vbt.to_mapped(
            dropna=True, dtype=int_, group_by=group_by, **kwargs
        )

    def total(
        self, wrap_kwargs: tp.KwargsLike = None, group_by: tp.GroupByLike = None
    ) -> tp.MaybeSeries:
        """Return the total number of True signals in each column or group.

        Args:
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            MaybeSeries: Series with the total count of True signals.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="total"), wrap_kwargs)
        return self.sum(group_by=group_by, wrap_kwargs=wrap_kwargs)

    def rate(
        self, wrap_kwargs: tp.KwargsLike = None, group_by: tp.GroupByLike = None, **kwargs
    ) -> tp.MaybeSeries:
        """Return the rate of True signals relative to the total index length for each column or group.

        The rate is computed as the total number of True signals divided by the total index length.

        Args:
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `SignalsAccessor.total`.

        Returns:
            MaybeSeries: Series with the rate of True signals.
        """
        total = reshaping.to_1d_array(self.total(group_by=group_by, **kwargs))
        wrap_kwargs = merge_dicts(dict(name_or_index="rate"), wrap_kwargs)
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        return self.wrapper.wrap_reduced(total / total_steps, group_by=group_by, **wrap_kwargs)

    def total_partitions(
        self,
        wrap_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the total number of partitions of True signals in each column or group.

        Args:
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `SignalsAccessor.partition_ranges`.

        Returns:
            MaybeSeries: Series with the count of True signal partitions.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="total_partitions"), wrap_kwargs)
        return self.partition_ranges(**kwargs).count(group_by=group_by, wrap_kwargs=wrap_kwargs)

    def partition_rate(
        self,
        wrap_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the ratio of total partitions to the total number of True signals in each column or group.

        The ratio is computed as `SignalsAccessor.total_partitions` divided by `SignalsAccessor.total`.

        Args:
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `SignalsAccessor.total_partitions` and `SignalsAccessor.total`.

        Returns:
            MaybeSeries: Series with the partition rate.
        """
        total_partitions = reshaping.to_1d_array(self.total_partitions(group_by=group_by, *kwargs))
        total = reshaping.to_1d_array(self.total(group_by=group_by, *kwargs))
        wrap_kwargs = merge_dicts(dict(name_or_index="partition_rate"), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_partitions / total, group_by=group_by, **wrap_kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `SignalsAccessor.stats`.

        Merges `vectorbtpro.generic.accessors.GenericAccessor.stats_defaults` with the
        `stats` configuration from `vectorbtpro._settings.signals`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        signals_stats_cfg = settings["signals"]["stats"]

        return merge_dicts(GenericAccessor.stats_defaults.__get__(self), signals_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            total=dict(title="Total", calc_func="total", tags="signals"),
            rate=dict(
                title="Rate [%]",
                calc_func="rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="signals",
            ),
            total_overlapping=dict(
                title="Total Overlapping",
                calc_func=lambda self, target, group_by: (self & target).vbt.signals.total(
                    group_by=group_by
                ),
                check_silent_has_target=True,
                tags=["signals", "target"],
            ),
            overlapping_rate=dict(
                title="Overlapping Rate [%]",
                calc_func=lambda self, target, group_by: (self & target).vbt.signals.total(
                    group_by=group_by
                )
                / (self | target).vbt.signals.total(group_by=group_by),
                post_calc_func=lambda self, out, settings: out * 100,
                check_silent_has_target=True,
                tags=["signals", "target"],
            ),
            first_index=dict(
                title="First Index",
                calc_func="nth_index",
                n=0,
                wrap_kwargs=dict(to_index=True),
                tags=["signals", "index"],
            ),
            last_index=dict(
                title="Last Index",
                calc_func="nth_index",
                n=-1,
                wrap_kwargs=dict(to_index=True),
                tags=["signals", "index"],
            ),
            norm_avg_index=dict(
                title="Norm Avg Index [-1, 1]",
                calc_func="norm_avg_index",
                tags=["signals", "index"],
            ),
            distance=dict(
                title=RepEval(
                    "f'Distance {self.get_relation_str(relation)} {target_name}' if target is not None else 'Distance'"
                ),
                calc_func="between_ranges.duration",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.min(),
                    "Median": out.median(),
                    "Max": out.max(),
                },
                apply_to_timedelta=True,
                tags=RepEval(
                    "['signals', 'distance', 'target'] if target is not None else ['signals', 'distance']"
                ),
            ),
            total_partitions=dict(
                title="Total Partitions",
                calc_func="total_partitions",
                tags=["signals", "partitions"],
            ),
            partition_rate=dict(
                title="Partition Rate [%]",
                calc_func="partition_rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=["signals", "partitions"],
            ),
            partition_len=dict(
                title="Partition Length",
                calc_func="partition_ranges.duration",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.min(),
                    "Median": out.median(),
                    "Max": out.max(),
                },
                apply_to_timedelta=True,
                tags=["signals", "partitions", "distance"],
            ),
            partition_distance=dict(
                title="Partition Distance",
                calc_func="between_partition_ranges.duration",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.min(),
                    "Median": out.median(),
                    "Max": out.max(),
                },
                apply_to_timedelta=True,
                tags=["signals", "partitions", "distance"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        yref: str = "y",
        column: tp.Optional[tp.Column] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals.

        Args:
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            column (hashable): Column to plot.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.lineplot`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or trace updater instance produced by the line plot.

        Examples:
            ```pycon
            >>> mask[['a', 'c']].vbt.signals.plot().show()
            ```

            ![](/assets/images/api/signals_df_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/signals_df_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        if column is not None:
            _self = self.select_col(column=column, group_by=False)
        else:
            _self = self
        default_kwargs = dict(trace_kwargs=dict(line=dict(shape="hv")))
        default_kwargs["yaxis" + yref[1:]] = dict(
            tickmode="array", tickvals=[0, 1], ticktext=["false", "true"]
        )
        return _self.obj.vbt.lineplot(**merge_dicts(default_kwargs, kwargs))

    def plot_as_markers(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Column] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot series as markers.

        Args:
            y (ArrayLike): Y-axis values to plot markers on.
            column (hashable): Column to plot.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.scatterplot`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or trace updater instance produced by the scatter plot.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> ts = pd.Series([1, 2, 3, 2, 1], index=mask.index)
            >>> fig = ts.vbt.lineplot()
            >>> mask['b'].vbt.signals.plot_as_entries(y=ts, fig=fig)
            >>> (~mask['b']).vbt.signals.plot_as_exits(y=ts, fig=fig).show()
            ```

            ![](/assets/images/api/signals_plot_as_markers.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/signals_plot_as_markers.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        obj = self.obj
        if isinstance(obj, pd.DataFrame):
            obj = self.select_col_from_obj(obj, column=column, group_by=False)
        if y is None:
            y = pd.Series.vbt.empty_like(obj, 1)
        else:
            y = reshaping.to_pd_array(y)
            if isinstance(y, pd.DataFrame):
                y = self.select_col_from_obj(y, column=column, group_by=False)
            obj, y = reshaping.broadcast(obj, y, columns_from="keep")
            obj = obj.fillna(False).astype(np.bool_)
            if y.name is None:
                y = y.rename("Y")

        def_kwargs = dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol="circle",
                    color=plotting_cfg["contrast_color_schema"]["blue"],
                    size=7,
                ),
                name=obj.name,
            )
        )
        kwargs = merge_dicts(def_kwargs, kwargs)
        if "marker_color" in kwargs["trace_kwargs"]:
            marker_color = kwargs["trace_kwargs"]["marker_color"]
        else:
            marker_color = kwargs["trace_kwargs"]["marker"]["color"]
        if isinstance(marker_color, str) and "rgba" not in marker_color:
            line_color = adjust_lightness(marker_color)
        else:
            line_color = marker_color
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    marker=dict(
                        line=dict(width=1, color=line_color),
                    ),
                ),
            ),
            kwargs,
        )
        return y[obj].vbt.scatterplot(**kwargs)

    def plot_as_entries(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Column] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals as entry markers.

        Args:
            y (Optional[ArrayLike]): Y-axis values for entry markers.
            column (Optional[Column]): Identifier of the column to plot.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.scatterplot`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or trace updater instance representing the entry markers.

        See:
            `SignalsSRAccessor.plot_as_markers`

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        return self.plot_as_markers(
            y=y,
            column=column,
            **merge_dicts(
                dict(
                    trace_kwargs=dict(
                        marker=dict(
                            symbol="triangle-up",
                            color=plotting_cfg["contrast_color_schema"]["green"],
                            size=8,
                        ),
                        name="Entries",
                    )
                ),
                kwargs,
            ),
        )

    def plot_as_exits(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Column] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals as exit markers.

        Args:
            y (Optional[ArrayLike]): Array-like data for plotting exit signals.
            column (Optional[Column]): Identifier of the column to plot.
            **kwargs: Keyword arguments for `SignalsAccessor.plot_as_markers`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or trace updater with plotted exit markers.

        See:
            `SignalsSRAccessor.plot_as_markers`

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        return self.plot_as_markers(
            y=y,
            column=column,
            **merge_dicts(
                dict(
                    trace_kwargs=dict(
                        marker=dict(
                            symbol="triangle-down",
                            color=plotting_cfg["contrast_color_schema"]["red"],
                            size=8,
                        ),
                        name="Exits",
                    )
                ),
                kwargs,
            ),
        )

    def plot_as_entry_marks(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Column] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals as marked entry markers.

        Args:
            y (Optional[ArrayLike]): Array-like data for plotting entry markers.
            column (Optional[Column]): Identifier of the column to plot.
            **kwargs: Keyword arguments for `SignalsAccessor.plot_as_markers`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or trace updater with plotted entry markers.

        See:
            `SignalsSRAccessor.plot_as_markers`

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        return self.plot_as_markers(
            y=y,
            column=column,
            **merge_dicts(
                dict(
                    trace_kwargs=dict(
                        marker=dict(
                            symbol="circle",
                            color="rgba(0, 0, 0, 0)",
                            size=20,
                            line=dict(
                                color=plotting_cfg["contrast_color_schema"]["green"],
                                width=2,
                            ),
                        ),
                        name="Entry marks",
                    )
                ),
                kwargs,
            ),
        )

    def plot_as_exit_marks(
        self,
        y: tp.Optional[tp.ArrayLike] = None,
        column: tp.Optional[tp.Column] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot signals as marked exit markers.

        Args:
            y (Optional[ArrayLike]): Array-like data for plotting exit markers.
            column (Optional[Column]): Identifier of the column to plot.
            **kwargs: Keyword arguments for `SignalsAccessor.plot_as_markers`.

        Returns:
            Union[BaseFigure, tp.TraceUpdater]: Figure or trace updater with plotted exit markers.

        See:
            `SignalsSRAccessor.plot_as_markers`

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        return self.plot_as_markers(
            y=y,
            column=column,
            **merge_dicts(
                dict(
                    trace_kwargs=dict(
                        marker=dict(
                            symbol="circle",
                            color="rgba(0, 0, 0, 0)",
                            size=20,
                            line=dict(
                                color=plotting_cfg["contrast_color_schema"]["red"],
                                width=2,
                            ),
                        ),
                        name="Exit marks",
                    )
                ),
                kwargs,
            ),
        )

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `SignalsAccessor.plots`.

        Merges `vectorbtpro.generic.accessors.GenericAccessor.plots_defaults` with the
        `plots` configuration from `vectorbtpro._settings.signals`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        signals_plots_cfg = settings["signals"]["plots"]

        return merge_dicts(GenericAccessor.plots_defaults.__get__(self), signals_plots_cfg)

    @property
    def subplots(self) -> Config:
        return self._subplots


SignalsAccessor.override_metrics_doc(__pdoc__)
SignalsAccessor.override_subplots_doc(__pdoc__)


@register_sr_vbt_accessor("signals")
class SignalsSRAccessor(SignalsAccessor, GenericSRAccessor):
    """Class representing an accessor for signal series on Pandas Series.

    Provides access to signal-related functionalities for a Pandas Series.

    Accessible via `pd.Series.vbt.signals`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Underlying series data.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericSRAccessor` and `SignalsAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        GenericSRAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            SignalsAccessor.__init__(self, wrapper, obj=obj, **kwargs)


@register_df_vbt_accessor("signals")
class SignalsDFAccessor(SignalsAccessor, GenericDFAccessor):
    """Class representing an accessor for signal series on Pandas DataFrame.

    Provides access to signal-related functionalities for a Pandas DataFrame.

    Accessible via `pd.DataFrame.vbt.signals`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Underlying DataFrame data.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericDFAccessor` and `SignalsAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        GenericDFAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            SignalsAccessor.__init__(self, wrapper, obj=obj, **kwargs)
