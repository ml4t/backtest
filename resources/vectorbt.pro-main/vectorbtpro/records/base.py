# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module for working with records.

Vectorbtpro works with two distinct representations of data: matrices and records.

A matrix, in this context, is simply an array of one-dimensional arrays, each corresponding
to a separate feature. A matrix typically holds data for a single attribute—for example,
entry signals with different strategy configurations as columns. However, when a matrix is huge,
sparse, or when each element must represent multiple pieces of information, creating multiple
matrices becomes inefficient.

Records enable the dense representation of complex, sparse information. They consist of a fixed-schema
array of one-dimensional arrays where each element contains heterogeneous data. Essentially,
records can be thought of as a DataFrame in which each row is a record and each column is a
specific attribute. Learn more about structured arrays [here](https://numpy.org/doc/stable/user/basics.rec.html).

For example, consider representing two DataFrames as a single record array:

```plaintext
               a     b
         0   1.0   5.0
attr1 =  1   2.0   NaN
         2   NaN   7.0
         3   4.0   8.0
               a     b
         0   9.0  13.0
attr2 =  1  10.0   NaN
         2   NaN  15.0
         3  12.0  16.0
            |
            v
      id  col  idx  attr1  attr2
0      0    0    0      1      9
1      1    0    1      2     10
2      2    0    3      4     12
3      0    1    0      5     13
4      1    1    2      7     15
5      2    1    3      8     16
```

Another advantage of records is that they are not constrained by size; multiple records can
correspond to a single element in a matrix. For instance, one can define multiple orders at
the same timestamp, which is difficult to represent in a matrix without duplicating index
entries or employing complex data types.

Consider the following example:

```pycon
>>> from vectorbtpro import *

>>> example_dt = np.dtype([
...     ('id', int_),
...     ('col', int_),
...     ('idx', int_),
...     ('some_field', float_)
... ])
>>> records_arr = np.array([
...     (0, 0, 0, 10.),
...     (1, 0, 1, 11.),
...     (2, 0, 2, 12.),
...     (0, 1, 0, 13.),
...     (1, 1, 1, 14.),
...     (2, 1, 2, 15.),
...     (0, 2, 0, 16.),
...     (1, 2, 1, 17.),
...     (2, 2, 2, 18.)
... ], dtype=example_dt)
>>> wrapper = vbt.ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b', 'c'], ndim=2, freq='1 day')
>>> records = vbt.Records(wrapper, records_arr)
```

## Printing

Records can be printed in two ways.

Raw DataFrame that preserves field names and data types:

```pycon
>>> records.records
   id  col  idx  some_field
0   0    0    0        10.0
1   1    0    1        11.0
2   2    0    2        12.0
3   0    1    0        13.0
4   1    1    1        14.0
5   2    1    2        15.0
6   0    2    0        16.0
7   1    2    1        17.0
8   2    2    2        18.0
```

Readable DataFrame that takes into account `Records.field_config`:

```pycon
>>> records.readable
   Id Column Timestamp  some_field
0   0      a         x        10.0
1   1      a         y        11.0
2   2      a         z        12.0
3   0      b         x        13.0
4   1      b         y        14.0
5   2      b         z        15.0
6   0      c         x        16.0
7   1      c         y        17.0
8   2      c         z        18.0
```

## Mapping

`Records` are structured arrays with numerous methods and properties for data processing.
Their main feature is the ability to map and reduce the records array by column, similar
to the MapReduce paradigm, all without converting to a matrix form and wasting memory.

`Records` can be mapped to a `vectorbtpro.records.mapped_array.MappedArray` in several ways.

Use `Records.map_field` to map a record field:

```pycon
>>> records.map_field('some_field')
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff49bd31a58>

>>> records.map_field('some_field').values
array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
```

Use `Records.map` to map records using a custom function.

```pycon
>>> @njit
... def power_map_nb(record, pow):
...     return record.some_field ** pow

>>> records.map(power_map_nb, 2)
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff49c990cf8>

>>> records.map(power_map_nb, 2).values
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])

>>> # Map using a meta function

>>> @njit
... def power_map_meta_nb(ridx, records, pow):
...     return records[ridx].some_field ** pow

>>> vbt.Records.map(power_map_meta_nb, records.values, 2, col_mapper=records.col_mapper).values
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])
```

Use `Records.map_array` to convert an array to a `vectorbtpro.records.mapped_array.MappedArray`:

```pycon
>>> records.map_array(records_arr['some_field'] ** 2)
<vectorbtpro.records.mapped_array.MappedArray object at 0x7fe9bccf2978>

>>> records.map_array(records_arr['some_field'] ** 2).values
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])
```

Use `Records.apply` to apply a function on each column or group:

```pycon
>>> @njit
... def cumsum_apply_nb(records):
...     return np.cumsum(records.some_field)

>>> records.apply(cumsum_apply_nb)
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff49c990cf8>

>>> records.apply(cumsum_apply_nb).values
array([10., 21., 33., 13., 27., 42., 16., 33., 51.])

>>> group_by = np.array(['first', 'first', 'second'])
>>> records.apply(cumsum_apply_nb, group_by=group_by, apply_per_group=True).values
array([10., 21., 33., 46., 60., 75., 16., 33., 51.])

>>> # Apply using a meta function

>>> @njit
... def cumsum_apply_meta_nb(idxs, col, records):
...     return np.cumsum(records[idxs].some_field)

>>> vbt.Records.apply(cumsum_apply_meta_nb, records.values, col_mapper=records.col_mapper).values
array([10., 21., 33., 13., 27., 42., 16., 33., 51.])
```

Notice that in the first example the cumulative sum resets at each column,
while in the second example it resets for each group.

## Filtering

Use `Records.apply_mask` to filter elements within each column or group:

```pycon
>>> mask = [True, False, True, False, True, False, True, False, True]
>>> filtered_records = records.apply_mask(mask)
>>> filtered_records.records
   id  col  idx  some_field
0   0    0    0        10.0
1   2    0    2        12.0
2   1    1    1        14.0
3   0    2    0        16.0
4   2    2    2        18.0
```

## Grouping

One of the key features of `Records` is the ability to perform reducing operations on a group
of columns as if they were a single column. Groups can be specified by `group_by`, which may be
defined as positions, names of column levels, or a NumPy array representing actual groups.

There are several ways to define grouping.

When creating `Records`, pass `group_by` to `vectorbtpro.base.wrapping.ArrayWrapper`:

```pycon
>>> group_by = np.array(['first', 'first', 'second'])
>>> grouped_wrapper = wrapper.replace(group_by=group_by)
>>> grouped_records = vbt.Records(grouped_wrapper, records_arr)

>>> grouped_records.map_field('some_field').mean()
first     12.5
second    17.0
dtype: float64
```

Regroup an existing `Records`:

```pycon
>>> records.regroup(group_by).map_field('some_field').mean()
first     12.5
second    17.0
dtype: float64
```

Pass `group_by` directly to the mapping method:

```pycon
>>> records.map_field('some_field', group_by=group_by).mean()
first     12.5
second    17.0
dtype: float64
```

Pass `group_by` directly to the reducing method:

```pycon
>>> records.map_field('some_field').mean(group_by=group_by)
a    11.0
b    14.0
c    17.0
dtype: float64
```

!!! note
    Grouping applies only to reducing operations; the underlying arrays remain unchanged.

## Indexing

Similar to other classes subclassing `vectorbtpro.base.wrapping.Wrapping`, `Records` supports Pandas
indexing on a per-column basis. Indexing operations are forwarded to each object representing a column:

```pycon
>>> records['a'].records
   id  col  idx  some_field
0   0    0    0        10.0
1   1    0    1        11.0
2   2    0    2        12.0

>>> grouped_records['first'].records
   id  col  idx  some_field
0   0    0    0        10.0
1   1    0    1        11.0
2   2    0    2        12.0
3   0    1    0        13.0
4   1    1    1        14.0
5   2    1    2        15.0
```

!!! note
    Changing the index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame. For example, use `some_field.iloc[0]` instead of `some_field.iloc[:, 0]`
    to get the first column.

    Indexing behavior depends solely on `vectorbtpro.base.wrapping.ArrayWrapper`. For instance,
    if `group_select` is enabled, indexing will target groups when grouped, otherwise individual columns.

## Caching

`Records` supports caching. Methods or properties that require heavy computation are decorated
with `vectorbtpro.utils.decorators.cached_method` and `vectorbtpro.utils.decorators.cached_property`.
Caching can be disabled globally via `vectorbtpro._settings.caching`.

!!! note
    Because of caching, the class is designed to be immutable and all properties are read-only.
    To modify any attribute, use the `Records.replace` method, passing the changes as keyword arguments.

## Saving and loading

Since `Records` subclasses `vectorbtpro.utils.pickling.Pickleable`, you can save an instance to disk
using `Records.save` and later load it with `Records.load`.

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Records.metrics`.

```pycon
>>> records.stats(column='a')
Start                          x
End                            z
Period           3 days 00:00:00
Total Records                  3
Name: a, dtype: object
```

`Records.stats` also supports (re-)grouping:

```pycon
>>> grouped_records.stats(column='first')
Start                          x
End                            z
Period           3 days 00:00:00
Total Records                  6
Name: first, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Records.subplots`.

This class does not include dedicated subplots by default, but you can add custom subplots in a subclass.

## Extending

The `Records` class can be extended by subclassing.

If certain fields share the same meaning but use different naming conventions (such as the base field `idx`),
you can override `field_config` using `vectorbtpro.records.decorators.override_field_config`.
This decorator searches for configurations in base classes and merges your configuration with them,
preserving any base class properties not explicitly overridden.

```pycon
>>> from vectorbtpro.records.decorators import override_field_config

>>> my_dt = np.dtype([
...     ('my_id', int_),
...     ('my_col', int_),
...     ('my_idx', int_)
... ])

>>> my_fields_config = dict(
...     dtype=my_dt,
...     settings=dict(
...         id=dict(name='my_id'),
...         col=dict(name='my_col'),
...         idx=dict(name='my_idx')
...     )
... )
>>> @override_field_config(my_fields_config)
... class MyRecords(vbt.Records):
...     pass

>>> records_arr = np.array([
...     (0, 0, 0),
...     (1, 0, 1),
...     (0, 1, 0),
...     (1, 1, 1)
... ], dtype=my_dt)
>>> wrapper = vbt.ArrayWrapper(index=['x', 'y'],
...     columns=['a', 'b'], ndim=2, freq='1 day')
>>> my_records = MyRecords(wrapper, records_arr)

>>> my_records.id_arr
array([0, 1, 0, 1])

>>> my_records.col_arr
array([0, 0, 1, 1])

>>> my_records.idx_arr
array([0, 1, 0, 1])
```

Alternatively, you can override the `_field_config` class attribute:

```pycon
>>> @override_field_config
... class MyRecords(vbt.Records):
...     _field_config = dict(
...         dtype=my_dt,
...         settings=dict(
...             id=dict(name='my_id'),
...             idx=dict(name='my_idx'),
...             col=dict(name='my_col')
...         )
...     )
```

!!! note
    Remember to decorate your subclass with `@override_field_config` to inherit configurations
    from base classes. Inheritance can be stopped by not applying the decorator or by passing
    `merge_configs=False` to it.
"""

import inspect
import string
from collections import defaultdict

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.reshaping import index_to_frame, index_to_series, to_1d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.records import nb
from vectorbtpro.records.col_mapper import ColumnMapper
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import cached_method, hybrid_method
from vectorbtpro.utils.random_ import set_seed_nb
from vectorbtpro.utils.template import Sub

__all__ = [
    "Records",
]

__pdoc__ = {}

RecordsT = tp.TypeVar("RecordsT", bound="Records")


class MetaRecords(type(Analyzable)):
    """Metaclass for `Records` that provides field configuration."""

    @property
    def field_config(cls) -> Config:
        """Field configuration.

        Returns:
            Config: Field configuration for the `Records` class.
        """
        return cls._field_config


class Records(Analyzable, metaclass=MetaRecords):
    """Class for wrapping and analyzing record arrays.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        records_arr (array_like): Structured NumPy array of records.

            Must contain the fields `id` (record index) and `col` (column index).
        col_mapper (ColumnMapper): Column mapper if already known.

            !!! note
                This depends on `records_arr`, so ensure that `col_mapper` is invalidated when
                creating a `Records` instance with a modified `records_arr`.

                `Records.replace` handles this automatically.
        **kwargs: Keyword arguments for `vectorbtpro.generic.analyzable.Analyzable`.

    !!! info
        For default settings, see `vectorbtpro._settings.records`.
    """

    _writeable_attrs: tp.WriteableAttrs = {"_field_config"}

    _field_config: tp.ClassVar[Config] = HybridConfig(
        dict(
            dtype=None,
            settings=dict(
                id=dict(name="id", title="Id", mapping="ids"),
                col=dict(name="col", title="Column", mapping="columns", as_customdata=False),
                idx=dict(name="idx", title="Index", mapping="index"),
            ),
        )
    )

    @property
    def field_config(self) -> Config:
        """Field configuration for `${cls_name}`.

        ```python
        ${field_config}
        ```

        Returns:
            Config: Field configuration copied for each instance. Changes to this configuration
                do not affect the class-level configuration.

        To modify the fields, update the config in-place, override this property,
        or set `${cls_name}._field_config` on the instance.
        """
        return self._field_config

    def __init__(
        self,
        wrapper: ArrayWrapper,
        records_arr: tp.RecordArray,
        col_mapper: tp.Optional[ColumnMapper] = None,
        **kwargs,
    ) -> None:
        # Check fields
        records_arr = np.asarray(records_arr)
        checks.assert_not_none(records_arr.dtype.fields)
        field_names = {
            dct.get("name", field_name)
            for field_name, dct in self.field_config.get("settings", {}).items()
        }
        dtype = self.field_config.get("dtype", None)
        if dtype is not None:
            for field in dtype.names:
                if field not in records_arr.dtype.names:
                    if field not in field_names:
                        raise TypeError(
                            f"Field '{field}' from {dtype} cannot be found in records or config"
                        )
        if col_mapper is None:
            col_mapper = ColumnMapper(wrapper, records_arr[self.get_field_name("col")])

        Analyzable.__init__(self, wrapper, records_arr=records_arr, col_mapper=col_mapper, **kwargs)

        self._records_arr = records_arr
        self._col_mapper = col_mapper

        # Only slices of rows can be selected
        self._range_only_select = True

        # Copy writeable attrs
        self._field_config = type(self)._field_config.copy()

    @classmethod
    def row_stack_records_arrs(
        cls, *objs: tp.MaybeSequence[tp.RecordArray], **kwargs
    ) -> tp.RecordArray:
        """Stack multiple record arrays along rows.

        Args:
            *objs (MaybeSequence[RecordArray]): Record arrays to stack.
            **kwargs: Additional keyword arguments.

                Must include a key "wrapper" representing the wrapper instance.

        Returns:
            RecordArray: Row-stacked record array.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        records_arrs = []
        for col in range(kwargs["wrapper"].shape_2d[1]):
            n_rows_sum = 0
            from_id = defaultdict(int)
            for obj in objs:
                col_idxs, col_lens = obj.col_mapper.col_map
                if len(col_idxs) > 0:
                    col_records = None
                    set_columns = False
                    if col > 0 and obj.wrapper.shape_2d[1] == 1:
                        col_records = obj.records_arr[col_idxs]
                        set_columns = True
                    elif col_lens[col] > 0:
                        col_end_idxs = np.cumsum(col_lens)
                        col_start_idxs = col_end_idxs - col_lens
                        col_records = obj.records_arr[
                            col_idxs[col_start_idxs[col] : col_end_idxs[col]]
                        ]
                    if col_records is not None:
                        col_records = col_records.copy()
                        for field in obj.values.dtype.names:
                            field_mapping = (
                                cls.field_config.get("settings", {})
                                .get(field, {})
                                .get("mapping", None)
                            )
                            if (
                                isinstance(field_mapping, str)
                                and field_mapping == "columns"
                                and set_columns
                            ):
                                col_records[field][:] = col
                            elif isinstance(field_mapping, str) and field_mapping == "index":
                                col_records[field][:] += n_rows_sum
                            elif isinstance(field_mapping, str) and field_mapping == "ids":
                                col_records[field][:] += from_id[field]
                                from_id[field] = col_records[field].max() + 1
                        records_arrs.append(col_records)
                n_rows_sum += obj.wrapper.shape_2d[0]

        if len(records_arrs) == 0:
            return np.array([], dtype=objs[0].values.dtype)
        return np.concatenate(records_arrs)

    @classmethod
    def get_row_stack_record_indices(
        cls, *objs: tp.MaybeSequence[tp.RecordArray], **kwargs
    ) -> tp.Array1d:
        """Get the indices mapping concatenated record arrays to the row-stacked record array.

        Args:
            *objs (MaybeSequence[RecordArray]): Record arrays to be stacked.
            **kwargs: Additional keyword arguments.

                Must include a key "wrapper" representing the wrapper instance.

        Returns:
            Array1d: Concatenated indices mapping the original record arrays to the row-stacked array.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        record_indices = []
        cum_n_rows_sum = []
        for i in range(len(objs)):
            if i == 0:
                cum_n_rows_sum.append(0)
            else:
                cum_n_rows_sum.append(cum_n_rows_sum[-1] + len(objs[i - 1].values))
        for col in range(kwargs["wrapper"].shape_2d[1]):
            for i, obj in enumerate(objs):
                col_idxs, col_lens = obj.col_mapper.col_map
                if len(col_idxs) > 0:
                    if col > 0 and obj.wrapper.shape_2d[1] == 1:
                        _record_indices = col_idxs + cum_n_rows_sum[i]
                        record_indices.append(_record_indices)
                    elif col_lens[col] > 0:
                        col_end_idxs = np.cumsum(col_lens)
                        col_start_idxs = col_end_idxs - col_lens
                        _record_indices = (
                            col_idxs[col_start_idxs[col] : col_end_idxs[col]] + cum_n_rows_sum[i]
                        )
                        record_indices.append(_record_indices)

        if len(record_indices) == 0:
            return np.array([], dtype=int_)
        return np.concatenate(record_indices)

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[RecordsT],
        *objs: tp.MaybeSequence[RecordsT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> RecordsT:
        """Stack multiple `Records` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers and
        `Records.row_stack_records_arrs` to stack the record arrays.

        Args:
            *objs (MaybeSequence[Records]): (Additional) `Records` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `Records` through
                `Records.resolve_row_stack_kwargs` and `Records.resolve_stack_kwargs`.

        Returns:
            Records: New `Records` instance representing the row-stacked result.

        !!! note
            Will produce a column-sorted array.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Records):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(
                *[obj.wrapper for obj in objs], **wrapper_kwargs
            )

        if "col_mapper" not in kwargs:
            kwargs["col_mapper"] = ColumnMapper.row_stack(
                *[obj.col_mapper for obj in objs],
                wrapper=kwargs["wrapper"],
            )
        if "records_arr" not in kwargs:
            kwargs["records_arr"] = cls.row_stack_records_arrs(*objs, **kwargs)

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack_records_arrs(
        cls,
        *objs: tp.MaybeSequence[tp.RecordArray],
        get_indexer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.RecordArray:
        """Stack multiple record arrays along columns.

        Args:
            *objs (MaybeSequence[RecordArray]): Record arrays to stack.
            get_indexer_kwargs (KwargsLike): Keyword arguments for `pd.Index.get_indexer`.
            **kwargs: Additional keyword arguments.

                Must include a key "wrapper" representing the wrapper instance.

        Returns:
            RecordArray: Column-stacked record array.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        if get_indexer_kwargs is None:
            get_indexer_kwargs = {}

        records_arrs = []
        col_sum = 0
        for obj in objs:
            col_idxs, col_lens = obj.col_mapper.col_map
            if len(col_idxs) > 0:
                col_end_idxs = np.cumsum(col_lens)
                col_start_idxs = col_end_idxs - col_lens
                for col in range(len(col_lens)):
                    if col_lens[col] > 0:
                        col_records = obj.records_arr[
                            col_idxs[col_start_idxs[col] : col_end_idxs[col]]
                        ]
                        col_records = col_records.copy()
                        for field in obj.values.dtype.names:
                            field_mapping = (
                                cls.field_config.get("settings", {})
                                .get(field, {})
                                .get("mapping", None)
                            )
                            if isinstance(field_mapping, str) and field_mapping == "columns":
                                col_records[field][:] += col_sum
                            elif isinstance(field_mapping, str) and field_mapping == "index":
                                old_idxs = col_records[field]
                                if not obj.wrapper.index.equals(kwargs["wrapper"].index):
                                    new_idxs = kwargs["wrapper"].index.get_indexer(
                                        obj.wrapper.index[old_idxs],
                                        **get_indexer_kwargs,
                                    )
                                else:
                                    new_idxs = old_idxs
                                col_records[field][:] = new_idxs
                        records_arrs.append(col_records)
            col_sum += obj.wrapper.shape_2d[1]

        if len(records_arrs) == 0:
            return np.array([], dtype=objs[0].values.dtype)
        return np.concatenate(records_arrs)

    @classmethod
    def get_column_stack_record_indices(
        cls, *objs: tp.MaybeSequence[tp.RecordArray], **kwargs
    ) -> tp.Array1d:
        """Get the indices that map concatenated record arrays into the column-stacked record array.

        Args:
            *objs (MaybeSequence[RecordArray]): Record arrays to be stacked.
            **kwargs: Additional keyword arguments.

        Returns:
            Array1d: Numpy array of indices mapping the concatenated record arrays into the column-stacked array.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        record_indices = []
        cum_n_rows_sum = []
        for i in range(len(objs)):
            if i == 0:
                cum_n_rows_sum.append(0)
            else:
                cum_n_rows_sum.append(cum_n_rows_sum[-1] + len(objs[i - 1].values))
        for i, obj in enumerate(objs):
            col_idxs, col_lens = obj.col_mapper.col_map
            if len(col_idxs) > 0:
                col_end_idxs = np.cumsum(col_lens)
                col_start_idxs = col_end_idxs - col_lens
                for col in range(len(col_lens)):
                    if col_lens[col] > 0:
                        _record_indices = (
                            col_idxs[col_start_idxs[col] : col_end_idxs[col]] + cum_n_rows_sum[i]
                        )
                        record_indices.append(_record_indices)

        if len(record_indices) == 0:
            return np.array([], dtype=int_)
        return np.concatenate(record_indices)

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[RecordsT],
        *objs: tp.MaybeSequence[RecordsT],
        wrapper_kwargs: tp.KwargsLike = None,
        get_indexer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> RecordsT:
        """Stack multiple `Records` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers and
        `Records.column_stack_records_arrs` to combine the record arrays. The `get_indexer_kwargs`
        are passed to `pd.Index.get_indexer` to map old indices to new ones after reindexing.

        Args:
            *objs (MaybeSequence[Records]): (Additional) `Records` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            get_indexer_kwargs (KwargsLike): Keyword arguments for `pd.Index.get_indexer`.
            **kwargs: Keyword arguments for `Records` through
                `Records.resolve_column_stack_kwargs` and `Records.resolve_stack_kwargs`.

        Returns:
            Records: New `Records` instance with column-stacked data.

        !!! note
            Produces a column-sorted array.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Records):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        if "col_mapper" not in kwargs:
            kwargs["col_mapper"] = ColumnMapper.column_stack(
                *[obj.col_mapper for obj in objs],
                wrapper=kwargs["wrapper"],
            )
        if "records_arr" not in kwargs:
            kwargs["records_arr"] = cls.column_stack_records_arrs(
                *objs,
                get_indexer_kwargs=get_indexer_kwargs,
                **kwargs,
            )

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    def replace(self: RecordsT, **kwargs) -> RecordsT:
        if self.config.get("col_mapper", None) is not None:
            if "wrapper" in kwargs:
                if self.wrapper is not kwargs.get("wrapper"):
                    kwargs["col_mapper"] = None
            if "records_arr" in kwargs:
                if self.records_arr is not kwargs.get("records_arr"):
                    kwargs["col_mapper"] = None
        return Analyzable.replace(self, **kwargs)

    def select_cols(
        self,
        col_idxs: tp.MaybeIndexArray,
        jitted: tp.JittedOption = None,
    ) -> tp.Tuple[tp.Array1d, tp.RecordArray]:
        """Select columns from the record array.

        Automatically determines whether to use column lengths or the column map for selecting the columns.

        Args:
            col_idxs (MaybeIndexArray): Column indices or slice to select.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            Tuple[Array1d, RecordArray]: Tuple containing the selected indices and the new record array.

        See:
            * `vectorbtpro.records.nb.record_col_lens_select_nb` if `Records.col_mapper` is sorted.
            * `vectorbtpro.records.nb.record_col_map_select_nb` if `Records.col_mapper` is not sorted.
        """
        if len(self.values) == 0:
            return np.arange(len(self.values)), self.values
        if isinstance(col_idxs, slice):
            if col_idxs.start is None and col_idxs.stop is None:
                return np.arange(len(self.values)), self.values
            col_idxs = np.arange(col_idxs.start, col_idxs.stop)
        if self.col_mapper.is_sorted():
            func = jit_reg.resolve_option(nb.record_col_lens_select_nb, jitted)
            new_indices, new_records_arr = func(
                self.values, self.col_mapper.col_lens, to_1d_array(col_idxs)
            )  # faster
        else:
            func = jit_reg.resolve_option(nb.record_col_map_select_nb, jitted)
            new_indices, new_records_arr = func(
                self.values, self.col_mapper.col_map, to_1d_array(col_idxs)
            )  # more flexible
        return new_indices, new_records_arr

    def indexing_func_meta(self, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on the `Records` instance and return corresponding metadata.

        By default, all fields mapped to an index are processed.
        Set a field's `noindex` setting to True to exclude it from indexing.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.
            wrapper_meta (DictLike): Metadata from the indexing operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.

        Returns:
            dict: Dictionary containing:

                * `wrapper_meta`: Updated metadata from the wrapper.
                * `new_indices`: Array of new indices after indexing.
                * `new_records_arr`: Updated record array after applying the index filter.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(
                *args,
                column_only_select=self.column_only_select,
                range_only_select=self.range_only_select,
                group_select=self.group_select,
                **kwargs,
            )
        if self.get_field_setting("col", "group_indexing", False):
            new_indices, new_records_arr = self.select_cols(wrapper_meta["group_idxs"])
        else:
            new_indices, new_records_arr = self.select_cols(wrapper_meta["col_idxs"])
        if wrapper_meta["rows_changed"]:
            row_idxs = wrapper_meta["row_idxs"]
            index_fields = []
            all_index_fields = []
            for field in new_records_arr.dtype.names:
                field_mapping = self.get_field_mapping(field)
                noindex = self.get_field_setting(field, "noindex", False)
                if isinstance(field_mapping, str) and field_mapping == "index":
                    all_index_fields.append(field)
                    if not noindex:
                        index_fields.append(field)
            if len(index_fields) > 0:
                masks = []
                for field in index_fields:
                    field_arr = new_records_arr[field]
                    masks.append((field_arr >= row_idxs.start) & (field_arr < row_idxs.stop))
                mask = np.array(masks).all(axis=0)
                new_indices = new_indices[mask]
                new_records_arr = new_records_arr[mask]
                for field in all_index_fields:
                    new_records_arr[field] = new_records_arr[field] - row_idxs.start
        return dict(
            wrapper_meta=wrapper_meta,
            new_indices=new_indices,
            new_records_arr=new_records_arr,
        )

    def indexing_func(
        self: RecordsT, *args, records_meta: tp.DictLike = None, **kwargs
    ) -> RecordsT:
        """Perform indexing on the `Records` instance.

        Args:
            *args: Positional arguments for `Records.indexing_func_meta`.
            records_meta (DictLike): Metadata from the indexing operation on the records.
            **kwargs: Keyword arguments for `Records.indexing_func_meta`.

        Returns:
            Records: New `Records` instance with updated indexing.
        """
        if records_meta is None:
            records_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=records_meta["new_records_arr"],
        )

    def resample_records_arr(
        self, resampler: tp.Union[Resampler, tp.PandasResampler]
    ) -> tp.RecordArray:
        """Resample the record array.

        Args:
            resampler (Union[Resampler, PandasResampler]): `vectorbtpro.base.resampling.base.Resampler` instance
                or a Pandas resampler.

        Returns:
            RecordArray: Resampled record array.
        """
        if isinstance(resampler, Resampler):
            _resampler = resampler
        else:
            _resampler = Resampler.from_pd_resampler(resampler)
        new_records_arr = self.records_arr.copy()
        for field_name in self.values.dtype.names:
            field_mapping = self.get_field_mapping(field_name)
            if isinstance(field_mapping, str) and field_mapping == "index":
                index_map = _resampler.map_to_target_index(return_index=False)
                new_records_arr[field_name] = index_map[new_records_arr[field_name]]
        return new_records_arr

    def resample_meta(self: RecordsT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Resample records and return metadata.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.
            wrapper_meta (DictLike): Metadata from the resampling operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.

        Returns:
            dict: Dictionary containing:

                * `wrapper_meta`: Metadata from the resampling operation on the wrapper.
                * `new_records_arr`: New resampled record array.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        new_records_arr = self.resample_records_arr(wrapper_meta["resampler"])
        return dict(wrapper_meta=wrapper_meta, new_records_arr=new_records_arr)

    def resample(self: RecordsT, *args, records_meta: tp.DictLike = None, **kwargs) -> RecordsT:
        """Resample records.

        Args:
            *args: Positional arguments for `Records.resample_meta`.
            records_meta (DictLike): Metadata from the resampling operation on the records.
            **kwargs: Keyword arguments for `Records.resample_meta`.

        Returns:
            Records: New instance of records with resampled data.
        """
        if records_meta is None:
            records_meta = self.resample_meta(*args, **kwargs)
        return self.replace(
            wrapper=records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=records_meta["new_records_arr"],
        )

    @property
    def records_arr(self) -> tp.RecordArray:
        """Record array.

        The internal array storing the records.

        Returns:
            RecordArray: Record array containing the records.
        """
        return self._records_arr

    @property
    def col_mapper(self) -> ColumnMapper:
        """Column mapper.

        Provides a mapping of record columns.

        Returns:
            ColumnMapper: Instance of `vectorbtpro.records.col_mapper.ColumnMapper`.
        """
        return self._col_mapper

    @property
    def values(self) -> tp.RecordArray:
        """Record array.

        Alias for `Records.records_arr`.

        Returns:
            RecordArray: Record array containing the records.
        """
        return self.records_arr

    def __len__(self) -> int:
        return len(self.values)

    @property
    def records(self) -> tp.Frame:
        """Records as DataFrame.

        Converts the record array into a Pandas DataFrame.

        Returns:
            Frame: DataFrame representation of the records.
        """
        return pd.DataFrame.from_records(self.values)

    @property
    def recarray(self) -> tp.RecArray:
        """Record array with attribute access.

        Returns a NumPy recarray that supports accessing fields as attributes.

        Returns:
            RecArray: NumPy recarray representation of the records.
        """
        return self.values.view(np.recarray)

    @property
    def field_names(self) -> tp.List[str]:
        """Field names.

        A list of field names extracted from the record array.

        Returns:
            List[str]: List of field names.
        """
        return list(self.values.dtype.fields.keys())

    def to_readable(self, expand_columns: bool = False) -> tp.Frame:
        """Convert records to a human-readable format.

        Args:
            expand_columns (bool): Expand MultiIndex columns into separate columns if present.

        Returns:
            Frame: DataFrame representing the records in a human-readable format.
        """
        new_columns = list()
        field_settings = self.field_config.get("settings", {})
        for field_name in self.field_names:
            if field_name in field_settings:
                dct = field_settings[field_name]
                if dct.get("ignore", False):
                    continue
                field_name = dct.get("name", field_name)
                if "title" in dct:
                    title = dct["title"]
                else:
                    title = field_name
                if "mapping" in dct:
                    if isinstance(dct["mapping"], str) and dct["mapping"] == "index":
                        new_columns.append(
                            pd.Series(self.get_map_field_to_index(field_name), name=title)
                        )
                    elif isinstance(dct["mapping"], str) and dct["mapping"] == "columns":
                        column_index = self.get_map_field_to_columns(field_name)
                        if expand_columns and isinstance(column_index, pd.MultiIndex):
                            column_frame = index_to_frame(column_index, reset_index=True)
                            new_columns.append(column_frame.add_prefix(f"{title}: "))
                        else:
                            column_sr = index_to_series(column_index, reset_index=True)
                            if (
                                expand_columns
                                and self.wrapper.ndim == 2
                                and column_sr.name is not None
                            ):
                                new_columns.append(column_sr.rename(f"{title}: {column_sr.name}"))
                            else:
                                new_columns.append(column_sr.rename(title))
                    else:
                        new_columns.append(
                            pd.Series(self.get_apply_mapping_arr(field_name), name=title)
                        )
                else:
                    new_columns.append(pd.Series(self.values[field_name], name=title))
            else:
                new_columns.append(pd.Series(self.values[field_name], name=field_name))
        records_readable = pd.concat(new_columns, axis=1)
        if all([isinstance(col, tuple) for col in records_readable.columns]):
            records_readable.columns = pd.MultiIndex.from_tuples(records_readable.columns)
        return records_readable

    @property
    def records_readable(self) -> tp.Frame:
        """Readable records.

        Returns the result of `Records.to_readable` with default arguments.

        Returns:
            Frame: DataFrame representing the records in a human-readable format.
        """
        return self.to_readable()

    readable = records_readable

    def get_field_setting(self, field: str, setting: str, default: tp.Any = None) -> tp.Any:
        """Retrieve a field's setting.

        Args:
            field (str): Field identifier.
            setting (str): Key of the setting.
            default (Any): Default value to return if the setting is not present.

        Returns:
            Any: Value of the specified setting or the default value.
        """
        return self.field_config.get("settings", {}).get(field, {}).get(setting, default)

    def get_field_name(self, field: str) -> str:
        """Retrieve the display name of a field.

        Args:
            field (str): Field identifier.

        Returns:
            str: Display name for the field as specified in the field configuration
                or the original field name.
        """
        return self.get_field_setting(field, "name", field)

    def get_field_title(self, field: str) -> str:
        """Retrieve the title of a field.

        Args:
            field (str): Field identifier.

        Returns:
            str: Title of the field based on the field configuration,
                or the field name if not specified.
        """
        return self.get_field_setting(field, "title", field)

    def get_field_mapping(self, field: str) -> tp.Optional[tp.MappingLike]:
        """Retrieve the mapping for a field.

        Args:
            field (str): Field identifier.

        Returns:
            Optional[MappingLike]: Mapping for the field as defined in the field configuration,
                or None if not set.
        """
        return self.get_field_setting(field, "mapping", None)

    def get_field_arr(self, field: str, copy: bool = False) -> tp.Array1d:
        """Retrieve the array for a given field.

        Args:
            field (str): Field identifier.
            copy (bool): Whether to return a copy of the array.

        Returns:
            Array1d: Array corresponding to the specified field.
        """
        out = self.values[self.get_field_name(field)]
        if copy:
            out = out.copy()
        return out

    def get_map_field(self, field: str, **kwargs) -> MappedArray:
        """Retrieve the mapped array for a field.

        Args:
            field (str): Field identifier.
            **kwargs: Keyword arguments for `Records.map_field`.

        Returns:
            MappedArray: Mapped array for the specified field.
        """
        mapping = self.get_field_mapping(field)
        if isinstance(mapping, str) and mapping == "ids":
            mapping = None
        return self.map_field(self.get_field_name(field), mapping=mapping, **kwargs)

    def get_map_field_to_index(
        self, field: str, minus_one_to_zero: bool = False, **kwargs
    ) -> tp.Index:
        """Retrieve the mapped index for a field.

        Args:
            field (str): Field identifier.
            minus_one_to_zero (bool): If True, convert index -1 to 0; if False,
                raise an error when -1 is present.
            **kwargs: Keyword arguments for `Records.get_map_field`.

        Returns:
            Index: Index derived from the mapped field.
        """
        return self.get_map_field(field, **kwargs).to_index(minus_one_to_zero=minus_one_to_zero)

    def get_map_field_to_columns(self, field: str, **kwargs) -> tp.Index:
        """Retrieve the mapped columns for a field.

        Args:
            field (str): Field identifier.
            **kwargs: Keyword arguments for `Records.get_map_field`.

        Returns:
            Index: Columns derived from the mapped field.
        """
        return self.get_map_field(field, **kwargs).to_columns()

    def get_apply_mapping_arr(
        self, field: str, mapping_kwargs: tp.KwargsLike = None, **kwargs
    ) -> tp.Array1d:
        """Retrieve the array for a field with applied mapping.

        Args:
            field (str): Field identifier.
            mapping_kwargs (KwargsLike): Keyword arguments for applying the mapping.

                See `vectorbtpro.records.mapped_array.MappedArray.apply_mapping`.
            **kwargs: Keyword arguments for `Records.get_map_field`.

        Returns:
            Array1d: Array with the applied mapping for the field.
        """
        mapping = self.get_field_mapping(field)
        if isinstance(mapping, str) and mapping == "index":
            return self.get_map_field_to_index(field, **kwargs).values
        if isinstance(mapping, str) and mapping == "columns":
            return self.get_map_field_to_columns(field, **kwargs).values
        return (
            self.get_map_field(field, **kwargs).apply_mapping(mapping_kwargs=mapping_kwargs).values
        )

    def get_apply_mapping_str_arr(
        self, field: str, mapping_kwargs: tp.KwargsLike = None, **kwargs
    ) -> tp.Array1d:
        """Retrieve a stringified array for a field with applied mapping.

        Args:
            field (str): Field identifier.
            mapping_kwargs (KwargsLike): Keyword arguments for applying the mapping.

                See `vectorbtpro.records.mapped_array.MappedArray.apply_mapping`.
            **kwargs: Keyword arguments for `Records.get_map_field`.

        Returns:
            Array1d: Stringified array with the applied mapping for the field.
        """
        mapping = self.get_field_mapping(field)
        if isinstance(mapping, str) and mapping == "index":
            return self.get_map_field_to_index(field, **kwargs).astype(str).values
        if isinstance(mapping, str) and mapping == "columns":
            return self.get_map_field_to_columns(field, **kwargs).astype(str).values
        return (
            self.get_map_field(field, **kwargs)
            .apply_mapping(mapping_kwargs=mapping_kwargs)
            .values.astype(str)
        )

    @property
    def id_arr(self) -> tp.Array1d:
        """ID array.

        Array of IDs extracted from the record array.

        Returns:
            Array1d: Array of IDs.
        """
        return self.values[self.get_field_name("id")]

    @property
    def col_arr(self) -> tp.Array1d:
        """Column array.

        Array of column identifiers extracted from the record array.

        Returns:
            Array1d: Array of column identifiers.
        """
        return self.values[self.get_field_name("col")]

    @property
    def idx_arr(self) -> tp.Optional[tp.Array1d]:
        """Index array.

        Array of index values extracted from the record array, or None if not available.

        Returns:
            Optional[Array1d]: Array of index values if available; otherwise, None.
        """
        idx_field_name = self.get_field_name("idx")
        if idx_field_name is None:
            return None
        return self.values[idx_field_name]

    # ############# Sorting ############# #

    @cached_method
    def is_sorted(self, incl_id: bool = False, jitted: tp.JittedOption = None) -> bool:
        """Check whether the records are sorted.

        Args:
            incl_id (bool): If True, include record ids in the sorting criteria.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            bool: True if the records are sorted, False otherwise.

        See:
            * `vectorbtpro.records.nb.is_col_id_sorted_nb` for sorting with IDs.
            * `vectorbtpro.records.nb.is_col_sorted_nb` for sorting without IDs.
        """
        if incl_id:
            func = jit_reg.resolve_option(nb.is_col_id_sorted_nb, jitted)
            return func(self.col_arr, self.id_arr)
        func = jit_reg.resolve_option(nb.is_col_sorted_nb, jitted)
        return func(self.col_arr)

    def sort(
        self: RecordsT, incl_id: bool = False, group_by: tp.GroupByLike = None, **kwargs
    ) -> RecordsT:
        """Sort records by column values with an optional secondary sort by record ids.

        Args:
            incl_id (bool): If True, include record ids in the sorting criteria.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `Records.replace`.

        Returns:
            Records: New instance with sorted records.

        !!! note
            Sorting is expensive. It is more efficient to append records that are already in the correct order.
        """
        if self.is_sorted(incl_id=incl_id):
            return self.replace(**kwargs).regroup(group_by)
        if incl_id:
            ind = np.lexsort((self.id_arr, self.col_arr))  # expensive!
        else:
            ind = np.argsort(self.col_arr)
        return self.replace(records_arr=self.values[ind], **kwargs).regroup(group_by)

    # ############# Filtering ############# #

    def apply_mask(
        self: RecordsT, mask: tp.Array1d, group_by: tp.GroupByLike = None, **kwargs
    ) -> RecordsT:
        """Return a new records instance filtered by a boolean mask.

        Args:
            mask (Array1d): Boolean array indicating which records to retain.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `Records.replace`.

        Returns:
            Records: New instance containing the filtered records.
        """
        mask_indices = np.flatnonzero(mask)
        return self.replace(records_arr=np.take(self.values, mask_indices), **kwargs).regroup(
            group_by
        )

    def first_n(
        self: RecordsT,
        n: int,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> RecordsT:
        """Return a new records instance with the first N records retained in each column.

        Args:
            n (int): Number of records to retain from the beginning of each column.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Records.apply_mask`.

        Returns:
            Records: New instance containing the first N records per column.

        See:
            `vectorbtpro.records.nb.first_n_nb`
        """
        col_map = self.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.first_n_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return self.apply_mask(func(col_map, n), **kwargs)

    def last_n(
        self: RecordsT,
        n: int,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> RecordsT:
        """Return a new records instance with the last N records retained in each column.

        Args:
            n (int): Number of records to retain from the end of each column.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Records.apply_mask`.

        Returns:
            Records: New instance containing the last N records per column.

        See:
            `vectorbtpro.records.nb.last_n_nb`
        """
        col_map = self.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.last_n_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return self.apply_mask(func(col_map, n), **kwargs)

    def random_n(
        self: RecordsT,
        n: int,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> RecordsT:
        """Return a new records instance with N randomly selected records from each column.

        Args:
            n (int): Number of random records to select per column.
            seed (Optional[int]): Random seed for deterministic output.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Records.apply_mask`.

        Returns:
            Records: New instance containing the randomly selected records.

        See:
            `vectorbtpro.records.nb.random_n_nb`
        """
        if seed is not None:
            set_seed_nb(seed)
        col_map = self.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.random_n_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return self.apply_mask(func(col_map, n), **kwargs)

    # ############# Mapping ############# #

    def map_array(
        self,
        a: tp.ArrayLike,
        idx_arr: tp.Union[None, str, tp.Array1d] = None,
        mapping: tp.Optional[tp.MappingLike] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> MappedArray:
        """Convert an array to a mapped array.

        The input array's length must match the number of records.

        Args:
            a (ArrayLike): Array to be converted into a mapped array.
            idx_arr (Union[None, str, Array1d]): Array of row indices or field name for retrieving row indices.

                If None, `Records.idx_arr` is used.
            mapping (Optional[MappingLike]): Mapping configuration.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray`.

        Returns:
            MappedArray: Resulting mapped array.
        """
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        checks.assert_shape_equal(a, self.values)
        if idx_arr is None:
            idx_arr = self.idx_arr
        elif isinstance(idx_arr, str):
            idx_arr = self.get_field_arr(idx_arr)
        return MappedArray(
            self.wrapper,
            a,
            self.col_arr,
            id_arr=self.id_arr,
            idx_arr=idx_arr,
            mapping=mapping,
            col_mapper=self.col_mapper,
            **kwargs,
        ).regroup(group_by)

    def map_field(self, field: str, **kwargs) -> MappedArray:
        """Convert a field to a mapped array.

        Args:
            field (str): Field identifier.
            **kwargs: Keyword arguments for `Records.map_array`.

        Returns:
            MappedArray: Resulting mapped array for the specified field.
        """
        mapped_arr = self.values[field]
        return self.map_array(mapped_arr, **kwargs)

    @hybrid_method
    def map(
        cls_or_self,
        map_func_nb: tp.AnyRecordsMapFunc,
        *args,
        dtype: tp.Optional[tp.DTypeLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        col_mapper: tp.Optional[ColumnMapper] = None,
        **kwargs,
    ) -> MappedArray:
        """Map each record to a scalar value and return a mapped array.

        For class method calls, `col_mapper` must be provided.

        Args:
            map_func_nb (AnyRecordsMapFunc): Callback function for mapping records.
            *args: Positional arguments for `map_func_nb`.
            dtype (Optional[DTypeLike]): Data type for the resulting mapped array.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            col_mapper (Optional[ColumnMapper]): Column mapper for the meta version.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray`
                or `Records.map_array`.

        Returns:
            MappedArray: Mapped array with scalar values for each record.

        See:
            * `vectorbtpro.records.nb.map_records_nb` for regular application.
            * `vectorbtpro.records.nb.map_records_meta_nb` for meta application.
        """
        if isinstance(cls_or_self, type):
            checks.assert_not_none(col_mapper, arg_name="col_mapper")
            func = jit_reg.resolve_option(nb.map_records_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            mapped_arr = func(len(col_mapper.col_arr), map_func_nb, *args)
            mapped_arr = np.asarray(mapped_arr, dtype=dtype)
            return MappedArray(
                col_mapper.wrapper, mapped_arr, col_mapper.col_arr, col_mapper=col_mapper, **kwargs
            )
        else:
            func = jit_reg.resolve_option(nb.map_records_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            mapped_arr = func(cls_or_self.values, map_func_nb, *args)
            mapped_arr = np.asarray(mapped_arr, dtype=dtype)
            return cls_or_self.map_array(mapped_arr, **kwargs)

    @hybrid_method
    def apply(
        cls_or_self,
        apply_func_nb: tp.AnyApplyFunc,
        *args,
        group_by: tp.GroupByLike = None,
        apply_per_group: bool = False,
        dtype: tp.Optional[tp.DTypeLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        col_mapper: tp.Optional[ColumnMapper] = None,
        **kwargs,
    ) -> MappedArray:
        """Apply a function to records on a per-column or per-group basis and return a mapped array.

        If `apply_per_group` is True, the function is applied separately to each group.

        Args:
            apply_func_nb (AnyApplyFunc): Callback function for applying to records.
            *args: Positional arguments for `apply_func_nb`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            apply_per_group (bool): If True, apply the function per group of columns.
            dtype (Optional[DTypeLike]): Data type for the resulting mapped array.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            col_mapper (Optional[ColumnMapper]): Column mapper for the meta version.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray`
                or `Records.map_array`.

        Returns:
            MappedArray: Mapped array resulting from applying the function to the records.

        See:
            * `vectorbtpro.records.nb.apply_nb` for regular application.
            * `vectorbtpro.records.nb.apply_meta_nb` for meta application.
        """
        if isinstance(cls_or_self, type):
            checks.assert_not_none(col_mapper, arg_name="col_mapper")
            col_map = col_mapper.get_col_map(group_by=group_by if apply_per_group else False)
            func = jit_reg.resolve_option(nb.apply_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            mapped_arr = func(len(col_mapper.col_arr), col_map, apply_func_nb, *args)
            mapped_arr = np.asarray(mapped_arr, dtype=dtype)
            return MappedArray(
                col_mapper.wrapper, mapped_arr, col_mapper.col_arr, col_mapper=col_mapper, **kwargs
            )
        else:
            col_map = cls_or_self.col_mapper.get_col_map(
                group_by=group_by if apply_per_group else False
            )
            func = jit_reg.resolve_option(nb.apply_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            mapped_arr = func(cls_or_self.values, col_map, apply_func_nb, *args)
            mapped_arr = np.asarray(mapped_arr, dtype=dtype)
            return cls_or_self.map_array(mapped_arr, group_by=group_by, **kwargs)

    # ############# Masking ############# #

    def get_pd_mask(
        self,
        idx_arr: tp.Union[None, str, tp.Array1d] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return a mask as a Series or DataFrame based on row and column indices.

        Args:
            idx_arr (Union[None, str, Array1d]): Array of row indices or field name for retrieving row indices.

                If None, `Records.idx_arr` is used.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Series or DataFrame representing the mask.
        """
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        elif isinstance(idx_arr, str):
            idx_arr = self.get_field_arr(idx_arr)
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        out_arr = np.full(target_shape, False)
        out_arr[idx_arr, col_arr] = True
        return self.wrapper.wrap(out_arr, group_by=group_by, **resolve_dict(wrap_kwargs))

    @property
    def pd_mask(self) -> tp.SeriesFrame:
        """Mask as a SeriesFrame produced by invoking
        `vectorbtpro.records.mapped_array.MappedArray.get_pd_mask` with default arguments.

        Returns:
            SeriesFrame: Series or DataFrame representing the mask.
        """
        return self.get_pd_mask()

    # ############# Reducing ############# #

    @cached_method
    def count(
        self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None
    ) -> tp.MaybeSeries:
        """Return count of records per column.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Series containing the count by column.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="count"), wrap_kwargs)
        return self.wrapper.wrap_reduced(
            self.col_mapper.get_col_map(group_by=group_by)[1],
            group_by=group_by,
            **wrap_kwargs,
        )

    # ############# Conflicts ############# #

    @cached_method
    def has_conflicts(self, **kwargs) -> bool:
        """Return whether conflicts exist in the mapped column records.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.has_conflicts`.

        Returns:
            bool: True if conflicts are present, otherwise False.
        """
        return self.get_map_field("col").has_conflicts(**kwargs)

    def coverage_map(self, **kwargs) -> tp.SeriesFrame:
        """Return the coverage map for the records using the mapped column field.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.coverage_map`.

        Returns:
            SeriesFrame: Resulting coverage map.
        """
        return self.get_map_field("col").coverage_map(**kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `Records.stats`.

        Merges the defaults from `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.records`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        records_stats_cfg = settings["records"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), records_stats_cfg)

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
            count=dict(title="Count", calc_func="count", tags="records"),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def prepare_customdata(
        self,
        incl_fields: tp.Optional[tp.Sequence[str]] = None,
        excl_fields: tp.Optional[tp.Sequence[str]] = None,
        append_info: tp.Optional[tp.Sequence[tp.Tuple]] = None,
        mask: tp.Optional[tp.Array1d] = None,
    ) -> tp.Tuple[tp.Array2d, str]:
        """Prepare `customdata` and `hoverinfo` for Plotly figures.

        This function constructs `customdata` arrays and hover templates by including available fields
        from the record's data type. If `incl_fields` is provided, only those fields are considered
        unless they have the field configuration `as_customdata` disabled or are explicitly listed
        in `excl_fields`. Additionally, field hover templates can be defined via
        `vectorbtpro.utils.template.Sub` with substitutions for `title` and `index`.
        Mapped fields are automatically converted to strings.

        To append additional custom arrays, provide `append_info` as a list of tuples, each containing
        a 1D NumPy array, a title, and optionally a `hoverinfo` template. If an array's data type is
        `object`, it is treated as strings; otherwise, it is treated as numbers.

        Args:
            incl_fields (Optional[Sequence[str]]): Fields to include in the `customdata`.
            excl_fields (Optional[Sequence[str]]): Fields to exclude from the `customdata`.
            append_info (Optional[Sequence[Tuple]]): Additional tuples with a 1D array, title,
                and optional `hoverinfo` template.
            mask (Optional[Array1d]): Boolean mask to apply to the `customdata` arrays.

        Returns:
            Tuple[Array2d, str]: Tuple where the first element is the `customdata` array and
                the second element is the `hoverinfo` string.
        """
        customdata_info = []
        if incl_fields is not None:
            iterate_over_names = incl_fields
        else:
            iterate_over_names = self.field_config.get("dtype").names
        for field in iterate_over_names:
            if excl_fields is not None and field in excl_fields:
                continue
            field_as_customdata = self.get_field_setting(field, "as_customdata", True)
            if field_as_customdata:
                numeric_customdata = self.get_field_setting(field, "mapping", None)
                if numeric_customdata is not None:
                    field_arr = self.get_apply_mapping_str_arr(field)
                    field_hovertemplate = self.get_field_setting(
                        field,
                        "hovertemplate",
                        "$title: %{customdata[$index]}",
                    )
                else:
                    field_arr = self.get_apply_mapping_arr(field)
                    field_hovertemplate = self.get_field_setting(
                        field,
                        "hovertemplate",
                        "$title: %{customdata[$index]:,}",
                    )
                if isinstance(field_hovertemplate, str):
                    field_hovertemplate = Sub(field_hovertemplate)
                field_title = self.get_field_title(field)
                customdata_info.append((field_arr, field_title, field_hovertemplate))
        if append_info is not None:
            for info in append_info:
                checks.assert_instance_of(info, tuple)
                if len(info) == 2:
                    if info[0].dtype == object:
                        info += ("$title: %{customdata[$index]}",)
                    else:
                        info += ("$title: %{customdata[$index]:,}",)
                if isinstance(info[2], str):
                    info = (info[0], info[1], Sub(info[2]))
                customdata_info.append(info)
        customdata = []
        hovertemplate = []
        for i in range(len(customdata_info)):
            if mask is not None:
                customdata.append(customdata_info[i][0][mask])
            else:
                customdata.append(customdata_info[i][0])
            _hovertemplate = customdata_info[i][2].substitute(
                dict(title=customdata_info[i][1], index=i)
            )
            if not _hovertemplate.startswith("<br>"):
                _hovertemplate = "<br>" + _hovertemplate
            hovertemplate.append(_hovertemplate)
        return np.stack(customdata, axis=1), "\n".join(hovertemplate)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `Records.plots`.

        Merges the defaults from `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.records`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        records_plots_cfg = settings["records"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), records_plots_cfg)

    @property
    def subplots(self) -> Config:
        return self._subplots

    # ############# Docs ############# #

    @classmethod
    def build_field_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Return the field configuration documentation for the class.

        Builds and returns a formatted string of the field configuration documentation by
        processing the docstring of the `Records.field_config` attribute from the provided source
        class and substituting class-specific values.

        Args:
            source_cls (Optional[type]): Source class providing the original configuration.

                Defaults to `Records` if not provided.

        Returns:
            str: Formatted field configuration documentation.
        """
        if source_cls is None:
            source_cls = Records
        return string.Template(
            inspect.cleandoc(get_dict_attr(source_cls, "field_config").__doc__)
        ).substitute(
            {"field_config": cls.field_config.prettify_doc(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_field_config_doc(
        cls, __pdoc__: dict, source_cls: tp.Optional[type] = None
    ) -> None:
        """Update the field configuration documentation in the provided doc dictionary.

        This method should be called on each subclass that overrides `Records.field_config`
        to update its documentation in the `__pdoc__` dictionary with the formatted field
        configuration documentation generated by `Records.build_field_config_doc`.

        Args:
            __pdoc__ (dict): Dictionary mapping objects to their documentation strings.
            source_cls (Optional[type]): Source class providing the original configuration.

        Returns:
            None
        """
        __pdoc__[cls.__name__ + ".field_config"] = cls.build_field_config_doc(source_cls=source_cls)


Records.override_field_config_doc(__pdoc__)
Records.override_metrics_doc(__pdoc__)
Records.override_subplots_doc(__pdoc__)
