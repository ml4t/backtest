# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing base classes and functions for grouping.

The `Grouper` class encapsulates metadata related to grouping an index. It provides details such as the number
of groups, the starting indices of groups, and other information useful for grouping reduction operations.
It also supports dynamically enabling, disabling, or modifying groups while enforcing allowed operations.
"""

import numpy as np
import pandas as pd
from pandas.core.groupby import GroupBy as PandasGroupBy
from pandas.core.resample import Resampler as PandasResampler

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import indexes
from vectorbtpro.base.grouping import nb
from vectorbtpro.base.indexes import ExceptLevel
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.array_ import is_sorted
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.decorators import cached_method
from vectorbtpro.utils.template import CustomTemplate

__all__ = [
    "Grouper",
]

GrouperT = tp.TypeVar("GrouperT", bound="Grouper")


class Grouper(Configured):
    """Class for grouping indices and managing group metadata.

    This class stores metadata and offers methods to create and manipulate groupings for a Pandas Index.
    It provides information such as the number of groups, the starting indices for each group, and other details
    that are useful for grouping reduction operations.

    Args:
        index (Index): Original Pandas Index to group.
        group_by (GroupByLike): Grouping specification.

            Accepts:

            * boolean (False for no grouping, True for one group).
            * integer (MultiIndex level by position).
            * string (MultiIndex level by name).
            * sequence of integers or strings that is shorter than `index` (multiple MultiIndex levels).
            * any other sequence such as Pandas Index that has the same length as `index`.
            * `vectorbtpro.base.indexes.ExceptLevel` object (to exclude levels).
            * `vectorbtpro.utils.template.CustomTemplate` object with `index` as context (to substitute levels).
        def_lvl_name (Hashable): Default level name for groups.
        allow_enable (bool): Indicates if enabling grouping is permitted when `group_by` is None.
        allow_disable (bool): Indicates if disabling grouping is permitted when `group_by` is not None.
        allow_modify (bool): Indicates if modifying groups is allowed.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(
        self,
        index: tp.Index,
        group_by: tp.GroupByLike = None,
        def_lvl_name: tp.Hashable = "group",
        allow_enable: bool = True,
        allow_disable: bool = True,
        allow_modify: bool = True,
        **kwargs,
    ) -> None:
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if group_by is None or group_by is False:
            group_by = None
        else:
            group_by = self.group_by_to_index(index, group_by, def_lvl_name=def_lvl_name)

        self._index = index
        self._group_by = group_by
        self._def_lvl_name = def_lvl_name
        self._allow_enable = allow_enable
        self._allow_disable = allow_disable
        self._allow_modify = allow_modify

        Configured.__init__(
            self,
            index=index,
            group_by=group_by,
            def_lvl_name=def_lvl_name,
            allow_enable=allow_enable,
            allow_disable=allow_disable,
            allow_modify=allow_modify,
            **kwargs,
        )

    @property
    def index(self) -> tp.Index:
        """Original Pandas Index used for grouping.

        Returns:
            Index: Original Pandas Index.
        """
        return self._index

    @property
    def group_by(self) -> tp.GroupBy:
        """Group-by mapping generated from the provided grouping criteria.

        Returns:
            GroupBy: Group-by mapping generated from the provided grouping criteria.
        """
        return self._group_by

    @property
    def def_lvl_name(self) -> tp.Hashable:
        """Default group level name.

        Returns:
            Hashable: Default level name for groups.
        """
        return self._def_lvl_name

    @property
    def allow_enable(self) -> bool:
        """Indicates if enabling grouping is permitted.

        Returns:
            bool: True if enabling grouping is allowed, False otherwise.
        """
        return self._allow_enable

    @property
    def allow_disable(self) -> bool:
        """Indicates if disabling grouping is permitted.

        Returns:
            bool: True if disabling grouping is allowed, False otherwise.
        """
        return self._allow_disable

    @property
    def allow_modify(self) -> bool:
        """Indicates if modifying groups is allowed.

        Returns:
            bool: True if modifying groups is allowed, False otherwise.
        """
        return self._allow_modify

    @classmethod
    def group_by_to_index(
        cls,
        index: tp.Index,
        group_by: tp.GroupByLike,
        def_lvl_name: tp.Hashable = "group",
    ) -> tp.GroupBy:
        """Convert the provided `group_by` specification into a Pandas Index.

        Args:
            index (Index): Original Pandas Index.
            group_by (GroupByLike): Grouping specification.
            def_lvl_name (Hashable): Default level name for groups.

        Returns:
            GroupBy: Resulting group-by mapping as a Pandas Index,
                or the original `group_by` if it is None or False.

        !!! note
            The index and the `group_by` mapper must have the same length.
        """
        if group_by is None or group_by is False:
            return group_by
        if isinstance(group_by, CustomTemplate):
            group_by = group_by.substitute(
                context=dict(index=index), strict=True, eval_id="group_by"
            )
        if group_by is True:
            group_by = pd.Index(["group"] * len(index), name=def_lvl_name)
        elif isinstance(index, pd.MultiIndex) or isinstance(group_by, (ExceptLevel, int, str)):
            if isinstance(group_by, ExceptLevel):
                except_levels = group_by.value
                if isinstance(except_levels, (int, str)):
                    except_levels = [except_levels]
                new_group_by = []
                for i, name in enumerate(index.names):
                    if i not in except_levels and name not in except_levels:
                        new_group_by.append(name)
                if len(new_group_by) == 0:
                    group_by = pd.Index(["group"] * len(index), name=def_lvl_name)
                else:
                    if len(new_group_by) == 1:
                        new_group_by = new_group_by[0]
                    group_by = indexes.select_levels(index, new_group_by)
            elif isinstance(group_by, (int, str)):
                group_by = indexes.select_levels(index, group_by)
            elif (
                isinstance(group_by, (tuple, list))
                and not isinstance(group_by[0], pd.Index)
                and len(group_by) <= len(index.names)
            ):
                try:
                    group_by = indexes.select_levels(index, group_by)
                except (IndexError, KeyError):
                    pass
        if not isinstance(group_by, pd.Index):
            if isinstance(group_by[0], pd.Index):
                group_by = pd.MultiIndex.from_arrays(group_by)
            else:
                group_by = pd.Index(group_by, name=def_lvl_name)
        if len(group_by) != len(index):
            raise ValueError("group_by and index must have the same length")
        return group_by

    @classmethod
    def group_by_to_groups_and_index(
        cls,
        index: tp.Index,
        group_by: tp.GroupByLike,
        def_lvl_name: tp.Hashable = "group",
    ) -> tp.Tuple[tp.Array1d, tp.Index]:
        """Return an array of group indices corresponding to the original index and the grouped index.

        Args:
            index (Index): Original Pandas Index.
            group_by (GroupByLike): Grouping specification.
            def_lvl_name (Hashable): Default level name for groups.

        Returns:
            Tuple[ndarray, Index]: Tuple containing:

                * Array of integer group codes for the original index.
                * Grouped Pandas Index.
        """
        if group_by is None or group_by is False:
            return np.arange(len(index)), index

        group_by = cls.group_by_to_index(index, group_by, def_lvl_name)
        codes, uniques = pd.factorize(group_by)
        if not isinstance(uniques, pd.Index):
            new_index = pd.Index(uniques)
        else:
            new_index = uniques
        if isinstance(group_by, pd.MultiIndex):
            new_index.names = group_by.names
        elif isinstance(group_by, (pd.Index, pd.Series)):
            new_index.name = group_by.name
        return codes, new_index

    @classmethod
    def iter_group_lens(cls, group_lens: tp.GroupLens) -> tp.Iterator[tp.GroupIdxs]:
        """Iterate over group indices based on group lengths.

        Args:
            group_lens (GroupLens): Array defining the number of columns in each group.

        Yields:
            GroupIdxs: Array of indices representing a single group.
        """
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        for group in range(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]
            yield np.arange(from_col, to_col)

    @classmethod
    def iter_group_map(cls, group_map: tp.GroupMap) -> tp.Iterator[tp.GroupIdxs]:
        """Iterate over group indices based on a group map.

        Args:
            group_map (GroupMap): Tuple of indices and lengths for each group.

        Yields:
            GroupIdxs: Array of indices representing a single group.
        """
        group_idxs, group_lens = group_map
        group_start = 0
        group_end = 0
        for group in range(len(group_lens)):
            group_len = group_lens[group]
            group_end += group_len
            yield group_idxs[group_start:group_end]
            group_start += group_len

    @classmethod
    def from_pd_group_by(
        cls: tp.Type[GrouperT],
        pd_group_by: tp.PandasGroupByLike,
        **kwargs,
    ) -> GrouperT:
        """Build a `Grouper` instance from a Pandas `GroupBy` object.

        Args:
            cls (Type[Grouper]): `Grouper` class.
            pd_group_by (PandasGroupByLike): Pandas `GroupBy` or
                `vectorbtpro.base.resampling.base.Resampler` object.
            **kwargs: Keyword arguments for `Grouper`.

        Returns:
            Grouper: New instance of `Grouper`.
        """
        from vectorbtpro.base.merging import concat_arrays

        if not isinstance(pd_group_by, (PandasGroupBy, PandasResampler)):
            raise TypeError("pd_group_by must be an instance of GroupBy or Resampler")
        indices = list(pd_group_by.indices.values())
        group_lens = np.asarray(list(map(len, indices)))
        groups = np.full(int(np.sum(group_lens)), 0, dtype=int_)
        group_start_idxs = np.cumsum(group_lens)[1:] - group_lens[1:]
        groups[group_start_idxs] = 1
        groups = np.cumsum(groups)
        index = pd.Index(concat_arrays(indices))
        group_by = pd.Index(list(pd_group_by.indices.keys()), name="group")[groups]
        return cls(
            index=index,
            group_by=group_by,
            **kwargs,
        )

    def is_grouped(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether the index is grouped.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.

        Returns:
            bool: True if the index is grouped, False otherwise.
        """
        if group_by is False:
            return False
        if group_by is None:
            group_by = self.group_by
        return group_by is not None

    def is_grouping_enabled(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether grouping is enabled.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.

        Returns:
            bool: True if grouping is enabled, False otherwise.
        """
        return self.group_by is None and self.is_grouped(group_by=group_by)

    def is_grouping_disabled(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether grouping is disabled.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.

        Returns:
            bool: True if grouping is disabled, False otherwise.
        """
        return self.group_by is not None and not self.is_grouped(group_by=group_by)

    @cached_method(whitelist=True)
    def is_grouping_modified(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether the grouping has been modified, disregarding changes in group labels.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.

        Returns:
            bool: True if the grouping has been modified, False otherwise.
        """
        if group_by is None or (group_by is False and self.group_by is None):
            return False
        group_by = self.group_by_to_index(self.index, group_by, def_lvl_name=self.def_lvl_name)
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            if not pd.Index.equals(group_by, self.group_by):
                groups1 = self.group_by_to_groups_and_index(
                    self.index,
                    group_by,
                    def_lvl_name=self.def_lvl_name,
                )[0]
                groups2 = self.group_by_to_groups_and_index(
                    self.index,
                    self.group_by,
                    def_lvl_name=self.def_lvl_name,
                )[0]
                if not np.array_equal(groups1, groups2):
                    return True
            return False
        return True

    @cached_method(whitelist=True)
    def is_grouping_changed(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether the grouping has changed in any way.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.

        Returns:
            bool: True if the grouping has changed, False otherwise.
        """
        if group_by is None or (group_by is False and self.group_by is None):
            return False
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            if pd.Index.equals(group_by, self.group_by):
                return False
        return True

    def is_group_count_changed(self, group_by: tp.GroupByLike = None) -> bool:
        """Check whether the number of groups has changed.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.

        Returns:
            bool: True if the number of groups has changed, False otherwise.
        """
        if group_by is None or (group_by is False and self.group_by is None):
            return False
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            return len(group_by) != len(self.group_by)
        return True

    def check_group_by(
        self,
        group_by: tp.GroupByLike = None,
        allow_enable: tp.Optional[bool] = None,
        allow_disable: tp.Optional[bool] = None,
        allow_modify: tp.Optional[bool] = None,
    ) -> None:
        """Check the provided `group_by` object against grouping restrictions.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.
            allow_enable (Optional[bool]): Whether enabling grouping is allowed.
            allow_disable (Optional[bool]): Whether disabling grouping is allowed.
            allow_modify (Optional[bool]): Whether modifying groups is allowed.

        Returns:
            None
        """
        if allow_enable is None:
            allow_enable = self.allow_enable
        if allow_disable is None:
            allow_disable = self.allow_disable
        if allow_modify is None:
            allow_modify = self.allow_modify

        if self.is_grouping_enabled(group_by=group_by):
            if not allow_enable:
                raise ValueError("Enabling grouping is not allowed")
        elif self.is_grouping_disabled(group_by=group_by):
            if not allow_disable:
                raise ValueError("Disabling grouping is not allowed")
        elif self.is_grouping_modified(group_by=group_by):
            if not allow_modify:
                raise ValueError("Modifying groups is not allowed")

    def resolve_group_by(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.GroupBy:
        """Resolve the `group_by` value using either the provided argument or the object's attribute.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.
            **kwargs: Keyword arguments for `Grouper.check_group_by`.

        Returns:
            GroupBy: Resolved grouping index.
        """
        if group_by is None:
            group_by = self.group_by
        if group_by is False and self.group_by is None:
            group_by = None
        self.check_group_by(group_by=group_by, **kwargs)
        return self.group_by_to_index(self.index, group_by, def_lvl_name=self.def_lvl_name)

    @cached_method(whitelist=True)
    def get_groups_and_index(
        self, group_by: tp.GroupByLike = None, **kwargs
    ) -> tp.Tuple[tp.Array1d, tp.Index]:
        """Return the groups array and associated index computed from the resolved grouping.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.
            **kwargs: Keyword arguments for `Grouper.resolve_group_by`.

        Returns:
            Tuple[Array1d, Index]: Tuple containing the groups array and the grouped index.
        """
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        return self.group_by_to_groups_and_index(
            self.index, group_by, def_lvl_name=self.def_lvl_name
        )

    def get_groups(self, **kwargs) -> tp.Array1d:
        """Return the groups array.

        Args:
            **kwargs: Keyword arguments for `Grouper.get_groups_and_index`.

        Returns:
            Array1d: Array representing group labels for each index entry.
        """
        return self.get_groups_and_index(**kwargs)[0]

    def get_index(self, **kwargs) -> tp.Index:
        """Return the grouped index.

        Args:
            **kwargs: Keyword arguments for `Grouper.get_groups_and_index`.

        Returns:
            Index: Grouped index.
        """
        return self.get_groups_and_index(**kwargs)[1]

    get_grouped_index = get_index

    @property
    def grouped_index(self) -> tp.Index:
        """Grouped index computed from the current grouping configuration.

        Returns:
            Index: Grouped index obtained via `Grouper.get_grouped_index`.
        """
        return self.get_grouped_index()

    def get_stretched_index(self, **kwargs) -> tp.Index:
        """Return the stretched index, computed by applying the groups mapping to the index.

        Args:
            **kwargs: Keyword arguments for `Grouper.get_groups_and_index`.

        Returns:
            Index: Stretched index.
        """
        groups, index = self.get_groups_and_index(**kwargs)
        return index[groups]

    def get_group_count(self, **kwargs) -> int:
        """Return the number of groups computed from the grouped index.

        Args:
            **kwargs: Keyword arguments for `Grouper.get_index`.

        Returns:
            int: Number of groups.
        """
        return len(self.get_index(**kwargs))

    @cached_method(whitelist=True)
    def is_sorted(self, group_by: tp.GroupByLike = None, **kwargs) -> bool:
        """Determine if groups are monolithic and sorted.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.
            **kwargs: Keyword arguments for `Grouper.resolve_group_by`.

        Returns:
            bool: True if groups are monolithic and sorted, False otherwise.
        """
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        groups = self.get_groups(group_by=group_by)
        return is_sorted(groups)

    @cached_method(whitelist=True)
    def get_group_lens(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        **kwargs,
    ) -> tp.GroupLens:
        """Return the lengths of each group computed from the current grouping.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            **kwargs: Keyword arguments for `Grouper.resolve_group_by`.

        Returns:
            GroupLens: Array containing the length of each group.

        Raises:
            ValueError: If the grouping is not monolithic and sorted.

        See:
            `vectorbtpro.base.grouping.nb.get_group_lens_nb`
        """
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        if group_by is None or group_by is False:  # no grouping
            return np.full(len(self.index), 1)
        if not self.is_sorted(group_by=group_by):
            raise ValueError("group_by must form monolithic, sorted groups")
        groups = self.get_groups(group_by=group_by)
        func = jit_reg.resolve_option(nb.get_group_lens_nb, jitted)
        return func(groups)

    def get_group_start_idxs(self, **kwargs) -> tp.Array1d:
        """Return the starting indices of each group.

        Args:
            **kwargs: Keyword arguments for `Grouper.get_group_lens`.

        Returns:
            Array1d: Array containing the first index of each group.
        """
        group_lens = self.get_group_lens(**kwargs)
        return np.cumsum(group_lens) - group_lens

    def get_group_end_idxs(self, **kwargs) -> tp.Array1d:
        """Return the ending indices of each group.

        Args:
            **kwargs: Keyword arguments for `Grouper.get_group_lens`.

        Returns:
            Array1d: Array containing the end index for each group.
        """
        group_lens = self.get_group_lens(**kwargs)
        return np.cumsum(group_lens)

    @cached_method(whitelist=True)
    def get_group_map(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        **kwargs,
    ) -> tp.GroupMap:
        """Return the group mapping computed from the resolved grouping.

        Args:
            group_by (GroupByLike): Grouping specification.

                If not provided, uses `Grouper.group_by`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            **kwargs: Keyword arguments for `Grouper.resolve_group_by`.

        Returns:
            GroupMap: Tuple containing the group mapping.

        See:
            `vectorbtpro.base.grouping.nb.get_group_map_nb`

        !!! note
            If no grouping is applied, a default mapping is returned.
        """
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        if group_by is None or group_by is False:  # no grouping
            return np.arange(len(self.index)), np.full(len(self.index), 1)
        groups, new_index = self.get_groups_and_index(group_by=group_by)
        func = jit_reg.resolve_option(nb.get_group_map_nb, jitted)
        return func(groups, len(new_index))

    def iter_group_idxs(self, **kwargs) -> tp.Iterator[tp.GroupIdxs]:
        """Iterate over the indices corresponding to each group.

        Args:
            **kwargs: Keyword arguments for `Grouper.get_group_map`.

        Returns:
            Iterator[GroupIdxs]: Iterator over the indices for each group.
        """
        group_map = self.get_group_map(**kwargs)
        return self.iter_group_map(group_map)

    def iter_groups(
        self,
        key_as_index: bool = False,
        **kwargs,
    ) -> tp.Iterator[tp.Tuple[tp.Union[tp.Hashable, tp.Index], tp.GroupIdxs]]:
        """Iterate over group keys and their associated indices.

        Args:
            key_as_index (bool): Whether to return the yielded key as an index.

            **kwargs: Keyword arguments for `Grouper.get_index` and `Grouper.iter_group_idxs`.

        Yields:
            Iterator[Tuple[Union[Hashable, Index], GroupIdxs]]: Tuple containing:

                * Identifier for the group.
                * Indices corresponding to the group.
        """
        index = self.get_index(**kwargs)
        for group, group_idxs in enumerate(self.iter_group_idxs(**kwargs)):
            if key_as_index:
                yield index[[group]], group_idxs
            else:
                yield index[group], group_idxs

    def select_groups(
        self,
        group_idxs: tp.Array1d,
        jitted: tp.JittedOption = None,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Select groups using provided indices while automatically choosing the selection method.

        Args:
            group_idxs (Array1d): Array of group indices to be selected.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            Tuple[Array1d, Array1d]: Tuple containing:

                * New group indices after selection.
                * New group array corresponding to the selected indices.

        See:
            * `vectorbtpro.base.grouping.nb.group_lens_select_nb` if `Grouper.is_sorted` returns True.
            * `vectorbtpro.base.grouping.nb.group_map_select_nb` if `Grouper.is_sorted` returns False.

        !!! note
            If `Grouper.is_sorted` returns True, selection is performed using group lengths (faster).
            Otherwise, selection is performed using a group map for greater flexibility.
        """
        from vectorbtpro.base.reshaping import to_1d_array

        if self.is_sorted():
            func = jit_reg.resolve_option(nb.group_lens_select_nb, jitted)
            new_group_idxs, new_groups = func(
                self.get_group_lens(), to_1d_array(group_idxs)
            )  # faster
        else:
            func = jit_reg.resolve_option(nb.group_map_select_nb, jitted)
            new_group_idxs, new_groups = func(
                self.get_group_map(), to_1d_array(group_idxs)
            )  # more flexible
        return new_group_idxs, new_groups
