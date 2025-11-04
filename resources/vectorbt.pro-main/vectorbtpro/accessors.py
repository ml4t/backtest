# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing root Pandas accessors.

An accessor adds an additional namespace to Pandas objects.

The `vectorbtpro.accessors` module registers a custom `vbt` accessor on each `pd.Index`, `pd.Series`,
and `pd.DataFrame` object. It is the main entry point for all other accessors:

```plaintext
vbt.base.accessors.BaseIDX/SR/DFAccessor       -> pd.Index/Series/DataFrame.vbt.*
vbt.generic.accessors.GenericSR/DFAccessor     -> pd.Series/DataFrame.vbt.*
vbt.signals.accessors.SignalsSR/DFAccessor     -> pd.Series/DataFrame.vbt.signals.*
vbt.returns.accessors.ReturnsSR/DFAccessor     -> pd.Series/DataFrame.vbt.returns.*
vbt.ohlcv.accessors.OHLCVDFAccessor            -> pd.DataFrame.vbt.ohlcv.*
vbt.px.accessors.PXSR/DFAccessor               -> pd.Series/DataFrame.vbt.px.*
```

Additionally, some accessors subclass other accessors, forming the following inheritance hierarchy:

```plaintext
vbt.base.accessors.BaseIDXAccessor
vbt.base.accessors.BaseSR/DFAccessor
    -> vbt.generic.accessors.GenericSR/DFAccessor
        -> vbt.signals.accessors.SignalsSR/DFAccessor
        -> vbt.returns.accessors.ReturnsSR/DFAccessor
        -> vbt.ohlcv.accessors.OHLCVDFAccessor
    -> vbt.px.accessors.PXSR/DFAccessor
```

For example, the method `pd.Series.vbt.to_2d_array` is also available as
`pd.Series.vbt.returns.to_2d_array`.

Class methods of any accessor can be accessed using the shortcuts `pd_acc`, `sr_acc`, and `df_acc`:

```pycon
>>> from vectorbtpro import *

>>> vbt.pd_acc.signals.generate
<bound method SignalsAccessor.generate of <class 'vectorbtpro.signals.accessors.SignalsAccessor'>>
```

!!! note
    Accessors in vectorbtpro are not cached, so querying `df.vbt` twice will invoke `Vbt_DFAccessor` twice.
    This behavior can be changed in global settings.
"""

import pandas as pd
from pandas.core.accessor import DirNamesMixin

from vectorbtpro import _typing as tp
from vectorbtpro.base.accessors import BaseIDXAccessor
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic.accessors import GenericAccessor, GenericDFAccessor, GenericSRAccessor
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "Vbt_Accessor",
    "Vbt_SRAccessor",
    "Vbt_DFAccessor",
    "idx_acc",
    "pd_acc",
    "sr_acc",
    "df_acc",
]

__pdoc__ = {}

ParentAccessorT = tp.TypeVar("ParentAccessorT", bound=object)
AccessorT = tp.TypeVar("AccessorT", bound=object)


class Accessor(Base):
    """Class representing a custom accessor.

    Args:
        name (str): Name under which the accessor is registered.
        accessor (Type[Accessor]): Accessor type for instantiation.
    """

    def __init__(self, name: str, accessor: tp.Type[AccessorT]) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj: ParentAccessorT, cls: DirNamesMixin) -> AccessorT:
        if obj is None:
            return self._accessor
        if isinstance(obj, (pd.Index, pd.Series, pd.DataFrame)):
            accessor_obj = self._accessor(obj)
        elif issubclass(self._accessor, type(obj)):
            accessor_obj = obj.replace(cls_=self._accessor)
        else:
            accessor_obj = self._accessor(obj.wrapper, obj=obj._obj)
        return accessor_obj


class CachedAccessor(Base):
    """Class representing a cached accessor.

    Args:
        name (str): Name under which the accessor is registered.
        accessor (Type[Accessor]): Accessor type for instantiation.
    """

    def __init__(self, name: str, accessor: tp.Type[AccessorT]) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj: ParentAccessorT, cls: DirNamesMixin) -> AccessorT:
        if obj is None:
            return self._accessor
        if isinstance(obj, (pd.Index, pd.Series, pd.DataFrame)):
            accessor_obj = self._accessor(obj)
        elif issubclass(self._accessor, type(obj)):
            accessor_obj = obj.replace(cls_=self._accessor)
        else:
            accessor_obj = self._accessor(obj.wrapper, obj=obj._obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def register_accessor(name: str, cls: tp.Type[DirNamesMixin]) -> tp.Callable:
    """Register a custom accessor.

    Args:
        name (str): Name to register the accessor under.
        cls (Type[DirNamesMixin]): Class extending `DirNamesMixin`.

    Returns:
        Callable: Decorator function to register the accessor.

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """

    def decorator(accessor: tp.Type[AccessorT]) -> tp.Type[AccessorT]:
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if hasattr(cls, name):
            warn(
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                "attribute with the same name."
            )
        if caching_cfg["use_cached_accessors"]:
            setattr(cls, name, CachedAccessor(name, accessor))
        else:
            setattr(cls, name, Accessor(name, accessor))
        cls._accessors.add(name)
        return accessor

    return decorator


def register_index_accessor(name: str) -> tp.Callable:
    """Decorator to register a custom `pd.Index` accessor.

    Args:
        name (str): Name to register the accessor under.

    Returns:
        Callable: Decorator function to register the accessor.
    """
    return register_accessor(name, pd.Index)


def register_series_accessor(name: str) -> tp.Callable:
    """Decorator to register a custom `pd.Series` accessor.

    Args:
        name (str): Name to register the accessor under.

    Returns:
        Callable: Decorator function to register the accessor.
    """
    return register_accessor(name, pd.Series)


def register_dataframe_accessor(name: str) -> tp.Callable:
    """Decorator to register a custom `pd.DataFrame` accessor.

    Args:
        name (str): Name to register the accessor under.

    Returns:
        Callable: Decorator function to register the accessor.
    """
    return register_accessor(name, pd.DataFrame)


@register_index_accessor("vbt")
class Vbt_IDXAccessor(DirNamesMixin, BaseIDXAccessor):
    """Class representing the main vectorbtpro accessor for `pd.Index`.

    Args:
        obj (Index): Pandas Index object.
        **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseIDXAccessor`.
    """

    def __init__(self, obj: tp.Index, **kwargs) -> None:
        self._obj = obj

        DirNamesMixin.__init__(self)
        BaseIDXAccessor.__init__(self, obj, **kwargs)


idx_acc = Vbt_IDXAccessor
"""Shortcut for `Vbt_IDXAccessor`."""

__pdoc__["idx_acc"] = False


class Vbt_Accessor(DirNamesMixin, GenericAccessor):
    """Class representing the main vectorbtpro accessor for `pd.Series` and `pd.DataFrame`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Optional object for initialization.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        DirNamesMixin.__init__(self)
        GenericAccessor.__init__(self, wrapper, obj=obj, **kwargs)


pd_acc = Vbt_Accessor
"""Shortcut for `Vbt_Accessor`."""

__pdoc__["pd_acc"] = False


@register_series_accessor("vbt")
class Vbt_SRAccessor(DirNamesMixin, GenericSRAccessor):
    """Class representing the main vectorbtpro accessor for `pd.Series`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Optional object for initialization.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericSRAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        DirNamesMixin.__init__(self)
        GenericSRAccessor.__init__(self, wrapper, obj=obj, **kwargs)


sr_acc = Vbt_SRAccessor
"""Shortcut for `Vbt_SRAccessor`."""

__pdoc__["sr_acc"] = False


@register_dataframe_accessor("vbt")
class Vbt_DFAccessor(DirNamesMixin, GenericDFAccessor):
    """Class representing the main vectorbtpro accessor for `pd.DataFrame`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Optional data object.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericDFAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        DirNamesMixin.__init__(self)
        GenericDFAccessor.__init__(self, wrapper, obj=obj, **kwargs)


df_acc = Vbt_DFAccessor
"""Shortcut for `Vbt_DFAccessor`."""

__pdoc__["df_acc"] = False


def register_vbt_accessor(name: str, parent: tp.Type[DirNamesMixin] = Vbt_Accessor) -> tp.Callable:
    """Decorator to register an accessor on top of a parent accessor.

    Args:
        name (str): Name to register the accessor.
        parent (Type[DirNamesMixin]): Parent accessor class to extend (e.g., `Vbt_Accessor`).

    Returns:
        Callable: Decorator function to register the accessor.
    """
    return register_accessor(name, parent)


def register_idx_vbt_accessor(
    name: str, parent: tp.Type[DirNamesMixin] = Vbt_IDXAccessor
) -> tp.Callable:
    """Decorator to register a `pd.Index` accessor.

    Args:
        name (str): Name to register the accessor.
        parent (Type[DirNamesMixin]): Parent accessor class to extend (e.g., `Vbt_IDXAccessor`).

    Returns:
        Callable: Decorator function to register the accessor.
    """
    return register_accessor(name, parent)


def register_sr_vbt_accessor(
    name: str, parent: tp.Type[DirNamesMixin] = Vbt_SRAccessor
) -> tp.Callable:
    """Decorator to register a `pd.Series` accessor.

    Args:
        name (str): Name to register the accessor.
        parent (Type[DirNamesMixin]): Parent accessor class to extend (e.g., `Vbt_SRAccessor`).

    Returns:
        Callable: Decorator function to register the accessor.
    """
    return register_accessor(name, parent)


def register_df_vbt_accessor(
    name: str, parent: tp.Type[DirNamesMixin] = Vbt_DFAccessor
) -> tp.Callable:
    """Decorator to register a `pd.DataFrame` accessor.

    Args:
        name (str): Name to register the accessor.
        parent (Type[DirNamesMixin]): Parent accessor class to extend (e.g., `Vbt_DFAccessor`).

    Returns:
        Callable: Decorator function to register the accessor.
    """
    return register_accessor(name, parent)
