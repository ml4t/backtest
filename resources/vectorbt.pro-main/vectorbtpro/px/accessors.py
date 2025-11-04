# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Pandas accessors for Plotly Express functions.

!!! note
    The accessors in this module do not utilize caching.
"""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("plotly")


from vectorbtpro import _typing as tp
from vectorbtpro.accessors import (
    register_df_vbt_accessor,
    register_sr_vbt_accessor,
    register_vbt_accessor,
)
from vectorbtpro.base.accessors import BaseAccessor, BaseDFAccessor, BaseSRAccessor
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.px.decorators import attach_px_methods

__all__ = [
    "PXAccessor",
    "PXSRAccessor",
    "PXDFAccessor",
]


@register_vbt_accessor("px")
@attach_px_methods
class PXAccessor(BaseAccessor):
    """Class that provides a Pandas accessor to invoke Plotly Express functions on Series and DataFrame objects.

    Accessible via `pd.Series.vbt.px` and `pd.DataFrame.vbt.px`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Object to be wrapped.
        **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseAccessor`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> pd.Series([1, 2, 3]).vbt.px.bar().show()
        ```

        ![](/assets/images/api/px_bar.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/px_bar.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        BaseAccessor.__init__(self, wrapper, obj=obj, **kwargs)


@register_sr_vbt_accessor("px")
class PXSRAccessor(PXAccessor, BaseSRAccessor):
    """Class that provides a Pandas accessor to apply Plotly Express functions on Series objects.

    Accessible via `pd.Series.vbt.px`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Object to be wrapped.
        **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        BaseSRAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            PXAccessor.__init__(self, wrapper, obj=obj, **kwargs)


@register_df_vbt_accessor("px")
class PXDFAccessor(PXAccessor, BaseDFAccessor):
    """Class that provides a Pandas accessor to apply Plotly Express functions on DataFrame objects.

    Accessible via `pd.DataFrame.vbt.px`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Object to be wrapped.
        **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        BaseDFAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            PXAccessor.__init__(self, wrapper, obj=obj, **kwargs)
