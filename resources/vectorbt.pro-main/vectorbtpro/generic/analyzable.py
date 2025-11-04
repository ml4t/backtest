# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `Analyzable` class for analyzing data."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.generic.plots_builder import PlotsBuilderMixin
from vectorbtpro.generic.stats_builder import StatsBuilderMixin

__all__ = [
    "Analyzable",
]


class MetaAnalyzable(type(Wrapping), type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    """Metaclass for the `Analyzable` class."""

    pass


AnalyzableT = tp.TypeVar("AnalyzableT", bound="Analyzable")


class Analyzable(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaAnalyzable):
    """Class that can be analyzed by computing and plotting various attributes.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.Wrapping`.
    """

    def __init__(self, wrapper: ArrayWrapper, **kwargs) -> None:
        Wrapping.__init__(self, wrapper, **kwargs)
        StatsBuilderMixin.__init__(self)
        PlotsBuilderMixin.__init__(self)
