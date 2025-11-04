# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for selection."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define

__all__ = [
    "PosSel",
    "LabelSel",
]


@define
class PosSel(DefineMixin):
    """Class representing a positional selection."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more positions."""


@define
class LabelSel(DefineMixin):
    """Class representing a label-based selection."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more labels."""
