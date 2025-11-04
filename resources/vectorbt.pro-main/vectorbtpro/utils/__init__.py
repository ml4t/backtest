# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing utilities used throughout vectorbtpro."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.utils.annotations import *
    from vectorbtpro.utils.array_ import *
    from vectorbtpro.utils.attr_ import *
    from vectorbtpro.utils.base import *
    from vectorbtpro.utils.caching import *
    from vectorbtpro.utils.chaining import *
    from vectorbtpro.utils.checks import *
    from vectorbtpro.utils.chunking import *
    from vectorbtpro.utils.colors import *
    from vectorbtpro.utils.config import *
    from vectorbtpro.utils.datetime_ import *
    from vectorbtpro.utils.datetime_nb import *
    from vectorbtpro.utils.decorators import *
    from vectorbtpro.utils.enum_ import *
    from vectorbtpro.utils.eval_ import *
    from vectorbtpro.utils.execution import *
    from vectorbtpro.utils.figure import *
    from vectorbtpro.utils.formatting import *
    from vectorbtpro.utils.hashing import *
    from vectorbtpro.utils.image_ import *
    from vectorbtpro.utils.jitting import *
    from vectorbtpro.utils.knowledge import *
    from vectorbtpro.utils.magic_decorators import *
    from vectorbtpro.utils.mapping import *
    from vectorbtpro.utils.math_ import *
    from vectorbtpro.utils.merging import *
    from vectorbtpro.utils.module_ import *
    from vectorbtpro.utils.params import *
    from vectorbtpro.utils.parsing import *
    from vectorbtpro.utils.path_ import *
    from vectorbtpro.utils.pbar import *
    from vectorbtpro.utils.pickling import *
    from vectorbtpro.utils.profiling import *
    from vectorbtpro.utils.random_ import *
    from vectorbtpro.utils.requests_ import *
    from vectorbtpro.utils.schedule_ import *
    from vectorbtpro.utils.search_ import *
    from vectorbtpro.utils.selection import *
    from vectorbtpro.utils.source import *
    from vectorbtpro.utils.tagging import *
    from vectorbtpro.utils.telegram import *
    from vectorbtpro.utils.template import *
    from vectorbtpro.utils.warnings_ import *

__import_if_installed__ = dict()
__import_if_installed__["figure"] = "plotly"
__import_if_installed__["telegram"] = "telegram"
