# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package for registering objects across vectorbtpro."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.registries.ca_registry import *
    from vectorbtpro.registries.ch_registry import *
    from vectorbtpro.registries.jit_registry import *
    from vectorbtpro.registries.pbar_registry import *
