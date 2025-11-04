# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for random number generation."""

import random

import numpy as np

from vectorbtpro.registries.jit_registry import register_jitted

__all__ = [
    "set_seed",
]


@register_jitted(cache=True)
def set_seed_nb(seed: int) -> None:
    """Set the seed for numba jitted functions.

    Args:
        seed (int): Random seed for deterministic output.

    Returns:
        None
    """
    np.random.seed(seed)


def set_seed(seed: int) -> None:
    """Set seeds across random, NumPy, and numba contexts.

    Args:
        seed (int): Random seed for deterministic output.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    set_seed_nb(seed)
