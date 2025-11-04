# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing math utility functions.

!!! info
    For default settings, see `vectorbtpro._settings.math`.
"""

import numpy as np

from vectorbtpro._settings import settings
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []

_use_tol = settings["math"]["use_tol"]
_rel_tol = settings["math"]["rel_tol"]
_abs_tol = settings["math"]["abs_tol"]
_use_round = settings["math"]["use_round"]
_decimals = settings["math"]["decimals"]


@register_jitted(cache=True)
def is_close_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Determine whether two float values are approximately equal, considering tolerance settings.

    Args:
        a (float): First float value.
        b (float): Second float value.
        use_tol (bool): Flag to enable tolerance-based comparison.
        rel_tol (float): Relative tolerance used for comparing values.
        abs_tol (float): Absolute tolerance used for comparing values.

    Returns:
        bool: True if the values are approximately equal based on the tolerance settings, otherwise False.
    """
    if np.isnan(a) or np.isnan(b):
        return False
    if np.isinf(a) or np.isinf(b):
        return False
    if a == b:
        return True
    return use_tol and abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@register_jitted(cache=True)
def is_less_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Determine whether the first float value is approximately less than the second value.

    Args:
        a (float): First float value.
        b (float): Second float value.
        use_tol (bool): Flag to enable tolerance-based comparison.
        rel_tol (float): Relative tolerance used for comparing values.
        abs_tol (float): Absolute tolerance used for comparing values.

    Returns:
        bool: True if a is less than b given the tolerance criteria, otherwise False.
    """
    if use_tol and is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return False
    return a < b


@register_jitted(cache=True)
def is_close_or_less_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Determine whether the first float value is approximately equal to or less than the second value.

    Args:
        a (float): First float value.
        b (float): Second float value.
        use_tol (bool): Flag to enable tolerance-based comparison.
        rel_tol (float): Relative tolerance used for comparing values.
        abs_tol (float): Absolute tolerance used for comparing values.

    Returns:
        bool: True if a is approximately equal to or less than b considering tolerance, otherwise False.
    """
    if use_tol and is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return a < b


@register_jitted(cache=True)
def is_close_or_greater_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Determine whether the first float value is approximately equal to or greater than the second value.

    Args:
        a (float): First float value.
        b (float): Second float value.
        use_tol (bool): Flag to enable tolerance-based comparison.
        rel_tol (float): Relative tolerance used for comparing values.
        abs_tol (float): Absolute tolerance used for comparing values.

    Returns:
        bool: True if a is approximately equal to or greater than b considering tolerance, otherwise False.
    """
    if use_tol and is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return a > b


@register_jitted(cache=True)
def is_greater_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Determine whether the first float value is approximately greater than the second value.

    Args:
        a (float): First float value.
        b (float): Second float value.
        use_tol (bool): Flag to enable tolerance-based comparison.
        rel_tol (float): Relative tolerance used for comparing values.
        abs_tol (float): Absolute tolerance used for comparing values.

    Returns:
        bool: True if a is greater than b given the tolerance criteria, otherwise False.
    """
    if use_tol and is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return False
    return a > b


@register_jitted(cache=True)
def is_addition_zero_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> bool:
    """Determine whether the sum of two float values is approximately zero, considering tolerance settings.

    Args:
        a (float): First operand.
        b (float): Second operand.
        use_tol (bool): Flag to enable tolerance-based comparison.
        rel_tol (float): Relative tolerance used for comparing values.
        abs_tol (float): Absolute tolerance used for comparing values.

    Returns:
        bool: True if the sum of a and b is approximately zero according to the tolerance settings, otherwise False.
    """
    if use_tol:
        if np.sign(a) != np.sign(b):
            return is_close_nb(abs(a), abs(b), rel_tol=rel_tol, abs_tol=abs_tol)
        return is_close_nb(a + b, 0.0, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == -b


@register_jitted(cache=True)
def add_nb(
    a: float,
    b: float,
    use_tol: bool = _use_tol,
    rel_tol: float = _rel_tol,
    abs_tol: float = _abs_tol,
) -> float:
    """Add two float values.

    Args:
        a (float): First addend.
        b (float): Second addend.
        use_tol (bool): Flag to enable tolerance-based comparison.
        rel_tol (float): Relative tolerance used for comparing values.
        abs_tol (float): Absolute tolerance used for comparing values.

    Returns:
        float: Sum of a and b, or 0.0 if the sum is approximately zero based on the tolerance settings.
    """
    if use_tol and is_addition_zero_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return 0.0
    return a + b


@register_jitted(cache=True)
def round_nb(a: float, use_round: bool = _use_round, decimals: int = _decimals) -> float:
    """Round a float value to a specified number of decimals.

    Args:
        a (float): Float value to round.
        use_round (bool): Flag that specifies whether rounding should be applied.
        decimals (int): Number of decimals to round to.

    Returns:
        float: Rounded float value if rounding is enabled; otherwise, the original value.
    """
    if use_round:
        return round(a, decimals)
    return a
