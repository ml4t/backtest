# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `RANDX` class for generating random exit signals."""

import numpy as np

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_place_nb

__all__ = [
    "RANDX",
]

__pdoc__ = {}

RANDX = SignalFactory(
    class_name="RANDX",
    module_name=__name__,
    short_name="randx",
    mode="exits",
).with_place_func(
    exit_place_func_nb=rand_place_nb,
    exit_settings=dict(
        pass_kwargs=dict(n=np.array([1])),
    ),
    seed=None,
)


class _RANDX(RANDX):
    """Class representing a random exit signal generator based on probabilities.

    See:
        * `RANDX.run` for the main entry point.
        * `vectorbtpro.signals.nb.rand_place_nb` for details on the exit placement.
        * `vectorbtpro.signals.generators.rand.RAND` for parameter notes.

    Examples:
        Generate an exit signal for each entry:

        ```pycon
        >>> from vectorbtpro import *

        >>> entries = pd.Series([True, False, False, True, False, False])
        >>> randx = vbt.RANDX.run(entries, seed=42)

        >>> randx.exits
        0    False
        1    False
        2     True
        3    False
        4     True
        5    False
        dtype: bool
        ```
    """

    pass


RANDX.clone_docstring(_RANDX)
