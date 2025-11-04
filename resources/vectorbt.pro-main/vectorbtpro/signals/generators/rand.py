# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `RAND` class for generating random entry signals."""

from vectorbtpro.indicators.configs import flex_col_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_place_nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "RAND",
]

__pdoc__ = {}

RAND = SignalFactory(
    class_name="RAND",
    module_name=__name__,
    short_name="rand",
    mode="entries",
    param_names=["n"],
).with_place_func(
    entry_place_func_nb=rand_place_nb,
    entry_settings=dict(
        pass_params=["n"],
    ),
    param_settings=dict(
        n=merge_dicts(
            flex_col_param_config,
            dict(
                doc="Number of entries to place, as a scalar or an array (per column).",
            ),
        ),
    ),
    seed=None,
)


class _RAND(RAND):
    """Class representing a random entry signal generator based on probabilities.

    See:
        * `RAND.run` for the main entry point.
        * `vectorbtpro.signals.nb.rand_place_nb` for details on the entry placement.

    !!! hint
        The parameter `n` may be provided as either a single value (per frame) or a NumPy array (per column).

        To create multiple combinations, specify `n` as a list.

    Examples:
        Test three different entry count values:

        ```pycon
        >>> from vectorbtpro import *

        >>> rand = vbt.RAND.run(input_shape=(6,), n=[1, 2, 3], seed=42)

        >>> rand.entries
        rand_n      1      2      3
        0        True   True   True
        1       False  False   True
        2       False  False  False
        3       False   True  False
        4       False  False   True
        5       False  False  False
        ```

        Entry count can also be set per column:

        ```pycon
        >>> rand = vbt.RAND.run(input_shape=(8, 2), n=[np.array([1, 2]), 3], seed=42)

        >>> rand.entries
        rand_n      1      2      3      3
                    0      1      0      1
        0       False  False   True  False
        1        True  False  False  False
        2       False  False  False   True
        3       False   True   True  False
        4       False  False  False  False
        5       False  False  False   True
        6       False  False   True  False
        7       False   True  False   True
        ```
    """

    pass


RAND.clone_docstring(_RAND)
