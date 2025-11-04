# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `RANDNX` class for generating random entry and exit signals."""

from vectorbtpro.indicators.configs import flex_col_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_enex_apply_nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "RANDNX",
]

__pdoc__ = {}

RANDNX = SignalFactory(
    class_name="RANDNX",
    module_name=__name__,
    short_name="randnx",
    mode="both",
    param_names=["n"],
).with_apply_func(
    rand_enex_apply_nb,
    require_input_shape=True,
    param_settings=dict(
        n=merge_dicts(
            flex_col_param_config,
            dict(
                doc="Number of entries to place, as a scalar or an array (per column).",
            ),
        ),
    ),
    kwargs_as_args=["entry_wait", "exit_wait"],
    entry_wait=1,
    exit_wait=1,
    seed=None,
)


class _RANDNX(RANDNX):
    """Class representing a random entry and exit signal generator based on probabilities.

    See:
        * `RANDNX.run` for the main entry point.
        * `vectorbtpro.signals.nb.rand_enex_apply_nb` for details on the entry and exit placement.
        * `vectorbtpro.signals.generators.rand.RAND` for parameter details.

    Examples:
        Test three different entry and exit signal configurations:

        ```pycon
        >>> from vectorbtpro import *

        >>> randnx = vbt.RANDNX.run(
        ...     input_shape=(6,),
        ...     n=[1, 2, 3],
        ...     seed=42)

        >>> randnx.entries
        randnx_n      1      2      3
        0          True   True   True
        1         False  False  False
        2         False   True   True
        3         False  False  False
        4         False  False   True
        5         False  False  False

        >>> randnx.exits
        randnx_n      1      2      3
        0         False  False  False
        1          True   True   True
        2         False  False  False
        3         False   True   True
        4         False  False  False
        5         False  False   True
        ```
    """

    pass


RANDNX.clone_docstring(_RANDNX)
