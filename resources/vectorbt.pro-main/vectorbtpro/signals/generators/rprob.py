# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `RPROB` class for generating random entry signals based on probabilities."""

from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_by_prob_place_nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "RPROB",
]

__pdoc__ = {}

RPROB = SignalFactory(
    class_name="RPROB",
    module_name=__name__,
    short_name="rprob",
    mode="entries",
    param_names=["prob"],
).with_place_func(
    entry_place_func_nb=rand_by_prob_place_nb,
    entry_settings=dict(
        pass_params=["prob"],
        pass_kwargs=["pick_first"],
    ),
    param_settings=dict(
        prob=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Probability of placing an entry, as a scalar or an array.",
            ),
        ),
    ),
    seed=None,
)


class _RPROB(RPROB):
    """Class representing a random entry signal generator based on probabilities.

    See:
        * `RPROB.run` for the main entry point.
        * `vectorbtpro.signals.nb.rand_by_prob_place_nb` for details on the entry placement.

    !!! hint
        All parameters can be provided as a single value (per frame) or as a NumPy array
        (per row, column, or element).

        To generate multiple combinations, pass them as lists.

    Examples:
        Generate three columns with different entry probabilities:

        ```pycon
        >>> from vectorbtpro import *

        >>> rprob = vbt.RPROB.run(input_shape=(5,), prob=[0., 0.5, 1.], seed=42)

        >>> rprob.entries
        rprob_prob    0.0    0.5   1.0
        0           False   True  True
        1           False   True  True
        2           False  False  True
        3           False  False  True
        4           False  False  True
        ```

        Probability can also be set per row, column, or element:

        ```pycon
        >>> rprob = vbt.RPROB.run(input_shape=(5,), prob=np.array([0., 0., 1., 1., 1.]), seed=42)

        >>> rprob.entries
        0    False
        1    False
        2     True
        3     True
        4     True
        Name: array_0, dtype: bool
        ```
    """

    pass


RPROB.clone_docstring(_RPROB)
