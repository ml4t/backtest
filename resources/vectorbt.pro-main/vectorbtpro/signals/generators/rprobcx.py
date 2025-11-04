# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `RPROBCX` class for generating random exit signals based on probabilities."""

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators.rprobx import rprobx_config, rprobx_func_config

__all__ = [
    "RPROBCX",
]

__pdoc__ = {}

RPROBCX = SignalFactory(
    **rprobx_config.merge_with(
        dict(
            class_name="RPROBCX",
            module_name=__name__,
            short_name="rprobcx",
            mode="chain",
        )
    ),
).with_place_func(**rprobx_func_config)


class _RPROBCX(RPROBCX):
    """Class representing a random exit signal generator based on probabilities.

    Generates a chain of `new_entries` and `exits` derived from input `entries`.

    See:
        * `RPROBCX.run` for the main entry point.
        * `vectorbtpro.signals.nb.rand_by_prob_place_nb` for details on the exit placement.
        * `vectorbtpro.signals.generators.rprob.RPROB` for parameter arguments.
    """

    pass


RPROBCX.clone_docstring(_RPROBCX)
