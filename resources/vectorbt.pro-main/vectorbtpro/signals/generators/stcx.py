# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `STCX` class for generating stop signals."""

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators.stx import stx_config, stx_func_config

__all__ = [
    "STCX",
]

__pdoc__ = {}

STCX = SignalFactory(
    **stx_config.merge_with(
        dict(
            class_name="STCX",
            module_name=__name__,
            short_name="stcx",
            mode="chain",
        )
    )
).with_place_func(**stx_func_config)


class _STCX(STCX):
    """Class representing a chained exit signal generator based on stop values.

    Generates a chain of `new_entries` and `exits` derived from input `entries`.

    See:
        * `STCX.run` for the main entry point.
        * `vectorbtpro.signals.nb.stop_place_nb` for details on the exit placement.
        * `vectorbtpro.signals.generators.stx.STX` for parameter details.
    """

    pass


STCX.clone_docstring(_STCX)
