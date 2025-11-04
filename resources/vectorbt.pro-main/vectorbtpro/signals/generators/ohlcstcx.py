# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `OHLCSTCX` class for generating stop signals based on OHLC data."""

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators.ohlcstx import (
    _bind_ohlcstx_plot,
    ohlcstx_config,
    ohlcstx_func_config,
)

__all__ = [
    "OHLCSTCX",
]

__pdoc__ = {}

OHLCSTCX = SignalFactory(
    **ohlcstx_config.merge_with(
        dict(
            class_name="OHLCSTCX",
            module_name=__name__,
            short_name="ohlcstcx",
            mode="chain",
        )
    ),
).with_place_func(
    **ohlcstx_func_config,
)


class _OHLCSTCX(OHLCSTCX):
    """Class representing a chained exit signal generator based on OHLC data and stop values.

    Generates a chain of `new_entries` and `exits` derived from input `entries`.

    See:
        * `OHLCSTCX.run` for the main entry point.
        * `vectorbtpro.signals.nb.ohlc_stop_place_nb` for details on the exit placement.
        * `vectorbtpro.signals.generators.ohlcstx.OHLCSTX` for parameter details.
    """

    plot = _bind_ohlcstx_plot(OHLCSTCX, "new_entries")


OHLCSTCX.clone_docstring(_OHLCSTCX)
OHLCSTCX.clone_method(_OHLCSTCX.plot)
