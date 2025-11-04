# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `BOLB` generator class for breakout labels."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "BOLB",
]

__pdoc__ = {}

BOLB = IndicatorFactory(
    class_name="BOLB",
    module_name=__name__,
    input_names=["high", "low"],
    param_names=["window", "up_th", "down_th", "wait"],
    output_names=["labels"],
    attr_settings=dict(
        high=dict(
            doc="High price series.",
        ),
        low=dict(
            doc="Low price series.",
        ),
        labels=dict(
            doc="Breakout labels.",
        ),
    ),
).with_apply_func(
    nb.breakout_labels_nb,
    param_settings=dict(
        window=dict(
            doc="Window size.",
        ),
        up_th=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Upper threshold for breakout labels, as a scalar or an array.",
            ),
        ),
        down_th=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Lower threshold for breakout labels, as a scalar or an array.",
            ),
        ),
        wait=dict(
            doc="Number of periods to wait before calculating the breakout labels.",
        ),
    ),
    window=14,
    up_th=np.inf,
    down_th=np.inf,
    wait=1,
)


class _BOLB(BOLB):
    """Class representing the look-ahead breakout label generator.

    See:
        * `BOLB.run` for the main entry point.
        * `vectorbtpro.labels.nb.breakout_labels_nb` for the underlying implementation.
    """

    def plot(self, column: tp.Optional[tp.Column] = None, **kwargs) -> tp.BaseFigure:
        """Plot the median of `BOLB.high` and `BOLB.low` and overlay it with a heatmap of `BOLB.labels`.

        Args:
            column (Label): Column to select for plotting.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Returns:
            BaseFigure: Figure displaying the plotted median with the heatmap overlay.

        Examples:
            ```pycon
            >>> vbt.BOLB.run(ohlcv['High'], ohlcv['Low'], up_th=0.2, down_th=0.2).plot().show()
            ```

            ![](/assets/images/api/BOLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/BOLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        median = (self_col.high + self_col.low) / 2
        return median.rename("Median").vbt.overlay_with_heatmap(
            self_col.labels.rename("Labels"), **kwargs
        )


BOLB.clone_docstring(_BOLB)
BOLB.clone_method(_BOLB.plot)
