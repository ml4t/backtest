# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `PIVOTLB` generator class for pivot labels."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "PIVOTLB",
]

__pdoc__ = {}

PIVOTLB = IndicatorFactory(
    class_name="PIVOTLB",
    module_name=__name__,
    input_names=["high", "low"],
    param_names=["up_th", "down_th"],
    output_names=["labels"],
    attr_settings=dict(
        high=dict(
            doc="High price series.",
        ),
        low=dict(
            doc="Low price series.",
        ),
        labels=dict(
            doc="Labels generated based on the pivot thresholds (see `vectorbtpro.indicators.enums.Pivot`).",
        ),
    ),
).with_apply_func(
    nb.pivots_nb,
    param_settings=dict(
        up_th=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Upper threshold for the pivot label, as a scalar or an array.",
            ),
        ),
        down_th=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Lower threshold for the pivot label, as a scalar or an array.",
            ),
        ),
    ),
)


class _PIVOTLB(PIVOTLB):
    """Class representing the look-ahead pivot label generator.

    This class generates labels based on pivot thresholds applied to high and low values.

    See:
        * `PIVOTLB.run` for the main entry point.
        * `vectorbtpro.labels.nb.pivots_nb` for the underlying implementation.
    """

    def plot(self, column: tp.Optional[tp.Column] = None, **kwargs) -> tp.BaseFigure:
        """Plot the median of `PIVOTLB.high` and `PIVOTLB.low` with a heatmap overlay of `PIVOTLB.labels`.

        Calculates the median of the high and low values and overlays it with a heatmap
        representation of the generated labels.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Returns:
            BaseFigure: Figure object displaying the plot.

        Examples:
            ```pycon
            >>> vbt.PIVOTLB.run(ohlcv['High'], ohlcv['Low'], up_th=0.2, down_th=0.2).plot().show()
            ```

            ![](/assets/images/api/PIVOTLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PIVOTLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        median = (self_col.high + self_col.low) / 2
        return median.rename("Median").vbt.overlay_with_heatmap(
            self_col.labels.rename("Labels"), **kwargs
        )


PIVOTLB.clone_docstring(_PIVOTLB)
PIVOTLB.clone_method(_PIVOTLB.plot)
