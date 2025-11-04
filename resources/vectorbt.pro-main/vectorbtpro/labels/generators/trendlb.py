# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `TRENDLB` generator class for trend labels."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.labels.enums import TrendLabelMode
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "TRENDLB",
]

__pdoc__ = {}

TRENDLB = IndicatorFactory(
    class_name="TRENDLB",
    module_name=__name__,
    input_names=["high", "low"],
    param_names=["up_th", "down_th", "mode"],
    output_names=["labels"],
    attr_settings=dict(
        high=dict(
            doc="High price series.",
        ),
        low=dict(
            doc="Low price series.",
        ),
        labels=dict(
            doc="Trend labels series.",
        ),
    ),
).with_apply_func(
    nb.trend_labels_nb,
    param_settings=dict(
        up_th=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Upper threshold for trend labels, as a scalar or an array.",
            ),
        ),
        down_th=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Lower threshold for trend labels, as a scalar or an array.",
            ),
        ),
        mode=dict(
            dtype=TrendLabelMode,
            post_index_func=lambda index: index.str.lower(),
            doc="Trend label mode (see `vectorbtpro.labels.enums.TrendLabelMode`).",
        ),
    ),
    mode="binary",
)


class _TRENDLB(TRENDLB):
    """Class representing the look-ahead trend label generator.

    See:
        * `TRENDLB.run` for the main entry point.
        * `vectorbtpro.labels.nb.trend_labels_nb` for the underlying implementation.
    """

    def plot(self, column: tp.Optional[tp.Column] = None, **kwargs) -> tp.BaseFigure:
        """Plot the median of `TRENDLB.high` and `TRENDLB.low` and overlay it with a heatmap of `TRENDLB.labels`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Returns:
            BaseFigure: Figure object displaying the overlay of the median and heatmap.

        Examples:
            ```pycon
            >>> vbt.TRENDLB.run(ohlcv['High'], ohlcv['Low'], up_th=0.2, down_th=0.2).plot().show()
            ```

            ![](/assets/images/api/TRENDLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/TRENDLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        median = (self_col.high + self_col.low) / 2
        return median.rename("Median").vbt.overlay_with_heatmap(
            self_col.labels.rename("Labels"), **kwargs
        )


TRENDLB.clone_docstring(_TRENDLB)
TRENDLB.clone_method(_TRENDLB.plot)
