# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `HURST` class for calculating the rolling Hurst exponent."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.enums import HurstMethod
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "HURST",
]

__pdoc__ = {}

HURST = IndicatorFactory(
    class_name="HURST",
    module_name=__name__,
    input_names=["close"],
    param_names=[
        "window",
        "method",
        "max_lag",
        "min_log",
        "max_log",
        "log_step",
        "min_chunk",
        "max_chunk",
        "num_chunks",
    ],
    output_names=["hurst"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        hurst=dict(
            doc="Hurst exponent series.",
        ),
    ),
).with_apply_func(
    nb.rolling_hurst_nb,
    kwargs_as_args=["minp", "stabilize"],
    param_settings=dict(
        window=dict(
            doc="Window size.",
        ),
        method=dict(
            dtype=HurstMethod,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
            doc="Hurst exponent computation method (see `vectorbtpro.indicators.enums.HurstMethod`)",
        ),
        max_lag=dict(
            doc="Maximum lag parameter for the standard computation.",
        ),
        min_log=dict(
            doc="Minimum logarithmic scale for the LogRS method.",
        ),
        max_log=dict(
            doc="Maximum logarithmic scale for the LogRS method.",
        ),
        log_step=dict(
            doc="Increment on the logarithmic scale for the LogRS method.",
        ),
        min_chunk=dict(
            doc="Minimum chunk size for RS and DMA methods.",
        ),
        max_chunk=dict(
            doc="Maximum chunk size for RS and DMA methods.",
        ),
        num_chunks=dict(
            doc="Number of chunks for RS and DMA methods.",
        ),
    ),
    window=200,
    method="standard",
    max_lag=20,
    min_log=1,
    max_log=2,
    log_step=0.25,
    min_chunk=8,
    max_chunk=100,
    num_chunks=5,
    minp=None,
    stabilize=False,
)


class _HURST(HURST):
    """Class representing the moving Hurst exponent indicator.

    This indicator measures the long-term memory of a time series.

    See:
        * `HURST.run` for the main entry point.
        * `vectorbtpro.indicators.nb.rolling_hurst_nb` for the underlying implementation.
        * https://de.wikipedia.org/wiki/Hurst-Exponent for the definition of the Hurst exponent.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        hurst_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the HURST traces.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            hurst_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `HURST.hurst`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure updated with the Hurst exponent plot.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> ohlcv = vbt.YFData.pull(
            ...     "BTC-USD",
            ...     start="2020-01-01",
            ...     end="2024-01-01"
            ... ).get()
            >>> vbt.HURST.run(ohlcv["Close"]).plot().show()
            ```

            ![](/assets/images/api/HURST.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/HURST.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if hurst_trace_kwargs is None:
            hurst_trace_kwargs = {}
        hurst_trace_kwargs = merge_dicts(
            dict(name="HURST", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            hurst_trace_kwargs,
        )

        fig = self_col.hurst.vbt.lineplot(
            trace_kwargs=hurst_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig


HURST.clone_docstring(_HURST)
HURST.clone_method(_HURST.plot)
