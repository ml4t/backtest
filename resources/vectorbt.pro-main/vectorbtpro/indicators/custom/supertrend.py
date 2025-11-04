# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `SUPERTREND` class for calculating the Supertrend indicator."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "SUPERTREND",
]

__pdoc__ = {}

SUPERTREND = IndicatorFactory(
    class_name="SUPERTREND",
    module_name=__name__,
    short_name="supertrend",
    input_names=["high", "low", "close"],
    param_names=["period", "multiplier"],
    output_names=["trend", "direction", "long", "short"],
    attr_settings=dict(
        high=dict(
            doc="High price series.",
        ),
        low=dict(
            doc="Low price series.",
        ),
        close=dict(
            doc="Close price series.",
        ),
        trend=dict(
            doc="Supertrend trend series.",
        ),
        direction=dict(
            doc="Supertrend direction series.",
        ),
        long=dict(
            doc="Supertrend long signal series.",
        ),
        short=dict(
            doc="Supertrend short signal series.",
        ),
    ),
).with_apply_func(
    nb.supertrend_nb,
    param_settings=dict(
        period=dict(
            doc="Period for the Supertrend calculation.",
        ),
        multiplier=dict(
            doc="Multiplier for the ATR calculation.",
        ),
    ),
    period=7,
    multiplier=3,
)


class _SUPERTREND(SUPERTREND):
    """Class representing the supertrend indicator.

    The supertrend indicator is a trend-following overlay that appears directly on price charts,
    providing clear buy and sell signals based on the underlying asset's price action and volatility.

    See:
        * `SUPERTREND.run` for the main entry point.
        * `vectorbtpro.indicators.nb.supertrend_nb` for the underlying implementation.
        * https://www.investopedia.com/supertrend-indicator-7976167 for the definition of the supertrend indicator.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        superl_trace_kwargs: tp.KwargsLike = None,
        supers_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the long and short signals of the Supertrend indicator, and optionally the close prices.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_close (bool): Whether to plot the close price.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `SUPERTREND.close`.
            superl_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `SUPERTREND.long`.
            supers_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `SUPERTREND.short`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure containing the plotted indicator traces.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.SUPERTREND.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/SUPERTREND.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/SUPERTREND.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if superl_trace_kwargs is None:
            superl_trace_kwargs = {}
        if supers_trace_kwargs is None:
            supers_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        superl_trace_kwargs = merge_dicts(
            dict(name="Long", line=dict(color=plotting_cfg["color_schema"]["green"])),
            superl_trace_kwargs,
        )
        supers_trace_kwargs = merge_dicts(
            dict(name="Short", line=dict(color=plotting_cfg["color_schema"]["red"])),
            supers_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.long.vbt.lineplot(
            trace_kwargs=superl_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.short.vbt.lineplot(
            trace_kwargs=supers_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


SUPERTREND.clone_docstring(_SUPERTREND)
SUPERTREND.clone_method(_SUPERTREND.plot)
