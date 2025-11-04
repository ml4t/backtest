# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `ADX` class for calculating the Average Directional Index indicator."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "ADX",
]

__pdoc__ = {}

ADX = IndicatorFactory(
    class_name="ADX",
    module_name=__name__,
    input_names=["high", "low", "close"],
    param_names=["window", "wtype"],
    output_names=["plus_di", "minus_di", "dx", "adx"],
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
        plus_di=dict(
            doc="Positive Directional Indicator (DI) series.",
        ),
        minus_di=dict(
            doc="Negative Directional Indicator (DI) series.",
        ),
        dx=dict(
            doc="Directional Movement Index (DX) series.",
        ),
        adx=dict(
            doc="Average Directional Index (ADX) series.",
        ),
    ),
).with_apply_func(
    nb.adx_nb,
    kwargs_as_args=["minp", "adjust"],
    param_settings=dict(
        window=dict(
            doc="Window size.",
        ),
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
            doc="Weighting type (see `vectorbtpro.generic.enums.WType`).",
        ),
    ),
    window=14,
    wtype="wilder",
    minp=None,
    adjust=False,
)


class _ADX(ADX):
    """Class representing the Average Directional Movement Index (ADX) indicator.

    Measures trend strength to assist traders in assessing market conditions.

    See:
        * `ADX.run` for the main entry point.
        * `vectorbtpro.indicators.nb.adx_nb` for the underlying implementation.
        * https://www.investopedia.com/terms/a/adx.asp for the definition of ADX.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        plus_di_trace_kwargs: tp.KwargsLike = None,
        minus_di_trace_kwargs: tp.KwargsLike = None,
        adx_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot ADX traces including `ADX.plus_di`, `ADX.minus_di`, and `ADX.adx`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plus_di_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `ADX.plus_di`.
            minus_di_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `ADX.minus_di`.
            adx_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `ADX.adx`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure containing the plotted ADX traces.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.ADX.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/ADX.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/ADX.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if plus_di_trace_kwargs is None:
            plus_di_trace_kwargs = {}
        if minus_di_trace_kwargs is None:
            minus_di_trace_kwargs = {}
        if adx_trace_kwargs is None:
            adx_trace_kwargs = {}
        plus_di_trace_kwargs = merge_dicts(
            dict(name="+DI", line=dict(color=plotting_cfg["color_schema"]["green"], dash="dot")),
            plus_di_trace_kwargs,
        )
        minus_di_trace_kwargs = merge_dicts(
            dict(name="-DI", line=dict(color=plotting_cfg["color_schema"]["red"], dash="dot")),
            minus_di_trace_kwargs,
        )
        adx_trace_kwargs = merge_dicts(
            dict(name="ADX", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            adx_trace_kwargs,
        )

        fig = self_col.plus_di.vbt.lineplot(
            trace_kwargs=plus_di_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.minus_di.vbt.lineplot(
            trace_kwargs=minus_di_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.adx.vbt.lineplot(
            trace_kwargs=adx_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


ADX.clone_docstring(_ADX)
ADX.clone_method(_ADX.plot)
