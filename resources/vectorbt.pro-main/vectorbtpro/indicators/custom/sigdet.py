# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `SIGDET` indicator class for robust peak detection using z-scores."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "SIGDET",
]

__pdoc__ = {}

SIGDET = IndicatorFactory(
    class_name="SIGDET",
    module_name=__name__,
    short_name="sigdet",
    input_names=["close"],
    param_names=[
        "lag",
        "factor",
        "influence",
        "up_factor",
        "down_factor",
        "mean_influence",
        "std_influence",
    ],
    output_names=["signal", "upper_band", "lower_band"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        signal=dict(
            doc="Signal series.",
        ),
        upper_band=dict(
            doc="Upper band series.",
        ),
        lower_band=dict(
            doc="Lower band series.",
        ),
    ),
).with_apply_func(
    nb.signal_detection_nb,
    param_settings=dict(
        lag=dict(
            doc="Lag period for the z-score calculation.",
        ),
        factor=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Factor to multiply the z-score by, as a scalar or an array.",
            ),
        ),
        influence=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Influence of the z-score on the signal, as a scalar or an array.",
            ),
        ),
        up_factor=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Factor to multiply the z-score by for upward signals, as a scalar or an array.",
            ),
        ),
        down_factor=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Factor to multiply the z-score by for downward signals, as a scalar or an array.",
            ),
        ),
        mean_influence=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Influence of the mean on the signal, as a scalar or an array.",
            ),
        ),
        std_influence=merge_dicts(
            flex_elem_param_config,
            dict(
                doc="Influence of the standard deviation on the signal, as a scalar or an array.",
            ),
        ),
    ),
    lag=14,
    factor=1.0,
    influence=1.0,
    up_factor=None,
    down_factor=None,
    mean_influence=None,
    std_influence=None,
)


class _SIGDET(SIGDET):
    """Class representing the robust peak detection algorithm using z-scores.

    For additional details, see [this StackOverflow answer]().

    See:
        * `SIGDET.run` for the main entry point.
        * `vectorbtpro.indicators.nb.signal_detection_nb` for the underlying implementation.
        * https://stackoverflow.com/a/22640362 for the original algorithm.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        signal_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the signal from `SIGDET.signal`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            signal_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `SIGDET.signal`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure object containing the plotted signal.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.SIGDET.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/SIGDET.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/SIGDET.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        signal_trace_kwargs = merge_dicts(
            dict(
                name="Signal",
                line=dict(color=plotting_cfg["color_schema"]["lightblue"], shape="hv"),
            ),
            signal_trace_kwargs,
        )
        fig = self_col.signal.vbt.lineplot(
            trace_kwargs=signal_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig

    def plot_bands(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        upper_band_trace_kwargs: tp.KwargsLike = None,
        lower_band_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the upper and lower bands from `SIGDET.upper_band` and `SIGDET.lower_band`
        against the close values from `SIGDET.close`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_close (bool): Whether to plot the close price.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `SIGDET.close`.
            upper_band_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `SIGDET.upper_band`.
            lower_band_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `SIGDET.lower_band`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure object containing the plotted bands (and close line if requested).

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.SIGDET.run(ohlcv['Close']).plot_bands().show()
            ```

            ![](/assets/images/api/SIGDET_plot_bands.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/SIGDET_plot_bands.dark.svg#only-dark){: .iimg loading=lazy }
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
        if upper_band_trace_kwargs is None:
            upper_band_trace_kwargs = {}
        if lower_band_trace_kwargs is None:
            lower_band_trace_kwargs = {}
        lower_band_trace_kwargs = merge_dicts(
            dict(
                name="Lower band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
            ),
            lower_band_trace_kwargs,
        )
        upper_band_trace_kwargs = merge_dicts(
            dict(
                name="Upper band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.2)",
            ),
            upper_band_trace_kwargs,
        )  # default kwargs
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )

        fig = self_col.lower_band.vbt.lineplot(
            trace_kwargs=lower_band_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.upper_band.vbt.lineplot(
            trace_kwargs=upper_band_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        return fig


SIGDET.clone_docstring(_SIGDET)
SIGDET.clone_method(_SIGDET.plot)
SIGDET.clone_method(_SIGDET.plot_bands)
