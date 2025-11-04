# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `FMIN` generator class for the future minimum."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FMIN",
]

__pdoc__ = {}

FMIN = IndicatorFactory(
    class_name="FMIN",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wait"],
    output_names=["fmin"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        fmin=dict(
            doc="Future minimum series.",
        ),
    ),
).with_apply_func(
    nb.future_min_nb,
    param_settings=dict(
        window=dict(
            doc="Window size.",
        ),
        wait=dict(
            doc="Number of periods to wait before calculating the future minimum.",
        ),
    ),
    window=14,
    wait=1,
)


class _FMIN(FMIN):
    """Class representing the look-ahead future minimum generator.

    See:
        * `FMIN.run` for the main entry point.
        * `vectorbtpro.labels.nb.future_min_nb` for the underlying implementation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        fmin_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `FMIN.fmin` against `FMIN.close`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_close (bool): Whether to plot the close price.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `FMIN.close`.
            fmin_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `FMIN.fmin`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure object with the plotted traces.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.FMIN.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FMIN.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FMIN.dark.svg#only-dark){: .iimg loading=lazy }
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
        if fmin_trace_kwargs is None:
            fmin_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        fmin_trace_kwargs = merge_dicts(
            dict(name="Future min", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fmin_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.fmin.vbt.lineplot(
            trace_kwargs=fmin_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


FMIN.clone_docstring(_FMIN)
FMIN.clone_method(_FMIN.plot)
