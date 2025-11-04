# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `FMAX` generator class for the future maximum."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FMAX",
]

__pdoc__ = {}

FMAX = IndicatorFactory(
    class_name="FMAX",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wait"],
    output_names=["fmax"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        fmax=dict(
            doc="Future maximum series.",
        ),
    ),
).with_apply_func(
    nb.future_max_nb,
    param_settings=dict(
        window=dict(
            doc="Window size.",
        ),
        wait=dict(
            doc="Number of periods to wait before calculating the future maximum.",
        ),
    ),
    window=14,
    wait=1,
)


class _FMAX(FMAX):
    """Class representing the look-ahead future maximum generator.

    See:
        * `FMAX.run` for the main entry point.
        * `vectorbtpro.labels.nb.future_max_nb` for the underlying implementation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        fmax_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `FMAX.fmax` against `FMAX.close`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_close (bool): Whether to plot the close price.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `FMAX.close`.
            fmax_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `FMAX.fmax`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure with the indicator traces plotted.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.FMAX.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FMAX.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FMAX.dark.svg#only-dark){: .iimg loading=lazy }
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
        if fmax_trace_kwargs is None:
            fmax_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        fmax_trace_kwargs = merge_dicts(
            dict(name="Future max", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fmax_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.fmax.vbt.lineplot(
            trace_kwargs=fmax_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


FMAX.clone_docstring(_FMAX)
FMAX.clone_method(_FMAX.plot)
