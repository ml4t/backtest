# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `FMEAN` generator class for the future mean."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FMEAN",
]

__pdoc__ = {}

FMEAN = IndicatorFactory(
    class_name="FMEAN",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype", "wait"],
    output_names=["fmean"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        fmean=dict(
            doc="Future mean series.",
        ),
    ),
).with_apply_func(
    nb.future_mean_nb,
    kwargs_as_args=["minp", "adjust"],
    param_settings=dict(
        window=dict(
            doc="Window size.",
        ),
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
            doc="Weighting type (see `vectorbtpro.generic.enums.WType`).",
        ),
        wait=dict(
            doc="Number of periods to wait before calculating the future mean.",
        ),
    ),
    window=14,
    wtype="simple",
    wait=1,
    minp=None,
    adjust=False,
)


class _FMEAN(FMEAN):
    """Class representing the look-ahead future mean generator.

    See:
        * `FMEAN.run` for the main entry point.
        * `vectorbtpro.labels.nb.future_mean_nb` for the underlying implementation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        fmean_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the `FMEAN.fmean` indicator against the `FMEAN.close` price.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_close (bool): Whether to plot the close price.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `FMEAN.close`.
            fmean_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `FMEAN.fmean`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure object with the plotted indicators.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.FMEAN.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FMEAN.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FMEAN.dark.svg#only-dark){: .iimg loading=lazy }
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
        if fmean_trace_kwargs is None:
            fmean_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        fmean_trace_kwargs = merge_dicts(
            dict(name="Future mean", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fmean_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.fmean.vbt.lineplot(
            trace_kwargs=fmean_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


FMEAN.clone_docstring(_FMEAN)
FMEAN.clone_method(_FMEAN.plot)
