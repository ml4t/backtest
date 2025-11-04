# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `FSTD` generator class for the future standard deviation."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FSTD",
]

__pdoc__ = {}

FSTD = IndicatorFactory(
    class_name="FSTD",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype", "wait"],
    output_names=["fstd"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        fstd=dict(
            doc="Future standard deviation series.",
        ),
    ),
).with_apply_func(
    nb.future_std_nb,
    kwargs_as_args=["minp", "adjust", "ddof"],
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
            doc="Number of periods to wait before calculating the future standard deviation.",
        ),
    ),
    window=14,
    wtype="simple",
    wait=1,
    minp=None,
    adjust=False,
    ddof=0,
)


class _FSTD(FSTD):
    """Class representing the look-ahead future standard deviation generator.

    See:
        * `FSTD.run` for the main entry point.
        * `vectorbtpro.labels.nb.future_std_nb` for the underlying implementation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        fstd_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the `FSTD.fstd` indicator.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            fstd_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `FSTD.fstd`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure with the `FSTD.fstd` indicator plotted.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.FSTD.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FSTD.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FSTD.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        if fstd_trace_kwargs is None:
            fstd_trace_kwargs = {}
        fstd_trace_kwargs = merge_dicts(
            dict(name="Future STD", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fstd_trace_kwargs,
        )
        fig = self_col.fstd.vbt.lineplot(
            trace_kwargs=fstd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        return fig


FSTD.clone_docstring(_FSTD)
FSTD.clone_method(_FSTD.plot)
