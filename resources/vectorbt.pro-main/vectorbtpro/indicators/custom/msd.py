# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `MSD` class for calculating the rolling squared deviation."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "MSD",
]

__pdoc__ = {}

MSD = IndicatorFactory(
    class_name="MSD",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["msd"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        msd=dict(
            doc="Moving Standard Deviation (MSD) series.",
        ),
    ),
).with_apply_func(
    nb.msd_nb,
    kwargs_as_args=["minp", "adjust", "ddof"],
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
    wtype="simple",
    minp=None,
    adjust=False,
    ddof=0,
)


class _MSD(MSD):
    """Class representing the Moving Standard Deviation (MSD) indicator.

    This class represents the moving standard deviation (MSD) indicator that measures
    the magnitude of recent price movements to assess asset volatility.

    See:
        * `MSD.run` for the main entry point.
        * `vectorbtpro.indicators.nb.msd_nb` for the underlying implementation.
        * https://en.wikipedia.org/wiki/Standard_deviation for the definition of standard deviation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        msd_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the `MSD.msd` indicator.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            msd_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `MSD.msd`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure instance containing the plotted MSD indicator.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.MSD.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MSD.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/MSD.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if msd_trace_kwargs is None:
            msd_trace_kwargs = {}
        msd_trace_kwargs = merge_dicts(
            dict(name="MSD", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            msd_trace_kwargs,
        )

        fig = self_col.msd.vbt.lineplot(
            trace_kwargs=msd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig


MSD.clone_docstring(_MSD)
MSD.clone_method(_MSD.plot)
