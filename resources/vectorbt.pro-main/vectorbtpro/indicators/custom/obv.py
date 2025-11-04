# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `OBV` class for calculating the on-balance volume indicator."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "OBV",
]

__pdoc__ = {}

OBV = IndicatorFactory(
    class_name="OBV",
    module_name=__name__,
    short_name="obv",
    input_names=["close", "volume"],
    param_names=[],
    output_names=["obv"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        volume=dict(
            doc="Volume series.",
        ),
        obv=dict(
            doc="On-Balance Volume (OBV) series.",
        ),
    ),
).with_custom_func(nb.obv_nb)


class _OBV(OBV):
    """Class representing the on-balance volume (OBV) indicator.

    Calculates OBV by accumulating total volume based on price movements,
    thereby relating price and volume in the stock market.

    See [On-Balance Volume (OBV)](https://www.investopedia.com/terms/o/onbalancevolume.asp).

    See:
        * `OBV.run` for the main entry point.
        * `vectorbtpro.indicators.nb.obv_nb` for the underlying implementation.
        * https://www.investopedia.com/terms/o/onbalancevolume.asp for the definition of OBV.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        obv_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `OBV.obv`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            obv_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `OBV.obv`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure displaying the plotted OBV data.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.OBV.run(ohlcv['Close'], ohlcv['Volume']).plot().show()
            ```

            ![](/assets/images/api/OBV.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/OBV.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if obv_trace_kwargs is None:
            obv_trace_kwargs = {}
        obv_trace_kwargs = merge_dicts(
            dict(name="OBV", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            obv_trace_kwargs,
        )

        fig = self_col.obv.vbt.lineplot(
            trace_kwargs=obv_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


OBV.clone_docstring(_OBV)
OBV.clone_method(_OBV.plot)
