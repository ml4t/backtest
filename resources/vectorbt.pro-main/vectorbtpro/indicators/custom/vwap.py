# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `VWAP` class for calculating the Volume-Weighted Average Price indicator."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.template import RepFunc

__all__ = [
    "VWAP",
]

__pdoc__ = {}


def substitute_anchor(wrapper: ArrayWrapper, anchor: tp.Optional[tp.FrequencyLike]) -> tp.Array1d:
    """Substitute the reset frequency with group lengths.

    Computes group lengths based on the provided `anchor`. If `anchor` is None, returns an array
    with the number of rows in the wrapper; otherwise, calculates group lengths using the
    wrapper's index grouper.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        anchor (Optional[FrequencyLike]): Reset frequency for grouping.

            See `vectorbtpro.base.wrapping.ArrayWrapper.get_index_grouper`.

    Returns:
        Array1d: Array containing the group lengths.
    """
    if anchor is None:
        return np.array([wrapper.shape[0]])
    return wrapper.get_index_grouper(anchor).get_group_lens()


VWAP = IndicatorFactory(
    class_name="VWAP",
    module_name=__name__,
    short_name="vwap",
    input_names=["high", "low", "close", "volume"],
    param_names=["anchor"],
    output_names=["vwap"],
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
        volume=dict(
            doc="Volume series.",
        ),
        vwap=dict(
            doc="Volume-Weighted Average Price (VWAP) series.",
        ),
    ),
).with_apply_func(
    nb.vwap_nb,
    param_settings=dict(
        anchor=dict(
            template=RepFunc(substitute_anchor),
            doc="Reset frequency for grouping (see `substitute_anchor`).",
        ),
    ),
    anchor="D",
)


class _VWAP(VWAP):
    """Class representing the Volume-Weighted Average Price (VWAP) indicator.

    Calculates the volume-weighted average price commonly used in intraday charts.
    The calculation resets at the beginning of each trading session.

    The `anchor` parameter specifies the grouping for when the VWAP resets and can be any valid index grouper.

    See:
        * `VWAP.run` for the main entry point.
        * `vectorbtpro.indicators.nb.vwap_nb` for the underlying implementation.
        * https://www.investopedia.com/terms/v/vwap.asp for the definition of VWAP.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        vwap_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `VWAP.vwap` against `VWAP.close` values.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_close (bool): Whether to plot the close price.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `VWAP.close`.
            vwap_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `VWAP.vwap`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure containing the plotted traces.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.VWAP.run(
            ...    ohlcv['High'],
            ...    ohlcv['Low'],
            ...    ohlcv['Close'],
            ...    ohlcv['Volume'],
            ...    anchor="W"
            ... ).plot().show()
            ```

            ![](/assets/images/api/VWAP.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/VWAP.dark.svg#only-dark){: .iimg loading=lazy }
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
        if vwap_trace_kwargs is None:
            vwap_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        vwap_trace_kwargs = merge_dicts(
            dict(name="VWAP", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            vwap_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.vwap.vbt.lineplot(
            trace_kwargs=vwap_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


VWAP.clone_docstring(_VWAP)
VWAP.clone_method(_VWAP.plot)
