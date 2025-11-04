# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `PATSIM` indicator class for calculating rolling pattern similarity."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "PATSIM",
]

__pdoc__ = {}

PATSIM = IndicatorFactory(
    class_name="PATSIM",
    module_name=__name__,
    short_name="patsim",
    input_names=["close"],
    param_names=[
        "pattern",
        "window",
        "max_window",
        "row_select_prob",
        "window_select_prob",
        "interp_mode",
        "rescale_mode",
        "vmin",
        "vmax",
        "pmin",
        "pmax",
        "invert",
        "error_type",
        "distance_measure",
        "max_error",
        "max_error_interp_mode",
        "max_error_as_maxdist",
        "max_error_strict",
        "min_pct_change",
        "max_pct_change",
        "min_similarity",
    ],
    output_names=["similarity"],
    attr_settings=dict(
        close=dict(
            doc="Close price series.",
        ),
        similarity=dict(
            doc="Rolling pattern similarity series.",
        ),
    ),
).with_apply_func(
    generic_nb.rolling_pattern_similarity_nb,
    param_settings=dict(
        pattern=dict(
            is_array_like=True,
            min_one_dim=True,
            doc="Pattern to compare against.",
        ),
        window=dict(
            doc="Window size.",
        ),
        max_window=dict(
            doc="Maximum window size.",
        ),
        row_select_prob=dict(
            doc="Probability of selecting a row in the close series.",
        ),
        window_select_prob=dict(
            doc="Probability of selecting a window in the close series.",
        ),
        interp_mode=dict(
            dtype=generic_enums.InterpMode,
            post_index_func=lambda index: index.str.lower(),
            doc="Interpolation mode (see `vectorbtpro.generic.enums.InterpMode`).",
        ),
        rescale_mode=dict(
            dtype=generic_enums.RescaleMode,
            post_index_func=lambda index: index.str.lower(),
            doc="Rescale mode (see `vectorbtpro.generic.enums.RescaleMode`).",
        ),
        vmin=dict(
            doc="Minimum value for rescaling the window.",
        ),
        vmax=dict(
            doc="Maximum value for rescaling the window.",
        ),
        pmin=dict(
            doc="Minimum value for rescaling the pattern.",
        ),
        pmax=dict(
            doc="Maximum value for rescaling the pattern.",
        ),
        invert=dict(
            doc="Whether to invert the pattern.",
        ),
        error_type=dict(
            dtype=generic_enums.ErrorType,
            post_index_func=lambda index: index.str.lower(),
            doc="Error type (see `vectorbtpro.generic.enums.ErrorType`).",
        ),
        distance_measure=dict(
            dtype=generic_enums.DistanceMeasure,
            post_index_func=lambda index: index.str.lower(),
            doc="Distance measure (see `vectorbtpro.generic.enums.DistanceMeasure`).",
        ),
        max_error=dict(
            is_array_like=True,
            min_one_dim=True,
            doc="Maximum error threshold.",
        ),
        max_error_interp_mode=dict(
            dtype=generic_enums.InterpMode,
            post_index_func=lambda index: index.str.lower(),
            doc="Interpolation mode for the maximum error (see `vectorbtpro.generic.enums.InterpMode`).",
        ),
        max_error_as_maxdist=dict(
            doc="Whether to use maximum error as maximum distance.",
        ),
        max_error_strict=dict(
            doc="Whether exceeding the maximum error is considered a failure.",
        ),
        min_pct_change=dict(
            doc="Minimum percentage change threshold.",
        ),
        max_pct_change=dict(
            doc="Maximum percentage change threshold.",
        ),
        min_similarity=dict(
            doc="Minimum similarity threshold.",
        ),
    ),
    window=None,
    max_window=None,
    row_select_prob=1.0,
    window_select_prob=1.0,
    interp_mode="mixed",
    rescale_mode="minmax",
    vmin=np.nan,
    vmax=np.nan,
    pmin=np.nan,
    pmax=np.nan,
    invert=False,
    error_type="absolute",
    distance_measure="mae",
    max_error=np.nan,
    max_error_interp_mode=None,
    max_error_as_maxdist=False,
    max_error_strict=False,
    min_pct_change=np.nan,
    max_pct_change=np.nan,
    min_similarity=np.nan,
)


class _PATSIM(PATSIM):
    """Class representing the rolling pattern similarity indicator.

    See:
        * `PATSIM.run` for the main entry point.
        * `vectorbtpro.generic.nb.rolling.rolling_pattern_similarity_nb` for the underlying implementation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        similarity_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `PATSIM.similarity` against `PATSIM.close`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.

            similarity_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `PATSIM.similarity`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated or newly created figure with the similarity line plotted.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.PATSIM.run(ohlcv['Close'], np.array([1, 2, 3, 2, 1]), 30).plot().show()
            ```

            ![](/assets/images/api/PATSIM.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PATSIM.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        similarity_trace_kwargs = merge_dicts(
            dict(name="Similarity", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            similarity_trace_kwargs,
        )
        fig = self_col.similarity.vbt.lineplot(
            trace_kwargs=similarity_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]
        default_layout = dict()
        default_layout[yaxis] = dict(tickformat=",.0%")
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        return fig

    def overlay_with_heatmap(
        self,
        column: tp.Optional[tp.Column] = None,
        close_trace_kwargs: tp.KwargsLike = None,
        similarity_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Overlay `PATSIM.similarity` as a heatmap on top of `PATSIM.close`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `PATSIM.close`.
            similarity_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Heatmap` for `PATSIM.similarity`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated or newly created figure with the heatmap overlay on `PATSIM.close`.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.PATSIM.run(ohlcv['Close'], np.array([1, 2, 3, 2, 1]), 30).overlay_with_heatmap().show()
            ```

            ![](/assets/images/api/PATSIM_heatmap.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PATSIM_heatmap.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if similarity_trace_kwargs is None:
            similarity_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        similarity_trace_kwargs = merge_dicts(
            dict(
                colorbar=dict(tickformat=",.0%"),
                colorscale=[
                    [0.0, "rgba(0, 0, 0, 0)"],
                    [1.0, plotting_cfg["color_schema"]["lightpurple"]],
                ],
                zmin=0,
                zmax=1,
            ),
            similarity_trace_kwargs,
        )
        fig = self_col.close.vbt.overlay_with_heatmap(
            self_col.similarity,
            trace_kwargs=close_trace_kwargs,
            heatmap_kwargs=dict(y_labels=["Similarity"], trace_kwargs=similarity_trace_kwargs),
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig


PATSIM.clone_docstring(_PATSIM)
PATSIM.clone_method(_PATSIM.plot)
PATSIM.clone_method(_PATSIM.overlay_with_heatmap)
