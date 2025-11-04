# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing base plotting functions.

Provides functions to visualize data using interactive figure widgets that integrate with
ipywidgets in Jupyter Notebook and JupyterLab.

For more details on using Plotly, refer to
[Getting Started with Plotly in Python](https://plotly.com/python/getting-started/).

!!! warning
    Errors related to plotting in the Jupyter environment may appear in the logs rather than in the cell output.
"""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("plotly")

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

from vectorbtpro import _typing as tp
from vectorbtpro.base import reshaping
from vectorbtpro.utils import checks
from vectorbtpro.utils.array_ import rescale
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.colors import map_value_to_cmap
from vectorbtpro.utils.config import Configured, merge_dicts, resolve_dict
from vectorbtpro.utils.figure import make_figure

__all__ = [
    "TraceUpdater",
    "Gauge",
    "Bar",
    "Scatter",
    "Histogram",
    "Box",
    "Heatmap",
    "Volume",
]


def clean_labels(labels: tp.Labels) -> tp.Labels:
    """Clean labels for Plotly compatibility.

    Args:
        labels (Labels): Sequence of labels, which may be a Pandas MultiIndex, PeriodIndex, or list.

    Returns:
        Labels: List of labels formatted for Plotly.
    """
    if isinstance(labels, pd.MultiIndex):
        labels = labels.to_flat_index()
    if isinstance(labels, pd.PeriodIndex):
        labels = labels.map(str)
    if isinstance(labels, pd.Index):
        labels = labels.tolist()
    if len(labels) > 0 and isinstance(labels[0], tuple):
        labels = list(map(str, labels))
    return labels


def clean_data(data: tp.Any) -> tp.Any:
    """Clean data for Plotly compatibility.

    Args:
        data (Any): Input data that may contain NaN values.

    Returns:
        Any: Data with NaN values replaced by None if it's a floating NumPy array;
            otherwise, the original data.
    """
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
        mask = np.isnan(data)
        if mask.any():
            return np.where(mask, None, data.astype(object))
    return data


class TraceType(Configured):
    """Class representing a trace type configuration for Plotly visualizations."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"


class TraceUpdater(Base):
    """Class for updating Plotly traces.

    Args:
        fig (BaseFigure): Figure to update.
        traces (Tuple[BaseTraceType, ...]): Tuple of Plotly trace objects to update.
    """

    def __init__(self, fig: tp.BaseFigure, traces: tp.Tuple[BaseTraceType, ...]) -> None:
        self._fig = fig
        self._traces = traces

    @property
    def fig(self) -> tp.BaseFigure:
        """Plotly figure widget containing the traces.

        Returns:
            BaseFigure: Plotly figure widget.
        """
        return self._fig

    @property
    def traces(self) -> tp.Tuple[BaseTraceType, ...]:
        """Tuple of Plotly trace objects that will be updated.

        Returns:
            Tuple[BaseTraceType, ...]: Tuple of Plotly trace objects.
        """
        return self._traces

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike, *args, **kwargs) -> None:
        """Update a single Plotly trace with new data.

        Args:
            trace (BaseTraceType): Plotly trace to update.
            data (ArrayLike): New data for the trace.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        """Update all Plotly traces with new data.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


class Gauge(TraceType, TraceUpdater):
    """Class for creating a gauge plot.

    Args:
        value (Optional[float]): Value to display on the gauge.
        label (Optional[str]): Label shown on the gauge.
        value_range (Optional[Tuple[float, float]]): Range of values for the gauge.
        cmap_name (str): Matplotlib-compatible colormap name.

            See the [list of available colormaps](https://matplotlib.org/tutorials/colors/colormaps.html).
        trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Indicator`.
        add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
            for example, `dict(row=1, col=1)`.
        make_figure_kwargs (KwargsLike): Keyword arguments for making the figure.

            See `vectorbtpro.utils.figure.make_figure`.
        fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
        **layout_kwargs: Keyword arguments for `fig.update_layout`.

    !!! info
        For default settings, see `vectorbtpro._settings.plotting`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> gauge = vbt.Gauge(
        ...     value=2,
        ...     value_range=(1, 3),
        ...     label='My Gauge'
        ... )
        >>> gauge.fig.show()
        ```

        ![](/assets/images/api/Gauge.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/Gauge.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        value: tp.Optional[float] = None,
        label: tp.Optional[str] = None,
        value_range: tp.Optional[tp.Tuple[float, float]] = None,
        cmap_name: str = "Spectral",
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> None:
        TraceType.__init__(
            self,
            value=value,
            label=label,
            value_range=value_range,
            cmap_name=cmap_name,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        from vectorbtpro._settings import settings

        layout_cfg = settings["plotting"]["layout"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
            if "width" in layout_cfg:
                # Calculate nice width and height
                fig.update_layout(
                    width=layout_cfg["width"] * 0.7,
                    height=layout_cfg["width"] * 0.5,
                    margin=dict(t=80),
                )
        fig.update_layout(**layout_kwargs)

        _trace_kwargs = merge_dicts(
            dict(
                domain=dict(x=[0, 1], y=[0, 1]),
                mode="gauge+number+delta",
                title=dict(text=label),
            ),
            trace_kwargs,
        )
        trace = go.Indicator(**_trace_kwargs)
        if value is not None:
            self.update_trace(trace, value, value_range=value_range, cmap_name=cmap_name)
        fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, (fig.data[-1],))
        self._value_range = value_range
        self._cmap_name = cmap_name

    @property
    def value_range(self) -> tp.Tuple[float, float]:
        """Value range of the gauge as a tuple of minimum and maximum values.

        Returns:
            Tuple[float, float]: Value range of the gauge.
        """
        return self._value_range

    @property
    def cmap_name(self) -> str:
        """Name of the matplotlib-compatible colormap used for the gauge.

        Returns:
            str: Name of the colormap.
        """
        return self._cmap_name

    @classmethod
    def update_trace(
        cls,
        trace: BaseTraceType,
        value: float,
        value_range: tp.Optional[tp.Tuple[float, float]] = None,
        cmap_name: str = "Spectral",
    ) -> None:
        """Update the gauge trace with a new value and optional range.

        Args:
            trace (BaseTraceType): Plotly trace to update.
            value (float): Value to display on the gauge.
            value_range (Optional[Tuple[float, float]]): Range of values for the gauge.
            cmap_name (str): Matplotlib-compatible colormap name.

                See the [list of available colormaps](https://matplotlib.org/tutorials/colors/colormaps.html).

        Returns:
            None
        """
        if value_range is not None:
            trace.gauge.axis.range = value_range
            if cmap_name is not None:
                trace.gauge.bar.color = map_value_to_cmap(
                    value, cmap_name, vmin=value_range[0], vmax=value_range[1]
                )
        trace.delta.reference = trace.value
        trace.value = value

    def update(self, value: float) -> None:
        """Update all gauge traces with a new value.

        Args:
            value (float): Value to display on the gauge.

        Returns:
            None
        """
        if self.value_range is None:
            self._value_range = value, value
        else:
            self._value_range = min(self.value_range[0], value), max(self.value_range[1], value)

        with self.fig.batch_update():
            self.update_trace(
                self.traces[0],
                value=value,
                value_range=self.value_range,
                cmap_name=self.cmap_name,
            )


class Bar(TraceType, TraceUpdater):
    """Class for creating a bar plot.

    Args:
        data (Optional[ArrayLike]): Data convertible to a NumPy array.

            Must have shape corresponding to (`x_labels`, `trace_names`).
        trace_names (TraceNames): Names for traces corresponding to data columns.
        x_labels (Optional[Labels]): X-axis labels corresponding to the index in pandas.
        trace_kwargs (KwargsLikeSequence): Keyword arguments for `plotly.graph_objects.Bar`.

            Can be provided per trace as a sequence of dictionaries.
        add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
            for example, `dict(row=1, col=1)`.
        make_figure_kwargs (KwargsLike): Keyword arguments for making the figure.

            See `vectorbtpro.utils.figure.make_figure`.
        fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
        **layout_kwargs: Keyword arguments for `fig.update_layout`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> bar = vbt.Bar(
        ...     data=[[1, 2], [3, 4]],
        ...     trace_names=['a', 'b'],
        ...     x_labels=['x', 'y']
        ... )
        >>> bar.fig.show()
        ```

        ![](/assets/images/api/Bar.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/Bar.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        trace_names: tp.TraceNames = None,
        x_labels: tp.Optional[tp.Labels] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> None:
        TraceType.__init__(
            self,
            data=data,
            trace_names=trace_names,
            x_labels=x_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if trace_names is not None:
                checks.assert_shape_equal(data, trace_names, (1, 0))
        else:
            if trace_names is None:
                raise ValueError("At least data or trace_names must be passed")
        if trace_names is None:
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]
        if x_labels is not None:
            x_labels = clean_labels(x_labels)

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)
            trace_name = _trace_kwargs.pop("name", trace_name)
            if trace_name is not None:
                trace_name = str(trace_name)
            _trace_kwargs = merge_dicts(
                dict(x=x_labels, name=trace_name, showlegend=trace_name is not None),
                _trace_kwargs,
            )
            trace = go.Bar(**_trace_kwargs)
            if data is not None:
                self.update_trace(trace, data, i)
            fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names) :])

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike, i: int) -> None:
        """Update a single bar trace with new data.

        Args:
            trace (BaseTraceType): Plotly trace to update.
            data (ArrayLike): Data convertible to a NumPy array.

                Must have shape corresponding to (`x_labels`, `trace_names`).
            i (int): Index of the trace to update.

        Returns:
            None
        """
        d = clean_data(reshaping.to_2d_array(data)[:, i])

        trace.y = d
        if trace.marker.colorscale is not None:
            trace.marker.color = d

    def update(self, data: tp.ArrayLike) -> None:
        """Update all bar traces with new data.

        Args:
            data (ArrayLike): Data convertible to a NumPy array.

                Must have shape corresponding to (`x_labels`, `trace_names`).

        Returns:
            None
        """
        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                self.update_trace(trace, data, i)


class Scatter(TraceType, TraceUpdater):
    """Class for creating a scatter plot.

    Args:
        data (Optional[ArrayLike]): Data convertible to a NumPy array.

            Must have shape corresponding to (`x_labels`, `trace_names`).
        trace_names (TraceNames): Names for traces corresponding to data columns.
        x_labels (Optional[Labels]): Labels for the x-axis, typically representing the index in a Pandas DataFrame.
        trace_kwargs (KwargsLikeSequence): Keyword arguments for `plotly.graph_objects.Scatter`.

            Can be provided per trace as a sequence of dictionaries.
        add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
            for example, `dict(row=1, col=1)`.
        make_figure_kwargs (KwargsLike): Keyword arguments for making the figure.

            See `vectorbtpro.utils.figure.make_figure`.
        fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
        use_gl (bool): Flag to use `plotly.graph_objects.Scattergl`.

            Defaults to the global setting. If the global configuration is None and the data has
            more than 10,000 points, this flag is set to True.
        **layout_kwargs: Keyword arguments for `fig.update_layout`.

    !!! info
        For default settings, see `vectorbtpro._settings.plotting`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> scatter = vbt.Scatter(
        ...     data=[[1, 2], [3, 4]],
        ...     trace_names=['a', 'b'],
        ...     x_labels=['x', 'y']
        ... )
        >>> scatter.fig.show()
        ```

        ![](/assets/images/api/Scatter.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/Scatter.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        trace_names: tp.TraceNames = None,
        x_labels: tp.Optional[tp.Labels] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        use_gl: tp.Optional[bool] = None,
        **layout_kwargs,
    ) -> None:
        TraceType.__init__(
            self,
            data=data,
            trace_names=trace_names,
            x_labels=x_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if trace_names is not None:
                checks.assert_shape_equal(data, trace_names, (1, 0))
        else:
            if trace_names is None:
                raise ValueError("At least data or trace_names must be passed")
        if trace_names is None:
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]
        if x_labels is not None:
            x_labels = clean_labels(x_labels)

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)
            _use_gl = _trace_kwargs.pop("use_gl", use_gl)
            if _use_gl is None:
                _use_gl = plotting_cfg["use_gl"]
            if _use_gl is None:
                _use_gl = _use_gl is None and data is not None and data.size >= 10000
            trace_name = _trace_kwargs.pop("name", trace_name)
            if trace_name is not None:
                trace_name = str(trace_name)
            if _use_gl:
                scatter_obj = go.Scattergl
            else:
                scatter_obj = go.Scatter
            try:
                from plotly_resampler.aggregation import AbstractFigureAggregator

                if isinstance(fig, AbstractFigureAggregator):
                    use_resampler = True
                else:
                    use_resampler = False
            except ImportError:
                use_resampler = False
            if use_resampler:
                if data is None:
                    raise ValueError(
                        "Cannot create empty scatter traces when using plotly-resampler"
                    )
                _trace_kwargs = merge_dicts(
                    dict(name=trace_name, showlegend=trace_name is not None),
                    _trace_kwargs,
                )
                trace = scatter_obj(**_trace_kwargs)
                fig.add_trace(trace, hf_x=x_labels, hf_y=data[:, i], **add_trace_kwargs)
            else:
                _trace_kwargs = merge_dicts(
                    dict(x=x_labels, name=trace_name, showlegend=trace_name is not None),
                    _trace_kwargs,
                )
                trace = scatter_obj(**_trace_kwargs)
                if data is not None:
                    self.update_trace(trace, data, i)
                fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names) :])

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike, i: int) -> None:
        """Update a single scatter trace with new data.

        Args:
            trace (BaseTraceType): Plotly trace to update.
            data (ArrayLike): Data convertible to a NumPy array.

                Must have shape corresponding to (`x_labels`, `trace_names`).
            i (int): Index of the trace to update.

        Returns:
            None
        """
        d = clean_data(reshaping.to_2d_array(data)[:, i])

        trace.y = d

    def update(self, data: tp.ArrayLike) -> None:
        """Update all scatter traces with new data.

                Args:
                    data (ArrayLike): Data convertible to a NumPy array.
        `
                        Must have shape corresponding to (`x_labels`, `trace_names`).

                Returns:
                    None
        """
        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                self.update_trace(trace, data, i)


class Histogram(TraceType, TraceUpdater):
    """Class for creating a histogram plot.

    Args:
        data (Optional[ArrayLike]): Data convertible to a NumPy array.

            The second axis must correspond to `trace_names`.
        trace_names (TraceNames): Names for traces corresponding to data columns.
        horizontal (bool): Flag indicating whether the plot is oriented horizontally.
        remove_nan (bool): Flag determining whether NaN values are removed from the data.
        from_quantile (float): Lower quantile threshold used to filter out data points.

            Must be in the range [0, 1].
        to_quantile (float): Upper quantile threshold used to filter out data points.

            Must be in the range [0, 1].
        trace_kwargs (KwargsLikeSequence): Keyword arguments for `plotly.graph_objects.Histogram`.

            Can be provided per trace as a sequence of dictionaries.
        add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
            for example, `dict(row=1, col=1)`.
        make_figure_kwargs (KwargsLike): Keyword arguments for making the figure.

            See `vectorbtpro.utils.figure.make_figure`.
        fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
        **layout_kwargs: Keyword arguments for `fig.update_layout`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> hist = vbt.Histogram(
        ...     data=[[1, 2], [3, 4], [2, 1]],
        ...     trace_names=['a', 'b']
        ... )
        >>> hist.fig.show()
        ```

        ![](/assets/images/api/Histogram.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/Histogram.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        trace_names: tp.TraceNames = None,
        horizontal: bool = False,
        remove_nan: bool = True,
        from_quantile: tp.Optional[float] = None,
        to_quantile: tp.Optional[float] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> None:
        TraceType.__init__(
            self,
            data=data,
            trace_names=trace_names,
            horizontal=horizontal,
            remove_nan=remove_nan,
            from_quantile=from_quantile,
            to_quantile=to_quantile,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if trace_names is not None:
                checks.assert_shape_equal(data, trace_names, (1, 0))
        else:
            if trace_names is None:
                raise ValueError("At least data or trace_names must be passed")
        if trace_names is None:
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
            fig.update_layout(barmode="overlay")
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)
            trace_name = _trace_kwargs.pop("name", trace_name)
            if trace_name is not None:
                trace_name = str(trace_name)
            _trace_kwargs = merge_dicts(
                dict(
                    opacity=0.75 if len(trace_names) > 1 else 1,
                    name=trace_name,
                    showlegend=trace_name is not None,
                ),
                _trace_kwargs,
            )
            trace = go.Histogram(**_trace_kwargs)
            if data is not None:
                self.update_trace(
                    trace,
                    data,
                    i,
                    horizontal=horizontal,
                    remove_nan=remove_nan,
                    from_quantile=from_quantile,
                    to_quantile=to_quantile,
                )
            fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names) :])
        self._horizontal = horizontal
        self._remove_nan = remove_nan
        self._from_quantile = from_quantile
        self._to_quantile = to_quantile

    @property
    def horizontal(self) -> bool:
        """Indicates whether the histogram is plotted horizontally.

        Returns:
            bool: True if the histogram is horizontal, False otherwise.
        """
        return self._horizontal

    @property
    def remove_nan(self) -> bool:
        """Indicates whether NaN values are removed from the data.

        Returns:
            bool: True if NaN values are removed, False otherwise.
        """
        return self._remove_nan

    @property
    def from_quantile(self) -> float:
        """Specifies the lower quantile threshold used to filter out data points.

        Returns:
            float: Lower quantile threshold.
        """
        return self._from_quantile

    @property
    def to_quantile(self) -> float:
        """Specifies the upper quantile threshold used to filter out data points.

        Returns:
            float: Upper quantile threshold.
        """
        return self._to_quantile

    @classmethod
    def update_trace(
        cls,
        trace: BaseTraceType,
        data: tp.ArrayLike,
        i: int,
        horizontal: bool = False,
        remove_nan: bool = True,
        from_quantile: tp.Optional[float] = None,
        to_quantile: tp.Optional[float] = None,
    ) -> None:
        """Update a single histogram trace with new data.

        Args:
            trace (BaseTraceType): Plotly trace to update.
            data (ArrayLike): Data convertible to a NumPy array.

                The second axis must correspond to `trace_names`.
            i (int): Index of the trace to update.
            horizontal (bool): Flag indicating whether the plot is oriented horizontally.
            remove_nan (bool): Flag determining whether NaN values are removed from the data.
            from_quantile (float): Lower quantile threshold used to filter out data points.

                Must be in the range [0, 1].
            to_quantile (float): Upper quantile threshold used to filter out data points.

                Must be in the range [0, 1].

        Returns:
            None
        """
        d = reshaping.to_2d_array(data)[:, i]
        if remove_nan:
            d = d[~np.isnan(d)]
        mask = np.full(d.shape, True)
        if from_quantile is not None:
            mask &= d >= np.quantile(d, from_quantile)
        if to_quantile is not None:
            mask &= d <= np.quantile(d, to_quantile)
        d = clean_data(d[mask])

        if horizontal:
            trace.x = None
            trace.y = d
        else:
            trace.x = d
            trace.y = None

    def update(self, data: tp.ArrayLike) -> None:
        """Update all histogram traces with new data.

        Args:
            data (ArrayLike): Data convertible to a NumPy array.

                The second axis must correspond to `trace_names`.

        Returns:
            None
        """
        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                self.update_trace(
                    trace,
                    data,
                    i,
                    horizontal=self.horizontal,
                    remove_nan=self.remove_nan,
                    from_quantile=self.from_quantile,
                    to_quantile=self.to_quantile,
                )


class Box(TraceType, TraceUpdater):
    """Class for creating a box plot.

    This class creates a box plot from the provided data and configuration parameters.
    For additional keyword arguments for trace customization, see `Histogram`.

    Args:
        data (Optional[ArrayLike]): Data convertible to a NumPy array.

            The second axis must correspond to `trace_names`.
        trace_names (TraceNames): Names for traces corresponding to data columns.
        horizontal (bool): Flag indicating whether the plot is oriented horizontally.
        remove_nan (bool): Flag determining whether NaN values are removed from the data.
        from_quantile (Optional[float]): Lower quantile threshold to filter out data.

            Data below this quantile are excluded.
        to_quantile (Optional[float]): Upper quantile threshold to filter out data.

            Data above this quantile are excluded.
        trace_kwargs (KwargsLikeSequence): Keyword arguments for `plotly.graph_objects.Box`.

            Can be provided per trace as a sequence of dictionaries.
        add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
            for example, `dict(row=1, col=1)`.
        make_figure_kwargs (KwargsLike): Keyword arguments for making the figure.

            See `vectorbtpro.utils.figure.make_figure`.
        fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
        **layout_kwargs: Keyword arguments for `fig.update_layout`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> box = vbt.Box(
        ...     data=[[1, 2], [3, 4], [2, 1]],
        ...     trace_names=['a', 'b']
        ... )
        >>> box.fig.show()
        ```

        ![](/assets/images/api/Box.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/Box.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        trace_names: tp.TraceNames = None,
        horizontal: bool = False,
        remove_nan: bool = True,
        from_quantile: tp.Optional[float] = None,
        to_quantile: tp.Optional[float] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> None:
        TraceType.__init__(
            self,
            data=data,
            trace_names=trace_names,
            horizontal=horizontal,
            remove_nan=remove_nan,
            from_quantile=from_quantile,
            to_quantile=to_quantile,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if trace_names is not None:
                checks.assert_shape_equal(data, trace_names, (1, 0))
        else:
            if trace_names is None:
                raise ValueError("At least data or trace_names must be passed")
        if trace_names is None:
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)
            trace_name = _trace_kwargs.pop("name", trace_name)
            if trace_name is not None:
                trace_name = str(trace_name)
            _trace_kwargs = merge_dicts(
                dict(name=trace_name, showlegend=trace_name is not None, boxmean="sd"),
                _trace_kwargs,
            )
            trace = go.Box(**_trace_kwargs)
            if data is not None:
                self.update_trace(
                    trace,
                    data,
                    i,
                    horizontal=horizontal,
                    remove_nan=remove_nan,
                    from_quantile=from_quantile,
                    to_quantile=to_quantile,
                )
            fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names) :])
        self._horizontal = horizontal
        self._remove_nan = remove_nan
        self._from_quantile = from_quantile
        self._to_quantile = to_quantile

    @property
    def horizontal(self) -> bool:
        """Indicates if the box plot is oriented horizontally.

        Returns:
            bool: True if the box plot is horizontal, False otherwise.
        """
        return self._horizontal

    @property
    def remove_nan(self) -> bool:
        """Specifies whether NaN values are removed from the data.

        Returns:
            bool: True if NaN values are removed, False otherwise.
        """
        return self._remove_nan

    @property
    def from_quantile(self) -> float:
        """Specifies the lower quantile threshold; data points below this value are excluded.

        Returns:
            float: Lower quantile threshold.
        """
        return self._from_quantile

    @property
    def to_quantile(self) -> float:
        """Specifies the upper quantile threshold; data points above this value are excluded.

        Returns:
            float: Upper quantile threshold.
        """
        return self._to_quantile

    @classmethod
    def update_trace(
        cls,
        trace: BaseTraceType,
        data: tp.ArrayLike,
        i: int,
        horizontal: bool = False,
        remove_nan: bool = True,
        from_quantile: tp.Optional[float] = None,
        to_quantile: tp.Optional[float] = None,
    ) -> None:
        """Update a single box trace with new data.

        Args:
            trace (BaseTraceType): Plotly trace to update.
            data (ArrayLike): Data convertible to a NumPy array.

                The second axis must correspond to `trace_names`.
            i (int): Index of the trace to update.
            horizontal (bool): Flag indicating whether the plot is oriented horizontally.
            remove_nan (bool): Flag determining whether NaN values are removed from the data.
            from_quantile (Optional[float]): Lower quantile threshold to filter out data.

                Data below this quantile are excluded.
            to_quantile (Optional[float]): Upper quantile threshold to filter out data.

                Data above this quantile are excluded.

        Returns:
            None
        """
        d = reshaping.to_2d_array(data)[:, i]
        if remove_nan:
            d = d[~np.isnan(d)]
        mask = np.full(d.shape, True)
        if from_quantile is not None:
            mask &= d >= np.quantile(d, from_quantile)
        if to_quantile is not None:
            mask &= d <= np.quantile(d, to_quantile)
        d = clean_data(d[mask])

        if horizontal:
            trace.x = d
            trace.y = None
        else:
            trace.x = None
            trace.y = d

    def update(self, data: tp.ArrayLike) -> None:
        """Update all box traces with new data.

        Args:
            data (ArrayLike): Data convertible to a NumPy array.

                The second axis must correspond to `trace_names`.

        Returns:
            None
        """
        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                self.update_trace(
                    trace,
                    data,
                    i,
                    horizontal=self.horizontal,
                    remove_nan=self.remove_nan,
                    from_quantile=self.from_quantile,
                    to_quantile=self.to_quantile,
                )


class Heatmap(TraceType, TraceUpdater):
    """Class for creating a heatmap plot.

    Args:
        data (Optional[ArrayLike]): Data convertible to a NumPy array.

            Must have shape (`y_labels`, `x_labels`).
        x_labels (Optional[Labels]): Labels for the x-axis corresponding to DataFrame columns.
        y_labels (Optional[Labels]): Labels for the y-axis corresponding to DataFrame index.
        is_x_category (bool): Indicates whether the x-axis represents categorical data.
        is_y_category (bool): Flag indicating whether to treat the y-axis as categorical.
        trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Heatmap`.
        add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
            for example, `dict(row=1, col=1)`.
        make_figure_kwargs (KwargsLike): Keyword arguments for making the figure.

            See `vectorbtpro.utils.figure.make_figure`.
        fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
        **layout_kwargs: Keyword arguments for `fig.update_layout`.

    !!! info
        For default settings, see `vectorbtpro._settings.plotting`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> heatmap = vbt.Heatmap(
        ...     data=[[1, 2], [3, 4]],
        ...     x_labels=['a', 'b'],
        ...     y_labels=['x', 'y']
        ... )
        >>> heatmap.fig.show()
        ```

        ![](/assets/images/api/Heatmap.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/Heatmap.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        x_labels: tp.Optional[tp.Labels] = None,
        y_labels: tp.Optional[tp.Labels] = None,
        is_x_category: bool = False,
        is_y_category: bool = False,
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> None:
        TraceType.__init__(
            self,
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        from vectorbtpro._settings import settings

        layout_cfg = settings["plotting"]["layout"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            data = reshaping.to_2d_array(data)
            if x_labels is not None:
                checks.assert_shape_equal(data, x_labels, (1, 0))
            if y_labels is not None:
                checks.assert_shape_equal(data, y_labels, (0, 0))
        else:
            if x_labels is None or y_labels is None:
                raise ValueError("At least data, or x_labels and y_labels must be passed")
        if x_labels is not None:
            x_labels = clean_labels(x_labels)
        if y_labels is not None:
            y_labels = clean_labels(y_labels)

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
            if "width" in layout_cfg:
                # Calculate nice width and height
                max_width = layout_cfg["width"]
                if data is not None:
                    x_len = data.shape[1]
                    y_len = data.shape[0]
                else:
                    x_len = len(x_labels)
                    y_len = len(y_labels)
                width = math.ceil(
                    rescale(x_len / (x_len + y_len), (0, 1), (0.3 * max_width, max_width))
                )
                width = min(width + 150, max_width)  # account for colorbar
                height = math.ceil(
                    rescale(y_len / (x_len + y_len), (0, 1), (0.3 * max_width, max_width))
                )
                height = min(height, max_width * 0.7)  # limit height
                fig.update_layout(width=width, height=height)

        _trace_kwargs = merge_dicts(
            dict(hoverongaps=False, colorscale="Plasma", x=x_labels, y=y_labels),
            trace_kwargs,
        )
        trace = go.Heatmap(**_trace_kwargs)
        if data is not None:
            self.update_trace(trace, data)
        fig.add_trace(trace, **add_trace_kwargs)

        xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]
        axis_kwargs = dict()
        if is_x_category:
            axis_kwargs[xaxis] = dict(type="category")
        if is_y_category:
            axis_kwargs[yaxis] = dict(type="category")
        fig.update_layout(**axis_kwargs)
        fig.update_layout(**layout_kwargs)

        TraceUpdater.__init__(self, fig, (fig.data[-1],))

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike) -> None:
        """Update a single heatmap trace with new data.

        Args:
            trace (BaseTraceType): Plotly trace to update.
            data (ArrayLike): Data convertible to a NumPy array.

                Must have shape (`y_labels`, `x_labels`).

        Returns:
            None
        """
        d = clean_data(reshaping.to_2d_array(data))

        trace.z = d

    def update(self, data: tp.ArrayLike) -> None:
        """Update all heatmap traces with new data.

        Args:
            data (ArrayLike): Data convertible to a NumPy array.

                Must have shape (`y_labels`, `x_labels`).

        Returns:
            None
        """
        with self.fig.batch_update():
            self.update_trace(self.traces[0], data)


class Volume(TraceType, TraceUpdater):
    """Class for creating a volume plot.

    Args:
        data (Optional[ArrayLike]): Data convertible to a NumPy array.

            Must have shape corresponding to (`x_labels`, `y_labels`, `z_labels`).
        x_labels (Optional[Labels]): Labels for the x-axis.
        y_labels (Optional[Labels]): Labels for the y-axis.
        z_labels (Optional[Labels]): Labels for the z-axis.
        trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Volume`.
        add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
            for example, `dict(row=1, col=1)`.
        scene_name (str): Name of the 3D scene.
        make_figure_kwargs (KwargsLike): Keyword arguments for making the figure.

            See `vectorbtpro.utils.figure.make_figure`.
        fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
        **layout_kwargs: Keyword arguments for `fig.update_layout`.

    !!! note
        Figure widgets currently have issues displaying NaNs. Use the `.show()` method for rendering.

    !!! info
        For default settings, see `vectorbtpro._settings.plotting`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> volume = vbt.Volume(
        ...     data=np.random.randint(1, 10, size=(3, 3, 3)),
        ...     x_labels=['a', 'b', 'c'],
        ...     y_labels=['d', 'e', 'f'],
        ...     z_labels=['g', 'h', 'i']
        ... )
        >>> volume.fig.show()
        ```

        ![](/assets/images/api/Volume.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/Volume.dark.svg#only-dark){: .iimg loading=lazy }
    """

    def __init__(
        self,
        data: tp.Optional[tp.ArrayLike] = None,
        x_labels: tp.Optional[tp.Labels] = None,
        y_labels: tp.Optional[tp.Labels] = None,
        z_labels: tp.Optional[tp.Labels] = None,
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        scene_name: str = "scene",
        make_figure_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> None:
        TraceType.__init__(
            self,
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            z_labels=z_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            scene_name=scene_name,
            make_figure_kwargs=make_figure_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        from vectorbtpro._settings import settings

        layout_cfg = settings["plotting"]["layout"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is not None:
            checks.assert_ndim(data, 3)
            data = np.asarray(data)
            x_len, y_len, z_len = data.shape
            if x_labels is not None:
                checks.assert_shape_equal(data, x_labels, (0, 0))
            if y_labels is not None:
                checks.assert_shape_equal(data, y_labels, (1, 0))
            if z_labels is not None:
                checks.assert_shape_equal(data, z_labels, (2, 0))
        else:
            if x_labels is None or y_labels is None or z_labels is None:
                raise ValueError("At least data, or x_labels, y_labels and z_labels must be passed")
            x_len = len(x_labels)
            y_len = len(y_labels)
            z_len = len(z_labels)
        if x_labels is None:
            x_labels = np.arange(x_len)
        else:
            x_labels = clean_labels(x_labels)
        if y_labels is None:
            y_labels = np.arange(y_len)
        else:
            y_labels = clean_labels(y_labels)
        if z_labels is None:
            z_labels = np.arange(z_len)
        else:
            z_labels = clean_labels(z_labels)
        x_labels = np.asarray(x_labels)
        y_labels = np.asarray(y_labels)
        z_labels = np.asarray(z_labels)

        if fig is None:
            fig = make_figure(**resolve_dict(make_figure_kwargs))
            if "width" in layout_cfg:
                # Calculate nice width and height
                fig.update_layout(width=layout_cfg["width"], height=0.7 * layout_cfg["width"])

        # Non-numeric data types are not supported by go.Volume, so use ticktext
        # Note: Currently plotly displays the entire tick array, in future versions it will be more sensible
        more_layout = dict()
        more_layout[scene_name] = dict()
        if not np.issubdtype(x_labels.dtype, np.number):
            x_ticktext = x_labels
            x_labels = np.arange(x_len)
            more_layout[scene_name]["xaxis"] = dict(
                ticktext=x_ticktext, tickvals=x_labels, tickmode="array"
            )
        if not np.issubdtype(y_labels.dtype, np.number):
            y_ticktext = y_labels
            y_labels = np.arange(y_len)
            more_layout[scene_name]["yaxis"] = dict(
                ticktext=y_ticktext, tickvals=y_labels, tickmode="array"
            )
        if not np.issubdtype(z_labels.dtype, np.number):
            z_ticktext = z_labels
            z_labels = np.arange(z_len)
            more_layout[scene_name]["zaxis"] = dict(
                ticktext=z_ticktext, tickvals=z_labels, tickmode="array"
            )
        fig.update_layout(**more_layout)
        fig.update_layout(**layout_kwargs)

        # Arrays must have the same length as the flattened data array
        x = np.repeat(x_labels, len(y_labels) * len(z_labels))
        y = np.tile(np.repeat(y_labels, len(z_labels)), len(x_labels))
        z = np.tile(z_labels, len(x_labels) * len(y_labels))

        _trace_kwargs = merge_dicts(
            dict(x=x, y=y, z=z, opacity=0.2, surface_count=15, colorscale="Plasma"),
            trace_kwargs,
        )
        trace = go.Volume(**_trace_kwargs)
        if data is not None:
            self.update_trace(trace, data)
        fig.add_trace(trace, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, (fig.data[-1],))

    @classmethod
    def update_trace(cls, trace: BaseTraceType, data: tp.ArrayLike) -> None:
        """Update a single volume trace with new data.

        Args:
            trace (BaseTraceType): Plotly trace to update.
            data (ArrayLike): Data convertible to a NumPy array.

                Must have shape corresponding to (`x_labels`, `y_labels`, `z_labels`).

        Returns:
            None
        """
        d = clean_data(np.asarray(data).flatten())

        trace.value = d

    def update(self, data: tp.ArrayLike) -> None:
        """Update all volume traces with new data.

        Args:
            data (ArrayLike): Data convertible to a NumPy array.

                Must have shape corresponding to (`x_labels`, `y_labels`, `z_labels`).

        Returns:
            None
        """
        with self.fig.batch_update():
            self.update_trace(self.traces[0], data)
