# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing custom Pandas accessors for OHLC(V) data.

Access the methods via:

* `OHLCVDFAccessor` -> `pd.DataFrame.vbt.ohlcv.*`

The accessors inherit from `vectorbtpro.generic.accessors`.

!!! note
    Accessors do not utilize caching.

## Column names

By default, vectorbtpro searches for columns with names 'open', 'high', 'low', 'close', and 'volume'
(case-insensitive). You can change the naming by either using `feature_map` in
`vectorbtpro._settings.ohlcv` or by providing a custom `feature_map` directly to the accessor.

```pycon
>>> from vectorbtpro import *

>>> df = pd.DataFrame({
...     'my_open1': [2, 3, 4, 3.5, 2.5],
...     'my_high2': [3, 4, 4.5, 4, 3],
...     'my_low3': [1.5, 2.5, 3.5, 2.5, 1.5],
...     'my_close4': [2.5, 3.5, 4, 3, 2],
...     'my_volume5': [10, 11, 10, 9, 10]
... })
>>> df.vbt.ohlcv.get_feature('open')
None

>>> my_feature_map = {
...     "my_open1": "Open",
...     "my_high2": "High",
...     "my_low3": "Low",
...     "my_close4": "Close",
...     "my_volume5": "Volume",
... }
>>> ohlcv_acc = df.vbt.ohlcv(freq='d', feature_map=my_feature_map)
>>> ohlcv_acc.get_feature('open')
0    2.0
1    3.0
2    4.0
3    3.5
4    2.5
Name: my_open1, dtype: float64
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `OHLCVDFAccessor.metrics`
    for more details.

```pycon
>>> ohlcv_acc.stats()
Start                           0
End                             4
Period            5 days 00:00:00
First Price                   2.0
Lowest Price                  1.5
Highest Price                 4.5
Last Price                    2.0
First Volume                   10
Lowest Volume                   9
Highest Volume                 11
Last Volume                    10
Name: agg_stats, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `OHLCVDFAccessor.subplots` for more details.

The `OHLCVDFAccessor` provides a single subplot based on `OHLCVDFAccessor.plot` (excluding volume):

```pycon
>>> ohlcv_acc.plots(settings=dict(ohlc_type='candlestick')).show()
```

![](/assets/images/api/ohlcv_plots.light.svg#only-light){: .iimg loading=lazy }
![](/assets/images/api/ohlcv_plots.dark.svg#only-dark){: .iimg loading=lazy }
"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.accessors import register_df_vbt_accessor
from vectorbtpro.base.reshaping import to_1d_array, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.data.base import OHLCDataMixin
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import GenericAccessor, GenericDFAccessor
from vectorbtpro.ohlcv import enums, nb
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts
from vectorbtpro.utils.decorators import hybrid_property
from vectorbtpro.utils.enum_ import map_enum_fields

if tp.TYPE_CHECKING:
    from vectorbtpro.data.base import Data as DataT
else:
    DataT = "vectorbtpro.data.base.Data"

__all__ = [
    "OHLCVDFAccessor",
]

__pdoc__ = {}

OHLCVDFAccessorT = tp.TypeVar("OHLCVDFAccessorT", bound="OHLCVDFAccessor")


@register_df_vbt_accessor("ohlcv")
class OHLCVDFAccessor(OHLCDataMixin, GenericDFAccessor):
    """Class representing an accessor on top of OHLCV data for Pandas DataFrames.

    Accessible via `pd.DataFrame.vbt.ohlcv`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Underlying data object.
        feature_map (KwargsLike): Dictionary mapping feature names to OHLCV components.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericDFAccessor`.

    !!! info
        For default settings, see `vectorbtpro._settings.ohlcv`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        feature_map: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        GenericDFAccessor.__init__(self, wrapper, obj=obj, feature_map=feature_map, **kwargs)

        self._feature_map = feature_map

    @hybrid_property
    def df_accessor_cls(cls_or_self) -> tp.Type["OHLCVDFAccessor"]:
        return OHLCVDFAccessor

    @property
    def feature_map(self) -> tp.Kwargs:
        """Merged feature mapping for OHLCV columns.

        Merges default feature settings from `vectorbtpro._settings` with the instance's feature map.

        Returns:
            Kwargs: Dictionary mapping OHLCV feature names to their corresponding column names.
        """
        from vectorbtpro._settings import settings

        ohlcv_cfg = settings["ohlcv"]

        return merge_dicts(ohlcv_cfg["feature_map"], self._feature_map)

    @property
    def feature_wrapper(self) -> ArrayWrapper:
        new_columns = self.wrapper.columns.map(
            lambda x: self.feature_map[x] if x in self.feature_map else x
        )
        return self.wrapper.replace(columns=new_columns)

    @property
    def symbol_wrapper(self) -> ArrayWrapper:
        return ArrayWrapper(self.wrapper.index, [None], 1)

    def select_feature_idxs(
        self: OHLCVDFAccessorT, idxs: tp.MaybeSequence[int], **kwargs
    ) -> OHLCVDFAccessorT:
        return self.iloc[:, idxs]

    def select_symbol_idxs(
        self: OHLCVDFAccessorT, idxs: tp.MaybeSequence[int], **kwargs
    ) -> OHLCVDFAccessorT:
        raise NotImplementedError

    def get(
        self,
        features: tp.Union[None, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, tp.MaybeSymbols] = None,
        feature: tp.Optional[tp.Feature] = None,
        symbol: tp.Optional[tp.Symbol] = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        if features is not None and feature is not None:
            raise ValueError("Must provide either features or feature, not both")
        if symbols is not None or symbol is not None:
            raise ValueError("Cannot provide symbols")
        if feature is not None:
            features = feature
            single_feature = True
        else:
            if features is None:
                return self.obj
            single_feature = not self.has_multiple_keys(features)
        if not single_feature:
            feature_idxs = [self.get_feature_idx(k, raise_error=True) for k in features]
        else:
            feature_idxs = self.get_feature_idx(features, raise_error=True)
        return self.obj.iloc[:, feature_idxs]

    # ############# Conversion ############# #

    def to_data(self, data_cls: tp.Optional[tp.Type[DataT]] = None, **kwargs) -> DataT:
        if data_cls is None:
            from vectorbtpro.data.base import Data

            data_cls = Data

        return data_cls.from_data(self.obj, columns_are_symbols=False, **kwargs)

    # ############# Transforming ############# #

    def mirror_ohlc(
        self: OHLCVDFAccessorT,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        start_value: tp.ArrayLike = np.nan,
        ref_feature: tp.ArrayLike = -1,
    ) -> tp.Frame:
        """Mirror OHLC features.

        Applies a mirror transformation to the OHLC columns ("Open", "High", "Low", "Close")
        using the specified starting value and reference feature.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            start_value (ArrayLike): Starting value used in the mirror operation.
            ref_feature (ArrayLike): Reference feature used for mirroring.

                Mapped using `vectorbtpro.ohlcv.enums.PriceFeature` if provided as a string.

        Returns:
            Frame: DataFrame with mirrored OHLC features.

        See:
            `vectorbtpro.ohlcv.nb.mirror_ohlc_nb`
        """
        if isinstance(ref_feature, str):
            ref_feature = map_enum_fields(ref_feature, enums.PriceFeature)

        open_idx = self.get_feature_idx("Open")
        high_idx = self.get_feature_idx("High")
        low_idx = self.get_feature_idx("Low")
        close_idx = self.get_feature_idx("Close")

        func = jit_reg.resolve_option(nb.mirror_ohlc_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        new_open, new_high, new_low, new_close = func(
            self.symbol_wrapper.shape_2d,
            open=to_2d_array(self.obj.iloc[:, open_idx]) if open_idx is not None else None,
            high=to_2d_array(self.obj.iloc[:, high_idx]) if high_idx is not None else None,
            low=to_2d_array(self.obj.iloc[:, low_idx]) if low_idx is not None else None,
            close=to_2d_array(self.obj.iloc[:, close_idx]) if close_idx is not None else None,
            start_value=to_1d_array(start_value),
            ref_feature=to_1d_array(ref_feature),
        )
        df = self.obj.copy()
        if open_idx is not None:
            df.iloc[:, open_idx] = new_open[:, 0]
        if high_idx is not None:
            df.iloc[:, high_idx] = new_high[:, 0]
        if low_idx is not None:
            df.iloc[:, low_idx] = new_low[:, 0]
        if close_idx is not None:
            df.iloc[:, close_idx] = new_close[:, 0]
        return df

    def resample(
        self: OHLCVDFAccessorT, *args, wrapper_meta: tp.DictLike = None, **kwargs
    ) -> OHLCVDFAccessorT:
        """Resample OHLCV data.

        Resamples each feature based on default reduction rules:

        * For "open", applies the first value.
        * For "high", applies the maximum value.
        * For "low", applies the minimum value.
        * For "close", applies the last value.
        * For "volume", applies the sum.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.
            wrapper_meta (DictLike): Metadata from the resampling operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.

        Returns:
            OHLCVDFAccessor: New accessor instance with resampled OHLCV data.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        sr_dct = {}
        for feature in self.feature_wrapper.columns:
            if isinstance(feature, str) and feature.lower() == "open":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.first_reduce_nb,
                )
            elif isinstance(feature, str) and feature.lower() == "high":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.max_reduce_nb,
                )
            elif isinstance(feature, str) and feature.lower() == "low":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.min_reduce_nb,
                )
            elif isinstance(feature, str) and feature.lower() == "close":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.last_reduce_nb,
                )
            elif isinstance(feature, str) and feature.lower() == "volume":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.sum_reduce_nb,
                )
            else:
                raise ValueError(f"Cannot match feature '{feature}'")
        new_obj = pd.DataFrame(sr_dct)
        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            obj=new_obj,
        )

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `OHLCVDFAccessor.stats`.

        Merges the defaults from `vectorbtpro.generic.accessors.GenericAccessor.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.ohlcv`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        ohlcv_stats_cfg = settings["ohlcv"]["stats"]

        return merge_dicts(GenericAccessor.stats_defaults.__get__(self), ohlcv_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            first_price=dict(
                title="First Price",
                calc_func=lambda ohlc: generic_nb.bfill_1d_nb(ohlc.values.flatten())[0],
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            lowest_price=dict(
                title="Lowest Price",
                calc_func=lambda ohlc: ohlc.values.min(),
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            highest_price=dict(
                title="Highest Price",
                calc_func=lambda ohlc: ohlc.values.max(),
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            last_price=dict(
                title="Last Price",
                calc_func=lambda ohlc: generic_nb.ffill_1d_nb(ohlc.values.flatten())[-1],
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            first_volume=dict(
                title="First Volume",
                calc_func=lambda volume: generic_nb.bfill_1d_nb(volume.values)[0],
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            lowest_volume=dict(
                title="Lowest Volume",
                calc_func=lambda volume: volume.values.min(),
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            highest_volume=dict(
                title="Highest Volume",
                calc_func=lambda volume: volume.values.max(),
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            last_volume=dict(
                title="Last Volume",
                calc_func=lambda volume: generic_nb.ffill_1d_nb(volume.values)[-1],
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_ohlc(
        self,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot OHLC data.

        Args:
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure containing the OHLC plot.

        !!! info
            For default settings, see `vectorbtpro._settings.ohlcv` and `vectorbtpro._settings.plotting`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]
        ohlcv_cfg = settings["ohlcv"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        # Set up figure
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if ohlc_type is None:
            ohlc_type = ohlcv_cfg["ohlc_type"]
        if isinstance(ohlc_type, str):
            if ohlc_type.lower() == "ohlc":
                plot_obj = go.Ohlc
            elif ohlc_type.lower() == "candlestick":
                plot_obj = go.Candlestick
            else:
                raise ValueError("Plot type can be either 'OHLC' or 'Candlestick'")
        else:
            plot_obj = ohlc_type
        def_trace_kwargs = dict(
            x=self.wrapper.index,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            name="OHLC",
            increasing=dict(
                fillcolor=plotting_cfg["color_schema"]["increasing"],
                line=dict(color=plotting_cfg["color_schema"]["increasing"]),
            ),
            decreasing=dict(
                fillcolor=plotting_cfg["color_schema"]["decreasing"],
                line=dict(color=plotting_cfg["color_schema"]["decreasing"]),
            ),
            opacity=0.75,
        )
        if plot_obj is go.Ohlc:
            del def_trace_kwargs["increasing"]["fillcolor"]
            del def_trace_kwargs["decreasing"]["fillcolor"]
        _trace_kwargs = merge_dicts(def_trace_kwargs, trace_kwargs)
        ohlc = plot_obj(**_trace_kwargs)
        fig.add_trace(ohlc, **add_trace_kwargs)
        xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]
        if "rangeslider_visible" not in layout_kwargs.get(xaxis, {}):
            fig.update_layout({xaxis: dict(rangeslider_visible=False)})
        return fig

    def plot_volume(
        self,
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot volume data.

        Args:
            trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Bar`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure with the volume bar plot.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        # Set up figure
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        marker_colors = np.empty(self.volume.shape, dtype=object)
        mask_greater = (self.close.values - self.open.values) > 0
        mask_less = (self.close.values - self.open.values) < 0
        marker_colors[mask_greater] = plotting_cfg["color_schema"]["increasing"]
        marker_colors[mask_less] = plotting_cfg["color_schema"]["decreasing"]
        marker_colors[~(mask_greater | mask_less)] = plotting_cfg["color_schema"]["gray"]
        _trace_kwargs = merge_dicts(
            dict(
                x=self.wrapper.index,
                y=self.volume,
                marker=dict(color=marker_colors, line_width=0),
                opacity=0.5,
                name="Volume",
            ),
            trace_kwargs,
        )
        volume_bar = go.Bar(**_trace_kwargs)
        fig.add_trace(volume_bar, **add_trace_kwargs)
        return fig

    def plot(
        self,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        plot_volume: tp.Optional[bool] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        volume_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        volume_add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot OHLC(V) data using Plotly.

        Args:
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            plot_volume (bool): Whether to plot volume below the OHLC chart.
            ohlc_trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            volume_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Bar` for the volume data.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            volume_add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for the volume trace.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure with the OHLC(V) plot.

        Examples:
            ```pycon
            >>> vbt.YFData.pull("BTC-USD").get().vbt.ohlcv.plot().show()
            ```

            [=100% "100%"]{: .candystripe .candystripe-animate }

            ![](/assets/images/api/ohlcv_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/ohlcv_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure, make_subplots

        if plot_volume is None:
            plot_volume = self.volume is not None
        if plot_volume:
            add_trace_kwargs = merge_dicts(dict(row=1, col=1), add_trace_kwargs)
            volume_add_trace_kwargs = merge_dicts(dict(row=2, col=1), volume_add_trace_kwargs)

        # Set up figure
        if fig is None:
            if plot_volume:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0,
                    row_heights=[0.7, 0.3],
                )
            else:
                fig = make_figure()
            fig.update_layout(
                showlegend=True,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
            )
            if plot_volume:
                fig.update_layout(
                    xaxis2=dict(showgrid=True),
                    yaxis2=dict(showgrid=True),
                )
        fig.update_layout(**layout_kwargs)

        fig = self.plot_ohlc(
            ohlc_type=ohlc_type,
            trace_kwargs=ohlc_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if plot_volume:
            fig = self.plot_volume(
                trace_kwargs=volume_trace_kwargs,
                add_trace_kwargs=volume_add_trace_kwargs,
                fig=fig,
            )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `OHLCVDFAccessor.plots`.

        Merges the defaults from `vectorbtpro.generic.accessors.GenericAccessor.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.ohlcv`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        ohlcv_plots_cfg = settings["ohlcv"]["plots"]

        return merge_dicts(GenericAccessor.plots_defaults.__get__(self), ohlcv_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="OHLC",
                xaxis_kwargs=dict(showgrid=True, rangeslider_visible=False),
                yaxis_kwargs=dict(showgrid=True),
                check_is_not_grouped=True,
                plot_func="plot",
                plot_volume=False,
                tags="ohlcv",
            )
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


OHLCVDFAccessor.override_metrics_doc(__pdoc__)
OHLCVDFAccessor.override_subplots_doc(__pdoc__)
