# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `OHLCSTX` class for generating stop signals based on OHLC data."""

import inspect

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.enums import StopType
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import ohlc_stop_place_nb
from vectorbtpro.utils.config import ReadonlyConfig, merge_dicts

__all__ = [
    "OHLCSTX",
]

__pdoc__ = {}

ohlcstx_config = ReadonlyConfig(
    dict(
        class_name="OHLCSTX",
        module_name=__name__,
        short_name="ohlcstx",
        mode="exits",
        input_names=["entry_price", "open", "high", "low", "close"],
        in_output_names=["stop_price", "stop_type"],
        param_names=["sl_stop", "tsl_th", "tsl_stop", "tp_stop", "reverse"],
        attr_settings=dict(
            entry_price=dict(
                doc="Entry price series.",
            ),
            open=dict(
                doc="Open price series.",
            ),
            high=dict(
                doc="High price series.",
            ),
            low=dict(
                doc="Low price series.",
            ),
            close=dict(
                doc="Close price series.",
            ),
            stop_price=dict(
                doc="Stop price series.",
            ),
            stop_type=dict(
                dtype=StopType,
                doc="Stop type series (see `vectorbtpro.signals.enums.StopType`).",
            ),
        ),
    )
)
"""Factory configuration for the `OHLCSTX` signal generator."""

ohlcstx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=ohlc_stop_place_nb,
        exit_settings=dict(
            pass_inputs=["entry_price", "open", "high", "low", "close"],
            pass_in_outputs=["stop_price", "stop_type"],
            pass_params=["sl_stop", "tsl_th", "tsl_stop", "tp_stop", "reverse"],
            pass_kwargs=["is_entry_open"],
        ),
        in_output_settings=dict(
            stop_price=dict(dtype=float_),
            stop_type=dict(dtype=int_),
        ),
        param_settings=dict(
            sl_stop=merge_dicts(
                flex_elem_param_config,
                dict(
                    doc="Stop loss value, as a scalar or an array.",
                ),
            ),
            tsl_th=merge_dicts(
                flex_elem_param_config,
                dict(
                    doc="Trailing stop threshold value, as a scalar or an array.",
                ),
            ),
            tsl_stop=merge_dicts(
                flex_elem_param_config,
                dict(
                    doc="Trailing stop value, as a scalar or an array.",
                ),
            ),
            tp_stop=merge_dicts(
                flex_elem_param_config,
                dict(
                    doc="Take profit value, as a scalar or an array.",
                ),
            ),
            reverse=merge_dicts(
                flex_elem_param_config,
                dict(
                    doc="Whether to reverse the position, as a scalar or an array.",
                ),
            ),
        ),
        open=np.nan,
        high=np.nan,
        low=np.nan,
        close=np.nan,
        stop_price=np.nan,
        stop_type=-1,
        sl_stop=np.nan,
        tsl_th=np.nan,
        tsl_stop=np.nan,
        tp_stop=np.nan,
        reverse=False,
        is_entry_open=False,
    )
)
"""Exit function configuration for the `OHLCSTX` signal generator."""

OHLCSTX = SignalFactory(**ohlcstx_config).with_place_func(**ohlcstx_func_config)


def _bind_ohlcstx_plot(base_cls: type, entries_attr: str) -> tp.Callable:
    base_cls_plot = base_cls.plot

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        ohlc_kwargs: tp.KwargsLike = None,
        entry_price_kwargs: tp.KwargsLike = None,
        entry_trace_kwargs: tp.KwargsLike = None,
        exit_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        _base_cls_plot: tp.Callable = base_cls_plot,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        self_col = self.select_col(column=column, group_by=False)

        if ohlc_kwargs is None:
            ohlc_kwargs = {}
        if entry_price_kwargs is None:
            entry_price_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        open_any = not self_col.open.isnull().all()
        high_any = not self_col.high.isnull().all()
        low_any = not self_col.low.isnull().all()
        close_any = not self_col.close.isnull().all()
        if open_any and high_any and low_any and close_any:
            ohlc_df = pd.concat(
                (self_col.open, self_col.high, self_col.low, self_col.close), axis=1
            )
            ohlc_df.columns = ["Open", "High", "Low", "Close"]
            ohlc_kwargs = merge_dicts(
                layout_kwargs, dict(ohlc_trace_kwargs=dict(opacity=0.5)), ohlc_kwargs
            )
            fig = ohlc_df.vbt.ohlcv.plot(fig=fig, **ohlc_kwargs)
        else:
            entry_price_kwargs = merge_dicts(layout_kwargs, entry_price_kwargs)
            fig = self_col.entry_price.rename("Entry price").vbt.lineplot(
                fig=fig, **entry_price_kwargs
            )

        _base_cls_plot(
            self_col,
            entry_y=self_col.entry_price,
            exit_y=self_col.stop_price,
            exit_types=self_col.stop_type_readable,
            entry_trace_kwargs=entry_trace_kwargs,
            exit_trace_kwargs=exit_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )
        return fig

    plot.__doc__ = inspect.cleandoc(
        """
        Plot OHLC, `{0}.{1}` and `{0}.exits`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.

                If None, a default column is used.
            ohlc_kwargs (KwargsLike): Keyword arguments for plotting OHLC data using
                `vectorbtpro.ohlcv.accessors.OHLCVDFAccessor.plot`.
            entry_price_kwargs (KwargsLike): Keyword arguments for plotting the entry price line.
            entry_trace_kwargs (KwargsLike): Keyword arguments for
                `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_entries` for `{0}.{1}`.
            exit_trace_kwargs (KwargsLike): Keyword arguments for
                `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_exits` for `{0}.exits`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated or newly created figure.
        """
    ).format(base_cls.__name__, entries_attr)
    if entries_attr == "entries":
        plot.__doc__ += "\n" + inspect.cleandoc(
            """
            Examples:
                ```pycon
                >>> ohlcstx.iloc[:, 0].plot().show()
                ```

                ![](/assets/images/api/OHLCSTX.light.svg#only-light){: .iimg loading=lazy }
                ![](/assets/images/api/OHLCSTX.dark.svg#only-dark){: .iimg loading=lazy }
            """
        )
    return plot


class _OHLCSTX(OHLCSTX):
    """Class representing an exit signal generator based on OHLC data and stop values.

    See:
        * `OHLCSTX.run` for the main entry point.
        * `vectorbtpro.signals.nb.ohlc_stop_place_nb` for details on the exit placement.

    !!! hint
        All parameters may be provided as a single value (per frame) or as a NumPy array
        (per row, column, or element).

        To generate multiple combinations, pass them as lists.

    !!! warning
        The generator checks for an exit after every entry. If two entries occur consecutively,
        no exit signal is generated. Consider cleaning up entry signals before passing them, or use `OHLCSTCX`.

    Examples:
        Test each stop type:

        ```pycon
        >>> from vectorbtpro import *

        >>> entries = pd.Series([True, False, False, False, False, False])
        >>> price = pd.DataFrame({
        ...     'open': [10, 11, 12, 11, 10, 9],
        ...     'high': [11, 12, 13, 12, 11, 10],
        ...     'low': [9, 10, 11, 10, 9, 8],
        ...     'close': [10, 11, 12, 11, 10, 9]
        ... })
        >>> ohlcstx = vbt.OHLCSTX.run(
        ...     entries,
        ...     price['open'],
        ...     price['open'],
        ...     price['high'],
        ...     price['low'],
        ...     price['close'],
        ...     sl_stop=[0.1, np.nan, np.nan, np.nan],
        ...     tsl_th=[np.nan, np.nan, 0.2, np.nan],
        ...     tsl_stop=[np.nan, 0.1, 0.3, np.nan],
        ...     tp_stop=[np.nan, np.nan, np.nan, 0.1],
        ...     is_entry_open=True
        ... )

        >>> ohlcstx.entries
        ohlcstx_sl_stop      0.1    NaN    NaN    NaN
        ohlcstx_tsl_th       NaN    NaN    0.2    NaN
        ohlcstx_tsl_stop     NaN    0.1    0.3    NaN
        ohlcstx_tp_stop      NaN    NaN    NaN    0.1
        0                   True   True   True   True
        1                  False  False  False  False
        2                  False  False  False  False
        3                  False  False  False  False
        4                  False  False  False  False
        5                  False  False  False  False

        >>> ohlcstx.exits
        ohlcstx_sl_stop      0.1    NaN    NaN    NaN
        ohlcstx_tsl_th       NaN    NaN    0.2    NaN
        ohlcstx_tsl_stop     NaN    0.1    0.3    NaN
        ohlcstx_tp_stop      NaN    NaN    NaN    0.1
        0                  False  False  False  False
        1                  False  False  False   True
        2                  False  False  False  False
        3                  False   True  False  False
        4                   True  False   True  False
        5                  False  False  False  False

        >>> ohlcstx.stop_price
        ohlcstx_sl_stop    0.1   NaN  NaN   NaN
        ohlcstx_tsl_th     NaN   NaN  0.2   NaN
        ohlcstx_tsl_stop   NaN   0.1  0.3   NaN
        ohlcstx_tp_stop    NaN   NaN  NaN   0.1
        0                  NaN   NaN  NaN   NaN
        1                  NaN   NaN  NaN  11.0
        2                  NaN   NaN  NaN   NaN
        3                  NaN  11.7  NaN   NaN
        4                  9.0   NaN  9.1   NaN
        5                  NaN   NaN  NaN   NaN

        >>> ohlcstx.stop_type_readable
        ohlcstx_sl_stop     0.1   NaN   NaN   NaN
        ohlcstx_tsl_th      NaN   NaN   0.2   NaN
        ohlcstx_tsl_stop    NaN   0.1   0.3   NaN
        ohlcstx_tp_stop     NaN   NaN   NaN   0.1
        0                  None  None  None  None
        1                  None  None  None    TP
        2                  None  None  None  None
        3                  None   TSL  None  None
        4                    SL  None   TTP  None
        5                  None  None  None  None
        ```
    """

    plot = _bind_ohlcstx_plot(OHLCSTX, "entries")


OHLCSTX.clone_docstring(_OHLCSTX)
OHLCSTX.clone_method(_OHLCSTX.plot)
