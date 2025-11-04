# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `OLS` class for rolling Ordinary Least Squares regression analysis."""


from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "OLS",
]

__pdoc__ = {}

OLS = IndicatorFactory(
    class_name="OLS",
    module_name=__name__,
    short_name="ols",
    input_names=["x", "y"],
    param_names=["window", "norm_window"],
    output_names=["slope", "intercept", "zscore"],
    lazy_outputs=dict(
        pred=lambda self: self.wrapper.wrap(
            nb.ols_pred_nb(
                to_2d_array(self.x),
                to_2d_array(self.slope),
                to_2d_array(self.intercept),
            ),
        ),
        error=lambda self: self.wrapper.wrap(
            nb.ols_error_nb(
                to_2d_array(self.y),
                to_2d_array(self.pred),
            ),
        ),
        angle=lambda self: self.wrapper.wrap(
            nb.ols_angle_nb(
                to_2d_array(self.slope),
            ),
        ),
    ),
    attr_settings=dict(
        x=dict(
            doc="Independent variable series.",
        ),
        y=dict(
            doc="Dependent variable series.",
        ),
        slope=dict(
            doc="Slope of the regression line.",
        ),
        intercept=dict(
            doc="Intercept of the regression line.",
        ),
        zscore=dict(
            doc="Z-score of the regression line.",
        ),
        pred=dict(
            doc="Predicted values based on the regression line.",
        ),
        error=dict(
            doc="Error between the actual and predicted values.",
        ),
        angle=dict(
            doc="Angle of the regression line in radians.",
        ),
    ),
).with_apply_func(
    nb.ols_nb,
    kwargs_as_args=["minp", "ddof", "with_zscore"],
    param_settings=dict(
        window=dict(
            doc="Window size.",
        ),
        norm_window=dict(
            doc="Normalization window size.",
        ),
    ),
    window=14,
    norm_window=None,
    minp=None,
    ddof=0,
    with_zscore=True,
)


class _OLS(OLS):
    """Class representing the Rolling Ordinary Least Squares (OLS) indicator.

    The `OLS` indicator is used to detect changes in the relationship between stocks and
    the market or between different stocks by computing rolling linear regressions.

    See:
        * `OLS.run` for the main entry point.
        * `vectorbtpro.indicators.nb.ols_nb` for the underlying implementation.
        * `vectorbtpro.indicators.nb.ols_pred_nb` for the underlying implementation of
            the `OLS.pred` property.
        * `vectorbtpro.indicators.nb.ols_error_nb` for the underlying implementation of
            the `OLS.error` property.
        * `vectorbtpro.indicators.nb.ols_angle_nb` for the underlying implementation of
            the `OLS.angle` property.
        * https://www.investopedia.com/articles/trading/09/linear-regression-time-price.asp
            for the definition of OLS.
    """

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_y: bool = True,
        y_trace_kwargs: tp.KwargsLike = None,
        pred_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the `OLS.pred` and (optionally) `OLS.y` values.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_y (bool): Whether to plot `OLS.y`.
            y_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `OLS.y`.
            pred_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `OLS.pred`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure with plotted OLS predictions and actual values.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.OLS.run(np.arange(len(ohlcv)), ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/OLS.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/OLS.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if y_trace_kwargs is None:
            y_trace_kwargs = {}
        if pred_trace_kwargs is None:
            pred_trace_kwargs = {}
        y_trace_kwargs = merge_dicts(
            dict(name="Y", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            y_trace_kwargs,
        )
        pred_trace_kwargs = merge_dicts(
            dict(name="Pred", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            pred_trace_kwargs,
        )

        if plot_y:
            fig = self_col.y.vbt.lineplot(
                trace_kwargs=y_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.pred.vbt.lineplot(
            trace_kwargs=pred_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig

    def plot_zscore(
        self,
        column: tp.Optional[tp.Column] = None,
        alpha: float = 0.05,
        zscore_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the `OLS.zscore` values with confidence intervals.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            alpha (float): Alpha level for the confidence interval.

                The default alpha value of 0.05 returns a 95% confidence interval.
            zscore_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for `OLS.zscore`.
            add_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for each shape.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure with plotted OLS z-score and confidence intervals.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.OLS.run(np.arange(len(ohlcv)), ohlcv['Close']).plot_zscore().show()
            ```

            ![](/assets/images/api/OLS_zscore.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/OLS_zscore.dark.svg#only-dark){: .iimg loading=lazy }
        """
        import scipy.stats as st

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        zscore_trace_kwargs = merge_dicts(
            dict(name="Z-score", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            zscore_trace_kwargs,
        )
        fig = self_col.zscore.vbt.lineplot(
            trace_kwargs=zscore_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        # Fill void between limits
        xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]
        add_shape_kwargs = merge_dicts(
            dict(
                type="rect",
                xref=xref,
                yref=yref,
                x0=self_col.wrapper.index[0],
                y0=st.norm.ppf(1 - alpha / 2),
                x1=self_col.wrapper.index[-1],
                y1=st.norm.ppf(alpha / 2),
                fillcolor="mediumslateblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
            add_shape_kwargs,
        )
        fig.add_shape(**add_shape_kwargs)

        return fig


OLS.clone_docstring(_OLS)
OLS.clone_method(_OLS.plot)
OLS.clone_method(_OLS.plot_zscore)
