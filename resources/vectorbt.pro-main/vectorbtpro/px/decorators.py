# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module containing class decorators for Plotly Express accessors."""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("plotly")

import inspect

import pandas as pd
import plotly.express as px

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.generic.plotting import clean_labels
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.figure import make_figure


def attach_px_methods(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
    """Attach Plotly Express methods to a class.

    This decorator scans the functions defined in `plotly.express` and attaches those
    that accept a `data_frame` parameter or are named `imshow` as methods to the given class.

    Args:
        cls (Type): Class to decorate by adding Plotly Express methods.

    Returns:
        Type: Decorated class with Plotly Express methods attached.

    !!! info
        For default settings, see `vectorbtpro._settings.plotting`.
    """
    for px_func_name, px_func in inspect.getmembers(px, inspect.isfunction):
        if checks.func_accepts_arg(px_func, "data_frame") or px_func_name == "imshow":

            def plot_method(
                self,
                *args,
                _px_func_name: str = px_func_name,
                _px_func: tp.Callable = px_func,
                layout: tp.KwargsLike = None,
                **kwargs,
            ) -> tp.BaseFigure:
                from vectorbtpro._settings import settings

                layout_cfg = settings["plotting"]["layout"]

                layout_kwargs = dict(
                    template=kwargs.pop("template", layout_cfg["template"]),
                    width=kwargs.pop("width", layout_cfg["width"]),
                    height=kwargs.pop("height", layout_cfg["height"]),
                )
                layout = merge_dicts(layout_kwargs, layout)
                # Fix category_orders
                if "color" in kwargs:
                    if isinstance(kwargs["color"], str):
                        if isinstance(self.obj, pd.DataFrame):
                            if kwargs["color"] in self.obj.columns:
                                category_orders = dict()
                                category_orders[kwargs["color"]] = sorted(
                                    self.obj[kwargs["color"]].unique()
                                )
                                kwargs = merge_dicts(dict(category_orders=category_orders), kwargs)

                # Fix Series name
                obj = self.obj.copy(deep=False)
                if isinstance(obj, pd.Series):
                    if obj.name is not None:
                        obj = obj.rename(str(obj.name))
                else:
                    obj.columns = clean_labels(obj.columns)
                obj.index = clean_labels(obj.index)

                if _px_func_name == "imshow":
                    return make_figure(
                        _px_func(to_2d_array(obj), *args, **layout_kwargs, **kwargs), layout=layout
                    )
                return make_figure(_px_func(obj, *args, **layout_kwargs, **kwargs), layout=layout)

            plot_method.__name__ = px_func_name
            plot_method.__module__ = cls.__module__
            plot_method.__qualname__ = f"{cls.__name__}.{plot_method.__name__}"
            plot_method.__doc__ = inspect.cleandoc(
                f"""
                Plot using `{px_func.__module__ + "." + px_func.__name__}`.

                Args:
                    *args: Positional arguments for the Plotly Express function.
                    layout (KwargsLike): Layout configuration overrides.
                    **kwargs: Keyword arguments for the Plotly Express function.

                Returns:
                    BaseFigure: Plotly figure created by the Plotly Express function.
                """
            )
            setattr(cls, px_func_name, plot_method)
    return cls
