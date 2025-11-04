# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for color manipulation, conversion, and adjustment."""

import numpy as np

from vectorbtpro import _typing as tp

__all__ = []


def map_value_to_cmap(
    value: tp.MaybeSequence[float],
    cmap: tp.Any,
    vmin: tp.Optional[float] = None,
    vcenter: tp.Optional[float] = None,
    vmax: tp.Optional[float] = None,
) -> tp.MaybeSequence[str]:
    """Return the RGB color(s) corresponding to the input value(s) according to the given colormap.

    Args:
        value (MaybeSequence[float]): Numeric value or sequence of values to map to colors.
        cmap (Any): Colormap identifier provided as a string name or a collection (list/tuple) of colors.
        vmin (Optional[float]): Minimum data value for colormap normalization.
        vcenter (Optional[float]): Midpoint for two-slope colormap normalization.
        vmax (Optional[float]): Maximum data value for colormap normalization.

    Returns:
        MaybeSequence[str]: Color string in `rgb(r,g,b)` format for each input value.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    value_is_scalar = np.isscalar(value)
    if value_is_scalar:
        value = np.array([value])
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    elif isinstance(cmap, (tuple, list)):
        cmap = mcolors.LinearSegmentedColormap.from_list("", cmap)
    if vmin is not None and vcenter is not None and vmin > vcenter:
        vmin = vcenter
    if vmin is not None and vcenter is not None and vmin == vcenter:
        vcenter = None
    if vmax is not None and vcenter is not None and vmax < vcenter:
        vmax = vcenter
    if vmax is not None and vcenter is not None and vmax == vcenter:
        vcenter = None
    if vcenter is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        value = norm(value)
    elif vmin is not None or vmax is not None:
        if vmin == vmax:
            value = value * 0 + 0.5
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            value = norm(value)
    rgbs = list(
        map(
            lambda x: "rgb(%d,%d,%d)" % (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            cmap(value),
        )
    )
    if value_is_scalar:
        return rgbs[0]
    return rgbs


def parse_rgba_tuple(color: str) -> tp.Tuple[float, float, float, float]:
    """Return a tuple of normalized RGBA components parsed from the provided color string.

    Args:
        color (str): RGBA color string in the format "rgba(r,g,b,a)".

    Returns:
        Tuple[float, float, float, float]: Tuple containing the red, green, and blue components
            normalized to [0, 1] and the alpha value.
    """
    rgba = color.replace("rgba", "").replace("(", "").replace(")", "").split(",")
    return int(rgba[0]) / 255, int(rgba[1]) / 255, int(rgba[2]) / 255, float(rgba[3])


def parse_rgb_tuple(color: str) -> tp.Tuple[float, float, float]:
    """Return a tuple of normalized RGB components parsed from the provided color string.

    Args:
        color (str): RGB color string in the format "rgb(r,g,b)".

    Returns:
        Tuple[float, float, float]: Tuple containing the red, green, and blue components
            normalized to [0, 1].
    """
    rgb = color.replace("rgb", "").replace("(", "").replace(")", "").split(",")
    return int(rgb[0]) / 255, int(rgb[1]) / 255, int(rgb[2]) / 255


def adjust_opacity(color: tp.Any, opacity: float) -> str:
    """Return a color string with the specified opacity adjustment.

    Args:
        color (Any): Color represented as a Matplotlib color string, hex string, or RGB/RGBA tuple.
        opacity (float): Desired opacity value.

    Returns:
        str: Color in `rgba(r,g,b,a)` format with the updated opacity.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import matplotlib.colors as mc

    if isinstance(color, str) and color.startswith("rgba"):
        color = parse_rgba_tuple(color)
    elif isinstance(color, str) and color.startswith("rgb"):
        color = parse_rgb_tuple(color)
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    rgb = mc.to_rgb(c)
    return "rgba(%d,%d,%d,%.4f)" % (
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255),
        opacity,
    )


def adjust_lightness(color: tp.Any, amount: float = 0.7) -> str:
    """Return an RGB color string with adjusted lightness.

    Args:
        color (Any): Color represented as a Matplotlib color string, hex string, or RGB/RGBA tuple.
        amount (float): Factor to adjust the lightness.

            Values less than 1 darken the color, while values greater than 1 lighten it.

    Returns:
        str: Adjusted color in `rgb(r,g,b)` format.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("matplotlib")
    import colorsys

    import matplotlib.colors as mc

    if isinstance(color, str) and color.startswith("rgba"):
        color = parse_rgba_tuple(color)
    elif isinstance(color, str) and color.startswith("rgb"):
        color = parse_rgb_tuple(color)
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    rgb = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return "rgb(%d,%d,%d)" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
