# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utility functions for image processing."""

from pathlib import Path

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.pbar import ProgressBar

__all__ = [
    "save_animation",
]


def hstack_image_arrays(a: tp.Array3d, b: tp.Array3d) -> tp.Array3d:
    """Horizontally stack two 3D NumPy image arrays with a white background fill.

    Args:
        a (Array3d): Left image array.
        b (Array3d): Right image array.

    Returns:
        Array3d: Combined image with `a` on the left and `b` on the right.
    """
    h1, w1, d = a.shape
    h2, w2, _ = b.shape
    c = np.full((max(h1, h2), w1 + w2, d), 255, np.uint8)
    c[:h1, :w1, :] = a
    c[:h2, w1 : w1 + w2, :] = b
    return c


def vstack_image_arrays(a: tp.Array3d, b: tp.Array3d) -> tp.Array3d:
    """Vertically stack two 3D NumPy image arrays with a white background fill.

    Args:
        a (Array3d): Top image array.
        b (Array3d): Bottom image array.

    Returns:
        Array3d: Combined image with `a` at the top and `b` at the bottom.
    """
    h1, w1, d = a.shape
    h2, w2, _ = b.shape
    c = np.full((h1 + h2, max(w1, w2), d), 255, np.uint8)
    c[:h1, :w1, :] = a
    c[h1 : h1 + h2, :w2, :] = b
    return c


def save_animation(
    fname: tp.PathLike,
    index: tp.Sequence,
    plot_func: tp.Callable,
    *args,
    delta: tp.Optional[int] = None,
    step: int = 1,
    fps: int = 3,
    writer_kwargs: tp.KwargsLike = None,
    show_progress: bool = True,
    pbar_kwargs: tp.KwargsLike = None,
    to_image_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Path:
    """Save an animation to a file by iterating over a provided index.

    Args:
        fname (PathLike): File name or path to save the animation.
        index (Sequence): Iterable index for generating frames.
        plot_func (Callable): Plotting function that accepts a slice of `index`, additional
            positional arguments, and keyword arguments, and returns either a Plotly figure, an image file
            path (readable by `imageio.imread`), or a NumPy array representing an image.
        *args: Positional arguments for `plot_func`.
        delta (Optional[int]): Window size for each iteration.

            Defaults to half the length of `index` if None.
        step (int): Step size between iterations.
        fps (int): Frames per second for the animation.

            Internally converted to a frame duration using `1000 / fps`.
        writer_kwargs (KwargsLike): Keyword arguments for `imageio.get_writer`.
        show_progress (bool): Flag indicating whether to display the progress bar.
        pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

            See `vectorbtpro.utils.pbar.ProgressBar`.
        to_image_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Figure.to_image`.
        **kwargs: Keyword arguments for `plot_func`.

    Returns:
        Path: Path to the saved animation file.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> def plot_data_window(index, data):
        ...     return data.loc[index].plot()

        >>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")
        >>> vbt.save_animation(
        ...     "plot_data_window.gif",
        ...     data.index,
        ...     plot_data_window,
        ...     data,
        ...     delta=90,
        ...     step=10
        ... )
        ```
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("plotly")
    import imageio
    import plotly.graph_objects as go

    if isinstance(fname, str):
        fname = Path(fname)
    if writer_kwargs is None:
        writer_kwargs = {}
    if "duration" not in writer_kwargs:
        writer_kwargs["duration"] = 1000 / fps
    if pbar_kwargs is None:
        pbar_kwargs = {}
    if "bar_id" not in pbar_kwargs:
        pbar_kwargs["bar_id"] = "save_animation"
    if to_image_kwargs is None:
        to_image_kwargs = {}
    if delta is None:
        delta = len(index) // 2

    with imageio.get_writer(fname, **writer_kwargs) as writer:
        index_steps = range(0, len(index) - delta + 1, step)
        with ProgressBar(index_steps, show_progress=show_progress, **pbar_kwargs) as pbar:
            pbar.set_description(f"{str(index[0])} → {str(index[0 + delta - 1])}")

            for i in range(len(index_steps)):
                j = index_steps[i]
                fig = plot_func(index[j : j + delta], *args, **kwargs)
                if fig is None:
                    continue
                if isinstance(fig, (go.Figure, go.FigureWidget)):
                    fig = fig.to_image(format="png", **to_image_kwargs)
                if not isinstance(fig, np.ndarray):
                    fig = imageio.imread(fig)
                writer.append_data(fig)

                if i + 1 < len(index_steps):
                    next_j = index_steps[i + 1]
                    pbar.set_description(
                        f"{str(index[next_j])} → {str(index[next_j + delta - 1])}"
                    )
                pbar.update()

    return fname
