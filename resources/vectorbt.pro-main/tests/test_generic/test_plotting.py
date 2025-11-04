import os

import vectorbtpro as vbt

from tests.utils import *

# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.chunking["n_chunks"] = 2


def teardown_module():
    vbt.settings.reset()


# ############# plotting ############# #


class TestPlotting:
    def test_gauge(self):
        vbt.Gauge(
            value=2,
            value_range=(1, 3),
            label="My Gauge",
            make_figure_kwargs=dict(
                use_widgets=False,
                use_resampler=False,
            ),
        )
        vbt.Gauge(
            value=2,
            value_range=(1, 3),
            label="My Gauge",
            make_figure_kwargs=dict(
                use_widgets=True,
                use_resampler=False,
            ),
        )

    def test_bar(self):
        vbt.Bar(
            data=[[1, 2], [3, 4]],
            trace_names=["a", "b"],
            x_labels=["x", "y"],
            make_figure_kwargs=dict(
                use_widgets=False,
                use_resampler=False,
            ),
        )
        vbt.Bar(
            data=[[1, 2], [3, 4]],
            trace_names=["a", "b"],
            x_labels=["x", "y"],
            make_figure_kwargs=dict(
                use_widgets=True,
                use_resampler=False,
            ),
        )

    def test_scatter(self):
        vbt.Scatter(
            data=[[1, 2], [3, 4]],
            trace_names=["a", "b"],
            x_labels=["x", "y"],
            make_figure_kwargs=dict(
                use_widgets=False,
                use_resampler=False,
            ),
        )
        vbt.Scatter(
            data=[[1, 2], [3, 4]],
            trace_names=["a", "b"],
            x_labels=["x", "y"],
            make_figure_kwargs=dict(
                use_widgets=True,
                use_resampler=False,
            ),
        )

    def test_histogram(self):
        vbt.Histogram(
            data=[[1, 2], [3, 4], [2, 1]],
            trace_names=["a", "b"],
            make_figure_kwargs=dict(
                use_widgets=False,
                use_resampler=False,
            ),
        )
        vbt.Histogram(
            data=[[1, 2], [3, 4], [2, 1]],
            trace_names=["a", "b"],
            make_figure_kwargs=dict(
                use_widgets=True,
                use_resampler=False,
            ),
        )

    def test_box(self):
        vbt.Box(
            data=[[1, 2], [3, 4], [2, 1]],
            trace_names=["a", "b"],
            make_figure_kwargs=dict(
                use_widgets=False,
                use_resampler=False,
            ),
        )
        vbt.Box(
            data=[[1, 2], [3, 4], [2, 1]],
            trace_names=["a", "b"],
            make_figure_kwargs=dict(
                use_widgets=True,
                use_resampler=False,
            ),
        )

    def test_heatmap(self):
        vbt.Heatmap(
            data=[[1, 2], [3, 4]],
            x_labels=["a", "b"],
            y_labels=["x", "y"],
            make_figure_kwargs=dict(
                use_widgets=False,
                use_resampler=False,
            ),
        )
        vbt.Heatmap(
            data=[[1, 2], [3, 4]],
            x_labels=["a", "b"],
            y_labels=["x", "y"],
            make_figure_kwargs=dict(
                use_widgets=True,
                use_resampler=False,
            ),
        )

    def test_volume(self):
        vbt.Volume(
            data=np.random.randint(1, 10, size=(3, 3, 3)),
            x_labels=["a", "b", "c"],
            y_labels=["d", "e", "f"],
            z_labels=["g", "h", "i"],
            make_figure_kwargs=dict(
                use_widgets=False,
                use_resampler=False,
            ),
        )
        vbt.Volume(
            data=np.random.randint(1, 10, size=(3, 3, 3)),
            x_labels=["a", "b", "c"],
            y_labels=["d", "e", "f"],
            z_labels=["g", "h", "i"],
            make_figure_kwargs=dict(
                use_widgets=True,
                use_resampler=False,
            ),
        )
