import os
from itertools import permutations

import vectorbtpro as vbt
from vectorbtpro.generic import enums, nb

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


# ############# nb ############# #


class TestPatterns:
    def test_linear_interp_nb(self):
        pattern_arr = np.array([3, 2, 1, 4])
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 1, enums.InterpMode.Linear),
            np.array([3.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 2, enums.InterpMode.Linear),
            np.array([3.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 3, enums.InterpMode.Linear),
            np.array([3.0, 1.5, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 4, enums.InterpMode.Linear),
            np.array([3.0, 2.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 5, enums.InterpMode.Linear),
            np.array([3.0, 2.25, 1.5, 1.75, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 6, enums.InterpMode.Linear),
            np.array([3.0, 2.4, 1.7999999999999998, 1.2000000000000002, 2.200000000000001, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 7, enums.InterpMode.Linear),
            np.array([3.0, 2.5, 2.0, 1.5, 1.0, 2.5, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 8, enums.InterpMode.Linear),
            np.array(
                [
                    3.0,
                    2.5714285714285716,
                    2.142857142857143,
                    1.7142857142857144,
                    1.2857142857142858,
                    1.4285714285714284,
                    2.7142857142857135,
                    4.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 9, enums.InterpMode.Linear),
            np.array([3.0, 2.625, 2.25, 1.875, 1.5, 1.125, 1.75, 2.875, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 10, enums.InterpMode.Linear),
            np.array(
                [
                    3.0,
                    2.6666666666666665,
                    2.3333333333333335,
                    2.0,
                    1.6666666666666667,
                    1.3333333333333333,
                    1.0,
                    2.0000000000000004,
                    2.9999999999999996,
                    4.0,
                ]
            ),
        )

    def test_nearest_interp_nb(self):
        pattern_arr = np.array([3, 2, 1, 4])
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 1, enums.InterpMode.Nearest),
            np.array([3.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 2, enums.InterpMode.Nearest),
            np.array([3.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 3, enums.InterpMode.Nearest),
            np.array([3.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 4, enums.InterpMode.Nearest),
            np.array([3.0, 2.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 5, enums.InterpMode.Nearest),
            np.array([3.0, 2.0, 1.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 6, enums.InterpMode.Nearest),
            np.array([3.0, 2.0, 2.0, 1.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 7, enums.InterpMode.Nearest),
            np.array([3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 8, enums.InterpMode.Nearest),
            np.array([3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 4.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 9, enums.InterpMode.Nearest),
            np.array([3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 4.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 10, enums.InterpMode.Nearest),
            np.array([3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 4.0, 4.0]),
        )

    def test_discrete_interp_nb(self):
        pattern_arr = np.array([3, 2, 1, 4])
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 1, enums.InterpMode.Discrete),
            np.array([3.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 2, enums.InterpMode.Discrete),
            np.array([3.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 3, enums.InterpMode.Discrete),
            np.array([3.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 4, enums.InterpMode.Discrete),
            np.array([3.0, 2.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 5, enums.InterpMode.Discrete),
            np.array([3.0, 2.0, np.nan, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 6, enums.InterpMode.Discrete),
            np.array([3.0, np.nan, 2.0, 1.0, np.nan, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 7, enums.InterpMode.Discrete),
            np.array([3.0, np.nan, 2.0, np.nan, 1.0, np.nan, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 8, enums.InterpMode.Discrete),
            np.array([3.0, np.nan, 2.0, np.nan, np.nan, 1.0, np.nan, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 9, enums.InterpMode.Discrete),
            np.array([3.0, np.nan, np.nan, 2.0, np.nan, 1.0, np.nan, np.nan, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 10, enums.InterpMode.Discrete),
            np.array([3.0, np.nan, np.nan, 2.0, np.nan, np.nan, 1.0, np.nan, np.nan, 4.0]),
        )

    def test_mixed_interp_nb(self):
        pattern_arr = np.array([3, 2, 1, 4])
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 1, enums.InterpMode.Mixed),
            np.array([3.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 2, enums.InterpMode.Mixed),
            np.array([3.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 3, enums.InterpMode.Mixed),
            np.array([3.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 4, enums.InterpMode.Mixed),
            np.array([3.0, 2.0, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 5, enums.InterpMode.Mixed),
            np.array([3.0, 2.0, 1.5, 1.0, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 6, enums.InterpMode.Mixed),
            np.array([3.0, 2.4, 2.0, 1.0, 2.200000000000001, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 7, enums.InterpMode.Mixed),
            np.array([3.0, 2.5, 2.0, 1.5, 1.0, 2.5, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 8, enums.InterpMode.Mixed),
            np.array(
                [
                    3.0,
                    2.5714285714285716,
                    2.0,
                    1.7142857142857144,
                    1.2857142857142858,
                    1.0,
                    2.7142857142857135,
                    4.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 9, enums.InterpMode.Mixed),
            np.array([3.0, 2.625, 2.25, 2.0, 1.5, 1.0, 1.75, 2.875, 4.0]),
        )
        np.testing.assert_array_equal(
            nb.interp_resize_1d_nb(pattern_arr, 10, enums.InterpMode.Mixed),
            np.array(
                [
                    3.0,
                    2.6666666666666665,
                    2.3333333333333335,
                    2.0,
                    1.6666666666666667,
                    1.3333333333333333,
                    1.0,
                    2.0000000000000004,
                    2.9999999999999996,
                    4.0,
                ]
            ),
        )

    def test_pattern_similarity_nb(self):
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(np.array([]), np.array([1, 2, 3])),
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(np.array([1, 2, 3]), np.array([])),
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(np.array([np.nan, np.nan, np.nan]), np.array([1, 2, 3])),
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(np.array([1, 2, 3]), np.array([np.nan, np.nan, np.nan])),
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 3, 1]),
                np.array([1, 2, 3]),
            )
            == 0.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, np.nan]),
                np.array([1, 2, 3, 100]),
                minp=1,
            )
            == 0.5051020408163265
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, np.nan]),
                np.array([1, 2, 3, 100]),
            )
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, 100]),
                np.array([1, 2, 3, np.nan]),
            )
            == 0.4121212121212121
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, np.nan]),
                np.array([1, 2, 3, np.nan]),
                minp=1,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                vmin=0,
            )
            == 0.8
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                vmax=4,
            )
            == 0.8
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                pmin=0,
            )
            == 0.7857142857142858
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                pmax=4,
            )
            == 0.7857142857142856
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                min_pct_change=2,
            )
            == 1.0
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                min_pct_change=3,
            )
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                max_pct_change=2,
            )
            == 1.0
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                max_pct_change=1,
            )
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([0, 2, 3]),
                np.array([1, 2, 3]),
                vmin=1,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 4]),
                np.array([1, 2, 3]),
                vmax=3,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([0, 2, 3]),
                pmin=1,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 4]),
                pmax=3,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 1, 2]),
                np.array([1, 2, 3]),
                max_error=np.array([1.0]),
            )
            == 0.19999999999999996
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 1, 2]),
                np.array([1, 2, 3]),
                max_error=np.array([0.5]),
            )
            == 0.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 1, 2]),
                np.array([1, 2, 3]),
                max_error=np.array([0.5, 0.5, 0.5]),
            )
            == 0.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 1, 2]),
                np.array([1, 2, 3]),
                max_error=np.array([1.0]),
                max_error_as_maxdist=True,
            )
            == 0.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 1, 2]),
                np.array([1, 2, 3]),
                max_error=np.array([2.0]),
                max_error_as_maxdist=True,
            )
            == 0.33333333333333337
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 1, 2]),
                np.array([1, 2, 3]),
                max_error=np.array([1.0, 2.0, 3.0]),
                max_error_as_maxdist=True,
            )
            == 0.5
        )
        assert not np.isnan(
            vbt.nb.pattern_similarity_nb(
                np.array([3, 1, 2]),
                np.array([1, 2, 3]),
                max_error=np.array([2.0]),
                max_error_strict=True,
            )
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(
                np.array([3, 1, 2]),
                np.array([1, 2, 3]),
                max_error=np.array([1.0]),
                max_error_strict=True,
            )
        )
        for arr in permutations(np.array([1, 2, 3, 4, 5])):
            sim = vbt.nb.pattern_similarity_nb(
                np.asarray(arr),
                np.array([1, 2, 3, 4, 5]),
            )
            assert (
                vbt.nb.pattern_similarity_nb(
                    np.asarray(arr),
                    np.array([1, 2, 3, 4, 5]),
                    min_similarity=sim,
                )
                == sim
            )
            assert np.isnan(
                vbt.nb.pattern_similarity_nb(
                    np.asarray(arr),
                    np.array([1, 2, 3, 4, 5]),
                    min_similarity=sim + 0.1,
                )
            )
        for arr in permutations(np.array([1, 2, 3, 4, np.nan])):
            sim = vbt.nb.pattern_similarity_nb(
                np.asarray(arr),
                np.array([1, 2, 3, 4, 5]),
                minp=1,
            )
            assert (
                vbt.nb.pattern_similarity_nb(
                    np.asarray(arr),
                    np.array([1, 2, 3, 4, 5]),
                    min_similarity=sim,
                    minp=1,
                )
                == sim
            )
            assert (
                vbt.nb.pattern_similarity_nb(
                    np.asarray(arr),
                    np.array([1, 2, 3, 4, 5]),
                    min_similarity=sim - 0.1,
                    minp=1,
                )
                == sim
            )
        for arr in permutations(np.array([1, 2, 3, 4, 5])):
            sim = vbt.nb.pattern_similarity_nb(
                np.asarray(arr),
                np.array([1, 2, 3, 4, 5]),
                max_error=np.array([2.0]),
            )
            assert (
                vbt.nb.pattern_similarity_nb(
                    np.asarray(arr),
                    np.array([1, 2, 3, 4, 5]),
                    max_error=np.array([2.0]),
                    min_similarity=sim,
                )
                == sim
            )
            assert np.isnan(
                vbt.nb.pattern_similarity_nb(
                    np.asarray(arr),
                    np.array([1, 2, 3, 4, 5]),
                    max_error=np.array([2.0]),
                    min_similarity=sim + 0.1,
                )
            )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Rebase,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([3, 2, 1]),
                rescale_mode=enums.RescaleMode.Rebase,
            )
            == 0.4285714285714286
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Rebase,
                max_error=np.array([1.0]),
            )
            == 0.368421052631579
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Rebase,
                max_error=np.array([0.5]),
            )
            == 0.3157894736842105
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Rebase,
                max_error=np.array([1.0]),
                max_error_as_maxdist=True,
            )
            == 0.33333333333333337
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Rebase,
                max_error=np.array([0.5]),
                max_error_as_maxdist=True,
            )
            == 0.16666666666666663
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Rebase,
                max_error=np.array([0.5]),
                max_error_strict=True,
            )
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Rebase,
                max_error=np.array([0.5]),
                min_similarity=0.31,
            )
            == 0.3157894736842105
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Rebase,
                max_error=np.array([0.5]),
                min_similarity=0.32,
            )
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Disable,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Disable,
                max_error=np.array([1.0]),
            )
            == 0.19999999999999996
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Disable,
                max_error=np.array([0.5]),
            )
            == 0.19999999999999996
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([3, 2, 1]),
                rescale_mode=enums.RescaleMode.Disable,
                max_error=np.array([0.0]),
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Disable,
                max_error=np.array([1.0]),
                max_error_as_maxdist=True,
            )
            == 0.33333333333333337
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                rescale_mode=enums.RescaleMode.Disable,
                max_error=np.array([0.5]),
                max_error_as_maxdist=True,
            )
            == 0.33333333333333337
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([3, 2, 1]),
                rescale_mode=enums.RescaleMode.Disable,
                max_error=np.array([0.0]),
                max_error_as_maxdist=True,
            )
            == 1.0
        )
        assert np.isnan(
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([3, 2, 2]),
                rescale_mode=enums.RescaleMode.Disable,
                max_error=np.array([0.0]),
                max_error_as_maxdist=True,
            )
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3]),
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3, 4, 5]),
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3]),
                interp_mode=enums.InterpMode.Nearest,
            )
            == 0.8888888888888888
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3, 4, 5]),
                interp_mode=enums.InterpMode.Nearest,
            )
            == 0.875
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3]),
                interp_mode=enums.InterpMode.Discrete,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3, 4, 5]),
                interp_mode=enums.InterpMode.Discrete,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.MAE,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 3, 1]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.MAE,
            )
            == 0.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.MAE,
            )
            == 0.19999999999999996
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.MSE,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 3, 1]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.MSE,
            )
            == 0.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.MSE,
            )
            == 0.11111111111111116
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.RMSE,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 3, 1]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.RMSE,
            )
            == 0.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                distance_measure=enums.DistanceMeasure.RMSE,
            )
            == 0.05719095841793653
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                error_type=enums.ErrorType.Relative,
            )
            == 1.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 3, 1]),
                np.array([1, 2, 3]),
                error_type=enums.ErrorType.Relative,
            )
            == 0.0
        )
        assert (
            vbt.nb.pattern_similarity_nb(
                np.array([3, 2, 1]),
                np.array([1, 2, 3]),
                error_type=enums.ErrorType.Relative,
            )
            == 0.1578947368421053
        )
        for arr in permutations(np.array([1, 2, 3, 4, 5])):
            assert vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, 4, 5]),
                np.asarray(arr),
                invert=True,
            ) == vbt.nb.pattern_similarity_nb(
                np.array([1, 2, 3, 4, 5]),
                np.max(arr) + np.min(arr) - np.asarray(arr),
            )
