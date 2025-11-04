import os

import pytest
import vectorbtpro as vbt
from vectorbtpro.base import resampling

from tests.utils import *

# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.broadcasting["index_from"] = "stack"
    vbt.settings.broadcasting["columns_from"] = "stack"


def teardown_module():
    vbt.settings.reset()


# ############# resampling ############# #


class TestResampler:
    @pytest.mark.parametrize("test_freq", ["1h", "3d", "7d"])
    @pytest.mark.parametrize("test_inclusive", ["left", "right"])
    def test_date_range_nb(self, test_freq, test_inclusive):
        source_index = pd.date_range("2020-01-01", "2020-02-01")
        np.testing.assert_array_equal(
            resampling.nb.date_range_nb(
                source_index[0].to_datetime64(),
                source_index[-1].to_datetime64(),
                pd.Timedelta(test_freq).to_timedelta64(),
                incl_left=test_inclusive == "left",
                incl_right=test_inclusive == "right",
            ),
            pd.date_range(
                source_index[0], source_index[-1], freq=test_freq, inclusive=test_inclusive
            ).values,
        )

    def test_from_pd_resample(self):
        source_index = pd.date_range("2020-01-01", "2020-02-01", freq="1h")
        resampler = vbt.Resampler.from_pd_resample(source_index, "1d")
        target_index = pd.Series(index=source_index).resample("1d").count().index
        assert_index_equal(resampler.source_index, source_index)
        assert_index_equal(resampler.target_index, target_index)
        assert resampler.source_freq == source_index.freq
        assert resampler.target_freq == target_index.freq

    def test_from_date_range(self):
        source_index = pd.date_range("2020-01-01", "2020-02-01", freq="1h", inclusive="left")
        resampler = vbt.Resampler.from_date_range(
            source_index, "2020-01-01", "2020-02-01", freq="1d"
        )
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="1d", inclusive="left")
        assert_index_equal(resampler.source_index, source_index)
        assert_index_equal(resampler.target_index, target_index)
        assert resampler.source_freq == source_index.freq
        assert resampler.target_freq == target_index.freq

    def test_downsample_map_to_target_index(self):
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="7d")
        resampler = vbt.Resampler(source_index, target_index)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True),
            pd.DatetimeIndex(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-15",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-06", "2020-02-01", freq="7d")
        resampler = vbt.Resampler(source_index, target_index)
        with pytest.raises(Exception):
            resampler.map_to_target_index(return_index=False)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, raise_missing=False),
            np.array([-1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, raise_missing=False),
            pd.DatetimeIndex(
                [
                    pd.NaT,
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-13",
                    "2020-01-13",
                    "2020-01-13",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="7d")
        resampler = vbt.Resampler(source_index, target_index)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, before=True),
            np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, before=True),
            pd.DatetimeIndex(
                [
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                    "2020-01-15",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-01-14", freq="7d")
        resampler = vbt.Resampler(source_index, target_index)
        with pytest.raises(Exception):
            resampler.map_to_target_index(return_index=False, before=True)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, raise_missing=False, before=True),
            np.array([1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, raise_missing=False, before=True),
            pd.DatetimeIndex(
                [
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    "2020-01-08",
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )

    def test_upsample_map_to_target_index(self):
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="10h")
        resampler = vbt.Resampler(source_index, target_index)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False),
            np.array([9, 12, 14, 16, 19, 21, 24, 26, 28, 31, 33]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True),
            pd.DatetimeIndex(
                [
                    "2020-01-04 18:00:00",
                    "2020-01-06 00:00:00",
                    "2020-01-06 20:00:00",
                    "2020-01-07 16:00:00",
                    "2020-01-08 22:00:00",
                    "2020-01-09 18:00:00",
                    "2020-01-11 00:00:00",
                    "2020-01-11 20:00:00",
                    "2020-01-12 16:00:00",
                    "2020-01-13 22:00:00",
                    "2020-01-14 18:00:00",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-06", "2020-02-01", freq="10h")
        resampler = vbt.Resampler(source_index, target_index)
        with pytest.raises(Exception):
            resampler.map_to_target_index(return_index=False)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, raise_missing=False),
            np.array([-1, 0, 2, 4, 7, 9, 12, 14, 16, 19, 21]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, raise_missing=False),
            pd.DatetimeIndex(
                [
                    pd.NaT,
                    "2020-01-06 00:00:00",
                    "2020-01-06 20:00:00",
                    "2020-01-07 16:00:00",
                    "2020-01-08 22:00:00",
                    "2020-01-09 18:00:00",
                    "2020-01-11 00:00:00",
                    "2020-01-11 20:00:00",
                    "2020-01-12 16:00:00",
                    "2020-01-13 22:00:00",
                    "2020-01-14 18:00:00",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-02-01", freq="10h")
        resampler = vbt.Resampler(source_index, target_index)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, before=True),
            np.array([10, 12, 15, 17, 20, 22, 24, 27, 29, 32, 34]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, before=True),
            pd.DatetimeIndex(
                [
                    "2020-01-05 04:00:00",
                    "2020-01-06 00:00:00",
                    "2020-01-07 06:00:00",
                    "2020-01-08 02:00:00",
                    "2020-01-09 08:00:00",
                    "2020-01-10 04:00:00",
                    "2020-01-11 00:00:00",
                    "2020-01-12 06:00:00",
                    "2020-01-13 02:00:00",
                    "2020-01-14 08:00:00",
                    "2020-01-15 04:00:00",
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        source_index = pd.date_range("2020-01-05", "2020-01-15", freq="1d")
        target_index = pd.date_range("2020-01-01", "2020-01-14", freq="10h")
        resampler = vbt.Resampler(source_index, target_index)
        with pytest.raises(Exception):
            resampler.map_to_target_index(return_index=False, before=True)
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=False, raise_missing=False, before=True),
            np.array([10, 12, 15, 17, 20, 22, 24, 27, 29, -1, -1]),
        )
        np.testing.assert_array_equal(
            resampler.map_to_target_index(return_index=True, raise_missing=False, before=True),
            pd.DatetimeIndex(
                [
                    "2020-01-05 04:00:00",
                    "2020-01-06 00:00:00",
                    "2020-01-07 06:00:00",
                    "2020-01-08 02:00:00",
                    "2020-01-09 08:00:00",
                    "2020-01-10 04:00:00",
                    "2020-01-11 00:00:00",
                    "2020-01-12 06:00:00",
                    "2020-01-13 02:00:00",
                    pd.NaT,
                    pd.NaT,
                ],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )

    def test_index_difference(self):
        source_index = ["2020-01-01", "2020-01-02", "2020-01-03"]
        target_index = [
            "2020-01-01T12:00:00",
            "2020-01-02T00:00:00",
            "2020-01-03T00:00:00",
            "2020-01-03T12:00:00",
        ]
        resampler = vbt.Resampler(source_index, target_index)
        assert_index_equal(
            resampler.index_difference(),
            pd.DatetimeIndex(["2020-01-01 12:00:00"], dtype="datetime64[ns]", freq=None),
        )
        assert_index_equal(
            resampler.index_difference(reverse=True),
            pd.DatetimeIndex(
                ["2020-01-01 12:00:00", "2020-01-03 12:00:00"], dtype="datetime64[ns]", freq=None
            ),
        )
