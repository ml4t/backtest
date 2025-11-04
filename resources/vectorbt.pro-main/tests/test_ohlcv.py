import os

import pytest
import vectorbtpro as vbt

from tests.utils import *

ohlcv_df = pd.DataFrame(
    {
        "open": [1, 2, 3, 4, 5],
        "high": [2.5, 3.5, 4.5, 5.5, 6.5],
        "low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "close": [2, 3, 4, 5, 6],
        "volume": [1, 2, 3, 2, 1],
    },
    index=pd.date_range("2020", periods=5),
)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.chunking["n_chunks"] = 2


def teardown_module():
    vbt.settings.reset()


# ############# accessors ############# #


class TestAccessors:
    def test_mirror(self):
        assert_frame_equal(
            ohlcv_df.vbt.ohlcv.mirror_ohlc(),
            pd.DataFrame(
                [
                    [1.0, 2.5, 0.5, 2.0, 1],
                    [0.5, 0.6666666666666666, 0.2857142857142857, 0.3333333333333333, 2],
                    [0.33333333333333337, 0.4, 0.22222222222222224, 0.25, 3],
                    [
                        0.25000000000000006,
                        0.28571428571428575,
                        0.18181818181818185,
                        0.20000000000000007,
                        2,
                    ],
                    [
                        0.20000000000000007,
                        0.2222222222222223,
                        0.1538461538461539,
                        0.16666666666666674,
                        1,
                    ],
                ],
                index=ohlcv_df.index,
                columns=ohlcv_df.columns,
            ),
        )
        assert_frame_equal(
            ohlcv_df.vbt.ohlcv.mirror_ohlc(start_value=100),
            pd.DataFrame(
                [
                    [100.0, 250.0, 50.0, 200.0, 1],
                    [50.0, 66.66666666666666, 28.57142857142857, 33.33333333333333, 2],
                    [33.333333333333336, 40.0, 22.22222222222222, 25.0, 3],
                    [
                        25.000000000000007,
                        28.571428571428577,
                        18.181818181818187,
                        20.000000000000007,
                        2,
                    ],
                    [
                        20.000000000000007,
                        22.222222222222232,
                        15.38461538461539,
                        16.666666666666675,
                        1,
                    ],
                ],
                index=ohlcv_df.index,
                columns=ohlcv_df.columns,
            ),
        )
        assert_frame_equal(
            ohlcv_df.vbt.ohlcv.mirror_ohlc(ref_feature="close"),
            pd.DataFrame(
                [
                    [1.0, 2.5, 0.5, 2.0, 1],
                    [2.0, 2.6666666666666665, 1.1428571428571428, 1.3333333333333333, 2],
                    [
                        1.3333333333333335,
                        1.6000000000000005,
                        0.8888888888888891,
                        1.0000000000000002,
                        3,
                    ],
                    [
                        1.0000000000000002,
                        1.142857142857143,
                        0.7272727272727274,
                        0.8000000000000002,
                        2,
                    ],
                    [0.8, 0.888888888888889, 0.6153846153846155, 0.6666666666666667, 1],
                ],
                index=ohlcv_df.index,
                columns=ohlcv_df.columns,
            ),
        )

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        assert_frame_equal(
            ohlcv_df.vbt.ohlcv.resample(test_freq).obj,
            ohlcv_df.resample(test_freq).agg(
                {
                    "open": lambda x: float(x[0] if len(x) > 0 else np.nan),
                    "high": lambda x: float(x.max() if len(x) > 0 else np.nan),
                    "low": lambda x: float(x.min() if len(x) > 0 else np.nan),
                    "close": lambda x: float(x[-1] if len(x) > 0 else np.nan),
                    "volume": lambda x: float(x.sum() if len(x) > 0 else np.nan),
                }
            ),
        )
