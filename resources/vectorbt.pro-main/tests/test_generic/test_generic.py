import os

import pytest
import vectorbtpro as vbt
from numba import njit
from vectorbtpro._dtypes import *
from vectorbtpro.generic import nb

from tests.utils import *

seed = 42

day_dt = np.timedelta64(86400000000000)

df = pd.DataFrame(
    {"a": [1, 2, 3, 4, np.nan], "b": [np.nan, 4, 3, 2, 1], "c": [1, 2, np.nan, 2, 1]},
    index=pd.date_range("2018", periods=5),
)
group_by = np.array(["g1", "g1", "g2"])


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
    def test_indexing(self):
        assert df.vbt["a"].min() == df["a"].vbt.min()

    def test_set_by_mask(self):
        np.testing.assert_array_equal(
            nb.set_by_mask_1d_nb(
                np.array([1, 2, 3, 1, 2, 3]), np.array([True, False, False, True, False, False]), 0
            ),
            np.array([0, 2, 3, 0, 2, 3]),
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_1d_nb(
                np.array([1, 2, 3, 1, 2, 3]),
                np.array([True, False, False, True, False, False]),
                0.0,
            ),
            np.array([0.0, 2.0, 3.0, 0.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_nb(
                np.array([1, 2, 3, 1, 2, 3])[:, None],
                np.array([True, False, False, True, False, False])[:, None],
                0,
            ),
            np.array([0, 2, 3, 0, 2, 3])[:, None],
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_nb(
                np.array([1, 2, 3, 1, 2, 3])[:, None],
                np.array([True, False, False, True, False, False])[:, None],
                0.0,
            ),
            np.array([0.0, 2.0, 3.0, 0.0, 2.0, 3.0])[:, None],
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_mult_1d_nb(
                np.array([1, 2, 3, 1, 2, 3]),
                np.array([True, False, False, True, False, False]),
                np.array([0, -1, -1, 0, -1, -1]),
            ),
            np.array([0, 2, 3, 0, 2, 3]),
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_mult_1d_nb(
                np.array([1, 2, 3, 1, 2, 3]),
                np.array([True, False, False, True, False, False]),
                np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0]),
            ),
            np.array([0.0, 2.0, 3.0, 0.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_mult_nb(
                np.array([1, 2, 3, 1, 2, 3])[:, None],
                np.array([True, False, False, True, False, False])[:, None],
                np.array([0, -1, -1, 0, -1, -1])[:, None],
            ),
            np.array([0, 2, 3, 0, 2, 3])[:, None],
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_mult_nb(
                np.array([1, 2, 3, 1, 2, 3])[:, None],
                np.array([True, False, False, True, False, False])[:, None],
                np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0])[:, None],
            ),
            np.array([0.0, 2.0, 3.0, 0.0, 2.0, 3.0])[:, None],
        )

    def test_shuffle(self):
        assert_series_equal(
            df["a"].vbt.shuffle(seed=seed),
            pd.Series(
                np.array([2.0, np.nan, 3.0, 1.0, 4.0]), index=df["a"].index, name=df["a"].name
            ),
        )
        np.testing.assert_array_equal(
            df["a"].vbt.shuffle(seed=seed).values,
            nb.shuffle_1d_nb(df["a"].values, seed=seed),
        )
        assert_frame_equal(
            df.vbt.shuffle(seed=seed),
            pd.DataFrame(
                np.array(
                    [
                        [2.0, 2.0, 2.0],
                        [np.nan, 4.0, 1.0],
                        [3.0, 3.0, 2.0],
                        [1.0, np.nan, 1.0],
                        [4.0, 1.0, np.nan],
                    ],
                ),
                index=df.index,
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize(
        "test_value",
        [-1, 0.0, np.nan],
    )
    def test_fillna(self, test_value):
        assert_series_equal(df["a"].vbt.fillna(test_value), df["a"].fillna(test_value))
        assert_frame_equal(df.vbt.fillna(test_value), df.fillna(test_value))
        assert_series_equal(pd.Series([1, 2, 3]).vbt.fillna(-1), pd.Series([1, 2, 3]))
        assert_series_equal(
            pd.Series([False, True, False]).vbt.fillna(False),
            pd.Series([False, True, False]),
        )
        assert_frame_equal(
            df.vbt.fillna(test_value, chunked=True), df.vbt.fillna(test_value, chunked=False)
        )

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_bshift(self, test_n):
        assert_series_equal(df["a"].vbt.bshift(test_n), df["a"].shift(-test_n))
        np.testing.assert_array_equal(
            df["a"].vbt.bshift(test_n).values, nb.bshift_1d_nb(df["a"].values, test_n)
        )
        assert_frame_equal(df.vbt.bshift(test_n), df.shift(-test_n))
        assert_series_equal(
            pd.Series([1, 2, 3]).vbt.bshift(1, fill_value=-1), pd.Series([2, 3, -1])
        )
        assert_series_equal(
            pd.Series([True, True, True]).vbt.bshift(1, fill_value=False),
            pd.Series([True, True, False]),
        )
        assert_frame_equal(
            df.vbt.bshift(test_n, chunked=True), df.vbt.bshift(test_n, chunked=False)
        )

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_fshift(self, test_n):
        assert_series_equal(df["a"].vbt.fshift(test_n), df["a"].shift(test_n))
        np.testing.assert_array_equal(
            df["a"].vbt.fshift(test_n).values, nb.fshift_1d_nb(df["a"].values, test_n)
        )
        assert_frame_equal(df.vbt.fshift(test_n), df.shift(test_n))
        assert_series_equal(
            pd.Series([1, 2, 3]).vbt.fshift(1, fill_value=-1), pd.Series([-1, 1, 2])
        )
        assert_series_equal(
            pd.Series([True, True, True]).vbt.fshift(1, fill_value=False),
            pd.Series([False, True, True]),
        )
        assert_frame_equal(
            df.vbt.fshift(test_n, chunked=True), df.vbt.fshift(test_n, chunked=False)
        )

    def test_diff(self):
        assert_series_equal(df["a"].vbt.diff(), df["a"].diff())
        np.testing.assert_array_equal(df["a"].vbt.diff().values, nb.diff_1d_nb(df["a"].values))
        assert_frame_equal(df.vbt.diff(), df.diff())
        assert_frame_equal(
            df.vbt.diff(jitted=dict(parallel=True)), df.vbt.diff(jitted=dict(parallel=False))
        )
        assert_frame_equal(df.vbt.diff(chunked=True), df.vbt.diff(chunked=False))

    def test_pct_change(self):
        assert_series_equal(df["a"].vbt.pct_change(), df["a"].pct_change(fill_method=None))
        np.testing.assert_array_equal(
            df["a"].vbt.pct_change().values, nb.pct_change_1d_nb(df["a"].values)
        )
        assert_frame_equal(df.vbt.pct_change(), df.pct_change(fill_method=None))
        assert_frame_equal(
            df.vbt.pct_change(jitted=dict(parallel=True)),
            df.vbt.pct_change(jitted=dict(parallel=False)),
        )
        assert_frame_equal(df.vbt.pct_change(chunked=True), df.vbt.pct_change(chunked=False))

    def test_bfill(self):
        assert_series_equal(df["b"].vbt.bfill(), df["b"].bfill())
        assert_frame_equal(df.vbt.bfill(), df.bfill())
        assert_frame_equal(df.vbt.bfill(chunked=True), df.vbt.bfill(chunked=False))

    def test_ffill(self):
        assert_series_equal(df["a"].vbt.ffill(), df["a"].ffill())
        assert_frame_equal(df.vbt.ffill(), df.ffill())
        assert_frame_equal(df.vbt.ffill(chunked=True), df.vbt.ffill(chunked=False))

    def test_product(self):
        assert df["a"].vbt.product() == df["a"].product()
        assert_series_equal(df.vbt.product(), df.product().rename("product"))
        assert_series_equal(
            df.vbt.product(jitted=dict(parallel=True)),
            df.vbt.product(jitted=dict(parallel=False)),
        )
        assert_series_equal(df.vbt.product(chunked=True), df.vbt.product(chunked=False))

    def test_cumsum(self):
        assert_series_equal(df["a"].vbt.cumsum(), df["a"].cumsum().ffill().fillna(0))
        assert_frame_equal(df.vbt.cumsum(), df.cumsum().ffill().fillna(0))
        assert_frame_equal(
            df.vbt.cumsum(jitted=dict(parallel=True)),
            df.vbt.cumsum(jitted=dict(parallel=False)),
        )
        assert_frame_equal(df.vbt.cumsum(chunked=True), df.vbt.cumsum(chunked=False))

    def test_cumprod(self):
        assert_series_equal(df["a"].vbt.cumprod(), df["a"].cumprod().ffill().fillna(1))
        assert_frame_equal(df.vbt.cumprod(), df.cumprod().ffill().fillna(1))
        assert_frame_equal(
            df.vbt.cumprod(jitted=dict(parallel=True)),
            df.vbt.cumprod(jitted=dict(parallel=False)),
        )
        assert_frame_equal(df.vbt.cumprod(chunked=True), df.vbt.cumprod(chunked=False))

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_sum(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_sum(test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).sum(),
        )
        assert_frame_equal(
            df.vbt.rolling_sum(test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).sum(),
        )
        assert_frame_equal(df.vbt.rolling_sum(test_window), df.rolling(test_window).sum())
        assert_frame_equal(
            df.vbt.rolling_sum(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_sum(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_sum(test_window, minp=test_minp, chunked=True),
            df.vbt.rolling_sum(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_prod(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_prod(test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).apply(np.prod),
        )
        assert_frame_equal(
            df.vbt.rolling_prod(test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).apply(np.prod),
        )
        assert_frame_equal(df.vbt.rolling_prod(test_window), df.rolling(test_window).apply(np.prod))
        assert_frame_equal(
            df.vbt.rolling_prod(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_prod(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_prod(test_window, minp=test_minp, chunked=True),
            df.vbt.rolling_prod(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_mean(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_mean(test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).mean(),
        )
        assert_frame_equal(
            df.vbt.rolling_mean(test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).mean(),
        )
        assert_frame_equal(df.vbt.rolling_mean(test_window), df.rolling(test_window).mean())
        assert_frame_equal(
            df.vbt.rolling_mean(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_mean(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_mean(test_window, minp=test_minp, chunked=True),
            df.vbt.rolling_mean(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_mean(self, test_minp):
        assert_series_equal(
            df["a"].vbt.expanding_mean(minp=test_minp),
            df["a"].expanding(min_periods=test_minp).mean(),
        )
        assert_frame_equal(
            df.vbt.expanding_mean(minp=test_minp), df.expanding(min_periods=test_minp).mean()
        )
        assert_frame_equal(df.vbt.expanding_mean(), df.expanding().mean())
        assert_frame_equal(
            df.vbt.expanding_mean(minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.expanding_mean(minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_mean(minp=test_minp, chunked=True),
            df.vbt.expanding_mean(minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    @pytest.mark.parametrize("test_ddof", [0, 1])
    def test_rolling_std(self, test_window, test_minp, test_ddof):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
            df["a"].rolling(test_window, min_periods=test_minp).std(ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
            df.rolling(test_window, min_periods=test_minp).std(ddof=test_ddof),
        )
        assert_frame_equal(df.vbt.rolling_std(test_window), df.rolling(test_window).std())
        assert_frame_equal(
            df.vbt.rolling_std(
                test_window, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=True)
            ),
            df.vbt.rolling_std(
                test_window, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof, chunked=True),
            df.vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    @pytest.mark.parametrize("test_ddof", [0, 1])
    def test_rolling_zscore(self, test_window, test_minp, test_ddof):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_zscore(test_window, minp=test_minp, ddof=test_ddof),
            (df["a"] - df["a"].vbt.rolling_mean(test_window, minp=test_minp))
            / df["a"].vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.rolling_zscore(test_window, minp=test_minp, ddof=test_ddof),
            (df - df.vbt.rolling_mean(test_window, minp=test_minp))
            / df.vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.rolling_zscore(test_window),
            (df - df.vbt.rolling_mean(test_window)) / df.vbt.rolling_std(test_window),
        )
        assert_frame_equal(
            df.vbt.rolling_zscore(
                test_window, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=True)
            ),
            df.vbt.rolling_zscore(
                test_window, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_zscore(test_window, minp=test_minp, ddof=test_ddof, chunked=True),
            df.vbt.rolling_zscore(test_window, minp=test_minp, ddof=test_ddof, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    @pytest.mark.parametrize("test_ddof", [0, 1])
    def test_expanding_std(self, test_minp, test_ddof):
        assert_series_equal(
            df["a"].vbt.expanding_std(minp=test_minp, ddof=test_ddof),
            df["a"].expanding(min_periods=test_minp).std(ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.expanding_std(minp=test_minp, ddof=test_ddof),
            df.expanding(min_periods=test_minp).std(ddof=test_ddof),
        )
        assert_frame_equal(df.vbt.expanding_std(), df.expanding().std())
        assert_frame_equal(
            df.vbt.expanding_std(minp=test_minp, ddof=test_ddof, jitted=dict(parallel=True)),
            df.vbt.expanding_std(minp=test_minp, ddof=test_ddof, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_std(minp=test_minp, ddof=test_ddof, chunked=True),
            df.vbt.expanding_std(minp=test_minp, ddof=test_ddof, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_wm_mean(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window

        def wma(sr):
            sr = sr[sr.notna()]
            weights = np.arange(len(sr)) + 1
            return np.sum(sr * weights) / np.sum(weights)

        assert_series_equal(
            df["a"].vbt.wm_mean(test_window, minp=test_minp),
            df["a"].rolling(window=test_window, min_periods=test_minp).apply(wma),
        )
        assert_frame_equal(
            df.vbt.wm_mean(test_window, minp=test_minp),
            df.rolling(window=test_window, min_periods=test_minp).apply(wma),
        )
        assert_frame_equal(df.vbt.wm_mean(test_window), df.rolling(window=test_window).apply(wma))
        assert_frame_equal(
            df.vbt.wm_mean(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.wm_mean(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.wm_mean(test_window, minp=test_minp, chunked=True),
            df.vbt.wm_mean(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    @pytest.mark.parametrize("test_adjust", [False, True])
    def test_ewm_mean(self, test_window, test_minp, test_adjust):
        if test_minp is None:
            test_minp = test_window

        def _adjust_minp(pd_obj, pd_result):
            valid_mask = (~pd_obj.isnull()).rolling(test_window, min_periods=1).sum() >= test_minp
            pd_result[~valid_mask] = np.nan
            return pd_result

        assert_series_equal(
            df["a"].vbt.ewm_mean(test_window, minp=test_minp, adjust=test_adjust),
            _adjust_minp(df["a"], df["a"].ewm(span=test_window, adjust=test_adjust).mean()),
        )
        assert_frame_equal(
            df.vbt.ewm_mean(test_window, minp=test_minp, adjust=test_adjust),
            _adjust_minp(df, df.ewm(span=test_window, adjust=test_adjust).mean()),
        )
        assert_frame_equal(df.vbt.ewm_mean(test_window), df.ewm(span=test_window).mean())
        assert_frame_equal(
            df.vbt.ewm_mean(
                test_window, minp=test_minp, adjust=test_adjust, jitted=dict(parallel=True)
            ),
            df.vbt.ewm_mean(
                test_window, minp=test_minp, adjust=test_adjust, jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            df.vbt.ewm_mean(test_window, minp=test_minp, adjust=test_adjust, chunked=True),
            df.vbt.ewm_mean(test_window, minp=test_minp, adjust=test_adjust, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    @pytest.mark.parametrize("test_adjust", [False, True])
    def test_ewm_std(self, test_window, test_minp, test_adjust):
        if test_minp is None:
            test_minp = test_window

        def _adjust_minp(pd_obj, pd_result):
            valid_mask = (~pd_obj.isnull()).rolling(test_window, min_periods=1).sum() >= test_minp
            pd_result[~valid_mask] = np.nan
            return pd_result

        assert_series_equal(
            df["a"].vbt.ewm_std(test_window, minp=test_minp, adjust=test_adjust),
            _adjust_minp(
                df["a"],
                df["a"].ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).std(),
            ),
        )
        assert_frame_equal(
            df.vbt.ewm_std(test_window, minp=test_minp, adjust=test_adjust),
            _adjust_minp(
                df, df.ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).std()
            ),
        )
        assert_frame_equal(df.vbt.ewm_std(test_window), df.ewm(span=test_window).std())
        assert_frame_equal(
            df.vbt.ewm_std(
                test_window, minp=test_minp, adjust=test_adjust, jitted=dict(parallel=True)
            ),
            df.vbt.ewm_std(
                test_window, minp=test_minp, adjust=test_adjust, jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            df.vbt.ewm_std(test_window, minp=test_minp, adjust=test_adjust, chunked=True),
            df.vbt.ewm_std(test_window, minp=test_minp, adjust=test_adjust, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_wwm_mean(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window

        def _adjust_minp(pd_obj, pd_result):
            valid_mask = (~pd_obj.isnull()).rolling(
                2 * test_window - 1, min_periods=1
            ).sum() >= test_minp
            pd_result[~valid_mask] = np.nan
            return pd_result

        assert_series_equal(
            df["a"].vbt.wwm_mean(test_window, minp=test_minp),
            _adjust_minp(df["a"], df["a"].ewm(alpha=1 / test_window, min_periods=test_minp).mean()),
        )
        assert_frame_equal(
            df.vbt.wwm_mean(test_window, minp=test_minp),
            _adjust_minp(df, df.ewm(alpha=1 / test_window, min_periods=test_minp).mean()),
        )
        assert_frame_equal(df.vbt.wwm_mean(test_window), df.ewm(alpha=1 / test_window).mean())
        assert_frame_equal(
            df.vbt.wwm_mean(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.wwm_mean(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.wwm_mean(test_window, minp=test_minp, chunked=True),
            df.vbt.wwm_mean(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_wwm_std(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window

        def _adjust_minp(pd_obj, pd_result):
            valid_mask = (~pd_obj.isnull()).rolling(
                2 * test_window - 1, min_periods=1
            ).sum() >= test_minp
            pd_result[~valid_mask] = np.nan
            return pd_result

        assert_series_equal(
            df["a"].vbt.wwm_std(test_window, minp=test_minp),
            _adjust_minp(df["a"], df["a"].ewm(alpha=1 / test_window, min_periods=test_minp).std()),
        )
        assert_frame_equal(
            df.vbt.wwm_std(test_window, minp=test_minp),
            _adjust_minp(df, df.ewm(alpha=1 / test_window, min_periods=test_minp).std()),
        )
        assert_frame_equal(df.vbt.wwm_std(test_window), df.ewm(alpha=1 / test_window).std())
        assert_frame_equal(
            df.vbt.wwm_std(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.wwm_std(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.wwm_std(test_window, minp=test_minp, chunked=True),
            df.vbt.wwm_std(test_window, minp=test_minp, chunked=False),
        )

    def test_vidya(self):
        assert_series_equal(
            df["a"].vbt.vidya(3),
            pd.Series([np.nan, np.nan, np.nan, 2.0, np.nan], index=df.index, name="a"),
        )
        assert_frame_equal(
            df.vbt.vidya(3),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [2.0, np.nan, np.nan],
                    [np.nan, 0.5, np.nan],
                ],
                index=df.index,
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.vidya(3, minp=1),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [1.0, np.nan, 1.0],
                    [2.0, 1.5, np.nan],
                    [3.0, 1.75, 1.0],
                    [np.nan, 1.375, 1.0],
                ],
                index=df.index,
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.vidya(3, jitted=dict(parallel=True)),
            df.vbt.vidya(3, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.vidya(3, chunked=True),
            df.vbt.vidya(3, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_ma(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.ma(test_window, wtype="simple", minp=test_minp),
            df["a"].vbt.rolling_mean(test_window, minp=test_minp),
        )
        assert_frame_equal(
            df.vbt.ma(test_window, wtype="simple", minp=test_minp),
            df.vbt.rolling_mean(test_window, minp=test_minp),
        )
        assert_frame_equal(
            df.vbt.ma(test_window, wtype="weighted", minp=test_minp),
            df.vbt.wm_mean(test_window, minp=test_minp),
        )
        assert_frame_equal(
            df.vbt.ma(test_window, wtype="exp", minp=test_minp),
            df.vbt.ewm_mean(test_window, minp=test_minp),
        )
        assert_frame_equal(
            df.vbt.ma(test_window, wtype="wilder", minp=test_minp),
            df.vbt.wwm_mean(test_window, minp=test_minp),
        )
        assert_frame_equal(
            df.vbt.ma(test_window, wtype="vidya", minp=test_minp),
            df.vbt.vidya(test_window, minp=test_minp),
        )
        assert_frame_equal(
            df.vbt.ma(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.ma(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.ma(test_window, minp=test_minp, chunked=True),
            df.vbt.ma(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    @pytest.mark.parametrize("test_ddof", [0, 1])
    def test_msd(self, test_window, test_minp, test_ddof):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.msd(test_window, wtype="simple", minp=test_minp, ddof=test_ddof),
            df["a"].vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.msd(test_window, wtype="simple", minp=test_minp, ddof=test_ddof),
            df.vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
        )
        with pytest.raises(Exception):
            df.vbt.msd(test_window, wtype="weighted", minp=test_minp)
        assert_frame_equal(
            df.vbt.msd(test_window, wtype="exp", minp=test_minp),
            df.vbt.ewm_std(test_window, minp=test_minp),
        )
        assert_frame_equal(
            df.vbt.msd(test_window, wtype="wilder", minp=test_minp),
            df.vbt.wwm_std(test_window, minp=test_minp),
        )
        assert_frame_equal(
            df.vbt.msd(test_window, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=True)),
            df.vbt.msd(test_window, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.msd(test_window, minp=test_minp, ddof=test_ddof, chunked=True),
            df.vbt.msd(test_window, minp=test_minp, ddof=test_ddof, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    @pytest.mark.parametrize("test_ddof", [0, 1])
    def test_rolling_cov(self, test_window, test_minp, test_ddof):
        if test_minp is None:
            test_minp = test_window
        df2 = df[["b", "c", "a"]].rename(columns={"b": "a", "c": "b", "a": "c"})
        assert_series_equal(
            df["a"].vbt.rolling_cov(df2["a"], test_window, minp=test_minp, ddof=test_ddof),
            df["a"].rolling(test_window, min_periods=test_minp).cov(df2["a"], ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.rolling_cov(df2, test_window, minp=test_minp, ddof=test_ddof),
            df.rolling(test_window, min_periods=test_minp).cov(df2, ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.rolling_cov(df2, test_window, ddof=test_ddof),
            df.rolling(test_window).cov(df2, ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.rolling_cov(
                df2, test_window, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=True)
            ),
            df.vbt.rolling_cov(
                df2, test_window, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_cov(df2, test_window, minp=test_minp, ddof=test_ddof, chunked=True),
            df.vbt.rolling_cov(df2, test_window, minp=test_minp, ddof=test_ddof, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    @pytest.mark.parametrize("test_ddof", [0, 1])
    def test_expanding_cov(self, test_minp, test_ddof):
        df2 = df[["b", "c", "a"]].rename(columns={"b": "a", "c": "b", "a": "c"})
        assert_series_equal(
            df["a"].vbt.expanding_cov(df2["a"], minp=test_minp, ddof=test_ddof),
            df["a"].expanding(min_periods=test_minp).cov(df2["a"], ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.expanding_cov(df2, minp=test_minp, ddof=test_ddof),
            df.expanding(min_periods=test_minp).cov(df2, ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.expanding_cov(df2, ddof=test_ddof),
            df.expanding().cov(df2, ddof=test_ddof),
        )
        assert_frame_equal(
            df.vbt.expanding_cov(df2, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=True)),
            df.vbt.expanding_cov(df2, minp=test_minp, ddof=test_ddof, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_cov(df2, minp=test_minp, ddof=test_ddof, chunked=True),
            df.vbt.expanding_cov(df2, minp=test_minp, ddof=test_ddof, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_corr(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        df2 = df[["b", "c", "a"]].rename(columns={"b": "a", "c": "b", "a": "c"})
        assert_series_equal(
            df["a"].vbt.rolling_corr(df2["a"], test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).corr(df2["a"]),
        )
        assert_frame_equal(
            df.vbt.rolling_corr(df2, test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).corr(df2),
        )
        assert_frame_equal(df.vbt.rolling_corr(df2, test_window), df.rolling(test_window).corr(df2))
        assert_frame_equal(
            df.vbt.rolling_corr(df2, test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_corr(df2, test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_corr(df2, test_window, minp=test_minp, chunked=True),
            df.vbt.rolling_corr(df2, test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_corr(self, test_minp):
        df2 = df[["b", "c", "a"]].rename(columns={"b": "a", "c": "b", "a": "c"})
        assert_series_equal(
            df["a"].vbt.expanding_corr(df2["a"], minp=test_minp),
            df["a"].expanding(min_periods=test_minp).corr(df2["a"]),
        )
        assert_frame_equal(
            df.vbt.expanding_corr(df2, minp=test_minp),
            df.expanding(min_periods=test_minp).corr(df2),
        )
        assert_frame_equal(df.vbt.expanding_corr(df2), df.expanding().corr(df2))
        assert_frame_equal(
            df.vbt.expanding_corr(df2, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.expanding_corr(df2, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_corr(df2, minp=test_minp, chunked=True),
            df.vbt.expanding_corr(df2, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [2, 3, 4, 5])
    def test_rolling_ols(self, test_window):
        def sr_ols(x, y):
            from statsmodels.regression.rolling import RollingOLS

            intercept_slope = (
                RollingOLS(
                    y,
                    np.column_stack((np.broadcast_to(1, x.shape), x)),
                    window=test_window,
                    min_nobs=test_window,
                )
                .fit()
                .params
            )
            intercept = intercept_slope.iloc[:, 0].rename(x.name)
            slope = intercept_slope.iloc[:, 1].rename(x.name)
            return slope, intercept

        df2 = df[["b", "c", "a"]].rename(columns={"b": "a", "c": "b", "a": "c"})
        assert_series_equal(
            df["a"].vbt.rolling_ols(df2["a"], test_window)[0],
            sr_ols(df["a"], df2["a"])[0],
        )
        assert_series_equal(
            df["a"].vbt.rolling_ols(df2["a"], test_window)[1],
            sr_ols(df["a"], df2["a"])[1],
        )
        assert_frame_equal(
            df.vbt.rolling_ols(df2, test_window)[0],
            pd.concat(
                (
                    sr_ols(df["a"], df2["a"])[0],
                    sr_ols(df["b"], df2["b"])[0],
                    sr_ols(df["c"], df2["c"])[0],
                ),
                axis=1,
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_ols(df2, test_window)[1],
            pd.concat(
                (
                    sr_ols(df["a"], df2["a"])[1],
                    sr_ols(df["b"], df2["b"])[1],
                    sr_ols(df["c"], df2["c"])[1],
                ),
                axis=1,
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_ols(df2, test_window, jitted=dict(parallel=True))[0],
            df.vbt.rolling_ols(df2, test_window, jitted=dict(parallel=False))[0],
        )
        assert_frame_equal(
            df.vbt.rolling_ols(df2, test_window, jitted=dict(parallel=True))[1],
            df.vbt.rolling_ols(df2, test_window, jitted=dict(parallel=False))[1],
        )
        assert_frame_equal(
            df.vbt.rolling_ols(df2, test_window, chunked=True)[0],
            df.vbt.rolling_ols(df2, test_window, chunked=False)[0],
        )
        assert_frame_equal(
            df.vbt.rolling_ols(df2, test_window, chunked=True)[1],
            df.vbt.rolling_ols(df2, test_window, chunked=False)[1],
        )

    def test_expanding_ols(self):
        def sr_ols(x, y):
            from statsmodels.regression.rolling import RollingOLS

            intercept_slope = (
                RollingOLS(y, np.column_stack((np.broadcast_to(1, x.shape), x)), expanding=True)
                .fit()
                .params
            )
            intercept = intercept_slope.iloc[:, 0].rename(x.name)
            slope = intercept_slope.iloc[:, 1].rename(x.name)
            return slope, intercept

        df2 = df[["b", "c", "a"]].rename(columns={"b": "a", "c": "b", "a": "c"})
        assert_series_equal(
            df["a"].vbt.expanding_ols(df2["a"])[0],
            sr_ols(df["a"], df2["a"])[0],
        )
        assert_series_equal(
            df["a"].vbt.expanding_ols(df2["a"])[1],
            sr_ols(df["a"], df2["a"])[1],
        )
        assert_frame_equal(
            df.vbt.expanding_ols(df2)[0],
            pd.concat(
                (
                    sr_ols(df["a"], df2["a"])[0],
                    sr_ols(df["b"], df2["b"])[0],
                    sr_ols(df["c"], df2["c"])[0],
                ),
                axis=1,
            ),
        )
        assert_frame_equal(
            df.vbt.expanding_ols(df2)[1],
            pd.concat(
                (
                    sr_ols(df["a"], df2["a"])[1],
                    sr_ols(df["b"], df2["b"])[1],
                    sr_ols(df["c"], df2["c"])[1],
                ),
                axis=1,
            ),
        )
        assert_frame_equal(
            df.vbt.expanding_ols(df2, jitted=dict(parallel=True))[0],
            df.vbt.expanding_ols(df2, jitted=dict(parallel=False))[0],
        )
        assert_frame_equal(
            df.vbt.expanding_ols(df2, jitted=dict(parallel=True))[1],
            df.vbt.expanding_ols(df2, jitted=dict(parallel=False))[1],
        )
        assert_frame_equal(
            df.vbt.expanding_ols(df2, chunked=True)[0],
            df.vbt.expanding_ols(df2, chunked=False)[0],
        )
        assert_frame_equal(
            df.vbt.expanding_ols(df2, chunked=True)[1],
            df.vbt.expanding_ols(df2, chunked=False)[1],
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    @pytest.mark.parametrize("test_pct", [False, True])
    def test_rolling_rank(self, test_window, test_minp, test_pct):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_rank(test_window, minp=test_minp, pct=test_pct),
            df["a"]
            .rolling(test_window, min_periods=test_minp)
            .apply(lambda sr: sr.rank(pct=test_pct).iloc[-1]),
        )
        assert_frame_equal(
            df.vbt.rolling_rank(test_window, minp=test_minp, pct=test_pct),
            df.rolling(test_window, min_periods=test_minp).apply(
                lambda sr: sr.rank(pct=test_pct).iloc[-1]
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_rank(test_window, pct=test_pct),
            df.rolling(test_window).apply(lambda sr: sr.rank(pct=test_pct).iloc[-1]),
        )
        assert_frame_equal(
            df.vbt.rolling_rank(
                test_window, minp=test_minp, pct=test_pct, jitted=dict(parallel=True)
            ),
            df.vbt.rolling_rank(
                test_window, minp=test_minp, pct=test_pct, jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_rank(test_window, minp=test_minp, pct=test_pct, chunked=True),
            df.vbt.rolling_rank(test_window, minp=test_minp, pct=test_pct, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    @pytest.mark.parametrize("test_pct", [False, True])
    def test_expanding_rank(self, test_minp, test_pct):
        assert_series_equal(
            df["a"].vbt.expanding_rank(minp=test_minp, pct=test_pct),
            df["a"]
            .expanding(min_periods=test_minp)
            .apply(lambda sr: sr.rank(pct=test_pct).iloc[-1]),
        )
        assert_frame_equal(
            df.vbt.expanding_rank(minp=test_minp, pct=test_pct),
            df.expanding(min_periods=test_minp).apply(lambda sr: sr.rank(pct=test_pct).iloc[-1]),
        )
        assert_frame_equal(
            df.vbt.expanding_rank(pct=test_pct),
            df.expanding().apply(lambda sr: sr.rank(pct=test_pct).iloc[-1]),
        )
        assert_frame_equal(
            df.vbt.expanding_rank(minp=test_minp, pct=test_pct, jitted=dict(parallel=True)),
            df.vbt.expanding_rank(minp=test_minp, pct=test_pct, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_rank(minp=test_minp, pct=test_pct, chunked=True),
            df.vbt.expanding_rank(minp=test_minp, pct=test_pct, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_min(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_min(test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).min(),
        )
        assert_frame_equal(
            df.vbt.rolling_min(test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).min(),
        )
        assert_frame_equal(df.vbt.rolling_min(test_window), df.rolling(test_window).min())
        assert_frame_equal(
            df.vbt.rolling_min(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_min(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_min(test_window, minp=test_minp, chunked=True),
            df.vbt.rolling_min(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_min(self, test_minp):
        assert_series_equal(
            df["a"].vbt.expanding_min(minp=test_minp),
            df["a"].expanding(min_periods=test_minp).min(),
        )
        assert_frame_equal(
            df.vbt.expanding_min(minp=test_minp), df.expanding(min_periods=test_minp).min()
        )
        assert_frame_equal(df.vbt.expanding_min(), df.expanding().min())
        assert_frame_equal(
            df.vbt.expanding_min(minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.expanding_min(minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_min(minp=test_minp, chunked=True),
            df.vbt.expanding_min(minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_max(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_max(test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).max(),
        )
        assert_frame_equal(
            df.vbt.rolling_max(test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).max(),
        )
        assert_frame_equal(df.vbt.rolling_max(test_window), df.rolling(test_window).max())
        assert_frame_equal(
            df.vbt.rolling_max(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_max(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_max(test_window, minp=test_minp, chunked=True),
            df.vbt.rolling_max(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_max(self, test_minp):
        assert_series_equal(
            df["a"].vbt.expanding_max(minp=test_minp),
            df["a"].expanding(min_periods=test_minp).max(),
        )
        assert_frame_equal(
            df.vbt.expanding_max(minp=test_minp), df.expanding(min_periods=test_minp).max()
        )
        assert_frame_equal(df.vbt.expanding_max(), df.expanding().max())
        assert_frame_equal(
            df.vbt.expanding_max(minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.expanding_max(minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_max(minp=test_minp, chunked=True),
            df.vbt.expanding_max(minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_idxmin(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_idxmin(test_window, minp=test_minp, local=True),
            df["a"]
            .rolling(test_window, min_periods=test_minp)
            .apply(np.argmin)
            .fillna(-1)
            .astype(int_),
        )
        assert_frame_equal(
            df.vbt.rolling_idxmin(test_window, minp=test_minp, local=True),
            df.rolling(test_window, min_periods=test_minp).apply(np.argmin).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.rolling_idxmin(test_window, local=True),
            df.rolling(test_window).apply(np.argmin).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.rolling_idxmin(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_idxmin(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_idxmin(test_window, minp=test_minp, chunked=True),
            df.vbt.rolling_idxmin(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_idxmin(self, test_minp):
        assert_series_equal(
            df["a"].vbt.expanding_idxmin(minp=test_minp, local=True),
            df["a"].expanding(min_periods=test_minp).apply(np.argmin).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.expanding_idxmin(minp=test_minp, local=True),
            df.expanding(min_periods=test_minp).apply(np.argmin).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.expanding_idxmin(local=True),
            df.expanding().apply(np.argmin).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.expanding_idxmin(minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.expanding_idxmin(minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_idxmin(minp=test_minp, chunked=True),
            df.vbt.expanding_idxmin(minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_idxmax(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_idxmax(test_window, minp=test_minp, local=True),
            df["a"]
            .rolling(test_window, min_periods=test_minp)
            .apply(np.argmax)
            .fillna(-1)
            .astype(int_),
        )
        assert_frame_equal(
            df.vbt.rolling_idxmax(test_window, minp=test_minp, local=True),
            df.rolling(test_window, min_periods=test_minp).apply(np.argmax).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.rolling_idxmax(test_window, local=True),
            df.rolling(test_window).apply(np.argmax).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.rolling_idxmax(test_window, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_idxmax(test_window, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_idxmax(test_window, minp=test_minp, chunked=True),
            df.vbt.rolling_idxmax(test_window, minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_idxmax(self, test_minp):
        assert_series_equal(
            df["a"].vbt.expanding_idxmax(minp=test_minp, local=True),
            df["a"].expanding(min_periods=test_minp).apply(np.argmax).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.expanding_idxmax(minp=test_minp, local=True),
            df.expanding(min_periods=test_minp).apply(np.argmax).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.expanding_idxmax(local=True),
            df.expanding().apply(np.argmax).fillna(-1).astype(int_),
        )
        assert_frame_equal(
            df.vbt.expanding_idxmax(minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.expanding_idxmax(minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_idxmax(minp=test_minp, chunked=True),
            df.vbt.expanding_idxmax(minp=test_minp, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    def test_rolling_any(self, test_window):
        mask = (df == 2) | (df == 3)
        assert_series_equal(
            mask["a"].vbt.rolling_any(test_window),
            mask["a"].rolling(test_window, min_periods=1).max().fillna(False).astype(np.bool_),
        )
        assert_frame_equal(
            mask.vbt.rolling_any(test_window),
            mask.rolling(test_window, min_periods=1).max().fillna(False).astype(np.bool_),
        )
        assert_frame_equal(
            mask.vbt.rolling_any(test_window, jitted=dict(parallel=True)),
            mask.vbt.rolling_any(test_window, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            mask.vbt.rolling_any(test_window, chunked=True),
            mask.vbt.rolling_any(test_window, chunked=False),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5])
    def test_rolling_all(self, test_window):
        mask = (df == 2) | (df == 3)
        assert_series_equal(
            mask["a"].vbt.rolling_all(test_window),
            mask["a"].rolling(test_window, min_periods=1).min().fillna(False).astype(np.bool_),
        )
        assert_frame_equal(
            mask.vbt.rolling_all(test_window),
            mask.rolling(test_window, min_periods=1).min().fillna(False).astype(np.bool_),
        )
        assert_frame_equal(
            mask.vbt.rolling_all(test_window, jitted=dict(parallel=True)),
            mask.vbt.rolling_all(test_window, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            mask.vbt.rolling_all(test_window, chunked=True),
            mask.vbt.rolling_all(test_window, chunked=False),
        )

    def test_rolling_pattern_similarity(self):
        assert_frame_equal(
            df.vbt.rolling_pattern_similarity([1, 2, 3], interp_mode="linear"),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan],
                    [1.0, 0.19999999999999996, np.nan],
                    [np.nan, 0.19999999999999996, np.nan],
                ],
                index=pd.DatetimeIndex(
                    ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"],
                    dtype="datetime64[ns]",
                    freq="D",
                ),
                columns=pd.Index(["a", "b", "c"], dtype="object"),
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_pattern_similarity([1, 2, 3], 4, interp_mode="linear"),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan],
                    [np.nan, 0.19999999999999996, np.nan],
                ],
                index=pd.DatetimeIndex(
                    ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"],
                    dtype="datetime64[ns]",
                    freq="D",
                ),
                columns=pd.Index(["a", "b", "c"], dtype="object"),
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_pattern_similarity(
                [1, 2, 3], interp_mode="linear", jitted=dict(parallel=True)
            ),
            df.vbt.rolling_pattern_similarity(
                [1, 2, 3], interp_mode="linear", jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            df.vbt.rolling_pattern_similarity([1, 2, 3], interp_mode="linear", chunked=True),
            df.vbt.rolling_pattern_similarity([1, 2, 3], interp_mode="linear", chunked=False),
        )

    def test_map(self):
        @njit
        def mult_nb(x, y):
            return x * y

        @njit
        def mult_meta_nb(i, col, x, y):
            return x[i, col] * y

        assert_series_equal(df["a"].vbt.map(mult_nb, 2), df["a"].map(lambda x: x * 2))
        assert_frame_equal(df.vbt.map(mult_nb, 2), df.applymap(lambda x: x * 2))
        assert_frame_equal(
            df.vbt.map(mult_nb, 2, jitted=dict(parallel=True)),
            df.vbt.map(mult_nb, 2, jitted=dict(parallel=False)),
        )
        chunked = dict(arg_take_spec=dict(args=vbt.ArgsTaker(None)))
        assert_frame_equal(
            df.vbt.map(mult_nb, 2, chunked=chunked), df.vbt.map(mult_nb, 2, chunked=False)
        )
        assert_frame_equal(
            pd.DataFrame.vbt.map(mult_meta_nb, df.vbt.to_2d_array(), 2, wrapper=df.vbt.wrapper),
            df.vbt.map(mult_nb, 2),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.map(
                mult_meta_nb,
                df.vbt.to_2d_array(),
                2,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.map(
                mult_meta_nb,
                df.vbt.to_2d_array(),
                2,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(arg_take_spec=dict(args=vbt.ArgsTaker(vbt.ArraySlicer(axis=1), None)))
        assert_frame_equal(
            pd.DataFrame.vbt.map(
                mult_meta_nb, df.vbt.to_2d_array(), 2, wrapper=df.vbt.wrapper, chunked=chunked
            ),
            pd.DataFrame.vbt.map(
                mult_meta_nb, df.vbt.to_2d_array(), 2, wrapper=df.vbt.wrapper, chunked=False
            ),
        )

        @njit
        def mult_meta2_nb(i, col, x, y):
            return x[i, col] * y[i, col]

        assert_frame_equal(
            pd.DataFrame.vbt.map(
                mult_meta2_nb,
                vbt.RepEval("to_2d_array(x)"),
                vbt.RepEval("to_2d_array(y)"),
                broadcast_named_args=dict(
                    x=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    y=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
            ),
            pd.DataFrame(
                [[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]],
                index=df.index,
                columns=df.columns,
            ),
        )

    def test_apply_along_axis(self):
        @njit
        def pow_nb(x, y):
            return x**y

        @njit
        def pow_meta_nb(col, x, y):
            return x[:, col] ** y

        @njit
        def row_pow_meta_nb(i, x, y):
            return x[i, :] ** y

        assert_frame_equal(
            df.vbt.apply_along_axis(pow_nb, 2, axis=0),
            df.apply(pow_nb, args=(2,), axis=0, raw=True),
        )
        assert_frame_equal(
            df.vbt.apply_along_axis(pow_nb, 2, axis=0, jitted=dict(parallel=True)),
            df.vbt.apply_along_axis(pow_nb, 2, axis=0, jitted=dict(parallel=False)),
        )
        chunked = dict(arg_take_spec=dict(args=vbt.ArgsTaker(None)))
        assert_frame_equal(
            df.vbt.apply_along_axis(pow_nb, 2, axis=0, chunked=chunked),
            df.vbt.apply_along_axis(pow_nb, 2, axis=0, chunked=False),
        )
        assert_frame_equal(
            df.vbt.apply_along_axis(pow_nb, 2, axis=1),
            df.apply(pow_nb, args=(2,), axis=1, raw=True),
        )
        assert_frame_equal(
            df.vbt.apply_along_axis(pow_nb, 2, axis=1, jitted=dict(parallel=True)),
            df.vbt.apply_along_axis(pow_nb, 2, axis=1, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.apply_along_axis(pow_nb, 2, axis=1, chunked=chunked),
            df.vbt.apply_along_axis(pow_nb, 2, axis=1, chunked=False),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.apply_along_axis(
                row_pow_meta_nb, df.vbt.to_2d_array(), 2, axis=0, wrapper=df.vbt.wrapper
            ),
            df.vbt.apply_along_axis(pow_nb, 2, axis=0),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.apply_along_axis(
                row_pow_meta_nb,
                df.vbt.to_2d_array(),
                2,
                axis=0,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.apply_along_axis(
                row_pow_meta_nb,
                df.vbt.to_2d_array(),
                2,
                axis=0,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(arg_take_spec=dict(args=vbt.ArgsTaker(vbt.ArraySlicer(axis=0), None)))
        assert_frame_equal(
            pd.DataFrame.vbt.apply_along_axis(
                row_pow_meta_nb,
                df.vbt.to_2d_array(),
                2,
                axis=0,
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.apply_along_axis(
                row_pow_meta_nb,
                df.vbt.to_2d_array(),
                2,
                axis=0,
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.apply_along_axis(
                pow_meta_nb, df.vbt.to_2d_array(), 2, axis=1, wrapper=df.vbt.wrapper
            ),
            df.vbt.apply_along_axis(pow_nb, 2, axis=1),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.apply_along_axis(
                pow_meta_nb,
                df.vbt.to_2d_array(),
                2,
                axis=1,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.apply_along_axis(
                pow_meta_nb,
                df.vbt.to_2d_array(),
                2,
                axis=1,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(arg_take_spec=dict(args=vbt.ArgsTaker(vbt.ArraySlicer(axis=1), None)))
        assert_frame_equal(
            pd.DataFrame.vbt.apply_along_axis(
                pow_meta_nb,
                df.vbt.to_2d_array(),
                2,
                axis=1,
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.apply_along_axis(
                pow_meta_nb,
                df.vbt.to_2d_array(),
                2,
                axis=1,
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

        @njit
        def pow_meta2_nb(col, x, y):
            return x[:, col] ** y[:, col]

        assert_frame_equal(
            pd.DataFrame.vbt.apply_along_axis(
                pow_meta2_nb,
                vbt.RepEval("to_2d_array(x)"),
                vbt.RepEval("to_2d_array(y)"),
                broadcast_named_args=dict(
                    x=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    y=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
            ),
            pd.DataFrame(
                [[1, 1, 1], [2, 4, 8], [3, 9, 27], [4, 16, 64], [5, 25, 125]],
                index=df.index,
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize("test_window", [1, 2, 3, 4, 5, "12h", "1d", "3d", "10d"])
    @pytest.mark.parametrize("test_minp", [1, None])
    def test_rolling_apply(self, test_window, test_minp):
        @njit
        def mean_nb(x):
            return np.nanmean(x)

        @njit
        def mean_meta_nb(from_i, to_i, col, x):
            return np.nanmean(x[from_i:to_i, col])

        if isinstance(test_window, str):
            test_minp = 1
        elif test_minp is None:
            test_minp = test_window
        assert_series_equal(
            df["a"].vbt.rolling_apply(test_window, mean_nb, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).apply(mean_nb, raw=True),
        )
        assert_frame_equal(
            df.vbt.rolling_apply(test_window, mean_nb, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).apply(mean_nb, raw=True),
        )
        assert_frame_equal(
            df.vbt.rolling_apply(test_window, mean_nb, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.rolling_apply(test_window, mean_nb, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.rolling_apply(test_window, mean_nb, minp=test_minp, chunked=True),
            df.vbt.rolling_apply(test_window, mean_nb, minp=test_minp, chunked=False),
        )
        result = df.vbt.rolling_apply(test_window, mean_nb, minp=1)
        result.iloc[: test_minp - 1] = np.nan
        assert_frame_equal(
            pd.DataFrame.vbt.rolling_apply(
                test_window,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
            ),
            result,
        )
        assert_frame_equal(
            pd.DataFrame.vbt.rolling_apply(
                test_window,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.rolling_apply(
                test_window,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.rolling_apply(
                test_window,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.rolling_apply(
                test_window,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

        @njit
        def mean_ratio_meta_nb(from_i, to_i, col, x, y):
            return np.nanmean(x[from_i:to_i, col]) / np.nanmean(y[from_i:to_i, col])

        assert_frame_equal(
            pd.DataFrame.vbt.rolling_apply(
                3,
                mean_ratio_meta_nb,
                vbt.RepEval("to_2d_array(x)"),
                vbt.RepEval("to_2d_array(y)"),
                broadcast_named_args=dict(
                    x=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    y=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
            ),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [2.0, 1.0, 0.6666666666666666],
                    [3.0, 1.5, 1.0],
                    [4.0, 2.0, 1.3333333333333333],
                ],
                index=df.index,
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_apply(self, test_minp):
        @njit
        def mean_nb(x):
            return np.nanmean(x)

        @njit
        def mean_meta_nb(from_i, to_i, col, x):
            return np.nanmean(x[from_i:to_i, col])

        assert_series_equal(
            df["a"].vbt.expanding_apply(mean_nb, minp=test_minp),
            df["a"].expanding(min_periods=test_minp).apply(mean_nb, raw=True),
        )
        assert_frame_equal(
            df.vbt.expanding_apply(mean_nb, minp=test_minp),
            df.expanding(min_periods=test_minp).apply(mean_nb, raw=True),
        )
        assert_frame_equal(
            df.vbt.expanding_apply(mean_nb, minp=test_minp, jitted=dict(parallel=True)),
            df.vbt.expanding_apply(mean_nb, minp=test_minp, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.expanding_apply(mean_nb, minp=test_minp, chunked=True),
            df.vbt.expanding_apply(mean_nb, minp=test_minp, chunked=False),
        )
        result = df.vbt.expanding_apply(mean_nb, minp=1)
        result.iloc[: test_minp - 1] = np.nan
        assert_frame_equal(
            pd.DataFrame.vbt.expanding_apply(
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
            ),
            result,
        )
        assert_frame_equal(
            pd.DataFrame.vbt.expanding_apply(
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.expanding_apply(
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.expanding_apply(
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.expanding_apply(
                mean_meta_nb,
                df.vbt.to_2d_array(),
                minp=test_minp,
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

    def test_groupby_apply(self):
        @njit
        def mean_nb(x):
            return np.nanmean(x)

        @njit
        def mean_meta_nb(idxs, group, col, x):
            return np.nanmean(x[idxs, col])

        assert_series_equal(
            df["a"].vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb),
            df["a"]
            .groupby(np.array([1, 1, 2, 2, 3]))
            .apply(lambda x: mean_nb(x.values))
            .rename_axis("group"),
        )
        assert_frame_equal(
            df.vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb),
            df.groupby(np.array([1, 1, 2, 2, 3]))
            .agg(
                {
                    "a": lambda x: mean_nb(x.values),
                    "b": lambda x: mean_nb(x.values),
                    "c": lambda x: mean_nb(x.values),
                }
            )
            .rename_axis("group"),  # any clean way to do column-wise grouping in pandas?
        )
        assert_frame_equal(
            df.vbt.groupby_apply(df.groupby(np.array([1, 1, 2, 2, 3])), mean_nb),
            df.vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb),
        )
        assert_frame_equal(
            df.vbt.groupby_apply(vbt.Grouper(df.index, np.array([1, 1, 2, 2, 3])), mean_nb),
            df.vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb),
        )
        assert_frame_equal(
            df.vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb, jitted=dict(parallel=True)),
            df.vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb, chunked=True),
            df.vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb, chunked=False),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.groupby_apply(
                np.array([1, 1, 2, 2, 3]),
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.groupby_apply(np.array([1, 1, 2, 2, 3]), mean_nb),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.groupby_apply(
                np.array([1, 1, 2, 2, 3]),
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.groupby_apply(
                np.array([1, 1, 2, 2, 3]),
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.groupby_apply(
                np.array([1, 1, 2, 2, 3]),
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.groupby_apply(
                np.array([1, 1, 2, 2, 3]),
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

        @njit
        def mean_ratio_meta_nb(idxs, group, col, x, y):
            return np.nanmean(x[idxs, col]) / np.nanmean(y[idxs, col])

        assert_frame_equal(
            pd.DataFrame.vbt.groupby_apply(
                vbt.RepEval("group_by_evenly_nb(wrapper.shape[0], 2)"),
                mean_ratio_meta_nb,
                vbt.RepEval("to_2d_array(x)"),
                vbt.RepEval("to_2d_array(y)"),
                broadcast_named_args=dict(
                    x=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    y=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(
                    to_2d_array=vbt.base.reshaping.to_2d_array,
                    group_by_evenly_nb=vbt.base.grouping.nb.group_by_evenly_nb,
                ),
            ),
            pd.DataFrame(
                [[2.0, 1.0, 0.6666666666666666], [4.5, 2.25, 1.5]], columns=df.columns
            ).rename_axis("group"),
        )

    def test_groupby_transform(self):
        def zscore(x):
            return (x - np.nanmean(x)) / np.nanstd(x)

        @njit
        def zscore_nb(x):
            return (x - np.nanmean(x)) / np.nanstd(x)

        @njit
        def zscore_meta_nb(idxs, group, x):
            return zscore_nb(x[idxs])

        assert_series_equal(
            df["a"].vbt.groupby_transform(np.array([1, 1, 2, 2, 3]), zscore_nb),
            df["a"].groupby(np.array([1, 1, 2, 2, 3]), group_keys=False).apply(zscore),
        )
        assert_frame_equal(
            df.vbt.groupby_transform(np.array([1, 1, 2, 2, 3]), zscore_nb),
            df.groupby(np.array([1, 1, 2, 2, 3]), group_keys=False).apply(zscore),
        )
        assert_frame_equal(
            df.vbt.groupby_transform(df.groupby(np.array([1, 1, 2, 2, 3])), zscore_nb),
            df.vbt.groupby_transform(np.array([1, 1, 2, 2, 3]), zscore_nb),
        )
        assert_frame_equal(
            df.vbt.groupby_transform(vbt.Grouper(df.index, np.array([1, 1, 2, 2, 3])), zscore_nb),
            df.vbt.groupby_transform(np.array([1, 1, 2, 2, 3]), zscore_nb),
        )
        assert_frame_equal(
            df.vbt.groupby_transform(
                np.array([1, 1, 2, 2, 3]), zscore_nb, jitted=dict(parallel=True)
            ),
            df.vbt.groupby_transform(
                np.array([1, 1, 2, 2, 3]), zscore_nb, jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.groupby_transform(
                np.array([1, 1, 2, 2, 3]),
                zscore_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.groupby_transform(np.array([1, 1, 2, 2, 3]), zscore_nb),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.groupby_transform(
                np.array([1, 1, 2, 2, 3]),
                zscore_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.groupby_transform(
                np.array([1, 1, 2, 2, 3]),
                zscore_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )

        @njit
        def zscore_ratio_meta_nb(idxs, group, x, y):
            return zscore_nb(x[idxs]) / zscore_nb(y[idxs])

        assert_frame_equal(
            pd.DataFrame.vbt.groupby_transform(
                vbt.RepEval("group_by_evenly_nb(wrapper.shape[0], 2)"),
                zscore_ratio_meta_nb,
                vbt.RepEval("to_2d_array(x)"),
                vbt.RepEval("to_2d_array(y)"),
                broadcast_named_args=dict(
                    x=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    y=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(
                    to_2d_array=vbt.base.reshaping.to_2d_array,
                    group_by_evenly_nb=vbt.base.grouping.nb.group_by_evenly_nb,
                ),
            ),
            pd.DataFrame(
                [
                    [1.0, -np.inf, -1.0],
                    [0.0, np.nan, 0.0],
                    [-1.0, np.inf, 1.0],
                    [0.8164965809277261, -np.inf, -0.8164965809277261],
                    [-0.8164965809277261, np.inf, 0.8164965809277261],
                ],
                index=df.index,
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample_apply(self, test_freq):
        @njit
        def mean_nb(x):
            return np.nanmean(x)

        @njit
        def mean_meta_nb(from_i, to_i, col, x):
            return np.nanmean(x[from_i:to_i, col])

        assert_series_equal(
            df["a"].vbt.resample_apply(test_freq, mean_nb),
            df["a"].resample(test_freq).apply(lambda x: mean_nb(x.values)),
        )
        assert_frame_equal(
            df.vbt.resample_apply(test_freq, mean_nb),
            df.resample(test_freq).apply(lambda x: mean_nb(x.values)),
        )
        assert_frame_equal(
            df.vbt.resample_apply(test_freq, mean_nb, use_groupby_apply=True),
            df.vbt.resample_apply(test_freq, mean_nb, use_groupby_apply=False).rename_axis("group"),
        )
        assert_frame_equal(
            df.vbt.resample_apply(df.resample(test_freq), mean_nb, use_groupby_apply=True),
            df.vbt.resample_apply(test_freq, mean_nb, use_groupby_apply=False).rename_axis("group"),
        )
        with pytest.raises(Exception):
            df.vbt.resample_apply(
                vbt.Resampler.from_pd_resampler(df.resample(test_freq)),
                mean_nb,
                use_groupby_apply=True,
            )
        assert_frame_equal(
            df.vbt.resample_apply(df.resample(test_freq), mean_nb),
            df.vbt.resample_apply(test_freq, mean_nb),
        )
        assert_frame_equal(
            df.vbt.resample_apply(vbt.Resampler.from_pd_resampler(df.resample(test_freq)), mean_nb),
            df.vbt.resample_apply(test_freq, mean_nb),
        )
        assert_frame_equal(
            df.vbt.resample_apply(test_freq, mean_nb, jitted=dict(parallel=True)),
            df.vbt.resample_apply(test_freq, mean_nb, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.resample_apply(test_freq, mean_nb, chunked=True),
            df.vbt.resample_apply(test_freq, mean_nb, chunked=False),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_apply(
                test_freq,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.resample_apply(test_freq, mean_nb),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_apply(
                test_freq,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.resample_apply(
                test_freq,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_apply(
                test_freq,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.resample_apply(
                test_freq,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

        @njit
        def mean_ratio_meta_nb(from_i, to_i, col, x, y):
            return np.nanmean(x[from_i:to_i, col]) / np.nanmean(y[from_i:to_i, col])

        assert_frame_equal(
            pd.DataFrame.vbt.resample_apply(
                "2d",
                mean_ratio_meta_nb,
                vbt.RepEval("to_2d_array(x)"),
                vbt.RepEval("to_2d_array(y)"),
                broadcast_named_args=dict(
                    x=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    y=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
            ),
            pd.DataFrame(
                [[1.5, 0.75, 0.5], [3.5, 1.75, 1.1666666666666667], [5.0, 2.5, 1.6666666666666667]],
                index=pd.DatetimeIndex(
                    ["2018-01-01", "2018-01-03", "2018-01-05"], dtype="datetime64[ns]", freq="2D"
                ),
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize(
        "test_freq",
        ["1h", "3d", "7d"],
    )
    def test_realign(self, test_freq):
        target_index = df.resample(test_freq, closed="right", label="right").count().index
        np.testing.assert_array_equal(
            df["a"].vbt.realign(target_index).values,
            df["a"].resample(test_freq, closed="right", label="right").last().ffill().values,
        )
        np.testing.assert_array_equal(
            df.vbt.realign(target_index).values,
            df.resample(test_freq, closed="right", label="right").last().ffill().values,
        )
        np.testing.assert_array_equal(
            df.vbt.realign(target_index, ffill=False).values,
            df.resample(test_freq, closed="right", label="right").last().values,
        )
        np.testing.assert_array_equal(
            df.vbt.realign(target_index, ffill=False, nan_value=-1).values,
            df.resample(test_freq, closed="right", label="right").last().fillna(-1).values,
        )
        np.testing.assert_array_equal(
            df.vbt.realign(target_index, jitted=dict(parallel=True)).values,
            df.vbt.realign(target_index, jitted=dict(parallel=False)).values,
        )
        assert_frame_equal(
            df.vbt.realign(target_index, chunked=True),
            df.vbt.realign(target_index, chunked=False),
        )

    def test_realign_bounds(self):
        source_index = np.array([1, 2, 3])
        target_index = np.array([0.5, 1, 1.5, 2.5, 3.0, 3.5])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=True,
                target_rbound=True,
            ).values,
            np.array([-1, -1, 0, 1, 1, 2]),
        )
        source_index = np.array([1, 2, 3])
        target_index = np.array([1.5, 2, 2.5])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=True,
                target_rbound=True,
            ).values,
            np.array([0, 0, 2]),
        )
        source_index = np.array([0.5, 1, 1.5, 2.5, 3.0, 3.5])
        target_index = np.array([1, 2, 3])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=True,
                target_rbound=True,
            ).values,
            np.array([1, 3, 5]),
        )
        source_index = np.array([1.5, 2, 2.5])
        target_index = np.array([1, 2, 3])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=True,
                target_rbound=True,
            ).values,
            np.array([0, 1, 2]),
        )
        source_index = np.array([1, 2, 3])
        target_index = np.array([0.5, 1, 1.5, 2.5, 3.0, 3.5])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=True,
                target_rbound=False,
            ).values,
            np.array([-1, -1, -1, 0, 1, 1]),
        )
        source_index = np.array([1, 2, 3])
        target_index = np.array([1.5, 2, 2.5])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=True,
                target_rbound=False,
            ).values,
            np.array([-1, 0, 0]),
        )
        source_index = np.array([0.5, 1, 1.5, 2.5, 3.0, 3.5])
        target_index = np.array([1, 2, 3])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=True,
                target_rbound=False,
            ).values,
            np.array([0, 1, 3]),
        )
        source_index = np.array([1.5, 2, 2.5])
        target_index = np.array([1, 2, 3])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=True,
                target_rbound=False,
            ).values,
            np.array([-1, 0, 1]),
        )
        source_index = np.array([1, 2, 3])
        target_index = np.array([0.5, 1, 1.5, 2.5, 3.0, 3.5])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=False,
                target_rbound=True,
            ).values,
            np.array([-1, 0, 1, 1, 2, 2]),
        )
        source_index = np.array([1, 2, 3])
        target_index = np.array([1.5, 2, 2.5])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=False,
                target_rbound=True,
            ).values,
            np.array([0, 1, 2]),
        )
        source_index = np.array([0.5, 1, 1.5, 2.5, 3.0, 3.5])
        target_index = np.array([1, 2, 3])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=False,
                target_rbound=True,
            ).values,
            np.array([2, 3, 5]),
        )
        source_index = np.array([1.5, 2, 2.5])
        target_index = np.array([1, 2, 3])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=False,
                target_rbound=True,
            ).values,
            np.array([0, 2, 2]),
        )
        source_index = np.array([1, 2, 3])
        target_index = np.array([0.5, 1, 1.5, 2.5, 3.0, 3.5])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=False,
                target_rbound=False,
            ).values,
            np.array([-1, 0, 0, 1, 2, 2]),
        )
        source_index = np.array([1, 2, 3])
        target_index = np.array([1.5, 2, 2.5])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=False,
                target_rbound=False,
            ).values,
            np.array([0, 1, 1]),
        )
        source_index = np.array([0.5, 1, 1.5, 2.5, 3.0, 3.5])
        target_index = np.array([1, 2, 3])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=False,
                target_rbound=False,
            ).values,
            np.array([1, 2, 4]),
        )
        source_index = np.array([1.5, 2, 2.5])
        target_index = np.array([1, 2, 3])
        sr = pd.Series(np.arange(len(source_index)), index=source_index)
        np.testing.assert_array_equal(
            sr.vbt.realign(
                target_index,
                nan_value=-1,
                source_rbound=False,
                target_rbound=False,
            ).values,
            np.array([-1, 1, 2]),
        )

    @pytest.mark.parametrize(
        "test_freq",
        ["1h", "3d", "7d"],
    )
    def test_resample_to_index(self, test_freq):
        @njit
        def mean_nb(x):
            return np.nanmean(x)

        @njit
        def mean_meta_nb(from_i, to_i, col, x):
            return np.nanmean(x[from_i:to_i, col])

        target_index = df.resample(test_freq).asfreq().index
        target_index_before = df.resample(test_freq, closed="right", label="right").asfreq().index

        assert_series_equal(
            df["a"].vbt.resample_to_index(target_index, mean_nb),
            df["a"].resample(test_freq).apply(lambda x: mean_nb(x.values)),
        )
        assert_frame_equal(
            df.vbt.resample_to_index(target_index, mean_nb),
            df.resample(test_freq).apply(lambda x: mean_nb(x.values)),
        )
        assert_frame_equal(
            df.vbt.resample_to_index(target_index_before, mean_nb, before=True),
            df.resample(test_freq, closed="right", label="right").apply(
                lambda x: mean_nb(x.values)
            ),
        )
        assert_frame_equal(
            df.vbt.resample_to_index(target_index, mean_nb, jitted=dict(parallel=True)),
            df.vbt.resample_to_index(target_index, mean_nb, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.resample_to_index(target_index, mean_nb, chunked=True),
            df.vbt.resample_to_index(target_index, mean_nb, chunked=False),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_to_index(
                target_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.resample_to_index(target_index, mean_nb),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_to_index(
                target_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.resample_to_index(
                target_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_to_index(
                target_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.resample_to_index(
                target_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

        @njit
        def mean_ratio_meta_nb(from_i, to_i, col, x, y):
            return np.nanmean(x[from_i:to_i, col]) / np.nanmean(y[from_i:to_i, col])

        assert_frame_equal(
            pd.DataFrame.vbt.resample_to_index(
                df.resample("2d").asfreq().index,
                mean_ratio_meta_nb,
                vbt.RepEval("to_2d_array(x)"),
                vbt.RepEval("to_2d_array(y)"),
                broadcast_named_args=dict(
                    x=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    y=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
            ),
            pd.DataFrame(
                [[1.5, 0.75, 0.5], [3.5, 1.75, 1.1666666666666667], [5.0, 2.5, 1.6666666666666667]],
                index=pd.DatetimeIndex(
                    ["2018-01-01", "2018-01-03", "2018-01-05"], dtype="datetime64[ns]", freq="2D"
                ),
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize(
        "test_freq",
        ["1h", "3d", "7d"],
    )
    def test_resample_between_bounds(self, test_freq):
        @njit
        def mean_nb(x):
            return np.nanmean(x)

        @njit
        def mean_meta_nb(from_i, to_i, col, x):
            return np.nanmean(x[from_i:to_i, col])

        target_lbound_index = df.resample(test_freq).asfreq().index
        target_rbound_index = target_lbound_index.shift()

        assert_series_equal(
            df["a"].vbt.resample_between_bounds(target_lbound_index, target_rbound_index, mean_nb),
            df["a"].resample(test_freq).apply(lambda x: mean_nb(x.values)),
        )
        assert_frame_equal(
            df.vbt.resample_between_bounds(target_lbound_index, target_rbound_index, mean_nb),
            df.resample(test_freq).apply(lambda x: mean_nb(x.values)),
        )
        assert_frame_equal(
            df.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index,
                mean_nb,
                closed_lbound=False,
                closed_rbound=True,
            ),
            df.vbt.resample_between_bounds(
                target_lbound_index + pd.Timedelta(nanoseconds=1),
                target_rbound_index + pd.Timedelta(nanoseconds=1),
                mean_nb,
                wrap_kwargs=dict(index=target_rbound_index),
            ),
        )
        assert_frame_equal(
            df.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index,
                mean_nb,
                closed_lbound=False,
                closed_rbound=False,
            ),
            df.vbt.resample_between_bounds(
                target_lbound_index + pd.Timedelta(nanoseconds=1),
                target_rbound_index - pd.Timedelta(nanoseconds=1),
                mean_nb,
                wrap_kwargs=dict(index=target_lbound_index),
            ),
        )
        assert_frame_equal(
            df.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index,
                mean_nb,
                closed_lbound=True,
                closed_rbound=True,
            ),
            df.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index + pd.Timedelta(nanoseconds=1),
                mean_nb,
                wrap_kwargs=dict(index=target_lbound_index),
            ),
        )
        assert_frame_equal(
            df.vbt.resample_between_bounds(
                target_lbound_index, target_rbound_index, mean_nb, jitted=dict(parallel=True)
            ),
            df.vbt.resample_between_bounds(
                target_lbound_index, target_rbound_index, mean_nb, jitted=dict(parallel=False)
            ),
        )
        assert_frame_equal(
            df.vbt.resample_between_bounds(
                target_lbound_index, target_rbound_index, mean_nb, chunked=True
            ),
            df.vbt.resample_between_bounds(
                target_lbound_index, target_rbound_index, mean_nb, chunked=False
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.resample_between_bounds(target_lbound_index, target_rbound_index, mean_nb),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.resample_between_bounds(
                target_lbound_index,
                target_rbound_index,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

        @njit
        def mean_ratio_meta_nb(from_i, to_i, col, x, y):
            return np.nanmean(x[from_i:to_i, col]) / np.nanmean(y[from_i:to_i, col])

        assert_frame_equal(
            pd.DataFrame.vbt.resample_between_bounds(
                df.resample("2d").asfreq().index,
                df.resample("2d").asfreq().index.shift(),
                mean_ratio_meta_nb,
                vbt.RepEval("to_2d_array(x)"),
                vbt.RepEval("to_2d_array(y)"),
                broadcast_named_args=dict(
                    x=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    y=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
            ),
            pd.DataFrame(
                [[1.5, 0.75, 0.5], [3.5, 1.75, 1.1666666666666667], [5.0, 2.5, 1.6666666666666667]],
                index=pd.DatetimeIndex(
                    ["2018-01-01", "2018-01-03", "2018-01-05"], dtype="datetime64[ns]", freq="2D"
                ),
                columns=df.columns,
            ),
        )

    def test_resample_between_bounds_rolling(self):
        @njit
        def mean_nb(x):
            if len(x) < 2:
                return np.nan
            return np.mean(x)

        assert_frame_equal(
            df.vbt.resample_between_bounds(
                pd.DatetimeIndex(
                    [
                        df.index[0],
                        df.index[0],
                        df.index[1],
                        df.index[2],
                        df.index[3],
                    ]
                ),
                df.index,
                mean_nb,
                closed_lbound=True,
                closed_rbound=True,
                wrap_kwargs=dict(index=df.index),
            ),
            df.vbt.rolling_apply(2, mean_nb),
        )

    def test_apply_and_reduce(self):
        @njit
        def every_nth_nb(a, n):
            return a[::n]

        @njit
        def sum_nb(a, b):
            return np.nansum(a) + b

        @njit
        def every_nth_meta_nb(col, a, n):
            return a[::n, col]

        @njit
        def sum_meta_nb(col, a, b):
            return np.nansum(a) + b

        assert (
            df["a"].vbt.apply_and_reduce(every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,))
            == df["a"].iloc[::2].sum() + 3
        )
        assert_series_equal(
            df.vbt.apply_and_reduce(every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,)),
            df.iloc[::2].sum().rename("apply_and_reduce") + 3,
        )
        assert_series_equal(
            df.vbt.apply_and_reduce(
                every_nth_nb,
                sum_nb,
                apply_args=(2,),
                reduce_args=(3,),
                jitted=dict(parallel=True),
            ),
            df.vbt.apply_and_reduce(
                every_nth_nb,
                sum_nb,
                apply_args=(2,),
                reduce_args=(3,),
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                apply_args=vbt.ArgsTaker(
                    None,
                ),
                reduce_args=vbt.ArgsTaker(
                    None,
                ),
            )
        )
        assert_series_equal(
            df.vbt.apply_and_reduce(
                every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,), chunked=chunked
            ),
            df.vbt.apply_and_reduce(
                every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,), chunked=False
            ),
        )
        assert_series_equal(
            pd.DataFrame.vbt.apply_and_reduce(
                every_nth_meta_nb,
                sum_meta_nb,
                apply_args=(
                    df.vbt.to_2d_array(),
                    2,
                ),
                reduce_args=(3,),
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.apply_and_reduce(every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,)),
        )
        assert_series_equal(
            pd.DataFrame.vbt.apply_and_reduce(
                every_nth_meta_nb,
                sum_meta_nb,
                apply_args=(
                    df.vbt.to_2d_array(),
                    2,
                ),
                reduce_args=(3,),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.apply_and_reduce(
                every_nth_meta_nb,
                sum_meta_nb,
                apply_args=(
                    df.vbt.to_2d_array(),
                    2,
                ),
                reduce_args=(3,),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                apply_args=vbt.ArgsTaker(vbt.ArraySlicer(axis=1), None),
                reduce_args=vbt.ArgsTaker(
                    None,
                ),
            )
        )
        assert_series_equal(
            pd.DataFrame.vbt.apply_and_reduce(
                every_nth_meta_nb,
                sum_meta_nb,
                apply_args=(
                    df.vbt.to_2d_array(),
                    2,
                ),
                reduce_args=(3,),
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.apply_and_reduce(
                every_nth_meta_nb,
                sum_meta_nb,
                apply_args=(
                    df.vbt.to_2d_array(),
                    2,
                ),
                reduce_args=(3,),
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

        @njit
        def every_2nd_sum_meta_nb(col, a, b):
            return a[::2, col] + b[::2, col]

        @njit
        def sum_meta2_nb(col, a):
            return np.nansum(a)

        assert_series_equal(
            pd.DataFrame.vbt.apply_and_reduce(
                every_2nd_sum_meta_nb,
                sum_meta2_nb,
                apply_args=(vbt.RepEval("to_2d_array(a)"), vbt.RepEval("to_2d_array(b)")),
                broadcast_named_args=dict(
                    a=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    b=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
            ),
            pd.Series([12, 15, 18], index=df.columns, name="apply_and_reduce"),
        )

    def test_reduce(self):
        @njit
        def sum_nb(a):
            return np.nansum(a)

        @njit
        def sum_meta_nb(col, a):
            return np.nansum(a[:, col])

        @njit
        def sum_grouped_meta_nb(group_idxs, group, a):
            return np.nansum(a[:, group_idxs])

        assert df["a"].vbt.reduce(sum_nb) == df["a"].sum()
        assert_series_equal(df.vbt.reduce(sum_nb), df.sum().rename("reduce"))
        assert_series_equal(
            df.vbt.reduce(sum_nb, jitted=dict(parallel=True)),
            df.vbt.reduce(sum_nb, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            df.vbt.reduce(sum_nb, chunked=True), df.vbt.reduce(sum_nb, chunked=False)
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(sum_meta_nb, df.vbt.to_2d_array(), wrapper=df.vbt.wrapper),
            df.vbt.reduce(sum_nb),
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                sum_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.reduce(
                sum_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        count_chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                sum_meta_nb, df.vbt.to_2d_array(), wrapper=df.vbt.wrapper, chunked=count_chunked
            ),
            pd.DataFrame.vbt.reduce(
                sum_meta_nb, df.vbt.to_2d_array(), wrapper=df.vbt.wrapper, chunked=False
            ),
        )
        assert_series_equal(
            df.vbt.reduce(sum_nb, group_by=group_by),
            pd.Series([20.0, 6.0], index=pd.Index(["g1", "g2"], name="group")).rename("reduce"),
        )
        assert_series_equal(
            df.vbt.reduce(sum_nb, group_by=group_by, flatten=True, order="C"),
            df.vbt.reduce(sum_nb, group_by=group_by),
        )
        assert_series_equal(
            df.vbt.reduce(sum_nb, group_by=group_by, flatten=True, order="F"),
            df.vbt.reduce(sum_nb, group_by=group_by),
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                sum_grouped_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                group_by=group_by,
            ),
            df.vbt.reduce(sum_nb, group_by=group_by),
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                sum_grouped_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.reduce(
                sum_grouped_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query=0)),
                )
            )
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                sum_grouped_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.reduce(
                sum_grouped_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                chunked=False,
            ),
        )

        @njit
        def sum_meta2_nb(col, a, b):
            return np.nansum(a[:, col]) + np.nansum(b[:, col])

        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                sum_meta2_nb,
                vbt.RepEval("to_2d_array(a)"),
                vbt.RepEval("to_2d_array(b)"),
                broadcast_named_args=dict(
                    a=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    b=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
            ),
            pd.Series([20, 25, 30], index=df.columns, name="reduce"),
        )

        @njit
        def sum_grouped_meta2_nb(group_idxs, group, a, b):
            return np.nansum(a[:, group_idxs]) + np.nansum(b[:, group_idxs])

        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                sum_grouped_meta2_nb,
                vbt.RepEval("to_2d_array(a)"),
                vbt.RepEval("to_2d_array(b)"),
                broadcast_named_args=dict(
                    a=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    b=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
                group_by=group_by,
            ),
            pd.Series([45, 30], index=pd.Index(["g1", "g2"], name="group"), name="reduce"),
        )

    def test_reduce_to_idx(self):
        @njit
        def argmax_nb(a):
            a = a.copy()
            a[np.isnan(a)] = -np.inf
            return np.argmax(a)

        @njit
        def argmax_meta_nb(col, a):
            a = a[:, col].copy()
            a[np.isnan(a)] = -np.inf
            return np.argmax(a)

        @njit
        def argmax_grouped_meta_nb(group_idxs, group, a):
            a = a[:, group_idxs].flatten()
            a[np.isnan(a)] = -np.inf
            return np.argmax(a) // len(group_idxs)

        assert df["a"].vbt.reduce(argmax_nb, returns_idx=True) == df["a"].idxmax()
        assert_series_equal(
            df.vbt.reduce(argmax_nb, returns_idx=True), df.idxmax().rename("reduce")
        )
        assert_series_equal(
            df.vbt.reduce(argmax_nb, returns_idx=True, jitted=dict(parallel=True)),
            df.vbt.reduce(argmax_nb, returns_idx=True, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            df.vbt.reduce(argmax_nb, returns_idx=True, chunked=True),
            df.vbt.reduce(argmax_nb, returns_idx=True, chunked=False),
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                argmax_meta_nb, df.vbt.to_2d_array(), returns_idx=True, wrapper=df.vbt.wrapper
            ),
            df.vbt.reduce(argmax_nb, returns_idx=True),
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.reduce(
                argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        count_chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                chunked=count_chunked,
            ),
            pd.DataFrame.vbt.reduce(
                argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )
        assert_series_equal(
            df.vbt.reduce(argmax_nb, returns_idx=True, group_by=group_by, flatten=True, order="C"),
            pd.Series(
                ["2018-01-02", "2018-01-02"],
                dtype="datetime64[ns]",
                index=pd.Index(["g1", "g2"], name="group"),
            ).rename("reduce"),
        )
        assert_series_equal(
            df.vbt.reduce(argmax_nb, returns_idx=True, group_by=group_by, flatten=True, order="F"),
            pd.Series(
                ["2018-01-04", "2018-01-02"],
                dtype="datetime64[ns]",
                index=pd.Index(["g1", "g2"], name="group"),
            ).rename("reduce"),
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
            ),
            df.vbt.reduce(argmax_nb, group_by=group_by, returns_idx=True, flatten=True),
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                flatten=True,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.reduce(
                argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                flatten=True,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query=0)),
                )
            )
        )
        assert_series_equal(
            pd.DataFrame.vbt.reduce(
                argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                flatten=True,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.reduce(
                argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                flatten=True,
                chunked=False,
            ),
        )

    def test_reduce_to_array(self):
        @njit
        def min_and_max_nb(a):
            out = np.empty(2)
            out[0] = np.nanmin(a)
            out[1] = np.nanmax(a)
            return out

        @njit
        def min_and_max_meta_nb(col, a):
            out = np.empty(2)
            out[0] = np.nanmin(a[:, col])
            out[1] = np.nanmax(a[:, col])
            return out

        @njit
        def min_and_max_grouped_meta_nb(group_idxs, group, a):
            out = np.empty(2)
            out[0] = np.nanmin(a[:, group_idxs])
            out[1] = np.nanmax(a[:, group_idxs])
            return out

        assert_series_equal(
            df["a"].vbt.reduce(
                min_and_max_nb, returns_array=True, wrap_kwargs=dict(name_or_index=["min", "max"])
            ),
            pd.Series([np.nanmin(df["a"]), np.nanmax(df["a"])], index=["min", "max"], name="a"),
        )
        assert_frame_equal(
            df.vbt.reduce(
                min_and_max_nb, returns_array=True, wrap_kwargs=dict(name_or_index=["min", "max"])
            ),
            df.apply(
                lambda x: pd.Series(np.array([np.nanmin(x), np.nanmax(x)]), index=["min", "max"]),
                axis=0,
            ),
        )
        assert_frame_equal(
            df.vbt.reduce(min_and_max_nb, returns_array=True, jitted=dict(parallel=True)),
            df.vbt.reduce(min_and_max_nb, returns_array=True, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.reduce(min_and_max_nb, returns_array=True, chunked=True),
            df.vbt.reduce(min_and_max_nb, returns_array=True, chunked=False),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                min_and_max_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.reduce(min_and_max_nb, returns_array=True),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                min_and_max_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.reduce(
                min_and_max_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        count_chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                min_and_max_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                chunked=count_chunked,
            ),
            pd.DataFrame.vbt.reduce(
                min_and_max_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )
        assert_frame_equal(
            df.vbt.reduce(
                min_and_max_nb,
                returns_array=True,
                group_by=group_by,
                wrap_kwargs=dict(name_or_index=["min", "max"]),
            ),
            pd.DataFrame(
                [[1.0, 1.0], [4.0, 2.0]],
                index=["min", "max"],
                columns=pd.Index(["g1", "g2"], name="group"),
            ),
        )
        assert_frame_equal(
            df.vbt.reduce(
                min_and_max_nb, returns_array=True, group_by=group_by, flatten=True, order="C"
            ),
            df.vbt.reduce(min_and_max_nb, returns_array=True, group_by=group_by),
        )
        assert_frame_equal(
            df.vbt.reduce(
                min_and_max_nb, returns_array=True, group_by=group_by, flatten=True, order="F"
            ),
            df.vbt.reduce(min_and_max_nb, returns_array=True, group_by=group_by),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                min_and_max_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
            ),
            df.vbt.reduce(min_and_max_nb, returns_array=True, group_by=group_by),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                min_and_max_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.reduce(
                min_and_max_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query=0)),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                min_and_max_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.reduce(
                min_and_max_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                chunked=False,
            ),
        )

    def test_reduce_to_idx_array(self):
        @njit
        def argmin_and_argmax_nb(a):
            # nanargmin and nanargmax
            out = np.empty(2)
            _a = a.copy()
            _a[np.isnan(_a)] = np.inf
            out[0] = np.argmin(_a)
            _a = a.copy()
            _a[np.isnan(_a)] = -np.inf
            out[1] = np.argmax(_a)
            return out

        @njit
        def argmin_and_argmax_meta_nb(col, a):
            # nanargmin and nanargmax
            out = np.empty(2)
            _a = a[:, col].copy()
            _a[np.isnan(_a)] = np.inf
            out[0] = np.argmin(_a)
            _a = a[:, col].copy()
            _a[np.isnan(_a)] = -np.inf
            out[1] = np.argmax(_a)
            return out

        @njit
        def argmin_and_argmax_grouped_meta_nb(group_idxs, group, a):
            # nanargmin and nanargmax
            out = np.empty(2)
            _a = a[:, group_idxs].flatten()
            _a[np.isnan(_a)] = np.inf
            out[0] = np.argmin(_a)
            _a = a[:, group_idxs].flatten()
            _a[np.isnan(_a)] = -np.inf
            out[1] = np.argmax(_a)
            return out // len(group_idxs)

        assert_series_equal(
            df["a"].vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                wrap_kwargs=dict(name_or_index=["idxmin", "idxmax"]),
            ),
            pd.Series([df["a"].idxmin(), df["a"].idxmax()], index=["idxmin", "idxmax"], name="a"),
        )
        assert_frame_equal(
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                wrap_kwargs=dict(name_or_index=["idxmin", "idxmax"]),
            ),
            df.apply(
                lambda x: pd.Series(np.array([x.idxmin(), x.idxmax()]), index=["idxmin", "idxmax"]),
                axis=0,
            ),
        )
        assert_frame_equal(
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                jitted=dict(parallel=True),
            ),
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                jitted=dict(parallel=False),
            ),
        )
        assert_frame_equal(
            df.vbt.reduce(argmin_and_argmax_nb, returns_idx=True, returns_array=True, chunked=True),
            df.vbt.reduce(
                argmin_and_argmax_nb, returns_idx=True, returns_array=True, chunked=False
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.reduce(argmin_and_argmax_nb, returns_idx=True, returns_array=True),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        count_chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                chunked=count_chunked,
            ),
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )
        assert_frame_equal(
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                flatten=True,
                order="C",
                group_by=group_by,
                wrap_kwargs=dict(name_or_index=["idxmin", "idxmax"]),
            ),
            pd.DataFrame(
                [["2018-01-01", "2018-01-01"], ["2018-01-02", "2018-01-02"]],
                dtype="datetime64[ns]",
                index=["idxmin", "idxmax"],
                columns=pd.Index(["g1", "g2"], name="group"),
            ),
        )
        assert_frame_equal(
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                flatten=True,
                order="F",
                group_by=group_by,
                wrap_kwargs=dict(name_or_index=["idxmin", "idxmax"]),
            ),
            pd.DataFrame(
                [["2018-01-01", "2018-01-01"], ["2018-01-04", "2018-01-02"]],
                dtype="datetime64[ns]",
                index=["idxmin", "idxmax"],
                columns=pd.Index(["g1", "g2"], name="group"),
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
            ),
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                group_by=group_by,
                flatten=True,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query=0)),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.reduce(
                argmin_and_argmax_grouped_meta_nb,
                df.vbt.to_2d_array(),
                returns_idx=True,
                returns_array=True,
                wrapper=df.vbt.wrapper,
                group_by=group_by,
                chunked=False,
            ),
        )

    def test_proximity_apply(self):
        @njit
        def mean_nb(a):
            return np.nanmean(a)

        @njit
        def mean_meta_nb(from_i, to_i, from_col, to_col, a):
            return np.nanmean(a[from_i:to_i, from_col:to_col])

        assert_series_equal(
            df["a"].vbt.proximity_apply(1, mean_nb),
            pd.Series([1.5, 2.0, 3.0, 3.5, 4.0], index=df.index, name="a"),
        )
        assert_series_equal(
            df["a"].vbt.proximity_apply(2, mean_nb),
            pd.Series([2.0, 2.5, 2.5, 3.0, 3.5], index=df.index, name="a"),
        )
        assert_frame_equal(
            df.vbt.proximity_apply(1, mean_nb),
            pd.DataFrame(
                [
                    [2.3333333333333335, 2.0, 2.3333333333333335],
                    [2.6, 2.2857142857142856, 2.5],
                    [3.0, 2.75, 2.6],
                    [2.6, 2.2857142857142856, 1.8],
                    [2.3333333333333335, 2.0, 1.5],
                ],
                index=df.index,
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.proximity_apply(2, mean_nb),
            pd.DataFrame(
                [
                    [2.2857142857142856, 2.2857142857142856, 2.2857142857142856],
                    [2.4, 2.4, 2.4],
                    [2.1666666666666665, 2.1666666666666665, 2.1666666666666665],
                    [2.4, 2.4, 2.4],
                    [2.2857142857142856, 2.2857142857142856, 2.2857142857142856],
                ],
                index=df.index,
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.proximity_apply(1, mean_nb, jitted=dict(parallel=True)),
            df.vbt.proximity_apply(1, mean_nb, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            pd.Series.vbt.proximity_apply(
                1,
                mean_meta_nb,
                df["a"].vbt.to_2d_array(),
                wrapper=df["a"].vbt.wrapper,
            ),
            df["a"].vbt.proximity_apply(1, mean_nb),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.proximity_apply(
                1,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.proximity_apply(1, mean_nb),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.proximity_apply(
                1,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.proximity_apply(
                1,
                mean_meta_nb,
                df.vbt.to_2d_array(),
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )

    def test_squeeze_grouped(self):
        @njit
        def mean_nb(a):
            return np.nanmean(a)

        @njit
        def mean_grouped_meta_nb(i, group_idxs, group, a):
            return np.nanmean(a[i][group_idxs])

        assert_frame_equal(
            df.vbt.squeeze_grouped(mean_nb, group_by=group_by),
            pd.DataFrame(
                [[1.0, 1.0], [3.0, 2.0], [3.0, np.nan], [3.0, 2.0], [1.0, 1.0]],
                index=df.index,
                columns=pd.Index(["g1", "g2"], name="group"),
            ),
        )
        assert_frame_equal(
            df.vbt.squeeze_grouped(mean_nb, group_by=group_by, jitted=dict(parallel=True)),
            df.vbt.squeeze_grouped(mean_nb, group_by=group_by, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.squeeze_grouped(mean_nb, group_by=group_by, chunked=True),
            df.vbt.squeeze_grouped(mean_nb, group_by=group_by, chunked=False),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.squeeze_grouped(
                mean_grouped_meta_nb,
                df.vbt.to_2d_array(),
                group_by=group_by,
                wrapper=df.vbt.wrapper,
            ),
            df.vbt.squeeze_grouped(mean_nb, group_by=group_by),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.squeeze_grouped(
                mean_grouped_meta_nb,
                df.vbt.to_2d_array(),
                group_by=group_by,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=True),
            ),
            pd.DataFrame.vbt.squeeze_grouped(
                mean_grouped_meta_nb,
                df.vbt.to_2d_array(),
                group_by=group_by,
                wrapper=df.vbt.wrapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query=1)),
                )
            )
        )
        assert_frame_equal(
            pd.DataFrame.vbt.squeeze_grouped(
                mean_grouped_meta_nb,
                df.vbt.to_2d_array(),
                group_by=group_by,
                wrapper=df.vbt.wrapper,
                chunked=chunked,
            ),
            pd.DataFrame.vbt.squeeze_grouped(
                mean_grouped_meta_nb,
                df.vbt.to_2d_array(),
                group_by=group_by,
                wrapper=df.vbt.wrapper,
                chunked=False,
            ),
        )

        @njit
        def sum_grouped_meta_nb(i, group_idxs, group, a, b):
            return np.nansum(a[i][group_idxs]) + np.nansum(b[i][group_idxs])

        assert_frame_equal(
            pd.DataFrame.vbt.squeeze_grouped(
                sum_grouped_meta_nb,
                vbt.RepEval("to_2d_array(a)"),
                vbt.RepEval("to_2d_array(b)"),
                broadcast_named_args=dict(
                    a=pd.Series([1, 2, 3, 4, 5], index=df.index),
                    b=pd.DataFrame([[1, 2, 3]], columns=df.columns),
                ),
                template_context=dict(to_2d_array=vbt.base.reshaping.to_2d_array),
                group_by=group_by,
            ),
            pd.DataFrame(
                [[5, 4], [7, 5], [9, 6], [11, 7], [13, 8]],
                index=df.index,
                columns=pd.Index(["g1", "g2"], name="group"),
            ),
        )

    def test_flatten_grouped(self):
        assert_frame_equal(
            df.vbt.flatten_grouped(group_by=group_by, order="C"),
            pd.DataFrame(
                [
                    [1.0, 1.0],
                    [np.nan, np.nan],
                    [2.0, 2.0],
                    [4.0, np.nan],
                    [3.0, np.nan],
                    [3.0, np.nan],
                    [4.0, 2.0],
                    [2.0, np.nan],
                    [np.nan, 1.0],
                    [1.0, np.nan],
                ],
                index=np.repeat(df.index, 2),
                columns=pd.Index(["g1", "g2"], name="group"),
            ),
        )
        assert_frame_equal(
            df.vbt.flatten_grouped(group_by=group_by, order="F"),
            pd.DataFrame(
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, np.nan],
                    [4.0, 2.0],
                    [np.nan, 1.0],
                    [np.nan, np.nan],
                    [4.0, np.nan],
                    [3.0, np.nan],
                    [2.0, np.nan],
                    [1.0, np.nan],
                ],
                index=np.tile(df.index, 2),
                columns=pd.Index(["g1", "g2"], name="group"),
            ),
        )
        assert_series_equal(
            pd.DataFrame([[False, True], [False, True]]).vbt.flatten_grouped(
                group_by=True, order="C"
            ),
            pd.Series([False, True, False, True], name="group"),
        )
        assert_series_equal(
            pd.DataFrame([[False, True], [False, True]]).vbt.flatten_grouped(
                group_by=True, order="F"
            ),
            pd.Series([False, False, True, True], name="group"),
        )

    @pytest.mark.parametrize(
        "test_name,test_func",
        [
            ("min", lambda x, **kwargs: x.min(**kwargs)),
            ("max", lambda x, **kwargs: x.max(**kwargs)),
            ("mean", lambda x, **kwargs: x.mean(**kwargs)),
            ("median", lambda x, **kwargs: x.median(**kwargs)),
            ("std", lambda x, **kwargs: x.std(**kwargs, ddof=0)),
            ("count", lambda x, **kwargs: x.count(**kwargs)),
            ("sum", lambda x, **kwargs: x.sum(**kwargs)),
        ],
    )
    def test_funcs(self, test_name, test_func):
        # numeric
        assert test_func(df["a"].vbt) == test_func(df["a"])
        assert_series_equal(test_func(df.vbt), test_func(df).rename(test_name))
        assert_series_equal(
            test_func(df.vbt, group_by=group_by),
            pd.Series(
                [test_func(df[["a", "b"]].stack()), test_func(df["c"])],
                index=pd.Index(["g1", "g2"], name="group"),
            ).rename(test_name),
        )
        assert_series_equal(test_func(df.vbt, use_jitted=True), test_func(df.vbt, use_jitted=False))
        assert_series_equal(
            test_func(df.vbt, use_jitted=True, jitted=dict(parallel=True)),
            test_func(df.vbt, use_jitted=True, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            test_func(df.vbt, use_jitted=True, chunked=True),
            test_func(df.vbt, use_jitted=True, chunked=False),
        )
        assert_series_equal(
            test_func(df.vbt, wrap_kwargs=dict(to_timedelta=True)),
            test_func(df).rename(test_name) * day_dt,
        )
        # boolean
        bool_ts = df == df
        assert test_func(bool_ts["a"].vbt) == test_func(bool_ts["a"])
        assert_series_equal(test_func(bool_ts.vbt), test_func(bool_ts).rename(test_name))

    def test_corr(self):
        df2 = df[["b", "c", "a"]].rename(columns={"b": "a", "c": "b", "a": "c"})
        assert df["a"].vbt.corr(df2["a"]) == df["a"].corr(df2["a"])
        assert_series_equal(df.vbt.corr(df2), df.corrwith(df2).rename("corr"))
        assert_series_equal(
            df.vbt.corr(df2, jitted=dict(parallel=True)),
            df.vbt.corr(df2, jitted=dict(parallel=False)),
        )
        assert_series_equal(df.vbt.corr(df2, chunked=True), df.vbt.corr(df2, chunked=False))
        flatten1 = pd.Series(df[["a", "b"]].values.flatten())
        flatten2 = pd.Series(df2[["a", "b"]].values.flatten())
        assert_series_equal(
            df.vbt.corr(df2, group_by=group_by),
            pd.Series(
                [flatten1.vbt.corr(flatten2), df["c"].vbt.corr(df2["c"])],
                index=pd.Index(["g1", "g2"], name="group"),
            ).rename("corr"),
        )

    @pytest.mark.parametrize("test_ddof", [0, 1])
    def test_cov(self, test_ddof):
        df2 = df[["b", "c", "a"]].rename(columns={"b": "a", "c": "b", "a": "c"})
        assert df["a"].vbt.cov(df2["a"], ddof=test_ddof) == df["a"].cov(df2["a"], ddof=test_ddof)
        assert_series_equal(
            df.vbt.cov(df2, ddof=test_ddof),
            pd.Series(
                [
                    df["a"].cov(df2["a"], ddof=test_ddof),
                    df["b"].cov(df2["b"], ddof=test_ddof),
                    df["c"].cov(df2["c"], ddof=test_ddof),
                ],
                index=pd.Index(["a", "b", "c"], dtype="object"),
                name="cov",
            ),
        )
        assert_series_equal(
            df.vbt.cov(df2, ddof=test_ddof, jitted=dict(parallel=True)),
            df.vbt.cov(df2, ddof=test_ddof, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            df.vbt.cov(df2, ddof=test_ddof, chunked=True),
            df.vbt.cov(df2, ddof=test_ddof, chunked=False),
        )
        flatten1 = pd.Series(df[["a", "b"]].values.flatten())
        flatten2 = pd.Series(df2[["a", "b"]].values.flatten())
        assert_series_equal(
            df.vbt.cov(df2, ddof=test_ddof, group_by=group_by),
            pd.Series(
                [
                    flatten1.vbt.cov(flatten2, ddof=test_ddof),
                    df["c"].vbt.cov(df2["c"], ddof=test_ddof),
                ],
                index=pd.Index(["g1", "g2"], name="group"),
            ).rename("cov"),
        )

    @pytest.mark.parametrize("test_pct", [False, True])
    def test_rank(self, test_pct):
        assert_series_equal(df["a"].vbt.rank(pct=test_pct), df["a"].rank(pct=test_pct))
        assert_frame_equal(df.vbt.rank(pct=test_pct), df.rank(pct=test_pct))
        assert_frame_equal(df.vbt.rank(), df.rank())
        assert_frame_equal(
            df.vbt.rank(jitted=dict(parallel=True)), df.vbt.rank(jitted=dict(parallel=False))
        )
        assert_frame_equal(df.vbt.rank(chunked=True), df.vbt.rank(chunked=False))

    @pytest.mark.parametrize(
        "test_name,test_func",
        [
            ("idxmin", lambda x, **kwargs: x.idxmin(**kwargs)),
            ("idxmax", lambda x, **kwargs: x.idxmax(**kwargs)),
        ],
    )
    def test_arg_funcs(self, test_name, test_func):
        assert test_func(df["a"].vbt) == test_func(df["a"])
        assert_series_equal(test_func(df.vbt), test_func(df).rename(test_name))
        assert_series_equal(test_func(df.vbt, chunked=True), test_func(df.vbt, chunked=False))
        assert_series_equal(
            test_func(df.vbt, group_by=group_by),
            pd.Series(
                [test_func(df[["a", "b"]].stack())[0], test_func(df["c"])],
                index=pd.Index(["g1", "g2"], name="group"),
                dtype="datetime64[ns]",
            ).rename(test_name),
        )

    def test_describe(self):
        assert_series_equal(df["a"].vbt.describe(), df["a"].describe())
        assert_frame_equal(df.vbt.describe(percentiles=None), df.describe(percentiles=None))
        assert_frame_equal(df.vbt.describe(percentiles=[]), df.describe(percentiles=[]))
        test_against = df.describe(percentiles=np.arange(0, 1, 0.1))
        assert_frame_equal(df.vbt.describe(percentiles=np.arange(0, 1, 0.1)), test_against)
        assert_frame_equal(
            df.vbt.describe(percentiles=np.arange(0, 1, 0.1), group_by=group_by),
            pd.DataFrame(
                {
                    "g1": df[["a", "b"]].stack().describe(percentiles=np.arange(0, 1, 0.1)).values,
                    "g2": df["c"].describe(percentiles=np.arange(0, 1, 0.1)).values,
                },
                index=test_against.index,
                columns=pd.Index(["g1", "g2"], name="group"),
            ),
        )
        assert_frame_equal(
            df.vbt.describe(jitted=dict(parallel=True)),
            df.vbt.describe(jitted=dict(parallel=False)),
        )
        assert_frame_equal(df.vbt.describe(chunked=True), df.vbt.describe(chunked=False))
        assert_frame_equal(
            df.vbt.describe(group_by=group_by, jitted=dict(parallel=True)),
            df.vbt.describe(group_by=group_by, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.describe(group_by=group_by, chunked=True),
            df.vbt.describe(group_by=group_by, chunked=False),
        )

    def test_value_counts(self):
        assert_series_equal(
            df["a"].vbt.value_counts(),
            pd.Series(
                np.array([1, 1, 1, 1, 1]),
                index=pd.Index([1.0, 2.0, 3.0, 4.0, np.nan], dtype="float64"),
                name="a",
            ),
        )
        mapping = {1.0: "one", 2.0: "two", 3.0: "three", 4.0: "four"}
        assert_series_equal(
            df["a"].vbt.value_counts(mapping=mapping),
            pd.Series(
                np.array([1, 1, 1, 1, 1]),
                index=pd.Index(["one", "two", "three", "four", None], dtype="object"),
                name="a",
            ),
        )
        assert_frame_equal(
            df.vbt.value_counts(),
            pd.DataFrame(
                np.array([[1, 1, 2], [1, 1, 2], [1, 1, 0], [1, 1, 0], [1, 1, 1]]),
                index=pd.Index([1.0, 2.0, 3.0, 4.0, np.nan], dtype="float64"),
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.value_counts(jitted=dict(parallel=True)),
            df.vbt.value_counts(jitted=dict(parallel=False)),
        )
        assert_frame_equal(df.vbt.value_counts(chunked=True), df.vbt.value_counts(chunked=False))
        assert_frame_equal(
            df.vbt.value_counts(axis=0),
            pd.DataFrame(
                np.array(
                    [
                        [2, 0, 0, 0, 2],
                        [0, 2, 0, 2, 0],
                        [0, 0, 2, 0, 0],
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                    ]
                ),
                index=pd.Index([1.0, 2.0, 3.0, 4.0, np.nan], dtype="float64"),
                columns=df.index,
            ),
        )
        assert_frame_equal(
            df.vbt.value_counts(axis=0, jitted=dict(parallel=True)),
            df.vbt.value_counts(axis=0, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.value_counts(axis=0, chunked=True),
            df.vbt.value_counts(axis=0, chunked=False),
        )
        assert_series_equal(
            df.vbt.value_counts(axis=-1),
            pd.Series(
                np.array([4, 4, 2, 2, 3]),
                index=pd.Index([1.0, 2.0, 3.0, 4.0, np.nan], dtype="float64"),
                name="value_counts",
            ),
        )
        assert_series_equal(
            df.vbt.value_counts(axis=-1, jitted=dict(parallel=True)),
            df.vbt.value_counts(axis=-1, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.value_counts(group_by=group_by),
            pd.DataFrame(
                np.array([[2, 2], [2, 2], [2, 0], [2, 0], [2, 1]]),
                index=pd.Index([1.0, 2.0, 3.0, 4.0, np.nan], dtype="float64"),
                columns=pd.Index(["g1", "g2"], dtype="object", name="group"),
            ),
        )
        assert_frame_equal(
            df.vbt.value_counts(sort_uniques=False),
            pd.DataFrame(
                np.array([[1, 1, 2], [1, 1, 1], [1, 1, 2], [1, 1, 0], [1, 1, 0]]),
                index=pd.Index([1.0, np.nan, 2.0, 4.0, 3.0], dtype="float64"),
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.value_counts(sort=True),
            pd.DataFrame(
                np.array([[1, 1, 2], [1, 1, 2], [1, 1, 1], [1, 1, 0], [1, 1, 0]]),
                index=pd.Index([1.0, 2.0, np.nan, 3.0, 4.0], dtype="float64"),
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.value_counts(sort=True, ascending=True),
            pd.DataFrame(
                np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 2]]),
                index=pd.Index([3.0, 4.0, np.nan, 1.0, 2.0], dtype="float64"),
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.value_counts(sort=True, normalize=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.06666666666666667, 0.06666666666666667, 0.13333333333333333],
                        [0.06666666666666667, 0.06666666666666667, 0.13333333333333333],
                        [0.06666666666666667, 0.06666666666666667, 0.06666666666666667],
                        [0.06666666666666667, 0.06666666666666667, 0.0],
                        [0.06666666666666667, 0.06666666666666667, 0.0],
                    ]
                ),
                index=pd.Index([1.0, 2.0, np.nan, 3.0, 4.0], dtype="float64"),
                columns=df.columns,
            ),
        )
        assert_frame_equal(
            df.vbt.value_counts(sort=True, normalize=True, dropna=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.08333333333333333, 0.08333333333333333, 0.16666666666666666],
                        [0.08333333333333333, 0.08333333333333333, 0.16666666666666666],
                        [0.08333333333333333, 0.08333333333333333, 0.0],
                        [0.08333333333333333, 0.08333333333333333, 0.0],
                    ]
                ),
                index=pd.Index([1.0, 2.0, 3.0, 4.0], dtype="float64"),
                columns=df.columns,
            ),
        )

    def test_demean(self):
        assert_frame_equal(
            df.vbt.demean(group_by=group_by),
            pd.DataFrame(
                {
                    "a": df["a"].values - df[["a", "b"]].mean(axis=1).values,
                    "b": df["b"].values - df[["a", "b"]].mean(axis=1).values,
                    "c": df["c"].values - df["c"].values,
                },
                index=df.index,
            ),
        )
        assert_frame_equal(
            df.vbt.demean(jitted=dict(parallel=True)),
            df.vbt.demean(jitted=dict(parallel=False)),
        )
        assert_frame_equal(df.vbt.demean(chunked=True), df.vbt.demean(chunked=False))

    def test_to_renko(self):
        sr = pd.Series([np.nan, 1, 2, 3, 2, 1, np.nan, 2, 3, 2])
        sr.index = pd.date_range("2020", periods=len(sr))
        assert_series_equal(
            sr.vbt.to_renko(1),
            pd.Series(
                [2.0, 3.0, 1.0, 3.0],
                pd.DatetimeIndex(
                    ["2020-01-03", "2020-01-04", "2020-01-06", "2020-01-09"],
                    dtype="datetime64[ns]",
                    freq=None,
                ),
            ),
        )
        assert_series_equal(
            sr.vbt.to_renko(np.array([1])),
            pd.Series(
                [2.0, 3.0, 1.0, 3.0],
                pd.DatetimeIndex(
                    ["2020-01-03", "2020-01-04", "2020-01-06", "2020-01-09"],
                    dtype="datetime64[ns]",
                    freq=None,
                ),
            ),
        )
        assert_series_equal(
            sr.vbt.to_renko(0.5, relative=True),
            pd.Series(
                [1.5, 2.25, 2.53125],
                pd.DatetimeIndex(
                    ["2020-01-03", "2020-01-04", "2020-01-09"], dtype="datetime64[ns]", freq=None
                ),
            ),
        )

    def test_drawdown(self):
        assert_series_equal(df["a"].vbt.drawdown(), df["a"] / df["a"].expanding().max() - 1)
        assert_frame_equal(df.vbt.drawdown(), df / df.expanding().max() - 1)
        assert_frame_equal(
            df.vbt.drawdown(jitted=dict(parallel=True)),
            df.vbt.drawdown(jitted=dict(parallel=False)),
        )
        assert_frame_equal(df.vbt.drawdown(chunked=True), df.vbt.drawdown(chunked=False))

    def test_drawdowns(self):
        assert type(df["a"].vbt.drawdowns) is vbt.Drawdowns
        assert df["a"].vbt.drawdowns.wrapper.freq == df["a"].vbt.wrapper.freq
        assert df["a"].vbt.drawdowns.wrapper.ndim == df["a"].ndim
        assert df.vbt.drawdowns.wrapper.ndim == df.ndim

    def test_to_mapped(self):
        np.testing.assert_array_equal(
            df.vbt.to_mapped().values,
            np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0]),
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped().col_arr, np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped().idx_arr, np.array([0, 1, 2, 3, 1, 2, 3, 4, 0, 1, 3, 4])
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped(dropna=False).values,
            np.array(
                [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, np.nan, 2.0, 1.0]
            ),
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped(dropna=False).col_arr,
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped(dropna=False).idx_arr,
            np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
        )

    def test_zscore(self):
        assert_series_equal(df["a"].vbt.zscore(), (df["a"] - df["a"].mean()) / df["a"].std(ddof=0))
        assert_frame_equal(df.vbt.zscore(), (df - df.mean()) / df.std(ddof=0))

    def test_crossed_above(self):
        sr1 = pd.Series([np.nan, 3, 2, 1, 2, 3, 4])
        sr2 = pd.Series([1, 2, 3, 4, 3, 2, 1])
        assert_series_equal(
            sr1.vbt.crossed_above(sr2),
            pd.Series([False, False, False, False, False, True, False]),
        )
        assert_series_equal(
            sr1.vbt.crossed_above(sr2, wait=1),
            pd.Series([False, False, False, False, False, False, True]),
        )
        sr3 = pd.Series([1, 2, 3, np.nan, 5, 1, 5])
        sr4 = pd.Series([3, 2, 1, 1, 1, 5, 1])
        assert_series_equal(
            sr3.vbt.crossed_above(sr4),
            pd.Series([False, False, True, False, False, False, True]),
        )
        assert_series_equal(
            sr3.vbt.crossed_above(sr4, wait=1, dropna=True),
            pd.Series([False, False, False, False, True, False, False]),
        )
        assert_series_equal(
            sr3.vbt.crossed_above(sr4, wait=1, dropna=False),
            pd.Series([False, False, False, False, False, False, False]),
        )
        assert_frame_equal(
            df.vbt.crossed_above(df.iloc[:, ::-1], jitted=dict(parallel=True)),
            df.vbt.crossed_above(df.iloc[:, ::-1], jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.crossed_above(df.iloc[:, ::-1], chunked=True),
            df.vbt.crossed_above(df.iloc[:, ::-1], chunked=False),
        )
        sr5 = pd.Series([1, 2, 1, 2, 1])
        sr6 = pd.Series([2, 1, 2, 1, 2])
        assert_series_equal(
            sr5.vbt.crossed_above(sr6), pd.Series([False, True, False, True, False])
        )
        assert_series_equal(
            sr5.vbt.crossed_above(sr6, wait=1),
            pd.Series([False, False, False, False, False]),
        )
        sr7 = pd.Series([1, 1, 2, 2, 1, 1, 2, 2, 1, 1])
        sr8 = pd.Series([2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
        assert_series_equal(
            sr7.vbt.crossed_above(sr8),
            pd.Series([False, False, True, False, False, False, True, False, False, False]),
        )
        assert_series_equal(
            sr7.vbt.crossed_above(sr8, wait=1),
            pd.Series([False, False, False, True, False, False, False, True, False, False]),
        )

    def test_crossed_below(self):
        sr1 = pd.Series([np.nan, 3, 2, 1, 2, 3, 4])
        sr2 = pd.Series([1, 2, 3, 4, 3, 2, 1])
        assert_series_equal(
            sr1.vbt.crossed_below(sr2),
            pd.Series([False, False, True, False, False, False, False]),
        )
        assert_series_equal(
            sr1.vbt.crossed_below(sr2, wait=1),
            pd.Series([False, False, False, True, False, False, False]),
        )
        sr3 = pd.Series([1, 2, 3, np.nan, 5, 1, 5])
        sr4 = pd.Series([3, 2, 1, 1, 1, 5, 1])
        assert_series_equal(
            sr3.vbt.crossed_above(sr4),
            pd.Series([False, False, True, False, False, False, True]),
        )
        assert_series_equal(
            sr3.vbt.crossed_above(sr4, wait=1, dropna=True),
            pd.Series([False, False, False, False, True, False, False]),
        )
        assert_series_equal(
            sr3.vbt.crossed_above(sr4, wait=1, dropna=False),
            pd.Series([False, False, False, False, False, False, False]),
        )
        assert_frame_equal(
            df.vbt.crossed_below(df.iloc[:, ::-1], jitted=dict(parallel=True)),
            df.vbt.crossed_below(df.iloc[:, ::-1], jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            df.vbt.crossed_below(df.iloc[:, ::-1], chunked=True),
            df.vbt.crossed_below(df.iloc[:, ::-1], chunked=False),
        )
        sr5 = pd.Series([1, 2, 1, 2, 1])
        sr6 = pd.Series([2, 1, 2, 1, 2])
        assert_series_equal(
            sr5.vbt.crossed_below(sr6), pd.Series([False, False, True, False, True])
        )
        assert_series_equal(
            sr5.vbt.crossed_below(sr6, wait=1),
            pd.Series([False, False, False, False, False]),
        )
        sr7 = pd.Series([1, 1, 2, 2, 1, 1, 2, 2, 1, 1])
        sr8 = pd.Series([2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
        assert_series_equal(
            sr7.vbt.crossed_below(sr8),
            pd.Series([False, False, False, False, True, False, False, False, True, False]),
        )
        assert_series_equal(
            sr7.vbt.crossed_below(sr8, wait=1),
            pd.Series([False, False, False, False, False, True, False, False, False, True]),
        )

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Count",
                "Mean",
                "Std",
                "Min",
                "Median",
                "Max",
                "Min Index",
                "Max Index",
            ],
            dtype="object",
        )
        assert_series_equal(
            df.vbt.stats(),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    4.0,
                    2.1666666666666665,
                    1.0531130555537456,
                    1.0,
                    2.1666666666666665,
                    3.3333333333333335,
                ],
                index=stats_index[:-2],
                name="agg_stats",
            ),
        )
        assert_series_equal(
            df.vbt.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    4,
                    2.5,
                    1.2909944487358056,
                    1.0,
                    2.5,
                    4.0,
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-04 00:00:00"),
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            df.vbt.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    8,
                    2.5,
                    1.1952286093343936,
                    1.0,
                    2.5,
                    4.0,
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-02 00:00:00"),
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(df["c"].vbt.stats(), df.vbt.stats(column="c"))
        assert_series_equal(df["c"].vbt.stats(), df.vbt.stats(column="c", group_by=False))
        assert_series_equal(
            df.vbt(group_by=group_by)["g2"].stats(),
            df.vbt(group_by=group_by).stats(column="g2"),
        )
        assert_series_equal(
            df.vbt(group_by=group_by)["g2"].stats(),
            df.vbt.stats(column="g2", group_by=group_by),
        )
        stats_df = df.vbt.stats(agg_func=None)
        assert stats_df.shape == (3, 11)
        assert_index_equal(stats_df.index, df.vbt.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_mapping_stats(self):
        mapping = {x: "test_" + str(x) for x in pd.unique(df.values.flatten())}
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Value Counts: test_1.0",
                "Value Counts: test_2.0",
                "Value Counts: test_3.0",
                "Value Counts: test_4.0",
                "Value Counts: test_nan",
            ],
            dtype="object",
        )
        assert_series_equal(
            df.vbt(mapping=mapping).stats(),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    1.3333333333333333,
                    1.3333333333333333,
                    0.6666666666666666,
                    0.6666666666666666,
                    1.0,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            df.vbt(mapping=mapping).stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            df.vbt(mapping=mapping).stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    2,
                    2,
                    2,
                    2,
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(
            df.vbt(mapping=mapping).stats(), df.vbt.stats(settings=dict(mapping=mapping))
        )
        assert_series_equal(
            df["c"].vbt(mapping=mapping).stats(settings=dict(incl_all_keys=True)),
            df.vbt(mapping=mapping).stats(column="c"),
        )
        assert_series_equal(
            df["c"].vbt(mapping=mapping).stats(settings=dict(incl_all_keys=True)),
            df.vbt(mapping=mapping).stats(column="c", group_by=False),
        )
        assert_series_equal(
            df.vbt(mapping=mapping, group_by=group_by)["g2"].stats(
                settings=dict(incl_all_keys=True)
            ),
            df.vbt(mapping=mapping, group_by=group_by).stats(column="g2"),
        )
        assert_series_equal(
            df.vbt(mapping=mapping, group_by=group_by)["g2"].stats(
                settings=dict(incl_all_keys=True)
            ),
            df.vbt(mapping=mapping).stats(column="g2", group_by=group_by),
        )
        stats_df = df.vbt(mapping=mapping).stats(agg_func=None)
        assert stats_df.shape == (3, 8)
        assert_index_equal(stats_df.index, df.vbt.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)
