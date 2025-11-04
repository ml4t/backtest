import os

import pytest
import vectorbtpro as vbt

from tests.utils import *

seed = 42


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.chunking["n_chunks"] = 2


def teardown_module():
    vbt.settings.reset()


# ############# base ############# #


index = pd.date_range("2020-01-01", "2020-02-01", inclusive="left")


class TestRelRange:
    def test_split(self):
        assert vbt.RelRange().to_slice(30) == slice(0, 30)
        assert vbt.RelRange(offset=1).to_slice(30) == slice(1, 30)
        assert vbt.RelRange(offset=0.5).to_slice(30) == slice(15, 30)
        assert vbt.RelRange(offset_anchor="end", offset=-1.0).to_slice(30) == slice(0, 30)
        assert vbt.RelRange(offset_anchor="prev_start").to_slice(30, prev_start=1) == slice(1, 30)
        assert vbt.RelRange(offset_anchor="prev_end").to_slice(30, prev_end=1) == slice(1, 30)
        assert vbt.RelRange(offset_anchor="prev_end", offset=0.5, offset_space="free").to_slice(
            30, prev_end=10
        ) == slice(20, 30)
        assert vbt.RelRange(offset_anchor="prev_end", offset=-0.5, offset_space="free").to_slice(
            30, prev_end=10
        ) == slice(5, 30)
        assert vbt.RelRange(offset_anchor="prev_end", offset=0.5, offset_space="all").to_slice(
            30, prev_end=10
        ) == slice(25, 30)
        assert vbt.RelRange(length=10).to_slice(30) == slice(0, 10)
        assert vbt.RelRange(length=0.5).to_slice(30) == slice(0, 15)
        assert vbt.RelRange(offset_anchor="prev_end", length=10).to_slice(30, prev_end=10) == slice(
            10, 20
        )
        assert vbt.RelRange(offset_anchor="prev_end", length=0.5).to_slice(
            30, prev_end=10
        ) == slice(10, 20)
        assert vbt.RelRange(offset_anchor="end", length=-0.5).to_slice(30, prev_end=0) == slice(
            15, 30
        )
        assert vbt.RelRange(offset_anchor="end", length=-0.5).to_slice(30, prev_end=10) == slice(
            15, 30
        )
        assert vbt.RelRange(offset_anchor="end", length=-0.5, length_space="free_or_prev").to_slice(
            30, prev_end=10
        ) == slice(20, 30)
        assert vbt.RelRange(offset_anchor="prev_end", length=-0.5).to_slice(
            30, prev_end=10
        ) == slice(5, 10)
        assert vbt.RelRange(offset_anchor="prev_end", length=0.5, length_space="all").to_slice(
            30, prev_end=10
        ) == slice(10, 25)
        assert vbt.RelRange(offset=-10, length=50).to_slice(30) == slice(0, 30)
        with pytest.raises(Exception):
            vbt.RelRange(offset=-10, length=50, out_of_bounds="raise").to_slice(30)
        with pytest.raises(Exception):
            vbt.RelRange(offset_anchor="hello")
        with pytest.raises(Exception):
            vbt.RelRange(offset_space="hello")
        with pytest.raises(Exception):
            vbt.RelRange(length_space="hello")
        with pytest.raises(Exception):
            vbt.RelRange(out_of_bounds="hello")
        assert vbt.RelRange(length=index[5] - index[0]).to_slice(
            len(index), index=index
        ) == vbt.RelRange(length=5).to_slice(len(index), index=index)
        assert vbt.RelRange(offset=1, length=index[5] - index[0]).to_slice(
            len(index), index=index
        ) == vbt.RelRange(offset=1, length=5).to_slice(len(index), index=index)
        assert vbt.RelRange(offset=index[1] - index[0], length=index[5] - index[0]).to_slice(
            len(index), index=index
        ) == vbt.RelRange(offset=1, length=5).to_slice(len(index), index=index)
        assert vbt.RelRange(offset_anchor="end", length=index[0] - index[5]).to_slice(
            len(index), index=index
        ) == vbt.RelRange(offset_anchor="end", length=-5).to_slice(len(index), index=index)
        assert vbt.RelRange(offset="-3 days", length="5 days").to_slice(
            len(index), index=index
        ) == vbt.RelRange(offset=-3, length=5).to_slice(len(index), index=index)
        assert vbt.RelRange(offset="3 days", length="-5 days").to_slice(
            len(index), index=index
        ) == vbt.RelRange(offset=3, length=-5).to_slice(len(index), index=index)
        assert vbt.RelRange(
            offset="-3 days", offset_anchor="end", length=index[5] - index[0]
        ).to_slice(len(index), index=index) == vbt.RelRange(
            offset=-3, offset_anchor="end", length=5
        ).to_slice(len(index), index=index)
        assert vbt.RelRange(
            offset="3 days", offset_anchor="end", length=index[0] - index[5]
        ).to_slice(len(index), index=index) == vbt.RelRange(
            offset=3, offset_anchor="end", length=-5
        ).to_slice(len(index), index=index)


class TestSplitter:
    def test_from_splits(self):
        np.testing.assert_array_equal(
            vbt.Splitter.from_splits(index, [0.5]).splits_arr,
            np.array([[slice(0, 15, None), slice(15, 31, None)]], dtype=object),
        )
        assert_index_equal(
            vbt.Splitter.from_splits(index, [0.5]).wrapper.index,
            pd.RangeIndex(start=0, stop=1, step=1, name="split"),
        )
        assert_index_equal(
            vbt.Splitter.from_splits(index, [0.5]).wrapper.columns,
            pd.Index(["set_0", "set_1"], dtype="object", name="set"),
        )
        assert vbt.Splitter.from_splits(index, [0.5]).wrapper.ndim == 2
        np.testing.assert_array_equal(
            vbt.Splitter.from_splits(index, [[0.5]]).splits_arr,
            np.array([[slice(0, 15, None)]], dtype=object),
        )
        assert_index_equal(
            vbt.Splitter.from_splits(index, [[0.5]]).wrapper.index,
            pd.RangeIndex(start=0, stop=1, step=1, name="split"),
        )
        assert_index_equal(
            vbt.Splitter.from_splits(index, [[0.5]]).wrapper.columns,
            pd.Index(["set_0"], dtype="object", name="set"),
        )
        assert vbt.Splitter.from_splits(index, [[0.5]]).wrapper.ndim == 2
        np.testing.assert_array_equal(
            vbt.Splitter.from_splits(index, [[0.5]], fix_ranges=False).splits_arr,
            np.array([[vbt.RelRange(length=0.5)]], dtype=object),
        )
        assert vbt.Splitter.from_splits(
            index, [[0.5]], split_range_kwargs=dict(range_format="mask")
        ).splits_arr.shape == (1, 1)
        np.testing.assert_array_equal(
            vbt.Splitter.from_splits(index, [[0.5]], split_range_kwargs=dict(range_format="mask"))
            .splits_arr[0, 0]
            .range_,
            np.array([*[True] * 15, *[False] * 16]),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_splits(index, [[0.5], [1.0]]).splits_arr,
            np.array([[slice(0, 15, None)], [slice(0, 31, None)]], dtype=object),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_splits(index, [[0.25, 0.5], [0.75, 1.0]]).splits_arr,
            np.array(
                [
                    [slice(0, 7, None), slice(7, 19, None)],
                    [slice(0, 23, None), slice(23, 31, None)],
                ],
                dtype=object,
            ),
        )
        assert_index_equal(
            vbt.Splitter.from_splits(
                index, [[0.25, 0.5], [0.75, 1.0]], split_labels=["s1", "s2"]
            ).wrapper.index,
            pd.Index(["s1", "s2"], name="split"),
        )
        assert_index_equal(
            vbt.Splitter.from_splits(
                index,
                [[0.25, 0.5], [0.75, 1.0]],
                split_labels=pd.Index(["s1", "s2"], name="my_split"),
            ).wrapper.index,
            pd.Index(["s1", "s2"], name="my_split"),
        )
        assert_index_equal(
            vbt.Splitter.from_splits(
                index, [[0.25, 0.5], [0.75, 1.0]], set_labels=["s1", "s2"]
            ).wrapper.columns,
            pd.Index(["s1", "s2"], name="set"),
        )
        assert_index_equal(
            vbt.Splitter.from_splits(
                index,
                [[0.25, 0.5], [0.75, 1.0]],
                set_labels=pd.Index(["s1", "s2"], name="my_set"),
            ).wrapper.columns,
            pd.Index(["s1", "s2"], name="my_set"),
        )

    def test_from_split_func(self):
        def split_func(split_idx, x, y=15):
            if split_idx == 0:
                return slice(x, y)
            return None

        np.testing.assert_array_equal(
            vbt.Splitter.from_split_func(
                index,
                split_func,
                split_args=(vbt.Rep("split_idx"), 10),
                split_kwargs=dict(y=20),
            ).splits_arr,
            np.array([[slice(10, 20, None)]], dtype=object),
        )

        def split_func(split_idx, splits, bounds):
            if split_idx == 0:
                return [slice(0, 5), slice(5, 10)]
            if split_idx == 1:
                return slice(splits[-1][-1].stop, 15), slice(15, 20)
            if split_idx == 2:
                return slice(bounds[-1][-1][1], 25), slice(25, 30)
            return None

        np.testing.assert_array_equal(
            vbt.Splitter.from_split_func(
                index,
                split_func,
                split_args=(vbt.Rep("split_idx"), vbt.Rep("splits"), vbt.Rep("bounds")),
            ).splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(10, 15, None), slice(15, 20, None)],
                    [slice(20, 25, None), slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )

        def split_func(split_idx, splits, bounds):
            if split_idx == 0:
                return [slice(0, 10)]
            if split_idx == 1:
                return slice(splits[-1][-1].stop, 20)
            if split_idx == 2:
                return slice(bounds[-1][-1][1], 30)
            return None

        np.testing.assert_array_equal(
            vbt.Splitter.from_split_func(
                index,
                split_func,
                split_args=(vbt.Rep("split_idx"), vbt.Rep("splits"), vbt.Rep("bounds")),
                split=0.5,
            ).splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(10, 15, None), slice(15, 20, None)],
                    [slice(20, 25, None), slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )

        def split_func(split_idx):
            if split_idx == 0:
                return 0.5
            if split_idx == 1:
                return 1.0
            return None

        np.testing.assert_array_equal(
            vbt.Splitter.from_split_func(
                index,
                vbt.Rep("split_func", context=dict(split_func=split_func)),
                split_args=(vbt.Rep("split_idx"),),
                fix_ranges=False,
            ).splits_arr,
            np.array([[vbt.RelRange(length=0.5)], [vbt.RelRange(length=1.0)]]),
        )

        def split_func(split_idx):
            if split_idx == 0:
                return 0.5
            if split_idx == 1:
                return 0.5, 1.0
            return None

        with pytest.raises(Exception):
            vbt.Splitter.from_split_func(
                index,
                split_func,
                split_args=(vbt.Rep("split_idx"),),
            )

    def test_from_single(self):
        np.testing.assert_array_equal(
            vbt.Splitter.from_single(index, 0.5).splits_arr,
            np.array([[slice(0, 15, None), slice(15, 31, None)]], dtype=object),
        )

    def test_from_rolling(self):
        with pytest.raises(Exception):
            vbt.Splitter.from_rolling(index, -1)
        with pytest.raises(Exception):
            vbt.Splitter.from_rolling(index, 0)
        with pytest.raises(Exception):
            vbt.Splitter.from_rolling(index, 1.5)
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 0.5).splits_arr,
            np.array([[slice(0, 15, None)], [slice(15, 30, None)]], dtype=object),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 0.7).splits_arr,
            np.array([[slice(0, 21, None)]], dtype=object),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 1.0).splits_arr,
            np.array([[slice(0, 31, None)]], dtype=object),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 10).splits_arr,
            np.array(
                [[slice(0, 10, None)], [slice(10, 20, None)], [slice(20, 30, None)]], dtype=object
            ),
        )
        with pytest.raises(Exception):
            vbt.Splitter.from_rolling(index, 10, offset=1.5)
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 10, offset=0.1).splits_arr,
            np.array([[slice(0, 10, None)], [slice(11, 21, None)]], dtype=object),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 10, offset=-1).splits_arr,
            np.array(
                [[slice(0, 10, None)], [slice(9, 19, None)], [slice(18, 28, None)]], dtype=object
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 10, offset=-0.1).splits_arr,
            np.array(
                [[slice(0, 10, None)], [slice(9, 19, None)], [slice(18, 28, None)]], dtype=object
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 10, offset=10, offset_anchor="prev_start").splits_arr,
            np.array(
                [[slice(0, 10, None)], [slice(10, 20, None)], [slice(20, 30, None)]], dtype=object
            ),
        )
        with pytest.raises(Exception):
            vbt.Splitter.from_rolling(index, 10, offset_anchor="prev_start")
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 10, split=0.5).splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(5, 10, None), slice(10, 15, None)],
                    [slice(10, 15, None), slice(15, 20, None)],
                    [slice(15, 20, None), slice(20, 25, None)],
                    [slice(20, 25, None), slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 10, split=0.5, offset_anchor_set=-1).splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(10, 15, None), slice(15, 20, None)],
                    [slice(20, 25, None), slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_rolling(index, 10, split=0.5, offset_anchor_set=None).splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(10, 15, None), slice(15, 20, None)],
                    [slice(20, 25, None), slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )

    def test_from_n_rolling(self):
        with pytest.raises(Exception):
            vbt.Splitter.from_n_rolling(index, 5, length=-1)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_rolling(index, 5, length=0)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_rolling(index, 5, length=1.5)
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_rolling(index, 5, split=0.5).splits_arr,
            vbt.Splitter.from_rolling(
                index, len(index) // 5, offset_anchor_set=None, split=0.5
            ).splits_arr,
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_rolling(index, 5, length=10).splits_arr,
            np.array(
                [
                    [slice(0, 10, None)],
                    [slice(5, 15, None)],
                    [slice(10, 20, None)],
                    [slice(16, 26, None)],
                    [slice(21, 31, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_rolling(index, 5, length=10, split=0.5).splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(5, 10, None), slice(10, 15, None)],
                    [slice(10, 15, None), slice(15, 20, None)],
                    [slice(16, 21, None), slice(21, 26, None)],
                    [slice(21, 26, None), slice(26, 31, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_rolling(index, 40, length=10).splits_arr,
            np.array([*[[slice(i, i + 10)] for i in range(22)]], dtype=object),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_rolling(index, 5, length="5 days").splits_arr,
            vbt.Splitter.from_n_rolling(index, 5, length=5).splits_arr,
        )

    def test_from_expanding(self):
        with pytest.raises(Exception):
            vbt.Splitter.from_expanding(index, -1, 1)
        with pytest.raises(Exception):
            vbt.Splitter.from_expanding(index, 0, 1)
        with pytest.raises(Exception):
            vbt.Splitter.from_expanding(index, 1.5, 1)
        with pytest.raises(Exception):
            vbt.Splitter.from_expanding(index, 1, -1)
        with pytest.raises(Exception):
            vbt.Splitter.from_expanding(index, 1, 0)
        with pytest.raises(Exception):
            vbt.Splitter.from_expanding(index, 1, 1.5)
        np.testing.assert_array_equal(
            vbt.Splitter.from_expanding(index, 10, 5).splits_arr,
            np.array(
                [
                    [slice(0, 10, None)],
                    [slice(0, 15, None)],
                    [slice(0, 20, None)],
                    [slice(0, 25, None)],
                    [slice(0, 30, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_expanding(index, 0.5, 5).splits_arr,
            np.array(
                [
                    [slice(0, 15, None)],
                    [slice(0, 20, None)],
                    [slice(0, 25, None)],
                    [slice(0, 30, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_expanding(index, 10, 1 / 6).splits_arr,
            np.array(
                [
                    [slice(0, 10, None)],
                    [slice(0, 15, None)],
                    [slice(0, 20, None)],
                    [slice(0, 25, None)],
                    [slice(0, 30, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_expanding(index, 10, 5, split=0.5).splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(0, 7, None), slice(7, 15, None)],
                    [slice(0, 10, None), slice(10, 20, None)],
                    [slice(0, 12, None), slice(12, 25, None)],
                    [slice(0, 15, None), slice(15, 30, None)],
                ],
                dtype=object,
            ),
        )

    def test_from_n_expanding(self):
        with pytest.raises(Exception):
            vbt.Splitter.from_n_expanding(index, 5, min_length=-1)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_expanding(index, 5, min_length=0)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_expanding(index, 5, min_length=1.5)
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_expanding(index, 5).splits_arr,
            np.array(
                [
                    [slice(0, 6, None)],
                    [slice(0, 12, None)],
                    [slice(0, 18, None)],
                    [slice(0, 25, None)],
                    [slice(0, 31, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_expanding(index, 5, min_length=10).splits_arr,
            np.array(
                [
                    [slice(0, 10, None)],
                    [slice(0, 15, None)],
                    [slice(0, 20, None)],
                    [slice(0, 26, None)],
                    [slice(0, 31, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_expanding(index, 5, min_length=10, split=0.5).splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(0, 7, None), slice(7, 15, None)],
                    [slice(0, 10, None), slice(10, 20, None)],
                    [slice(0, 13, None), slice(13, 26, None)],
                    [slice(0, 15, None), slice(15, 31, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_expanding(index, 40, min_length=10).splits_arr,
            np.array([*[[slice(0, i + 10)] for i in range(22)]], dtype=object),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_expanding(index, 5, min_length="10 days").splits_arr,
            vbt.Splitter.from_n_expanding(index, 5, min_length=10).splits_arr,
        )

    def test_from_n_random(self):
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, min_start=-1)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, min_start=-0.1)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, min_start=1.5)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, min_start=100)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, max_end=-1)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, max_end=0)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, max_end=1.5)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, max_end=100)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, -1)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 0)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 1.5)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 100)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, max_length=-1)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, max_length=0)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, max_length=1.5)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 10, max_length=100)
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(index, 5, 5, 10, seed=seed).splits_arr,
            np.array(
                [
                    [slice(20, 25, None)],
                    [slice(10, 18, None)],
                    [slice(21, 28, None)],
                    [slice(18, 23, None)],
                    [slice(2, 8, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(index, 5, 6, split=0.5, seed=seed).splits_arr,
            np.array(
                [
                    [slice(2, 5, None), slice(5, 8, None)],
                    [slice(20, 23, None), slice(23, 26, None)],
                    [slice(17, 20, None), slice(20, 23, None)],
                    [slice(11, 14, None), slice(14, 17, None)],
                    [slice(11, 14, None), slice(14, 17, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(index, 5, 5, 10, min_start=20, seed=seed).splits_arr,
            np.array(
                [
                    [slice(25, 30, None)],
                    [slice(21, 29, None)],
                    [slice(24, 31, None)],
                    [slice(24, 29, None)],
                    [slice(20, 26, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index, 5, 5, 10, min_start=10, max_end=20, seed=seed
            ).splits_arr,
            np.array(
                [
                    [slice(14, 19, None)],
                    [slice(11, 19, None)],
                    [slice(13, 20, None)],
                    [slice(14, 19, None)],
                    [slice(10, 16, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index,
                5,
                5,
                10,
                length_p_func=lambda i, x: np.arange(len(x)) / np.arange(len(x)).sum(),
                seed=seed,
            ).splits_arr,
            np.array(
                [
                    [slice(14, 24, None)],
                    [slice(9, 19, None)],
                    [slice(4, 14, None)],
                    [slice(2, 12, None)],
                    [slice(15, 25, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index,
                5,
                5,
                10,
                start_p_func=lambda i, x: np.arange(len(x)) / np.arange(len(x)).sum(),
                seed=seed,
            ).splits_arr,
            np.array(
                [
                    [slice(18, 23, None)],
                    [slice(21, 30, None)],
                    [slice(8, 13, None)],
                    [slice(22, 31, None)],
                    [slice(20, 29, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index,
                5,
                5,
                10,
                length_choice_func=lambda i, x: np.random.choice(
                    x, p=np.arange(len(x)) / np.arange(len(x)).sum()
                ),
                seed=seed,
            ).splits_arr,
            np.array(
                [
                    [slice(2, 10, None)],
                    [slice(17, 27, None)],
                    [slice(14, 24, None)],
                    [slice(10, 19, None)],
                    [slice(10, 17, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index,
                5,
                5,
                10,
                start_choice_func=lambda i, x: np.random.choice(
                    x, p=np.arange(len(x)) / np.arange(len(x)).sum()
                ),
                seed=seed,
            ).splits_arr,
            np.array(
                [
                    [slice(16, 21, None)],
                    [slice(22, 31, None)],
                    [slice(20, 28, None)],
                    [slice(19, 26, None)],
                    [slice(10, 17, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index, 5, 5, 10, min_start="2020-01-03", seed=seed
            ).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 5, 10, min_start=2, seed=seed).splits_arr,
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(index, 5, 5, 10, max_end="2020-01-29", seed=seed).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 5, 10, max_end=28, seed=seed).splits_arr,
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(index, 5, "5 days", seed=seed).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 5, seed=seed).splits_arr,
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(index, 5, "5 days", "10 days", seed=seed).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 5, 10, seed=seed).splits_arr,
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index, 5, "5 days", "10 days", min_start="2020-01-03", seed=seed
            ).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 5, 10, min_start=2, seed=seed).splits_arr,
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index, 5, "5 days", "10 days", max_end="2020-01-29", seed=seed
            ).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 5, 10, max_end=28, seed=seed).splits_arr,
        )
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(
                index, 5, 1, min_start=index[0] - pd.Timedelta(days=1), seed=seed
            )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(index, 5, 1, min_start=index[0], seed=seed).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 1, min_start=0, seed=seed).splits_arr,
        )
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(
                index, 5, 1, min_start=index[-1] + pd.Timedelta(days=1), seed=seed
            )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(index, 5, 1, min_start=index[-1], seed=seed).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 1, min_start=len(index) - 1, seed=seed).splits_arr,
        )
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, 1, max_end=index[0], seed=seed)
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index, 5, 1, max_end=index[0] + pd.Timedelta(days=1), seed=seed
            ).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 1, max_end=1, seed=seed).splits_arr,
        )
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(
                index, 5, 1, max_end=index[-1] + pd.Timedelta(days=2), seed=seed
            )
        np.testing.assert_array_equal(
            vbt.Splitter.from_n_random(
                index, 5, 1, max_end=index[-1] + pd.Timedelta(days=1), seed=seed
            ).splits_arr,
            vbt.Splitter.from_n_random(index, 5, 1, max_end=len(index), seed=seed).splits_arr,
        )
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, "5 days", max_end="2020-01-05", seed=seed)
        with pytest.raises(Exception):
            vbt.Splitter.from_n_random(index, 5, "5 days", "4 days", seed=seed)

    def test_from_ranges(self):
        np.testing.assert_array_equal(
            vbt.Splitter.from_ranges(index, every="W").splits_arr,
            np.array(
                [
                    [slice(5, 12, None)],
                    [slice(12, 19, None)],
                    [slice(19, 26, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_ranges(index, every="W", split=0.5).splits_arr,
            np.array(
                [
                    [slice(5, 8, None), slice(8, 12, None)],
                    [slice(12, 15, None), slice(15, 19, None)],
                    [slice(19, 22, None), slice(22, 26, None)],
                ],
                dtype=object,
            ),
        )

    def test_from_grouper(self):
        np.testing.assert_array_equal(
            vbt.Splitter.from_grouper(index, by="W").splits_arr,
            np.array(
                [
                    [slice(0, 6, None)],
                    [slice(6, 13, None)],
                    [slice(13, 20, None)],
                    [slice(20, 27, None)],
                    [slice(27, 31, None)],
                ],
                dtype=object,
            ),
        )
        assert_index_equal(
            vbt.Splitter.from_grouper(index, by="W").wrapper.index,
            pd.PeriodIndex(
                [
                    "2019-12-31/2020-01-06",
                    "2020-01-07/2020-01-13",
                    "2020-01-14/2020-01-20",
                    "2020-01-21/2020-01-27",
                    "2020-01-28/2020-02-02",
                ],
                dtype="period[W-MON]",
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_grouper(index, by="W", split=0.5).splits_arr,
            np.array(
                [
                    [slice(0, 3, None), slice(3, 6, None)],
                    [slice(6, 9, None), slice(9, 13, None)],
                    [slice(13, 16, None), slice(16, 20, None)],
                    [slice(20, 23, None), slice(23, 27, None)],
                    [slice(27, 29, None), slice(29, 31, None)],
                ],
                dtype=object,
            ),
        )

    def test_from_sklearn(self):
        from sklearn.model_selection import TimeSeriesSplit

        np.testing.assert_array_equal(
            vbt.Splitter.from_sklearn(index, TimeSeriesSplit(n_splits=5)).splits_arr,
            np.array(
                [
                    [slice(0, 6, None), slice(6, 11, None)],
                    [slice(0, 11, None), slice(11, 16, None)],
                    [slice(0, 16, None), slice(16, 21, None)],
                    [slice(0, 21, None), slice(21, 26, None)],
                    [slice(0, 26, None), slice(26, 31, None)],
                ],
                dtype=object,
            ),
        )

    def test_from_purged_walkforward(self):
        np.testing.assert_array_equal(
            vbt.Splitter.from_purged_walkforward(index, n_folds=5).splits_arr,
            np.array(
                [
                    [slice(0, 13, None), slice(13, 19, None)],
                    [slice(0, 19, None), slice(19, 25, None)],
                    [slice(0, 25, None), slice(25, 31, None)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.from_purged_walkforward(
                index,
                n_folds=5,
                pred_times=index,
                eval_times=index + pd.Timedelta(days=3),
            ).splits_arr,
            np.array(
                [
                    [slice(0, 10, None), slice(13, 19, None)],
                    [slice(0, 16, None), slice(19, 25, None)],
                    [slice(0, 22, None), slice(25, 31, None)],
                ],
                dtype=object,
            ),
        )

    def test_from_purged_kfold(self):
        splits_arr = vbt.Splitter.from_purged_kfold(index, n_folds=4).splits_arr
        assert splits_arr[0][0] == slice(0, 16, None)
        assert splits_arr[0][1] == slice(16, 31, None)
        np.testing.assert_array_equal(
            splits_arr[1][0].range_,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]),
        )
        np.testing.assert_array_equal(
            splits_arr[1][1].range_,
            np.array([8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30]),
        )
        np.testing.assert_array_equal(
            splits_arr[2][0].range_,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30]),
        )
        assert splits_arr[2][1] == slice(8, 24, None)
        assert splits_arr[3][0] == slice(8, 24, None)
        np.testing.assert_array_equal(
            splits_arr[3][1].range_,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30]),
        )
        np.testing.assert_array_equal(
            splits_arr[4][0].range_,
            np.array([8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30]),
        )
        np.testing.assert_array_equal(
            splits_arr[4][1].range_,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]),
        )
        assert splits_arr[5][0] == slice(16, 31, None)
        assert splits_arr[5][1] == slice(0, 16, None)

        splits_arr = vbt.Splitter.from_purged_kfold(
            index,
            n_folds=4,
            pred_times=index,
            eval_times=index + pd.Timedelta(days=3),
        ).splits_arr
        assert splits_arr[0][0] == slice(0, 13, None)
        assert splits_arr[0][1] == slice(16, 31, None)
        np.testing.assert_array_equal(
            splits_arr[1][0].range_,
            np.array([0, 1, 2, 3, 4, 19, 20]),
        )
        np.testing.assert_array_equal(
            splits_arr[1][1].range_,
            np.array([8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30]),
        )
        np.testing.assert_array_equal(
            splits_arr[2][0].range_,
            np.array([0, 1, 2, 3, 4, 27, 28, 29, 30]),
        )
        assert splits_arr[2][1] == slice(8, 24, None)
        assert splits_arr[3][0] == slice(11, 21, None)
        np.testing.assert_array_equal(
            splits_arr[3][1].range_,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30]),
        )
        np.testing.assert_array_equal(
            splits_arr[4][0].range_,
            np.array([11, 12, 27, 28, 29, 30]),
        )
        np.testing.assert_array_equal(
            splits_arr[4][1].range_,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]),
        )
        assert splits_arr[5][0] == slice(19, 31, None)
        assert splits_arr[5][1] == slice(0, 16, None)

        splits_arr = vbt.Splitter.from_purged_kfold(
            index,
            n_folds=4,
            embargo_td="2 days",
            pred_times=index,
            eval_times=index + pd.Timedelta(days=3),
        ).splits_arr
        assert splits_arr[0][0] == slice(0, 13, None)
        assert splits_arr[0][1] == slice(16, 31, None)
        assert splits_arr[1][0] == slice(0, 5, None)
        np.testing.assert_array_equal(
            splits_arr[1][1].range_,
            np.array([8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30]),
        )
        np.testing.assert_array_equal(
            splits_arr[2][0].range_,
            np.array([0, 1, 2, 3, 4, 29, 30]),
        )
        assert splits_arr[2][1] == slice(8, 24, None)
        assert splits_arr[3][0] == slice(13, 21, None)
        np.testing.assert_array_equal(
            splits_arr[3][1].range_,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30]),
        )
        assert splits_arr[4][0] == slice(29, 31, None)
        np.testing.assert_array_equal(
            splits_arr[4][1].range_,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]),
        )
        assert splits_arr[5][0] == slice(21, 31, None)
        assert splits_arr[5][1] == slice(0, 16, None)

    def test_row_stack(self):
        splitter1 = vbt.Splitter.from_splits(
            index, [slice(0, 10), slice(10, 20), slice(20, 30)], split_labels=["a", "b", "c"]
        )
        splitter2 = vbt.Splitter.from_splits(
            index, [slice(0, 10), slice(10, 20), slice(20, 30)], split_labels=["d", "e", "f"]
        )
        splitter = vbt.Splitter.row_stack(splitter1, splitter2)
        assert isinstance(splitter, vbt.Splitter)
        assert_index_equal(
            splitter.wrapper.index, pd.Index(["a", "b", "c", "d", "e", "f"], name="split")
        )
        assert_index_equal(splitter.wrapper.columns, pd.Index(["set_0"], name="set"))
        assert splitter.wrapper.ndim == 1

        splitter3 = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, 30)],
            ],
            split_labels=["d", "e", "f"],
        )
        with pytest.raises(Exception):
            vbt.Splitter.row_stack(splitter1, splitter3)
        with pytest.raises(Exception):
            vbt.Splitter.row_stack(splitter1, splitter2.replace(index=index + pd.Timedelta(days=1)))
        with pytest.raises(Exception):
            vbt.Splitter.row_stack(
                splitter1, splitter2.replace(hello="world", check_expected_keys_=False)
            )
        with pytest.raises(Exception):
            vbt.Splitter.row_stack(
                splitter1.replace(hello="world1", check_expected_keys_=False),
                splitter2.replace(hello="world2", check_expected_keys_=False),
            )
        assert (
            vbt.Splitter.row_stack(
                splitter1.replace(hello="world", check_expected_keys_=False),
                splitter2.replace(hello="world", check_expected_keys_=False),
            ).config["hello"]
            == "world"
        )

    def test_column_stack(self):
        splitter1 = vbt.Splitter.from_splits(
            index,
            [slice(0, 10), slice(10, 20), slice(20, 30)],
            split_labels=["a", "b", "c"],
            set_labels=["a"],
        )
        splitter2 = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, 30)],
            ],
            split_labels=["a", "b", "c"],
            set_labels=["b", "c"],
        )
        splitter = vbt.Splitter.column_stack(splitter1, splitter2)
        assert isinstance(splitter, vbt.Splitter)
        assert_index_equal(splitter.wrapper.index, pd.Index(["a", "b", "c"], name="split"))
        assert_index_equal(splitter.wrapper.columns, pd.Index(["a", "b", "c"], name="set"))
        assert splitter.wrapper.ndim == 2

        splitter3 = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, 30)],
            ],
            split_labels=["b", "c", "d"],
            set_labels=["b", "c"],
        )
        with pytest.raises(Exception):
            vbt.Splitter.column_stack(splitter1, splitter3)
        with pytest.raises(Exception):
            vbt.Splitter.column_stack(
                splitter1, splitter2.replace(index=index + pd.Timedelta(days=1))
            )
        with pytest.raises(Exception):
            vbt.Splitter.column_stack(
                splitter1, splitter2.replace(hello="world", check_expected_keys_=False)
            )
        with pytest.raises(Exception):
            vbt.Splitter.column_stack(
                splitter1.replace(hello="world1", check_expected_keys_=False),
                splitter2.replace(hello="world2", check_expected_keys_=False),
            )
        assert (
            vbt.Splitter.column_stack(
                splitter1.replace(hello="world", check_expected_keys_=False),
                splitter2.replace(hello="world", check_expected_keys_=False),
            ).config["hello"]
            == "world"
        )

    def test_config(self, tmp_path):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, 30)],
            ],
            split_labels=["a", "b", "c"],
            set_labels=["d", "e"],
        )
        assert vbt.Splitter.loads(splitter.dumps()) == splitter
        splitter.save(tmp_path / "splitter")
        assert vbt.Splitter.load(tmp_path / "splitter") == splitter
        splitter.save(tmp_path / "splitter", file_format="ini")
        assert vbt.Splitter.load(tmp_path / "splitter", file_format="ini") == splitter

    def test_indexing(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, 30)],
            ],
            split_labels=["a", "b", "c"],
            set_labels=["d", "e"],
        )
        np.testing.assert_array_equal(
            splitter.loc[["a", "c"], "d"].wrapper, splitter.wrapper.loc[["a", "c"], "d"]
        )
        np.testing.assert_array_equal(splitter.loc[["a", "c"], "d"].index, splitter.index)
        np.testing.assert_array_equal(
            splitter.loc[["a", "c"], "d"].splits_arr, splitter.splits_arr[[0, 2]][:, [0]]
        )

    def test_get_range(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 15), slice(10, 20)],
                [slice(10, 20), slice(15, 25)],
                [slice(15, 25), slice(20, None)],
            ],
            split_labels=[10, 11, 12],
        )
        assert splitter.select_range() == slice(5, 31)

        assert splitter.select_range(split=10, set_="set_0") == slice(5, 15)
        assert splitter.select_range(split=[10], set_="set_0") == slice(5, 15)
        assert splitter.select_range(split=10, set_=["set_0", "set_1"]) == slice(5, 20)
        assert splitter.select_range(split=[10, 11, 12], set_="set_0") == slice(5, 25)
        assert splitter.select_range(split=None, set_="set_0") == slice(5, 25)

        with pytest.raises(Exception):
            splitter.select_range(split=0, set_=0)
        assert splitter.select_range(split=vbt.PosSel(0), set_=0) == slice(5, 15)
        assert splitter.select_range(split=vbt.PosSel(-1), set_=-1) == slice(20, 31)
        assert splitter.select_range(split=vbt.PosSel([0]), set_=0) == slice(5, 15)
        assert splitter.select_range(split=vbt.PosSel(0), set_=[0, 1]) == slice(5, 20)
        assert splitter.select_range(split=vbt.PosSel([0, 1, 2]), set_=0) == slice(5, 25)
        assert splitter.select_range(
            split=[vbt.PosSel(0), vbt.PosSel(1), vbt.PosSel(2)], set_=0
        ) == slice(5, 25)
        assert splitter.select_range(split=vbt.PosSel(0)) == slice(5, 20)

        split_group_by = [10, 11, 10]
        set_group_by = [12, 13]

        assert splitter.select_range(
            split=10,
            set_=12,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=[10],
            set_=12,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=[10, 11],
            set_=12,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=None,
            set_=12,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=10,
            set_=[12],
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=10,
            set_=[12, 13],
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 31, None)
        assert splitter.select_range(
            split=10,
            set_=None,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 31, None)

        with pytest.raises(Exception):
            splitter.select_range(
                split=vbt.PosSel(10),
                set_=vbt.PosSel(0),
                split_group_by=split_group_by,
                set_group_by=set_group_by,
            )
        with pytest.raises(Exception):
            splitter.select_range(
                split=vbt.PosSel(0),
                set_=vbt.PosSel(12),
                split_group_by=split_group_by,
                set_group_by=set_group_by,
            )

        assert splitter.select_range(
            split=vbt.PosSel(0),
            set_=vbt.PosSel(0),
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=vbt.PosSel([0]),
            set_=vbt.PosSel(0),
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=vbt.PosSel([0, 1]),
            set_=vbt.PosSel(0),
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=[vbt.PosSel(0), vbt.PosSel(1)],
            set_=vbt.PosSel(0),
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=None,
            set_=vbt.PosSel(0),
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=vbt.PosSel(0),
            set_=vbt.PosSel([0]),
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 25, None)
        assert splitter.select_range(
            split=vbt.PosSel(0),
            set_=vbt.PosSel([0, 1]),
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 31, None)
        assert splitter.select_range(
            split=vbt.PosSel(0),
            set_=[vbt.PosSel(0), vbt.PosSel(1)],
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 31, None)
        assert splitter.select_range(
            split=vbt.PosSel(0),
            set_=None,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        ) == slice(5, 31, None)

    def test_get_ready_range(self):
        assert vbt.Splitter.get_ready_range(
            vbt.Rep("range_", context=dict(range_=slice(10, 20))), index=index
        ) == slice(10, 20)
        assert vbt.Splitter.get_ready_range(
            vbt.RepEval("slice(10, len(index))"), index=index
        ) == slice(10, len(index))
        assert vbt.Splitter.get_ready_range(
            lambda index: slice(10, len(index)), index=index
        ) == slice(10, len(index))
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(10, index=index)
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(0.5, index=index)
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(vbt.RelRange(), index=index)
        assert vbt.Splitter.get_ready_range(vbt.hslice(10, len(index)), index=index) == slice(
            10, len(index)
        )
        assert vbt.Splitter.get_ready_range(slice(None), index=index) == slice(0, len(index))
        assert vbt.Splitter.get_ready_range(slice(None, len(index)), index=index) == slice(
            0, len(index)
        )
        assert vbt.Splitter.get_ready_range(slice(0, len(index)), index=index) == slice(
            0, len(index)
        )
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(slice(0, len(index), 2), index=index)
        assert vbt.Splitter.get_ready_range(slice(-5, 0), index=index) == slice(
            len(index) - 5, len(index)
        )
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(slice(-5, 1), index=index)
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(slice(0, 0), index=index)
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(slice(5, 10), range_format="indices", index=index),
            np.array([5, 6, 7, 8, 9]),
        )
        mask = np.full(len(index), False)
        mask[5:10] = True
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(slice(5, 10), range_format="mask", index=index),
            mask,
        )
        assert vbt.Splitter.get_ready_range(slice(None, index[4]), index=index) == slice(0, 4)
        assert vbt.Splitter.get_ready_range(slice(index[1], None), index=index) == slice(
            1, len(index)
        )
        assert vbt.Splitter.get_ready_range(slice(1, index[4]), index=index) == slice(1, 4)
        assert vbt.Splitter.get_ready_range(slice(index[1], 4), index=index) == slice(1, 4)
        assert vbt.Splitter.get_ready_range(slice(index[1], index[4]), index=index) == slice(1, 4)
        assert vbt.Splitter.get_ready_range(
            slice(index[0] - pd.Timedelta(days=1), index[-1] + pd.Timedelta(days=1)), index=index
        ) == slice(0, len(index))
        assert vbt.Splitter.get_ready_range(
            slice(index[1].to_datetime64(), index[4].to_datetime64()), index=index
        ) == slice(1, 4)
        assert vbt.Splitter.get_ready_range(
            slice(str(index[1].to_datetime64()), str(index[4].to_datetime64())), index=index
        ) == slice(1, 4)
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(slice("hello", str(index[4].to_datetime64())), index=index)
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(slice(str(index[1].to_datetime64()), "hello"), index=index)

        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(np.array([3, 2, 1]), index=index),
            np.array([3, 2, 1]),
        )
        assert vbt.Splitter.get_ready_range(np.array([1, 2, 3]), index=index) == slice(1, 4)
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(np.array([1, 2, 3]), range_format="indices", index=index),
            np.array([1, 2, 3]),
        )
        mask = np.full(len(index), False)
        mask[[1, 2, 3]] = True
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(np.array([1, 2, 3]), range_format="mask", index=index),
            mask,
        )
        assert vbt.Splitter.get_ready_range(
            np.array([1, 2, 3]), range_format="slice", index=index
        ) == slice(1, 4)
        assert vbt.Splitter.get_ready_range(
            np.array([1, 2, 3]), range_format="slice_or_indices", index=index
        ) == slice(1, 4)
        assert vbt.Splitter.get_ready_range(
            np.array([1, 2, 3]), range_format="slice_or_mask", index=index
        ) == slice(1, 4)
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(np.array([1, 3]), index=index),
            np.array([1, 3]),
        )
        mask = np.full(len(index), False)
        mask[[1, 3]] = True
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(np.array([1, 3]), range_format="mask", index=index),
            mask,
        )
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(np.array([1, 3]), range_format="slice", index=index)
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(
                np.array([1, 3]), range_format="slice_or_indices", index=index
            ),
            np.array([1, 3]),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(
                np.array([1, 3]), range_format="slice_or_mask", index=index
            ),
            mask,
        )
        assert vbt.Splitter.get_ready_range(index[1:3], index=index) == slice(1, 3)
        assert vbt.Splitter.get_ready_range([index[1], index[2]], index=index) == slice(1, 3)
        assert vbt.Splitter.get_ready_range([str(index[1]), str(index[2])], index=index) == slice(
            1, 3
        )
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(["hello", "world"], index=index)

        mask = np.full(len(index), False)
        mask[[1, 2, 3]] = True
        assert vbt.Splitter.get_ready_range(mask, index=index) == slice(1, 4)
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(mask, range_format="indices", index=index),
            np.array([1, 2, 3]),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(mask, range_format="mask", index=index),
            mask.astype(bool),
        )
        assert vbt.Splitter.get_ready_range(mask, range_format="slice", index=index) == slice(1, 4)
        assert vbt.Splitter.get_ready_range(
            mask, range_format="slice_or_indices", index=index
        ) == slice(1, 4)
        assert vbt.Splitter.get_ready_range(
            mask, range_format="slice_or_mask", index=index
        ) == slice(1, 4)
        mask = np.full(len(index), False)
        mask[[1, 3]] = True
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(mask, index=index),
            mask.astype(bool),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(mask, range_format="indices", index=index),
            np.array([1, 3]),
        )
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(mask, range_format="slice", index=index)
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(mask, range_format="slice_or_mask", index=index),
            mask.astype(bool),
        )
        np.testing.assert_array_equal(
            vbt.Splitter.get_ready_range(mask, range_format="slice_or_indices", index=index),
            np.array([1, 3]),
        )

        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(np.array([-1, -2]), index=index)
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(np.array([0, 100]), index=index)
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(np.array([100, 0]), index=index)
        with pytest.raises(Exception):
            vbt.Splitter.get_ready_range(np.array([100, 200]), index=index)
        assert vbt.Splitter.get_ready_range(
            vbt.FixRange(vbt.Rep("range_", context=dict(range_=lambda index: vbt.hslice(10, 20)))),
            index=index,
            return_meta=True,
        ) == {
            "was_fixed": True,
            "was_template": True,
            "was_callable": True,
            "was_relative": False,
            "was_hslice": True,
            "was_slice": True,
            "was_neg_slice": False,
            "was_datetime": False,
            "was_mask": False,
            "was_indices": False,
            "is_constant": True,
            "start": 10,
            "stop": 20,
            "length": 10,
            "range_": slice(10, 20, None),
        }
        mask = np.full(len(index), False)
        mask[[1, 2, 3]] = True
        assert vbt.Splitter.get_ready_range(mask, index=index, return_meta=True) == {
            "was_fixed": False,
            "was_template": False,
            "was_callable": False,
            "was_relative": False,
            "was_hslice": False,
            "was_slice": False,
            "was_neg_slice": False,
            "was_datetime": False,
            "was_mask": True,
            "was_indices": False,
            "is_constant": True,
            "start": 1,
            "stop": 4,
            "length": 3,
            "range_": slice(1, 4, None),
        }
        assert vbt.Splitter.get_ready_range(np.array([1, 2, 3]), index=index, return_meta=True) == {
            "was_fixed": False,
            "was_template": False,
            "was_callable": False,
            "was_relative": False,
            "was_hslice": False,
            "was_slice": False,
            "was_neg_slice": False,
            "was_datetime": False,
            "was_mask": False,
            "was_indices": True,
            "is_constant": True,
            "start": 1,
            "stop": 4,
            "length": 3,
            "range_": slice(1, 4, None),
        }

    def test_split_range(self):
        with pytest.raises(Exception):
            vbt.Splitter.split_range(20, slice(None), index=index)
        with pytest.raises(Exception):
            vbt.Splitter.split_range(0.5, slice(None), index=index)
        with pytest.raises(Exception):
            vbt.Splitter.split_range(vbt.RelRange(), slice(None), index=index)
        assert vbt.Splitter.split_range(slice(None), slice(None), index=index) == (
            slice(0, len(index)),
        )
        assert vbt.Splitter.split_range(slice(None), 0.75, index=index) == (
            slice(0, 23, None),
            slice(23, 31, None),
        )
        assert vbt.Splitter.split_range(slice(None), 0.75, backwards=True, index=index) == (
            slice(0, 8, None),
            slice(8, 31, None),
        )
        assert vbt.Splitter.split_range(slice(None), -0.25, index=index) == (
            slice(0, 24, None),
            slice(24, 31, None),
        )
        assert vbt.Splitter.split_range(slice(None), -0.25, backwards=True, index=index) == (
            slice(0, 7, None),
            slice(7, 31, None),
        )
        assert vbt.Splitter.split_range(
            slice(None), (vbt.RelRange(length=10), vbt.RelRange(length=5)), index=index
        ) == (slice(0, 10, None), slice(10, 15, None))
        assert vbt.Splitter.split_range(
            slice(None),
            (vbt.RelRange(length=10), vbt.RelRange(length=5)),
            backwards=True,
            index=index,
        ) == (slice(16, 26, None), slice(26, 31, None))
        assert vbt.Splitter.split_range(
            slice(None),
            (
                vbt.RelRange(length=10, offset_anchor="prev_start", offset=10),
                vbt.RelRange(length=5, offset_anchor="prev_start"),
            ),
            index=index,
        ) == (slice(10, 20, None), slice(10, 15, None))
        assert vbt.Splitter.split_range(
            slice(None),
            (
                vbt.RelRange(length=10, offset_anchor="prev_start", offset=10),
                vbt.RelRange(length=5, offset_anchor="prev_start"),
            ),
            backwards=True,
            index=index,
        ) == (slice(11, 21, None), slice(26, 31, None))
        assert vbt.Splitter.split_range(
            slice(None),
            (
                np.arange(10, 20),
                vbt.RelRange(length=5, offset_anchor="prev_start"),
            ),
            index=index,
        ) == (slice(10, 20, None), slice(10, 15, None))
        assert vbt.Splitter.split_range(
            slice(None),
            (
                vbt.RelRange(length=10, offset_anchor="prev_start", offset=10),
                np.arange(26, 31),
            ),
            backwards=True,
            index=index,
        ) == (slice(11, 21, None), slice(26, 31, None))
        mask = np.full(len(index), False)
        mask[10:20] = True
        assert vbt.Splitter.split_range(
            slice(None),
            (
                mask,
                vbt.RelRange(length=5, offset_anchor="prev_start"),
            ),
            index=index,
        ) == (slice(10, 20, None), slice(10, 15, None))
        mask = np.full(len(index), False)
        mask[26:31] = True
        assert vbt.Splitter.split_range(
            slice(None),
            (
                vbt.RelRange(length=10, offset_anchor="prev_start", offset=10),
                mask,
            ),
            backwards=True,
            index=index,
        ) == (slice(11, 21, None), slice(26, 31, None))
        assert vbt.Splitter.split_range(
            slice(None),
            (
                vbt.RelRange(length=10, offset_anchor="prev_start", offset=10),
                vbt.RelRange(length=5, is_gap=True),
                vbt.RelRange(length=5),
            ),
            index=index,
        ) == (slice(10, 20, None), slice(25, 30, None))
        mask = np.full(len(index), False)
        mask[15:20] = True
        assert vbt.Splitter.split_range(
            vbt.hslice(),
            [
                slice(0, 5),
                vbt.hslice(5, 10),
                np.arange(10, 15),
                mask,
            ],
            index=index,
        ) == (
            vbt.hslice(start=0, stop=5, step=None),
            vbt.hslice(start=5, stop=10, step=None),
            vbt.hslice(start=10, stop=15, step=None),
            vbt.hslice(start=15, stop=20, step=None),
        )
        target_mask = np.full((4, len(index)), False)
        target_mask[0, 0:5] = True
        target_mask[1, 5:10] = True
        target_mask[2, 10:15] = True
        target_mask[3, 15:20] = True
        np.testing.assert_array_equal(
            np.asarray(
                vbt.Splitter.split_range(
                    vbt.hslice(),
                    [
                        slice(0, 5),
                        vbt.hslice(5, 10),
                        np.arange(10, 15),
                        mask,
                    ],
                    index=index,
                    range_format="mask",
                )
            ),
            target_mask,
        )
        mask = np.full(len(index), False)
        mask[[16, 18, 20]] = True
        new_split = vbt.Splitter.split_range(
            vbt.hslice(),
            [
                slice(0, 5),
                vbt.hslice(5, 10),
                np.array([10, 12, 14]),
                mask,
            ],
            index=index,
            backwards=True,
        )
        assert new_split[0] == vbt.hslice(start=0, stop=5, step=None)
        assert new_split[1] == vbt.hslice(start=5, stop=10, step=None)
        np.testing.assert_array_equal(new_split[2], np.array([10, 12, 14]))
        np.testing.assert_array_equal(new_split[3], np.array([16, 18, 20]))
        assert vbt.Splitter.split_range(
            np.array([0, 2, 4, 5, 7, 8, 9, 11]), "by_gap", index=index
        ) == (
            slice(0, 1, None),
            slice(2, 3, None),
            slice(4, 6, None),
            slice(7, 10, None),
            slice(11, 12, None),
        )
        mask = np.full(len(index), False)
        mask[[0, 2, 4, 5, 7, 8, 9, 11]] = True
        assert vbt.Splitter.split_range(mask, "by_gap", index=index) == (
            slice(0, 1, None),
            slice(2, 3, None),
            slice(4, 6, None),
            slice(7, 10, None),
            slice(11, 12, None),
        )

    def test_merge_split(self):
        assert vbt.Splitter.merge_split((slice(10, 20), slice(20, 30)), index=index) == slice(
            10, 30
        )
        assert vbt.Splitter.merge_split(
            (vbt.hslice(10, 20), vbt.hslice(20, 30)), wrap_with_hslice=None, index=index
        ) == vbt.hslice(10, 30)
        assert vbt.Splitter.merge_split((np.array([1, 3]), np.array([2, 4])), index=index) == slice(
            1, 5
        )
        mask1 = np.full(len(index), False)
        mask1[10:20] = True
        mask2 = np.full(len(index), False)
        mask2[20:30] = True
        assert vbt.Splitter.merge_split((mask1, mask2), index=index) == slice(10, 30)
        mask1 = np.full(len(index), False)
        mask1[[2, 4, 6]] = True
        mask2 = np.full(len(index), False)
        mask2[[8, 10, 12]] = True
        target_mask = np.full(len(index), False)
        target_mask[[2, 4, 6, 8, 10, 12]] = True
        np.testing.assert_array_equal(
            vbt.Splitter.merge_split((mask1, mask2), index=index),
            target_mask,
        )
        np.testing.assert_array_equal(
            vbt.Splitter.merge_split((slice(10, 20), mask2), index=index),
            np.array([8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        )

    def test_to_fixed(self):
        rel_splitter = vbt.Splitter.from_splits(index, [[0.25, 0.5], [0.75, 1.0]], fix_ranges=False)
        np.testing.assert_array_equal(
            rel_splitter.to_fixed().splits_arr,
            np.array(
                [[slice(0, 7, None), slice(7, 19, None)], [slice(0, 23, None), slice(23, 31, None)]]
            ),
        )
        target_mask = np.full((2, 2, len(index)), False)
        target_mask[0, 0, slice(0, 7, None)] = True
        target_mask[0, 1, slice(7, 19, None)] = True
        target_mask[1, 0, slice(0, 23, None)] = True
        target_mask[1, 1, slice(23, 31, None)] = True
        assert rel_splitter.to_fixed(
            split_range_kwargs=dict(range_format="mask")
        ).splits_arr.shape == (2, 2)
        np.testing.assert_array_equal(
            rel_splitter.to_fixed(split_range_kwargs=dict(range_format="mask"))
            .splits_arr[0, 0]
            .range_,
            target_mask[0, 0],
        )
        np.testing.assert_array_equal(
            rel_splitter.to_fixed(split_range_kwargs=dict(range_format="mask"))
            .splits_arr[0, 1]
            .range_,
            target_mask[0, 1],
        )
        np.testing.assert_array_equal(
            rel_splitter.to_fixed(split_range_kwargs=dict(range_format="mask"))
            .splits_arr[1, 0]
            .range_,
            target_mask[1, 0],
        )
        np.testing.assert_array_equal(
            rel_splitter.to_fixed(split_range_kwargs=dict(range_format="mask"))
            .splits_arr[1, 1]
            .range_,
            target_mask[1, 1],
        )

    def test_to_grouped(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 15), slice(10, 20)],
                [slice(10, 20), slice(15, 25)],
                [slice(15, 25), slice(20, None)],
            ],
        )
        new_splitter = splitter.to_grouped(split_group_by=[0, 1, 0])
        np.testing.assert_array_equal(
            new_splitter.splits_arr,
            np.array(
                [
                    [slice(5, 25, None), slice(10, 31, None)],
                    [slice(10, 20, None), slice(15, 25, None)],
                ],
                dtype=object,
            ),
        )
        assert_index_equal(
            new_splitter.wrapper.index, pd.Index([0, 1], dtype="int64", name="split_group")
        )
        assert_index_equal(
            new_splitter.wrapper.columns, pd.Index(["set_0", "set_1"], dtype="object", name="set")
        )
        assert new_splitter.wrapper.ndim == 2
        new_splitter = splitter.to_grouped(split_group_by=[0, 1, 0], set_group_by=True)
        np.testing.assert_array_equal(
            new_splitter.splits_arr,
            np.array([[slice(5, 31, None)], [slice(10, 25, None)]], dtype=object),
        )
        assert_index_equal(
            new_splitter.wrapper.index, pd.Index([0, 1], dtype="int64", name="split_group")
        )
        assert_index_equal(
            new_splitter.wrapper.columns, pd.Index(["group"], dtype="object", name="set_group")
        )
        assert new_splitter.wrapper.ndim == 1

    def test_remap_range(self):
        def test_mask(out, slice_):
            mask = np.full(len(out), False)
            mask[slice_] = True
            np.testing.assert_array_equal(out, mask)

        source_index = pd.date_range("2020-01-02", "2020-01-04", tz="utc")
        target_index = pd.date_range("2020-01-01", "2020-01-03", tz="utc")
        test_mask(
            vbt.Splitter.remap_range(slice(None), target_index, index=source_index), slice(1, 3)
        )
        target_index = pd.date_range("2020-01-02", "2020-01-04", tz="utc")
        assert vbt.Splitter.remap_range(slice(None), target_index, index=source_index) == slice(
            None, None, None
        )
        target_index = pd.date_range("2020-01-03", "2020-01-05", tz="utc")
        test_mask(
            vbt.Splitter.remap_range(slice(None), target_index, index=source_index), slice(0, 2)
        )
        target_index = pd.date_range(
            "2020-01-01 21:00:00", "2020-01-02 21:00:00", freq="4h", tz="utc"
        )
        test_mask(
            vbt.Splitter.remap_range(slice(None), target_index, index=source_index), slice(1, 7)
        )
        target_index = pd.date_range(
            "2020-01-02 21:00:00", "2020-01-03 21:00:00", freq="4h", tz="utc"
        )
        test_mask(
            vbt.Splitter.remap_range(slice(None), target_index, index=source_index), slice(0, 7)
        )
        target_index = pd.date_range(
            "2020-01-03 21:00:00", "2020-01-04 21:00:00", freq="4h", tz="utc"
        )
        test_mask(
            vbt.Splitter.remap_range(slice(None), target_index, index=source_index), slice(0, 6)
        )
        target_index = pd.date_range("2020-01-01", "2020-01-04", freq="2d", tz="utc")
        test_mask(
            vbt.Splitter.remap_range(slice(None), target_index, index=source_index), slice(1, 2)
        )
        target_index = pd.date_range("2020-01-02", "2020-01-05", freq="2d", tz="utc")
        test_mask(
            vbt.Splitter.remap_range(slice(None), target_index, index=source_index), slice(0, 1)
        )

        target_index = pd.date_range("2020-01-01", "2020-01-03", tz="utc")
        test_mask(vbt.Splitter.remap_range([0, 2], target_index, index=source_index), slice(1, 2))
        target_index = pd.date_range("2020-01-02", "2020-01-04", tz="utc")
        np.testing.assert_array_equal(
            vbt.Splitter.remap_range([0, 2], target_index, index=source_index),
            np.array([0, 2]),
        )
        target_index = pd.date_range("2020-01-03", "2020-01-05", tz="utc")
        test_mask(vbt.Splitter.remap_range([0, 2], target_index, index=source_index), slice(1, 2))
        target_index = pd.date_range(
            "2020-01-01 21:00:00", "2020-01-02 21:00:00", freq="4h", tz="utc"
        )
        test_mask(vbt.Splitter.remap_range([0, 2], target_index, index=source_index), slice(1, 6))
        target_index = pd.date_range(
            "2020-01-02 21:00:00", "2020-01-03 21:00:00", freq="4h", tz="utc"
        )
        test_mask(vbt.Splitter.remap_range([0, 2], target_index, index=source_index), slice(0, 0))
        target_index = pd.date_range(
            "2020-01-03 21:00:00", "2020-01-04 21:00:00", freq="4h", tz="utc"
        )
        test_mask(vbt.Splitter.remap_range([0, 2], target_index, index=source_index), slice(1, 6))
        target_index = pd.date_range("2020-01-01", "2020-01-04", freq="2d", tz="utc")
        test_mask(vbt.Splitter.remap_range([0, 2], target_index, index=source_index), slice(0, 0))
        target_index = pd.date_range("2020-01-02", "2020-01-05", freq="2d", tz="utc")
        test_mask(vbt.Splitter.remap_range([0, 2], target_index, index=source_index), slice(0, 0))

    def test_get_ready_obj_range(self):
        arr = np.arange(len(index))
        assert vbt.Splitter.get_ready_obj_range(arr, slice(None), index=index) == slice(0, 31)
        assert vbt.Splitter.get_ready_obj_range(arr, slice(5, 10), index=index) == slice(5, 10)
        assert vbt.Splitter.get_ready_obj_range(index, slice(None), index=index) == slice(0, 31)
        assert vbt.Splitter.get_ready_obj_range(index, slice(5, 10), index=index) == slice(5, 10)
        sr = pd.Series(np.arange(len(index)), index=index)
        assert vbt.Splitter.get_ready_obj_range(sr, slice(None), index=index) == slice(0, 31)
        assert vbt.Splitter.get_ready_obj_range(sr, slice(5, 10), index=index) == slice(5, 10)
        df = pd.DataFrame(np.arange(len(index)), index=index)
        assert vbt.Splitter.get_ready_obj_range(df, slice(None), index=index) == slice(0, 31)
        assert vbt.Splitter.get_ready_obj_range(df, slice(5, 10), index=index) == slice(5, 10)

        obj_index = index.shift(-10)
        arr = np.arange(len(obj_index))
        assert vbt.Splitter.get_ready_obj_range(
            arr, slice(None), obj_index=obj_index, index=index
        ) == slice(10, 31)
        assert vbt.Splitter.get_ready_obj_range(
            arr, slice(5, 10), obj_index=obj_index, index=index
        ) == slice(15, 20)
        assert vbt.Splitter.get_ready_obj_range(obj_index, slice(None), index=index) == slice(
            10, 31
        )
        assert vbt.Splitter.get_ready_obj_range(obj_index, slice(5, 10), index=index) == slice(
            15, 20
        )
        sr = pd.Series(np.arange(len(obj_index)), index=obj_index)
        assert vbt.Splitter.get_ready_obj_range(sr, slice(None), index=index) == slice(10, 31)
        assert vbt.Splitter.get_ready_obj_range(sr, slice(5, 10), index=index) == slice(15, 20)
        assert vbt.Splitter.get_ready_obj_range(
            sr, slice(None), remap_to_obj=False, index=index
        ) == slice(0, 31)
        assert vbt.Splitter.get_ready_obj_range(
            sr, slice(5, 10), remap_to_obj=False, index=index
        ) == slice(5, 10)
        df = pd.DataFrame(np.arange(len(obj_index)), index=obj_index)
        assert vbt.Splitter.get_ready_obj_range(df, slice(None), index=index) == slice(10, 31)
        assert vbt.Splitter.get_ready_obj_range(df, slice(5, 10), index=index) == slice(15, 20)

    def test_take_split_major_meta(self):
        sr = pd.Series(np.arange(len(index)), index=index.shift(-5))
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 15), slice(10, 25)],
                [slice(5, 20), slice(15, None)],
            ],
        )
        new_obj = list(
            splitter.take(
                sr,
                into="split_major_meta",
                attach_bounds="target_index",
            )[1]
        )
        assert len(new_obj) == 4
        assert new_obj[0]["split_idx"] == 0
        assert new_obj[0]["set_idx"] == 0
        assert new_obj[0]["range_meta"]["range_"] == slice(0, 15, None)
        assert new_obj[0]["obj_range_meta"]["range_"] == slice(5, 20, None)
        assert new_obj[0]["bounds"] == (
            pd.Timestamp("2020-01-01 00:00:00"),
            pd.Timestamp("2020-01-16 00:00:00"),
        )
        assert_series_equal(new_obj[0]["obj_slice"], sr.iloc[5:20])
        assert new_obj[1]["split_idx"] == 0
        assert new_obj[1]["set_idx"] == 1
        assert new_obj[1]["range_meta"]["range_"] == slice(10, 25, None)
        assert new_obj[1]["obj_range_meta"]["range_"] == slice(15, 30, None)
        assert new_obj[1]["bounds"] == (
            pd.Timestamp("2020-01-11 00:00:00"),
            pd.Timestamp("2020-01-26 00:00:00"),
        )
        assert_series_equal(new_obj[1]["obj_slice"], sr.iloc[15:30])
        assert new_obj[2]["split_idx"] == 1
        assert new_obj[2]["set_idx"] == 0
        assert new_obj[2]["range_meta"]["range_"] == slice(5, 20, None)
        assert new_obj[2]["obj_range_meta"]["range_"] == slice(10, 25, None)
        assert new_obj[2]["bounds"] == (
            pd.Timestamp("2020-01-06 00:00:00"),
            pd.Timestamp("2020-01-21 00:00:00"),
        )
        assert_series_equal(new_obj[2]["obj_slice"], sr.iloc[10:25])
        assert new_obj[3]["split_idx"] == 1
        assert new_obj[3]["set_idx"] == 1
        assert new_obj[3]["range_meta"]["range_"] == slice(15, 31, None)
        assert new_obj[3]["obj_range_meta"]["range_"] == slice(20, 31, None)
        assert new_obj[3]["bounds"] == (
            pd.Timestamp("2020-01-16 00:00:00"),
            pd.Timestamp("2020-01-27 00:00:00"),
        )
        assert_series_equal(new_obj[3]["obj_slice"], sr.iloc[20:31])

        new_obj = list(
            splitter.take(
                sr,
                into="split_major_meta",
                split_group_by=["a", "a"],
                set_group_by=["b", "b"],
                attach_bounds="target_index",
            )[1]
        )
        assert len(new_obj) == 1
        assert new_obj[0]["split_idx"] == 0
        assert new_obj[0]["set_idx"] == 0
        assert new_obj[0]["range_meta"]["range_"] == slice(0, 31, None)
        assert new_obj[0]["obj_range_meta"]["range_"] == slice(5, 31, None)
        assert new_obj[0]["bounds"] == (
            pd.Timestamp("2020-01-01 00:00:00"),
            pd.Timestamp("2020-01-27 00:00:00"),
        )
        assert_series_equal(new_obj[0]["obj_slice"], sr.iloc[5:31])

    def test_take_set_major_meta(self):
        sr = pd.Series(np.arange(len(index)), index=index.shift(-5))
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 15), slice(10, 25)],
                [slice(5, 20), slice(15, None)],
            ],
        )
        new_obj = list(
            splitter.take(
                sr,
                into="set_major_meta",
                attach_bounds="target_index",
            )[1]
        )
        assert len(new_obj) == 4
        assert new_obj[0]["split_idx"] == 0
        assert new_obj[0]["set_idx"] == 0
        assert new_obj[0]["range_meta"]["range_"] == slice(0, 15, None)
        assert new_obj[0]["obj_range_meta"]["range_"] == slice(5, 20, None)
        assert new_obj[0]["bounds"] == (
            pd.Timestamp("2020-01-01 00:00:00"),
            pd.Timestamp("2020-01-16 00:00:00"),
        )
        assert_series_equal(new_obj[0]["obj_slice"], sr.iloc[5:20])
        assert new_obj[1]["split_idx"] == 1
        assert new_obj[1]["set_idx"] == 0
        assert new_obj[1]["range_meta"]["range_"] == slice(5, 20, None)
        assert new_obj[1]["obj_range_meta"]["range_"] == slice(10, 25, None)
        assert new_obj[1]["bounds"] == (
            pd.Timestamp("2020-01-06 00:00:00"),
            pd.Timestamp("2020-01-21 00:00:00"),
        )
        assert_series_equal(new_obj[1]["obj_slice"], sr.iloc[10:25])
        assert new_obj[2]["split_idx"] == 0
        assert new_obj[2]["set_idx"] == 1
        assert new_obj[2]["range_meta"]["range_"] == slice(10, 25, None)
        assert new_obj[2]["obj_range_meta"]["range_"] == slice(15, 30, None)
        assert new_obj[2]["bounds"] == (
            pd.Timestamp("2020-01-11 00:00:00"),
            pd.Timestamp("2020-01-26 00:00:00"),
        )
        assert_series_equal(new_obj[2]["obj_slice"], sr.iloc[15:30])
        assert new_obj[3]["split_idx"] == 1
        assert new_obj[3]["set_idx"] == 1
        assert new_obj[3]["range_meta"]["range_"] == slice(15, 31, None)
        assert new_obj[3]["obj_range_meta"]["range_"] == slice(20, 31, None)
        assert new_obj[3]["bounds"] == (
            pd.Timestamp("2020-01-16 00:00:00"),
            pd.Timestamp("2020-01-27 00:00:00"),
        )
        assert_series_equal(new_obj[3]["obj_slice"], sr.iloc[20:31])

        new_obj = list(
            splitter.take(
                sr,
                into="set_major_meta",
                split_group_by=["a", "a"],
                set_group_by=["b", "b"],
                attach_bounds="target_index",
            )[1]
        )
        assert len(new_obj) == 1
        assert new_obj[0]["split_idx"] == 0
        assert new_obj[0]["set_idx"] == 0
        assert new_obj[0]["range_meta"]["range_"] == slice(0, 31, None)
        assert new_obj[0]["obj_range_meta"]["range_"] == slice(5, 31, None)
        assert new_obj[0]["bounds"] == (
            pd.Timestamp("2020-01-01 00:00:00"),
            pd.Timestamp("2020-01-27 00:00:00"),
        )
        assert_series_equal(new_obj[0]["obj_slice"], sr.iloc[5:31])

    def test_take_none(self):
        sr = pd.Series(np.arange(len(index)), index=index.shift(-5))
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 15), slice(10, 25)],
                [slice(5, 20), slice(15, None)],
            ],
        )
        new_obj = splitter.take(sr)
        assert_index_equal(
            new_obj.index,
            pd.MultiIndex.from_tuples(
                [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
            ),
        )
        new_obj = splitter.take(sr, attach_bounds="target_index")
        assert_index_equal(
            new_obj.index,
            pd.MultiIndex.from_tuples(
                [
                    (0, "set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                    (0, "set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                    (1, "set_0", pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                    (1, "set_1", pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                ],
                names=["split", "set", "start", "end"],
            ),
        )
        assert_series_equal(
            new_obj.iloc[0],
            pd.Series(np.arange(5, 20), index=sr.index[5:20]),
        )
        assert_series_equal(
            new_obj.iloc[1],
            pd.Series(np.arange(15, 30), index=sr.index[15:30]),
        )
        assert_series_equal(
            new_obj.iloc[2],
            pd.Series(np.arange(10, 25), index=sr.index[10:25]),
        )
        assert_series_equal(
            new_obj.iloc[3],
            pd.Series(np.arange(20, 31), index=sr.index[20:31]),
        )
        new_obj = splitter.take(
            sr,
            split_group_by=["a", "a"],
            set_group_by=["b", "b"],
            attach_bounds="target_index",
        )
        assert_series_equal(
            new_obj,
            pd.Series(np.arange(5, 31), index=sr.index[5:31]),
        )
        new_obj = splitter.take(
            sr,
            split_group_by=["a", "a"],
            set_group_by=["b", "b"],
            attach_bounds="target_index",
            squeeze_one_split=False,
            squeeze_one_set=False,
        )
        assert_index_equal(
            new_obj.index,
            pd.MultiIndex.from_tuples(
                [
                    ("a", "b", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-27")),
                ],
                names=["split_group", "set_group", "start", "end"],
            ),
        )
        assert_series_equal(
            new_obj.iloc[0],
            pd.Series(np.arange(5, 31), index=sr.index[5:31]),
        )

    def test_take_stacked(self):
        sr = pd.Series(np.arange(len(index)), index=index.shift(-5))
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 15), slice(10, 25)],
                [slice(5, 20), slice(15, None)],
            ],
        )
        new_obj = splitter.take(sr, into="stacked", attach_bounds="target_index")
        assert_frame_equal(
            new_obj,
            pd.DataFrame(
                [
                    [5.0, np.nan, np.nan, np.nan],
                    [6.0, np.nan, np.nan, np.nan],
                    [7.0, np.nan, np.nan, np.nan],
                    [8.0, np.nan, np.nan, np.nan],
                    [9.0, np.nan, np.nan, np.nan],
                    [10.0, np.nan, 10.0, np.nan],
                    [11.0, np.nan, 11.0, np.nan],
                    [12.0, np.nan, 12.0, np.nan],
                    [13.0, np.nan, 13.0, np.nan],
                    [14.0, np.nan, 14.0, np.nan],
                    [15.0, 15.0, 15.0, np.nan],
                    [16.0, 16.0, 16.0, np.nan],
                    [17.0, 17.0, 17.0, np.nan],
                    [18.0, 18.0, 18.0, np.nan],
                    [19.0, 19.0, 19.0, np.nan],
                    [np.nan, 20.0, 20.0, 20.0],
                    [np.nan, 21.0, 21.0, 21.0],
                    [np.nan, 22.0, 22.0, 22.0],
                    [np.nan, 23.0, 23.0, 23.0],
                    [np.nan, 24.0, 24.0, 24.0],
                    [np.nan, 25.0, np.nan, 25.0],
                    [np.nan, 26.0, np.nan, 26.0],
                    [np.nan, 27.0, np.nan, 27.0],
                    [np.nan, 28.0, np.nan, 28.0],
                    [np.nan, 29.0, np.nan, 29.0],
                    [np.nan, np.nan, np.nan, 30.0],
                ],
                index=sr.index[5:31],
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        (0, "set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                        (1, "set_0", pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                        (1, "set_1", pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["split", "set", "start", "end"],
                ),
            ),
        )
        new_obj = splitter.take(sr, into="stacked", stack_axis=0)
        assert_series_equal(
            new_obj,
            pd.Series(
                [
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                ],
                pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0", pd.Timestamp("2020-01-01")),
                        (0, "set_0", pd.Timestamp("2020-01-02")),
                        (0, "set_0", pd.Timestamp("2020-01-03")),
                        (0, "set_0", pd.Timestamp("2020-01-04")),
                        (0, "set_0", pd.Timestamp("2020-01-05")),
                        (0, "set_0", pd.Timestamp("2020-01-06")),
                        (0, "set_0", pd.Timestamp("2020-01-07")),
                        (0, "set_0", pd.Timestamp("2020-01-08")),
                        (0, "set_0", pd.Timestamp("2020-01-09")),
                        (0, "set_0", pd.Timestamp("2020-01-10")),
                        (0, "set_0", pd.Timestamp("2020-01-11")),
                        (0, "set_0", pd.Timestamp("2020-01-12")),
                        (0, "set_0", pd.Timestamp("2020-01-13")),
                        (0, "set_0", pd.Timestamp("2020-01-14")),
                        (0, "set_0", pd.Timestamp("2020-01-15")),
                        (0, "set_1", pd.Timestamp("2020-01-11")),
                        (0, "set_1", pd.Timestamp("2020-01-12")),
                        (0, "set_1", pd.Timestamp("2020-01-13")),
                        (0, "set_1", pd.Timestamp("2020-01-14")),
                        (0, "set_1", pd.Timestamp("2020-01-15")),
                        (0, "set_1", pd.Timestamp("2020-01-16")),
                        (0, "set_1", pd.Timestamp("2020-01-17")),
                        (0, "set_1", pd.Timestamp("2020-01-18")),
                        (0, "set_1", pd.Timestamp("2020-01-19")),
                        (0, "set_1", pd.Timestamp("2020-01-20")),
                        (0, "set_1", pd.Timestamp("2020-01-21")),
                        (0, "set_1", pd.Timestamp("2020-01-22")),
                        (0, "set_1", pd.Timestamp("2020-01-23")),
                        (0, "set_1", pd.Timestamp("2020-01-24")),
                        (0, "set_1", pd.Timestamp("2020-01-25")),
                        (1, "set_0", pd.Timestamp("2020-01-06")),
                        (1, "set_0", pd.Timestamp("2020-01-07")),
                        (1, "set_0", pd.Timestamp("2020-01-08")),
                        (1, "set_0", pd.Timestamp("2020-01-09")),
                        (1, "set_0", pd.Timestamp("2020-01-10")),
                        (1, "set_0", pd.Timestamp("2020-01-11")),
                        (1, "set_0", pd.Timestamp("2020-01-12")),
                        (1, "set_0", pd.Timestamp("2020-01-13")),
                        (1, "set_0", pd.Timestamp("2020-01-14")),
                        (1, "set_0", pd.Timestamp("2020-01-15")),
                        (1, "set_0", pd.Timestamp("2020-01-16")),
                        (1, "set_0", pd.Timestamp("2020-01-17")),
                        (1, "set_0", pd.Timestamp("2020-01-18")),
                        (1, "set_0", pd.Timestamp("2020-01-19")),
                        (1, "set_0", pd.Timestamp("2020-01-20")),
                        (1, "set_1", pd.Timestamp("2020-01-16")),
                        (1, "set_1", pd.Timestamp("2020-01-17")),
                        (1, "set_1", pd.Timestamp("2020-01-18")),
                        (1, "set_1", pd.Timestamp("2020-01-19")),
                        (1, "set_1", pd.Timestamp("2020-01-20")),
                        (1, "set_1", pd.Timestamp("2020-01-21")),
                        (1, "set_1", pd.Timestamp("2020-01-22")),
                        (1, "set_1", pd.Timestamp("2020-01-23")),
                        (1, "set_1", pd.Timestamp("2020-01-24")),
                        (1, "set_1", pd.Timestamp("2020-01-25")),
                        (1, "set_1", pd.Timestamp("2020-01-26")),
                    ],
                    names=["split", "set", None],
                ),
            ),
        )

        new_obj = splitter.take(
            sr,
            into="stacked",
            split_group_by=["a", "a"],
            set_group_by=["b", "b"],
            attach_bounds="target_index",
        )
        assert_series_equal(
            new_obj,
            pd.Series(np.arange(5, 31), index=sr.index[5:31]),
        )
        new_obj = splitter.take(
            sr,
            into="stacked",
            split_group_by=["a", "a"],
            set_group_by=["b", "b"],
            attach_bounds="target_index",
            squeeze_one_split=False,
            squeeze_one_set=False,
        )
        assert_frame_equal(
            new_obj,
            pd.DataFrame(
                np.arange(5, 31)[:, None],
                index=sr.index[5:31],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", "b", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["split_group", "set_group", "start", "end"],
                ),
            ),
        )

    def test_take_stacked_by_split(self):
        sr = pd.Series(np.arange(len(index)), index=index.shift(-5))
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 15), slice(10, 25)],
                [slice(5, 20), slice(15, None)],
            ],
        )
        new_obj = splitter.take(
            sr,
            into="stacked_by_split",
            attach_bounds="target_index",
        )
        assert isinstance(new_obj, pd.Series)
        assert_index_equal(new_obj.index, pd.Index([0, 1], dtype="int64", name="split"))
        assert_frame_equal(
            new_obj.iloc[0],
            pd.DataFrame(
                [
                    [5.0, np.nan],
                    [6.0, np.nan],
                    [7.0, np.nan],
                    [8.0, np.nan],
                    [9.0, np.nan],
                    [10.0, np.nan],
                    [11.0, np.nan],
                    [12.0, np.nan],
                    [13.0, np.nan],
                    [14.0, np.nan],
                    [15.0, 15.0],
                    [16.0, 16.0],
                    [17.0, 17.0],
                    [18.0, 18.0],
                    [19.0, 19.0],
                    [np.nan, 20.0],
                    [np.nan, 21.0],
                    [np.nan, 22.0],
                    [np.nan, 23.0],
                    [np.nan, 24.0],
                    [np.nan, 25.0],
                    [np.nan, 26.0],
                    [np.nan, 27.0],
                    [np.nan, 28.0],
                    [np.nan, 29.0],
                ],
                index=sr.index[5:30],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        ("set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                    ],
                    names=["set", "start", "end"],
                ),
            ),
        )
        assert_frame_equal(
            new_obj.iloc[1],
            pd.DataFrame(
                [
                    [10.0, np.nan],
                    [11.0, np.nan],
                    [12.0, np.nan],
                    [13.0, np.nan],
                    [14.0, np.nan],
                    [15.0, np.nan],
                    [16.0, np.nan],
                    [17.0, np.nan],
                    [18.0, np.nan],
                    [19.0, np.nan],
                    [20.0, 20.0],
                    [21.0, 21.0],
                    [22.0, 22.0],
                    [23.0, 23.0],
                    [24.0, 24.0],
                    [np.nan, 25.0],
                    [np.nan, 26.0],
                    [np.nan, 27.0],
                    [np.nan, 28.0],
                    [np.nan, 29.0],
                    [np.nan, 30.0],
                ],
                index=sr.index[10:31],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("set_0", pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                        ("set_1", pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            splitter.take(
                sr,
                into="stacked_by_split",
                attach_bounds="target_index",
                split_group_by=True,
                set_group_by=True,
            ),
            sr.iloc[5:31],
        )
        new_obj = splitter.take(
            sr,
            into="stacked_by_split",
            attach_bounds="target_index",
            split_group_by=False,
            set_group_by=True,
        )
        assert_index_equal(
            new_obj.index,
            pd.MultiIndex.from_tuples(
                [
                    (0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-26")),
                    (1, pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-27")),
                ],
                names=["split", "start", "end"],
            ),
        )
        assert_series_equal(
            new_obj.iloc[0],
            pd.Series(sr.iloc[5:30], index=sr.index[5:30]),
        )
        assert_series_equal(
            new_obj.iloc[1],
            pd.Series(sr.iloc[10:31], index=sr.index[10:31]),
        )

    def test_take_stacked_by_set(self):
        sr = pd.Series(np.arange(len(index)), index=index.shift(-5))
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 15), slice(10, 25)],
                [slice(5, 20), slice(15, None)],
            ],
        )
        new_obj = splitter.take(
            sr,
            into="stacked_by_set",
            attach_bounds="target_index",
        )
        assert isinstance(new_obj, pd.Series)
        assert_index_equal(new_obj.index, pd.Index(["set_0", "set_1"], dtype="object", name="set"))
        assert_frame_equal(
            new_obj.iloc[0],
            pd.DataFrame(
                [
                    [5.0, np.nan],
                    [6.0, np.nan],
                    [7.0, np.nan],
                    [8.0, np.nan],
                    [9.0, np.nan],
                    [10.0, 10.0],
                    [11.0, 11.0],
                    [12.0, 12.0],
                    [13.0, 13.0],
                    [14.0, 14.0],
                    [15.0, 15.0],
                    [16.0, 16.0],
                    [17.0, 17.0],
                    [18.0, 18.0],
                    [19.0, 19.0],
                    [np.nan, 20.0],
                    [np.nan, 21.0],
                    [np.nan, 22.0],
                    [np.nan, 23.0],
                    [np.nan, 24.0],
                ],
                index=sr.index[5:25],
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        (1, pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                    ],
                    names=["split", "start", "end"],
                ),
            ),
        )
        assert_frame_equal(
            new_obj.iloc[1],
            pd.DataFrame(
                [
                    [15.0, np.nan],
                    [16.0, np.nan],
                    [17.0, np.nan],
                    [18.0, np.nan],
                    [19.0, np.nan],
                    [20.0, 20.0],
                    [21.0, 21.0],
                    [22.0, 22.0],
                    [23.0, 23.0],
                    [24.0, 24.0],
                    [25.0, 25.0],
                    [26.0, 26.0],
                    [27.0, 27.0],
                    [28.0, 28.0],
                    [29.0, 29.0],
                    [np.nan, 30.0],
                ],
                index=sr.index[15:31],
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                        (1, pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["split", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            splitter.take(
                sr,
                into="stacked_by_set",
                attach_bounds="target_index",
                split_group_by=True,
                set_group_by=True,
            ),
            sr.iloc[5:31],
        )
        new_obj = splitter.take(
            sr,
            into="stacked_by_set",
            attach_bounds="target_index",
            split_group_by=True,
            set_group_by=False,
        )
        assert_index_equal(
            new_obj.index,
            pd.MultiIndex.from_tuples(
                [
                    ("set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-21")),
                    ("set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-27")),
                ],
                names=["set", "start", "end"],
            ),
        )
        assert_series_equal(
            new_obj.iloc[0],
            pd.Series(sr.iloc[5:25], index=sr.index[5:25]),
        )
        assert_series_equal(
            new_obj.iloc[1],
            pd.Series(sr.iloc[15:31], index=sr.index[15:31]),
        )

    def test_apply(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 15), slice(10, 25)],
                [slice(5, 20), slice(15, None)],
            ],
        )

        def apply_func(a, *args, b=2, **kwargs):
            return a + np.sum(args) + b + np.sum(list(kwargs.values()))

        assert_series_equal(
            splitter.apply(
                apply_func,
                1,
                2,
                vbt.Rep("split_idx"),
                c=4,
                d=vbt.Rep("set_idx"),
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [9, 10, 10, 11],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )

        def apply_func(a, b=None):
            return a.sum() + b.sum()

        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(np.arange(len(index))),
                b=vbt.Takeable(np.arange(len(index))),
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [210, 510, 360, 720],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.RepEval("np.arange(len(index))[range_]"),
                b=vbt.RepEval("np.arange(len(index))[range_]"),
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [210, 510, 360, 720],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.RepEval("np.arange(index_len)[range_]"),
                b=vbt.RepEval("np.arange(index_len)[range_]"),
                merge_all=True,
                merge_func="concat",
                template_context=dict(index_len=len(index)),
            ),
            pd.Series(
                [210, 510, 360, 720],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        sr = pd.Series(np.arange(len(index)), index=index.shift(-5))
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr),
                b=vbt.Takeable(sr),
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [360, 660, 510, 550],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values),
                b=vbt.Takeable(sr.values),
                obj_index=sr.index,
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [360, 660, 510, 550],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 674, 555, 489],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                attach_bounds=True,
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 674, 555, 489],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0", 0, 15),
                        (0, "set_1", 10, 25),
                        (1, "set_0", 5, 20),
                        (1, "set_1", 15, 31),
                    ],
                    names=["split", "set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                attach_bounds=True,
                right_inclusive=True,
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 674, 555, 489],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0", 0, 14),
                        (0, "set_1", 10, 24),
                        (1, "set_0", 5, 19),
                        (1, "set_1", 15, 30),
                    ],
                    names=["split", "set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                attach_bounds="index",
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 674, 555, 489],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        (0, "set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                        (1, "set_0", pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                        (1, "set_1", pd.Timestamp("2020-01-16"), pd.Timestamp("2020-02-01")),
                    ],
                    names=["split", "set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                attach_bounds="target",
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 674, 555, 489],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0", 5, 20),
                        (0, "set_1", 15, 30),
                        (1, "set_0", 10, 25),
                        (1, "set_1", 20, 31),
                    ],
                    names=["split", "set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                attach_bounds="target_index",
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 674, 555, 489],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        (0, "set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                        (1, "set_0", pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                        (1, "set_1", pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["split", "set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                iteration="split_major",
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 674, 555, 489],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                iteration="split_wise",
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 674, 555, 489],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                iteration="set_major",
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 555, 674, 489],
                index=pd.MultiIndex.from_tuples(
                    [("set_0", 0), ("set_0", 1), ("set_1", 0), ("set_1", 1)], names=["set", "split"]
                ),
            ),
        )
        assert_series_equal(
            splitter.apply(
                apply_func,
                vbt.Takeable(sr.values, index=sr.index.shift(-1)),
                b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
                obj_index=sr.index,
                iteration="set_wise",
                merge_all=True,
                merge_func="concat",
            ),
            pd.Series(
                [405, 555, 674, 489],
                index=pd.MultiIndex.from_tuples(
                    [("set_0", 0), ("set_0", 1), ("set_1", 0), ("set_1", 1)], names=["set", "split"]
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_major",
            merge_all=False,
            merge_func="concat",
        )
        assert_series_equal(
            r.iloc[0],
            pd.Series(
                [405, 674],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_series_equal(
            r.iloc[1],
            pd.Series(
                [555, 489],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_wise",
            merge_all=False,
            merge_func="concat",
        )
        assert_series_equal(
            r.iloc[0],
            pd.Series(
                [405, 674],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_series_equal(
            r.iloc[1],
            pd.Series(
                [555, 489],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="set_major",
            merge_all=False,
            merge_func="concat",
        )
        assert_series_equal(
            r.iloc[0],
            pd.Series(
                [405, 555],
                index=pd.Index([0, 1], dtype="int64", name="split"),
            ),
        )
        assert_series_equal(
            r.iloc[1],
            pd.Series(
                [674, 489],
                index=pd.Index([0, 1], dtype="int64", name="split"),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="set_wise",
            merge_all=False,
            merge_func="concat",
        )
        assert_series_equal(
            r.iloc[0],
            pd.Series(
                [405, 555],
                index=pd.Index([0, 1], dtype="int64", name="split"),
            ),
        )
        assert_series_equal(
            r.iloc[1],
            pd.Series(
                [674, 489],
                index=pd.Index([0, 1], dtype="int64", name="split"),
            ),
        )

        def apply_func(a, b=None):
            return a.sum(), b.sum()

        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_major",
            merge_all=True,
            merge_func="concat",
        )
        assert_series_equal(
            r[0],
            pd.Series(
                [195, 345, 270, 255],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            r[1],
            pd.Series(
                [210, 329, 285, 234],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_wise",
            merge_all=True,
            merge_func="concat",
        )
        assert_series_equal(
            r[0],
            pd.Series(
                [195, 345, 270, 255],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            r[1],
            pd.Series(
                [210, 329, 285, 234],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="set_major",
            merge_all=True,
            merge_func="concat",
        )
        assert_series_equal(
            r[0],
            pd.Series(
                [195, 270, 345, 255],
                index=pd.MultiIndex.from_tuples(
                    [("set_0", 0), ("set_0", 1), ("set_1", 0), ("set_1", 1)], names=["set", "split"]
                ),
            ),
        )
        assert_series_equal(
            r[1],
            pd.Series(
                [210, 285, 329, 234],
                index=pd.MultiIndex.from_tuples(
                    [("set_0", 0), ("set_0", 1), ("set_1", 0), ("set_1", 1)], names=["set", "split"]
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="set_wise",
            merge_all=True,
            merge_func="concat",
        )
        assert_series_equal(
            r[0],
            pd.Series(
                [195, 270, 345, 255],
                index=pd.MultiIndex.from_tuples(
                    [("set_0", 0), ("set_0", 1), ("set_1", 0), ("set_1", 1)], names=["set", "split"]
                ),
            ),
        )
        assert_series_equal(
            r[1],
            pd.Series(
                [210, 285, 329, 234],
                index=pd.MultiIndex.from_tuples(
                    [("set_0", 0), ("set_0", 1), ("set_1", 0), ("set_1", 1)], names=["set", "split"]
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_major",
            merge_all=False,
            merge_func="concat",
        )
        assert_series_equal(
            r[0].iloc[0],
            pd.Series(
                [195, 345],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_series_equal(
            r[0].iloc[1],
            pd.Series(
                [270, 255],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_series_equal(
            r[1].iloc[0],
            pd.Series(
                [210, 329],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_series_equal(
            r[1].iloc[1],
            pd.Series(
                [285, 234],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_wise",
            merge_all=False,
            merge_func="concat",
        )
        assert_series_equal(
            r[0].iloc[0],
            pd.Series(
                [195, 345],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_series_equal(
            r[0].iloc[1],
            pd.Series(
                [270, 255],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_series_equal(
            r[1].iloc[0],
            pd.Series(
                [210, 329],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_series_equal(
            r[1].iloc[1],
            pd.Series(
                [285, 234],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="set_major",
            merge_all=False,
            merge_func="concat",
        )
        assert_series_equal(
            r[0].iloc[0],
            pd.Series(
                [195, 270],
                index=pd.Index([0, 1], dtype="int64", name="split"),
            ),
        )
        assert_series_equal(
            r[0].iloc[1],
            pd.Series(
                [345, 255],
                index=pd.Index([0, 1], dtype="int64", name="split"),
            ),
        )
        assert_series_equal(
            r[1].iloc[0],
            pd.Series(
                [210, 285],
                index=pd.Index([0, 1], dtype="int64", name="split"),
            ),
        )
        assert_series_equal(
            r[1].iloc[1],
            pd.Series(
                [329, 234],
                index=pd.Index([0, 1], dtype="int64", name="split"),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="set_wise",
            merge_all=False,
            merge_func="concat",
            attach_bounds="target_index",
        )
        assert_series_equal(
            r[0].iloc[0],
            pd.Series(
                [195, 270],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        (1, pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                    ],
                    names=["split", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            r[0].iloc[1],
            pd.Series(
                [345, 255],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                        (1, pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["split", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            r[1].iloc[0],
            pd.Series(
                [210, 285],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        (1, pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                    ],
                    names=["split", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            r[1].iloc[1],
            pd.Series(
                [329, 234],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                        (1, pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["split", "start", "end"],
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_major",
            merge_all=True,
        )
        assert_series_equal(
            r[0],
            pd.Series(
                [195, 345, 270, 255],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        assert_series_equal(
            r[1],
            pd.Series(
                [210, 329, 285, 234],
                index=pd.MultiIndex.from_tuples(
                    [(0, "set_0"), (0, "set_1"), (1, "set_0"), (1, "set_1")], names=["split", "set"]
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_major",
            merge_all=False,
            attach_bounds="target_index",
            set_group_by=True,
        )
        assert_series_equal(
            r[0],
            pd.Series(
                [450, 410],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-26")),
                        (1, pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["split", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            r[1],
            pd.Series(
                [444, 399],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-26")),
                        (1, pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["split", "start", "end"],
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_major",
            merge_all=False,
            attach_bounds="target_index",
            split_group_by=True,
        )
        assert_series_equal(
            r[0],
            pd.Series(
                [310, 345],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-21")),
                        ("set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            r[1],
            pd.Series(
                [330, 329],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-21")),
                        ("set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["set", "start", "end"],
                ),
            ),
        )
        r = splitter.apply(
            apply_func,
            vbt.Takeable(sr.values, index=sr.index.shift(-1)),
            b=vbt.Takeable(sr.values, index=sr.index.shift(-2)),
            obj_index=sr.index,
            iteration="split_major",
            merge_all=False,
            attach_bounds="target_index",
        )
        assert_index_equal(
            r[0].index,
            pd.Index([0, 1], dtype="int64", name="split"),
        )
        assert_series_equal(
            r[0].iloc[0],
            pd.Series(
                [195, 345],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        ("set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                    ],
                    names=["set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            r[0].iloc[1],
            pd.Series(
                [270, 255],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("set_0", pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                        ("set_1", pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["set", "start", "end"],
                ),
            ),
        )
        assert_index_equal(
            r[1].index,
            pd.Index([0, 1], dtype="int64", name="split"),
        )
        assert_series_equal(
            r[1].iloc[0],
            pd.Series(
                [210, 329],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("set_0", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-16")),
                        ("set_1", pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-26")),
                    ],
                    names=["set", "start", "end"],
                ),
            ),
        )
        assert_series_equal(
            r[1].iloc[1],
            pd.Series(
                [285, 234],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("set_0", pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-21")),
                        ("set_1", pd.Timestamp("2020-01-16"), pd.Timestamp("2020-01-27")),
                    ],
                    names=["set", "start", "end"],
                ),
            ),
        )

    def test_shuffle_splits(self):
        splitter = vbt.Splitter.from_splits(
            index, [slice(0, 10), slice(10, 20), slice(20, 30)]
        ).split_set(0.5)
        new_splitter = splitter.shuffle_splits(seed=42)
        np.testing.assert_array_equal(
            new_splitter.splits_arr,
            np.array(
                [
                    [slice(20, 25, None), slice(25, 30, None)],
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(10, 15, None), slice(15, 20, None)],
                ],
                dtype=object,
            ),
        )
        new_splitter = splitter.shuffle_splits(size=1, seed=42)
        np.testing.assert_array_equal(
            new_splitter.splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                ],
                dtype=object,
            ),
        )

    def test_break_up_splits(self):
        splitter = vbt.Splitter.from_splits(index, [slice(0, 10), slice(10, 20), slice(20, 30)])
        new_splitter = splitter.split_set(0.5)
        with pytest.raises(Exception):
            new_splitter.break_up_splits(0.5)
        np.testing.assert_array_equal(
            splitter.break_up_splits(0.5).splits_arr,
            np.array(
                [
                    [slice(0, 5, None)],
                    [slice(5, 10, None)],
                    [slice(10, 15, None)],
                    [slice(15, 20, None)],
                    [slice(20, 25, None)],
                    [slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )
        assert_index_equal(
            splitter.break_up_splits(0.5).split_labels,
            pd.MultiIndex.from_tuples(
                [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], names=["split", "split_part"]
            ),
        )
        splitter = vbt.Splitter.from_splits(index, [slice(20, 30), slice(10, 20), slice(0, 10)])
        np.testing.assert_array_equal(
            splitter.break_up_splits(0.5, sort=True).splits_arr,
            np.array(
                [
                    [slice(0, 5, None)],
                    [slice(5, 10, None)],
                    [slice(10, 15, None)],
                    [slice(15, 20, None)],
                    [slice(20, 25, None)],
                    [slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )
        assert_index_equal(
            splitter.break_up_splits(0.5, sort=True).split_labels,
            pd.MultiIndex.from_tuples(
                [(2, 0), (2, 1), (1, 0), (1, 1), (0, 0), (0, 1)], names=["split", "split_part"]
            ),
        )

    def test_split_set(self):
        splitter = vbt.Splitter.from_splits(index, [slice(0, 10), slice(10, 20), slice(20, 30)])
        new_splitter = splitter.split_set(0.5)
        assert_index_equal(new_splitter.wrapper.index, splitter.wrapper.index)
        assert_index_equal(
            new_splitter.wrapper.columns,
            pd.Index(["set_0/0", "set_0/1"], dtype="object", name="set"),
        )
        assert new_splitter.wrapper.ndim == 2
        np.testing.assert_array_equal(
            new_splitter.splits_arr,
            np.array(
                [
                    [slice(0, 5, None), slice(5, 10, None)],
                    [slice(10, 15, None), slice(15, 20, None)],
                    [slice(20, 25, None), slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )
        with pytest.raises(Exception):
            splitter.split_set(0.5, new_set_labels=["a"])
        with pytest.raises(Exception):
            splitter.split_set(0.5, new_set_labels=["a", "b", "c"])
        new_splitter = splitter.split_set(0.5, new_set_labels=["a", "b"])
        assert_index_equal(
            new_splitter.wrapper.columns,
            pd.Index(["a", "b"], dtype="object", name="set"),
        )
        splitter = vbt.Splitter.from_splits(
            index, [slice(0, 10), slice(10, 20), slice(20, 30)], set_labels=["a+b"]
        )
        new_splitter = splitter.split_set(0.5)
        assert_index_equal(
            new_splitter.wrapper.columns,
            pd.Index(["a", "b"], dtype="object", name="set"),
        )
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, 30)],
            ],
        )
        with pytest.raises(Exception):
            splitter.split_set(0.5)
        new_splitter = splitter.split_set(0.5, column=0)
        assert_index_equal(new_splitter.wrapper.index, splitter.wrapper.index)
        assert_index_equal(
            new_splitter.wrapper.columns,
            pd.Index(["set_0/0", "set_0/1", "set_1"], dtype="object", name="set"),
        )
        assert new_splitter.wrapper.ndim == 2
        np.testing.assert_array_equal(
            new_splitter.splits_arr,
            np.array(
                [
                    [slice(0, 2, None), slice(2, 5, None), slice(5, 10, None)],
                    [slice(10, 12, None), slice(12, 15, None), slice(15, 20, None)],
                    [slice(20, 22, None), slice(22, 25, None), slice(25, 30, None)],
                ],
                dtype=object,
            ),
        )
        with pytest.raises(Exception):
            splitter.split_set(0.5, column=0, new_set_labels=["a"])
        with pytest.raises(Exception):
            splitter.split_set(0.5, column=0, new_set_labels=["a", "b", "c"])
        new_splitter = splitter.split_set(0.5, column=0, new_set_labels=["a", "b"])
        assert_index_equal(
            new_splitter.wrapper.columns,
            pd.Index(["a", "b", "set_1"], dtype="object", name="set"),
        )

    def test_merge_sets(self):
        splitter = vbt.Splitter.from_splits(index, [slice(0, 10), slice(10, 20), slice(20, 30)])
        with pytest.raises(Exception):
            splitter.merge_sets()
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, 30)],
            ],
        )
        new_splitter = splitter.merge_sets()
        assert_index_equal(new_splitter.wrapper.index, splitter.wrapper.index)
        assert_index_equal(
            new_splitter.wrapper.columns,
            pd.Index(["set_0+set_1"], dtype="object", name="set"),
        )
        assert new_splitter.wrapper.ndim == 1
        np.testing.assert_array_equal(
            new_splitter.splits_arr,
            np.array(
                [
                    [slice(0, 10, None)],
                    [slice(10, 20, None)],
                    [slice(20, 30, None)],
                ],
                dtype=object,
            ),
        )
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10), slice(10, 15)],
                [slice(5, 10), slice(10, 15), slice(15, 20)],
                [slice(10, 15), slice(15, 20), slice(25, 30)],
            ],
            set_labels=["set_0/0", "set_1", "set_0/1"],
        )
        new_splitter = splitter.merge_sets(columns=[0, 2])
        assert_index_equal(new_splitter.wrapper.index, splitter.wrapper.index)
        assert_index_equal(
            new_splitter.wrapper.columns,
            pd.Index(["set_0", "set_1"], dtype="object", name="set"),
        )
        assert new_splitter.wrapper.ndim == 2
        np.testing.assert_array_equal(
            new_splitter.splits_arr[0, 0].range_,
            np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14]),
        )
        np.testing.assert_array_equal(
            new_splitter.splits_arr[1, 0].range_,
            np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19]),
        )
        np.testing.assert_array_equal(
            new_splitter.splits_arr[2, 0].range_,
            np.array([10, 11, 12, 13, 14, 25, 26, 27, 28, 29]),
        )
        np.testing.assert_array_equal(
            new_splitter.splits_arr[:, [1]],
            np.array(
                [
                    [slice(5, 10)],
                    [slice(10, 15)],
                    [slice(15, 20)],
                ],
                dtype=object,
            ),
        )
        new_splitter = splitter.merge_sets(columns=[0, 2], insert_at_last=True)
        assert_index_equal(new_splitter.wrapper.index, splitter.wrapper.index)
        assert_index_equal(
            new_splitter.wrapper.columns,
            pd.Index(["set_1", "set_0"], dtype="object", name="set"),
        )
        assert new_splitter.wrapper.ndim == 2
        np.testing.assert_array_equal(
            new_splitter.splits_arr[:, [0]],
            np.array(
                [
                    [slice(5, 10)],
                    [slice(10, 15)],
                    [slice(15, 20)],
                ],
                dtype=object,
            ),
        )
        np.testing.assert_array_equal(
            new_splitter.splits_arr[0, 1].range_,
            np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14]),
        )
        np.testing.assert_array_equal(
            new_splitter.splits_arr[1, 1].range_,
            np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19]),
        )
        np.testing.assert_array_equal(
            new_splitter.splits_arr[2, 1].range_,
            np.array([10, 11, 12, 13, 14, 25, 26, 27, 28, 29]),
        )

    def test_get_range_bounds(self):
        assert vbt.Splitter.get_range_bounds(slice(None), index=index) == (0, 31)
        assert vbt.Splitter.get_range_bounds(slice(5, 10), index=index) == (5, 10)
        assert vbt.Splitter.get_range_bounds(np.array([3, 4, 5]), index=index) == (3, 6)
        with pytest.raises(Exception):
            vbt.Splitter.get_range_bounds(np.array([3, 5]), index=index)
        assert vbt.Splitter.get_range_bounds(
            np.array([3, 5]), check_constant=False, index=index
        ) == (3, 6)
        with pytest.raises(Exception):
            vbt.Splitter.get_range_bounds(slice(5, 5), index=index)
        with pytest.raises(Exception):
            vbt.Splitter.get_range_bounds(np.array([]), index=index)
        assert vbt.Splitter.get_range_bounds(slice(None), index_bounds=True, index=index) == (
            index[0],
            index[-1] + index.freq,
        )
        assert vbt.Splitter.get_range_bounds(slice(5, 10), index_bounds=True, index=index) == (
            index[5],
            index[10],
        )
        assert vbt.Splitter.get_range_bounds(
            slice(None), index_bounds=True, index=index[[0, 2]], freq=pd.Timedelta(days=1)
        ) == (index[0], index[2] + pd.Timedelta(days=1))

    def test_get_bounds_arr(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, None)],
            ],
        )
        np.testing.assert_array_equal(
            splitter.bounds_arr,
            np.array([[[0, 5], [5, 10]], [[10, 15], [15, 20]], [[20, 25], [25, 31]]]),
        )
        np.testing.assert_array_equal(
            splitter.get_bounds_arr(index_bounds=True),
            np.array(
                [
                    [
                        [1577836800000000000, 1578268800000000000],
                        [1578268800000000000, 1578700800000000000],
                    ],
                    [
                        [1578700800000000000, 1579132800000000000],
                        [1579132800000000000, 1579564800000000000],
                    ],
                    [
                        [1579564800000000000, 1579996800000000000],
                        [1579996800000000000, 1580515200000000000],
                    ],
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            splitter.get_bounds_arr(split_group_by=[0, 1, 0], check_constant=False),
            np.array([[[0, 25], [5, 31]], [[10, 15], [15, 20]]]),
        )

    def test_get_bounds(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, None)],
            ],
        )
        assert_frame_equal(
            splitter.bounds,
            pd.DataFrame(
                [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 31]],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0"),
                        (0, "set_1"),
                        (1, "set_0"),
                        (1, "set_1"),
                        (2, "set_0"),
                        (2, "set_1"),
                    ],
                    names=["split", "set"],
                ),
                columns=pd.Index(["start", "end"], dtype="object", name="bound"),
            ),
        )
        assert_frame_equal(
            splitter.index_bounds,
            pd.DataFrame(
                [
                    [1577836800000000000, 1578268800000000000],
                    [1578268800000000000, 1578700800000000000],
                    [1578700800000000000, 1579132800000000000],
                    [1579132800000000000, 1579564800000000000],
                    [1579564800000000000, 1579996800000000000],
                    [1579996800000000000, 1580515200000000000],
                ],
                dtype="datetime64[ns]",
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0"),
                        (0, "set_1"),
                        (1, "set_0"),
                        (1, "set_1"),
                        (2, "set_0"),
                        (2, "set_1"),
                    ],
                    names=["split", "set"],
                ),
                columns=pd.Index(["start", "end"], dtype="object", name="bound"),
            ),
        )
        assert_frame_equal(
            splitter.get_bounds(
                split_group_by=["a", "b", "a"],
                set_group_by=["c", "c"],
                check_constant=False,
                squeeze_one_set=False,
            ),
            pd.DataFrame(
                [[0, 31], [10, 20]],
                index=pd.MultiIndex.from_tuples(
                    [("a", "c"), ("b", "c")], names=["split_group", "set_group"]
                ),
                columns=pd.Index(["start", "end"], dtype="object", name="bound"),
            ),
        )
        assert_frame_equal(
            splitter.get_bounds(
                split_group_by=["a", "b", "a"], set_group_by=["c", "c"], check_constant=False
            ),
            pd.DataFrame(
                [[0, 31], [10, 20]],
                index=pd.Index(["a", "b"], dtype="object", name="split_group"),
                columns=pd.Index(["start", "end"], dtype="object", name="bound"),
            ),
        )

    def test_get_duration(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, None)],
            ],
        )
        assert_series_equal(
            splitter.duration,
            pd.Series(
                [5, 5, 5, 5, 5, 6],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0"),
                        (0, "set_1"),
                        (1, "set_0"),
                        (1, "set_1"),
                        (2, "set_0"),
                        (2, "set_1"),
                    ],
                    names=["split", "set"],
                ),
                name="duration",
            ),
        )
        assert_series_equal(
            splitter.index_duration,
            pd.Series(
                [
                    432000000000000,
                    432000000000000,
                    432000000000000,
                    432000000000000,
                    432000000000000,
                    518400000000000,
                ],
                dtype="timedelta64[ns]",
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0"),
                        (0, "set_1"),
                        (1, "set_0"),
                        (1, "set_1"),
                        (2, "set_0"),
                        (2, "set_1"),
                    ],
                    names=["split", "set"],
                ),
                name="duration",
            ),
        )

    def test_get_range_mask(self):
        mask = np.full(len(index), True)
        np.testing.assert_array_equal(vbt.Splitter.get_range_mask(slice(None), index=index), mask)
        mask = np.full(len(index), False)
        np.testing.assert_array_equal(vbt.Splitter.get_range_mask(slice(0, 0), index=index), mask)
        mask = np.full(len(index), False)
        mask[5:10] = True
        np.testing.assert_array_equal(vbt.Splitter.get_range_mask(slice(5, 10), index=index), mask)
        mask = np.full(len(index), False)
        mask[[2, 4, 6]] = True
        np.testing.assert_array_equal(vbt.Splitter.get_range_mask([2, 4, 6], index=index), mask)
        np.testing.assert_array_equal(vbt.Splitter.get_range_mask(mask, index=index), mask)

    def test_get_iter_split_masks(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, None)],
            ],
        )
        assert_frame_equal(
            list(splitter.iter_split_masks)[0],
            pd.DataFrame(
                [
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ],
                index=index,
                columns=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_frame_equal(
            list(splitter.iter_split_masks)[1],
            pd.DataFrame(
                [
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ],
                index=index,
                columns=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_frame_equal(
            list(splitter.iter_split_masks)[2],
            pd.DataFrame(
                [
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                ],
                index=index,
                columns=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
            ),
        )
        assert_frame_equal(
            list(splitter.get_iter_split_masks(split_group_by=[0, 1, 0], set_group_by=[0, 0]))[0],
            pd.DataFrame(
                [
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                ],
                index=index,
                columns=pd.Index([0], dtype="int64", name="set_group"),
            ),
        )
        assert_frame_equal(
            list(splitter.get_iter_split_masks(split_group_by=[0, 1, 0], set_group_by=[0, 0]))[1],
            pd.DataFrame(
                [
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [True],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                    [False],
                ],
                index=index,
                columns=pd.Index([0], dtype="int64", name="set_group"),
            ),
        )

    def test_get_iter_set_masks(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, None)],
            ],
        )
        assert_frame_equal(
            list(splitter.iter_set_masks)[0],
            pd.DataFrame(
                [
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=index,
                columns=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
            ),
        )
        assert_frame_equal(
            list(splitter.iter_set_masks)[1],
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True],
                ],
                index=index,
                columns=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
            ),
        )
        assert_frame_equal(
            list(splitter.get_iter_set_masks(split_group_by=[0, 1, 0], set_group_by=[0, 0]))[0],
            pd.DataFrame(
                [
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="split_group"),
            ),
        )

    def test_get_mask_arr(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, None)],
            ],
        )
        np.testing.assert_array_equal(
            splitter.mask_arr,
            np.array(
                [
                    [
                        [
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ],
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ],
                    ],
                    [
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ],
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ],
                    ],
                    [
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ],
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                        ],
                    ],
                ]
            ),
        )
        np.testing.assert_array_equal(
            splitter.get_mask_arr(split_group_by=[0, 1, 0], set_group_by=[0, 0]),
            np.array(
                [
                    [
                        [
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                        ]
                    ],
                    [
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ]
                    ],
                ]
            ),
        )

    def test_get_mask(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(0, 5), slice(5, 10)],
                [slice(10, 15), slice(15, 20)],
                [slice(20, 25), slice(25, None)],
            ],
        )
        assert_frame_equal(
            splitter.mask,
            pd.DataFrame(
                [
                    [True, False, False, False, False, False],
                    [True, False, False, False, False, False],
                    [True, False, False, False, False, False],
                    [True, False, False, False, False, False],
                    [True, False, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, False, True],
                    [False, False, False, False, False, True],
                    [False, False, False, False, False, True],
                    [False, False, False, False, False, True],
                    [False, False, False, False, False, True],
                    [False, False, False, False, False, True],
                ],
                index=index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0"),
                        (0, "set_1"),
                        (1, "set_0"),
                        (1, "set_1"),
                        (2, "set_0"),
                        (2, "set_1"),
                    ],
                    names=["split", "set"],
                ),
            ),
        )
        assert_frame_equal(
            splitter.get_mask(split_group_by=[0, 1, 0], set_group_by=[0, 0]),
            pd.DataFrame(
                [
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, True],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="split_group"),
            ),
        )

    def test_get_split_coverage(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 15), slice(10, 20)],
                [slice(10, 20), slice(15, 25)],
                [slice(15, 25), slice(20, None)],
            ],
        )
        assert_series_equal(
            splitter.get_split_coverage(normalize=False, overlapping=False),
            pd.Series(
                [15, 15, 16],
                index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
                name="split_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_split_coverage(normalize=True, relative=False, overlapping=False),
            pd.Series(
                [0.4838709677419355, 0.4838709677419355, 0.5161290322580645],
                index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
                name="split_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_split_coverage(normalize=True, relative=True, overlapping=False),
            pd.Series(
                [0.5769230769230769, 0.5769230769230769, 0.6153846153846154],
                index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
                name="split_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_split_coverage(normalize=False, overlapping=True),
            pd.Series(
                [5, 5, 5],
                index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
                name="split_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_split_coverage(normalize=True, overlapping=True),
            pd.Series(
                [0.3333333333333333, 0.3333333333333333, 0.3125],
                index=pd.RangeIndex(start=0, stop=3, step=1, name="split"),
                name="split_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_split_coverage(split_group_by=[0, 1, 0], set_group_by=[0, 0]),
            pd.Series(
                [0.8387096774193549, 0.4838709677419355],
                index=pd.Index([0, 1], dtype="int64", name="split_group"),
                name="split_coverage",
            ),
        )

    def test_get_set_coverage(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 15), slice(10, 20)],
                [slice(10, 20), slice(15, 25)],
                [slice(15, 25), slice(20, None)],
            ],
        )
        assert_series_equal(
            splitter.get_set_coverage(normalize=False, overlapping=False),
            pd.Series(
                [20, 21],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
                name="set_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_set_coverage(normalize=True, relative=False, overlapping=False),
            pd.Series(
                [0.6451612903225806, 0.6774193548387096],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
                name="set_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_set_coverage(normalize=True, relative=True, overlapping=False),
            pd.Series(
                [0.7692307692307693, 0.8076923076923077],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
                name="set_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_set_coverage(normalize=False, overlapping=True),
            pd.Series(
                [10, 10],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
                name="set_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_set_coverage(normalize=True, overlapping=True),
            pd.Series(
                [0.5, 0.47619047619047616],
                index=pd.Index(["set_0", "set_1"], dtype="object", name="set"),
                name="set_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_set_coverage(
                split_group_by=[0, 1, 0], set_group_by=[0, 0], squeeze_one_set=False
            ),
            pd.Series(
                [0.8387096774193549],
                index=pd.Index([0], dtype="int64", name="set_group"),
                name="set_coverage",
            ),
        )
        splitter.get_set_coverage(
            split_group_by=[0, 1, 0], set_group_by=[0, 0]
        ) == 0.8387096774193549

    def test_get_range_coverage(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 5), slice(5, 6)],
                [slice(6, 8), slice(8, 11)],
                [slice(11, 15), slice(15, 20)],
                [slice(20, 26), slice(26, None)],
            ],
            split_range_kwargs=dict(allow_zero_len=True),
        )
        assert_series_equal(
            splitter.get_range_coverage(normalize=False),
            pd.Series(
                [0, 1, 2, 3, 4, 5, 6, 5],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0"),
                        (0, "set_1"),
                        (1, "set_0"),
                        (1, "set_1"),
                        (2, "set_0"),
                        (2, "set_1"),
                        (3, "set_0"),
                        (3, "set_1"),
                    ],
                    names=["split", "set"],
                ),
                name="range_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_range_coverage(normalize=True, relative=False),
            pd.Series(
                [
                    0.0,
                    0.03225806451612903,
                    0.06451612903225806,
                    0.0967741935483871,
                    0.12903225806451613,
                    0.16129032258064516,
                    0.1935483870967742,
                    0.16129032258064516,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0"),
                        (0, "set_1"),
                        (1, "set_0"),
                        (1, "set_1"),
                        (2, "set_0"),
                        (2, "set_1"),
                        (3, "set_0"),
                        (3, "set_1"),
                    ],
                    names=["split", "set"],
                ),
                name="range_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_range_coverage(normalize=True, relative=True),
            pd.Series(
                [
                    0.0,
                    1.0,
                    0.4,
                    0.6,
                    0.4444444444444444,
                    0.5555555555555556,
                    0.5454545454545454,
                    0.45454545454545453,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, "set_0"),
                        (0, "set_1"),
                        (1, "set_0"),
                        (1, "set_1"),
                        (2, "set_0"),
                        (2, "set_1"),
                        (3, "set_0"),
                        (3, "set_1"),
                    ],
                    names=["split", "set"],
                ),
                name="range_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_range_coverage(
                split_group_by=[0, 1, 0, 2], set_group_by=[0, 0], squeeze_one_set=False
            ),
            pd.Series(
                [0.3225806451612903, 0.16129032258064516, 0.3548387096774194],
                index=pd.MultiIndex.from_tuples(
                    [
                        (0, 0),
                        (1, 0),
                        (2, 0),
                    ],
                    names=["split_group", "set_group"],
                ),
                name="range_coverage",
            ),
        )
        assert_series_equal(
            splitter.get_range_coverage(split_group_by=[0, 1, 0, 2], set_group_by=[0, 0]),
            pd.Series(
                [0.3225806451612903, 0.16129032258064516, 0.3548387096774194],
                index=pd.Index([0, 1, 2], dtype="int64", name="split_group"),
                name="range_coverage",
            ),
        )

    def test_get_coverage(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 15), slice(10, 20)],
                [slice(10, 20), slice(15, 25)],
                [slice(15, 25), slice(20, None)],
            ],
        )
        assert splitter.get_coverage(normalize=False, overlapping=False) == 26
        assert splitter.get_coverage(normalize=True, overlapping=False) == 0.8387096774193549
        assert splitter.get_coverage(normalize=False, overlapping=True) == 15
        assert splitter.get_coverage(normalize=True, overlapping=True) == 0.5769230769230769

    def test_get_overlap_matrix(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 15), slice(10, 20)],
                [slice(10, 20), slice(15, 25)],
                [slice(15, 25), slice(20, None)],
            ],
        )
        np.testing.assert_array_equal(
            splitter.get_overlap_matrix(by="split", normalize=False).values,
            np.array([[15, 10, 5], [10, 15, 10], [5, 10, 16]]),
        )
        np.testing.assert_array_equal(
            splitter.get_overlap_matrix(by="split").values,
            np.array(
                [
                    [1.0, 0.5, 0.19230769230769232],
                    [0.5, 1.0, 0.47619047619047616],
                    [0.19230769230769232, 0.47619047619047616, 1.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            splitter.get_overlap_matrix(by="set", normalize=False).values,
            np.array([[20, 15], [15, 21]]),
        )
        np.testing.assert_array_equal(
            splitter.get_overlap_matrix(by="set").values,
            np.array([[1.0, 0.5769230769230769], [0.5769230769230769, 1.0]]),
        )
        np.testing.assert_array_equal(
            splitter.get_overlap_matrix(by="range", normalize=False).values,
            np.array(
                [
                    [10, 5, 5, 0, 0, 0],
                    [5, 10, 10, 5, 5, 0],
                    [5, 10, 10, 5, 5, 0],
                    [0, 5, 5, 10, 10, 5],
                    [0, 5, 5, 10, 10, 5],
                    [0, 0, 0, 5, 5, 11],
                ]
            ),
        )
        np.testing.assert_array_equal(
            splitter.get_overlap_matrix(by="range").values,
            np.array(
                [
                    [1.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0],
                    [0.3333333333333333, 1.0, 1.0, 0.3333333333333333, 0.3333333333333333, 0.0],
                    [0.3333333333333333, 1.0, 1.0, 0.3333333333333333, 0.3333333333333333, 0.0],
                    [0.0, 0.3333333333333333, 0.3333333333333333, 1.0, 1.0, 0.3125],
                    [0.0, 0.3333333333333333, 0.3333333333333333, 1.0, 1.0, 0.3125],
                    [0.0, 0.0, 0.0, 0.3125, 0.3125, 1.0],
                ]
            ),
        )

    def test_stats(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 5), slice(5, 6)],
                [slice(6, 8), slice(8, 11)],
                [slice(11, 15), slice(15, 20)],
                [slice(20, 26), slice(26, None)],
            ],
            split_range_kwargs=dict(allow_zero_len=True),
        )
        assert_series_equal(
            splitter.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-31 00:00:00"),
                    31,
                    4,
                    2,
                    83.87096774193549,
                    38.70967741935484,
                    45.16129032258064,
                    34.74747474747475,
                    65.25252525252525,
                    0.0,
                    0.0,
                    0.0,
                ],
                index=pd.Index(
                    [
                        "Index Start",
                        "Index End",
                        "Index Length",
                        "Splits",
                        "Sets",
                        "Coverage [%]",
                        "Coverage [%]: set_0",
                        "Coverage [%]: set_1",
                        "Mean Rel Coverage [%]: set_0",
                        "Mean Rel Coverage [%]: set_1",
                        "Overlap Coverage [%]",
                        "Overlap Coverage [%]: set_0",
                        "Overlap Coverage [%]: set_1",
                    ],
                    dtype="object",
                ),
                name="agg_stats",
            ),
        )
        assert_series_equal(
            splitter.stats(settings=dict(normalize=False)),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-31 00:00:00"),
                    31,
                    4,
                    2,
                    26,
                    12,
                    14,
                    0,
                    0,
                    0,
                ],
                index=pd.Index(
                    [
                        "Index Start",
                        "Index End",
                        "Index Length",
                        "Splits",
                        "Sets",
                        "Coverage",
                        "Coverage: set_0",
                        "Coverage: set_1",
                        "Overlap Coverage",
                        "Overlap Coverage: set_0",
                        "Overlap Coverage: set_1",
                    ],
                    dtype="object",
                ),
                name="agg_stats",
            ),
        )
        assert_series_equal(
            splitter.stats(settings=dict(split_group_by=[0, 1, 0, 2], set_group_by=[0, 0])),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-31 00:00:00"),
                    31,
                    3,
                    1,
                    83.87096774193549,
                    0.0,
                ],
                index=pd.Index(
                    [
                        "Index Start",
                        "Index End",
                        "Index Length",
                        "Splits",
                        "Sets",
                        "Coverage [%]",
                        "Overlap Coverage [%]",
                    ],
                    dtype="object",
                ),
                name="agg_stats",
            ),
        )

    def test_plots(self):
        splitter = vbt.Splitter.from_splits(
            index,
            [
                [slice(5, 5), slice(5, 6)],
                [slice(6, 8), slice(8, 11)],
                [slice(11, 15), slice(15, 20)],
                [slice(20, 26), slice(26, None)],
            ],
            split_range_kwargs=dict(allow_zero_len=True),
        )
        splitter.plots()
        (splitter.plots(settings=dict(split_group_by=[0, 1, 0, 2], set_group_by=[0, 0])),)


class TestSplitterCV:
    def test_class(self):
        skl_splitter = vbt.SplitterCV(
            splits=[
                [slice(5, 5), slice(5, 6)],
                [slice(6, 8), slice(8, 11)],
                [slice(11, 15), slice(15, 20)],
                [slice(20, 26), slice(26, None)],
            ],
            split_range_kwargs=dict(allow_zero_len=True),
        )
        sr = pd.Series(np.arange(len(index)), index=index)
        splits = list(skl_splitter.split(sr))
        np.testing.assert_array_equal(splits[0][0], np.array([], dtype=np.int64))
        np.testing.assert_array_equal(splits[0][1], np.array([5], dtype=np.int64))
        np.testing.assert_array_equal(splits[1][0], np.array([6, 7], dtype=np.int64))
        np.testing.assert_array_equal(splits[1][1], np.array([8, 9, 10], dtype=np.int64))
        np.testing.assert_array_equal(splits[2][0], np.array([11, 12, 13, 14], dtype=np.int64))
        np.testing.assert_array_equal(splits[2][1], np.array([15, 16, 17, 18, 19], dtype=np.int64))
        np.testing.assert_array_equal(
            splits[3][0], np.array([20, 21, 22, 23, 24, 25], dtype=np.int64)
        )
        np.testing.assert_array_equal(splits[3][1], np.array([26, 27, 28, 29, 30], dtype=np.int64))


# ############# decorators ############# #


class TestDecorators:
    def test_split(self):
        def f(index, a, *my_args, b=None, **my_kwargs):
            return (
                len(index)
                + len(a)
                + sum(map(len, my_args))
                + len(b)
                + sum(map(len, my_kwargs.values()))
            )

        sr = pd.Series(np.arange(len(index)), index=index)
        splitter = vbt.Splitter.from_single(index, vbt.RelRange(length=5))
        split_f = vbt.split(f, splitter=splitter)
        assert split_f(index, sr, sr, b=sr, c=sr) == 155
        with pytest.raises(Exception):
            split_f = vbt.split(
                f, splitter="from_single", splitter_kwargs=dict(split=vbt.RelRange(length=5))
            )
            split_f(index, sr, sr, b=sr, c=sr)
        split_f = vbt.split(
            f,
            splitter="from_single",
            splitter_kwargs=dict(split=vbt.RelRange(length=5)),
            index=index,
        )
        assert split_f(index, sr, sr, b=sr, c=sr) == 155
        split_f = vbt.split(
            f,
            splitter="from_single",
            splitter_kwargs=dict(split=vbt.RelRange(length=5)),
            index_from="index",
        )
        assert split_f(index, sr, sr, b=sr, c=sr) == 155
        split_f = vbt.split(
            f,
            splitter="from_single",
            splitter_kwargs=dict(split=vbt.RelRange(length=5)),
            index_from="a",
        )
        assert split_f(index, sr, sr, b=sr, c=sr) == 155
        split_f = vbt.split(
            f,
            splitter="from_single",
            splitter_kwargs=dict(split=vbt.RelRange(length=5)),
            index_from="my_args_0",
        )
        assert split_f(index, sr, sr, b=sr, c=sr) == 155
        split_f = vbt.split(
            f,
            splitter="from_single",
            splitter_kwargs=dict(split=vbt.RelRange(length=5)),
            index_from=1,
        )
        assert split_f(index, sr, sr, b=sr, c=sr) == 155
        split_f = vbt.split(
            f,
            splitter="from_single",
            splitter_kwargs=dict(split=vbt.RelRange(length=5)),
            takeable_args=["index"],
        )
        assert split_f(index, sr, sr, b=sr, c=sr) == 129
        split_f = vbt.split(
            f, splitter=splitter, takeable_args=["index", "a", "my_args_0", "b", "c"]
        )
        assert split_f(index, sr, sr, b=sr, c=sr) == 25

        split_f1 = vbt.split(
            f,
            splitter="from_single",
            splitter_kwargs=dict(split=vbt.RelRange(length=5)),
            takeable_args=["index"],
        )
        split_f2 = vbt.split(
            f,
            splitter_kwargs=dict(split=vbt.RelRange(length=5)),
            takeable_args=["index"],
        )
        split_f3 = vbt.split(
            f,
            split=vbt.RelRange(length=5),
            takeable_args=["index"],
        )
        assert split_f1(index, sr, sr, b=sr, c=sr) == split_f2(index, sr, sr, b=sr, c=sr)
        assert split_f1(index, sr, sr, b=sr, c=sr) == split_f3(index, sr, sr, b=sr, c=sr)
        split_f1 = vbt.split(
            f,
            splitter="from_n_rolling",
            splitter_kwargs=dict(n=2, split=vbt.RelRange(length=5)),
            takeable_args=["index"],
        )
        split_f2 = vbt.split(
            f,
            splitter_kwargs=dict(n=2, split=vbt.RelRange(length=5)),
            takeable_args=["index"],
        )
        split_f3 = vbt.split(
            f,
            n=2,
            split=vbt.RelRange(length=5),
            takeable_args=["index"],
        )
        assert_series_equal(
            split_f1(index, sr, sr, b=sr, c=sr), split_f2(index, sr, sr, b=sr, c=sr)
        )
        assert_series_equal(
            split_f1(index, sr, sr, b=sr, c=sr), split_f3(index, sr, sr, b=sr, c=sr)
        )
        split_f1 = vbt.split(
            f,
            splitter="from_n_rolling",
            splitter_kwargs=dict(n=2, split=vbt.RelRange(length=5)),
            apply_kwargs=dict(split_group_by=True, set_group_by=True),
            takeable_args=["index"],
        )
        split_f2 = vbt.split(
            f,
            splitter_kwargs=dict(n=2, split=vbt.RelRange(length=5)),
            apply_kwargs=dict(split_group_by=True, set_group_by=True),
            takeable_args=["index"],
        )
        split_f3 = vbt.split(
            f,
            n=2,
            split=vbt.RelRange(length=5),
            split_group_by=True,
            set_group_by=True,
            takeable_args=["index"],
        )
        assert split_f1(index, sr, sr, b=sr, c=sr) == split_f2(index, sr, sr, b=sr, c=sr)
        assert split_f1(index, sr, sr, b=sr, c=sr) == split_f3(index, sr, sr, b=sr, c=sr)

    def test_cv_split(self):
        def f(sr, i):
            return sr[i]

        sr = pd.Series(np.arange(len(index)), index=index)
        splitter = vbt.Splitter.from_ranges(index, start=[0, 10], end=[10, 20], split=0.5)
        cv_split_f = vbt.cv_split(
            f,
            splitter=splitter,
            takeable_args=["sr"],
            selection=vbt.RepFunc(lambda grid_results: np.argmin(grid_results)),
        )
        assert cv_split_f(sr, vbt.Param([0, 1, 2])).values.tolist() == [0, 5, 10, 15]
        assert cv_split_f(
            sr,
            vbt.Param([0, 1, 2]),
            _selection=vbt.RepFunc(lambda grid_results: np.argmax(grid_results)),
        ).values.tolist() == [2, 7, 12, 17]
        x, y = cv_split_f(sr, vbt.Param([0, 1, 2]), _return_grid=True)
        assert x.values.tolist() == [[0, 1, 2], [0, 1, 2], [10, 11, 12], [10, 11, 12]]
        assert y.values.tolist() == [0, 5, 10, 15]
        x, y = cv_split_f(sr, vbt.Param([0, 1, 2]), _return_grid="all")
        assert x.values.tolist() == [[0, 1, 2], [5, 6, 7], [10, 11, 12], [15, 16, 17]]
        assert y.values.tolist() == [0, 5, 10, 15]
        assert cv_split_f(
            sr,
            vbt.Param([0, 1, 2]),
            _selection=vbt.RepFunc(lambda grid_results: np.argmin(grid_results)),
        ).values.tolist() == [0, 5, 10, 15]

        def f2(sr, split_idx, set_idx, i):
            return sr[split_idx + set_idx + i]

        cv_split_f2 = vbt.cv_split(
            f2,
            splitter=splitter,
            takeable_args=["sr"],
            selection=vbt.RepFunc(lambda grid_results: np.argmin(grid_results)),
        )
        assert cv_split_f2(
            sr, vbt.Rep("split_idx"), vbt.Rep("set_idx"), vbt.Param([0, 1, 2])
        ).values.tolist() == [
            0,
            6,
            11,
            17,
        ]
