import os

import pytest
import vectorbtpro as vbt
from numba import njit
from vectorbtpro._dtypes import *
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.enums import range_dt

from tests.utils import *

seed = 42

day_dt = np.timedelta64(86400000000000)

mask = pd.DataFrame(
    [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, False, False],
        [False, True, False],
    ],
    index=pd.date_range("2020", periods=5),
    columns=["a", "b", "c"],
)

ts = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0], index=mask.index)

price = pd.DataFrame(
    {
        "open": [10, 11, 12, 11, 10],
        "high": [11, 12, 13, 12, 11],
        "low": [9, 10, 11, 10, 9],
        "close": [11, 12, 11, 10, 9],
    }
)

group_by = pd.Index(["g1", "g1", "g2"])


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


# ############# accessors ############# #


class TestAccessors:
    def test_indexing(self):
        assert mask.vbt.signals["a"].total() == mask["a"].vbt.signals.total()

    def test_freq(self):
        assert mask.vbt.signals.wrapper.freq == day_dt
        assert mask["a"].vbt.signals.wrapper.freq == day_dt
        assert mask.vbt.signals(freq="2D").wrapper.freq == day_dt * 2
        assert mask["a"].vbt.signals(freq="2D").wrapper.freq == day_dt * 2
        assert pd.Series([False, True]).vbt.signals.wrapper.freq is None
        assert pd.Series([False, True]).vbt.signals(freq="3D").wrapper.freq == day_dt * 3
        assert (
            pd.Series([False, True]).vbt.signals(freq=np.timedelta64(4, "D")).wrapper.freq
            == day_dt * 4
        )

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_fshift(self, test_n):
        assert_series_equal(
            mask["a"].vbt.signals.fshift(test_n), mask["a"].shift(test_n, fill_value=False)
        )
        np.testing.assert_array_equal(
            mask["a"].vbt.signals.fshift(test_n).values,
            generic_nb.fshift_1d_nb(mask["a"].values, test_n, fill_value=False),
        )
        assert_frame_equal(mask.vbt.signals.fshift(test_n), mask.shift(test_n, fill_value=False))

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_bshift(self, test_n):
        assert_series_equal(
            mask["a"].vbt.signals.bshift(test_n), mask["a"].shift(-test_n, fill_value=False)
        )
        np.testing.assert_array_equal(
            mask["a"].vbt.signals.bshift(test_n).values,
            generic_nb.bshift_1d_nb(mask["a"].values, test_n, fill_value=False),
        )
        assert_frame_equal(mask.vbt.signals.bshift(test_n), mask.shift(-test_n, fill_value=False))

    def test_empty(self):
        assert_series_equal(
            pd.Series.vbt.signals.empty(5, index=np.arange(10, 15), name="a"),
            pd.Series(np.full(5, False), index=np.arange(10, 15), name="a"),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.empty(
                (5, 3), index=np.arange(10, 15), columns=["a", "b", "c"]
            ),
            pd.DataFrame(np.full((5, 3), False), index=np.arange(10, 15), columns=["a", "b", "c"]),
        )
        assert_series_equal(
            pd.Series.vbt.signals.empty_like(mask["a"]),
            pd.Series(np.full(mask["a"].shape, False), index=mask["a"].index, name="a"),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.empty_like(mask),
            pd.DataFrame(np.full(mask.shape, False), index=mask.index, columns=mask.columns),
        )

    def test_generate(self):
        @njit
        def place_func_nb(c, n):
            c.out[-n] = True
            return len(c.out) - 1

        assert_series_equal(
            pd.Series.vbt.signals.generate(
                5, place_func_nb, 1, wrap_kwargs=dict(index=mask["a"].index, columns=["a"])
            ),
            pd.Series(
                np.array([False, False, False, False, True]), index=mask["a"].index, name="a"
            ),
        )
        with pytest.raises(Exception):
            pd.Series.vbt.signals.generate((5, 2), place_func_nb, 1)
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate(
                (5, 3),
                place_func_nb,
                1,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [True, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate((5, 3), place_func_nb, 1, jitted=dict(parallel=True)),
            pd.DataFrame.vbt.signals.generate(
                (5, 3), place_func_nb, 1, jitted=dict(parallel=False)
            ),
        )
        chunked = dict(arg_take_spec=dict(args=vbt.ArgsTaker(None)))
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate((5, 3), place_func_nb, 1, chunked=chunked),
            pd.DataFrame.vbt.signals.generate((5, 3), place_func_nb, 1, chunked=False),
        )

        @njit
        def place_func2_nb(c, temp):
            i = temp[c.from_i, c.col]
            c.out[i] = True
            return i

        assert_series_equal(
            pd.Series.vbt.signals.generate(
                5,
                place_func2_nb,
                vbt.Rep("temp"),
                broadcast_named_args=dict(temp=0),
                wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
            ),
            pd.Series(
                np.array([True, False, False, False, False]), index=mask["a"].index, name="a"
            ),
        )

    def test_generate_both(self):
        @njit
        def entry_place_func_nb(c):
            c.out[0] = True
            return 0

        @njit
        def exit_place_func_nb(c):
            c.out[0] = True
            return 0

        en, ex = pd.Series.vbt.signals.generate_both(
            5,
            entry_place_func_nb=entry_place_func_nb,
            exit_place_func_nb=exit_place_func_nb,
            wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
        )
        assert_series_equal(
            en,
            pd.Series(np.array([True, False, True, False, True]), index=mask["a"].index, name="a"),
        )
        assert_series_equal(
            ex,
            pd.Series(np.array([False, True, False, True, False]), index=mask["a"].index, name="a"),
        )
        en, ex = pd.DataFrame.vbt.signals.generate_both(
            (5, 3),
            entry_place_func_nb=entry_place_func_nb,
            exit_place_func_nb=exit_place_func_nb,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [False, False, False],
                        [True, True, True],
                        [False, False, False],
                        [True, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, True, True],
                        [False, False, False],
                        [True, True, True],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        en, ex = pd.Series.vbt.signals.generate_both(
            (5,),
            entry_place_func_nb=entry_place_func_nb,
            exit_place_func_nb=exit_place_func_nb,
            entry_wait=1,
            exit_wait=0,
            wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
        )
        assert_series_equal(
            en,
            pd.Series(np.array([True, True, True, True, True]), index=mask["a"].index, name="a"),
        )
        assert_series_equal(
            ex,
            pd.Series(np.array([True, True, True, True, True]), index=mask["a"].index, name="a"),
        )
        en, ex = pd.Series.vbt.signals.generate_both(
            (5,),
            entry_place_func_nb=entry_place_func_nb,
            exit_place_func_nb=exit_place_func_nb,
            entry_wait=0,
            exit_wait=1,
            wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
        )
        assert_series_equal(
            en,
            pd.Series(np.array([True, True, True, True, True]), index=mask["a"].index, name="a"),
        )
        assert_series_equal(
            ex,
            pd.Series(np.array([False, True, True, True, True]), index=mask["a"].index, name="a"),
        )

        @njit
        def entry_place_func2_nb(c):
            c.out[0] = True
            if c.from_i + 1 < c.to_i:
                c.out[1] = True
                return 1
            return 0

        @njit
        def exit_place_func2_nb(c):
            c.out[0] = True
            if c.from_i + 1 < c.to_i:
                c.out[1] = True
                return 1
            return 0

        en, ex = pd.DataFrame.vbt.signals.generate_both(
            (5, 3),
            entry_place_func_nb=entry_place_func2_nb,
            exit_place_func_nb=exit_place_func2_nb,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [True, True, True],
                        [False, False, False],
                        [False, False, False],
                        [True, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [True, True, True],
                        [True, True, True],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_both(
                (5, 3),
                entry_place_func_nb=entry_place_func2_nb,
                exit_place_func_nb=exit_place_func2_nb,
                chunked=True,
            )[0],
            pd.DataFrame.vbt.signals.generate_both(
                (5, 3),
                entry_place_func_nb=entry_place_func2_nb,
                exit_place_func_nb=exit_place_func2_nb,
                chunked=False,
            )[0],
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_both(
                (5, 3),
                entry_place_func_nb=entry_place_func2_nb,
                exit_place_func_nb=exit_place_func2_nb,
                chunked=True,
            )[1],
            pd.DataFrame.vbt.signals.generate_both(
                (5, 3),
                entry_place_func_nb=entry_place_func2_nb,
                exit_place_func_nb=exit_place_func2_nb,
                chunked=False,
            )[1],
        )

        @njit
        def entry_place_func3_nb(c, temp):
            i = temp[c.from_i, c.col]
            c.out[i] = True
            return i

        @njit
        def exit_place_func3_nb(c, temp):
            i = temp[c.from_i, c.col]
            c.out[i] = True
            return i

        en, ex = pd.Series.vbt.signals.generate_both(
            5,
            entry_place_func_nb=entry_place_func3_nb,
            entry_place_args=(vbt.Rep("temp"),),
            exit_place_func_nb=exit_place_func3_nb,
            exit_place_args=(vbt.Rep("temp"),),
            broadcast_named_args=dict(temp=0),
            wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
        )
        assert_series_equal(
            en,
            pd.Series(np.array([True, False, True, False, True]), index=mask["a"].index, name="a"),
        )
        assert_series_equal(
            ex,
            pd.Series(np.array([False, True, False, True, False]), index=mask["a"].index, name="a"),
        )

    def test_generate_exits(self):
        @njit
        def place_func_nb(c):
            c.out[0] = True
            return 0

        assert_series_equal(
            mask["a"].vbt.signals.generate_exits(place_func_nb),
            pd.Series(np.array([False, True, False, False, True]), index=mask["a"].index, name="a"),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_exits(place_func_nb),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_exits(place_func_nb, wait=0),
            pd.DataFrame(
                np.array(
                    [
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                        [False, True, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )

        @njit
        def place_func2_nb(c):
            c.out[:] = True
            return len(c.out) - 1

        assert_frame_equal(
            mask.vbt.signals.generate_exits(place_func2_nb, until_next=False),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [True, True, False],
                        [True, True, True],
                        [True, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )

        mask2 = pd.Series([True, True, True, True, True], index=mask.index)
        assert_series_equal(
            mask2.vbt.signals.generate_exits(place_func_nb, until_next=False, skip_until_exit=True),
            pd.Series(np.array([False, True, False, True, False]), index=mask.index),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_exits(place_func_nb, jitted=dict(parallel=True)),
            mask.vbt.signals.generate_exits(place_func_nb, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_exits(place_func_nb, chunked=True),
            mask.vbt.signals.generate_exits(place_func_nb, chunked=False),
        )

        @njit
        def place_func3_nb(c, temp):
            i = temp[c.from_i, c.col]
            c.out[i] = True
            return i

        assert_series_equal(
            mask["a"].vbt.signals.generate_exits(
                place_func3_nb,
                vbt.RepEval("temp"),
                broadcast_named_args=dict(temp=0),
            ),
            pd.Series(np.array([False, True, False, False, True]), index=mask["a"].index, name="a"),
        )

    def test_clean(self):
        entries = pd.DataFrame(
            [
                [True, False, True],
                [True, False, False],
                [True, True, True],
                [False, True, False],
                [False, True, True],
            ],
            index=mask.index,
            columns=mask.columns,
        )
        exits = pd.Series([True, False, True, False, True], index=mask.index)
        assert_frame_equal(
            entries.vbt.signals.clean(),
            pd.DataFrame(
                np.array(
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, True, True],
                        [False, False, False],
                        [False, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.clean(entries),
            pd.DataFrame(
                np.array(
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, True, True],
                        [False, False, False],
                        [False, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits)[0],
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, False, False],
                        [False, True, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits)[1],
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits, force_first=False)[0],
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, False, False],
                        [False, True, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits, force_first=False)[1],
            pd.DataFrame(
                np.array(
                    [
                        [False, True, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits, reverse_order=True)[0],
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, True, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits, reverse_order=True)[1],
            pd.DataFrame(
                np.array(
                    [
                        [False, True, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.clean(entries, exits)[0],
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, False, False],
                        [False, True, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.clean(entries, exits)[1],
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        with pytest.raises(Exception):
            pd.Series.vbt.signals.clean(entries, entries, entries)
        assert_frame_equal(
            entries.vbt.signals.clean(jitted=dict(parallel=True)),
            entries.vbt.signals.clean(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits, jitted=dict(parallel=True))[0],
            entries.vbt.signals.clean(exits, jitted=dict(parallel=False))[0],
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits, jitted=dict(parallel=True))[1],
            entries.vbt.signals.clean(exits, jitted=dict(parallel=False))[1],
        )
        assert_frame_equal(
            entries.vbt.signals.clean(chunked=True), entries.vbt.signals.clean(chunked=False)
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits, chunked=True)[0],
            entries.vbt.signals.clean(exits, chunked=False)[0],
        )
        assert_frame_equal(
            entries.vbt.signals.clean(exits, chunked=True)[1],
            entries.vbt.signals.clean(exits, chunked=False)[1],
        )

    def test_generate_random(self):
        assert_series_equal(
            pd.Series.vbt.signals.generate_random(
                5,
                n=3,
                seed=seed,
                wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
            ),
            pd.Series(np.array([False, False, True, True, True]), index=mask["a"].index, name="a"),
        )
        with pytest.raises(Exception):
            pd.Series.vbt.signals.generate_random((5, 2), n=3)
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                n=3,
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, True, False],
                        [True, True, True],
                        [True, False, True],
                        [True, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                n=[0, 1, 2],
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, True],
                        [False, True, False],
                        [False, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                n=[0, 1, 2],
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
                chunked=True,
            ),
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                n=[0, 1, 2],
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
                chunked=False,
            ),
        )
        assert_series_equal(
            pd.Series.vbt.signals.generate_random(
                5,
                prob=0.5,
                seed=seed,
                wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
            ),
            pd.Series(np.array([True, False, False, False, True]), index=mask["a"].index, name="a"),
        )
        with pytest.raises(Exception):
            pd.Series.vbt.signals.generate_random((5, 2), prob=3)
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                prob=0.5,
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [False, True, False],
                        [False, False, False],
                        [False, False, True],
                        [True, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                prob=[[0.0, 0.5, 1.0]],
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [False, True, True],
                        [False, True, True],
                        [False, False, True],
                        [False, False, True],
                        [False, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        with pytest.raises(Exception):
            pd.DataFrame.vbt.signals.generate_random((5, 3))
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                prob=[[0.0, 0.5, 1.0]],
                pick_first=True,
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [False, True, True],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                prob=[[0.0, 0.5, 1.0]],
                pick_first=True,
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
                chunked=True,
            ),
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3),
                prob=[[0.0, 0.5, 1.0]],
                pick_first=True,
                seed=seed,
                wrap_kwargs=dict(index=mask.index, columns=mask.columns),
                chunked=False,
            ),
        )

    def test_generate_random_both(self):
        # n
        en, ex = pd.Series.vbt.signals.generate_random_both(
            5,
            n=2,
            seed=seed,
            wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
        )
        assert_series_equal(
            en,
            pd.Series(np.array([False, True, False, True, False]), index=mask["a"].index, name="a"),
        )
        assert_series_equal(
            ex,
            pd.Series(np.array([False, False, True, False, True]), index=mask["a"].index, name="a"),
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            n=2,
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, True],
                        [True, True, False],
                        [False, False, False],
                        [True, True, True],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, True],
                        [True, True, False],
                        [False, False, False],
                        [True, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            n=[0, 1, 2],
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, True],
                        [False, False, False],
                        [False, True, True],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, True],
                        [False, False, False],
                        [False, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        en2, ex2 = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            n=[0, 1, 2],
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
            chunked=True,
        )
        assert_frame_equal(en2, en)
        assert_frame_equal(ex2, ex)
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (2, 3), n=2, seed=seed, entry_wait=1, exit_wait=0
        )
        assert_frame_equal(en, pd.DataFrame(np.array([[True, True, True], [True, True, True]])))
        assert_frame_equal(ex, pd.DataFrame(np.array([[True, True, True], [True, True, True]])))
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (3, 3), n=2, seed=seed, entry_wait=0, exit_wait=1
        )
        assert_frame_equal(
            en,
            pd.DataFrame(np.array([[True, True, True], [True, True, True], [False, False, False]])),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, True, True],
                        [True, True, True],
                    ]
                )
            ),
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (7, 3), n=2, seed=seed, entry_wait=2, exit_wait=2
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [True, True, True],
                        [False, False, False],
                        [False, False, False],
                    ]
                )
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [True, True, True],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [True, True, True],
                    ]
                )
            ),
        )
        n = 10
        a = np.full(n * 2, 0.0)
        for i in range(10000):
            en, ex = pd.Series.vbt.signals.generate_random_both(1000, n, entry_wait=2, exit_wait=2)
            _a = np.empty((n * 2,), dtype=int_)
            _a[0::2] = np.flatnonzero(en)
            _a[1::2] = np.flatnonzero(ex)
            a += _a
        greater = a > 10000000 / (2 * n + 1) * np.arange(0, 2 * n)
        less = a < 10000000 / (2 * n + 1) * np.arange(2, 2 * n + 2)
        assert np.all(greater & less)

        # probs
        en, ex = pd.Series.vbt.signals.generate_random_both(
            5,
            entry_prob=0.5,
            exit_prob=1.0,
            seed=seed,
            wrap_kwargs=dict(index=mask["a"].index, columns=["a"]),
        )
        assert_series_equal(
            en,
            pd.Series(np.array([True, False, False, False, True]), index=mask["a"].index, name="a"),
        )
        assert_series_equal(
            ex,
            pd.Series(
                np.array([False, True, False, False, False]), index=mask["a"].index, name="a"
            ),
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            entry_prob=0.5,
            exit_prob=1.0,
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [False, False, False],
                        [False, False, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, True, True],
                        [False, False, False],
                        [False, False, False],
                        [False, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            entry_prob=[[0.0, 0.5, 1.0]],
            exit_prob=[[0.0, 0.5, 1.0]],
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [False, True, True],
                        [False, False, False],
                        [False, False, True],
                        [False, False, False],
                        [False, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, True, True],
                        [False, False, False],
                        [False, False, True],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        en2, ex2 = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            entry_prob=[[0.0, 0.5, 1.0]],
            exit_prob=[[0.0, 0.5, 1.0]],
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
            chunked=True,
        )
        assert_frame_equal(en2, en)
        assert_frame_equal(ex2, ex)
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            entry_prob=1.0,
            exit_prob=1.0,
            exit_wait=0,
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                    ],
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                    ],
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            entry_prob=1.0,
            exit_prob=1.0,
            entry_pick_first=False,
            exit_pick_first=True,
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                    ],
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3),
            entry_prob=1.0,
            exit_prob=1.0,
            entry_pick_first=True,
            exit_pick_first=False,
            seed=seed,
            wrap_kwargs=dict(index=mask.index, columns=mask.columns),
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [True, True, True],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        # none
        with pytest.raises(Exception):
            pd.DataFrame.vbt.signals.generate_random((5, 3))

    def test_generate_random_exits(self):
        assert_series_equal(
            mask["a"].vbt.signals.generate_random_exits(seed=seed),
            pd.Series(np.array([False, True, False, False, True]), index=mask["a"].index, name="a"),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_random_exits(seed=seed),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, False, False],
                        [False, True, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_random_exits(seed=seed, chunked=True),
            mask.vbt.signals.generate_random_exits(seed=seed, chunked=False),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_random_exits(seed=seed, wait=0),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, True, False],
                        [True, False, False],
                        [False, False, False],
                        [True, True, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_series_equal(
            mask["a"].vbt.signals.generate_random_exits(prob=1.0, seed=seed),
            pd.Series(np.array([False, True, False, False, True]), index=mask["a"].index, name="a"),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_random_exits(prob=1.0, seed=seed),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_random_exits(prob=[[0.0, 0.5, 1.0]], seed=seed),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, True, True],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_random_exits(prob=[[0.0, 0.5, 1.0]], seed=seed, chunked=True),
            mask.vbt.signals.generate_random_exits(
                prob=[[0.0, 0.5, 1.0]], seed=seed, chunked=False
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_random_exits(prob=1.0, wait=0, seed=seed),
            pd.DataFrame(
                np.array(
                    [
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                        [False, True, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_random_exits(prob=1.0, until_next=False, seed=seed),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )

    def test_generate_stop_exits(self):
        e = pd.Series([True, False, False, False, False, False])
        t = pd.Series([2, 3, 4, 3, 2, 1]).astype(np.float64)

        # stop loss
        assert_series_equal(
            e.vbt.signals.generate_stop_exits(t, stop=-0.1),
            pd.Series(np.array([False, False, False, False, False, True])),
        )
        assert_series_equal(
            e.vbt.signals.generate_stop_exits(t, stop=-0.1, trailing=True),
            pd.Series(np.array([False, False, False, True, False, False])),
        )
        # take profit
        assert_series_equal(
            e.vbt.signals.generate_stop_exits(4 - t, stop=0.1),
            pd.Series(np.array([False, False, False, False, False, True])),
        )
        assert_series_equal(
            e.vbt.signals.generate_stop_exits(4 - t, stop=0.1, trailing=True),
            pd.Series(np.array([False, False, False, True, False, False])),
        )
        # chain
        e = pd.Series([True, True, True, True, True, True])
        en, ex = e.vbt.signals.generate_stop_exits(t, stop=-0.1, trailing=True, chain=True)
        assert_series_equal(en, pd.Series(np.array([True, False, False, False, True, False])))
        assert_series_equal(ex, pd.Series(np.array([False, False, False, True, False, True])))
        # until_next
        e2 = pd.Series([True, True, True, True, True, True])
        t2 = pd.Series([6, 5, 4, 3, 2, 1]).astype(np.float64)
        ex = e2.vbt.signals.generate_stop_exits(t2, stop=-0.1, until_next=False)
        assert_series_equal(ex, pd.Series(np.array([False, True, True, True, True, True])))
        assert_frame_equal(
            e.vbt.signals.generate_stop_exits(
                t.vbt.tile(3),
                stop=[[np.nan, -0.5, -1.0]],
                trailing=True,
                chain=True,
                chunked=True,
            )[0],
            e.vbt.signals.generate_stop_exits(
                t.vbt.tile(3),
                stop=[[np.nan, -0.5, -1.0]],
                trailing=True,
                chain=True,
                chunked=False,
            )[0],
        )
        assert_frame_equal(
            e.vbt.signals.generate_stop_exits(
                t.vbt.tile(3),
                stop=[[np.nan, -0.5, -1.0]],
                trailing=True,
                chain=True,
                chunked=True,
            )[1],
            e.vbt.signals.generate_stop_exits(
                t.vbt.tile(3),
                stop=[[np.nan, -0.5, -1.0]],
                trailing=True,
                chain=True,
                chunked=False,
            )[1],
        )

    def test_generate_ohlc_stop_exits(self):
        assert_frame_equal(
            mask.vbt.signals.generate_stop_exits(ts, stop=-0.1),
            mask.vbt.signals.generate_ohlc_stop_exits(ts, sl_stop=0.1),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_stop_exits(ts, stop=-0.1, trailing=True),
            mask.vbt.signals.generate_ohlc_stop_exits(ts, tsl_stop=0.1),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_stop_exits(ts, stop=0.1),
            mask.vbt.signals.generate_ohlc_stop_exits(ts, tp_stop=0.1),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_stop_exits(ts, stop=0.1),
            mask.vbt.signals.generate_ohlc_stop_exits(ts, sl_stop=0.1, reverse=True),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_stop_exits(ts, stop=0.1, trailing=True),
            mask.vbt.signals.generate_ohlc_stop_exits(ts, tsl_stop=0.1, reverse=True),
        )
        assert_frame_equal(
            mask.vbt.signals.generate_stop_exits(ts, stop=-0.1),
            mask.vbt.signals.generate_ohlc_stop_exits(ts, tp_stop=0.1, reverse=True),
        )

        def _test_ohlc_stop_exits(**kwargs):
            out_dict = {"stop_price": np.nan, "stop_type": -1}
            result = mask.vbt.signals.generate_ohlc_stop_exits(
                price["open"],
                price["open"],
                price["high"],
                price["low"],
                price["close"],
                out_dict=out_dict,
                is_entry_open=True,
                **kwargs,
            )
            if isinstance(result, tuple):
                _, ex = result
            else:
                ex = result
            return result, out_dict["stop_price"], out_dict["stop_type"]

        ex, stop_price, stop_type = _test_ohlc_stop_exits()
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_type,
            pd.DataFrame(
                np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        ex, stop_price, stop_type = _test_ohlc_stop_exits(sl_stop=0.1)
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, 10.8],
                        [9.9, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_type,
            pd.DataFrame(
                np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, 0], [0, -1, -1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        ex, stop_price, stop_type = _test_ohlc_stop_exits(tsl_stop=0.1)
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, True, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, 11.7, 11.7],
                        [10.8, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_type,
            pd.DataFrame(
                np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, 1, 1], [1, -1, -1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        ex, stop_price, stop_type = _test_ohlc_stop_exits(tp_stop=0.1)
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, True, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [11.0, np.nan, np.nan],
                        [np.nan, 12.1, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_type,
            pd.DataFrame(
                np.array([[-1, -1, -1], [3, -1, -1], [-1, 3, -1], [-1, -1, -1], [-1, -1, -1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        ex, stop_price, stop_type = _test_ohlc_stop_exits(tsl_stop=0.1, tp_stop=0.1)
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [11.0, np.nan, np.nan],
                        [np.nan, 12.1, np.nan],
                        [np.nan, np.nan, 11.7],
                        [10.8, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_type,
            pd.DataFrame(
                np.array([[-1, -1, -1], [3, -1, -1], [-1, 3, -1], [-1, -1, 1], [1, -1, -1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        ex, stop_price, stop_type = _test_ohlc_stop_exits(
            tsl_stop=[[np.nan, 0.1, 0.2]],
            tp_stop=[[np.nan, 0.1, 0.2]],
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, True, False],
                        [False, False, True],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, 12.1, np.nan],
                        [np.nan, np.nan, 10.4],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_type,
            pd.DataFrame(
                np.array([[-1, -1, -1], [-1, -1, -1], [-1, 3, -1], [-1, -1, 1], [-1, -1, -1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]], tp_stop=[[np.nan, 0.1, 0.2]], chunked=True
            )[0],
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]], tp_stop=[[np.nan, 0.1, 0.2]], chunked=False
            )[0],
        )
        assert_frame_equal(
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]], tp_stop=[[np.nan, 0.1, 0.2]], chunked=True
            )[1],
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]], tp_stop=[[np.nan, 0.1, 0.2]], chunked=False
            )[1],
        )
        assert_frame_equal(
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]], tp_stop=[[np.nan, 0.1, 0.2]], chunked=True
            )[2],
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]], tp_stop=[[np.nan, 0.1, 0.2]], chunked=False
            )[2],
        )
        ex, stop_price, stop_type = _test_ohlc_stop_exits(tsl_stop=0.1, tp_stop=0.1, exit_wait=0)
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [True, False, False],
                        [False, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, True, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [9.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, 12.1, np.nan],
                        [np.nan, np.nan, 11.7],
                        [10.8, 9.0, np.nan],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_type,
            pd.DataFrame(
                np.array([[1, -1, -1], [-1, -1, -1], [-1, 3, -1], [-1, -1, 1], [1, 1, -1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        (en, ex), stop_price, stop_type = _test_ohlc_stop_exits(
            tsl_stop=0.1, tp_stop=0.1, chain=True
        )
        assert_frame_equal(
            en,
            pd.DataFrame(
                np.array(
                    [
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                        [False, True, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [11.0, np.nan, np.nan],
                        [np.nan, 12.1, np.nan],
                        [np.nan, np.nan, 11.7],
                        [10.8, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            stop_type,
            pd.DataFrame(
                np.array([[-1, -1, -1], [3, -1, -1], [-1, 3, -1], [-1, -1, 1], [1, -1, -1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]],
                chain=True,
                tp_stop=[[np.nan, 0.1, 0.2]],
                chunked=True,
            )[0][0],
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]],
                chain=True,
                tp_stop=[[np.nan, 0.1, 0.2]],
                chunked=False,
            )[0][0],
        )
        assert_frame_equal(
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]],
                chain=True,
                tp_stop=[[np.nan, 0.1, 0.2]],
                chunked=True,
            )[0][0],
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]],
                chain=True,
                tp_stop=[[np.nan, 0.1, 0.2]],
                chunked=False,
            )[0][0],
        )
        assert_frame_equal(
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]],
                chain=True,
                tp_stop=[[np.nan, 0.1, 0.2]],
                chunked=True,
            )[1],
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]],
                chain=True,
                tp_stop=[[np.nan, 0.1, 0.2]],
                chunked=False,
            )[1],
        )
        assert_frame_equal(
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]],
                chain=True,
                tp_stop=[[np.nan, 0.1, 0.2]],
                chunked=True,
            )[2],
            _test_ohlc_stop_exits(
                tsl_stop=[[np.nan, 0.1, 0.2]],
                chain=True,
                tp_stop=[[np.nan, 0.1, 0.2]],
                chunked=False,
            )[2],
        )

    def test_between_ranges(self):
        ranges = mask.vbt.signals.between_ranges()
        assert_records_close(
            ranges.values, np.array([(0, 0, 0, 3, 1), (0, 1, 1, 4, 1)], dtype=range_dt)
        )
        assert ranges.wrapper == mask.vbt.wrapper
        ranges = mask.vbt.signals.between_ranges(incl_open=True)
        assert_records_close(
            ranges.values,
            np.array(
                [(0, 0, 0, 3, 1), (1, 0, 3, 4, 0), (0, 1, 1, 4, 1), (0, 2, 2, 4, 0)], dtype=range_dt
            ),
        )
        assert ranges.wrapper == mask.vbt.wrapper

        source = pd.DataFrame(
            [
                [True, True, True],
                [True, True, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
            index=mask.index,
            columns=mask.columns,
        )
        target = pd.DataFrame(
            [
                [False, False, False],
                [True, False, False],
                [True, True, False],
                [False, True, True],
                [False, False, True],
            ],
            index=mask.index,
            columns=mask.columns,
        )
        ranges = source.vbt.signals.between_ranges(target=target)
        assert_records_close(
            ranges.values,
            np.array(
                [
                    (0, 0, 1, 1, 1),
                    (1, 0, 1, 2, 1),
                    (0, 1, 1, 2, 1),
                    (1, 1, 1, 3, 1),
                    (0, 2, 1, 3, 1),
                    (1, 2, 1, 4, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert ranges.wrapper == source.vbt.wrapper
        ranges = source.vbt.signals.between_ranges(target=target, relation="manyone")
        assert_records_close(
            ranges.values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 1, 1, 1),
                    (0, 1, 0, 2, 1),
                    (1, 1, 1, 2, 1),
                    (0, 2, 0, 3, 1),
                    (1, 2, 1, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert ranges.wrapper == source.vbt.wrapper
        ranges = source.vbt.signals.between_ranges(target=target, relation="manymany")
        assert_records_close(
            ranges.values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 1, 1, 1),
                    (2, 0, 1, 2, 1),
                    (0, 1, 0, 2, 1),
                    (1, 1, 1, 2, 1),
                    (2, 1, 1, 3, 1),
                    (0, 2, 0, 3, 1),
                    (1, 2, 1, 3, 1),
                    (2, 2, 1, 4, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert ranges.wrapper == source.vbt.wrapper
        ranges = source.vbt.signals.between_ranges(target=target, relation="oneone")
        assert_records_close(
            ranges.values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 1, 2, 1),
                    (0, 1, 0, 2, 1),
                    (1, 1, 1, 3, 1),
                    (0, 2, 0, 3, 1),
                    (1, 2, 1, 4, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert ranges.wrapper == source.vbt.wrapper
        assert_records_close(
            mask.vbt.signals.between_ranges(jitted=dict(parallel=True)).values,
            mask.vbt.signals.between_ranges(jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            mask.vbt.signals.between_ranges(chunked=True).values,
            mask.vbt.signals.between_ranges(chunked=False).values,
        )
        assert_records_close(
            mask.vbt.signals.between_ranges(target=target, jitted=dict(parallel=True)).values,
            mask.vbt.signals.between_ranges(target=target, jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            mask.vbt.signals.between_ranges(target=target, chunked=True).values,
            mask.vbt.signals.between_ranges(target=target, chunked=False).values,
        )

    def test_partition_ranges(self):
        mask2 = pd.DataFrame(
            [
                [False, False, False],
                [True, False, False],
                [True, True, False],
                [False, True, True],
                [True, False, True],
            ],
            index=mask.index,
            columns=mask.columns,
        )

        ranges = mask2.vbt.signals.partition_ranges()
        assert_records_close(
            ranges.values,
            np.array(
                [(0, 0, 1, 3, 1), (1, 0, 4, 4, 0), (0, 1, 2, 4, 1), (0, 2, 3, 4, 0)], dtype=range_dt
            ),
        )
        assert ranges.wrapper == mask2.vbt.wrapper

        assert_records_close(
            mask.vbt.signals.partition_ranges(jitted=dict(parallel=True)).values,
            mask.vbt.signals.partition_ranges(jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            mask.vbt.signals.partition_ranges(chunked=True).values,
            mask.vbt.signals.partition_ranges(chunked=False).values,
        )

    def test_between_partition_ranges(self):
        mask2 = pd.DataFrame(
            [
                [True, False, False],
                [True, True, False],
                [False, True, True],
                [True, False, True],
                [False, True, False],
            ],
            index=mask.index,
            columns=mask.columns,
        )

        ranges = mask2.vbt.signals.between_partition_ranges()
        assert_records_close(
            ranges.values, np.array([(0, 0, 1, 3, 1), (0, 1, 2, 4, 1)], dtype=range_dt)
        )
        assert ranges.wrapper == mask2.vbt.wrapper

        assert_records_close(
            mask.vbt.signals.between_partition_ranges(jitted=dict(parallel=True)).values,
            mask.vbt.signals.between_partition_ranges(jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            mask.vbt.signals.between_partition_ranges(chunked=True).values,
            mask.vbt.signals.between_partition_ranges(chunked=False).values,
        )

    def test_unravel(self):
        assert_frame_equal(
            mask.vbt.signals.unravel(),
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, False, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (0, "b"), (1, "b"), (0, "c")], names=["signal", None]
                ),
            ),
        )
        mask2 = mask.copy()
        mask2.iloc[:, 1] = False
        assert_frame_equal(
            mask2.vbt.signals.unravel(),
            pd.DataFrame(
                [
                    [True, False, False, False],
                    [False, False, False, False],
                    [False, False, False, True],
                    [False, True, False, False],
                    [False, False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (-1, "b"), (0, "c")], names=["signal", None]
                ),
            ),
        )
        assert_frame_equal(
            mask2.vbt.signals.unravel(incl_empty_cols=False),
            pd.DataFrame(
                [
                    [True, False, False],
                    [False, False, False],
                    [False, False, True],
                    [False, True, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (0, "c")], names=["signal", None]
                ),
            ),
        )
        mask3 = mask.copy()
        mask3.iloc[:] = False
        assert_frame_equal(
            mask3.vbt.signals.unravel(),
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.Index(["a", "b", "c"], dtype="object"),
            ),
        )
        assert_frame_equal(
            mask3.vbt.signals.unravel(force_signal_index=True),
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (-1, "a"),
                        (-1, "b"),
                        (-1, "c"),
                    ],
                    names=["signal", None],
                ),
            ),
        )
        with pytest.raises(Exception):
            mask3.vbt.signals.unravel(incl_empty_cols=False)
        assert_frame_equal(
            mask.vbt.signals.unravel(signal_index_type="positions"),
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, False, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (3, "a"), (1, "b"), (4, "b"), (2, "c")], names=["signal", None]
                ),
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.unravel(signal_index_type="labels"),
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, False, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (mask.index[0], "a"),
                        (mask.index[3], "a"),
                        (mask.index[1], "b"),
                        (mask.index[4], "b"),
                        (mask.index[2], "c"),
                    ],
                    names=["signal", None],
                ),
            ),
        )

    def test_unravel_between(self):
        assert_frame_equal(
            mask.vbt.signals.unravel_between(incl_open_source=True),
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [True, True, False, False, False],
                    [False, False, True, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, "a"),
                        (1, "a"),
                        (0, "b"),
                        (1, "b"),
                        (0, "c"),
                    ],
                    names=["signal", None],
                ),
            ),
        )
        mask2 = mask.copy()
        mask2.iloc[:, 1] = False
        assert_frame_equal(
            mask2.vbt.signals.unravel_between(incl_open_source=True),
            pd.DataFrame(
                [
                    [True, False, False, False],
                    [False, False, False, False],
                    [False, False, False, True],
                    [True, True, False, False],
                    [False, False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, "a"),
                        (1, "a"),
                        (-1, "b"),
                        (0, "c"),
                    ],
                    names=["signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask2.vbt.signals.unravel_between(incl_open_source=True, incl_empty_cols=False),
            pd.DataFrame(
                [
                    [True, False, False],
                    [False, False, False],
                    [False, False, True],
                    [True, True, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, "a"),
                        (1, "a"),
                        (0, "c"),
                    ],
                    names=["signal", None],
                ),
            ),
        )
        mask3 = mask.copy()
        mask3.iloc[:] = False
        assert_frame_equal(
            mask3.vbt.signals.unravel_between(incl_open_source=True),
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.Index(["a", "b", "c"], dtype="object"),
            ),
        )
        assert_frame_equal(
            mask3.vbt.signals.unravel_between(force_signal_index=True),
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (-1, "a"),
                        (-1, "b"),
                        (-1, "c"),
                    ],
                    names=["signal", None],
                ),
            ),
        )
        with pytest.raises(Exception):
            mask3.vbt.signals.unravel_between(incl_empty_cols=False)
        assert_frame_equal(
            mask.vbt.signals.unravel_between(incl_open_source=True, signal_index_type="positions"),
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [True, True, False, False, False],
                    [False, False, True, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, 3, "a"),
                        (3, -1, "a"),
                        (1, 4, "b"),
                        (4, -1, "b"),
                        (2, -1, "c"),
                    ],
                    names=["source_signal", "target_signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.unravel_between(signal_index_type="labels", incl_empty_cols=False),
            pd.DataFrame(
                [
                    [True, False],
                    [False, True],
                    [False, False],
                    [True, False],
                    [False, True],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (mask.index[0], mask.index[3], "a"),
                        (mask.index[1], mask.index[4], "b"),
                    ],
                    names=["source_signal", "target_signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.unravel_between(mask.vbt.signals.fshift(), incl_open_source=True)[0],
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, False, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (0, "b"), (1, "b"), (0, "c")],
                    names=["signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.unravel_between(mask.vbt.signals.fshift(), incl_open_source=True)[1],
            pd.DataFrame(
                [
                    [False, False, False, False, False],
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (0, "b"), (1, "b"), (0, "c")],
                    names=["signal", None],
                ),
            ),
        )
        mask2 = mask.copy()
        mask2.iloc[:, 1] = False
        assert_frame_equal(
            mask2.vbt.signals.unravel_between(mask2.vbt.signals.fshift(), incl_open_source=True)[0],
            pd.DataFrame(
                [
                    [True, False, False, False],
                    [False, False, False, False],
                    [False, False, False, True],
                    [False, True, False, False],
                    [False, False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (-1, "b"), (0, "c")],
                    names=["signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask2.vbt.signals.unravel_between(mask2.vbt.signals.fshift(), incl_open_source=True)[1],
            pd.DataFrame(
                [
                    [False, False, False, False],
                    [True, False, False, False],
                    [False, False, False, False],
                    [False, False, False, True],
                    [False, True, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (-1, "b"), (0, "c")],
                    names=["signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask2.vbt.signals.unravel_between(
                mask2.vbt.signals.fshift(),
                incl_open_source=True,
                incl_empty_cols=False,
            )[0],
            pd.DataFrame(
                [
                    [True, False, False],
                    [False, False, False],
                    [False, False, True],
                    [False, True, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (0, "c")], names=["signal", None]
                ),
            ),
        )
        assert_frame_equal(
            mask2.vbt.signals.unravel_between(
                mask2.vbt.signals.fshift(),
                incl_open_source=True,
                incl_empty_cols=False,
            )[1],
            pd.DataFrame(
                [
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, False, True],
                    [False, True, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (0, "c")], names=["signal", None]
                ),
            ),
        )
        mask3 = mask.copy()
        mask3.iloc[:] = False
        assert_frame_equal(
            mask3.vbt.signals.unravel_between(mask3.vbt.signals.fshift(), incl_open_source=True)[0],
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.Index(["a", "b", "c"], dtype="object"),
            ),
        )
        assert_frame_equal(
            mask3.vbt.signals.unravel_between(mask3.vbt.signals.fshift(), incl_open_source=True)[1],
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.Index(["a", "b", "c"], dtype="object"),
            ),
        )
        assert_frame_equal(
            mask3.vbt.signals.unravel_between(
                mask3.vbt.signals.fshift(),
                incl_open_source=True,
                force_signal_index=True,
            )[0],
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (-1, "a"),
                        (-1, "b"),
                        (-1, "c"),
                    ],
                    names=["signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask3.vbt.signals.unravel_between(
                mask3.vbt.signals.fshift(),
                incl_open_source=True,
                force_signal_index=True,
            )[1],
            pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                index=mask2.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (-1, "a"),
                        (-1, "b"),
                        (-1, "c"),
                    ],
                    names=["signal", None],
                ),
            ),
        )
        with pytest.raises(Exception):
            mask3.vbt.signals.unravel_between(mask3.vbt.signals.fshift(), incl_empty_cols=False)
        assert_frame_equal(
            mask.vbt.signals.unravel_between(
                mask.vbt.signals.fshift(),
                incl_open_source=True,
                signal_index_type="range",
            )[0],
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, False, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, 0, "a"), (1, 1, "a"), (0, 0, "b"), (1, -1, "b"), (0, 0, "c")],
                    names=["source_signal", "target_signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.unravel_between(
                mask.vbt.signals.fshift(),
                incl_open_source=True,
                signal_index_type="source_range",
            )[0],
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, False, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (0, "b"), (1, "b"), (0, "c")],
                    names=["signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.unravel_between(
                mask.vbt.signals.fshift(),
                incl_open_source=True,
                signal_index_type="target_range",
            )[0],
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, False, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (1, "a"), (0, "b"), (-1, "b"), (0, "c")],
                    names=["signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.unravel_between(
                mask.vbt.signals.fshift(),
                incl_open_source=True,
                signal_index_type="positions",
            )[0],
            pd.DataFrame(
                [
                    [True, False, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, True],
                    [False, True, False, False, False],
                    [False, False, False, True, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1, "a"), (3, 4, "a"), (1, 2, "b"), (4, -1, "b"), (2, 3, "c")],
                    names=["source_signal", "target_signal", None],
                ),
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.unravel_between(
                mask.vbt.signals.fshift(),
                signal_index_type="labels",
                incl_empty_cols=False,
            )[0],
            pd.DataFrame(
                [
                    [True, False, False, False],
                    [False, False, True, False],
                    [False, False, False, True],
                    [False, True, False, False],
                    [False, False, False, False],
                ],
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (mask.index[0], mask.index[1], "a"),
                        (mask.index[3], mask.index[4], "a"),
                        (mask.index[1], mask.index[2], "b"),
                        (mask.index[2], mask.index[3], "c"),
                    ],
                    names=["source_signal", "target_signal", None],
                ),
            ),
        )

    def test_rank(self):
        mask2 = pd.Series([True, True, False, True, True, False, True, True, True])
        mask3 = pd.Series([False, False, False, False, True, False, False, True, False])

        last_false_i = []
        last_reset_i = []
        all_sig_cnt = []
        all_part_cnt = []
        all_sig_in_part_cnt = []
        nonres_sig_cnt = []
        nonres_part_cnt = []
        nonres_sig_in_part_cnt = []
        sig_cnt = []
        part_cnt = []
        sig_in_part_cnt = []

        def rank_func(c):
            last_false_i.append(c.last_false_i)
            last_reset_i.append(c.last_reset_i)
            all_sig_cnt.append(c.all_sig_cnt)
            all_part_cnt.append(c.all_part_cnt)
            all_sig_in_part_cnt.append(c.all_sig_in_part_cnt)
            nonres_sig_cnt.append(c.nonres_sig_cnt)
            nonres_part_cnt.append(c.nonres_part_cnt)
            nonres_sig_in_part_cnt.append(c.nonres_sig_in_part_cnt)
            sig_cnt.append(c.sig_cnt)
            part_cnt.append(c.part_cnt)
            sig_in_part_cnt.append(c.sig_in_part_cnt)
            return -1

        mask2.vbt.signals.rank(rank_func, jitted=False)

        assert last_false_i == [-1, -1, 2, 2, 5, 5, 5]
        assert last_reset_i == [-1, -1, -1, -1, -1, -1, -1]
        assert all_sig_cnt == [1, 2, 3, 4, 5, 6, 7]
        assert all_part_cnt == [1, 1, 2, 2, 3, 3, 3]
        assert all_sig_in_part_cnt == [1, 2, 1, 2, 1, 2, 3]
        assert nonres_sig_cnt == [1, 2, 3, 4, 5, 6, 7]
        assert nonres_part_cnt == [1, 1, 2, 2, 3, 3, 3]
        assert nonres_sig_in_part_cnt == [1, 2, 1, 2, 1, 2, 3]
        assert sig_cnt == [1, 2, 3, 4, 5, 6, 7]
        assert part_cnt == [1, 1, 2, 2, 3, 3, 3]
        assert sig_in_part_cnt == [1, 2, 1, 2, 1, 2, 3]

        last_false_i = []
        last_reset_i = []
        all_sig_cnt = []
        all_part_cnt = []
        all_sig_in_part_cnt = []
        nonres_sig_cnt = []
        nonres_part_cnt = []
        nonres_sig_in_part_cnt = []
        sig_cnt = []
        part_cnt = []
        sig_in_part_cnt = []

        mask2.vbt.signals.rank(rank_func, after_false=True, jitted=False)

        assert last_false_i == [2, 2, 5, 5, 5]
        assert last_reset_i == [-1, -1, -1, -1, -1]
        assert all_sig_cnt == [3, 4, 5, 6, 7]
        assert all_part_cnt == [2, 2, 3, 3, 3]
        assert all_sig_in_part_cnt == [1, 2, 1, 2, 3]
        assert nonres_sig_cnt == [1, 2, 3, 4, 5]
        assert nonres_part_cnt == [1, 1, 2, 2, 2]
        assert nonres_sig_in_part_cnt == [1, 2, 1, 2, 3]
        assert sig_cnt == [1, 2, 3, 4, 5]
        assert part_cnt == [1, 1, 2, 2, 2]
        assert sig_in_part_cnt == [1, 2, 1, 2, 3]

        last_false_i = []
        last_reset_i = []
        all_sig_cnt = []
        all_part_cnt = []
        all_sig_in_part_cnt = []
        nonres_sig_cnt = []
        nonres_part_cnt = []
        nonres_sig_in_part_cnt = []
        sig_cnt = []
        part_cnt = []
        sig_in_part_cnt = []

        mask2.vbt.signals.rank(
            rank_func, reset_by=mask3, after_reset=False, reset_wait=0, jitted=False
        )

        assert last_false_i == [-1, -1, 2, 2, 5, 5, 5]
        assert last_reset_i == [-1, -1, -1, 4, 4, 7, 7]
        assert all_sig_cnt == [1, 2, 3, 4, 5, 6, 7]
        assert all_part_cnt == [1, 1, 2, 2, 3, 3, 3]
        assert all_sig_in_part_cnt == [1, 2, 1, 2, 1, 2, 3]
        assert nonres_sig_cnt == [1, 2, 3, 4, 5, 6, 7]
        assert nonres_part_cnt == [1, 1, 2, 2, 3, 3, 3]
        assert nonres_sig_in_part_cnt == [1, 2, 1, 2, 1, 2, 3]
        assert sig_cnt == [1, 2, 3, 1, 2, 1, 2]
        assert part_cnt == [1, 1, 2, 1, 2, 1, 1]
        assert sig_in_part_cnt == [1, 2, 1, 1, 1, 1, 2]

        last_false_i = []
        last_reset_i = []
        all_sig_cnt = []
        all_part_cnt = []
        all_sig_in_part_cnt = []
        nonres_sig_cnt = []
        nonres_part_cnt = []
        nonres_sig_in_part_cnt = []
        sig_cnt = []
        part_cnt = []
        sig_in_part_cnt = []

        mask2.vbt.signals.rank(
            rank_func, reset_by=mask3, after_reset=True, reset_wait=0, jitted=False
        )

        assert last_false_i == [2, 5, 5, 5]
        assert last_reset_i == [4, 4, 7, 7]
        assert all_sig_cnt == [4, 5, 6, 7]
        assert all_part_cnt == [2, 3, 3, 3]
        assert all_sig_in_part_cnt == [2, 1, 2, 3]
        assert nonres_sig_cnt == [1, 2, 3, 4]
        assert nonres_part_cnt == [1, 2, 2, 2]
        assert nonres_sig_in_part_cnt == [1, 1, 2, 3]
        assert sig_cnt == [1, 2, 1, 2]
        assert part_cnt == [1, 2, 1, 1]
        assert sig_in_part_cnt == [1, 1, 1, 2]

        last_false_i = []
        last_reset_i = []
        all_sig_cnt = []
        all_part_cnt = []
        all_sig_in_part_cnt = []
        nonres_sig_cnt = []
        nonres_part_cnt = []
        nonres_sig_in_part_cnt = []
        sig_cnt = []
        part_cnt = []
        sig_in_part_cnt = []

        mask2.vbt.signals.rank(
            rank_func, reset_by=mask3, after_reset=True, reset_wait=1, jitted=False
        )

        assert last_false_i == [5, 5, 5]
        assert last_reset_i == [4, 7, 7]
        assert all_sig_cnt == [5, 6, 7]
        assert all_part_cnt == [3, 3, 3]
        assert all_sig_in_part_cnt == [1, 2, 3]
        assert nonres_sig_cnt == [1, 2, 3]
        assert nonres_part_cnt == [1, 1, 1]
        assert nonres_sig_in_part_cnt == [1, 2, 3]
        assert sig_cnt == [1, 2, 1]
        assert part_cnt == [1, 1, 1]
        assert sig_in_part_cnt == [1, 2, 1]

    def test_pos_rank(self):
        assert_series_equal(
            (~mask["a"]).vbt.signals.pos_rank(),
            pd.Series([-1, 0, 1, -1, 0], index=mask["a"].index, name="a"),
        )
        assert_frame_equal(
            (~mask).vbt.signals.pos_rank(),
            pd.DataFrame(
                np.array([[-1, 0, 0], [0, -1, 1], [1, 0, -1], [-1, 1, 0], [0, -1, 1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.pos_rank(after_false=True),
            pd.DataFrame(
                np.array([[-1, -1, -1], [0, -1, -1], [1, 0, -1], [-1, 1, 0], [0, -1, 1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.pos_rank(allow_gaps=True),
            pd.DataFrame(
                np.array([[-1, 0, 0], [0, -1, 1], [1, 1, -1], [-1, 2, 2], [2, -1, 3]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.pos_rank(reset_by=mask["a"], reset_wait=0, allow_gaps=True),
            pd.DataFrame(
                np.array([[-1, 0, 0], [0, -1, 1], [1, 1, -1], [-1, 0, 0], [0, -1, 1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.pos_rank(reset_by=mask, reset_wait=0, allow_gaps=True),
            pd.DataFrame(
                np.array([[-1, 0, 0], [0, -1, 1], [1, 0, -1], [-1, 1, 0], [0, -1, 1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.pos_rank(jitted=dict(parallel=True)),
            (~mask).vbt.signals.pos_rank(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            (~mask).vbt.signals.pos_rank(chunked=True),
            (~mask).vbt.signals.pos_rank(chunked=False),
        )

    def test_partition_pos_rank(self):
        assert_series_equal(
            (~mask["a"]).vbt.signals.partition_pos_rank(),
            pd.Series([-1, 0, 0, -1, 1], index=mask["a"].index, name="a"),
        )
        assert_frame_equal(
            (~mask).vbt.signals.partition_pos_rank(),
            pd.DataFrame(
                np.array([[-1, 0, 0], [0, -1, 0], [0, 1, -1], [-1, 1, 1], [1, -1, 1]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.partition_pos_rank(after_false=True),
            pd.DataFrame(
                np.array([[-1, -1, -1], [0, -1, -1], [0, 0, -1], [-1, 0, 0], [1, -1, 0]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.partition_pos_rank(reset_by=mask["a"], reset_wait=0),
            pd.DataFrame(
                np.array([[-1, 0, 0], [0, -1, 0], [0, 1, -1], [-1, 0, 0], [0, -1, 0]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.partition_pos_rank(reset_by=mask, reset_wait=0),
            pd.DataFrame(
                np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [-1, 0, 0], [0, -1, 0]]),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.partition_pos_rank(jitted=dict(parallel=True)),
            (~mask).vbt.signals.partition_pos_rank(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            (~mask).vbt.signals.partition_pos_rank(chunked=True),
            (~mask).vbt.signals.partition_pos_rank(chunked=False),
        )

    def test_pos_rank_fns(self):
        assert_frame_equal(
            (~mask).vbt.signals.first(),
            pd.DataFrame(
                np.array(
                    [
                        [False, True, True],
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.nth(1),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, True],
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.nth(2),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            (~mask).vbt.signals.from_nth(0),
            pd.DataFrame(
                np.array(
                    [
                        [False, True, True],
                        [True, False, True],
                        [True, True, False],
                        [False, True, True],
                        [True, False, True],
                    ]
                ),
                index=mask.index,
                columns=mask.columns,
            ),
        )

    def test_pos_rank_mapped(self):
        mask2 = pd.DataFrame(
            [
                [True, False, False],
                [True, True, False],
                [False, True, True],
                [True, False, True],
                [False, True, False],
            ],
            index=mask.index,
            columns=mask.columns,
        )

        mapped = mask2.vbt.signals.pos_rank_mapped()
        np.testing.assert_array_equal(mapped.values, np.array([0, 1, 0, 0, 1, 0, 0, 1]))
        np.testing.assert_array_equal(mapped.col_arr, np.array([0, 0, 0, 1, 1, 1, 2, 2]))
        np.testing.assert_array_equal(mapped.idx_arr, np.array([0, 1, 3, 1, 2, 4, 2, 3]))
        assert mapped.wrapper == mask2.vbt.wrapper

    def test_partition_pos_rank_mapped(self):
        mask2 = pd.DataFrame(
            [
                [True, False, False],
                [True, True, False],
                [False, True, True],
                [True, False, True],
                [False, True, False],
            ],
            index=mask.index,
            columns=mask.columns,
        )

        mapped = mask2.vbt.signals.partition_pos_rank_mapped()
        np.testing.assert_array_equal(mapped.values, np.array([0, 0, 1, 0, 0, 1, 0, 0]))
        np.testing.assert_array_equal(mapped.col_arr, np.array([0, 0, 0, 1, 1, 1, 2, 2]))
        np.testing.assert_array_equal(mapped.idx_arr, np.array([0, 1, 3, 1, 2, 4, 2, 3]))
        assert mapped.wrapper == mask2.vbt.wrapper

    def test_distance_from_last(self):
        assert_series_equal(
            mask["a"].vbt.signals.distance_from_last(),
            pd.Series([-1, 1, 2, 3, 1], index=mask.index, name="a"),
        )
        assert_frame_equal(
            mask.vbt.signals.distance_from_last(),
            pd.DataFrame(
                [[-1, -1, -1], [1, -1, -1], [2, 1, -1], [3, 2, 1], [1, 3, 2]],
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.distance_from_last(nth=0),
            pd.DataFrame(
                [[0, -1, -1], [1, 0, -1], [2, 1, 0], [0, 2, 1], [1, 0, 2]],
                index=mask.index,
                columns=mask.columns,
            ),
        )
        assert_frame_equal(
            mask.vbt.signals.distance_from_last(nth=2),
            pd.DataFrame(
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [3, -1, -1], [4, 3, -1]],
                index=mask.index,
                columns=mask.columns,
            ),
        )

    def test_nth_index(self):
        assert mask["a"].vbt.signals.nth_index(0) == pd.Timestamp("2020-01-01 00:00:00")
        assert_series_equal(
            mask.vbt.signals.nth_index(0),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-02 00:00:00"),
                    pd.Timestamp("2020-01-03 00:00:00"),
                ],
                index=mask.columns,
                name="nth_index",
                dtype="datetime64[ns]",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.nth_index(-1),
            pd.Series(
                [
                    pd.Timestamp("2020-01-04 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timestamp("2020-01-03 00:00:00"),
                ],
                index=mask.columns,
                name="nth_index",
                dtype="datetime64[ns]",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.nth_index(-2),
            pd.Series(
                [pd.Timestamp("2020-01-01 00:00:00"), pd.Timestamp("2020-01-02 00:00:00"), np.nan],
                index=mask.columns,
                name="nth_index",
                dtype="datetime64[ns]",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.nth_index(-1, jitted=dict(parallel=True)),
            mask.vbt.signals.nth_index(-1, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            mask.vbt.signals.nth_index(0, group_by=group_by),
            pd.Series(
                [pd.Timestamp("2020-01-01 00:00:00"), pd.Timestamp("2020-01-03 00:00:00")],
                index=["g1", "g2"],
                name="nth_index",
                dtype="datetime64[ns]",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.nth_index(-1, group_by=group_by),
            pd.Series(
                [pd.Timestamp("2020-01-05 00:00:00"), pd.Timestamp("2020-01-03 00:00:00")],
                index=["g1", "g2"],
                name="nth_index",
                dtype="datetime64[ns]",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.nth_index(-1, group_by=group_by, jitted=dict(parallel=True)),
            mask.vbt.signals.nth_index(-1, group_by=group_by, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            mask.vbt.signals.nth_index(-1, group_by=group_by, chunked=True),
            mask.vbt.signals.nth_index(-1, group_by=group_by, chunked=False),
        )

    def test_norm_avg_index(self):
        assert mask["a"].vbt.signals.norm_avg_index() == -0.25
        assert_series_equal(
            mask.vbt.signals.norm_avg_index(),
            pd.Series([-0.25, 0.25, 0.0], index=mask.columns, name="norm_avg_index"),
        )
        assert_series_equal(
            mask.vbt.signals.norm_avg_index(jitted=dict(parallel=True)),
            mask.vbt.signals.norm_avg_index(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            mask.vbt.signals.norm_avg_index(chunked=True),
            mask.vbt.signals.norm_avg_index(chunked=False),
        )
        assert_series_equal(
            mask.vbt.signals.norm_avg_index(group_by=group_by),
            pd.Series([0.0, 0.0], index=["g1", "g2"], name="norm_avg_index"),
        )
        assert_series_equal(
            mask.vbt.signals.norm_avg_index(group_by=group_by, jitted=dict(parallel=True)),
            mask.vbt.signals.norm_avg_index(group_by=group_by, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            mask.vbt.signals.norm_avg_index(group_by=group_by, chunked=True),
            mask.vbt.signals.norm_avg_index(group_by=group_by, chunked=False),
        )

    def test_index_mapped(self):
        mapped = mask.vbt.signals.index_mapped()
        np.testing.assert_array_equal(mapped.values, np.array([0, 3, 1, 4, 2]))
        np.testing.assert_array_equal(mapped.col_arr, np.array([0, 0, 1, 1, 2]))
        np.testing.assert_array_equal(mapped.idx_arr, np.array([0, 3, 1, 4, 2]))
        assert mapped.wrapper == mask.vbt.wrapper

    def test_total(self):
        assert mask["a"].vbt.signals.total() == 2
        assert_series_equal(
            mask.vbt.signals.total(), pd.Series([2, 2, 1], index=mask.columns, name="total")
        )
        assert_series_equal(
            mask.vbt.signals.total(group_by=group_by),
            pd.Series([4, 1], index=["g1", "g2"], name="total"),
        )

    def test_rate(self):
        assert mask["a"].vbt.signals.rate() == 0.4
        assert_series_equal(
            mask.vbt.signals.rate(),
            pd.Series([0.4, 0.4, 0.2], index=mask.columns, name="rate"),
        )
        assert_series_equal(
            mask.vbt.signals.rate(group_by=group_by),
            pd.Series([0.4, 0.2], index=["g1", "g2"], name="rate"),
        )

    def test_total_partitions(self):
        assert mask["a"].vbt.signals.total_partitions() == 2
        assert_series_equal(
            mask.vbt.signals.total_partitions(),
            pd.Series([2, 2, 1], index=mask.columns, name="total_partitions"),
        )
        assert_series_equal(
            mask.vbt.signals.total_partitions(group_by=group_by),
            pd.Series([4, 1], index=["g1", "g2"], name="total_partitions"),
        )

    def test_partition_rate(self):
        assert mask["a"].vbt.signals.partition_rate() == 1.0
        assert_series_equal(
            mask.vbt.signals.partition_rate(),
            pd.Series([1.0, 1.0, 1.0], index=mask.columns, name="partition_rate"),
        )
        assert_series_equal(
            mask.vbt.signals.partition_rate(group_by=group_by),
            pd.Series([1.0, 1.0], index=["g1", "g2"], name="partition_rate"),
        )

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total",
                "Rate [%]",
                "First Index",
                "Last Index",
                "Norm Avg Index [-1, 1]",
                "Distance: Min",
                "Distance: Median",
                "Distance: Max",
                "Total Partitions",
                "Partition Rate [%]",
                "Partition Length: Min",
                "Partition Length: Median",
                "Partition Length: Max",
                "Partition Distance: Min",
                "Partition Distance: Median",
                "Partition Distance: Max",
            ],
            dtype="object",
        )
        assert_series_equal(
            mask.vbt.signals.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    1.6666666666666667,
                    33.333333333333336,
                    pd.Timestamp("2020-01-02 00:00:00"),
                    pd.Timestamp("2020-01-04 00:00:00"),
                    0.0,
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    1.6666666666666667,
                    100.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    40.0,
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-04 00:00:00"),
                    -0.25,
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    2,
                    100.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.stats(column="a", settings=dict(to_timedelta=False)),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    5,
                    2,
                    40.0,
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-04 00:00:00"),
                    -0.25,
                    3.0,
                    3.0,
                    3.0,
                    2,
                    100.0,
                    1.0,
                    1.0,
                    1.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.stats(column="a", settings=dict(target=mask["b"], relation="manyone")),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    40.0,
                    0,
                    0.0,
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-04 00:00:00"),
                    -0.25,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    2,
                    100.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                ],
                index=pd.Index(
                    [
                        "Start Index",
                        "End Index",
                        "Total Duration",
                        "Total",
                        "Rate [%]",
                        "Total Overlapping",
                        "Overlapping Rate [%]",
                        "First Index",
                        "Last Index",
                        "Norm Avg Index [-1, 1]",
                        "Distance <- Target: Min",
                        "Distance <- Target: Median",
                        "Distance <- Target: Max",
                        "Total Partitions",
                        "Partition Rate [%]",
                        "Partition Length: Min",
                        "Partition Length: Median",
                        "Partition Length: Max",
                        "Partition Distance: Min",
                        "Partition Distance: Median",
                        "Partition Distance: Max",
                    ],
                    dtype="object",
                ),
                name="a",
            ),
        )
        assert_series_equal(
            mask.vbt.signals.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    4,
                    40.0,
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    0.0,
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    4,
                    100.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(mask["c"].vbt.signals.stats(), mask.vbt.signals.stats(column="c"))
        assert_series_equal(
            mask["c"].vbt.signals.stats(),
            mask.vbt.signals.stats(column="c", group_by=False),
        )
        assert_series_equal(
            mask.vbt.signals(group_by=group_by)["g2"].stats(),
            mask.vbt.signals(group_by=group_by).stats(column="g2"),
        )
        assert_series_equal(
            mask.vbt.signals(group_by=group_by)["g2"].stats(),
            mask.vbt.signals.stats(column="g2", group_by=group_by),
        )
        stats_df = mask.vbt.signals.stats(agg_func=None)
        assert stats_df.shape == (3, 19)
        assert_index_equal(stats_df.index, mask.vbt.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)


# ############# factory ############# #


class TestFactory:
    def test_entries(self):
        @njit
        def place_nb(c, ts, in_out, n, arg, temp_idx_arr, kw):
            in_out[c.from_i, c.col] = ts[c.from_i, c.col] * n + arg + kw
            c.out[0] = True
            return 0

        MySignals = vbt.SignalFactory(
            mode="entries",
            input_names=["ts2"],
            in_output_names=["in_out2"],
            param_names=["n2"],
        ).with_place_func(
            entry_place_func_nb=place_nb,
            entry_settings=dict(
                pass_inputs=["ts2"],
                pass_in_outputs=["in_out2"],
                pass_params=["n2"],
                pass_kwargs=["temp_idx_arr2", ("kw2", 1000)],
                pass_cache=True,
            ),
            in_output_settings=dict(in_out2=dict(dtype=float_)),
            in_out2=np.nan,
            var_args=True,
        )
        my_sig = MySignals.run(np.arange(5), [1, 0], 100)
        assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(
                np.array(
                    [[True, True], [False, False], [False, False], [False, False], [False, False]]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(
                np.array(
                    [
                        [1100.0, 1100.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )

    def test_exits(self):
        @njit
        def place_nb(c, ts, in_out, n, arg, temp_idx_arr, kw):
            in_out[c.from_i, c.col] = ts[c.from_i, c.col] * n + arg + kw
            c.out[0] = True
            return 0

        MySignals = vbt.SignalFactory(
            mode="exits",
            input_names=["ts2"],
            in_output_names=["in_out2"],
            param_names=["n2"],
        ).with_place_func(
            exit_place_func_nb=place_nb,
            exit_settings=dict(
                pass_inputs=["ts2"],
                pass_in_outputs=["in_out2"],
                pass_params=["n2"],
                pass_kwargs=["temp_idx_arr2", ("kw2", 1000)],
                pass_cache=True,
            ),
            in_output_settings=dict(in_out2=dict(dtype=float_)),
            in_out2=np.nan,
            var_args=True,
        )
        e = np.array([True, False, True, False, True])
        my_sig = MySignals.run(e, np.arange(5), [1, 0], 100)
        assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(
                np.array(
                    [[True, True], [False, False], [True, True], [False, False], [True, True]]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(
                np.array(
                    [[False, False], [True, True], [False, False], [True, True], [False, False]]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan],
                        [1101.0, 1100.0],
                        [np.nan, np.nan],
                        [1103.0, 1100.0],
                        [np.nan, np.nan],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        e = np.array([True, False, False, True, False, False])
        my_sig = MySignals.run(e, np.arange(6), [1, 0], 100, wait=2)
        assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(
                np.array(
                    [
                        [True, True],
                        [False, False],
                        [False, False],
                        [True, True],
                        [False, False],
                        [False, False],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, True],
                        [False, False],
                        [False, False],
                        [True, True],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [1102.0, 1100.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [1105.0, 1100.0],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )

    def test_chain(self):
        @njit
        def place_nb(c, ts, in_out, n, arg, temp_idx_arr, kw):
            in_out[c.from_i, c.col] = ts[c.from_i, c.col] * n + arg + kw
            c.out[0] = True
            return 0

        MySignals = vbt.SignalFactory(
            mode="chain",
            input_names=["ts2"],
            in_output_names=["in_out2"],
            param_names=["n2"],
        ).with_place_func(
            exit_place_func_nb=place_nb,
            exit_settings=dict(
                pass_inputs=["ts2"],
                pass_in_outputs=["in_out2"],
                pass_params=["n2"],
                pass_kwargs=["temp_idx_arr2", ("kw2", 1000)],
                pass_cache=True,
            ),
            in_output_settings=dict(in_out2=dict(dtype=float_)),
            in_out2=np.nan,
            var_args=True,
        )
        e = np.array([True, True, True, True, True])
        my_sig = MySignals.run(e, np.arange(5), [1, 0], 100)
        assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(
                np.array([[True, True], [True, True], [True, True], [True, True], [True, True]]),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.new_entries,
            pd.DataFrame(
                np.array(
                    [[True, True], [False, False], [True, True], [False, False], [True, True]]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(
                np.array(
                    [[False, False], [True, True], [False, False], [True, True], [False, False]]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan],
                        [1101.0, 1100.0],
                        [np.nan, np.nan],
                        [1103.0, 1100.0],
                        [np.nan, np.nan],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        e = np.array([True, True, True, True, True, True])
        my_sig = MySignals.run(e, np.arange(6), [1, 0], 100, wait=2)
        assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(
                np.array(
                    [
                        [True, True],
                        [True, True],
                        [True, True],
                        [True, True],
                        [True, True],
                        [True, True],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.new_entries,
            pd.DataFrame(
                np.array(
                    [
                        [True, True],
                        [False, False],
                        [False, False],
                        [True, True],
                        [False, False],
                        [False, False],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, True],
                        [False, False],
                        [False, False],
                        [True, True],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )
        assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [1102.0, 1100.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [1105.0, 1100.0],
                    ]
                ),
                columns=pd.Index([1, 0], dtype="int64", name="custom_n2"),
            ),
        )

    def test_both(self):
        @njit
        def cache_nb(ts1, ts2, in_out1, in_out2, n1, n2, arg0, temp_idx_arr0, kw0):
            return arg0

        @njit
        def place_nb(c, ts, in_out, n, arg, temp_idx_arr, kw, cache):
            in_out[c.from_i, c.col] = ts[c.from_i, c.col] * n + arg + kw + cache
            c.out[0] = True
            return 0

        MySignals = vbt.SignalFactory(
            input_names=["ts1", "ts2"],
            in_output_names=["in_out1", "in_out2"],
            param_names=["n1", "n2"],
        ).with_place_func(
            cache_func=cache_nb,
            cache_settings=dict(
                pass_inputs=["ts1", "ts2"],
                pass_in_outputs=["in_out1", "in_out2"],
                pass_params=["n1", "n2"],
                pass_kwargs=["temp_idx_arr0", ("kw0", 1000)],
            ),
            entry_place_func_nb=place_nb,
            entry_settings=dict(
                pass_inputs=["ts1"],
                pass_in_outputs=["in_out1"],
                pass_params=["n1"],
                pass_kwargs=["temp_idx_arr1", ("kw1", 1000)],
                pass_cache=True,
            ),
            exit_place_func_nb=place_nb,
            exit_settings=dict(
                pass_inputs=["ts2"],
                pass_in_outputs=["in_out2"],
                pass_params=["n2"],
                pass_kwargs=["temp_idx_arr2", ("kw2", 1000)],
                pass_cache=True,
            ),
            in_output_settings=dict(in_out1=dict(dtype=float_), in_out2=dict(dtype=float_)),
            in_out1=np.nan,
            in_out2=np.nan,
            var_args=True,
            require_input_shape=False,
        )
        my_sig = MySignals.run(
            np.arange(5),
            np.arange(5),
            [0, 1],
            [1, 0],
            cache_args=(0,),
            entry_place_args=(100,),
            exit_place_args=(100,),
        )
        assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(
                np.array(
                    [[True, True], [False, False], [True, True], [False, False], [True, True]]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1), (1, 0)], names=["custom_n1", "custom_n2"]
                ),
            ),
        )
        assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False],
                        [True, True],
                        [False, False],
                        [True, True],
                        [False, False],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1), (1, 0)], names=["custom_n1", "custom_n2"]
                ),
            ),
        )
        assert_frame_equal(
            my_sig.in_out1,
            pd.DataFrame(
                np.array(
                    [
                        [1100.0, 1100.0],
                        [np.nan, np.nan],
                        [1100.0, 1102.0],
                        [np.nan, np.nan],
                        [1100.0, 1104.0],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1), (1, 0)], names=["custom_n1", "custom_n2"]
                ),
            ),
        )
        assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan],
                        [1101.0, 1100.0],
                        [np.nan, np.nan],
                        [1103.0, 1100.0],
                        [np.nan, np.nan],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1), (1, 0)], names=["custom_n1", "custom_n2"]
                ),
            ),
        )
        my_sig = MySignals.run(
            np.arange(7),
            np.arange(7),
            [0, 1],
            [1, 0],
            cache_args=(0,),
            entry_place_args=(100,),
            exit_place_args=(100,),
            entry_kwargs=dict(wait=2),
            exit_kwargs=dict(wait=2),
        )
        assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(
                np.array(
                    [
                        [True, True],
                        [False, False],
                        [False, False],
                        [False, False],
                        [True, True],
                        [False, False],
                        [False, False],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1), (1, 0)], names=["custom_n1", "custom_n2"]
                ),
            ),
        )
        assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False],
                        [False, False],
                        [True, True],
                        [False, False],
                        [False, False],
                        [False, False],
                        [True, True],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1), (1, 0)], names=["custom_n1", "custom_n2"]
                ),
            ),
        )
        assert_frame_equal(
            my_sig.in_out1,
            pd.DataFrame(
                np.array(
                    [
                        [1100.0, 1100.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [1100.0, 1104.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1), (1, 0)], names=["custom_n1", "custom_n2"]
                ),
            ),
        )
        assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [1102.0, 1100.0],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                        [1106.0, 1100.0],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0, 1), (1, 0)], names=["custom_n1", "custom_n2"]
                ),
            ),
        )


# ############# generators ############# #


class TestGenerators:
    def test_RAND(self):
        rand = vbt.RAND.run(n=1, input_shape=(6,), seed=seed)
        assert_series_equal(
            rand.entries,
            pd.Series(np.array([False, False, False, True, False, False]), name=1),
        )
        rand = vbt.RAND.run(n=[1, 2, 3], input_shape=(6,), seed=seed)
        assert_frame_equal(
            rand.entries,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, True],
                        [False, True, True],
                        [True, False, False],
                        [False, True, True],
                        [False, False, False],
                    ]
                ),
                columns=pd.Index([1, 2, 3], dtype="int64", name="rand_n"),
            ),
        )
        rand = vbt.RAND.run(n=[np.array([1, 2]), np.array([3, 4])], input_shape=(8, 2), seed=seed)
        assert_frame_equal(
            rand.entries,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, True],
                        [False, False, True, True],
                        [False, True, False, False],
                        [False, True, False, True],
                        [False, False, False, False],
                        [True, False, True, True],
                        [False, False, True, False],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(1, 0), (2, 1), (3, 0), (4, 1)], names=["rand_n", None]
                ),
            ),
        )

    def test_RANDX(self):
        randx = vbt.RANDX.run(mask, seed=seed)
        assert_frame_equal(
            randx.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, False, False],
                        [False, True, True],
                        [True, False, False],
                    ]
                ),
                columns=mask.columns,
                index=mask.index,
            ),
        )

    def test_RANDNX(self):
        randnx = vbt.RANDNX.run(n=1, input_shape=(6,), seed=seed)
        assert_series_equal(
            randnx.entries,
            pd.Series(np.array([False, False, False, True, False, False]), name=1),
        )
        assert_series_equal(
            randnx.exits,
            pd.Series(np.array([False, False, False, False, True, False]), name=1),
        )
        randnx = vbt.RANDNX.run(n=[1, 2, 3], input_shape=(6,), seed=seed)
        assert_frame_equal(
            randnx.entries,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, True],
                        [False, True, False],
                        [False, False, True],
                        [True, True, False],
                        [False, False, True],
                        [False, False, False],
                    ]
                ),
                columns=pd.Index([1, 2, 3], dtype="int64", name="randnx_n"),
            ),
        )
        assert_frame_equal(
            randnx.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, True],
                        [False, True, False],
                        [False, False, True],
                        [True, True, False],
                        [False, False, True],
                    ]
                ),
                columns=pd.Index([1, 2, 3], dtype="int64", name="randnx_n"),
            ),
        )
        randnx = vbt.RANDNX.run(
            n=[np.array([1, 2]), np.array([3, 4])], input_shape=(8, 2), seed=seed
        )
        assert_frame_equal(
            randnx.entries,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False, True],
                        [False, False, True, False],
                        [False, True, False, True],
                        [True, False, True, False],
                        [False, False, False, True],
                        [False, False, False, False],
                        [False, True, True, True],
                        [False, False, False, False],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(1, 0), (2, 1), (3, 0), (4, 1)], names=["randnx_n", None]
                ),
            ),
        )
        assert_frame_equal(
            randnx.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, True],
                        [False, False, True, False],
                        [False, False, False, True],
                        [False, True, True, False],
                        [False, False, False, True],
                        [True, False, False, False],
                        [False, True, True, True],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(1, 0), (2, 1), (3, 0), (4, 1)], names=["randnx_n", None]
                ),
            ),
        )

    def test_RPROB(self):
        rprob = vbt.RPROB.run(prob=1, input_shape=(5,), seed=seed)
        assert_series_equal(
            rprob.entries, pd.Series(np.array([True, True, True, True, True]), name=1)
        )
        rprob = vbt.RPROB.run(prob=[0.0, 0.5, 1.0], input_shape=(5,), seed=seed)
        assert_frame_equal(
            rprob.entries,
            pd.DataFrame(
                np.array(
                    [
                        [False, True, True],
                        [False, True, True],
                        [False, False, True],
                        [False, False, True],
                        [False, False, True],
                    ]
                ),
                columns=pd.Index([0.0, 0.5, 1.0], dtype="float64", name="rprob_prob"),
            ),
        )
        rprob = vbt.RPROB.run(
            prob=[np.array([[0.0, 0.25]]), np.array([[0.75, 1.0]])], input_shape=(5, 2), seed=seed
        )
        assert_frame_equal(
            rprob.entries,
            pd.DataFrame(
                np.array(
                    [
                        [False, True, True, True],
                        [False, True, False, True],
                        [False, False, False, True],
                        [False, False, True, True],
                        [False, False, True, True],
                    ]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [("array_0", 0), ("array_0", 1), ("array_1", 0), ("array_1", 1)],
                    names=["rprob_prob", None],
                ),
            ),
        )

    def test_RPROBX(self):
        rprobx = vbt.RPROBX.run(mask, prob=[0.0, 0.5, 1.0], seed=seed)
        assert_frame_equal(
            rprobx.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False, False, False, False, False, False, False],
                        [False, False, False, False, False, False, True, False, False],
                        [False, False, False, False, True, False, False, True, False],
                        [False, False, False, False, False, False, False, False, True],
                        [False, False, False, False, False, False, True, False, False],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0.0, "a"),
                        (0.0, "b"),
                        (0.0, "c"),
                        (0.5, "a"),
                        (0.5, "b"),
                        (0.5, "c"),
                        (1.0, "a"),
                        (1.0, "b"),
                        (1.0, "c"),
                    ],
                    names=["rprobx_prob", None],
                ),
            ),
        )

    def test_RPROBCX(self):
        rprobcx = vbt.RPROBCX.run(mask, prob=[0.0, 0.5, 1.0], seed=seed)
        assert_frame_equal(
            rprobcx.new_entries,
            pd.DataFrame(
                np.array(
                    [
                        [True, False, False, True, False, False, True, False, False],
                        [False, True, False, False, True, False, False, True, False],
                        [False, False, True, False, False, True, False, False, True],
                        [False, False, False, True, False, False, True, False, False],
                        [False, False, False, False, True, False, False, True, False],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0.0, "a"),
                        (0.0, "b"),
                        (0.0, "c"),
                        (0.5, "a"),
                        (0.5, "b"),
                        (0.5, "c"),
                        (1.0, "a"),
                        (1.0, "b"),
                        (1.0, "c"),
                    ],
                    names=["rprobcx_prob", None],
                ),
            ),
        )
        assert_frame_equal(
            rprobcx.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False, False, False, False, False, False, False],
                        [False, False, False, False, False, False, True, False, False],
                        [False, False, False, True, False, False, False, True, False],
                        [False, False, False, False, True, True, False, False, True],
                        [False, False, False, False, False, False, True, False, False],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0.0, "a"),
                        (0.0, "b"),
                        (0.0, "c"),
                        (0.5, "a"),
                        (0.5, "b"),
                        (0.5, "c"),
                        (1.0, "a"),
                        (1.0, "b"),
                        (1.0, "c"),
                    ],
                    names=["rprobcx_prob", None],
                ),
            ),
        )

    def test_RPROBNX(self):
        rprobnx = vbt.RPROBNX.run(entry_prob=1.0, exit_prob=1.0, input_shape=(5,), seed=seed)
        assert_series_equal(
            rprobnx.entries,
            pd.Series(np.array([True, False, True, False, True]), name=(1.0, 1.0)),
        )
        assert_series_equal(
            rprobnx.exits,
            pd.Series(np.array([False, True, False, True, False]), name=(1.0, 1.0)),
        )
        rprobnx = vbt.RPROBNX.run(
            entry_prob=np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
            exit_prob=np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
            input_shape=(5,),
            seed=seed,
        )
        assert_series_equal(
            rprobnx.entries,
            pd.Series(np.array([True, False, True, False, True]), name=("array", "array")),
        )
        assert_series_equal(
            rprobnx.exits,
            pd.Series(np.array([False, True, False, True, False]), name=("array", "array")),
        )
        rprobnx = vbt.RPROBNX.run(
            entry_prob=[0.5, 1.0], exit_prob=[1.0, 0.5], input_shape=(5,), seed=seed
        )
        assert_frame_equal(
            rprobnx.entries,
            pd.DataFrame(
                np.array(
                    [[True, True], [False, False], [False, True], [False, False], [True, False]]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0.5, 1.0), (1.0, 0.5)],
                    names=["rprobnx_entry_prob", "rprobnx_exit_prob"],
                ),
            ),
        )
        assert_frame_equal(
            rprobnx.exits,
            pd.DataFrame(
                np.array(
                    [[False, False], [True, True], [False, False], [False, False], [False, False]]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [(0.5, 1.0), (1.0, 0.5)],
                    names=["rprobnx_entry_prob", "rprobnx_exit_prob"],
                ),
            ),
        )

    def test_STX(self):
        stx = vbt.STX.run(mask, ts, 0.1)
        assert_frame_equal(
            stx.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, True, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, "a"), (0.1, "b"), (0.1, "c")], names=["stx_stop", None]
                ),
            ),
        )
        assert_frame_equal(
            stx.stop_ts,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [1.1, np.nan, np.nan],
                        [np.nan, 2.2, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, "a"), (0.1, "b"), (0.1, "c")], names=["stx_stop", None]
                ),
            ),
        )
        stx = vbt.STX.run(mask, ts, np.array([0.1, 0.1, -0.1, -0.1, -0.1])[:, None])
        assert_frame_equal(
            stx.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [True, False, False],
                        [False, True, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [("array", "a"), ("array", "b"), ("array", "c")],
                    names=["stx_stop", None],
                ),
            ),
        )
        stx = vbt.STX.run(mask, ts, [0.1, 0.1, -0.1, -0.1], trailing=[False, True, False, True])
        assert_frame_equal(
            stx.exits,
            pd.DataFrame(
                np.array(
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
                        ],
                        [
                            True,
                            False,
                            False,
                            True,
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
                            True,
                            False,
                            False,
                            True,
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
                            True,
                            False,
                            True,
                            True,
                        ],
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                        ],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0.1, False, "a"),
                        (0.1, False, "b"),
                        (0.1, False, "c"),
                        (0.1, True, "a"),
                        (0.1, True, "b"),
                        (0.1, True, "c"),
                        (-0.1, False, "a"),
                        (-0.1, False, "b"),
                        (-0.1, False, "c"),
                        (-0.1, True, "a"),
                        (-0.1, True, "b"),
                        (-0.1, True, "c"),
                    ],
                    names=["stx_stop", "stx_trailing", None],
                ),
            ),
        )

    def test_STCX(self):
        stcx = vbt.STCX.run(mask, ts, [0.1, 0.1, -0.1, -0.1], trailing=[False, True, False, True])
        target_columns = pd.MultiIndex.from_tuples(
            [
                (0.1, False, "a"),
                (0.1, False, "b"),
                (0.1, False, "c"),
                (0.1, True, "a"),
                (0.1, True, "b"),
                (0.1, True, "c"),
                (-0.1, False, "a"),
                (-0.1, False, "b"),
                (-0.1, False, "c"),
                (-0.1, True, "a"),
                (-0.1, True, "b"),
                (-0.1, True, "c"),
            ],
            names=["stcx_stop", "stcx_trailing", None],
        )
        assert_frame_equal(
            stcx.new_entries,
            pd.DataFrame(
                np.array(
                    [
                        [
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                        ],
                        [
                            False,
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                            True,
                            False,
                        ],
                        [
                            False,
                            False,
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                            True,
                        ],
                        [
                            True,
                            False,
                            False,
                            True,
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
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            False,
                        ],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )
        assert_frame_equal(
            stcx.stop_ts,
            pd.DataFrame(
                np.array(
                    [
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                        [
                            1.1,
                            np.nan,
                            np.nan,
                            1.1,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                        [
                            np.nan,
                            2.2,
                            np.nan,
                            np.nan,
                            2.2,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            2.7,
                            2.7,
                            2.7,
                            2.7,
                        ],
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            1.8,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )
        assert_frame_equal(
            stcx.exits,
            pd.DataFrame(
                np.array(
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
                        ],
                        [
                            True,
                            False,
                            False,
                            True,
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
                            True,
                            False,
                            False,
                            True,
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
                            True,
                            True,
                            True,
                            True,
                        ],
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                            False,
                            False,
                            False,
                            False,
                        ],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )

    def test_OHLCSTX(self):
        ohlcstx = vbt.OHLCSTX.run(
            mask,
            price["open"],
            price["open"],
            price["high"],
            price["low"],
            price["close"],
            sl_stop=0.1,
            is_entry_open=False,
        )
        assert_frame_equal(
            ohlcstx.exits,
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                        [False, False, True],
                        [True, False, False],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, "a"), (0.1, "b"), (0.1, "c")],
                    names=["ohlcstx_sl_stop", None],
                ),
            ),
        )
        assert_frame_equal(
            ohlcstx.stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, 10.8],
                        [9.9, np.nan, np.nan],
                    ]
                ),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, "a"), (0.1, "b"), (0.1, "c")],
                    names=["ohlcstx_sl_stop", None],
                ),
            ),
        )
        assert_frame_equal(
            ohlcstx.stop_type,
            pd.DataFrame(
                np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, 0], [0, -1, -1]]),
                index=mask.index,
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, "a"), (0.1, "b"), (0.1, "c")],
                    names=["ohlcstx_sl_stop", None],
                ),
            ),
        )
        ohlcstx = vbt.OHLCSTX.run(
            mask,
            price["open"],
            price["open"],
            price["high"],
            price["low"],
            price["close"],
            sl_stop=[0.1, np.nan, np.nan, np.nan],
            tsl_th=[np.nan, np.nan, 0.1, np.nan],
            tsl_stop=[np.nan, 0.1, 0.05, np.nan],
            tp_stop=[np.nan, np.nan, np.nan, 0.1],
            is_entry_open=False,
        )
        target_columns = pd.MultiIndex.from_tuples(
            [
                (0.1, np.nan, np.nan, np.nan, "a"),
                (0.1, np.nan, np.nan, np.nan, "b"),
                (0.1, np.nan, np.nan, np.nan, "c"),
                (np.nan, np.nan, 0.1, np.nan, "a"),
                (np.nan, np.nan, 0.1, np.nan, "b"),
                (np.nan, np.nan, 0.1, np.nan, "c"),
                (np.nan, 0.1, 0.05, np.nan, "a"),
                (np.nan, 0.1, 0.05, np.nan, "b"),
                (np.nan, 0.1, 0.05, np.nan, "c"),
                (np.nan, np.nan, np.nan, 0.1, "a"),
                (np.nan, np.nan, np.nan, 0.1, "b"),
                (np.nan, np.nan, np.nan, 0.1, "c"),
            ],
            names=[
                "ohlcstx_sl_stop",
                "ohlcstx_tsl_th",
                "ohlcstx_tsl_stop",
                "ohlcstx_tp_stop",
                None,
            ],
        )
        assert_frame_equal(
            ohlcstx.exits,
            pd.DataFrame(
                np.array(
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
                            True,
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
                            True,
                            False,
                            False,
                            False,
                            True,
                            False,
                        ],
                        [
                            False,
                            False,
                            True,
                            False,
                            True,
                            True,
                            False,
                            True,
                            False,
                            False,
                            False,
                            False,
                        ],
                        [
                            True,
                            False,
                            False,
                            True,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )
        assert_frame_equal(
            ohlcstx.stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            11.0,
                            np.nan,
                            np.nan,
                        ],
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            11.399999999999999,
                            np.nan,
                            np.nan,
                            np.nan,
                            12.100000000000001,
                            np.nan,
                        ],
                        [
                            np.nan,
                            np.nan,
                            10.8,
                            np.nan,
                            11.700000000000001,
                            10.8,
                            np.nan,
                            12.35,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                        [
                            9.9,
                            np.nan,
                            np.nan,
                            9.9,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )
        assert_frame_equal(
            ohlcstx.stop_type,
            pd.DataFrame(
                np.array(
                    [
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1],
                        [-1, -1, -1, -1, -1, -1, 2, -1, -1, -1, 3, -1],
                        [-1, -1, 0, -1, 1, 1, -1, 2, -1, -1, -1, -1],
                        [0, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )
        np.testing.assert_array_equal(
            vbt.OHLCSTX.run(
                mask,
                price["open"],
                price["open"],
                price["high"],
                price["low"],
                price["close"],
                sl_stop=[0.1, np.nan],
                tp_stop=[np.nan, 0.1],
                reverse=False,
                is_entry_open=False,
            ).exits.values,
            vbt.OHLCSTX.run(
                mask,
                price["open"],
                price["open"],
                price["high"],
                price["low"],
                price["close"],
                sl_stop=[np.nan, 0.1],
                tp_stop=[0.1, np.nan],
                reverse=True,
                is_entry_open=False,
            ).exits.values,
        )

    def test_OHLCSTCX(self):
        ohlcstcx = vbt.OHLCSTCX.run(
            mask,
            price["open"],
            price["open"],
            price["high"],
            price["low"],
            price["close"],
            sl_stop=[0.1, np.nan, np.nan, np.nan],
            tsl_th=[np.nan, np.nan, 0.1, np.nan],
            tsl_stop=[np.nan, 0.1, 0.05, np.nan],
            tp_stop=[np.nan, np.nan, np.nan, 0.1],
            is_entry_open=False,
        )
        target_columns = pd.MultiIndex.from_tuples(
            [
                (0.1, np.nan, np.nan, np.nan, "a"),
                (0.1, np.nan, np.nan, np.nan, "b"),
                (0.1, np.nan, np.nan, np.nan, "c"),
                (np.nan, np.nan, 0.1, np.nan, "a"),
                (np.nan, np.nan, 0.1, np.nan, "b"),
                (np.nan, np.nan, 0.1, np.nan, "c"),
                (np.nan, 0.1, 0.05, np.nan, "a"),
                (np.nan, 0.1, 0.05, np.nan, "b"),
                (np.nan, 0.1, 0.05, np.nan, "c"),
                (np.nan, np.nan, np.nan, 0.1, "a"),
                (np.nan, np.nan, np.nan, 0.1, "b"),
                (np.nan, np.nan, np.nan, 0.1, "c"),
            ],
            names=[
                "ohlcstcx_sl_stop",
                "ohlcstcx_tsl_th",
                "ohlcstcx_tsl_stop",
                "ohlcstcx_tp_stop",
                None,
            ],
        )
        assert_frame_equal(
            ohlcstcx.exits,
            pd.DataFrame(
                np.array(
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
                            True,
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
                            True,
                            False,
                            False,
                            False,
                            True,
                            False,
                        ],
                        [
                            False,
                            False,
                            True,
                            True,
                            True,
                            True,
                            False,
                            True,
                            False,
                            False,
                            False,
                            False,
                        ],
                        [
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
                        ],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )
        assert_frame_equal(
            ohlcstcx.stop_price,
            pd.DataFrame(
                np.array(
                    [
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            11.0,
                            np.nan,
                            np.nan,
                        ],
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            11.399999999999999,
                            np.nan,
                            np.nan,
                            np.nan,
                            12.100000000000001,
                            np.nan,
                        ],
                        [
                            np.nan,
                            np.nan,
                            10.8,
                            11.700000000000001,
                            11.700000000000001,
                            10.8,
                            np.nan,
                            12.35,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                        [
                            9.0,
                            9.9,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )
        assert_frame_equal(
            ohlcstcx.stop_type,
            pd.DataFrame(
                np.array(
                    [
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1],
                        [-1, -1, -1, -1, -1, -1, 2, -1, -1, -1, 3, -1],
                        [-1, -1, 0, 1, 1, 1, -1, 2, -1, -1, -1, -1],
                        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    ]
                ),
                index=mask.index,
                columns=target_columns,
            ),
        )
