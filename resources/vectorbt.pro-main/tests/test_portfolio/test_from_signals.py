import os

import pytest
import vectorbtpro as vbt
from numba import njit
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.enums import *

from tests.utils import *

seed = 42

day_dt = np.timedelta64(86400000000000)

price = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2020", periods=5))
price_wide = price.vbt.tile(3, keys=["a", "b", "c"])


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# from_signals ############# #

entries = pd.Series([True, True, True, False, False], index=price.index)
entries_wide = entries.vbt.tile(3, keys=["a", "b", "c"])

exits = pd.Series([False, False, True, True, True], index=price.index)
exits_wide = exits.vbt.tile(3, keys=["a", "b", "c"])


def from_signals_both(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, entries, exits, direction="both", **kwargs)


def from_signals_longonly(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, entries, exits, direction="longonly", **kwargs)


def from_signals_shortonly(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, entries, exits, direction="shortonly", **kwargs)


def from_ls_signals_both(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(
        close, entries, False, short_entries=exits, short_exits=False, **kwargs
    )


def from_ls_signals_longonly(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(
        close, entries, exits, short_entries=False, short_exits=False, **kwargs
    )


def from_ls_signals_shortonly(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(
        close, False, False, short_entries=entries, short_exits=exits, **kwargs
    )


@njit
def adjust_func_nb(c):
    pass


@njit
def signal_func_nb(c, long_num_arr, short_num_arr):
    long_num = nb.select_nb(c, long_num_arr)
    short_num = nb.select_nb(c, short_num_arr)
    is_long_entry = long_num > 0
    is_long_exit = long_num < 0
    is_short_entry = short_num > 0
    is_short_exit = short_num < 0
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit


class TestFromSignals:
    def test_data(self):
        data = vbt.RandomOHLCData.pull(
            [0, 1],
            start="2020-01-01",
            end="2020-02-01",
            tick_freq="1h",
            freq="1d",
            seed=42,
        )
        pf = vbt.Portfolio.from_signals(data)
        assert pf.open is not None
        assert pf.high is not None
        assert pf.low is not None
        assert pf.close is not None
        pf = vbt.Portfolio.from_signals(data.get("Close"))
        assert pf.open is None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_signals(data[["Open", "Close"]])
        assert pf.open is not None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_signals(data["Close"])
        assert pf.open is None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_signals(data["Close"], open=data.get("Open"))
        assert pf.open is not None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None

    @pytest.mark.parametrize("test_ls", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_one_column(self, test_ls, test_flexible):
        if test_ls:
            if test_flexible:
                _from_signals_both = partial(from_ls_signals_both, adjust_func_nb=adjust_func_nb)
                _from_signals_longonly = partial(
                    from_ls_signals_longonly, adjust_func_nb=adjust_func_nb
                )
                _from_signals_shortonly = partial(
                    from_ls_signals_shortonly, adjust_func_nb=adjust_func_nb
                )
            else:
                _from_signals_both = from_ls_signals_both
                _from_signals_longonly = from_ls_signals_longonly
                _from_signals_shortonly = from_ls_signals_shortonly
        else:
            if test_flexible:
                _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
                _from_signals_longonly = partial(
                    from_signals_longonly, adjust_func_nb=adjust_func_nb
                )
                _from_signals_shortonly = partial(
                    from_signals_shortonly, adjust_func_nb=adjust_func_nb
                )
            else:
                _from_signals_both = from_signals_both
                _from_signals_longonly = from_signals_longonly
                _from_signals_shortonly = from_signals_shortonly
        assert_records_close(
            _from_signals_both().order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly().order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly().order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        pf = _from_signals_both()
        assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
            ),
        )
        assert_index_equal(pf.wrapper.columns, pd.Index([0], dtype="int64"))
        assert pf.wrapper.ndim == 1
        assert pf.wrapper.freq == day_dt
        assert pf.wrapper.grouper.group_by is None

    @pytest.mark.parametrize("test_ls", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_multiple_columns(self, test_ls, test_flexible):
        if test_ls:
            if test_flexible:
                _from_signals_both = partial(from_ls_signals_both, adjust_func_nb=adjust_func_nb)
                _from_signals_longonly = partial(
                    from_ls_signals_longonly, adjust_func_nb=adjust_func_nb
                )
                _from_signals_shortonly = partial(
                    from_ls_signals_shortonly, adjust_func_nb=adjust_func_nb
                )
            else:
                _from_signals_both = from_ls_signals_both
                _from_signals_longonly = from_ls_signals_longonly
                _from_signals_shortonly = from_ls_signals_shortonly
        else:
            if test_flexible:
                _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
                _from_signals_longonly = partial(
                    from_signals_longonly, adjust_func_nb=adjust_func_nb
                )
                _from_signals_shortonly = partial(
                    from_signals_shortonly, adjust_func_nb=adjust_func_nb
                )
            else:
                _from_signals_both = from_signals_both
                _from_signals_longonly = from_signals_longonly
                _from_signals_shortonly = from_signals_shortonly
        assert_records_close(
            _from_signals_both(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        pf = _from_signals_both(close=price_wide)
        assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
            ),
        )
        assert_index_equal(pf.wrapper.columns, pd.Index(["a", "b", "c"], dtype="object"))
        assert pf.wrapper.ndim == 2
        assert pf.wrapper.freq == day_dt
        assert pf.wrapper.grouper.group_by is None

    def test_custom_signal_func(self):
        pf_base = vbt.Portfolio.from_signals(
            pd.Series([1, 2, 3, 4, 5]),
            entries=pd.Series([True, False, False, False, False]),
            exits=pd.Series([False, False, True, False, False]),
            short_entries=pd.Series([False, True, False, True, False]),
            short_exits=pd.Series([False, False, False, False, True]),
            size=1,
            upon_opposite_entry="ignore",
        )
        pf = vbt.Portfolio.from_signals(
            pd.Series([1, 2, 3, 4, 5]),
            signal_func_nb=signal_func_nb,
            signal_args=(vbt.Rep("long_num_arr"), vbt.Rep("short_num_arr")),
            broadcast_named_args=dict(
                long_num_arr=pd.Series([1, 0, -1, 0, 0]),
                short_num_arr=pd.Series([0, 1, 0, 1, -1]),
            ),
            size=1,
            upon_opposite_entry="ignore",
        )
        assert_records_close(pf_base.order_records, pf.order_records)

    def test_pos_info(self):
        out = np.empty((5, 3), dtype=vbt.pf_enums.trade_dt)

        @njit
        def post_segment_func_nb(c, out):
            for col in range(c.from_col, c.to_col):
                out[c.i, col] = c.last_pos_info[col]

        _ = from_signals_both(
            close=pd.concat(
                (
                    price.rename("a"),
                    price.rename("b"),
                    price.rename("c"),
                ),
                axis=1,
            ),
            entries=pd.concat(
                (
                    entries.rename("a"),
                    pd.Series(np.roll(entries.values, 1), index=entries.index, name="b"),
                    pd.Series(np.roll(entries.values, 2), index=entries.index, name="c"),
                ),
                axis=1,
            ),
            exits=pd.concat(
                (
                    exits.rename("a"),
                    pd.Series(np.roll(exits.values, 1), index=exits.index, name="b"),
                    pd.Series(np.roll(exits.values, 2), index=exits.index, name="c"),
                ),
                axis=1,
            ),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(out,),
        )
        assert_records_close(
            out,
            np.array(
                [
                    [
                        (0, 0, 100.0, 0, 0, 1.0, 0.0, -1, -1, np.nan, 0.0, 0.0, 0.0, 0, 0, 0),
                        (0, 1, 100.0, 0, 0, 1.0, 0.0, -1, -1, np.nan, 0.0, 0.0, 0.0, 1, 0, 0),
                        (0, 2, 100.0, 0, 0, 1.0, 0.0, -1, -1, np.nan, 0.0, 0.0, 0.0, 1, 0, 0),
                    ],
                    [
                        (0, 0, 100.0, 0, 0, 1.0, 0.0, -1, -1, np.nan, 0.0, 100.0, 1.0, 0, 0, 0),
                        (0, 1, 100.0, 0, 0, 1.0, 0.0, 1, 1, 2.0, 0.0, -100.0, -1.0, 1, 1, 0),
                        (0, 2, 100.0, 0, 0, 1.0, 0.0, -1, -1, np.nan, 0.0, -100.0, -1.0, 1, 0, 0),
                    ],
                    [
                        (0, 0, 100.0, 0, 0, 1.0, 0.0, -1, -1, np.nan, 0.0, 200.0, 2.0, 0, 0, 0),
                        (0, 1, 100.0, 0, 0, 1.0, 0.0, 1, 1, 2.0, 0.0, -100.0, -1.0, 1, 1, 0),
                        (0, 2, 100.0, 0, 0, 1.0, 0.0, 1, -1, 3.0, 0.0, -200.0, -2.0, 1, 0, 0),
                    ],
                    [
                        (1, 0, 100.0, 1, 3, 4.0, 0.0, -1, -1, np.nan, 0.0, 0.0, 0.0, 1, 0, 1),
                        (0, 1, 100.0, 0, 0, 1.0, 0.0, 1, 1, 2.0, 0.0, -100.0, -1.0, 1, 1, 0),
                        (
                            0,
                            2,
                            100.0,
                            0,
                            0,
                            1.0,
                            0.0,
                            1,
                            -1,
                            3.0,
                            0.0,
                            -233.33333333333331,
                            -2.333333333333333,
                            1,
                            0,
                            0,
                        ),
                    ],
                    [
                        (1, 0, 100.0, 1, 3, 4.0, 0.0, -1, -1, np.nan, 0.0, -100.0, -0.25, 1, 0, 1),
                        (0, 1, 100.0, 0, 0, 1.0, 0.0, 1, 1, 2.0, 0.0, -100.0, -1.0, 1, 1, 0),
                        (
                            0,
                            2,
                            100.0,
                            0,
                            0,
                            1.0,
                            0.0,
                            1,
                            -1,
                            3.0,
                            0.0,
                            -266.66666666666663,
                            -2.666666666666666,
                            1,
                            0,
                            0,
                        ),
                    ],
                ],
                dtype=vbt.pf_enums.trade_dt,
            ),
        )

    def test_ladder(self):
        @njit
        def adjust_func_nb(c, tp_stops, tp_sizes):
            tp_info = c.last_tp_info[c.col]
            if tp_info["step_idx"] == c.i - 1:
                tp_info["ladder"] = True
                tp_info["stop"] = tp_stops[tp_info["step"]]
                tp_info["exit_size"] = tp_sizes[tp_info["step"]]
                tp_info["exit_size_type"] = vbt.pf_enums.SizeType.Percent

        assert_records_close(
            from_signals_both(
                adjust_func_nb=adjust_func_nb,
                adjust_args=(np.array([1.0, 2.0]), np.array([0.5, 1.0])),
                use_stops=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 50.0, 2.0, 0.0, 1, 0, 3),
                    (2, 0, 0, 2, 2, 50.0, 3.0, 0.0, 1, 0, 3),
                    (3, 0, 3, 3, 3, 62.5, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_amount(self):
        assert_records_close(
            from_signals_both(size=[[0, 1, np.inf]], size_type="amount").order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=[[0, 1, np.inf]], size_type="amount").order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=[[0, 1, np.inf]], size_type="amount").order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_value(self):
        assert_records_close(
            from_signals_both(size=[[0, 1, np.inf]], size_type="value").order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.25, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=[[0, 1, np.inf]], size_type="value").order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=[[0, 1, np.inf]], size_type="value").order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_percent(self):
        assert_records_close(
            from_signals_both(size=0.5, size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 81.25, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                size=0.5, size_type="percent", upon_opposite_entry="close"
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 1, 0, -1),
                    (2, 0, 4, 4, 4, 25.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                size=0.5,
                size_type="percent",
                upon_opposite_entry="close",
                accumulate=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 1, 1, 1, 12.5, 2.0, 0.0, 0, 0, -1),
                    (2, 0, 3, 3, 3, 62.5, 4.0, 0.0, 1, 0, -1),
                    (3, 0, 4, 4, 4, 27.5, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=0.5, size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=0.5, size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 37.5, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=price_wide,
                size=0.5,
                size_type="percent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 12.5, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 12.5, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=price_wide,
                size=0.5,
                size_type="percent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            from_signals_longonly(
                close=price_wide,
                size=50,
                size_type="percent100",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
        )

    def test_value_percent(self):
        assert_records_close(
            from_signals_both(size=0.5, size_type="valuepercent").order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 81.25, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=0.5, size_type="valuepercent").order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=0.5, size_type="valuepercent").order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 37.5, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=price_wide,
                size=0.5,
                size_type="valuepercent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 50.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 50.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=price_wide,
                size=0.5,
                size_type="valuepercent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            from_signals_longonly(
                close=price_wide,
                size=50,
                size_type="valuepercent100",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
        )

    def test_price(self):
        assert_records_close(
            from_signals_both(price=price * 1.01).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 99.00990099009901, 1.01, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 198.01980198019803, 4.04, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(price=price * 1.01).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 99.00990099, 1.01, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 99.00990099, 4.04, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(price=price * 1.01).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 99.00990099009901, 1.01, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 49.504950495049506, 4.04, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(price=np.inf).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(price=np.inf).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(price=np.inf).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(price=-np.inf, open=price.shift(1)).order_records,
            np.array(
                [
                    (0, 0, 1, 1, 1, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 3.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(price=-np.inf, open=price.shift(1)).order_records,
            np.array(
                [
                    (0, 0, 1, 1, 1, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 3.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(price=-np.inf, open=price.shift(1)).order_records,
            np.array(
                [
                    (0, 0, 1, 1, 1, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 66.66666666666667, 3.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_price_area(self):
        assert_records_close(
            from_signals_both(
                open=2,
                high=4,
                low=1,
                close=3,
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 0.55, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 3.3, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 5.5, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 0.55, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 3.3, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 5.5, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 0.45, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 2.7, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 4.5, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_longonly(
                entries=True,
                exits=False,
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=0.5,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=np.inf,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=5,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=0.5,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=np.inf,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=5,
                size=1,
                slippage=0.1,
            )

    def test_val_price(self):
        price_nan = pd.Series([1, 2, np.nan, 4, 5], index=price.index)
        assert_records_close(
            from_signals_both(
                close=price_nan, size=1, val_price=np.inf, size_type="value"
            ).order_records,
            from_signals_both(
                close=price_nan, size=1, val_price=price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan, size=1, val_price=np.inf, size_type="value"
            ).order_records,
            from_signals_longonly(
                close=price_nan, size=1, val_price=price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan, size=1, val_price=np.inf, size_type="value"
            ).order_records,
            from_signals_shortonly(
                close=price_nan, size=1, val_price=price, size_type="value"
            ).order_records,
        )
        shift_price = price_nan.ffill().shift(1)
        assert_records_close(
            from_signals_longonly(
                close=price_nan, size=1, val_price=-np.inf, size_type="value"
            ).order_records,
            from_signals_longonly(
                close=price_nan, size=1, val_price=shift_price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan, size=1, val_price=-np.inf, size_type="value"
            ).order_records,
            from_signals_longonly(
                close=price_nan, size=1, val_price=shift_price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan, size=1, val_price=-np.inf, size_type="value"
            ).order_records,
            from_signals_shortonly(
                close=price_nan, size=1, val_price=shift_price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=price_nan,
                size=1,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_both(
                close=price_nan,
                size=1,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan,
                size=1,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_longonly(
                close=price_nan,
                size=1,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan,
                size=1,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_shortonly(
                close=price_nan,
                size=1,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        price_all_nan = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=price.index)
        assert_records_close(
            from_signals_both(
                close=price_nan,
                size=1,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_both(
                close=price_nan,
                size=1,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan,
                size=1,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_longonly(
                close=price_nan,
                size=1,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan,
                size=1,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_shortonly(
                close=price_nan,
                size=1,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=price_nan, size=1, val_price=np.nan, size_type="value"
            ).order_records,
            from_signals_both(
                close=price_nan, size=1, val_price=shift_price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan, size=1, val_price=np.nan, size_type="value"
            ).order_records,
            from_signals_longonly(
                close=price_nan, size=1, val_price=shift_price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan, size=1, val_price=np.nan, size_type="value"
            ).order_records,
            from_signals_shortonly(
                close=price_nan, size=1, val_price=shift_price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=price_nan,
                open=price_nan,
                size=1,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_signals_both(
                close=price_nan, size=1, val_price=price_nan, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan,
                open=price_nan,
                size=1,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_signals_longonly(
                close=price_nan, size=1, val_price=price_nan, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan,
                open=price_nan,
                size=1,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_signals_shortonly(
                close=price_nan, size=1, val_price=price_nan, size_type="value"
            ).order_records,
        )

    def test_fees(self):
        assert_records_close(
            from_signals_both(size=1, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, -0.1, 0, 0, -1),
                    (1, 0, 3, 3, 3, 2.0, 4.0, -0.8, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.1, 0, 0, -1),
                    (1, 2, 3, 3, 3, 2.0, 4.0, 0.8, 1, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 1.0, 0, 0, -1),
                    (1, 3, 3, 3, 3, 2.0, 4.0, 8.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, -0.1, 0, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, -0.4, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.1, 0, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.4, 1, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 1.0, 0, 0, -1),
                    (1, 3, 3, 3, 3, 1.0, 4.0, 4.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, -0.1, 1, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, -0.4, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.1, 1, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.4, 0, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 1.0, 1, 0, -1),
                    (1, 3, 3, 3, 3, 1.0, 4.0, 4.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_fixed_fees(self):
        assert_records_close(
            from_signals_both(size=1, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, -0.1, 0, 0, -1),
                    (1, 0, 3, 3, 3, 2.0, 4.0, -0.1, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.1, 0, 0, -1),
                    (1, 2, 3, 3, 3, 2.0, 4.0, 0.1, 1, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 1.0, 0, 0, -1),
                    (1, 3, 3, 3, 3, 2.0, 4.0, 1.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, -0.1, 0, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, -0.1, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.1, 0, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.1, 1, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 1.0, 0, 0, -1),
                    (1, 3, 3, 3, 3, 1.0, 4.0, 1.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, -0.1, 1, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, -0.1, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.1, 1, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.1, 0, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 1.0, 1, 0, -1),
                    (1, 3, 3, 3, 3, 1.0, 4.0, 1.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_slippage(self):
        assert_records_close(
            from_signals_both(size=1, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.1, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 2.0, 3.6, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 2.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 2.0, 0.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.1, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 3.6, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 2.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 0.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 0.9, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.4, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 0.0, 0.0, 1, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 8.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_min_size(self):
        assert_records_close(
            from_signals_both(size=1, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_max_size(self):
        assert_records_close(
            from_signals_both(size=1, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 0.5, 4.0, 0.0, 1, 0, -1),
                    (2, 0, 4, 4, 4, 0.5, 5.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (2, 1, 4, 4, 4, 1.0, 5.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 0.5, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 0.5, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 0.5, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 0.5, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_reject_prob(self):
        assert_records_close(
            from_signals_both(size=1.0, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1.0, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1.0, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_allow_partial(self):
        assert_records_close(
            from_signals_both(size=1000, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1000, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1000, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 50.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_raise_reject(self):
        assert_records_close(
            from_signals_both(size=1000, allow_partial=True, raise_reject=True).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1000, allow_partial=True, raise_reject=True).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_shortonly(size=1000, allow_partial=True, raise_reject=True).order_records
        with pytest.raises(Exception):
            from_signals_both(size=1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception):
            from_signals_longonly(size=1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception):
            from_signals_shortonly(size=1000, allow_partial=False, raise_reject=True).order_records

    def test_log(self):
        assert_records_close(
            from_signals_both(log=True).log_records,
            np.array(
                [
                    (
                        0,
                        0,
                        0,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        100.0,
                        0.0,
                        0.0,
                        0.0,
                        100.0,
                        1.0,
                        100.0,
                        np.inf,
                        np.inf,
                        0,
                        2,
                        0.0,
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        0,
                        0.0,
                        0,
                        True,
                        False,
                        True,
                        100.0,
                        1.0,
                        0.0,
                        0,
                        0,
                        -1,
                        0.0,
                        100.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        100.0,
                        0,
                    ),
                    (
                        1,
                        0,
                        0,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        0.0,
                        100.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        400.0,
                        -np.inf,
                        np.inf,
                        0,
                        2,
                        0.0,
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        0,
                        0.0,
                        0,
                        True,
                        False,
                        True,
                        200.0,
                        4.0,
                        0.0,
                        1,
                        0,
                        -1,
                        800.0,
                        -100.0,
                        400.0,
                        400.0,
                        0.0,
                        4.0,
                        400.0,
                        1,
                    ),
                ],
                dtype=log_dt,
            ),
        )

    def test_accumulate(self):
        assert_records_close(
            from_signals_both(
                size=1, accumulate=[["disabled", "addonly", "removeonly", "both"]]
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 1, 3, 3, 3, 3.0, 4.0, 0.0, 1, 0, -1),
                    (3, 1, 4, 4, 4, 1.0, 5.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (2, 2, 4, 4, 4, 1.0, 5.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 3, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 3, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (3, 3, 4, 4, 4, 1.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                size=1, accumulate=[["disabled", "addonly", "removeonly", "both"]]
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 1, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 3, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 3, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (3, 3, 4, 4, 4, 1.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                size=1, accumulate=[["disabled", "addonly", "removeonly", "both"]]
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 1, 3, 3, 3, 2.0, 4.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 3, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                    (3, 3, 4, 4, 4, 1.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_long_conflict(self):
        kwargs = dict(
            close=price[:3],
            entries=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, True, False],
                    [True, True, True, True, True, True, True],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [False, False, False, False, True, False, True],
                    [True, True, True, True, True, True, True],
                ]
            ),
            size=1.0,
            accumulate=True,
            upon_long_conflict=[
                ["ignore", "entry", "exit", "adjacent", "adjacent", "opposite", "opposite"]
            ],
        )
        assert_records_close(
            from_signals_longonly(**kwargs).order_records,
            np.array(
                [
                    (0, 0, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 1, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 2, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (1, 2, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 3, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (1, 3, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 5, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 5, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 5, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 6, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 6, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 6, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_short_conflict(self):
        kwargs = dict(
            close=price[:3],
            entries=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, True, False],
                    [True, True, True, True, True, True, True],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [False, False, False, False, True, False, True],
                    [True, True, True, True, True, True, True],
                ]
            ),
            size=1.0,
            accumulate=True,
            upon_short_conflict=[
                ["ignore", "entry", "exit", "adjacent", "adjacent", "opposite", "opposite"]
            ],
        )
        assert_records_close(
            from_signals_shortonly(**kwargs).order_records,
            np.array(
                [
                    (0, 0, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 1, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 2, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (1, 2, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 3, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (1, 3, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 5, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 5, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 5, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 6, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 6, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 6, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_dir_conflict(self):
        kwargs = dict(
            close=price[:3],
            entries=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, True, False],
                    [True, True, True, True, True, True, True],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [False, False, False, False, True, False, True],
                    [True, True, True, True, True, True, True],
                ]
            ),
            size=1.0,
            accumulate=True,
            upon_dir_conflict=[
                ["ignore", "long", "short", "adjacent", "adjacent", "opposite", "opposite"]
            ],
        )
        assert_records_close(
            from_signals_both(**kwargs).order_records,
            np.array(
                [
                    (0, 0, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 1, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 2, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 3, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (1, 3, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 4, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (1, 4, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 5, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (1, 5, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 6, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (1, 6, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_opposite_entry(self):
        kwargs = dict(
            close=price[:3],
            entries=pd.DataFrame(
                [
                    [True, False, True, False, True, False, True, False, True, False],
                    [False, True, False, True, False, True, False, True, False, True],
                    [True, False, True, False, True, False, True, False, True, False],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [False, True, False, True, False, True, False, True, False, True],
                    [True, False, True, False, True, False, True, False, True, False],
                    [False, True, False, True, False, True, False, True, False, True],
                ]
            ),
            size=1.0,
            upon_opposite_entry=[
                [
                    "ignore",
                    "ignore",
                    "close",
                    "close",
                    "closereduce",
                    "closereduce",
                    "reverse",
                    "reverse",
                    "reversereduce",
                    "reversereduce",
                ]
            ],
        )
        assert_records_close(
            from_signals_both(**kwargs).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 2, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 3, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 4, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 4, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 4, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 5, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 5, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 5, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 6, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 6, 1, 1, 1, 2.0, 2.0, 0.0, 1, 0, -1),
                    (2, 6, 2, 2, 2, 2.0, 3.0, 0.0, 0, 0, -1),
                    (0, 7, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 7, 1, 1, 1, 2.0, 2.0, 0.0, 0, 0, -1),
                    (2, 7, 2, 2, 2, 2.0, 3.0, 0.0, 1, 0, -1),
                    (0, 8, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 8, 1, 1, 1, 2.0, 2.0, 0.0, 1, 0, -1),
                    (2, 8, 2, 2, 2, 2.0, 3.0, 0.0, 0, 0, -1),
                    (0, 9, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 9, 1, 1, 1, 2.0, 2.0, 0.0, 0, 0, -1),
                    (2, 9, 2, 2, 2, 2.0, 3.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(**kwargs, accumulate=True).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 2, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 3, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 4, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 4, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 4, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 5, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 5, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 5, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                    (0, 6, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 6, 1, 1, 1, 2.0, 2.0, 0.0, 1, 0, -1),
                    (2, 6, 2, 2, 2, 2.0, 3.0, 0.0, 0, 0, -1),
                    (0, 7, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 7, 1, 1, 1, 2.0, 2.0, 0.0, 0, 0, -1),
                    (2, 7, 2, 2, 2, 2.0, 3.0, 0.0, 1, 0, -1),
                    (0, 8, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 8, 1, 1, 1, 1.0, 2.0, 0.0, 1, 0, -1),
                    (2, 8, 2, 2, 2, 1.0, 3.0, 0.0, 0, 0, -1),
                    (0, 9, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 9, 1, 1, 1, 1.0, 2.0, 0.0, 0, 0, -1),
                    (2, 9, 2, 2, 2, 1.0, 3.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_init_cash(self):
        assert_records_close(
            from_signals_both(
                close=price_wide, size=1.0, init_cash=[0.0, 1.0, 100.0]
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 2.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=price_wide, size=1.0, init_cash=[0.0, 1.0, 100.0]
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_wide, size=1.0, init_cash=[0.0, 1.0, 100.0]
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 0.5, 4.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 3, 3, 3, 1.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_both(init_cash=np.inf).order_records
        with pytest.raises(Exception):
            from_signals_longonly(init_cash=np.inf).order_records
        with pytest.raises(Exception):
            from_signals_shortonly(init_cash=np.inf).order_records

    def test_init_position(self):
        pf = vbt.Portfolio.from_signals(
            close=1,
            entries=False,
            exits=True,
            init_cash=0.0,
            init_position=1.0,
            direction="longonly",
        )
        assert pf.init_position == 1.0
        assert_records_close(
            pf.order_records,
            np.array([(0, 0, 0, 0, 0, 1.0, 1.0, 0.0, 1, 0, -1)], dtype=fs_order_dt),
        )

    def test_group_by(self):
        pf = from_signals_both(close=price_wide, group_by=np.array([0, 0, 1]))
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_index_equal(
            pf.wrapper.grouper.group_by, pd.Index([0, 0, 1], dtype="int64", name="group")
        )
        assert_series_equal(
            pf.init_cash,
            pd.Series([200.0, 100.0], index=pd.Index([0, 1], dtype="int64"))
            .rename("init_cash")
            .rename_axis("group"),
        )
        assert not pf.cash_sharing

    def test_cash_sharing(self):
        pf = from_signals_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_index_equal(
            pf.wrapper.grouper.group_by,
            pd.Index([0, 0, 1], dtype="int64", name="group"),
        )
        assert_series_equal(
            pf.init_cash,
            pd.Series([100.0, 100.0], index=pd.Index([0, 1], dtype="int64"))
            .rename("init_cash")
            .rename_axis("group"),
        )
        assert pf.cash_sharing
        with pytest.raises(Exception):
            pf.regroup(group_by=False)

    @pytest.mark.parametrize(
        "test_ls",
        [False, True],
    )
    def test_from_ago(self, test_ls):
        _from_signals_both = from_ls_signals_both if test_ls else from_signals_both
        _from_signals_longonly = from_ls_signals_longonly if test_ls else from_signals_longonly
        _from_signals_shortonly = from_ls_signals_shortonly if test_ls else from_signals_shortonly

        assert_records_close(
            _from_signals_both(from_ago=1).order_records,
            np.array(
                [
                    (0, 0, 0, 1, 1, 50.0, 2.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 4, 4, 100.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(from_ago=1).order_records,
            np.array(
                [
                    (0, 0, 0, 1, 1, 50.0, 2.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 4, 4, 50.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(from_ago=1).order_records,
            np.array(
                [
                    (0, 0, 0, 1, 1, 50.0, 2.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 4, 4, 40.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(open=price * 0.9, price="nextopen").order_records,
            _from_signals_both(open=price * 0.9, from_ago=1, price=-np.inf).order_records,
        )
        assert_records_close(
            _from_signals_both(price="nextclose").order_records,
            _from_signals_both(from_ago=1, price=np.inf).order_records,
        )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_call_seq(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
            _from_signals_longonly = partial(from_signals_longonly, adjust_func_nb=adjust_func_nb)
            _from_signals_shortonly = partial(from_signals_shortonly, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
            _from_signals_longonly = from_signals_longonly
            _from_signals_shortonly = from_signals_shortonly
        pf = _from_signals_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        )
        pf = _from_signals_both(
            close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True, call_seq="reversed"
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        pf = _from_signals_both(
            close=price_wide,
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            call_seq="random",
            seed=seed,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 3, 3, 3, 200.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        kwargs = dict(
            close=1.0,
            entries=pd.DataFrame(
                [
                    [False, False, True],
                    [False, True, False],
                    [True, False, False],
                    [False, False, True],
                    [False, True, False],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, True],
                    [False, True, False],
                    [True, False, False],
                    [False, False, True],
                ]
            ),
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq="auto",
        )
        pf = _from_signals_both(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 1, 1, 1, 200.0, 1.0, 0.0, 1, 0, -1),
                    (2, 2, 3, 3, 3, 200.0, 1.0, 0.0, 0, 0, -1),
                    (3, 2, 4, 4, 4, 200.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [0, 1, 2], [2, 0, 1], [1, 0, 2], [0, 1, 2]]),
        )
        pf = _from_signals_longonly(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 2, 2, 2, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 1, 1, 1, 1, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 2, 2, 2, 100.0, 1.0, 0.0, 1, 0, -1),
                    (2, 1, 4, 4, 4, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 1, 1, 1, 100.0, 1.0, 0.0, 1, 0, -1),
                    (2, 2, 3, 3, 3, 100.0, 1.0, 0.0, 0, 0, -1),
                    (3, 2, 4, 4, 4, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 0, 1]]),
        )
        pf = _from_signals_shortonly(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 2, 2, 2, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 3, 3, 3, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 1, 1, 1, 1, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 2, 2, 2, 100.0, 1.0, 0.0, 0, 0, -1),
                    (2, 1, 4, 4, 4, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 1, 1, 1, 100.0, 1.0, 0.0, 0, 0, -1),
                    (2, 2, 3, 3, 3, 100.0, 1.0, 0.0, 1, 0, -1),
                    (3, 2, 4, 4, 4, 100.0, 1.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 0, 1]]),
        )
        _ = _from_signals_both(attach_call_seq=False, **kwargs)
        _ = _from_signals_longonly(attach_call_seq=False, **kwargs)
        _ = _from_signals_shortonly(attach_call_seq=False, **kwargs)

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_limit_order_type(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
            _from_signals_longonly = partial(from_signals_longonly, adjust_func_nb=adjust_func_nb)
            _from_signals_shortonly = partial(from_signals_shortonly, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
            _from_signals_longonly = from_signals_longonly
            _from_signals_shortonly = from_signals_shortonly
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([3.0, 4.0, 5.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                price=[[np.nan, 3.0, 4.5, 2.5, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 0, 22.22222222222222, 4.5, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 40.0, 2.5, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 0, 1, -1),
                    (0, 5, 0, 0, 3, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 1, 22.22222222222222, 4.5, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 40.0, 2.5, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_reverse=True,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 2, 20.0, 5.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 1, 22.22222222222222, 4.5, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 40.0, 2.5, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_order_price="close",
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 50.0, 2.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 50.0, 2.0, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1.5, 0.5, 2.0, np.inf]],
                delta_format="absolute",
            ).order_records,
        )
        target_limit_delta = pd.concat((close, close, close, close, close, close), axis=1)
        target_limit_delta.iloc[:, 0] = close * (1 + np.inf)
        target_limit_delta.iloc[:, 1] = close * (1 - 0.0)
        target_limit_delta.iloc[:, 2] = close * (1 + 1 / 2)
        target_limit_delta.iloc[:, 3] = close * (1 - 1 / 6)
        target_limit_delta.iloc[:, 4] = close * (1 - 2 / 3)
        target_limit_delta.iloc[:, 5] = close * (1 - np.inf)
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=target_limit_delta,
                delta_format="target",
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                slippage=0.01,
                order_type="limit",
                price=[[np.nan, 3.0, 4.5, 2.5, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 0, 22.22222222222222, 4.5, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 40.0, 2.5, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 0, 1, -1),
                    (0, 5, 0, 0, 3, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                slippage=0.01,
                order_type="limit",
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 1, 22.22222222222222, 4.5, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 40.0, 2.5, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                slippage=0.01,
                order_type="limit",
                limit_order_price="limit",
                limit_reverse=True,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 2, 19.047619047619047, 5.25, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                slippage=0.01,
                order_type="limit",
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 1, 22.22222222222222, 4.5, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 40.0, 2.5, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                slippage=0.01,
                order_type="limit",
                limit_order_price="close",
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 50.0, 2.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 1, 25.0, 4.0, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 50.0, 2.0, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                price=[[np.nan, 3.0, 4.5, 2.5, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 2, 22.22222222222222, 4.5, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 0, 40.0, 2.5, 0.0, 1, 1, -1),
                    (0, 4, 0, 0, 0, 100.0, 1.0, 0.0, 1, 1, -1),
                    (0, 5, 0, 0, 1, 33.333333333333336, 3.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 25.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 1, 33.333333333333336, 3.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 1, 66.66666666666667, 1.5, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 1, 28.571428571428573, 3.5, 0.0, 1, 1, -1),
                    (0, 4, 0, 0, 2, 20.0, 5.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_reverse=True,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 25.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 50.0, 2.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 1, 25.0, 4.0, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 3, 50.0, 2.0, 0.0, 1, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1.5, 0.5, 2.0, np.inf]],
                delta_format="absolute",
            ).order_records,
        )
        assert_records_close(
            _from_signals_shortonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                price=[[np.nan, 3.0, 4.5, 2.5, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 1, 22.22222222222222, 4.5, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 0, 40.0, 2.5, 0.0, 1, 1, -1),
                    (0, 4, 0, 0, 0, 100.0, 1.0, 0.0, 1, 1, -1),
                    (0, 5, 0, 0, 1, 33.333333333333336, 3.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 1, 33.333333333333336, 3.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 1, 66.66666666666667, 1.5, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 1, 28.571428571428573, 3.5, 0.0, 1, 1, -1),
                    (0, 4, 0, 0, 2, 20.0, 5.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                slippage=0.01,
                limit_reverse=True,
                limit_delta=[[-np.inf, 0.0, -1 / 2, 1 / 6, 2 / 3, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 44.44444444444444, 2.25, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 1, 23.529411764705884, 4.25, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 3, 44.44444444444444, 2.25, 0.0, 1, 1, -1),
                    (0, 4, 0, 0, 4, 100.0, 1.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_limit_tif(self):
        entries = pd.Series(
            [True, False, False, False, False], index=price.index.tz_localize("+0200")
        )
        exits = pd.Series(
            [False, False, False, False, False], index=price.index.tz_localize("+0200")
        )
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index.tz_localize("+0200"))
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_tif=[[-1, 0, 1, 2, 3]],
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_tif=[[-1, 0, 1, 2, 3]],
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_tif=[[-1, 0, 1, 2, 3]],
                time_delta_format="rows",
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_tif=[[1000 * day_dt, 0 * day_dt, 1 * day_dt, 2 * day_dt, 3 * day_dt]],
                time_delta_format="index",
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_tif=[[2 * day_dt, 2.5 * day_dt]],
                time_delta_format="index",
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_tif="2d",
                time_delta_format="index",
            ).order_records,
            np.array(
                [],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_tif=pd.Timedelta("2d"),
                time_delta_format="index",
            ).order_records,
            np.array(
                [],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_tif=pd.Timedelta("3d"),
                time_delta_format="index",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

        entries2 = pd.Series(
            [True, False, True, False, False], index=price.index.tz_localize("+0200")
        )
        limit_tif = pd.Series([2, -1, 2, -1, -1], index=price.index.tz_localize("+0200"))
        price2 = pd.Series([4.0, 3.0, 1.0, 1.0, 1.0], index=price.index.tz_localize("+0200"))
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries2,
                exits=exits,
                size=1,
                accumulate=True,
                order_type="limit",
                price=price2,
                limit_tif=limit_tif,
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 1.0, 4.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_limit_expiry(self):
        entries = pd.Series(
            [True, False, False, False, False], index=price.index.tz_localize("+0200")
        )
        exits = pd.Series(
            [False, False, False, False, False], index=price.index.tz_localize("+0200")
        )
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index.tz_localize("+0200"))
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=[[-1, 0, 1, 2, 3]],
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=[[-1, 0, 1, 2, 3]],
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                    (0, 4, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=[[-1, 0, 1, 2, 3]],
                time_delta_format="rows",
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=[
                    [
                        pd.Timestamp.max,
                        close.index[0] + 0 * day_dt,
                        close.index[0] + 1 * day_dt,
                        close.index[0] + 2 * day_dt,
                        close.index[0] + 3 * day_dt,
                    ]
                ],
                time_delta_format="index",
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                high=high,
                low=low,
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=[[close.index[0] + 2 * day_dt, close.index[0] + 2.5 * day_dt]],
                time_delta_format="index",
            ).order_records,
            np.array(
                [
                    (0, 1, 0, 0, 2, 33.333333333333336, 3.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry="2d",
                time_delta_format="index",
            ).order_records,
            np.array(
                [],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=pd.Timedelta("2d"),
                time_delta_format="index",
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=close.index[2],
                time_delta_format="index",
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=pd.Timedelta("3d"),
                time_delta_format="index",
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                order_type="limit",
                price=3,
                limit_expiry=close.index[3],
                time_delta_format="index",
            ).order_records,
        )

        entries2 = pd.Series(
            [True, False, True, False, False], index=price.index.tz_localize("+0200")
        )
        limit_expiry = pd.Series([2, -1, 4, -1, -1], index=price.index.tz_localize("+0200"))
        price2 = pd.Series([4.0, 3.0, 1.0, 1.0, 1.0], index=price.index.tz_localize("+0200"))
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries2,
                exits=exits,
                size=1,
                accumulate=True,
                order_type="limit",
                price=price2,
                limit_expiry=limit_expiry,
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 1, 1.0, 4.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_adj_limit_conflict_long(self):
        open = pd.Series([5.1, 4.1, 3.1, 2.1, 1.1], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        entries = pd.Series([True, True, False, False, False], index=price.index)
        entries2 = pd.Series([True, False, False, True, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        size = pd.Series([1.0, 0.5, 0.5, 0.5, 0.5], index=price.index)
        order_type = pd.Series(["limit", "market", "market", "market", "market"], index=price.index)
        price2 = pd.Series([2.0, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)
        price3 = pd.Series([2.0, np.inf, np.inf, 2.05, np.inf], index=price.index)
        price4 = pd.Series([2.0, np.inf, np.inf, np.inf, np.inf], index=price.index)
        price5 = pd.Series([2.5, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)

        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price4,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 1, 1, 1, 1, 0.5, 4.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 3, 1, 1, 1, 0.5, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                size=size,
                accumulate=True,
                order_type="limit",
                price=price4,
                upon_adj_limit_conflict="KeepExecute",
            )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price2,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 1, 3, 3, 3, 0.5, 2.1, 0.0, 0, 0, -1),
                    (1, 1, 0, 0, 4, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 3, 3, 3, 3, 0.5, 2.1, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_longonly(
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type="limit",
                price=price2,
                upon_adj_limit_conflict="KeepExecute",
            )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price3,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price4,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price5,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.5, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 2.5, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 2.5, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 2.5, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_adj_limit_conflict_short(self):
        open = pd.Series([0.9, 1.9, 2.9, 3.9, 4.9], index=price.index)
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)
        entries = pd.Series([True, True, False, False, False], index=price.index)
        entries2 = pd.Series([True, False, False, True, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        size = pd.Series([1.0, 0.5, 0.5, 0.5, 0.5], index=price.index)
        order_type = pd.Series(["limit", "market", "market", "market", "market"], index=price.index)
        price2 = pd.Series([4.0, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)
        price3 = pd.Series([4.0, np.inf, np.inf, 2.05, np.inf], index=price.index)
        price4 = pd.Series([4.0, np.inf, np.inf, np.inf, np.inf], index=price.index)
        price5 = pd.Series([3.5, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)

        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price4,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 1, 1, 1, 0.5, 2.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 3, 1, 1, 1, 0.5, 2.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                size=size,
                accumulate=True,
                order_type="limit",
                price=price4,
                upon_adj_limit_conflict="KeepExecute",
            )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price2,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 3, 3, 3, 0.5, 3.9, 0.0, 1, 0, -1),
                    (1, 1, 0, 0, 4, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 3, 3, 3, 3, 0.5, 3.9, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_shortonly(
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type="limit",
                price=price2,
                upon_adj_limit_conflict="KeepExecute",
            )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price3,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price4,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price5,
                upon_adj_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 3.5, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 3.5, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 3.5, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 3.5, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_opp_limit_conflict_long(self):
        open = pd.Series([5.1, 4.1, 3.1, 2.1, 1.1], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, True, False, False, False], index=price.index)
        exits2 = pd.Series([False, False, False, True, False], index=price.index)
        size = pd.Series([1.0, 0.5, 0.5, 0.5, 0.5], index=price.index)
        order_type = pd.Series(["limit", "market", "market", "market", "market"], index=price.index)
        price2 = pd.Series([2.0, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)
        price3 = pd.Series([2.0, np.inf, np.inf, 2.05, np.inf], index=price.index)
        price4 = pd.Series([2.0, np.inf, np.inf, np.inf, np.inf], index=price.index)
        price5 = pd.Series([2.5, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)

        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price4,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price2,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price3,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price4,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 2.0, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price5,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 2.5, 0.0, 0, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 2.5, 0.0, 0, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 2.5, 0.0, 0, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 2.5, 0.0, 0, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_opp_limit_conflict_short(self):
        open = pd.Series([0.9, 1.9, 2.9, 3.9, 4.9], index=price.index)
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, True, False, False, False], index=price.index)
        exits2 = pd.Series([False, False, False, True, False], index=price.index)
        size = pd.Series([1.0, 0.5, 0.5, 0.5, 0.5], index=price.index)
        order_type = pd.Series(["limit", "market", "market", "market", "market"], index=price.index)
        price2 = pd.Series([4.0, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)
        price3 = pd.Series([4.0, np.inf, np.inf, 2.05, np.inf], index=price.index)
        price4 = pd.Series([4.0, np.inf, np.inf, np.inf, np.inf], index=price.index)
        price5 = pd.Series([3.5, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)

        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price4,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price2,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price3,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price4,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 4.0, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                size=size,
                accumulate=True,
                order_type=order_type,
                price=price5,
                upon_opp_limit_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 3, 1.0, 3.5, 0.0, 1, 1, -1),
                    (0, 1, 0, 0, 3, 1.0, 3.5, 0.0, 1, 1, -1),
                    (0, 2, 0, 0, 3, 1.0, 3.5, 0.0, 1, 1, -1),
                    (0, 3, 0, 0, 3, 1.0, 3.5, 0.0, 1, 1, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_sl_stop(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
            _from_signals_longonly = partial(from_signals_longonly, adjust_func_nb=adjust_func_nb)
            _from_signals_shortonly = partial(from_signals_shortonly, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
            _from_signals_longonly = from_signals_longonly
            _from_signals_shortonly = from_signals_shortonly
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 20.0, 4.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 20.0, 4.25, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 20.0, 4.25, 0.0, 1, 0, 0),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 1, 1, 20.0, 4.0, 0.0, 1, 0, 0),
                    (0, 4, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 4, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 5 * 0.5, 5 * 0.75, 5 * 1.0, np.inf]],
                delta_format="absolute",
            ).order_records,
        )
        target_sl_stop = pd.concat((close, close, close, close, close), axis=1)
        target_sl_stop.iloc[:, 0] = close * (1 - np.nan)
        target_sl_stop.iloc[:, 1] = close * (1 - 0.5)
        target_sl_stop.iloc[:, 2] = close * (1 - 0.75)
        target_sl_stop.iloc[:, 3] = close * (1 - 1.0)
        target_sl_stop.iloc[:, 4] = close * (1 - np.inf)
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=target_sl_stop,
                delta_format="target",
            ).order_records,
        )

        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 1, 1, 100.0, 2.0, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 4, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 1, 1, 100.0, 1.75, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 1, 1, 100.0, 1.75, 0.0, 0, 0, 0),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 1, 1, 100.0, 2.0, 0.0, 0, 0, 0),
                    (0, 4, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_tsl_stop(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
            _from_signals_longonly = partial(from_signals_longonly, adjust_func_nb=adjust_func_nb)
            _from_signals_shortonly = partial(from_signals_shortonly, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
            _from_signals_longonly = from_signals_longonly
            _from_signals_shortonly = from_signals_shortonly
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([4.0, 5.0, 4.0, 3.0, 2.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 25.0, 4.0, 0.0, 1, 0, 1),
                    (0, 2, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 4, 4, 25.0, 2.0, 0.0, 1, 0, 1),
                    (0, 3, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 1, 1, 25.0, 5.0, 0.0, 0, 0, 1),
                    (0, 2, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.15, 0.2, 0.25, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 25.0, 4.25, 0.0, 1, 0, 1),
                    (0, 2, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 2, 2, 25.0, 4.25, 0.0, 1, 0, 1),
                    (0, 3, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 2, 2, 25.0, 4.125, 0.0, 1, 0, 1),
                    (0, 4, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.15, 0.2, 0.25, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 1, 1, 25.0, 5.25, 0.0, 0, 0, 1),
                    (0, 2, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 1, 1, 25.0, 5.25, 0.0, 0, 0, 1),
                    (0, 3, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 1, 1, 25.0, 5.25, 0.0, 0, 0, 1),
                    (0, 4, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.15, 0.2, 0.25, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 5.5 * 0.15, 5.5 * 0.2, 5.5 * 0.25, np.inf]],
                delta_format="absolute",
            ).order_records,
        )
        target_tsl_stop = pd.concat((close, close, close, close, close), axis=1)
        target_tsl_stop.iloc[:, 0] = close - np.nan
        target_tsl_stop.iloc[:, 1] = close - 1
        target_tsl_stop.iloc[:, 2] = close - 2
        target_tsl_stop.iloc[:, 3] = close - 3
        target_tsl_stop.iloc[:, 4] = close - np.inf
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 1, 2, 3, np.inf]],
                delta_format="absolute",
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=target_tsl_stop,
                delta_format="target",
            ).order_records,
        )

        close = pd.Series([2.0, 1.0, 2.0, 3.0, 4.0], index=price.index)
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 50.0, 1.0, 0.0, 1, 0, 1),
                    (0, 2, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 2, 2, 50.0, 2.0, 0.0, 0, 0, 1),
                    (0, 2, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 4, 4, 50.0, 4.0, 0.0, 0, 0, 1),
                    (0, 3, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 50.0, 0.75, 0.0, 1, 0, 1),
                    (0, 2, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 50.0, 0.5, 0.0, 1, 0, 1),
                    (0, 3, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                    (0, 4, 0, 0, 0, 50.0, 2.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 1, 1, 50.0, 1.125, 0.0, 0, 0, 1),
                    (0, 2, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 1, 1, 50.0, 1.3125, 0.0, 0, 0, 1),
                    (0, 3, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 1, 1, 50.0, 1.5, 0.0, 0, 0, 1),
                    (0, 4, 0, 0, 0, 50.0, 2.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                open=pd.Series([11.0, 10.0]),
                high=pd.Series([12.0, 11.0]),
                low=pd.Series([10.0, 9.0]),
                close=pd.Series([11.0, 10.0]),
                entries=pd.Series([True, False]),
                exits=pd.Series([False, False]),
                price=pd.Series([10.0, 10.0]),
                tsl_stop=[[np.nan, 0.05, 0.25, np.inf]],
                stop_entry_price="price",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 10.0, 10.0, 0.0, 1, 0, 1),
                    (0, 2, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                open=pd.Series([11.0, 10.0]),
                high=pd.Series([12.0, 11.0]),
                low=pd.Series([10.0, 9.0]),
                close=pd.Series([11.0, 10.0]),
                entries=pd.Series([True, False]),
                exits=pd.Series([False, False]),
                price=-np.inf,
                tsl_stop=[[np.nan, 0.05, 0.25, np.inf]],
                stop_entry_price="price",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 9.090909090909092, 11.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 9.090909090909092, 11.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 9.090909090909092, 10.0, 0.0, 1, 0, 1),
                    (0, 2, 0, 0, 0, 9.090909090909092, 11.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 9.090909090909092, 9.0, 0.0, 1, 0, 1),
                    (0, 3, 0, 0, 0, 9.090909090909092, 11.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        target_tsl_stop = pd.concat((close, close, close, close, close), axis=1)
        target_tsl_stop.iloc[:, 0] = close + np.nan
        target_tsl_stop.iloc[:, 1] = close + 1
        target_tsl_stop.iloc[:, 2] = close + 2
        target_tsl_stop.iloc[:, 3] = close + 3
        target_tsl_stop.iloc[:, 4] = close + np.inf
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 1, 2, 3, np.inf]],
                delta_format="absolute",
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=target_tsl_stop,
                delta_format="target",
            ).order_records,
        )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_tsl_th(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
            _from_signals_longonly = partial(from_signals_longonly, adjust_func_nb=adjust_func_nb)
            _from_signals_shortonly = partial(from_signals_shortonly, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
            _from_signals_longonly = from_signals_longonly
            _from_signals_shortonly = from_signals_shortonly
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([4.0, 3.0, 5.0, 4.0, 2.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 3, 3, 25.0, 4.0, 0.0, 1, 0, 2),
                    (0, 2, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 4, 4, 25.0, 2.0, 0.0, 1, 0, 2),
                    (0, 4, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (0, 5, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 2, 2, 25.0, 5.0, 0.0, 0, 0, 2),
                    (0, 2, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 2, 2, 25.0, 5.0, 0.0, 0, 0, 2),
                    (0, 4, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 5, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 25.0, 4.7250000000000005, 0.0, 1, 0, 2),
                    (0, 2, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 4, 4, 25.0, 2.25, 0.0, 1, 0, 2),
                    (0, 4, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                    (0, 5, 0, 0, 0, 25.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 1, 1, 25.0, 3.0, 0.0, 0, 0, 2),
                    (0, 2, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 4, 4, 25.0, 2.0, 0.0, 0, 0, 2),
                    (0, 3, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 2, 2, 25.0, 5.25, 0.0, 0, 0, 2),
                    (0, 4, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                    (0, 5, 0, 0, 0, 25.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_th=[[np.nan, 4 * 0.1, 4 * 0.5, 4 * 0.1, 4 * 0.5, np.inf]],
                tsl_stop=[[np.nan, 5.25 * 0.1, 5.5 * 0.1, 5.5 * 0.5, 5.5 * 0.5, np.inf]],
                delta_format="absolute",
            ).order_records,
        )
        target_tsl_th = pd.concat((close, close, close, close, close, close), axis=1)
        target_tsl_th.iloc[:, 0] = close + np.nan
        target_tsl_th.iloc[:, 1] = close + 1
        target_tsl_th.iloc[:, 2] = close + 2
        target_tsl_th.iloc[:, 3] = close + 1
        target_tsl_th.iloc[:, 4] = close + 2
        target_tsl_th.iloc[:, 5] = close + np.inf
        target_tsl_stop = pd.concat((close, close, close, close, close, close), axis=1)
        target_tsl_stop.iloc[:, 0] = close - np.nan
        target_tsl_stop.iloc[:, 1] = close - 1
        target_tsl_stop.iloc[:, 2] = close - 1
        target_tsl_stop.iloc[:, 3] = close - 2
        target_tsl_stop.iloc[:, 4] = close - 2
        target_tsl_stop.iloc[:, 4] = close - np.inf
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_th=[[np.nan, 1, 2, 1, 2, np.inf]],
                tsl_stop=[[np.nan, 1, 1, 2, 2, np.inf]],
                delta_format="absolute",
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_th=target_tsl_th,
                tsl_stop=target_tsl_stop,
                delta_format="target",
            ).order_records,
        )

        close = pd.Series([3.0, 4.0, 2.0, 3.0, 4.0], index=price.index)
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 33.333333333333336, 2.0, 0.0, 1, 0, 2),
                    (0, 2, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 2, 2, 33.333333333333336, 2.0, 0.0, 1, 0, 2),
                    (0, 4, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (0, 5, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 3, 3, 33.333333333333336, 3.0, 0.0, 0, 0, 2),
                    (0, 2, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 3, 3, 33.333333333333336, 3.0, 0.0, 0, 0, 2),
                    (0, 4, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (0, 5, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 33.333333333333336, 4.0, 0.0, 1, 0, 2),
                    (0, 2, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 33.333333333333336, 4.0, 0.0, 1, 0, 2),
                    (0, 3, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 2, 2, 33.333333333333336, 1.75, 0.0, 1, 0, 2),
                    (0, 4, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                    (1, 4, 0, 2, 2, 33.333333333333336, 1.75, 0.0, 1, 0, 2),
                    (0, 5, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                tsl_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 2, 2, 33.333333333333336, 1.9250000000000003, 0.0, 0, 0, 2),
                    (0, 2, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 2, 2, 33.333333333333336, 2.0, 0.0, 0, 0, 2),
                    (0, 3, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 3, 3, 33.333333333333336, 2.75, 0.0, 0, 0, 2),
                    (0, 4, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                    (1, 4, 0, 3, 3, 33.333333333333336, 2.75, 0.0, 0, 0, 2),
                    (0, 5, 0, 0, 0, 33.333333333333336, 3.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                open=pd.Series([11.0, 10.0]),
                high=pd.Series([12.0, 11.0]),
                low=pd.Series([10.0, 9.0]),
                close=pd.Series([11.0, 10.0]),
                entries=pd.Series([True, False]),
                exits=pd.Series([False, False]),
                price=pd.Series([10.0, 10.0]),
                tsl_th=[[0.05, 0.1, 0.2]],
                tsl_stop=0.1,
                stop_entry_price="price",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 10.0, 9.9, 0.0, 1, 0, 2),
                    (0, 1, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 10.0, 9.9, 0.0, 1, 0, 2),
                    (0, 2, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(
                open=pd.Series([11.0, 10.0]),
                high=pd.Series([12.0, 11.0]),
                low=pd.Series([10.0, 9.0]),
                close=pd.Series([11.0, 10.0]),
                entries=pd.Series([True, False]),
                exits=pd.Series([False, False]),
                price=-np.inf,
                tsl_th=[[0.05, 0.1, 0.2]],
                tsl_stop=0.1,
                stop_entry_price="price",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 9.090909090909092, 11.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 9.090909090909092, 10.0, 0.0, 1, 0, 2),
                    (0, 1, 0, 0, 0, 9.090909090909092, 11.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 9.090909090909092, 11.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_tp_stop(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
            _from_signals_longonly = partial(from_signals_longonly, adjust_func_nb=adjust_func_nb)
            _from_signals_shortonly = partial(from_signals_shortonly, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
            _from_signals_longonly = from_signals_longonly
            _from_signals_shortonly = from_signals_shortonly
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 1, 1, 20.0, 4.0, 0.0, 0, 0, 3),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 3, 3, 20.0, 2.0, 0.0, 0, 0, 3),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 4, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 1, 1, 20.0, 4.25, 0.0, 0, 0, 3),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 1, 1, 20.0, 4.25, 0.0, 0, 0, 3),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 1, 1, 20.0, 4.0, 0.0, 0, 0, 3),
                    (0, 4, 0, 0, 0, 20.0, 5.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 5 * 0.1, 5 * 0.15, 5 * 0.2, np.inf]],
                delta_format="absolute",
            ).order_records,
        )
        target_tp_stop = pd.concat((close, close, close, close, close), axis=1)
        target_tp_stop.iloc[:, 0] = close * (1 - np.nan)
        target_tp_stop.iloc[:, 1] = close * (1 - 0.1)
        target_tp_stop.iloc[:, 2] = close * (1 - 0.15)
        target_tp_stop.iloc[:, 3] = close * (1 - 0.2)
        target_tp_stop.iloc[:, 4] = close * (1 - np.inf)
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
                delta_format="percent",
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=target_tp_stop,
                delta_format="target",
            ).order_records,
        )

        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 100.0, 2.0, 0.0, 1, 0, 3),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 3, 3, 100.0, 4.0, 0.0, 1, 0, 3),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            _from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 100.0, 1.75, 0.0, 1, 0, 3),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 100.0, 1.75, 0.0, 1, 0, 3),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 1, 1, 100.0, 2.0, 0.0, 1, 0, 3),
                    (0, 4, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 4, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_td_stop(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
        entries = pd.Series([True, False, True, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                td_stop=[[-1, 0, 1, 2]],
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 20.0, 4.0, 0.0, 1, 0, 4),
                    (2, 1, 2, 2, 2, 26.666666666666668, 3.0, 0.0, 0, 0, -1),
                    (3, 1, 2, 3, 3, 26.666666666666668, 2.0, 0.0, 1, 0, 4),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 20.0, 4.0, 0.0, 1, 0, 4),
                    (2, 2, 2, 2, 2, 26.666666666666668, 3.0, 0.0, 0, 0, -1),
                    (3, 2, 2, 3, 3, 26.666666666666668, 2.0, 0.0, 1, 0, 4),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 2, 2, 20.0, 3.0, 0.0, 1, 0, 4),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                td_stop=[[-1, 0, 1, 2]],
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 20.0, 4.25, 0.0, 1, 0, 4),
                    (2, 1, 2, 2, 2, 28.333333333333332, 3.0, 0.0, 0, 0, -1),
                    (3, 1, 2, 3, 3, 28.333333333333332, 2.25, 0.0, 1, 0, 4),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 20.0, 4.25, 0.0, 1, 0, 4),
                    (2, 2, 2, 2, 2, 28.333333333333332, 3.0, 0.0, 0, 0, -1),
                    (3, 2, 2, 3, 3, 28.333333333333332, 2.25, 0.0, 1, 0, 4),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 2, 2, 20.0, 3.25, 0.0, 1, 0, 4),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                td_stop=[[-1, 0 * day_dt, 1 * day_dt, 2 * day_dt]],
                time_delta_format="index",
            ).order_records,
            _from_signals_both(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                td_stop=[[-1, 0, 1, 2]],
                time_delta_format="rows",
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                td_stop="3 days",
                time_delta_format="index",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.25, 0.0, 1, 0, 4),
                ],
                dtype=fs_order_dt,
            ),
        )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_dt_stop(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
        entries = pd.Series([True, False, True, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        assert_records_close(
            _from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                dt_stop=[[-1, 0, 1, 2]],
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 20.0, 4.0, 0.0, 1, 0, 5),
                    (2, 1, 2, 2, 2, 26.666666666666668, 3.0, 0.0, 0, 0, -1),
                    (3, 1, 2, 3, 3, 26.666666666666668, 2.0, 0.0, 1, 0, 5),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 20.0, 4.0, 0.0, 1, 0, 5),
                    (2, 2, 2, 2, 2, 26.666666666666668, 3.0, 0.0, 0, 0, -1),
                    (3, 2, 2, 3, 3, 26.666666666666668, 2.0, 0.0, 1, 0, 5),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 2, 2, 20.0, 3.0, 0.0, 1, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                dt_stop=[[-1, 0, 1, 2]],
                time_delta_format="rows",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 20.0, 4.25, 0.0, 1, 0, 5),
                    (2, 1, 2, 2, 2, 28.333333333333332, 3.0, 0.0, 0, 0, -1),
                    (3, 1, 2, 3, 3, 28.333333333333332, 2.25, 0.0, 1, 0, 5),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 20.0, 4.25, 0.0, 1, 0, 5),
                    (2, 2, 2, 2, 2, 28.333333333333332, 3.0, 0.0, 0, 0, -1),
                    (3, 2, 2, 3, 3, 28.333333333333332, 2.25, 0.0, 1, 0, 5),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 2, 2, 20.0, 3.25, 0.0, 1, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                dt_stop=[
                    [-1, price.index[0], price.index[0] + day_dt, price.index[0] + 2 * day_dt]
                ],
                time_delta_format="index",
            ).order_records,
            _from_signals_both(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                dt_stop=[[-1, 0, 1, 2]],
                time_delta_format="rows",
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                dt_stop="3 days",
                time_delta_format="index",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 2, 2, 20.0, 3.0, 0.0, 1, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_stop_ladder(self, test_flexible):
        if test_flexible:
            _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
        else:
            _from_signals_both = from_signals_both
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([10, 9, 11, 12, 13], index=price.index)
        assert_records_close(
            _from_signals_both(
                close,
                entries=entries,
                exits=exits,
                stop_ladder="uniform",
                sl_stop=[0.1, 0.15, 0.3],
                tp_stop=[0.1, 0.15, 0.3],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 1, 0, 0),
                    (2, 0, 0, 2, 2, 3.333333333333333, 11.0, 0.0, 1, 0, 3),
                    (3, 0, 0, 3, 3, 3.333333333333333, 12.0, 0.0, 1, 0, 3),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=entries,
                exits=exits,
                stop_ladder="weighted",
                sl_stop=[0.1, 0.15, 0.3],
                tp_stop=[0.1, 0.15, 0.3],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 1, 0, 0),
                    (2, 0, 0, 2, 2, 3.333333333333333, 11.0, 0.0, 1, 0, 3),
                    (3, 0, 0, 3, 3, 1.6666666666666665, 12.0, 0.0, 1, 0, 3),
                    (4, 0, 0, 4, 4, 1.6666666666666674, 13.0, 0.0, 1, 0, 3),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=entries,
                exits=exits,
                stop_ladder="adaptuniform",
                sl_stop=[0.1, 0.15, 0.3],
                tp_stop=[0.1, 0.15, 0.3],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 1, 0, 0),
                    (2, 0, 0, 2, 2, 2.2222222222222223, 11.0, 0.0, 1, 0, 3),
                    (3, 0, 0, 3, 3, 2.2222222222222223, 12.0, 0.0, 1, 0, 3),
                    (4, 0, 0, 4, 4, 2.2222222222222223, 13.0, 0.0, 1, 0, 3),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=entries,
                exits=exits,
                stop_ladder="adaptweighted",
                sl_stop=[0.1, 0.15, 0.3],
                tp_stop=[0.1, 0.15, 0.3],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 1, 0, 0),
                    (2, 0, 0, 2, 2, 2.2222222222222223, 11.0, 0.0, 1, 0, 3),
                    (3, 0, 0, 3, 3, 1.1111111111111112, 12.0, 0.0, 1, 0, 3),
                    (4, 0, 0, 4, 4, 3.3333333333333335, 13.0, 0.0, 1, 0, 3),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=exits,
                exits=entries,
                stop_ladder="uniform",
                sl_stop=[0.1, 0.15, 0.3],
                tp_stop=[0.1, 0.15, 0.3],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 0, 0, 3),
                    (2, 0, 0, 2, 2, 3.333333333333333, 11.0, 0.0, 0, 0, 0),
                    (3, 0, 0, 3, 3, 3.333333333333333, 12.0, 0.0, 0, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=exits,
                exits=entries,
                stop_ladder="weighted",
                sl_stop=[0.1, 0.15, 0.3],
                tp_stop=[0.1, 0.15, 0.3],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 0, 0, 3),
                    (2, 0, 0, 2, 2, 3.333333333333333, 11.0, 0.0, 0, 0, 0),
                    (3, 0, 0, 3, 3, 1.6666666666666665, 12.0, 0.0, 0, 0, 0),
                    (4, 0, 0, 4, 4, 1.6666666666666674, 13.0, 0.0, 0, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=exits,
                exits=entries,
                stop_ladder="adaptuniform",
                sl_stop=[0.1, 0.15, 0.3],
                tp_stop=[0.1, 0.15, 0.3],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 0, 0, 3),
                    (2, 0, 0, 2, 2, 2.2222222222222223, 11.0, 0.0, 0, 0, 0),
                    (3, 0, 0, 3, 3, 2.2222222222222223, 12.0, 0.0, 0, 0, 0),
                    (4, 0, 0, 4, 4, 2.2222222222222223, 13.0, 0.0, 0, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=exits,
                exits=entries,
                stop_ladder="adaptweighted",
                sl_stop=[0.1, 0.15, 0.3],
                tp_stop=[0.1, 0.15, 0.3],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 0, 0, 3),
                    (2, 0, 0, 2, 2, 2.2222222222222223, 11.0, 0.0, 0, 0, 0),
                    (3, 0, 0, 3, 3, 1.1111111111111112, 12.0, 0.0, 0, 0, 0),
                    (4, 0, 0, 4, 4, 3.3333333333333335, 13.0, 0.0, 0, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=entries,
                exits=exits,
                stop_ladder="uniform",
                sl_stop=[0.1, 0.15, 0.3],
                dt_stop=close.index[[2, 3, 4]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 1, 0, 0),
                    (2, 0, 0, 2, 2, 3.333333333333333, 11.0, 0.0, 1, 0, 5),
                    (3, 0, 0, 3, 3, 3.333333333333333, 12.0, 0.0, 1, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=entries,
                exits=exits,
                stop_ladder="weighted",
                sl_stop=[0.1, 0.15, 0.3],
                dt_stop=close.index[[2, 3, 4]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 1, 0, 0),
                    (2, 0, 0, 2, 2, 5.0, 11.0, 0.0, 1, 0, 5),
                    (3, 0, 0, 3, 3, 1.666666666666667, 12.0, 0.0, 1, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=entries,
                exits=exits,
                stop_ladder="adaptuniform",
                sl_stop=[0.1, 0.15, 0.3],
                dt_stop=close.index[[2, 3, 4]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 1, 0, 0),
                    (2, 0, 0, 2, 2, 2.2222222222222223, 11.0, 0.0, 1, 0, 5),
                    (3, 0, 0, 3, 3, 2.2222222222222223, 12.0, 0.0, 1, 0, 5),
                    (4, 0, 0, 4, 4, 2.2222222222222223, 13.0, 0.0, 1, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=entries,
                exits=exits,
                stop_ladder="adaptweighted",
                sl_stop=[0.1, 0.15, 0.3],
                dt_stop=close.index[[2, 3, 4]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 1, 0, 0),
                    (2, 0, 0, 2, 2, 3.3333333333333335, 11.0, 0.0, 1, 0, 5),
                    (3, 0, 0, 3, 3, 1.6666666666666667, 12.0, 0.0, 1, 0, 5),
                    (4, 0, 0, 4, 4, 1.6666666666666667, 13.0, 0.0, 1, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=exits,
                exits=entries,
                stop_ladder="uniform",
                tp_stop=[0.1, 0.15, 0.3],
                dt_stop=close.index[[2, 3, 4]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 0, 0, 3),
                    (2, 0, 0, 2, 2, 3.333333333333333, 11.0, 0.0, 0, 0, 5),
                    (3, 0, 0, 3, 3, 3.333333333333333, 12.0, 0.0, 0, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=exits,
                exits=entries,
                stop_ladder="weighted",
                tp_stop=[0.1, 0.15, 0.3],
                dt_stop=close.index[[2, 3, 4]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 0, 0, 3),
                    (2, 0, 0, 2, 2, 5.0, 11.0, 0.0, 0, 0, 5),
                    (3, 0, 0, 3, 3, 1.666666666666667, 12.0, 0.0, 0, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=exits,
                exits=entries,
                stop_ladder="adaptuniform",
                tp_stop=[0.1, 0.15, 0.3],
                dt_stop=close.index[[2, 3, 4]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 0, 0, 3),
                    (2, 0, 0, 2, 2, 2.2222222222222223, 11.0, 0.0, 0, 0, 5),
                    (3, 0, 0, 3, 3, 2.2222222222222223, 12.0, 0.0, 0, 0, 5),
                    (4, 0, 0, 4, 4, 2.2222222222222223, 13.0, 0.0, 0, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            _from_signals_both(
                close,
                entries=exits,
                exits=entries,
                stop_ladder="adaptweighted",
                tp_stop=[0.1, 0.15, 0.3],
                dt_stop=close.index[[2, 3, 4]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 1, 1, 3.333333333333333, 9.0, 0.0, 0, 0, 3),
                    (2, 0, 0, 2, 2, 3.3333333333333335, 11.0, 0.0, 0, 0, 5),
                    (3, 0, 0, 3, 3, 1.6666666666666667, 12.0, 0.0, 0, 0, 5),
                    (4, 0, 0, 4, 4, 1.6666666666666667, 13.0, 0.0, 0, 0, 5),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_stop_entry_price(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        slippage = pd.Series([0.1, 0.0, 0.0, 0.0, 0.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                slippage=slippage,
                stop_entry_price="val_price",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 4.25, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 2.625, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 2, 0, 4, 4, 16.52892561983471, 1.25, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                slippage=slippage,
                stop_entry_price="price",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 4.25, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 2.75, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 2, 0, 4, 4, 16.52892561983471, 1.25, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                slippage=slippage,
                stop_entry_price="fillprice",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 4.25, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 3.0250000000000004, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 2, 0, 3, 3, 16.52892561983471, 1.5125000000000002, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                slippage=slippage,
                stop_entry_price="open",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 4.25, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 2.625, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 2, 0, 4, 4, 16.52892561983471, 1.25, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                slippage=slippage,
                stop_entry_price="close",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 4.25, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 2.5, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 2, 0, 4, 4, 16.52892561983471, 1.25, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                slippage=slippage,
                stop_entry_price=close,
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                slippage=slippage,
                stop_entry_price="close",
            ).order_records,
        )

    def test_stop_exit_price(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        slippage = pd.Series([0.1, 0.0, 0.0, 0.0, 0.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                slippage=slippage,
                stop_exit_price="stop",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 4.25, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 2.5, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 2, 0, 4, 4, 16.52892561983471, 1.25, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                slippage=slippage,
                stop_exit_price="hardstop",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 4.75, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 2.5, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 2, 0, 4, 4, 16.52892561983471, 1.25, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                slippage=0.1,
                stop_exit_price="close",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 3.6, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 2.7, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 2, 0, 4, 4, 16.52892561983471, 0.9, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_exit_order_type(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 4.0, 5.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                slippage=0.1,
                stop_order_type="limit",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 16.52892561983471, 4.25, 0.0, 1, 1, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 2, 16.52892561983471, 2.5, 0.0, 1, 1, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                slippage=0.1,
                stop_order_type="limit",
                stop_limit_delta=0.25,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 4, 16.52892561983471, 5.3125, 0.0, 1, 1, 0),
                    (0, 1, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                    (1, 1, 0, 2, 3, 16.52892561983471, 3.125, 0.0, 1, 1, 0),
                    (0, 2, 0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_stop_exit_type(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                size=1,
                sl_stop=0.1,
                stop_exit_type=[["close", "closereduce", "reverse", "reversereduce"]],
                accumulate=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 1.0, 4.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 1.0, 4.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 2.0, 4.0, 0.0, 1, 0, 0),
                    (0, 3, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 1, 1, 1.0, 4.0, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                size=1,
                sl_stop=0.1,
                stop_exit_type=[["close", "closereduce", "reverse", "reversereduce"]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 1, 1, 1.0, 4.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 1, 1, 1.0, 4.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 1, 1, 2.0, 4.0, 0.0, 1, 0, 0),
                    (0, 3, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 1, 1, 2.0, 4.0, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_stop_update(self):
        entries = pd.Series([True, True, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        sl_stop = pd.Series([0.4, np.nan, np.nan, np.nan, np.nan])
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                accumulate=True,
                size=1.0,
                sl_stop=sl_stop,
                upon_stop_update=[["keep", "override", "overridenan"]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 1, 1, 1, 1.0, 4.0, 0.0, 0, 0, -1),
                    (2, 0, 0, 2, 2, 2.0, 3.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 1, 1, 1, 1.0, 4.0, 0.0, 0, 0, -1),
                    (2, 1, 0, 2, 2, 2.0, 3.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 1, 1, 1, 1.0, 4.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        sl_stop = pd.Series([0.4, 0.4, np.nan, np.nan, np.nan])
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                accumulate=True,
                size=1.0,
                sl_stop=sl_stop,
                upon_stop_update=[["keep", "override"]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 1, 1, 1, 1.0, 4.0, 0.0, 0, 0, -1),
                    (2, 0, 0, 2, 2, 2.0, 3.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 1.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 1, 1, 1, 1.0, 4.0, 0.0, 0, 0, -1),
                    (2, 1, 1, 3, 3, 2.0, 2.0, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_adj_stop_conflict_long(self):
        open = pd.Series([5.1, 4.1, 3.1, 2.1, 1.1], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, True, False, False, False], index=price.index)
        exits2 = pd.Series([False, False, False, True, False], index=price.index)
        price2 = pd.Series([5.0, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)
        price3 = pd.Series([5.0, np.inf, np.inf, 2.05, np.inf], index=price.index)
        price4 = pd.Series([5.0, np.inf, np.inf, np.inf, np.inf], index=price.index)

        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                accumulate=True,
                price=price4,
                sl_stop=0.6,
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 1, 1, 1, 20.0, 4.0, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 1, 1, 1, 20.0, 4.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                accumulate=True,
                price=price2,
                sl_stop=0.6,
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 20.0, 2.1, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 3, 3, 3, 20.0, 2.1, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                accumulate=True,
                price=price3,
                sl_stop=0.6,
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                accumulate=True,
                price=price3,
                sl_stop=0.6,
                stop_exit_price="close",
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 3, 3, 3, 20.0, 2.05, 0.0, 1, 0, -1),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 3, 3, 3, 20.0, 2.05, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                accumulate=True,
                price=price4,
                sl_stop=0.6,
                stop_exit_price="close",
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_adj_stop_conflict_short(self):
        open = pd.Series([0.9, 1.9, 2.9, 3.9, 4.9], index=price.index)
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, True, False, False, False], index=price.index)
        exits2 = pd.Series([False, False, False, True, False], index=price.index)
        price2 = pd.Series([1.0, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)
        price3 = pd.Series([1.0, np.inf, np.inf, 3.95, np.inf], index=price.index)
        price4 = pd.Series([1.0, np.inf, np.inf, np.inf, np.inf], index=price.index)

        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                accumulate=True,
                price=price4,
                sl_stop=3.0,
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 1, 1, 1, 100.0, 2.0, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 1, 1, 1, 100.0, 2.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                accumulate=True,
                price=price2,
                sl_stop=3.0,
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 51.282051282051285, 3.9, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 3, 3, 3, 51.282051282051285, 3.9, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                accumulate=True,
                price=price3,
                sl_stop=3.0,
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                accumulate=True,
                price=price3,
                sl_stop=3.0,
                stop_exit_price="close",
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 3, 3, 3, 50.63291139240506, 3.95, 0.0, 0, 0, -1),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 3, 3, 3, 50.63291139240506, 3.95, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits2,
                accumulate=True,
                price=price4,
                sl_stop=3.0,
                stop_exit_price="close",
                upon_adj_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_opp_stop_conflict_long(self):
        open = pd.Series([5.1, 4.1, 3.1, 2.1, 1.1], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        entries = pd.Series([True, True, False, False, False], index=price.index)
        entries2 = pd.Series([True, False, False, True, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        price2 = pd.Series([5.0, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)
        price3 = pd.Series([5.0, np.inf, np.inf, 2.05, np.inf], index=price.index)
        price4 = pd.Series([5.0, np.inf, np.inf, np.inf, np.inf], index=price.index)

        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                accumulate=True,
                price=price4,
                sl_stop=0.6,
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                accumulate=True,
                price=price2,
                sl_stop=0.6,
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 4, 4, 20.0, 1.1, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                accumulate=True,
                price=price3,
                sl_stop=0.6,
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                accumulate=True,
                price=price3,
                sl_stop=0.6,
                stop_exit_price="close",
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 4, 4, 20.0, 1.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                accumulate=True,
                price=price4,
                sl_stop=0.6,
                stop_exit_price="close",
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 1, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 1, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 2, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 2, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                    (0, 3, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 3, 0, 3, 3, 20.0, 2.0, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_upon_opp_stop_conflict_short(self):
        open = pd.Series([0.9, 1.9, 2.9, 3.9, 4.9], index=price.index)
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)
        entries = pd.Series([True, True, False, False, False], index=price.index)
        entries2 = pd.Series([True, False, False, True, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        price2 = pd.Series([1.0, -np.inf, -np.inf, -np.inf, -np.inf], index=price.index)
        price3 = pd.Series([1.0, np.inf, np.inf, 3.95, np.inf], index=price.index)
        price4 = pd.Series([1.0, np.inf, np.inf, np.inf, np.inf], index=price.index)

        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries,
                exits=exits,
                accumulate=True,
                price=price4,
                sl_stop=3.0,
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                accumulate=True,
                price=price2,
                sl_stop=3.0,
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 4, 4, 40.816326530612244, 4.9, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                accumulate=True,
                price=price3,
                sl_stop=3.0,
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                accumulate=True,
                price=price3,
                sl_stop=3.0,
                stop_exit_price="close",
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 4, 4, 40.0, 5.0, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=open,
                close=close,
                entries=entries2,
                exits=exits,
                accumulate=True,
                price=price4,
                sl_stop=3.0,
                stop_exit_price="close",
                upon_opp_stop_conflict=[
                    ["KeepIgnore", "KeepExecute", "CancelIgnore", "CancelExecute"]
                ],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 0, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 1, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 2, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 2, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                    (0, 3, 0, 0, 0, 100.0, 1.0, 0.0, 1, 0, -1),
                    (1, 3, 0, 3, 3, 50.0, 4.0, 0.0, 0, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_adjust_func(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)

        @njit
        def adjust_sl_func_nb(c, dur):
            if c.last_sl_info["init_idx"][c.col] != -1:
                if c.i - c.last_sl_info["init_idx"][c.col] >= dur:
                    c.last_sl_info["stop"][c.col] = 0.0

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=np.inf,
                adjust_func_nb=adjust_sl_func_nb,
                adjust_args=(2,),
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 20.0, 5.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 2, 2, 20.0, 3.0, 0.0, 1, 0, 0),
                ],
                dtype=fs_order_dt,
            ),
        )

        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)

        @njit
        def adjust_tp_func_nb(c, dur):
            if c.last_tp_info["init_idx"][c.col] != -1:
                if c.i - c.last_tp_info["init_idx"][c.col] >= dur:
                    c.last_tp_info["stop"][c.col] = 0.0

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=np.inf,
                adjust_func_nb=adjust_tp_func_nb,
                adjust_args=(2,),
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 0, 2, 2, 100.0, 3.0, 0.0, 1, 0, 3),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_max_order_records(self):
        assert from_signals_both(close=price_wide).order_records.shape[0] == 6
        assert from_signals_both(close=price_wide, max_order_records=2).order_records.shape[0] == 6
        assert from_signals_both(close=price_wide, max_order_records=0).order_records.shape[0] == 0
        with pytest.raises(Exception):
            from_signals_both(close=price_wide, max_order_records=1)

    def test_max_log_records(self):
        assert from_signals_both(close=price_wide, log=True).log_records.shape[0] == 6
        assert (
            from_signals_both(close=price_wide, log=True, max_log_records=2).log_records.shape[0]
            == 6
        )
        assert (
            from_signals_both(close=price_wide, log=True, max_log_records=0).log_records.shape[0]
            == 0
        )
        with pytest.raises(Exception):
            from_signals_both(close=price_wide, log=True, max_log_records=1)

    def test_jitted_parallel(self):
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        entries2 = pd.concat(
            (entries, entries.vbt.signals.fshift(1), entries.vbt.signals.fshift(2)), axis=1
        )
        entries2.columns = price_wide2.columns
        exits2 = pd.concat(
            (exits, exits.vbt.signals.fshift(1), exits.vbt.signals.fshift(2)), axis=1
        )
        exits2.columns = price_wide2.columns
        pf = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200, 300],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            log=True,
            jitted=dict(parallel=True),
        )
        pf2 = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200, 300],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            log=True,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            jitted=dict(parallel=True),
        )
        pf2 = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)

    def test_chunked(self):
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        entries2 = pd.concat(
            (entries, entries.vbt.signals.fshift(1), entries.vbt.signals.fshift(2)), axis=1
        )
        entries2.columns = price_wide2.columns
        exits2 = pd.concat(
            (exits, exits.vbt.signals.fshift(1), exits.vbt.signals.fshift(2)), axis=1
        )
        exits2.columns = price_wide2.columns
        pf = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200, 300],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            log=True,
            chunked=True,
        )
        pf2 = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200, 300],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            log=True,
            chunked=False,
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            chunked=True,
        )
        pf2 = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            chunked=False,
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)

    def test_cash_earnings(self):
        pf = vbt.Portfolio.from_signals(
            1, entries=True, cash_earnings=[0, 1, 2, 3], accumulate=True
        )
        assert_series_equal(pf.cash_earnings, pd.Series([0.0, 1.0, 2.0, 3.0]))
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 2, 2, 2, 1.0, 1.0, 0.0, 0, 0, -1),
                    (2, 0, 3, 3, 3, 2.0, 1.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    def test_cash_dividends(self):
        pf = vbt.Portfolio.from_signals(
            1, entries=True, size=np.inf, cash_dividends=[0, 1, 2, 3], accumulate=True
        )
        assert_series_equal(pf.cash_earnings, pd.Series([0.0, 100.0, 400.0, 1800.0]))
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 0, 0, 100.0, 1.0, 0.0, 0, 0, -1),
                    (1, 0, 2, 2, 2, 100.0, 1.0, 0.0, 0, 0, -1),
                    (2, 0, 3, 3, 3, 400.0, 1.0, 0.0, 0, 0, -1),
                ],
                dtype=fs_order_dt,
            ),
        )

    @pytest.mark.parametrize("test_group_by", [False, np.array([0, 0, 1])])
    @pytest.mark.parametrize("test_cash_sharing", [False, True])
    def test_save_returns(self, test_group_by, test_cash_sharing):
        assert_frame_equal(
            from_signals_both(
                close=price_wide,
                save_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_signals_both(
                close=price_wide,
                save_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )
        assert_frame_equal(
            from_signals_longonly(
                close=price_wide,
                save_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_signals_longonly(
                close=price_wide,
                save_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )
        assert_frame_equal(
            from_signals_shortonly(
                close=price_wide,
                save_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_signals_shortonly(
                close=price_wide,
                save_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )

    def test_staticized(self, tmp_path):
        close = [1, 2, 3, 4]
        entries = [True, False, False, False]
        exits = [False, True, False, False]
        short_entries = [False, False, True, False]
        short_exits = [False, False, False, True]
        assert_records_close(
            vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
            ).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                direction="both",
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                direction="both",
            ).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_signals(
                close,
                size=0.5,
                size_type="targetpercent",
                order_mode=True,
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            vbt.Portfolio.from_signals(
                close,
                size=0.5,
                size_type="targetpercent",
                order_mode=True,
            ).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_signals(
                close,
                size=0.5,
                size_type="targetpercent",
                order_mode=True,
                adjust_func_nb=adjust_func_nb,
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            vbt.Portfolio.from_signals(
                close,
                size=0.5,
                size_type="targetpercent",
                order_mode=True,
                adjust_func_nb=adjust_func_nb,
            ).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_signals(
                pd.Series([1, 2, 3, 4, 5]),
                signal_func_nb=signal_func_nb,
                signal_args=(vbt.Rep("long_num_arr"), vbt.Rep("short_num_arr")),
                broadcast_named_args=dict(
                    long_num_arr=pd.Series([1, 0, -1, 0, 0]),
                    short_num_arr=pd.Series([0, 1, 0, 1, -1]),
                ),
                size=1,
                upon_opposite_entry="ignore",
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            vbt.Portfolio.from_signals(
                pd.Series([1, 2, 3, 4, 5]),
                signal_func_nb=signal_func_nb,
                signal_args=(vbt.Rep("long_num_arr"), vbt.Rep("short_num_arr")),
                broadcast_named_args=dict(
                    long_num_arr=pd.Series([1, 0, -1, 0, 0]),
                    short_num_arr=pd.Series([0, 1, 0, 1, -1]),
                ),
                size=1,
                upon_opposite_entry="ignore",
            ).order_records,
        )


# ############# from_holding ############# #


class TestFromHolding:
    def test_from_holding(self, tmp_path):
        df = pd.DataFrame(
            [
                [1, np.nan, np.nan],
                [2, 5, np.nan],
                [3, 6, 8],
                [4, 7, 9],
            ]
        )
        assert_records_close(
            vbt.Portfolio.from_holding(df[0], dynamic_mode=False).order_records,
            vbt.Portfolio.from_holding(df[0], dynamic_mode=True).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_holding(df[0], dynamic_mode=False, close_at_end=True).order_records,
            vbt.Portfolio.from_holding(df[0], dynamic_mode=True, close_at_end=True).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_holding(df[2], dynamic_mode=False, close_at_end=True).order_records,
            vbt.Portfolio.from_holding(df[2], dynamic_mode=True, close_at_end=True).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_holding(df, dynamic_mode=False).order_records,
            vbt.Portfolio.from_holding(df, dynamic_mode=True).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_holding(df, dynamic_mode=False, close_at_end=True).order_records,
            vbt.Portfolio.from_holding(df, dynamic_mode=True, close_at_end=True).order_records,
        )
        entries = pd.Series.vbt.signals.empty_like(df[0])
        entries.iloc[0] = True
        exits = pd.Series.vbt.signals.empty_like(df[0])
        exits.iloc[-1] = True
        assert_records_close(
            vbt.Portfolio.from_holding(
                df[0], at_first_valid_in=None, close_at_end=True
            ).order_records,
            vbt.Portfolio.from_signals(df[0], entries, exits, accumulate=False).order_records,
        )
        entries = pd.Series.vbt.signals.empty_like(df[2])
        entries.iloc[0] = True
        exits = pd.Series.vbt.signals.empty_like(df[2])
        exits.iloc[-1] = True
        assert_records_close(
            vbt.Portfolio.from_holding(
                df[2], at_first_valid_in=None, close_at_end=True
            ).order_records,
            vbt.Portfolio.from_signals(df[2], entries, exits, accumulate=False).order_records,
        )
        entries = pd.DataFrame.vbt.signals.empty_like(df)
        entries.iloc[0] = True
        exits = pd.DataFrame.vbt.signals.empty_like(df)
        exits.iloc[-1] = True
        assert_records_close(
            vbt.Portfolio.from_holding(df, at_first_valid_in=None, close_at_end=True).order_records,
            vbt.Portfolio.from_signals(df, entries, exits, accumulate=False).order_records,
        )

    def test_staticized(self, tmp_path):
        df = pd.DataFrame(
            [
                [1, np.nan, np.nan],
                [2, 5, np.nan],
                [3, 6, 8],
                [4, 7, 9],
            ]
        )
        assert_records_close(
            vbt.Portfolio.from_holding(
                df,
                dynamic_mode=True,
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            vbt.Portfolio.from_holding(df, dynamic_mode=True).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_signals(
                df,
                signal_func_nb="holding_enex_signal_func_nb",
                signal_args=(Direction.LongOnly, True),
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            vbt.Portfolio.from_holding(
                df,
                dynamic_mode=True,
                close_at_end=True,
            ).order_records,
        )

    @pytest.mark.parametrize("test_ls", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_sim_range(self, test_ls, test_flexible):
        if test_ls:
            if test_flexible:
                _from_signals_both = partial(from_ls_signals_both, adjust_func_nb=adjust_func_nb)
            else:
                _from_signals_both = from_ls_signals_both
        else:
            if test_flexible:
                _from_signals_both = partial(from_signals_both, adjust_func_nb=adjust_func_nb)
            else:
                _from_signals_both = from_signals_both
        _entries_wide = entries_wide.copy()
        _exits_wide = exits_wide.copy()
        _entries_wide.iloc[:1] = False
        _exits_wide.iloc[:1] = False
        assert_records_close(
            _from_signals_both(entries=entries_wide, exits=exits_wide, sim_start=1).order_records,
            _from_signals_both(entries=_entries_wide, exits=_exits_wide).order_records,
        )
        np.testing.assert_array_equal(
            _from_signals_both(entries=entries_wide, exits=exits_wide, sim_start=1).sim_start,
            np.array([1, 1, 1]),
        )
        assert_records_close(
            _from_signals_both(entries=entries_wide, exits=exits_wide, sim_start=1).order_records,
            _from_signals_both(
                entries=_entries_wide, exits=_exits_wide, sim_start="auto"
            ).order_records,
        )
        _entries_wide = entries_wide.copy()
        _exits_wide = exits_wide.copy()
        _entries_wide.iloc[2:] = False
        _exits_wide.iloc[2:] = False
        assert_records_close(
            _from_signals_both(entries=entries_wide, exits=exits_wide, sim_end=2).order_records,
            _from_signals_both(entries=_entries_wide, exits=_exits_wide).order_records,
        )
        np.testing.assert_array_equal(
            _from_signals_both(entries=entries_wide, exits=exits_wide, sim_end=2).sim_end,
            np.array([2, 2, 2]),
        )
        assert_records_close(
            _from_signals_both(entries=entries_wide, exits=exits_wide, sim_end=2).order_records,
            _from_signals_both(
                entries=_entries_wide, exits=_exits_wide, sim_end="auto"
            ).order_records,
        )
        _entries_wide = entries_wide.copy()
        _exits_wide = exits_wide.copy()
        _entries_wide.iloc[:1, 0] = False
        _entries_wide.iloc[:2, 1] = False
        _entries_wide.iloc[:3, 2] = False
        _exits_wide.iloc[:1, 0] = False
        _exits_wide.iloc[:2, 1] = False
        _exits_wide.iloc[:3, 2] = False
        assert_records_close(
            _from_signals_both(
                entries=entries_wide,
                exits=exits_wide,
                sim_start=[1, 2, 3],
            ).order_records,
            _from_signals_both(
                entries=_entries_wide,
                exits=_exits_wide,
            ).order_records,
        )
        np.testing.assert_array_equal(
            _from_signals_both(
                entries=entries_wide, exits=exits_wide, sim_start=[1, 2, 3]
            ).sim_start,
            np.array([1, 2, 3]),
        )
        assert_records_close(
            _from_signals_both(
                entries=entries_wide,
                exits=exits_wide,
                sim_start=[1, 2, 3],
            ).order_records,
            _from_signals_both(
                entries=_entries_wide,
                exits=_exits_wide,
                sim_start="auto",
            ).order_records,
        )
        with pytest.raises(Exception):
            _from_signals_both(
                entries=entries_wide,
                exits=exits_wide,
                sim_start=[1, 2],
            )
        if not test_flexible:
            assert_records_close(
                _from_signals_both(
                    close=price_wide,
                    entries=entries_wide,
                    exits=exits_wide,
                    group_by=[0, 0, 1],
                    sim_start=[1, 2, 3],
                ).order_records,
                _from_signals_both(
                    close=price_wide,
                    entries=_entries_wide,
                    exits=_exits_wide,
                    group_by=[0, 0, 1],
                ).order_records,
            )
            assert_records_close(
                _from_signals_both(
                    close=price_wide,
                    entries=entries_wide,
                    exits=exits_wide,
                    group_by=[0, 0, 1],
                    sim_start=[1, 2, 3],
                ).order_records,
                _from_signals_both(
                    close=price_wide,
                    entries=_entries_wide,
                    exits=_exits_wide,
                    group_by=[0, 0, 1],
                    sim_start="auto",
                ).order_records,
            )
        if not test_flexible:
            np.testing.assert_array_equal(
                _from_signals_both(
                    entries=entries_wide, exits=exits_wide, sim_start=[1, 2, 3], group_by=[0, 0, 1]
                ).sim_start,
                np.array([1, 3]),
            )
        else:
            np.testing.assert_array_equal(
                _from_signals_both(
                    entries=entries_wide, exits=exits_wide, sim_start=[1, 2, 3], group_by=[0, 0, 1]
                ).sim_start,
                np.array([1, 3]),
            )
        _entries_wide = entries_wide.copy()
        _exits_wide = exits_wide.copy()
        _entries_wide.iloc[:1, 0] = False
        _entries_wide.iloc[:2, 1] = False
        _entries_wide.iloc[:2, 2] = False
        _exits_wide.iloc[:1, 0] = False
        _exits_wide.iloc[:2, 1] = False
        _exits_wide.iloc[:2, 2] = False
        assert_records_close(
            _from_signals_both(
                entries=entries_wide,
                exits=exits_wide,
                group_by=[0, 0, 1],
                cash_sharing=True,
                sim_start=[1, 2],
            ).order_records,
            _from_signals_both(
                entries=_entries_wide,
                exits=_exits_wide,
                group_by=[0, 0, 1],
                cash_sharing=True,
            ).order_records,
        )
        assert_records_close(
            _from_signals_both(
                entries=entries_wide,
                exits=exits_wide,
                group_by=[0, 0, 1],
                cash_sharing=True,
                sim_start=[1, 2],
            ).order_records,
            _from_signals_both(
                entries=_entries_wide,
                exits=_exits_wide,
                group_by=[0, 0, 1],
                cash_sharing=True,
                sim_start="auto",
            ).order_records,
        )

    def test_ctx_helpers(self):
        @njit
        def post_signal_func_nb(c):
            _ = vbt.pf_nb.get_position_nb(c)
            _ = vbt.pf_nb.in_position_nb(c)
            _ = vbt.pf_nb.in_long_position_nb(c)
            _ = vbt.pf_nb.in_short_position_nb(c)
            _ = vbt.pf_nb.get_n_active_positions_nb(c)
            _ = vbt.pf_nb.get_n_active_positions_nb(c, all_groups=True)
            _ = vbt.pf_nb.get_cash_nb(c)
            _ = vbt.pf_nb.get_locked_cash_nb(c)
            _ = vbt.pf_nb.get_free_cash_nb(c)
            _ = vbt.pf_nb.has_free_cash_nb(c)
            _ = vbt.pf_nb.get_val_price_nb(c)
            _ = vbt.pf_nb.get_value_nb(c)
            _ = vbt.pf_nb.get_leverage_nb(c)
            _ = vbt.pf_nb.get_position_value_nb(c)
            _ = vbt.pf_nb.get_allocation_nb(c)
            _ = vbt.pf_nb.get_order_count_nb(c)
            _ = vbt.pf_nb.get_order_records_nb(c)
            _ = vbt.pf_nb.has_orders_nb(c)
            _ = vbt.pf_nb.order_filled_nb(c)
            _ = vbt.pf_nb.order_opened_position_nb(c)
            _ = vbt.pf_nb.order_increased_position_nb(c)
            _ = vbt.pf_nb.order_decreased_position_nb(c)
            _ = vbt.pf_nb.order_closed_position_nb(c)
            _ = vbt.pf_nb.order_reversed_position_nb(c)
            _ = vbt.pf_nb.get_limit_target_price_nb(c)
            _ = vbt.pf_nb.get_sl_target_price_nb(c)
            _ = vbt.pf_nb.get_tsl_target_price_nb(c)
            _ = vbt.pf_nb.get_tp_target_price_nb(c)
            _ = vbt.pf_nb.get_entry_trade_records_nb(c)
            _ = vbt.pf_nb.get_exit_trade_records_nb(c)
            _ = vbt.pf_nb.get_position_records_nb(c)
            _ = vbt.pf_nb.stop_sim_nb(c)
            _ = vbt.pf_nb.get_exec_state_nb(c)
            _ = vbt.pf_nb.get_price_area_nb(c)
            _ = vbt.pf_nb.get_order_size_nb(c, np.inf)
            _ = vbt.pf_nb.get_order_value_nb(c, np.inf)
            _ = vbt.pf_nb.get_order_result_nb(c, vbt.pf_nb.close_position_nb())

        _ = vbt.PF.from_signals(
            [1],
            long_entries=[[True, False]],
            short_entries=[[False, True]],
            init_cash=[100, 200],
            post_signal_func_nb=post_signal_func_nb,
            sl_stop=0.1,
            tsl_stop=0.2,
            tp_stop=0.3,
        )

        @njit
        def post_segment_func_nb(c):
            for col in range(c.from_col, c.to_col):
                _ = vbt.pf_nb.get_col_position_nb(c, col)
                _ = vbt.pf_nb.col_in_position_nb(c, col)
                _ = vbt.pf_nb.col_in_long_position_nb(c, col)
                _ = vbt.pf_nb.col_in_short_position_nb(c, col)
                _ = vbt.pf_nb.get_col_cash_nb(c, col)
                _ = vbt.pf_nb.get_col_locked_cash_nb(c, col)
                _ = vbt.pf_nb.get_col_free_cash_nb(c, col)
                _ = vbt.pf_nb.col_has_free_cash_nb(c, col)
                _ = vbt.pf_nb.get_col_val_price_nb(c, col)
                _ = vbt.pf_nb.get_col_value_nb(c, col)
                _ = vbt.pf_nb.get_col_leverage_nb(c, col)
                _ = vbt.pf_nb.get_col_position_value_nb(c, col)
                _ = vbt.pf_nb.get_col_allocation_nb(c, col)
                _ = vbt.pf_nb.get_col_order_count_nb(c, col)
                _ = vbt.pf_nb.get_col_order_records_nb(c, col)
                _ = vbt.pf_nb.col_has_orders_nb(c, col)
                _ = vbt.pf_nb.get_col_limit_target_price_nb(c, col)
                _ = vbt.pf_nb.get_col_sl_target_price_nb(c, col)
                _ = vbt.pf_nb.get_col_tsl_target_price_nb(c, col)
                _ = vbt.pf_nb.get_col_tp_target_price_nb(c, col)
                _ = vbt.pf_nb.get_col_entry_trade_records_nb(c, col)
                _ = vbt.pf_nb.get_col_exit_trade_records_nb(c, col)
                _ = vbt.pf_nb.get_col_position_records_nb(c, col)

            _ = vbt.pf_nb.get_n_active_positions_nb(c)
            _ = vbt.pf_nb.get_n_active_positions_nb(c, all_groups=True)
            _ = vbt.pf_nb.get_group_cash_nb(c, c.group)
            _ = vbt.pf_nb.get_group_free_cash_nb(c, c.group)
            _ = vbt.pf_nb.get_group_value_nb(c, c.group)
            _ = vbt.pf_nb.get_group_position_value_nb(c, c.group)
            _ = vbt.pf_nb.stop_group_sim_nb(c, c.group)
            _ = vbt.pf_nb.stop_sim_nb(c)

        _ = vbt.PF.from_signals(
            [1],
            long_entries=[[True, False]],
            short_entries=[[False, True]],
            init_cash=[100, 200],
            post_segment_func_nb=post_segment_func_nb,
            group_by=True,
            sl_stop=0.1,
            tsl_stop=0.2,
            tp_stop=0.3,
        )


# ############# from_random_signals ############# #


class TestFromRandomSignals:
    def test_from_random_n(self):
        result = vbt.Portfolio.from_random_signals(price, n=2, seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [False, True, False, True, False],
                [False, False, True, False, True],
            ).order_records,
        )
        assert_index_equal(result.wrapper.index, price.vbt.wrapper.index)
        assert_index_equal(result.wrapper.columns, price.vbt.wrapper.columns)
        result = vbt.Portfolio.from_random_signals(price, n=[1, 2], seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [[False, False], [False, True], [False, False], [True, True], [False, False]],
                [[False, False], [False, False], [False, True], [False, False], [True, True]],
            ).order_records,
        )
        assert_index_equal(
            result.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        assert_index_equal(result.wrapper.columns, pd.Index([1, 2], dtype="int64", name="randnx_n"))

    def test_from_random_prob(self):
        result = vbt.Portfolio.from_random_signals(price, prob=0.5, seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [True, False, False, False, False],
                [False, False, False, False, True],
            ).order_records,
        )
        assert_index_equal(result.wrapper.index, price.vbt.wrapper.index)
        assert_index_equal(result.wrapper.columns, price.vbt.wrapper.columns)
        result = vbt.Portfolio.from_random_signals(price, prob=[0.25, 0.5], seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [[False, True], [False, False], [False, False], [False, False], [True, False]],
                [[False, False], [False, True], [False, False], [False, False], [False, False]],
            ).order_records,
        )
        assert_index_equal(
            result.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        assert_index_equal(
            result.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(0.25, 0.25), (0.5, 0.5)], names=["rprobnx_entry_prob", "rprobnx_exit_prob"]
            ),
        )
