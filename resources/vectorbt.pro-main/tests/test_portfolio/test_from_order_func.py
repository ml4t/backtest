import os
import uuid
from copy import deepcopy
from typing import NamedTuple

import pytest
import vectorbtpro as vbt
from numba import njit, typeof
from numba.typed import List
from vectorbtpro._dtypes import *
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.enums import *

from tests.utils import *

seed = 42

day_dt = np.timedelta64(86400000000000)

price = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    index=pd.date_range("2020", periods=5),
)
price_wide = price.vbt.tile(3, keys=["a", "b", "c"])

group_by = pd.Index(["first", "first", "second"], name="group")


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# from_order_func ############# #


@njit
def order_func_nb(c, size):
    _size = nb.select_nb(c, size)
    return nb.order_nb(_size if c.i % 2 == 0 else -_size)


@njit
def log_order_func_nb(c, size):
    _size = nb.select_nb(c, size)
    return nb.order_nb(_size if c.i % 2 == 0 else -_size, log=True)


@njit
def flex_order_func_nb(c, size):
    if c.call_idx < c.group_len:
        _size = nb.select_from_col_nb(c, c.from_col + c.call_idx, size)
        return c.from_col + c.call_idx, nb.order_nb(_size if c.i % 2 == 0 else -_size)
    return -1, nb.order_nothing_nb()


@njit
def log_flex_order_func_nb(c, size):
    if c.call_idx < c.group_len:
        _size = nb.select_from_col_nb(c, c.from_col + c.call_idx, size)
        return c.from_col + c.call_idx, nb.order_nb(_size if c.i % 2 == 0 else -_size, log=True)
    return -1, nb.order_nothing_nb()


def from_order_func(close, order_func_nb, *args, flexible=False, **kwargs):
    if not flexible:
        return vbt.Portfolio.from_order_func(
            close,
            order_func_nb=order_func_nb,
            order_args=args,
            **kwargs,
        )
    return vbt.Portfolio.from_order_func(
        close,
        flex_order_func_nb=order_func_nb,
        flex_order_args=args,
        **kwargs,
    )


class InOutputs(NamedTuple):
    custom_1d_arr: vbt.RepEval
    custom_2d_arr: vbt.RepEval
    custom_rec_arr: vbt.RepEval


class TestFromOrderFunc:
    def test_data(self):
        data = vbt.RandomOHLCData.pull(
            [0, 1],
            start="2020-01-01",
            end="2020-02-01",
            tick_freq="1h",
            freq="1d",
            seed=42,
        )
        pf = from_order_func(data, order_func_nb, np.array([[np.inf]]))
        assert pf.open is not None
        assert pf.high is not None
        assert pf.low is not None
        assert pf.close is not None
        pf = from_order_func(data.get("Close"), order_func_nb, np.array([[np.inf]]))
        assert pf.open is None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = from_order_func(data[["Open", "Close"]], order_func_nb, np.array([[np.inf]]))
        assert pf.open is not None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = from_order_func(data["Close"], order_func_nb, np.array([[np.inf]]))
        assert pf.open is None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = from_order_func(
            data["Close"], order_func_nb, np.array([[np.inf]]), open=data.get("Open")
        )
        assert pf.open is not None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_one_column(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = from_order_func(
            price.tolist(),
            order_func,
            np.array([[np.inf]]),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_order_func(
            price,
            order_func,
            np.array([[np.inf]]),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
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

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    @pytest.mark.parametrize("test_jitted", [False, True])
    def test_multiple_columns(self, test_row_wise, test_flexible, test_jitted):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = from_order_func(
            price_wide,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            row_wise=test_row_wise,
            flexible=test_flexible,
            jitted=test_jitted,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 2, 1.0, 3.0, 0.0, 0),
                    (3, 1, 3, 1.0, 4.0, 0.0, 1),
                    (4, 1, 4, 1.0, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
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

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_price_area(self, test_row_wise, test_flexible):
        @njit
        def order_func2_nb(c, price, price_area_vio_mode):
            _price = nb.select_nb(c, price)
            _price_area_vio_mode = nb.select_nb(c, price_area_vio_mode)
            return nb.order_nb(
                1 if c.i % 2 == 0 else -1,
                _price,
                slippage=0.1,
                price_area_vio_mode=_price_area_vio_mode,
            )

        @njit
        def flex_order_func2_nb(c, price, price_area_vio_mode):
            if c.call_idx < c.group_len:
                _price = nb.select_from_col_nb(c, c.from_col + c.call_idx, price)
                _price_area_vio_mode = nb.select_from_col_nb(
                    c, c.from_col + c.call_idx, price_area_vio_mode
                )
                return c.from_col + c.call_idx, nb.order_nb(
                    1 if c.i % 2 == 0 else -1,
                    _price,
                    slippage=0.1,
                    price_area_vio_mode=_price_area_vio_mode,
                )
            return -1, nb.order_nothing_nb()

        order_func = flex_order_func2_nb if test_flexible else order_func2_nb
        assert_records_close(
            from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(
                    price=[[0.5, np.inf, 5]], price_area_vio_mode=PriceAreaVioMode.Ignore
                ),
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 0.55, 0.0, 0),
                    (0, 1, 0, 1.0, 3.3, 0.0, 0),
                    (0, 2, 0, 1.0, 5.5, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(
                    price=[[0.5, np.inf, 5]], price_area_vio_mode=PriceAreaVioMode.Cap
                ),
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (0, 1, 0, 1.0, 3.0, 0.0, 0),
                    (0, 2, 0, 1.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(price=0.5, price_area_vio_mode=PriceAreaVioMode.Error),
            )
        with pytest.raises(Exception):
            from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(price=np.inf, price_area_vio_mode=PriceAreaVioMode.Error),
            )
        with pytest.raises(Exception):
            from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(price=5, price_area_vio_mode=PriceAreaVioMode.Error),
            )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_group_by(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = from_order_func(
            price_wide,
            order_func,
            np.array([[np.inf]]),
            group_by=np.array([0, 0, 1]),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 1, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 1, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
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

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_cash_sharing(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = from_order_func(
            price_wide,
            order_func,
            np.array([[np.inf]]),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
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

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_call_seq(self, test_row_wise):
        pf = from_order_func(
            price_wide,
            order_func_nb,
            np.array([[np.inf]]),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            row_wise=test_row_wise,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        )
        pf = from_order_func(
            price_wide,
            order_func_nb,
            np.array([[np.inf]]),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            call_seq="reversed",
            row_wise=test_row_wise,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 1, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 1, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        pf = from_order_func(
            price_wide,
            order_func_nb,
            np.array([[np.inf]]),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            call_seq="random",
            seed=seed,
            row_wise=test_row_wise,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 1, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 1, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        with pytest.raises(Exception):
            from_order_func(
                price_wide,
                order_func_nb,
                np.array([[np.inf]]),
                group_by=np.array([0, 0, 1]),
                cash_sharing=True,
                call_seq="auto",
                row_wise=test_row_wise,
            )

        target_hold_value = pd.DataFrame(
            {
                "a": [0.0, 70.0, 30.0, 0.0, 70.0],
                "b": [30.0, 0.0, 70.0, 30.0, 30.0],
                "c": [70.0, 30.0, 0.0, 70.0, 0.0],
            },
            index=price.index,
        )

        @njit
        def pre_segment_func_nb(c, target_hold_value):
            order_size = np.copy(target_hold_value[c.i, c.from_col : c.to_col])
            order_size_type = np.full(c.group_len, SizeType.TargetValue)
            direction = np.full(c.group_len, Direction.Both)
            order_value_out = np.empty(c.group_len, dtype=float_)
            c.last_val_price[c.from_col : c.to_col] = c.close[c.i, c.from_col : c.to_col]
            nb.sort_call_seq_1d_nb(c, order_size, order_size_type, direction, order_value_out)
            return order_size, order_size_type, direction

        @njit
        def pct_order_func_nb(c, order_size, order_size_type, direction):
            col_i = c.call_seq_now[c.call_idx]
            return nb.order_nb(
                order_size[col_i],
                c.close[c.i, col_i],
                size_type=order_size_type[col_i],
                direction=direction[col_i],
            )

        pf = from_order_func(
            price_wide * 0 + 1,
            pct_order_func_nb,
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(target_hold_value.values,),
            row_wise=test_row_wise,
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 1, 0]]),
        )
        assert_frame_equal(pf.get_asset_value(group_by=False), target_hold_value)

        _ = from_order_func(
            price_wide,
            order_func_nb,
            np.array([[np.inf]]),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            row_wise=test_row_wise,
            call_seq=None,
            attach_call_seq=False,
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_target_value(self, test_row_wise, test_flexible):
        @njit
        def target_val_pre_segment_func_nb(c, val_price):
            c.last_val_price[c.from_col : c.to_col] = val_price[c.i]
            return ()

        if test_flexible:

            @njit
            def target_val_order_func_nb(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(
                        50.0, nb.select_from_col_nb(c, col, c.close), size_type=SizeType.TargetValue
                    )
                return -1, nb.order_nothing_nb()

        else:

            @njit
            def target_val_order_func_nb(c):
                return nb.order_nb(50.0, nb.select_nb(c, c.close), size_type=SizeType.TargetValue)

        pf = from_order_func(
            price.iloc[1:],
            target_val_order_func_nb,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 1, 25.0, 3.0, 0.0, 0),
                    (1, 0, 2, 8.333333333333332, 4.0, 0.0, 1),
                    (2, 0, 3, 4.166666666666668, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_order_func(
            price.iloc[1:],
            target_val_order_func_nb,
            pre_segment_func_nb=target_val_pre_segment_func_nb,
            pre_segment_args=(price.iloc[:-1].values,),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 2.0, 0.0, 0),
                    (1, 0, 1, 25.0, 3.0, 0.0, 1),
                    (2, 0, 2, 8.333333333333332, 4.0, 0.0, 1),
                    (3, 0, 3, 4.166666666666668, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_target_percent(self, test_row_wise, test_flexible):
        @njit
        def target_pct_pre_segment_func_nb(c, val_price):
            c.last_val_price[c.from_col : c.to_col] = val_price[c.i]
            return ()

        if test_flexible:

            @njit
            def target_pct_order_func_nb(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(
                        0.5,
                        nb.select_from_col_nb(c, col, c.close),
                        size_type=SizeType.TargetPercent,
                    )
                return -1, nb.order_nothing_nb()

        else:

            @njit
            def target_pct_order_func_nb(c):
                return nb.order_nb(0.5, nb.select_nb(c, c.close), size_type=SizeType.TargetPercent)

        pf = from_order_func(
            price.iloc[1:],
            target_pct_order_func_nb,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 1, 25.0, 3.0, 0.0, 0),
                    (1, 0, 2, 8.333333333333332, 4.0, 0.0, 1),
                    (2, 0, 3, 1.0416666666666679, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_order_func(
            price.iloc[1:],
            target_pct_order_func_nb,
            pre_segment_func_nb=target_pct_pre_segment_func_nb,
            pre_segment_args=(price.iloc[:-1].values,),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 2.0, 0.0, 0),
                    (1, 0, 1, 25.0, 3.0, 0.0, 1),
                    (2, 0, 3, 3.125, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_update_value(self, test_row_wise, test_flexible):
        if test_flexible:

            @njit
            def order_func_nb(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(
                        np.inf if c.i % 2 == 0 else -np.inf,
                        nb.select_from_col_nb(c, col, c.close),
                        fees=0.01,
                        fixed_fees=1.0,
                        slippage=0.01,
                    )
                return -1, nb.order_nothing_nb()

        else:

            @njit
            def order_func_nb(c):
                return nb.order_nb(
                    np.inf if c.i % 2 == 0 else -np.inf,
                    nb.select_nb(c, c.close),
                    fees=0.01,
                    fixed_fees=1.0,
                    slippage=0.01,
                )

        @njit
        def post_order_func_nb(c, value_before, value_now):
            value_before[c.i, c.col] = c.value_before
            value_now[c.i, c.col] = c.value_now

        value_before = np.empty_like(price.values[:, None])
        value_now = np.empty_like(price.values[:, None])

        from_order_func(
            price,
            order_func_nb,
            post_order_func_nb=post_order_func_nb,
            post_order_args=(value_before, value_now),
            row_wise=test_row_wise,
            update_value=False,
            flexible=test_flexible,
        )

        np.testing.assert_array_equal(value_before, value_now)

        from_order_func(
            price,
            order_func_nb,
            post_order_func_nb=post_order_func_nb,
            post_order_args=(value_before, value_now),
            row_wise=test_row_wise,
            update_value=True,
            flexible=test_flexible,
        )

        np.testing.assert_array_equal(
            value_before,
            np.array(
                [
                    [100.0],
                    [97.04930889128518],
                    [185.46988117104038],
                    [82.47853456223027],
                    [104.65775576218029],
                ]
            ),
        )
        np.testing.assert_array_equal(
            value_now,
            np.array(
                [
                    [98.01980198019803],
                    [187.36243097890815],
                    [83.30331990785257],
                    [105.72569204546784],
                    [73.54075125567472],
                ]
            ),
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_states(self, test_row_wise, test_flexible):
        cash_deposits = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [100.0, 0.0], [0.0, 0.0]])
        cash_earnings = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        close = np.array([[1, 1, 1], [np.nan, 2, 2], [3, np.nan, 3], [4, 4, np.nan], [5, 5, 5]])
        open = close - 0.1
        size = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 1], [-1, -1, -1], [1, 1, 1]])
        value_arr1 = np.empty((size.shape[0], 2), dtype=float_)
        value_arr2 = np.empty(size.shape, dtype=float_)
        value_arr3 = np.empty(size.shape, dtype=float_)
        return_arr1 = np.empty((size.shape[0], 2), dtype=float_)
        return_arr2 = np.empty(size.shape, dtype=float_)
        return_arr3 = np.empty(size.shape, dtype=float_)
        pos_info_arr1 = np.empty(size.shape, dtype=trade_dt)
        pos_info_arr2 = np.empty(size.shape, dtype=trade_dt)
        pos_info_arr3 = np.empty(size.shape, dtype=trade_dt)

        def pre_segment_func_nb(c):
            value_arr1[c.i, c.group] = c.last_value[c.group]
            return_arr1[c.i, c.group] = c.last_return[c.group]
            for col in range(c.from_col, c.to_col):
                pos_info_arr1[c.i, col] = c.last_pos_info[col]
            c.last_val_price[c.from_col : c.to_col] = c.last_val_price[c.from_col : c.to_col] + 0.5
            return ()

        if test_flexible:

            def order_func_nb(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    value_arr2[c.i, col] = c.last_value[c.group]
                    return_arr2[c.i, col] = c.last_return[c.group]
                    pos_info_arr2[c.i, col] = c.last_pos_info[col]
                    return col, nb.order_nb(size[c.i, col], fixed_fees=1.0)
                return -1, nb.order_nothing_nb()

        else:

            def order_func_nb(c):
                value_arr2[c.i, c.col] = c.value_now
                return_arr2[c.i, c.col] = c.return_now
                pos_info_arr2[c.i, c.col] = c.pos_info_now
                return nb.order_nb(size[c.i, c.col], fixed_fees=1.0)

        def post_order_func_nb(c):
            value_arr3[c.i, c.col] = c.value_now
            return_arr3[c.i, c.col] = c.return_now
            pos_info_arr3[c.i, c.col] = c.pos_info_now

        from_order_func(
            close,
            order_func_nb,
            pre_segment_func_nb=pre_segment_func_nb,
            post_order_func_nb=post_order_func_nb,
            jitted=False,
            open=open,
            update_value=True,
            ffill_val_price=True,
            group_by=[0, 0, 1],
            cash_sharing=True,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )

        np.testing.assert_array_equal(
            value_arr1,
            np.array([[100.0, 100.0], [98.9, 99.9], [99.9, 99.0], [100.8, 98.0], [200.0, 99.9]]),
        )
        np.testing.assert_array_equal(
            value_arr2,
            np.array(
                [
                    [100.0, 99.0, 100.0],
                    [99.9, 99.9, 100.4],
                    [100.4, 99.0, 99.0],
                    [201.8, 200.0, 98.5],
                    [200.0, 198.6, 100.4],
                ]
            ),
        )
        np.testing.assert_array_equal(
            value_arr3,
            np.array(
                [
                    [99.0, 98.0, 99.0],
                    [99.9, 98.5, 99.0],
                    [99.0, 99.0, 98.0],
                    [200.0, 199.0, 98.5],
                    [198.6, 198.0, 99.0],
                ],
            ),
        )
        np.testing.assert_array_equal(
            return_arr1,
            np.array(
                [
                    [0.0, 0.0],
                    [0.009183673469387813, 0.009090909090909148],
                    [0.014213197969543205, 0.0],
                    [0.018181818181818153, 0.0],
                    [0.0, 0.014213197969543205],
                ]
            ),
        )
        np.testing.assert_array_equal(
            return_arr2,
            np.array(
                [
                    [0.0, -0.01, 0.0],
                    [0.019387755102040875, 0.019387755102040875, 0.0141414141414142],
                    [0.0192893401015229, 0.005076142131979695, 0.0],
                    [0.0282828282828284, 0.010101010101010102, 0.00510204081632653],
                    [0.0, -0.007000000000000029, 0.0192893401015229],
                ]
            ),
        )
        np.testing.assert_array_equal(
            return_arr3,
            np.array(
                [
                    [-0.01, -0.02, -0.01],
                    [0.019387755102040875, 0.00510204081632653, 0.0],
                    [0.005076142131979695, 0.005076142131979695, -0.010101010101010102],
                    [0.010101010101010102, 0.0, 0.00510204081632653],
                    [-0.007000000000000029, -0.01, 0.005076142131979695],
                ]
            ),
        )
        assert_records_close(
            pos_info_arr1.flatten()[3:],
            np.array(
                [
                    (0, 0, 1.0, 0, 0, 1.0, 1.0, -1, -1, np.nan, 0.0, -1.0, -1.0, 0, 0, 0),
                    (
                        0,
                        1,
                        1.0,
                        0,
                        0,
                        1.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        -0.10000000000000009,
                        -0.10000000000000009,
                        0,
                        0,
                        0,
                    ),
                    (
                        0,
                        2,
                        1.0,
                        0,
                        0,
                        1.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        -0.10000000000000009,
                        -0.10000000000000009,
                        0,
                        0,
                        0,
                    ),
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        0.8999999999999999,
                        0.8999999999999999,
                        0,
                        0,
                        0,
                    ),
                    (0, 1, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 2, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (
                        0,
                        0,
                        2.0,
                        0,
                        0,
                        2.0,
                        2.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        1.7999999999999998,
                        0.44999999999999996,
                        0,
                        0,
                        0,
                    ),
                    (0, 1, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (
                        1,
                        2,
                        1.0,
                        2,
                        2,
                        3.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        -1.0,
                        -0.3333333333333333,
                        0,
                        0,
                        1,
                    ),
                    (
                        0,
                        0,
                        2.0,
                        0,
                        0,
                        2.0,
                        2.0,
                        2,
                        -1,
                        4.0,
                        1.0,
                        1.9000000000000004,
                        0.4750000000000001,
                        0,
                        0,
                        0,
                    ),
                    (
                        1,
                        1,
                        1.0,
                        2,
                        3,
                        4.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        -1.9000000000000004,
                        -0.4750000000000001,
                        1,
                        0,
                        1,
                    ),
                    (
                        1,
                        2,
                        1.0,
                        2,
                        2,
                        3.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        0.9000000000000004,
                        0.3000000000000001,
                        0,
                        0,
                        1,
                    ),
                ],
                dtype=trade_dt,
            ),
        )
        assert_records_close(
            pos_info_arr2.flatten()[3:],
            np.array(
                [
                    (0, 0, 1.0, 0, 0, 1.0, 1.0, -1, -1, np.nan, 0.0, -0.5, -0.5, 0, 0, 0),
                    (
                        0,
                        1,
                        1.0,
                        0,
                        0,
                        1.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        0.3999999999999999,
                        0.3999999999999999,
                        0,
                        0,
                        0,
                    ),
                    (
                        0,
                        2,
                        1.0,
                        0,
                        0,
                        1.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        0.3999999999999999,
                        0.3999999999999999,
                        0,
                        0,
                        0,
                    ),
                    (0, 0, 1.0, 0, 0, 1.0, 1.0, -1, -1, np.nan, 0.0, 1.4, 1.4, 0, 0, 0),
                    (0, 1, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 2, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (
                        0,
                        0,
                        2.0,
                        0,
                        0,
                        2.0,
                        2.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        2.8000000000000007,
                        0.7000000000000002,
                        0,
                        0,
                        0,
                    ),
                    (0, 1, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (
                        1,
                        2,
                        1.0,
                        2,
                        2,
                        3.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        -0.5,
                        -0.16666666666666666,
                        0,
                        0,
                        1,
                    ),
                    (
                        0,
                        0,
                        2.0,
                        0,
                        0,
                        2.0,
                        2.0,
                        2,
                        -1,
                        4.0,
                        1.0,
                        2.4000000000000004,
                        0.6000000000000001,
                        0,
                        0,
                        0,
                    ),
                    (
                        1,
                        1,
                        1.0,
                        2,
                        3,
                        4.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        -2.4000000000000004,
                        -0.6000000000000001,
                        1,
                        0,
                        1,
                    ),
                    (
                        1,
                        2,
                        1.0,
                        2,
                        2,
                        3.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        1.4000000000000004,
                        0.4666666666666668,
                        0,
                        0,
                        1,
                    ),
                ],
                dtype=trade_dt,
            ),
        )
        assert_records_close(
            pos_info_arr3.flatten(),
            np.array(
                [
                    (0, 0, 1.0, 0, 0, 1.0, 1.0, -1, -1, np.nan, 0.0, -1.0, -1.0, 0, 0, 0),
                    (0, 1, 1.0, 0, 0, 1.0, 1.0, -1, -1, np.nan, 0.0, -1.0, -1.0, 0, 0, 0),
                    (0, 2, 1.0, 0, 0, 1.0, 1.0, -1, -1, np.nan, 0.0, -1.0, -1.0, 0, 0, 0),
                    (0, 0, 1.0, 0, 0, 1.0, 1.0, -1, -1, np.nan, 0.0, -0.5, -0.5, 0, 0, 0),
                    (0, 1, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 2, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 0, 2.0, 0, 0, 2.0, 2.0, -1, -1, np.nan, 0.0, 0.0, 0.0, 0, 0, 0),
                    (0, 1, 1.0, 0, 0, 1.0, 1.0, 1, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (
                        1,
                        2,
                        1.0,
                        2,
                        2,
                        3.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        -1.0,
                        -0.3333333333333333,
                        0,
                        0,
                        1,
                    ),
                    (0, 0, 2.0, 0, 0, 2.0, 2.0, 2, -1, 4.0, 1.0, 1.0, 0.25, 0, 0, 0),
                    (1, 1, 1.0, 2, 3, 4.0, 1.0, -1, -1, np.nan, 0.0, -1.0, -0.25, 1, 0, 1),
                    (
                        1,
                        2,
                        1.0,
                        2,
                        2,
                        3.0,
                        1.0,
                        -1,
                        -1,
                        np.nan,
                        0.0,
                        -0.5,
                        -0.16666666666666666,
                        0,
                        0,
                        1,
                    ),
                    (0, 0, 3.0, 0, 0, 3.0, 3.0, 2, -1, 4.0, 1.0, 1.0, 0.1111111111111111, 0, 0, 0),
                    (1, 1, 1.0, 2, 3, 4.0, 1.0, 3, 4, 5.0, 1.0, -3.0, -0.75, 1, 1, 1),
                    (1, 2, 2.0, 2, 2, 4.0, 2.0, -1, -1, np.nan, 0.0, 0.0, 0.0, 0, 0, 1),
                ],
                dtype=trade_dt,
            ),
        )

        cash_arr = np.empty((size.shape[0], 2), dtype=float_)
        position_arr = np.empty(size.shape, dtype=float_)
        val_price_arr = np.empty(size.shape, dtype=float_)
        value_arr = np.empty((size.shape[0], 2), dtype=float_)
        return_arr = np.empty((size.shape[0], 2), dtype=float_)
        pos_info_arr = np.empty(size.shape[1], dtype=trade_dt)

        def post_segment_func_nb(c):
            cash_arr[c.i, c.group] = c.last_cash[c.group]
            for col in range(c.from_col, c.to_col):
                position_arr[c.i, col] = c.last_position[col]
                val_price_arr[c.i, col] = c.last_val_price[col]
            value_arr[c.i, c.group] = c.last_value[c.group]
            return_arr[c.i, c.group] = c.last_return[c.group]

        def post_sim_func_nb(c):
            pos_info_arr[:] = c.last_pos_info

        pf = from_order_func(
            close,
            order_func_nb,
            post_segment_func_nb=post_segment_func_nb,
            post_sim_func_nb=post_sim_func_nb,
            jitted=False,
            update_value=True,
            ffill_val_price=True,
            group_by=[0, 0, 1],
            cash_sharing=True,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )

        np.testing.assert_array_equal(cash_arr, pf.cash.values)
        np.testing.assert_array_equal(position_arr, pf.assets.values)
        np.testing.assert_array_equal(val_price_arr, pf.filled_close.values)
        np.testing.assert_array_equal(value_arr, pf.value.values)
        np.testing.assert_array_equal(return_arr, pf.returns.values)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_post_sim_ctx(self, test_row_wise, test_flexible):
        if test_flexible:

            def order_func(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(
                        1.0,
                        nb.select_from_col_nb(c, col, c.close),
                        fees=0.01,
                        fixed_fees=1.0,
                        slippage=0.01,
                        log=True,
                    )
                return -1, nb.order_nothing_nb()

        else:

            def order_func(c):
                return nb.order_nb(
                    1.0,
                    nb.select_nb(c, c.close),
                    fees=0.01,
                    fixed_fees=1.0,
                    slippage=0.01,
                    log=True,
                )

        def post_sim_func(c, lst):
            lst.append(deepcopy(c))

        lst = []

        from_order_func(
            price_wide,
            order_func,
            post_sim_func_nb=post_sim_func,
            post_sim_args=(lst,),
            row_wise=test_row_wise,
            update_value=True,
            jitted=False,
            group_by=[0, 0, 1],
            cash_sharing=True,
            keep_inout_flex=False,
            max_log_records=price_wide.shape[0],
            flexible=test_flexible,
        )

        c = lst[-1]

        assert c.target_shape == price_wide.shape
        np.testing.assert_array_equal(c.close, price_wide.values)
        np.testing.assert_array_equal(c.group_lens, np.array([2, 1]))
        assert c.cash_sharing
        if test_flexible:
            assert c.call_seq is None
        else:
            np.testing.assert_array_equal(
                c.call_seq, np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
            )
        np.testing.assert_array_equal(c.init_cash, np.array([100.0, 100.0]))
        np.testing.assert_array_equal(c.init_position, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(
            c.segment_mask,
            np.array([[True, True], [True, True], [True, True], [True, True], [True, True]]),
        )
        assert c.ffill_val_price
        assert c.update_value
        assert_records_close(
            c.order_records.flatten(order="F"),
            np.array(
                [
                    (0, 0, 0, 1.0, 1.01, 1.0101, 0),
                    (1, 0, 1, 1.0, 2.02, 1.0202, 0),
                    (2, 0, 2, 1.0, 3.0300000000000002, 1.0303, 0),
                    (3, 0, 3, 1.0, 4.04, 1.0404, 0),
                    (4, 0, 4, 1.0, 5.05, 1.0505, 0),
                    (0, 1, 0, 1.0, 1.01, 1.0101, 0),
                    (1, 1, 1, 1.0, 2.02, 1.0202, 0),
                    (2, 1, 2, 1.0, 3.0300000000000002, 1.0303, 0),
                    (3, 1, 3, 1.0, 4.04, 1.0404, 0),
                    (4, 1, 4, 1.0, 5.05, 1.0505, 0),
                    (0, 2, 0, 1.0, 1.01, 1.0101, 0),
                    (1, 2, 1, 1.0, 2.02, 1.0202, 0),
                    (2, 2, 2, 1.0, 3.0300000000000002, 1.0303, 0),
                    (3, 2, 3, 1.0, 4.04, 1.0404, 0),
                    (4, 2, 4, 1.0, 5.05, 1.0505, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            c.log_records.flatten(order="F"),
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
                        np.nan,
                        100.0,
                        1.0,
                        1.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        1.01,
                        1.0101,
                        0,
                        0,
                        -1,
                        97.9799,
                        1.0,
                        0.0,
                        0.0,
                        97.9799,
                        1.01,
                        98.9899,
                        0,
                    ),
                    (
                        1,
                        0,
                        0,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        95.9598,
                        1.0,
                        0.0,
                        0.0,
                        95.9598,
                        1.0,
                        97.9598,
                        1.0,
                        2.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        2.02,
                        1.0202,
                        0,
                        0,
                        -1,
                        92.9196,
                        2.0,
                        0.0,
                        0.0,
                        92.9196,
                        2.02,
                        97.95960000000001,
                        1,
                    ),
                    (
                        2,
                        0,
                        0,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        89.8794,
                        2.0,
                        0.0,
                        0.0,
                        89.8794,
                        2.0,
                        97.8794,
                        1.0,
                        3.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        3.0300000000000002,
                        1.0303,
                        0,
                        0,
                        -1,
                        85.8191,
                        3.0,
                        0.0,
                        0.0,
                        85.8191,
                        3.0300000000000002,
                        98.90910000000001,
                        2,
                    ),
                    (
                        3,
                        0,
                        0,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        81.75880000000001,
                        3.0,
                        0.0,
                        0.0,
                        81.75880000000001,
                        3.0,
                        99.75880000000001,
                        1.0,
                        4.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        4.04,
                        1.0404,
                        0,
                        0,
                        -1,
                        76.67840000000001,
                        4.0,
                        0.0,
                        0.0,
                        76.67840000000001,
                        4.04,
                        101.83840000000001,
                        3,
                    ),
                    (
                        4,
                        0,
                        0,
                        4,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        71.59800000000001,
                        4.0,
                        0.0,
                        0.0,
                        71.59800000000001,
                        4.0,
                        103.59800000000001,
                        1.0,
                        5.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        5.05,
                        1.0505,
                        0,
                        0,
                        -1,
                        65.49750000000002,
                        5.0,
                        0.0,
                        0.0,
                        65.49750000000002,
                        5.05,
                        106.74750000000002,
                        4,
                    ),
                    (
                        0,
                        0,
                        1,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        97.9799,
                        0.0,
                        0.0,
                        0.0,
                        97.9799,
                        np.nan,
                        98.9899,
                        1.0,
                        1.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        1.01,
                        1.0101,
                        0,
                        0,
                        -1,
                        95.9598,
                        1.0,
                        0.0,
                        0.0,
                        95.9598,
                        1.01,
                        97.97980000000001,
                        0,
                    ),
                    (
                        1,
                        0,
                        1,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        92.9196,
                        1.0,
                        0.0,
                        0.0,
                        92.9196,
                        1.0,
                        97.95960000000001,
                        1.0,
                        2.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        2.02,
                        1.0202,
                        0,
                        0,
                        -1,
                        89.8794,
                        2.0,
                        0.0,
                        0.0,
                        89.8794,
                        2.02,
                        97.95940000000002,
                        1,
                    ),
                    (
                        2,
                        0,
                        1,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        85.8191,
                        2.0,
                        0.0,
                        0.0,
                        85.8191,
                        2.0,
                        98.90910000000001,
                        1.0,
                        3.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        3.0300000000000002,
                        1.0303,
                        0,
                        0,
                        -1,
                        81.75880000000001,
                        3.0,
                        0.0,
                        0.0,
                        81.75880000000001,
                        3.0300000000000002,
                        99.93880000000001,
                        2,
                    ),
                    (
                        3,
                        0,
                        1,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        76.67840000000001,
                        3.0,
                        0.0,
                        0.0,
                        76.67840000000001,
                        3.0,
                        101.83840000000001,
                        1.0,
                        4.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        4.04,
                        1.0404,
                        0,
                        0,
                        -1,
                        71.59800000000001,
                        4.0,
                        0.0,
                        0.0,
                        71.59800000000001,
                        4.04,
                        103.918,
                        3,
                    ),
                    (
                        4,
                        0,
                        1,
                        4,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        65.49750000000002,
                        4.0,
                        0.0,
                        0.0,
                        65.49750000000002,
                        4.0,
                        106.74750000000002,
                        1.0,
                        5.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        5.05,
                        1.0505,
                        0,
                        0,
                        -1,
                        59.39700000000002,
                        5.0,
                        0.0,
                        0.0,
                        59.39700000000002,
                        5.05,
                        109.89700000000002,
                        4,
                    ),
                    (
                        0,
                        1,
                        2,
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
                        np.nan,
                        100.0,
                        1.0,
                        1.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        1.01,
                        1.0101,
                        0,
                        0,
                        -1,
                        97.9799,
                        1.0,
                        0.0,
                        0.0,
                        97.9799,
                        1.01,
                        98.9899,
                        0,
                    ),
                    (
                        1,
                        1,
                        2,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        97.9799,
                        1.0,
                        0.0,
                        0.0,
                        97.9799,
                        1.0,
                        98.9799,
                        1.0,
                        2.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        2.02,
                        1.0202,
                        0,
                        0,
                        -1,
                        94.9397,
                        2.0,
                        0.0,
                        0.0,
                        94.9397,
                        2.02,
                        98.97970000000001,
                        1,
                    ),
                    (
                        2,
                        1,
                        2,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        94.9397,
                        2.0,
                        0.0,
                        0.0,
                        94.9397,
                        2.0,
                        98.9397,
                        1.0,
                        3.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        3.0300000000000002,
                        1.0303,
                        0,
                        0,
                        -1,
                        90.8794,
                        3.0,
                        0.0,
                        0.0,
                        90.8794,
                        3.0300000000000002,
                        99.96940000000001,
                        2,
                    ),
                    (
                        3,
                        1,
                        2,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        90.8794,
                        3.0,
                        0.0,
                        0.0,
                        90.8794,
                        3.0,
                        99.8794,
                        1.0,
                        4.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        4.04,
                        1.0404,
                        0,
                        0,
                        -1,
                        85.799,
                        4.0,
                        0.0,
                        0.0,
                        85.799,
                        4.04,
                        101.959,
                        3,
                    ),
                    (
                        4,
                        1,
                        2,
                        4,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        85.799,
                        4.0,
                        0.0,
                        0.0,
                        85.799,
                        4.0,
                        101.799,
                        1.0,
                        5.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
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
                        1.0,
                        5.05,
                        1.0505,
                        0,
                        0,
                        -1,
                        79.69850000000001,
                        5.0,
                        0.0,
                        0.0,
                        79.69850000000001,
                        5.05,
                        104.94850000000001,
                        4,
                    ),
                ],
                dtype=log_dt,
            ),
        )
        np.testing.assert_array_equal(c.last_cash, np.array([59.39700000000002, 79.69850000000001]))
        np.testing.assert_array_equal(c.last_position, np.array([5.0, 5.0, 5.0]))
        np.testing.assert_array_equal(c.last_val_price, np.array([5.0, 5.0, 5.0]))
        np.testing.assert_array_equal(
            c.last_value, np.array([109.39700000000002, 104.69850000000001])
        )
        np.testing.assert_array_equal(
            c.last_return, np.array([0.05597598409235705, 0.028482598060884715])
        )
        np.testing.assert_array_equal(c.last_debt, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(
            c.last_free_cash, np.array([59.39700000000002, 79.69850000000001])
        )
        np.testing.assert_array_equal(c.order_counts, np.array([5, 5, 5]))
        np.testing.assert_array_equal(c.log_counts, np.array([5, 5, 5]))

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_free_cash(self, test_row_wise, test_flexible):
        if test_flexible:

            def order_func(c, size):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(
                        size[c.i, col],
                        nb.select_from_col_nb(c, col, c.close),
                        fees=0.01,
                        fixed_fees=1.0,
                        slippage=0.01,
                    )
                return -1, nb.order_nothing_nb()

        else:

            def order_func(c, size):
                return nb.order_nb(
                    size[c.i, c.col],
                    nb.select_nb(c, c.close),
                    fees=0.01,
                    fixed_fees=1.0,
                    slippage=0.01,
                )

        def post_order_func(c, debt, free_cash):
            debt[c.i, c.col] = c.debt_now
            if c.cash_sharing:
                free_cash[c.i, c.group] = c.free_cash_now
            else:
                free_cash[c.i, c.col] = c.free_cash_now

        size = np.array([[5, -5, 5], [5, -5, -10], [-5, 5, 10], [-5, 5, -10], [-5, 5, 10]])
        debt = np.empty(price_wide.shape, dtype=float_)
        free_cash = np.empty(price_wide.shape, dtype=float_)
        pf = from_order_func(
            price_wide,
            order_func,
            size,
            post_order_func_nb=post_order_func,
            post_order_args=(
                debt,
                free_cash,
            ),
            row_wise=test_row_wise,
            jitted=False,
            flexible=test_flexible,
        )
        np.testing.assert_array_equal(
            debt,
            np.array(
                [
                    [0.0, 4.95, 0.0],
                    [0.0, 14.850000000000001, 9.9],
                    [0.0, 7.425000000000001, 0.0],
                    [0.0, 0.0, 19.8],
                    [24.75, 0.0, 0.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            free_cash,
            np.array(
                [
                    [93.8995, 94.0005, 93.8995],
                    [82.6985, 83.00150000000001, 92.70150000000001],
                    [96.39999999999999, 81.55000000000001, 80.89850000000001],
                    [115.002, 74.99800000000002, 79.50250000000001],
                    [89.0045, 48.49550000000002, 67.09750000000001],
                ]
            ),
        )
        np.testing.assert_almost_equal(free_cash, pf.get_cash(free=True).values)

        debt = np.empty(price_wide.shape, dtype=float_)
        free_cash = np.empty(price_wide.shape, dtype=float_)
        pf = from_order_func(
            price_wide.vbt.wrapper.wrap(price_wide.values[::-1]),
            order_func,
            size,
            post_order_func_nb=post_order_func,
            post_order_args=(
                debt,
                free_cash,
            ),
            row_wise=test_row_wise,
            jitted=False,
            flexible=test_flexible,
        )
        np.testing.assert_array_equal(
            debt,
            np.array(
                [
                    [0.0, 24.75, 0.0],
                    [0.0, 44.55, 19.8],
                    [0.0, 22.275, 0.0],
                    [0.0, 0.0, 9.9],
                    [4.95, 0.0, 0.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            free_cash,
            np.array(
                [
                    [73.4975, 74.0025, 73.4975],
                    [52.0955, 53.00449999999999, 72.1015],
                    [65.797, 81.25299999999999, 80.0985],
                    [74.598, 114.60199999999998, 78.90050000000001],
                    [68.5985, 108.50149999999998, 87.49950000000001],
                ]
            ),
        )
        np.testing.assert_almost_equal(free_cash, pf.get_cash(free=True).values)

        debt = np.empty(price_wide.shape, dtype=float_)
        free_cash = np.empty((price_wide.shape[0], 2), dtype=float_)
        pf = from_order_func(
            price_wide,
            order_func,
            size,
            post_order_func_nb=post_order_func,
            post_order_args=(
                debt,
                free_cash,
            ),
            row_wise=test_row_wise,
            jitted=False,
            group_by=[0, 0, 1],
            cash_sharing=True,
            flexible=test_flexible,
        )
        np.testing.assert_array_equal(
            debt,
            np.array(
                [
                    [0.0, 4.95, 0.0],
                    [0.0, 14.850000000000001, 9.9],
                    [0.0, 7.425000000000001, 0.0],
                    [0.0, 0.0, 19.8],
                    [24.75, 0.0, 0.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            free_cash,
            np.array(
                [
                    [87.9, 93.8995],
                    [65.70000000000002, 92.70150000000001],
                    [77.95000000000002, 80.89850000000001],
                    [90.00000000000003, 79.50250000000001],
                    [37.50000000000003, 67.09750000000001],
                ]
            ),
        )
        np.testing.assert_almost_equal(free_cash, pf.get_cash(free=True).values)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_init_cash(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = from_order_func(
            price_wide,
            order_func,
            np.array([[10.0]]),
            row_wise=test_row_wise,
            init_cash=[1.0, 10.0, np.inf],
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 2.0, 2.0, 0.0, 1),
                    (2, 0, 2, 1.333333333333333, 3.0, 0.0, 0),
                    (3, 0, 3, 0.666666666666667, 4.0, 0.0, 1),
                    (4, 0, 4, 0.533333333333333, 5.0, 0.0, 0),
                    (0, 1, 0, 10.0, 1.0, 0.0, 0),
                    (1, 1, 1, 10.0, 2.0, 0.0, 1),
                    (2, 1, 2, 6.666666666666667, 3.0, 0.0, 0),
                    (3, 1, 3, 10.0, 4.0, 0.0, 1),
                    (4, 1, 4, 8.0, 5.0, 0.0, 0),
                    (0, 2, 0, 10.0, 1.0, 0.0, 0),
                    (1, 2, 1, 10.0, 2.0, 0.0, 1),
                    (2, 2, 2, 10.0, 3.0, 0.0, 0),
                    (3, 2, 3, 10.0, 4.0, 0.0, 1),
                    (4, 2, 4, 10.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert type(pf._init_cash) == np.ndarray
        base_pf = from_order_func(
            price_wide,
            order_func,
            np.array([[10.0]]),
            row_wise=test_row_wise,
            init_cash=np.inf,
            flexible=test_flexible,
        )
        pf = from_order_func(
            price_wide,
            order_func,
            np.array([[10.0]]),
            row_wise=test_row_wise,
            init_cash=InitCashMode.Auto,
            flexible=test_flexible,
        )
        assert_records_close(pf.order_records, base_pf.orders.values)
        assert pf._init_cash == InitCashMode.Auto
        pf = from_order_func(
            price_wide,
            order_func,
            np.array([[10.0]]),
            row_wise=test_row_wise,
            init_cash=InitCashMode.AutoAlign,
            flexible=test_flexible,
        )
        assert_records_close(pf.order_records, base_pf.orders.values)
        assert pf._init_cash == InitCashMode.AutoAlign

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_init_position(self, test_row_wise, test_flexible):
        pos_info_arr = np.empty(1, dtype=trade_dt)

        def pre_segment_func_nb(c):
            pos_info_arr[:] = c.last_pos_info[:]
            return ()

        if test_flexible:

            def order_func_nb(c):
                if c.call_idx < c.group_len:
                    return c.from_col + c.call_idx, nb.order_nb(
                        -np.inf, direction=Direction.LongOnly
                    )
                return -1, nb.order_nothing_nb()

        else:

            def order_func_nb(c):
                return nb.order_nb(-np.inf, direction=Direction.LongOnly)

        pf = from_order_func(
            1,
            order_func_nb,
            open=0.5,
            jitted=False,
            init_cash=0.0,
            init_position=1.0,
            init_price=0.5,
            pre_segment_func_nb=pre_segment_func_nb,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert pf.init_position == 1.0
        assert_records_close(
            pf.order_records, np.array([(0, 0, 0, 1.0, 1.0, 0.0, 1)], dtype=order_dt)
        )
        assert_records_close(
            pos_info_arr,
            np.array(
                [(0, 0, 1.0, -1, -1, 0.5, 0.0, -1, -1, np.nan, 0.0, 0.0, 0.0, 0, 0, 0)],
                dtype=trade_dt,
            ),
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_cash_earnings(self, test_row_wise, test_flexible):
        if test_flexible:

            @njit
            def order_func_nb(c):
                if c.call_idx < c.group_len:
                    return c.from_col + c.call_idx, nb.order_nb()
                return -1, nb.order_nothing_nb()

        else:

            @njit
            def order_func_nb(c):
                return nb.order_nb()

        pf = from_order_func(
            1,
            order_func_nb,
            cash_earnings=np.array([0, 1, 2, 3]),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_series_equal(pf.cash_earnings, pd.Series([0.0, 1.0, 2.0, 3.0]))
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 2, 1.0, 1.0, 0.0, 0),
                    (2, 0, 3, 2.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_func_calls(self):
        @njit
        def pre_sim_func_nb(c, call_i, pre_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_sim_func_nb(c, call_i, post_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_group_func_nb(c, call_i, pre_group_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_group_func_nb(c, call_i, post_group_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_segment_func_nb(c, call_i, pre_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_segment_func_nb(c, call_i, post_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def order_func_nb(c, call_i, order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            order_lst.append(call_i[0])
            return NoOrder

        @njit
        def post_order_func_nb(c, call_i, post_order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_order_lst.append(call_i[0])

        sub_arg = vbt.RepEval("np.prod([target_shape[0], target_shape[1]])")

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            row_wise=False,
            template_context=dict(np=np),
        )
        assert call_i[0] == 56
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [56]
        assert list(pre_group_lst) == [2, 34]
        assert list(post_group_lst) == [33, 55]
        assert list(pre_segment_lst) == [3, 9, 15, 21, 27, 35, 39, 43, 47, 51]
        assert list(post_segment_lst) == [8, 14, 20, 26, 32, 38, 42, 46, 50, 54]
        assert list(order_lst) == [4, 6, 10, 12, 16, 18, 22, 24, 28, 30, 36, 40, 44, 48, 52]
        assert list(post_order_lst) == [5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 45, 49, 53]

        segment_mask = np.array(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=True,
            call_post_segment=True,
            row_wise=False,
            template_context=dict(np=np),
        )
        assert call_i[0] == 38
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [38]
        assert list(pre_group_lst) == [2, 22]
        assert list(post_group_lst) == [21, 37]
        assert list(pre_segment_lst) == [3, 5, 7, 13, 19, 23, 25, 29, 31, 35]
        assert list(post_segment_lst) == [4, 6, 12, 18, 20, 24, 28, 30, 34, 36]
        assert list(order_lst) == [8, 10, 14, 16, 26, 32]
        assert list(post_order_lst) == [9, 11, 15, 17, 27, 33]

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=False,
            call_post_segment=False,
            row_wise=False,
            template_context=dict(np=np),
        )
        assert call_i[0] == 26
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [26]
        assert list(pre_group_lst) == [2, 16]
        assert list(post_group_lst) == [15, 25]
        assert list(pre_segment_lst) == [3, 9, 17, 21]
        assert list(post_segment_lst) == [8, 14, 20, 24]
        assert list(order_lst) == [4, 6, 10, 12, 18, 22]
        assert list(post_order_lst) == [5, 7, 11, 13, 19, 23]

    def test_func_calls_flexible(self):
        @njit
        def pre_sim_func_nb(c, call_i, pre_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_sim_func_nb(c, call_i, post_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_group_func_nb(c, call_i, pre_group_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_group_func_nb(c, call_i, post_group_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_segment_func_nb(c, call_i, pre_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_segment_func_nb(c, call_i, post_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def flex_order_func_nb(c, call_i, order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            order_lst.append(call_i[0])
            col = c.from_col + c.call_idx
            if c.call_idx < c.group_len:
                return col, NoOrder
            return -1, NoOrder

        @njit
        def post_order_func_nb(c, call_i, post_order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_order_lst.append(call_i[0])

        sub_arg = vbt.RepEval("np.prod([target_shape[0], target_shape[1]])")

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            row_wise=False,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 66
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [66]
        assert list(pre_group_lst) == [2, 39]
        assert list(post_group_lst) == [38, 65]
        assert list(pre_segment_lst) == [3, 10, 17, 24, 31, 40, 45, 50, 55, 60]
        assert list(post_segment_lst) == [9, 16, 23, 30, 37, 44, 49, 54, 59, 64]
        assert list(order_lst) == [
            4,
            6,
            8,
            11,
            13,
            15,
            18,
            20,
            22,
            25,
            27,
            29,
            32,
            34,
            36,
            41,
            43,
            46,
            48,
            51,
            53,
            56,
            58,
            61,
            63,
        ]
        assert list(post_order_lst) == [5, 7, 12, 14, 19, 21, 26, 28, 33, 35, 42, 47, 52, 57, 62]

        segment_mask = np.array(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=True,
            call_post_segment=True,
            row_wise=False,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 42
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [42]
        assert list(pre_group_lst) == [2, 24]
        assert list(post_group_lst) == [23, 41]
        assert list(pre_segment_lst) == [3, 5, 7, 14, 21, 25, 27, 32, 34, 39]
        assert list(post_segment_lst) == [4, 6, 13, 20, 22, 26, 31, 33, 38, 40]
        assert list(order_lst) == [8, 10, 12, 15, 17, 19, 28, 30, 35, 37]
        assert list(post_order_lst) == [9, 11, 16, 18, 29, 36]

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=False,
            call_post_segment=False,
            row_wise=False,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 30
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [30]
        assert list(pre_group_lst) == [2, 18]
        assert list(post_group_lst) == [17, 29]
        assert list(pre_segment_lst) == [3, 10, 19, 24]
        assert list(post_segment_lst) == [9, 16, 23, 28]
        assert list(order_lst) == [4, 6, 8, 11, 13, 15, 20, 22, 25, 27]
        assert list(post_order_lst) == [5, 7, 12, 14, 21, 26]

    def test_func_calls_row_wise(self):
        @njit
        def pre_sim_func_nb(c, call_i, pre_sim_lst):
            call_i[0] += 1
            pre_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_sim_func_nb(c, call_i, post_sim_lst):
            call_i[0] += 1
            post_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_row_func_nb(c, call_i, pre_row_lst):
            call_i[0] += 1
            pre_row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_row_func_nb(c, call_i, post_row_lst):
            call_i[0] += 1
            post_row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_segment_func_nb(c, call_i, pre_segment_lst):
            call_i[0] += 1
            pre_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_segment_func_nb(c, call_i, post_segment_lst):
            call_i[0] += 1
            post_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def order_func_nb(c, call_i, order_lst):
            call_i[0] += 1
            order_lst.append(call_i[0])
            return NoOrder

        @njit
        def post_order_func_nb(c, call_i, post_order_lst):
            call_i[0] += 1
            post_order_lst.append(call_i[0])

        sub_arg = vbt.RepEval("np.prod([target_shape[0], target_shape[1]])")

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst,),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst,),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst,),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst,),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst,),
            row_wise=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 62
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [62]
        assert list(pre_row_lst) == [2, 14, 26, 38, 50]
        assert list(post_row_lst) == [13, 25, 37, 49, 61]
        assert list(pre_segment_lst) == [3, 9, 15, 21, 27, 33, 39, 45, 51, 57]
        assert list(post_segment_lst) == [8, 12, 20, 24, 32, 36, 44, 48, 56, 60]
        assert list(order_lst) == [4, 6, 10, 16, 18, 22, 28, 30, 34, 40, 42, 46, 52, 54, 58]
        assert list(post_order_lst) == [5, 7, 11, 17, 19, 23, 29, 31, 35, 41, 43, 47, 53, 55, 59]

        segment_mask = np.array(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst,),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst,),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst,),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst,),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst,),
            segment_mask=segment_mask,
            call_pre_segment=True,
            call_post_segment=True,
            row_wise=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 44
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [44]
        assert list(pre_row_lst) == [2, 8, 16, 26, 38]
        assert list(post_row_lst) == [7, 15, 25, 37, 43]
        assert list(pre_segment_lst) == [3, 5, 9, 11, 17, 23, 27, 33, 39, 41]
        assert list(post_segment_lst) == [4, 6, 10, 14, 22, 24, 32, 36, 40, 42]
        assert list(order_lst) == [12, 18, 20, 28, 30, 34]
        assert list(post_order_lst) == [13, 19, 21, 29, 31, 35]

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst,),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst,),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst,),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst,),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst,),
            segment_mask=segment_mask,
            call_pre_segment=False,
            call_post_segment=False,
            row_wise=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 32
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [32]
        assert list(pre_row_lst) == [2, 4, 10, 18, 30]
        assert list(post_row_lst) == [3, 9, 17, 29, 31]
        assert list(pre_segment_lst) == [5, 11, 19, 25]
        assert list(post_segment_lst) == [8, 16, 24, 28]
        assert list(order_lst) == [6, 12, 14, 20, 22, 26]
        assert list(post_order_lst) == [7, 13, 15, 21, 23, 27]

    def test_func_calls_row_wise_flexible(self):
        @njit
        def pre_sim_func_nb(c, call_i, pre_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_sim_func_nb(c, call_i, post_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_row_func_nb(c, call_i, pre_row_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_row_func_nb(c, call_i, post_row_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_segment_func_nb(c, call_i, pre_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_segment_func_nb(c, call_i, post_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def flex_order_func_nb(c, call_i, order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            order_lst.append(call_i[0])
            col = c.from_col + c.call_idx
            if c.call_idx < c.group_len:
                return col, NoOrder
            return -1, NoOrder

        @njit
        def post_order_func_nb(c, call_i, post_order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_order_lst.append(call_i[0])

        sub_arg = vbt.RepEval("np.prod([target_shape[0], target_shape[1]])")

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst, sub_arg),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            row_wise=True,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 72
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [72]
        assert list(pre_row_lst) == [2, 16, 30, 44, 58]
        assert list(post_row_lst) == [15, 29, 43, 57, 71]
        assert list(pre_segment_lst) == [3, 10, 17, 24, 31, 38, 45, 52, 59, 66]
        assert list(post_segment_lst) == [9, 14, 23, 28, 37, 42, 51, 56, 65, 70]
        assert list(order_lst) == [
            4,
            6,
            8,
            11,
            13,
            18,
            20,
            22,
            25,
            27,
            32,
            34,
            36,
            39,
            41,
            46,
            48,
            50,
            53,
            55,
            60,
            62,
            64,
            67,
            69,
        ]
        assert list(post_order_lst) == [5, 7, 12, 19, 21, 26, 33, 35, 40, 47, 49, 54, 61, 63, 68]

        segment_mask = np.array(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst, sub_arg),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=True,
            call_post_segment=True,
            row_wise=True,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 48
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [48]
        assert list(pre_row_lst) == [2, 8, 17, 28, 42]
        assert list(post_row_lst) == [7, 16, 27, 41, 47]
        assert list(pre_segment_lst) == [3, 5, 9, 11, 18, 25, 29, 36, 43, 45]
        assert list(post_segment_lst) == [4, 6, 10, 15, 24, 26, 35, 40, 44, 46]
        assert list(order_lst) == [12, 14, 19, 21, 23, 30, 32, 34, 37, 39]
        assert list(post_order_lst) == [13, 20, 22, 31, 33, 38]

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst, sub_arg),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=False,
            call_post_segment=False,
            row_wise=True,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 36
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [36]
        assert list(pre_row_lst) == [2, 4, 11, 20, 34]
        assert list(post_row_lst) == [3, 10, 19, 33, 35]
        assert list(pre_segment_lst) == [5, 12, 21, 28]
        assert list(post_segment_lst) == [9, 18, 27, 32]
        assert list(order_lst) == [6, 8, 13, 15, 17, 22, 24, 26, 29, 31]
        assert list(post_order_lst) == [7, 14, 16, 23, 25, 30]

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_max_order_records(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        assert (
            from_order_func(
                price_wide,
                order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                flexible=test_flexible,
            ).order_records.shape[0]
            == 15
        )
        assert (
            from_order_func(
                price_wide,
                order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                max_order_records=5,
                flexible=test_flexible,
            ).order_records.shape[0]
            == 15
        )
        assert (
            from_order_func(
                price_wide,
                order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                max_order_records=0,
                flexible=test_flexible,
            ).order_records.shape[0]
            == 0
        )
        with pytest.raises(Exception):
            from_order_func(
                price_wide,
                order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                max_order_records=4,
                flexible=test_flexible,
            )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_max_log_records(self, test_row_wise, test_flexible):
        log_order_func = log_flex_order_func_nb if test_flexible else log_order_func_nb
        assert (
            from_order_func(
                price_wide,
                log_order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                flexible=test_flexible,
            ).log_records.shape[0]
            == 15
        )
        assert (
            from_order_func(
                price_wide,
                log_order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                max_log_records=5,
                flexible=test_flexible,
            ).log_records.shape[0]
            == 15
        )
        assert (
            from_order_func(
                price_wide,
                log_order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                max_log_records=0,
                flexible=test_flexible,
            ).log_records.shape[0]
            == 0
        )
        with pytest.raises(Exception):
            from_order_func(
                price_wide,
                log_order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                max_log_records=4,
                flexible=test_flexible,
            )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_jitted_parallel(self, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        pf = from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            group_by=group_by,
            row_wise=False,
            flexible=test_flexible,
            jitted=dict(parallel=True),
        )
        pf2 = from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            group_by=group_by,
            row_wise=False,
            flexible=test_flexible,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            group_by=group_by,
            cash_sharing=True,
            row_wise=False,
            flexible=test_flexible,
            jitted=dict(parallel=True),
        )
        pf2 = from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            group_by=group_by,
            cash_sharing=True,
            row_wise=False,
            flexible=test_flexible,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_chunked(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        chunked = dict(
            arg_take_spec=dict(
                order_args=vbt.ArgsTaker(vbt.flex_array_gl_slicer),
                flex_order_args=vbt.ArgsTaker(vbt.flex_array_gl_slicer),
            )
        )
        pf = from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            group_by=group_by,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=chunked,
        )
        pf2 = from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            group_by=group_by,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=False,
        )
        if test_row_wise:
            assert_series_equal(pf.total_profit, pf2.total_profit)
            assert_series_equal(pf.total_profit, pf2.total_profit)
        else:
            assert_records_close(pf.order_records, pf2.order_records)
            assert_records_close(pf.log_records, pf2.log_records)
        pf = from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            group_by=group_by,
            cash_sharing=True,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=chunked,
        )
        pf2 = from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            group_by=group_by,
            cash_sharing=True,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=False,
        )
        if test_row_wise:
            assert_series_equal(pf.total_profit, pf2.total_profit)
            assert_series_equal(pf.total_profit, pf2.total_profit)
        else:
            assert_records_close(pf.order_records, pf2.order_records)
            assert_records_close(pf.log_records, pf2.log_records)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_in_outputs(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb

        @njit
        def post_sim_func_nb(c):
            c.in_outputs.custom_1d_arr[:] = 10.0
            c.in_outputs.custom_2d_arr[:] = 100
            c.in_outputs.custom_rec_arr["col"][:5] = 0
            c.in_outputs.custom_rec_arr["col"][5:10] = 1
            c.in_outputs.custom_rec_arr["col"][10:15] = 2

        class CustomMapper(vbt.ChunkMapper):
            def map(self, chunk_meta, ann_args=None, **kwargs):
                mapper = vbt.GroupLensMapper(arg_query="group_lens")
                chunk_meta = mapper.apply(chunk_meta, ann_args=ann_args, **kwargs)
                target_shape = ann_args["target_shape"]["value"]
                new_chunk_meta = vbt.ChunkMeta(
                    uuid=str(uuid.uuid4()),
                    idx=chunk_meta.idx,
                    start=chunk_meta.start * target_shape[0],
                    end=chunk_meta.end * target_shape[0],
                    indices=None,
                )
                return new_chunk_meta

        custom_dtype = np.dtype([("col", int_)])
        chunked = dict(
            arg_take_spec=dict(
                order_args=vbt.ArgsTaker(vbt.flex_array_gl_slicer),
                flex_order_args=vbt.ArgsTaker(vbt.flex_array_gl_slicer),
                in_outputs=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0, mapper=vbt.GroupLensMapper(arg_query="group_lens")),
                    vbt.ArraySlicer(axis=1),
                    vbt.ArraySlicer(axis=0, mapper=CustomMapper()),
                ),
            )
        )
        in_outputs = dict(
            custom_1d_arr=vbt.RepEval("np.full(target_shape[1], 0., dtype=float_)"),
            custom_2d_arr=vbt.RepEval("np.empty((target_shape[0], len(group_lens)), dtype=int_)"),
            custom_rec_arr=vbt.RepEval(
                "np.empty(target_shape[0] * target_shape[1], dtype=custom_dtype)"
            ),
        )
        in_outputs = InOutputs(**in_outputs)
        pf = from_order_func(
            price_wide,
            order_func,
            vbt.Rep("size"),
            post_sim_func_nb=post_sim_func_nb,
            broadcast_named_args=dict(size=[[0, 1, np.inf]]),
            in_outputs=in_outputs,
            template_context=dict(custom_dtype=custom_dtype),
            group_by=group_by,
            cash_sharing=False,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=chunked,
        )

        custom_1d_arr = np.array([10.0, 10.0, 10.0])
        custom_2d_arr = np.array([[100, 100], [100, 100], [100, 100], [100, 100], [100, 100]])
        custom_rec_arr = np.array(
            [
                (0,),
                (0,),
                (0,),
                (0,),
                (0,),
                (1,),
                (1,),
                (1,),
                (1,),
                (1,),
                (2,),
                (2,),
                (2,),
                (2,),
                (2,),
            ],
            dtype=custom_dtype,
        )

        np.testing.assert_array_equal(pf.in_outputs.custom_1d_arr, custom_1d_arr)
        np.testing.assert_array_equal(pf.in_outputs.custom_2d_arr, custom_2d_arr)
        np.testing.assert_array_equal(pf.in_outputs.custom_rec_arr, custom_rec_arr)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_staticized(self, test_row_wise, test_flexible, tmp_path):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        assert_records_close(
            from_order_func(
                price.tolist(),
                order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                flexible=test_flexible,
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            from_order_func(
                price.tolist(),
                order_func,
                np.array([[np.inf]]),
                row_wise=test_row_wise,
                flexible=test_flexible,
            ).order_records,
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    @pytest.mark.parametrize("test_jitted", [False, True])
    def test_sim_range(self, test_row_wise, test_flexible, test_jitted):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        order_size = pd.Series([np.inf, -np.inf, np.nan, np.inf, -np.inf], index=price.index)
        order_size_wide = order_size.vbt.tile(3, keys=["a", "b", "c"])

        def _from_order_func(size, **kwargs):
            return from_order_func(
                price_wide,
                order_func,
                vbt.Rep("size"),
                broadcast_named_args=dict(size=size),
                row_wise=test_row_wise,
                flexible=test_flexible,
                jitted=test_jitted,
                **kwargs,
            )

        _order_size_wide = order_size_wide.copy()
        _order_size_wide.iloc[:1] = np.nan
        assert_records_close(
            _from_order_func(order_size_wide, sim_start=1).order_records,
            _from_order_func(_order_size_wide).order_records,
        )
        np.testing.assert_array_equal(
            _from_order_func(order_size_wide, sim_start=1).sim_start,
            np.array([1, 1, 1]),
        )
        _order_size_wide = order_size_wide.copy()
        _order_size_wide.iloc[2:] = np.nan
        assert_records_close(
            _from_order_func(order_size_wide, sim_end=2).order_records,
            _from_order_func(_order_size_wide).order_records,
        )
        np.testing.assert_array_equal(
            _from_order_func(order_size_wide, sim_end=2).sim_end,
            np.array([2, 2, 2]),
        )
        _order_size_wide = order_size_wide.copy()
        _order_size_wide.iloc[:1, 0] = np.nan
        _order_size_wide.iloc[:2, 1] = np.nan
        _order_size_wide.iloc[:3, 2] = np.nan
        assert_records_close(
            _from_order_func(order_size_wide, sim_start=[1, 2, 3]).order_records,
            _from_order_func(_order_size_wide).order_records,
        )
        np.testing.assert_array_equal(
            _from_order_func(order_size_wide, sim_start=[1, 2, 3]).sim_start,
            np.array([1, 2, 3]),
        )
        with pytest.raises(Exception):
            _from_order_func(order_size_wide, sim_start=[1, 2])
        _order_size_wide = order_size_wide.copy()
        _order_size_wide.iloc[:1, 0] = np.nan
        _order_size_wide.iloc[:2, 1] = np.nan
        _order_size_wide.iloc[:2, 2] = np.nan
        assert_records_close(
            _from_order_func(
                order_size_wide, sim_start=[1, 2], group_by=[0, 0, 1], cash_sharing=True
            ).order_records,
            _from_order_func(_order_size_wide, group_by=[0, 0, 1], cash_sharing=True).order_records,
        )
        np.testing.assert_array_equal(
            _from_order_func(
                order_size_wide, sim_start=[1, 2], group_by=[0, 0, 1], cash_sharing=True
            ).sim_start,
            np.array([1, 2]),
        )


# ############# from_def_order_func ############# #


class TestFromDefOrderFunc:
    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_compare(self, test_row_wise, test_flexible):
        target_hold_value = pd.DataFrame(
            {
                "a": [0.0, 70.0, 30.0, 0.0, 70.0],
                "b": [30.0, 0.0, 70.0, 30.0, 30.0],
                "c": [70.0, 30.0, 0.0, 70.0, 0.0],
            },
            index=price.index,
        )
        pf = vbt.Portfolio.from_def_order_func(
            close=1.0,
            size=target_hold_value,
            size_type="targetvalue",
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq="auto",
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        pf2 = vbt.Portfolio.from_orders(
            close=1.0,
            size=target_hold_value,
            size_type="targetvalue",
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq="auto",
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        assert pf.wrapper == pf2.wrapper

        pf = vbt.Portfolio.from_def_order_func(
            close=1.0,
            size=target_hold_value / 100,
            size_type="targetpercent",
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq="auto",
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        pf2 = vbt.Portfolio.from_orders(
            close=1.0,
            size=target_hold_value / 100,
            size_type="targetpercent",
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq="auto",
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        assert pf.wrapper == pf2.wrapper

        if not test_row_wise:
            assert_records_close(
                vbt.Portfolio.from_def_order_func(
                    price_wide,
                    size=[[0, 1, np.inf]],
                    log=True,
                    row_wise=test_row_wise,
                    flexible=test_flexible,
                    jitted=dict(parallel=True),
                ).log_records,
                vbt.Portfolio.from_def_order_func(
                    price_wide,
                    size=[[0, 1, np.inf]],
                    log=True,
                    row_wise=test_row_wise,
                    flexible=test_flexible,
                    jitted=dict(parallel=False),
                ).log_records,
            )
        assert_records_close(
            vbt.Portfolio.from_def_order_func(
                price_wide,
                size=[[0, 1, np.inf]],
                log=True,
                row_wise=test_row_wise,
                flexible=test_flexible,
                chunked=True,
            ).log_records,
            vbt.Portfolio.from_def_order_func(
                price_wide,
                size=[[0, 1, np.inf]],
                log=True,
                row_wise=test_row_wise,
                flexible=test_flexible,
                chunked=False,
            ).log_records,
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_staticized(self, test_row_wise, test_flexible, tmp_path):
        target_hold_value = pd.DataFrame(
            {
                "a": [0.0, 70.0, 30.0, 0.0, 70.0],
                "b": [30.0, 0.0, 70.0, 30.0, 30.0],
                "c": [70.0, 30.0, 0.0, 70.0, 0.0],
            },
            index=price.index,
        )

        assert_records_close(
            vbt.Portfolio.from_def_order_func(
                close=1.0,
                size=target_hold_value,
                size_type="targetvalue",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                call_seq="auto",
                row_wise=test_row_wise,
                flexible=test_flexible,
                staticized=dict(path=tmp_path, override=True),
            ).order_records,
            vbt.Portfolio.from_def_order_func(
                close=1.0,
                size=target_hold_value,
                size_type="targetvalue",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                call_seq="auto",
                row_wise=test_row_wise,
                flexible=test_flexible,
            ).order_records,
        )
