import os

import pytest
import vectorbtpro as vbt
from vectorbtpro.portfolio.enums import *

from tests.utils import *

seed = 42

day_dt = np.timedelta64(86400000000000)

price = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    index=pd.date_range("2020", periods=5),
)
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


# ############# from_orders ############# #

order_size = pd.Series([np.inf, -np.inf, np.nan, np.inf, -np.inf], index=price.index)
order_size_wide = order_size.vbt.tile(3, keys=["a", "b", "c"])
order_size_one = pd.Series([1, -1, np.nan, 1, -1], index=price.index)


def from_orders_both(close=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(close, size, direction="both", **kwargs)


def from_orders_longonly(close=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(close, size, direction="longonly", **kwargs)


def from_orders_shortonly(close=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(close, size, direction="shortonly", **kwargs)


class TestFromOrders:
    def test_data(self):
        data = vbt.RandomOHLCData.pull(
            [0, 1],
            start="2020-01-01",
            end="2020-02-01",
            tick_freq="1h",
            freq="1d",
            seed=42,
        )
        pf = vbt.Portfolio.from_orders(data)
        assert pf.open is not None
        assert pf.high is not None
        assert pf.low is not None
        assert pf.close is not None
        pf = vbt.Portfolio.from_orders(data.get("Close"))
        assert pf.open is None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_orders(data[["Open", "Close"]])
        assert pf.open is not None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_orders(data["Close"])
        assert pf.open is None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_orders(data["Close"], open=data.get("Open"))
        assert pf.open is not None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None

    def test_one_column(self):
        assert_records_close(
            from_orders_both().order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly().order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly().order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 0, 1, 100.0, 2.0, 0.0, 0)], dtype=order_dt
            ),
        )
        pf = from_orders_both()
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

    def test_multiple_columns(self):
        assert_records_close(
            from_orders_both(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 2.0, 0.0, 1),
                    (2, 1, 3, 50.0, 4.0, 0.0, 0),
                    (3, 1, 4, 50.0, 5.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 100.0, 2.0, 0.0, 1),
                    (2, 2, 3, 50.0, 4.0, 0.0, 0),
                    (3, 2, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (1, 0, 1, 100.0, 2.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (1, 1, 1, 100.0, 2.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 1, 100.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_both(close=price_wide)
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

    def test_size_inf(self):
        assert_records_close(
            from_orders_both(size=[[np.inf, -np.inf]]).order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (0, 1, 0, 100.0, 1.0, 0.0, 1)], dtype=order_dt
            ),
        )
        assert_records_close(
            from_orders_longonly(size=[[np.inf, -np.inf]]).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_shortonly(size=[[np.inf, -np.inf]]).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 1)], dtype=order_dt),
        )

    def test_size_granularity(self):
        assert_records_close(
            from_orders_both(fees=0.1, fixed_fees=0.1, size_granularity=1).order_records,
            np.array(
                [
                    (0, 0, 0, 90.0, 1.0, 9.1, 0),
                    (1, 0, 1, 164.0, 2.0, 32.9, 1),
                    (2, 0, 3, 67.0, 4.0, 26.9, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(fees=0.1, fixed_fees=0.1, size_granularity=1).order_records,
            np.array(
                [
                    (0, 0, 0, 90.0, 1.0, 9.1, 0),
                    (1, 0, 1, 90.0, 2.0, 18.1, 1),
                    (2, 0, 3, 36.0, 4.0, 14.5, 0),
                    (3, 0, 4, 36.0, 5.0, 18.1, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(fees=0.1, fixed_fees=0.1, size_granularity=1).order_records,
            np.array([(0, 0, 0, 90.0, 1.0, 9.1, 1), (1, 0, 1, 82.0, 2.0, 16.5, 0)], dtype=order_dt),
        )

    def test_price(self):
        assert_records_close(
            from_orders_both(price=price * 1.01).order_records,
            np.array(
                [
                    (0, 0, 0, 99.00990099009901, 1.01, 0.0, 0),
                    (1, 0, 1, 198.01980198019803, 2.02, 0.0, 1),
                    (2, 0, 3, 99.00990099009901, 4.04, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(price=price * 1.01).order_records,
            np.array(
                [
                    (0, 0, 0, 99.00990099009901, 1.01, 0.0, 0),
                    (1, 0, 1, 99.00990099009901, 2.02, 0.0, 1),
                    (2, 0, 3, 49.504950495049506, 4.04, 0.0, 0),
                    (3, 0, 4, 49.504950495049506, 5.05, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(price=price * 1.01).order_records,
            np.array(
                [
                    (0, 0, 0, 99.00990099009901, 1.01, 0.0, 1),
                    (1, 0, 1, 99.00990099009901, 2.02, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(price=np.inf).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(price=np.inf).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(price=np.inf).order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 0, 1, 100.0, 2.0, 0.0, 0)], dtype=order_dt
            ),
        )
        assert_records_close(
            from_orders_both(price=-np.inf, open=price.shift(1)).order_records,
            np.array(
                [(0, 0, 1, 100.0, 1.0, 0.0, 1), (1, 0, 3, 66.66666666666667, 3.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(price=-np.inf, open=price.shift(1)).order_records,
            np.array(
                [
                    (0, 0, 3, 33.333333333333336, 3.0, 0.0, 0),
                    (1, 0, 4, 33.333333333333336, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(price=-np.inf, open=price.shift(1)).order_records,
            np.array(
                [
                    (0, 0, 3, 33.333333333333336, 3.0, 0.0, 1),
                    (1, 0, 4, 33.333333333333336, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_price_area(self):
        assert_records_close(
            from_orders_both(
                open=2,
                high=4,
                low=1,
                close=3,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
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
            from_orders_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
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
            from_orders_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 0.45, 0.0, 1),
                    (0, 1, 0, 1.0, 2.7, 0.0, 1),
                    (0, 2, 0, 1.0, 4.5, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
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
        assert_records_close(
            from_orders_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
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
        assert_records_close(
            from_orders_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (0, 1, 0, 1.0, 3.0, 0.0, 1),
                    (0, 2, 0, 1.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_orders_longonly(
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
            from_orders_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=np.inf,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_orders_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=5,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_orders_shortonly(
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
            from_orders_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=np.inf,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_orders_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=5,
                size=1,
                slippage=0.1,
            )

    def test_val_price(self):
        price_nan = pd.Series([1, 2, np.nan, 4, 5], index=price.index)
        assert_records_close(
            from_orders_both(
                close=price_nan, size=order_size_one, val_price=np.inf, size_type="value"
            ).order_records,
            from_orders_both(
                close=price_nan, size=order_size_one, val_price=price, size_type="value"
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=price,
                size_type="value",
            ).order_records,
        )
        shift_price = price_nan.ffill().shift(1)
        assert_records_close(
            from_orders_both(
                close=price_nan, size=order_size_one, val_price=-np.inf, size_type="value"
            ).order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        price_all_nan = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=price.index)
        assert_records_close(
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                close=price_nan, size=order_size_one, val_price=np.nan, size_type="value"
            ).order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                close=price_nan,
                open=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                open=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                open=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
            ).order_records,
        )

    def test_fees(self):
        assert_records_close(
            from_orders_both(size=order_size_one, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 1, 1.0, 2.0, -0.2, 1),
                    (2, 0, 3, 1.0, 4.0, -0.4, 0),
                    (3, 0, 4, 1.0, 5.0, -0.5, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 1, 1.0, 2.0, 0.2, 1),
                    (2, 2, 3, 1.0, 4.0, 0.4, 0),
                    (3, 2, 4, 1.0, 5.0, 0.5, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 1, 1.0, 2.0, 2.0, 1),
                    (2, 3, 3, 1.0, 4.0, 4.0, 0),
                    (3, 3, 4, 1.0, 5.0, 5.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 1, 1.0, 2.0, -0.2, 1),
                    (2, 0, 3, 1.0, 4.0, -0.4, 0),
                    (3, 0, 4, 1.0, 5.0, -0.5, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 1, 1.0, 2.0, 0.2, 1),
                    (2, 2, 3, 1.0, 4.0, 0.4, 0),
                    (3, 2, 4, 1.0, 5.0, 0.5, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 1, 1.0, 2.0, 2.0, 1),
                    (2, 3, 3, 1.0, 4.0, 4.0, 0),
                    (3, 3, 4, 1.0, 5.0, 5.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 1),
                    (1, 0, 1, 1.0, 2.0, -0.2, 0),
                    (2, 0, 3, 1.0, 4.0, -0.4, 1),
                    (3, 0, 4, 1.0, 5.0, -0.5, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 1.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.1, 1),
                    (1, 2, 1, 1.0, 2.0, 0.2, 0),
                    (2, 2, 3, 1.0, 4.0, 0.4, 1),
                    (3, 2, 4, 1.0, 5.0, 0.5, 0),
                    (0, 3, 0, 1.0, 1.0, 1.0, 1),
                    (1, 3, 1, 1.0, 2.0, 2.0, 0),
                    (2, 3, 3, 1.0, 4.0, 4.0, 1),
                    (3, 3, 4, 1.0, 5.0, 5.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_fixed_fees(self):
        assert_records_close(
            from_orders_both(size=order_size_one, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 1, 1.0, 2.0, -0.1, 1),
                    (2, 0, 3, 1.0, 4.0, -0.1, 0),
                    (3, 0, 4, 1.0, 5.0, -0.1, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 1, 1.0, 2.0, 0.1, 1),
                    (2, 2, 3, 1.0, 4.0, 0.1, 0),
                    (3, 2, 4, 1.0, 5.0, 0.1, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 1, 1.0, 2.0, 1.0, 1),
                    (2, 3, 3, 1.0, 4.0, 1.0, 0),
                    (3, 3, 4, 1.0, 5.0, 1.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(
                size=order_size_one, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 1, 1.0, 2.0, -0.1, 1),
                    (2, 0, 3, 1.0, 4.0, -0.1, 0),
                    (3, 0, 4, 1.0, 5.0, -0.1, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 1, 1.0, 2.0, 0.1, 1),
                    (2, 2, 3, 1.0, 4.0, 0.1, 0),
                    (3, 2, 4, 1.0, 5.0, 0.1, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 1, 1.0, 2.0, 1.0, 1),
                    (2, 3, 3, 1.0, 4.0, 1.0, 0),
                    (3, 3, 4, 1.0, 5.0, 1.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(
                size=order_size_one, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 1),
                    (1, 0, 1, 1.0, 2.0, -0.1, 0),
                    (2, 0, 3, 1.0, 4.0, -0.1, 1),
                    (3, 0, 4, 1.0, 5.0, -0.1, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 1.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.1, 1),
                    (1, 2, 1, 1.0, 2.0, 0.1, 0),
                    (2, 2, 3, 1.0, 4.0, 0.1, 1),
                    (3, 2, 4, 1.0, 5.0, 0.1, 0),
                    (0, 3, 0, 1.0, 1.0, 1.0, 1),
                    (1, 3, 1, 1.0, 2.0, 1.0, 0),
                    (2, 3, 3, 1.0, 4.0, 1.0, 1),
                    (3, 3, 4, 1.0, 5.0, 1.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_slippage(self):
        assert_records_close(
            from_orders_both(size=order_size_one, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.1, 0.0, 0),
                    (1, 1, 1, 1.0, 1.8, 0.0, 1),
                    (2, 1, 3, 1.0, 4.4, 0.0, 0),
                    (3, 1, 4, 1.0, 4.5, 0.0, 1),
                    (0, 2, 0, 1.0, 2.0, 0.0, 0),
                    (1, 2, 1, 1.0, 0.0, 0.0, 1),
                    (2, 2, 3, 1.0, 8.0, 0.0, 0),
                    (3, 2, 4, 1.0, 0.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.1, 0.0, 0),
                    (1, 1, 1, 1.0, 1.8, 0.0, 1),
                    (2, 1, 3, 1.0, 4.4, 0.0, 0),
                    (3, 1, 4, 1.0, 4.5, 0.0, 1),
                    (0, 2, 0, 1.0, 2.0, 0.0, 0),
                    (1, 2, 1, 1.0, 0.0, 0.0, 1),
                    (2, 2, 3, 1.0, 8.0, 0.0, 0),
                    (3, 2, 4, 1.0, 0.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 1, 1.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1.0, 4.0, 0.0, 1),
                    (3, 0, 4, 1.0, 5.0, 0.0, 0),
                    (0, 1, 0, 1.0, 0.9, 0.0, 1),
                    (1, 1, 1, 1.0, 2.2, 0.0, 0),
                    (2, 1, 3, 1.0, 3.6, 0.0, 1),
                    (3, 1, 4, 1.0, 5.5, 0.0, 0),
                    (0, 2, 0, 1.0, 0.0, 0.0, 1),
                    (1, 2, 1, 1.0, 4.0, 0.0, 0),
                    (2, 2, 3, 1.0, 0.0, 0.0, 1),
                    (3, 2, 4, 1.0, 10.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_min_size(self):
        assert_records_close(
            from_orders_both(size=order_size_one, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 1, 1.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1.0, 4.0, 0.0, 1),
                    (3, 0, 4, 1.0, 5.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 1.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_max_size(self):
        assert_records_close(
            from_orders_both(size=order_size_one, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 0),
                    (1, 0, 1, 0.5, 2.0, 0.0, 1),
                    (2, 0, 3, 0.5, 4.0, 0.0, 0),
                    (3, 0, 4, 0.5, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 1, 1.0, 2.0, 0.0, 1),
                    (2, 2, 3, 1.0, 4.0, 0.0, 0),
                    (3, 2, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 0),
                    (1, 0, 1, 0.5, 2.0, 0.0, 1),
                    (2, 0, 3, 0.5, 4.0, 0.0, 0),
                    (3, 0, 4, 0.5, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 1, 1.0, 2.0, 0.0, 1),
                    (2, 2, 3, 1.0, 4.0, 0.0, 0),
                    (3, 2, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 1),
                    (1, 0, 1, 0.5, 2.0, 0.0, 0),
                    (2, 0, 3, 0.5, 4.0, 0.0, 1),
                    (3, 0, 4, 0.5, 5.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 1.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.0, 1),
                    (1, 2, 1, 1.0, 2.0, 0.0, 0),
                    (2, 2, 3, 1.0, 4.0, 0.0, 1),
                    (3, 2, 4, 1.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_reject_prob(self):
        assert_records_close(
            from_orders_both(
                size=order_size_one, reject_prob=[[0.0, 0.5, 1.0]], seed=42
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 1, 1.0, 2.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                    (2, 1, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(
                size=order_size_one, reject_prob=[[0.0, 0.5, 1.0]], seed=42
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 3, 1.0, 4.0, 0.0, 0),
                    (1, 1, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(
                size=order_size_one, reject_prob=[[0.0, 0.5, 1.0]], seed=42
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 1, 1.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1.0, 4.0, 0.0, 1),
                    (3, 0, 4, 1.0, 5.0, 0.0, 0),
                    (0, 1, 3, 1.0, 4.0, 0.0, 1),
                    (1, 1, 4, 1.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_cash_locking(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([1, 1]),
            pd.DataFrame([[-25, -25], [np.inf, np.inf]]),
            group_by=True,
            cash_sharing=True,
            fees=0.01,
            fixed_fees=1.0,
            slippage=0.01,
        )
        np.testing.assert_array_equal(
            pf.asset_flow.values,
            np.array([[-25.0, -25.0], [94.6034702480149, 47.544358396235666]]),
        )
        pf = vbt.Portfolio.from_orders(
            pd.Series([1, 100]),
            pd.DataFrame([[-25, -25], [np.inf, np.inf]]),
            group_by=True,
            cash_sharing=True,
            fees=0.01,
            fixed_fees=1.0,
            slippage=0.01,
        )
        np.testing.assert_array_equal(
            pf.asset_flow.values,
            np.array([[-25.0, -25.0], [0.946034702480149, 0.008559442318505028]]),
        )
        pf = from_orders_both(size=order_size_one * 1000)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_longonly(size=order_size_one * 1000)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_shortonly(size=order_size_one * 1000)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (1, 0, 1, 100.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_leverage(self):
        pf = from_orders_longonly(
            close=[[1, 1], [2, 2]],
            size=[300, -300],
            group_by=True,
            cash_sharing=True,
            leverage=1,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_longonly(
            close=[[1, 1], [2, 2]],
            size=[300, -300],
            group_by=True,
            cash_sharing=True,
            leverage=2,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 200.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_longonly(
            close=[[1, 1], [2, 2]],
            size=[200, -200],
            group_by=True,
            cash_sharing=True,
            leverage=3,
            leverage_mode="eager",
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 200.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 2.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_longonly(
            close=[[1, 1], [2, 2]],
            size=[[50, 200], [-50, -200]],
            group_by=True,
            cash_sharing=True,
            leverage=2,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 50.0, 2.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 2.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_longonly(
            close=[[1, 1], [2, 1]],
            size=[[200, 0], [-200, np.inf]],
            group_by=True,
            cash_sharing=True,
            leverage=2,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 200.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (0, 1, 1, 600.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_longonly(
            close=[[1, 1], [0.75, 1]],
            size=[[200, 0], [-200, np.inf]],
            group_by=True,
            cash_sharing=True,
            leverage=2,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 200.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 0.75, 0.0, 1),
                    (0, 1, 1, 100.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_allow_partial(self):
        assert_records_close(
            from_orders_both(
                size=order_size_one * 1000, allow_partial=[[True, False]]
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(
                size=order_size_one * 1000, allow_partial=[[True, False]]
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(
                size=order_size_one * 1000, allow_partial=[[True, False]]
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (1, 0, 1, 100.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 2.0, 0.0, 1),
                    (2, 1, 3, 50.0, 4.0, 0.0, 0),
                    (3, 1, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (1, 0, 1, 100.0, 2.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (1, 1, 1, 100.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_raise_reject(self):
        with pytest.raises(Exception):
            from_orders_both(size=order_size_one * 1000, allow_partial=True, raise_reject=True)
        assert_records_close(
            from_orders_longonly(
                size=order_size_one * 1000, allow_partial=True, raise_reject=True
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_orders_shortonly(size=order_size_one * 1000, allow_partial=True, raise_reject=True)
        with pytest.raises(Exception):
            from_orders_both(size=order_size_one * 1000, allow_partial=False, raise_reject=True)
        with pytest.raises(Exception):
            from_orders_longonly(size=order_size_one * 1000, allow_partial=False, raise_reject=True)
        with pytest.raises(Exception):
            from_orders_shortonly(
                size=order_size_one * 1000, allow_partial=False, raise_reject=True
            )

    def test_log(self):
        assert_records_close(
            from_orders_both(log=True).log_records,
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
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        0.0,
                        100.0,
                        0.0,
                        0.0,
                        0.0,
                        2.0,
                        200.0,
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
                        2.0,
                        0.0,
                        1,
                        0,
                        -1,
                        400.0,
                        -100.0,
                        200.0,
                        200.0,
                        0.0,
                        2.0,
                        200.0,
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
                        400.0,
                        -100.0,
                        200.0,
                        200.0,
                        0.0,
                        3.0,
                        100.0,
                        np.nan,
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
                        np.nan,
                        np.nan,
                        np.nan,
                        -1,
                        1,
                        0,
                        400.0,
                        -100.0,
                        200.0,
                        200.0,
                        0.0,
                        3.0,
                        100.0,
                        -1,
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
                        400.0,
                        -100.0,
                        200.0,
                        200.0,
                        0.0,
                        4.0,
                        0.0,
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
                        4.0,
                        0.0,
                        0,
                        0,
                        -1,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        2,
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
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        5.0,
                        0.0,
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
                        np.nan,
                        np.nan,
                        np.nan,
                        -1,
                        2,
                        6,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        5.0,
                        0.0,
                        -1,
                    ),
                ],
                dtype=log_dt,
            ),
        )

    def test_group_by(self):
        pf = from_orders_both(close=price_wide, group_by=np.array([0, 0, 1]))
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
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
            pd.Series([200.0, 100.0], index=pd.Index([0, 1], dtype="int64"))
            .rename("init_cash")
            .rename_axis("group"),
        )
        assert not pf.cash_sharing

    def test_cash_sharing(self):
        pf = from_orders_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
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
        with pytest.raises(Exception):
            pf.regroup(group_by=False)

    def test_from_ago(self):
        assert_records_close(
            from_orders_both(from_ago=2, price=price * 0.9).order_records,
            from_orders_both(size=order_size.shift(2), price=(price * 0.9).shift(2)).order_records,
        )
        assert_records_close(
            from_orders_both(price="nextclose").order_records,
            from_orders_both(price=np.inf, from_ago=1).order_records,
        )
        assert_records_close(
            from_orders_both(open=price * 0.9, price="nextopen").order_records,
            from_orders_both(open=price * 0.9, price=-np.inf, from_ago=1).order_records,
        )

    def test_call_seq(self):
        pf = from_orders_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        )
        pf = from_orders_both(
            close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True, call_seq="reversed"
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        pf = from_orders_both(
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
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        kwargs = dict(
            close=1.0,
            size=pd.DataFrame(
                [
                    [0.0, 0.0, np.inf],
                    [0.0, np.inf, -np.inf],
                    [np.inf, -np.inf, 0.0],
                    [-np.inf, 0.0, np.inf],
                    [0.0, np.inf, -np.inf],
                ]
            ),
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq="auto",
        )
        pf = from_orders_both(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 1.0, 0.0, 1),
                    (2, 2, 3, 200.0, 1.0, 0.0, 0),
                    (3, 2, 4, 200.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [0, 1, 2], [2, 0, 1], [1, 0, 2], [0, 1, 2]]),
        )
        pf = from_orders_longonly(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 2, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 100.0, 1.0, 0.0, 1),
                    (0, 1, 1, 100.0, 1.0, 0.0, 0),
                    (1, 1, 2, 100.0, 1.0, 0.0, 1),
                    (2, 1, 4, 100.0, 1.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 100.0, 1.0, 0.0, 1),
                    (2, 2, 3, 100.0, 1.0, 0.0, 0),
                    (3, 2, 4, 100.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 0, 1]]),
        )
        pf = from_orders_shortonly(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 2, 100.0, 1.0, 0.0, 1),
                    (1, 0, 3, 100.0, 1.0, 0.0, 0),
                    (0, 1, 1, 100.0, 1.0, 0.0, 1),
                    (1, 1, 2, 100.0, 1.0, 0.0, 0),
                    (2, 1, 4, 100.0, 1.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 1, 100.0, 1.0, 0.0, 0),
                    (2, 2, 3, 100.0, 1.0, 0.0, 1),
                    (3, 2, 4, 100.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 0, 1]]),
        )
        _ = from_orders_both(attach_call_seq=False, **kwargs)
        _ = from_orders_longonly(attach_call_seq=False, **kwargs)
        _ = from_orders_shortonly(attach_call_seq=False, **kwargs)

    def test_value(self):
        assert_records_close(
            from_orders_both(size=order_size_one, size_type="value").order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 0.5, 2.0, 0.0, 1),
                    (2, 0, 3, 0.25, 4.0, 0.0, 0),
                    (3, 0, 4, 0.2, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, size_type="value").order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 0.5, 2.0, 0.0, 1),
                    (2, 0, 3, 0.25, 4.0, 0.0, 0),
                    (3, 0, 4, 0.2, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, size_type="value").order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 1, 0.5, 2.0, 0.0, 0),
                    (2, 0, 3, 0.25, 4.0, 0.0, 1),
                    (3, 0, 4, 0.2, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_target_amount(self):
        assert_records_close(
            from_orders_both(size=[[75.0, -75.0]], size_type="targetamount").order_records,
            np.array([(0, 0, 0, 75.0, 1.0, 0.0, 0), (0, 1, 0, 75.0, 1.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_longonly(size=[[75.0, -75.0]], size_type="targetamount").order_records,
            np.array([(0, 0, 0, 75.0, 1.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_shortonly(size=[[75.0, -75.0]], size_type="targetamount").order_records,
            np.array([(0, 0, 0, 75.0, 1.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=75.0,
                size_type="targetamount",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array([(0, 0, 0, 75.0, 1.0, 0.0, 0), (0, 1, 0, 25.0, 1.0, 0.0, 0)], dtype=order_dt),
        )

    def test_target_value(self):
        assert_records_close(
            from_orders_both(size=[[50.0, -50.0]], size_type="targetvalue").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 25.0, 2.0, 0.0, 1),
                    (2, 0, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (3, 0, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (4, 0, 4, 2.5, 5.0, 0.0, 1),
                    (0, 1, 0, 50.0, 1.0, 0.0, 1),
                    (1, 1, 1, 25.0, 2.0, 0.0, 0),
                    (2, 1, 2, 8.333333333333332, 3.0, 0.0, 0),
                    (3, 1, 3, 4.166666666666668, 4.0, 0.0, 0),
                    (4, 1, 4, 2.5, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=[[50.0, -50.0]], size_type="targetvalue").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 25.0, 2.0, 0.0, 1),
                    (2, 0, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (3, 0, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (4, 0, 4, 2.5, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=[[50.0, -50.0]], size_type="targetvalue").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 1),
                    (1, 0, 1, 25.0, 2.0, 0.0, 0),
                    (2, 0, 2, 8.333333333333332, 3.0, 0.0, 0),
                    (3, 0, 3, 4.166666666666668, 4.0, 0.0, 0),
                    (4, 0, 4, 2.5, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=50.0,
                size_type="targetvalue",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 25.0, 2.0, 0.0, 1),
                    (2, 0, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (3, 0, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (4, 0, 4, 2.5, 5.0, 0.0, 1),
                    (0, 1, 0, 50.0, 1.0, 0.0, 0),
                    (1, 1, 1, 25.0, 2.0, 0.0, 1),
                    (2, 1, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (3, 1, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (4, 1, 4, 2.5, 5.0, 0.0, 1),
                    (0, 2, 1, 25.0, 2.0, 0.0, 0),
                    (1, 2, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (2, 2, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (3, 2, 4, 2.5, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_target_percent(self):
        assert_records_close(
            from_orders_both(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 1),
                    (2, 0, 2, 6.25, 3.0, 0.0, 1),
                    (3, 0, 3, 3.90625, 4.0, 0.0, 1),
                    (4, 0, 4, 2.734375, 5.0, 0.0, 1),
                    (0, 1, 0, 50.0, 1.0, 0.0, 1),
                    (1, 1, 1, 37.5, 2.0, 0.0, 0),
                    (2, 1, 2, 6.25, 3.0, 0.0, 0),
                    (3, 1, 3, 2.34375, 4.0, 0.0, 0),
                    (4, 1, 4, 1.171875, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 1),
                    (2, 0, 2, 6.25, 3.0, 0.0, 1),
                    (3, 0, 3, 3.90625, 4.0, 0.0, 1),
                    (4, 0, 4, 2.734375, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 1),
                    (1, 0, 1, 37.5, 2.0, 0.0, 0),
                    (2, 0, 2, 6.25, 3.0, 0.0, 0),
                    (3, 0, 3, 2.34375, 4.0, 0.0, 0),
                    (4, 0, 4, 1.171875, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=0.5,
                size_type="targetpercent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array([(0, 0, 0, 50.0, 1.0, 0.0, 0), (0, 1, 0, 50.0, 1.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_both(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            from_orders_both(size=[[50, -50]], size_type="targetpercent100").order_records,
        )
        assert_records_close(
            from_orders_longonly(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            from_orders_longonly(size=[[50, -50]], size_type="targetpercent100").order_records,
        )
        assert_records_close(
            from_orders_shortonly(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            from_orders_shortonly(size=[[50, -50]], size_type="targetpercent100").order_records,
        )

    def test_update_value(self):
        assert_records_close(
            from_orders_both(
                size=0.5,
                size_type="targetpercent",
                fees=0.01,
                slippage=0.01,
                update_value=False,
            ).order_records,
            from_orders_both(
                size=0.5,
                size_type="targetpercent",
                fees=0.01,
                slippage=0.01,
                update_value=True,
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=0.5,
                size_type="targetpercent",
                fees=0.01,
                slippage=0.01,
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                update_value=False,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.01, 0.505, 0),
                    (1, 0, 1, 0.9851975296539592, 1.98, 0.019506911087148394, 1),
                    (2, 0, 2, 0.019315704924103727, 2.9699999999999998, 0.0005736764362458806, 1),
                    (3, 0, 3, 0.00037870218456959037, 3.96, 1.4996606508955778e-05, 1),
                    (4, 0, 4, 7.424805112066224e-06, 4.95, 3.675278530472781e-07, 1),
                    (0, 1, 0, 48.02960494069208, 1.01, 0.485099009900992, 0),
                    (1, 1, 1, 0.9465661198057499, 2.02, 0.019120635620076154, 0),
                    (2, 1, 2, 0.018558300554959377, 3.0300000000000002, 0.0005623165068152705, 0),
                    (3, 1, 3, 0.0003638525743521767, 4.04, 1.4699644003827875e-05, 0),
                    (4, 1, 4, 7.133664827307231e-06, 5.05, 3.6025007377901643e-07, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=0.5,
                size_type="targetpercent",
                fees=0.01,
                slippage=0.01,
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                update_value=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.01, 0.505, 0),
                    (1, 0, 1, 0.9851975296539592, 1.98, 0.019506911087148394, 1),
                    (2, 0, 2, 0.019315704924103727, 2.9699999999999998, 0.0005736764362458806, 1),
                    (3, 0, 3, 0.0005670876809631409, 3.96, 2.2456672166140378e-05, 1),
                    (4, 0, 4, 1.8523501267964093e-05, 4.95, 9.169133127642227e-07, 1),
                    (0, 1, 0, 48.02960494069208, 1.01, 0.485099009900992, 0),
                    (1, 1, 1, 0.7303208018821721, 2.02, 0.014752480198019875, 0),
                    (2, 1, 2, 0.009608602243410758, 2.9699999999999998, 0.00028537548662929945, 1),
                    (3, 1, 3, 0.00037770350099464167, 3.96, 1.4957058639387809e-05, 1),
                    (4, 1, 4, 1.2972670177191503e-05, 4.95, 6.421471737709794e-07, 1),
                    (0, 2, 1, 0.21624531792357785, 2.02, 0.0043681554220562635, 0),
                    (1, 2, 2, 0.02779013180558861, 3.0300000000000002, 0.0008420409937093393, 0),
                    (2, 2, 3, 0.0009077441794302741, 4.04, 3.6672864848982974e-05, 0),
                    (3, 2, 4, 3.0261148547590434e-05, 5.05, 1.5281880016533242e-06, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_percent(self):
        assert_records_close(
            from_orders_both(size=[[0.5, -0.5]], size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 0),
                    (2, 0, 2, 4.16666667, 3.0, 0.0, 0),
                    (3, 0, 3, 1.5625, 4.0, 0.0, 0),
                    (4, 0, 4, 0.625, 5.0, 0.0, 0),
                    (0, 1, 0, 50.0, 1.0, 0.0, 1),
                    (1, 1, 1, 12.5, 2.0, 0.0, 1),
                    (2, 1, 2, 4.16666667, 3.0, 0.0, 1),
                    (3, 1, 3, 1.5625, 4.0, 0.0, 1),
                    (4, 1, 4, 0.625, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=[[0.5, -0.5]], size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 0),
                    (2, 0, 2, 4.16666667, 3.0, 0.0, 0),
                    (3, 0, 3, 1.5625, 4.0, 0.0, 0),
                    (4, 0, 4, 0.625, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=[[0.5, -0.5]], size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 1),
                    (1, 0, 1, 12.5, 2.0, 0.0, 1),
                    (2, 0, 2, 4.16666667, 3.0, 0.0, 1),
                    (3, 0, 3, 1.5625, 4.0, 0.0, 1),
                    (4, 0, 4, 0.625, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=0.5,
                size_type="percent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 3.125, 2.0, 0.0, 0),
                    (2, 0, 2, 0.2604166666666667, 3.0, 0.0, 0),
                    (3, 0, 3, 0.0244140625, 4.0, 0.0, 0),
                    (4, 0, 4, 0.00244140625, 5.0, 0.0, 0),
                    (0, 1, 0, 25.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.5625, 2.0, 0.0, 0),
                    (2, 1, 2, 0.13020833333333334, 3.0, 0.0, 0),
                    (3, 1, 3, 0.01220703125, 4.0, 0.0, 0),
                    (4, 1, 4, 0.001220703125, 5.0, 0.0, 0),
                    (0, 2, 0, 12.5, 1.0, 0.0, 0),
                    (1, 2, 1, 0.78125, 2.0, 0.0, 0),
                    (2, 2, 2, 0.06510416666666667, 3.0, 0.0, 0),
                    (3, 2, 3, 0.006103515625, 4.0, 0.0, 0),
                    (4, 2, 4, 0.0006103515625, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_auto_seq(self):
        target_hold_value = pd.DataFrame(
            {
                "a": [0.0, 70.0, 30.0, 0.0, 70.0],
                "b": [30.0, 0.0, 70.0, 30.0, 30.0],
                "c": [70.0, 30.0, 0.0, 70.0, 0.0],
            },
            index=price.index,
        )
        assert_frame_equal(
            from_orders_both(
                close=1.0,
                size=target_hold_value,
                size_type="targetvalue",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                call_seq="auto",
            ).get_asset_value(group_by=False),
            target_hold_value,
        )
        assert_frame_equal(
            from_orders_both(
                close=1.0,
                size=target_hold_value / 100,
                size_type="targetpercent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                call_seq="auto",
            ).get_asset_value(group_by=False),
            target_hold_value,
        )

    def test_max_order_records(self):
        assert from_orders_both(close=price_wide).order_records.shape[0] == 9
        assert from_orders_both(close=price_wide, max_order_records=3).order_records.shape[0] == 9
        assert from_orders_both(close=price_wide, max_order_records=0).order_records.shape[0] == 0
        with pytest.raises(Exception):
            from_orders_both(close=price_wide, max_order_records=2)

    def test_max_log_records(self):
        assert from_orders_both(close=price_wide, log=True).log_records.shape[0] == 15
        assert (
            from_orders_both(close=price_wide, log=True, max_log_records=5).log_records.shape[0]
            == 15
        )
        assert (
            from_orders_both(close=price_wide, log=True, max_log_records=0).log_records.shape[0]
            == 0
        )
        with pytest.raises(Exception):
            from_orders_both(close=price_wide, log=True, max_log_records=4)

    def test_jitted_parallel(self):
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        pf = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200, 300],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            log=True,
            jitted=dict(parallel=True),
        )
        pf2 = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200, 300],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            log=True,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            jitted=dict(parallel=True),
        )
        pf2 = from_orders_both(
            close=price_wide2,
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
        pf = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200, 300],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            log=True,
            chunked=True,
        )
        pf2 = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200, 300],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            log=True,
            chunked=False,
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            chunked=True,
        )
        pf2 = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200],
            size=[[1, 2, 3]],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            chunked=False,
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)

    def test_init_position(self):
        pf = vbt.Portfolio.from_orders(
            close=1, init_cash=0.0, init_position=1.0, size=-np.inf, direction="longonly"
        )
        assert pf.init_position == 1.0
        assert_records_close(
            pf.order_records, np.array([(0, 0, 0, 1.0, 1.0, 0.0, 1)], dtype=order_dt)
        )

    def test_cash_earnings(self):
        pf = vbt.Portfolio.from_orders(1, cash_earnings=[0, 1, 2, 3])
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

    def test_cash_dividends(self):
        pf = vbt.Portfolio.from_orders(1, size=np.inf, cash_dividends=[0, 1, 2, 3])
        assert_series_equal(pf.cash_earnings, pd.Series([0.0, 100.0, 400.0, 1800.0]))
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 2, 100.0, 1.0, 0.0, 0),
                    (2, 0, 3, 400.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    @pytest.mark.parametrize("test_group_by", [False, np.array([0, 0, 1])])
    @pytest.mark.parametrize("test_cash_sharing", [False, True])
    def test_save_returns(self, test_group_by, test_cash_sharing):
        assert_frame_equal(
            from_orders_both(
                close=price_wide,
                save_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_orders_both(
                close=price_wide,
                save_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )
        assert_frame_equal(
            from_orders_longonly(
                close=price_wide,
                save_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_orders_longonly(
                close=price_wide,
                save_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )
        assert_frame_equal(
            from_orders_shortonly(
                close=price_wide,
                save_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_orders_shortonly(
                close=price_wide,
                save_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )

    def test_records(self):
        assert_records_close(
            from_orders_both(
                close=price,
                records=[
                    dict(row=0, size=0.5, size_type="percent"),
                    dict(row=4, size=-np.inf, direction="longonly"),
                ],
                size=np.nan,
            ).order_records,
            np.array([(0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 0, 4, 50.0, 5.0, 0.0, 1)], dtype=order_dt),
        )

    def test_sim_range(self):
        _order_size_wide = order_size_wide.copy()
        _order_size_wide.iloc[:1] = np.nan
        assert_records_close(
            from_orders_both(size=order_size_wide, sim_start=1).order_records,
            from_orders_both(size=_order_size_wide).order_records,
        )
        np.testing.assert_array_equal(
            from_orders_both(size=order_size_wide, sim_start=1).sim_start, np.array([1, 1, 1])
        )
        assert_records_close(
            from_orders_both(size=order_size_wide, sim_start=1).order_records,
            from_orders_both(size=_order_size_wide, sim_start="auto").order_records,
        )
        _order_size_wide = order_size_wide.copy()
        _order_size_wide.iloc[2:] = np.nan
        assert_records_close(
            from_orders_both(size=order_size_wide, sim_end=2).order_records,
            from_orders_both(size=_order_size_wide).order_records,
        )
        np.testing.assert_array_equal(
            from_orders_both(size=order_size_wide, sim_end=2).sim_end, np.array([2, 2, 2])
        )
        assert_records_close(
            from_orders_both(size=order_size_wide, sim_end=2).order_records,
            from_orders_both(size=_order_size_wide, sim_end="auto").order_records,
        )
        _order_size_wide = order_size_wide.copy()
        _order_size_wide.iloc[:1, 0] = np.nan
        _order_size_wide.iloc[:2, 1] = np.nan
        _order_size_wide.iloc[:3, 2] = np.nan
        assert_records_close(
            from_orders_both(size=order_size_wide, sim_start=[1, 2, 3]).order_records,
            from_orders_both(size=_order_size_wide).order_records,
        )
        np.testing.assert_array_equal(
            from_orders_both(size=order_size_wide, sim_start=[1, 2, 3]).sim_start,
            np.array([1, 2, 3]),
        )
        assert_records_close(
            from_orders_both(size=order_size_wide, sim_start=[1, 2, 3]).order_records,
            from_orders_both(size=_order_size_wide, sim_start="auto").order_records,
        )
        with pytest.raises(Exception):
            from_orders_both(size=order_size_wide, sim_start=[1, 2])
        assert_records_close(
            from_orders_both(
                size=order_size_wide, group_by=[0, 0, 1], sim_start=[1, 2, 3]
            ).order_records,
            from_orders_both(size=_order_size_wide, group_by=[0, 0, 1]).order_records,
        )
        np.testing.assert_array_equal(
            from_orders_both(
                size=order_size_wide, group_by=[0, 0, 1], sim_start=[1, 2, 3]
            ).sim_start,
            np.array([1, 3]),
        )
        assert_records_close(
            from_orders_both(
                size=order_size_wide, group_by=[0, 0, 1], sim_start=[1, 2, 3]
            ).order_records,
            from_orders_both(
                size=_order_size_wide, group_by=[0, 0, 1], sim_start="auto"
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                size=order_size_wide, group_by=[0, 0, 1], sim_start=[1, 1, 2]
            ).order_records,
            from_orders_both(
                size=order_size_wide, group_by=[0, 0, 1], sim_start=[1, 2]
            ).order_records,
        )
        _order_size_wide = order_size_wide.copy()
        _order_size_wide.iloc[:1, 0] = np.nan
        _order_size_wide.iloc[:2, 1] = np.nan
        _order_size_wide.iloc[:2, 2] = np.nan
        assert_records_close(
            from_orders_both(
                size=order_size_wide,
                group_by=[0, 0, 1],
                cash_sharing=True,
                sim_start=[1, 2],
            ).order_records,
            from_orders_both(
                size=_order_size_wide,
                group_by=[0, 0, 1],
                cash_sharing=True,
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                size=order_size_wide,
                group_by=[0, 0, 1],
                cash_sharing=True,
                sim_start=[1, 2],
            ).order_records,
            from_orders_both(
                size=_order_size_wide,
                group_by=[0, 0, 1],
                cash_sharing=True,
                sim_start="auto",
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                size=order_size_wide,
                group_by=[0, 0, 1],
                sim_start=[1, 1, 2],
                cash_sharing=True,
            ).order_records,
            from_orders_both(
                size=order_size_wide,
                group_by=[0, 0, 1],
                sim_start=[1, 2],
                cash_sharing=True,
            ).order_records,
        )
