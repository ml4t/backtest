import os
from collections import namedtuple

import pandas as pd
import pytest
import vectorbtpro as vbt
from vectorbtpro._dtypes import *
from vectorbtpro.generic.enums import drawdown_dt
from vectorbtpro.portfolio.enums import *

from tests.utils import *

qs_available = True
try:
    pass
except:
    qs_available = False

price = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    index=pd.date_range("2020", periods=5),
)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# Portfolio ############# #

open_na = pd.DataFrame(
    {
        "a": [np.nan, 1.0, 2.0, 3.0, 4.0],
        "b": [0.5, 1.0, np.nan, 3.0, 4.0],
        "c": [0.5, 1.0, 2.0, 3.0, np.nan],
    },
    index=price.index,
)
high_na = pd.DataFrame(
    {
        "a": [np.nan, 2.5, 3.5, 4.5, 5.5],
        "b": [1.5, 2.5, np.nan, 4.5, 5.5],
        "c": [1.5, 2.5, 3.5, 4.5, np.nan],
    },
    index=price.index,
)
low_na = pd.DataFrame(
    {
        "a": [np.nan, 0.5, 1.5, 2.5, 3.5],
        "b": [0.25, 0.5, np.nan, 2.5, 3.5],
        "c": [0.25, 0.5, 1.5, 2.5, np.nan],
    },
    index=price.index,
)
close_na = pd.DataFrame(
    {
        "a": [np.nan, 2.0, 3.0, 4.0, 5.0],
        "b": [1.0, 2.0, np.nan, 4.0, 5.0],
        "c": [1.0, 2.0, 3.0, 4.0, np.nan],
    },
    index=price.index,
)
order_size_new = pd.DataFrame(
    {
        "a": [0.0, 0.1, -1.0, -0.1, 1.0],
        "b": [0.0, 0.1, -1.0, -0.1, 1.0],
        "c": [1.0, 0.1, -1.0, -0.1, 1.0],
    },
    index=price.index,
)
bm_close_na = pd.DataFrame(
    {
        "a": [5.0, 4.0, 3.0, 2.0, np.nan],
        "b": [5.0, 4.0, np.nan, 2.0, 1.0],
        "c": [np.nan, 4.0, 3.0, 2.0, 1.0],
    },
    index=price.index,
)
init_position = [1.0, -1.0, 0.0]
directions = [["longonly", "shortonly", "both"]]
group_by = pd.Index(["first", "first", "second"], name="group")
group_columns = pd.Index(["first", "second"], dtype="object", name="group")

pf_kwargs = dict(
    open=open_na,
    high=high_na,
    low=low_na,
    close=close_na,
    size=order_size_new,
    size_type="amount",
    direction=directions,
    fees=0.01,
    fixed_fees=0.1,
    slippage=0.01,
    log=True,
    call_seq="reversed",
    group_by=None,
    init_cash=[100.0, 100.0, 100.0],
    init_position=init_position,
    init_price=close_na.bfill().values[0],
    cash_deposits=pd.DataFrame(
        {
            "a": [0.0, 0.0, 100.0, 0.0, 0.0],
            "b": [0.0, 0.0, 100.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=price.index,
    ),
    cash_earnings=0.0,
    freq="1D",
    attach_call_seq=True,
    bm_close=bm_close_na,
)

pf = vbt.Portfolio.from_orders(**pf_kwargs)  # independent
pf_filled = vbt.Portfolio.from_orders(**pf_kwargs, save_returns=True)

pf_grouped_kwargs = dict(
    open=open_na,
    high=high_na,
    low=low_na,
    close=close_na,
    size=order_size_new,
    size_type="amount",
    direction=directions,
    fees=0.01,
    fixed_fees=0.1,
    slippage=0.01,
    log=True,
    call_seq="reversed",
    group_by=group_by,
    cash_sharing=False,
    init_cash=[100.0, 100.0, 100.0],
    init_position=init_position,
    init_price=close_na.bfill().values[0],
    cash_deposits=pd.DataFrame(
        {
            "a": [0.0, 0.0, 100.0, 0.0, 0.0],
            "b": [0.0, 0.0, 100.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=price.index,
    ),
    cash_earnings=0.0,
    freq="1D",
    attach_call_seq=True,
    bm_close=bm_close_na,
)

pf_grouped = vbt.Portfolio.from_orders(**pf_grouped_kwargs)  # grouped
pf_grouped_filled = vbt.Portfolio.from_orders(**pf_grouped_kwargs, save_returns=True)

pf_shared_kwargs = dict(
    open=open_na,
    high=high_na,
    low=low_na,
    close=close_na,
    size=order_size_new,
    size_type="amount",
    direction=directions,
    fees=0.01,
    fixed_fees=0.1,
    slippage=0.01,
    log=True,
    call_seq="reversed",
    group_by=group_by,
    cash_sharing=True,
    init_cash=[200.0, 100.0],
    init_position=init_position,
    init_price=close_na.bfill().values[0],
    cash_deposits=pd.DataFrame(
        {"first": [0.0, 0.0, 200.0, 0.0, 0.0], "second": [0.0, 0.0, 0.0, 0.0, 0.0]},
        index=price.index,
    ),
    cash_earnings=0.0,
    freq="1D",
    attach_call_seq=True,
    bm_close=bm_close_na,
)

pf_shared = vbt.Portfolio.from_orders(**pf_shared_kwargs)  # shared
pf_shared_filled = vbt.Portfolio.from_orders(**pf_shared_kwargs, save_returns=True)


class TestPortfolio:
    def test_row_stack(self):
        pf_kwargs1 = dict(pf_kwargs)
        del pf_kwargs1["direction"]
        del pf_kwargs1["init_position"]
        del pf_kwargs1["init_price"]
        del pf_kwargs1["cash_deposits"]
        del pf_kwargs1["cash_earnings"]
        pf_kwargs1["init_cash"] = "auto"
        pf_kwargs1["size"] = pf_kwargs1["size"].loc["2020-01-01":"2020-01-04", "a"]
        pf_kwargs1["open"] = pf_kwargs1["open"].loc["2020-01-01":"2020-01-04", "a"]
        pf_kwargs1["high"] = pf_kwargs1["high"].loc["2020-01-01":"2020-01-04", "a"]
        pf_kwargs1["low"] = pf_kwargs1["low"].loc["2020-01-01":"2020-01-04", "a"]
        pf_kwargs1["close"] = pf_kwargs1["close"].loc["2020-01-01":"2020-01-04", "a"]
        pf_kwargs1["bm_close"] = pf_kwargs1["bm_close"].loc["2020-01-01":"2020-01-04", "a"]

        def reindex_second_obj(df):
            df = df.copy()
            df.index = pd.date_range("2020-01-06", "2020-01-10")
            return df

        pf_kwargs2 = dict(pf_kwargs)
        del pf_kwargs2["init_position"]
        del pf_kwargs2["init_price"]
        pf_kwargs2["init_cash"] = "auto"
        pf_kwargs2["size"] = reindex_second_obj(pf_kwargs2["size"])
        pf_kwargs2["open"] = reindex_second_obj(pf_kwargs2["open"])
        pf_kwargs2["high"] = reindex_second_obj(pf_kwargs2["high"])
        pf_kwargs2["low"] = reindex_second_obj(pf_kwargs2["low"])
        pf_kwargs2["close"] = reindex_second_obj(pf_kwargs2["close"])
        pf_kwargs2["bm_close"] = reindex_second_obj(pf_kwargs2["bm_close"])
        pf_kwargs2["cash_deposits"] = reindex_second_obj(pf_kwargs2["cash_deposits"])
        pf_kwargs2["cash_earnings"] = pf_kwargs2["cash_deposits"].shift(1).fillna(0) // 5

        pf1 = vbt.Portfolio.from_orders(**pf_kwargs1)
        pf2 = vbt.Portfolio.from_orders(**pf_kwargs2)
        new_pf = vbt.Portfolio.row_stack(pf1, pf2)

        def tile_first_obj(arr):
            return np.column_stack((arr, arr, arr))

        assert_index_equal(new_pf.wrapper.index, pf1.wrapper.index.append(pf2.wrapper.index))
        assert_index_equal(new_pf.wrapper.columns, pf2.wrapper.columns)
        assert new_pf.wrapper.grouper.group_by is None
        np.testing.assert_array_equal(
            new_pf._open,
            np.row_stack((tile_first_obj(pf1._open), pf2._open)),
        )
        np.testing.assert_array_equal(
            new_pf._high,
            np.row_stack((tile_first_obj(pf1._high), pf2._high)),
        )
        np.testing.assert_array_equal(
            new_pf._low,
            np.row_stack((tile_first_obj(pf1._low), pf2._low)),
        )
        np.testing.assert_array_equal(
            new_pf._close,
            np.row_stack((tile_first_obj(pf1._close), pf2._close)),
        )
        assert_records_close(
            new_pf.orders.values,
            vbt.Orders.row_stack(pf1.orders, pf2.orders).values,
        )
        assert_records_close(
            new_pf.logs.values,
            vbt.Logs.row_stack(pf1.logs, pf2.logs).values,
        )
        assert not new_pf.cash_sharing
        assert new_pf._init_cash == InitCashMode.Auto
        np.testing.assert_array_equal(new_pf._init_position, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(new_pf._init_price, np.array([np.nan, np.nan, np.nan]))
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.row_stack((tile_first_obj(pf1.wrapper.wrap(0.0)), pf2._cash_deposits)),
        )
        np.testing.assert_array_equal(
            new_pf._cash_earnings,
            np.row_stack((tile_first_obj(pf1.wrapper.wrap(0.0)), pf2._cash_earnings)),
        )
        np.testing.assert_array_equal(
            new_pf._call_seq,
            np.row_stack((tile_first_obj(pf1._call_seq), pf2._call_seq)),
        )
        np.testing.assert_array_equal(
            new_pf._bm_close,
            np.row_stack((tile_first_obj(pf1._bm_close), pf2._bm_close)),
        )
        new_pf = vbt.Portfolio.row_stack(pf1.replace(init_cash=100.0), pf2)
        np.testing.assert_array_equal(
            new_pf._init_cash,
            np.array([100.0, 100.0, 100.0]),
        )
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.30402, 0.29998, 1.4241200000000003],
                    [0.0, 0.0, 0.0],
                    [100.0, 100.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        new_pf = vbt.Portfolio.row_stack(pf1.replace(init_cash=100.0), pf2.replace(init_cash=50.0))
        np.testing.assert_array_equal(
            new_pf._init_cash,
            np.array([100.0, 100.0, 100.0]),
        )
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [50.0, 50.0, 50.0],
                    [0.0, 0.0, 0.0],
                    [100.0, 100.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(pf1.replace(init_cash=InitCashMode.AutoAlign), pf2)
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(init_cash=100.0, cash_deposits=np.array([[0.0]])),
            pf2.replace(init_cash=0.0, cash_deposits=np.array([[0.0]])),
        )
        np.testing.assert_array_equal(
            new_pf._init_cash,
            np.array([100.0, 100.0, 100.0]),
        )
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.array([[0.0]]),
        )
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(cash_deposits=np.array([[0.0]])),
            pf2.replace(init_cash=np.array([10, 20, 30]), cash_deposits=np.array([[0.0]])),
            combine_init_cash=True,
        )
        np.testing.assert_array_equal(
            new_pf._init_cash,
            np.array([13.30968, 23.30968, 33.30968]),
        )
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.array([[0.0]]),
        )
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(init_cash=100.0, cash_deposits=np.array([[0.0]])),
            pf2.replace(init_cash=np.array([10, 20, 30]), cash_deposits=np.array([[0.0]])),
            combine_init_cash=True,
        )
        np.testing.assert_array_equal(
            new_pf._init_cash,
            np.array([110.0, 120.0, 130.0]),
        )
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.array([[0.0]]),
        )
        new_pf = vbt.Portfolio.row_stack(pf1.replace(init_position=1.0), pf2)
        np.testing.assert_array_equal(
            new_pf._init_position,
            np.array([1.0, 1.0, 1.0]),
        )
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(init_position=1.0), pf2.replace(init_position=0.0)
        )
        np.testing.assert_array_equal(
            new_pf._init_position,
            np.array([1.0, 1.0, 1.0]),
        )
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(init_position=0.0),
            pf2.replace(init_position=1.0),
            combine_init_position=True,
        )
        np.testing.assert_array_equal(
            new_pf._init_position,
            np.array([1.0, 1.0, 1.0]),
        )
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(
                pf1.replace(init_position=0.0),
                pf2.replace(init_position=1.0),
            )
        new_pf = vbt.Portfolio.row_stack(pf1.replace(init_position=1.0, init_price=10.0), pf2)
        np.testing.assert_array_equal(
            new_pf._init_position,
            np.array([1.0, 1.0, 1.0]),
        )
        np.testing.assert_array_equal(
            new_pf._init_price,
            np.array([10.0, 10.0, 10.0]),
        )
        new_pf = vbt.Portfolio.row_stack(
            pf1,
            pf2.replace(init_position=1.0, init_price=10.0),
            combine_init_position=True,
            combine_init_price=True,
        )
        np.testing.assert_array_equal(
            new_pf._init_position,
            np.array([1.0, 1.0, 1.0]),
        )
        np.testing.assert_array_equal(
            new_pf._init_price,
            np.array([10.0, 10.0, 10.0]),
        )
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(init_position=1.0, init_price=10.0),
            pf2.replace(init_position=2.0, init_price=15.0),
            combine_init_position=True,
            combine_init_price=True,
        )
        np.testing.assert_array_equal(
            new_pf._init_position,
            np.array([3.0, 3.0, 3.0]),
        )
        np.testing.assert_array_equal(
            new_pf._init_price,
            np.array([13.333333333333334, 13.333333333333334, 13.333333333333334]),
        )
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(cash_deposits=np.array([[0.0]]), cash_earnings=np.array([[0.0]])),
            pf2.replace(cash_deposits=np.array([[0.0]]), cash_earnings=np.array([[0.0]])),
        )
        np.testing.assert_array_equal(new_pf._cash_deposits, np.array([[0.0]]))
        np.testing.assert_array_equal(new_pf._cash_earnings, np.array([[0.0]]))
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(pf1.replace(call_seq=None), pf2)
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(pf1, pf2.replace(call_seq=None))
        new_pf = vbt.Portfolio.row_stack(pf1.replace(call_seq=None), pf2.replace(call_seq=None))
        assert new_pf._call_seq is None
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(pf1.replace(bm_close=None), pf2)
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(pf1, pf2.replace(bm_close=None))
        new_pf = vbt.Portfolio.row_stack(pf1.replace(bm_close=None), pf2.replace(bm_close=None))
        assert new_pf._bm_close is None
        in_outputs1 = dict(cash_flow=np.arange(4).reshape((4, 1)))
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(cash_flow=np.arange(15).reshape((5, 3)))
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.cash_flow,
            np.row_stack((tile_first_obj(in_outputs1.cash_flow), in_outputs2.cash_flow)),
        )
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(
                pf1.replace(in_outputs=None),
                pf2.replace(in_outputs=in_outputs2),
            )
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(
                pf1.replace(in_outputs=in_outputs2),
                pf2.replace(in_outputs=None),
            )
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(in_outputs=None),
            pf2.replace(in_outputs=None),
        )
        assert new_pf.in_outputs is None
        in_outputs1 = dict(cash_flow1=np.arange(4).reshape((4, 1)))
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(cash_flow2=np.arange(15).reshape((5, 3)))
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(
                pf1.replace(in_outputs=in_outputs1),
                pf2.replace(in_outputs=in_outputs2),
            )
        in_outputs1 = dict(total_return=np.arange(4).reshape((4, 1)))
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(total_return=np.arange(15).reshape((5, 3)))
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(
                pf1.replace(in_outputs=in_outputs1),
                pf2.replace(in_outputs=in_outputs2),
            )

        pf_kwargs1 = dict(pf_grouped_kwargs)
        del pf_kwargs1["init_position"]
        del pf_kwargs1["init_price"]
        pf_kwargs1["size"] = pf_kwargs1["size"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["open"] = pf_kwargs1["open"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["high"] = pf_kwargs1["high"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["low"] = pf_kwargs1["low"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["close"] = pf_kwargs1["close"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["bm_close"] = pf_kwargs1["bm_close"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["cash_deposits"] = pf_kwargs1["cash_deposits"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["cash_earnings"] = pf_kwargs1["cash_deposits"].shift(1).fillna(0) // 5

        pf_kwargs2 = dict(pf_grouped_kwargs)
        del pf_kwargs2["init_position"]
        del pf_kwargs2["init_price"]
        pf_kwargs2["size"] = reindex_second_obj(pf_kwargs2["size"])
        pf_kwargs2["open"] = reindex_second_obj(pf_kwargs2["open"])
        pf_kwargs2["high"] = reindex_second_obj(pf_kwargs2["high"])
        pf_kwargs2["low"] = reindex_second_obj(pf_kwargs2["low"])
        pf_kwargs2["close"] = reindex_second_obj(pf_kwargs2["close"])
        pf_kwargs2["bm_close"] = reindex_second_obj(pf_kwargs2["bm_close"])
        pf_kwargs2["cash_deposits"] = reindex_second_obj(pf_kwargs2["cash_deposits"])
        pf_kwargs2["cash_earnings"] = pf_kwargs2["cash_deposits"].shift(1).fillna(0) // 5

        pf1 = vbt.Portfolio.from_orders(**pf_kwargs1)
        pf2 = vbt.Portfolio.from_orders(**pf_kwargs2)
        new_pf = vbt.Portfolio.row_stack(pf1, pf2, combine_init_cash=True)
        assert_index_equal(new_pf.wrapper.index, pf1.wrapper.index.append(pf2.wrapper.index))
        assert_index_equal(new_pf.wrapper.columns, pf2.wrapper.columns)
        assert_index_equal(
            new_pf.wrapper.grouper.group_by,
            pd.Index(["first", "first", "second"], dtype="object", name="group"),
        )
        np.testing.assert_array_equal(
            new_pf._open,
            np.row_stack((pf1._open, pf2._open)),
        )
        np.testing.assert_array_equal(
            new_pf._high,
            np.row_stack((pf1._high, pf2._high)),
        )
        np.testing.assert_array_equal(
            new_pf._low,
            np.row_stack((pf1._low, pf2._low)),
        )
        np.testing.assert_array_equal(
            new_pf._close,
            np.row_stack((pf1._close, pf2._close)),
        )
        assert_records_close(
            new_pf.orders.values,
            vbt.Orders.row_stack(pf1.orders, pf2.orders).values,
        )
        assert_records_close(
            new_pf.logs.values,
            vbt.Logs.row_stack(pf1.logs, pf2.logs).values,
        )
        assert not new_pf.cash_sharing
        np.testing.assert_array_equal(new_pf._init_cash, pf1._init_cash + pf2._init_cash)
        np.testing.assert_array_equal(new_pf._init_position, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(new_pf._init_price, np.array([np.nan, np.nan, np.nan]))
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.row_stack((pf1._cash_deposits, pf2._cash_deposits)),
        )
        np.testing.assert_array_equal(
            new_pf._cash_earnings,
            np.row_stack((pf1._cash_earnings, pf2._cash_earnings)),
        )
        np.testing.assert_array_equal(
            new_pf._call_seq,
            np.row_stack((pf1._call_seq, pf2._call_seq)),
        )
        np.testing.assert_array_equal(
            new_pf._bm_close,
            np.row_stack((pf1._bm_close, pf2._bm_close)),
        )
        new_pf = vbt.Portfolio.row_stack(
            pf1, pf2, combine_init_cash=True, wrapper_kwargs=dict(group_by=False)
        )
        assert_index_equal(new_pf.wrapper.index, pf1.wrapper.index.append(pf2.wrapper.index))
        assert_index_equal(new_pf.wrapper.columns, pf2.wrapper.columns)
        assert new_pf.wrapper.grouper.group_by is None
        np.testing.assert_array_equal(
            new_pf._open,
            np.row_stack((pf1._open, pf2._open)),
        )
        np.testing.assert_array_equal(
            new_pf._high,
            np.row_stack((pf1._high, pf2._high)),
        )
        np.testing.assert_array_equal(
            new_pf._low,
            np.row_stack((pf1._low, pf2._low)),
        )
        np.testing.assert_array_equal(
            new_pf._close,
            np.row_stack((pf1._close, pf2._close)),
        )
        assert_records_close(
            new_pf.orders.values,
            vbt.Orders.row_stack(pf1.orders, pf2.orders).values,
        )
        assert_records_close(
            new_pf.logs.values,
            vbt.Logs.row_stack(pf1.logs, pf2.logs).values,
        )
        assert not new_pf.cash_sharing
        np.testing.assert_array_equal(new_pf._init_cash, pf1._init_cash + pf2._init_cash)
        np.testing.assert_array_equal(new_pf._init_position, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(new_pf._init_price, np.array([np.nan, np.nan, np.nan]))
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.row_stack((pf1._cash_deposits, pf2._cash_deposits)),
        )
        np.testing.assert_array_equal(
            new_pf._cash_earnings,
            np.row_stack((pf1._cash_earnings, pf2._cash_earnings)),
        )
        np.testing.assert_array_equal(
            new_pf._call_seq,
            np.row_stack((pf1._call_seq, pf2._call_seq)),
        )
        np.testing.assert_array_equal(
            new_pf._bm_close,
            np.row_stack((pf1._bm_close, pf2._bm_close)),
        )
        in_outputs1 = dict(
            arr_2d_cs=np.arange(12).reshape((4, 3)),
            arr_2d_pcg=np.arange(8).reshape((4, 2)),
            arr_2d_pg=np.arange(8).reshape((4, 2)),
            arr_2d_pc=np.arange(12).reshape((4, 3)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr_2d_cs=np.arange(15).reshape((5, 3)),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_cs,
            np.row_stack((in_outputs1.arr_2d_cs, in_outputs2.arr_2d_cs)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pcg,
            np.row_stack((in_outputs1.arr_2d_pcg, in_outputs2.arr_2d_pcg)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pg,
            np.row_stack((in_outputs1.arr_2d_pg, in_outputs2.arr_2d_pg)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pc,
            np.row_stack((in_outputs1.arr_2d_pc, in_outputs2.arr_2d_pc)),
        )
        in_outputs1 = dict(
            arr=np.arange(8).reshape((4, 2)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr=np.arange(10).reshape((5, 2)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr, np.row_stack((in_outputs1.arr, in_outputs2.arr))
        )
        in_outputs1 = dict(
            arr=np.arange(12).reshape((4, 3)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr=np.arange(15).reshape((5, 3)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(
                pf1.replace(in_outputs=in_outputs1),
                pf2.replace(in_outputs=in_outputs2),
            )

        pf_kwargs1 = dict(pf_shared_kwargs)
        pf_kwargs1["cash_earnings"] = pf_kwargs["cash_deposits"].shift(1).fillna(0) // 5
        del pf_kwargs1["init_position"]
        del pf_kwargs1["init_price"]
        pf_kwargs1["size"] = pf_kwargs1["size"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["open"] = pf_kwargs1["open"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["high"] = pf_kwargs1["high"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["low"] = pf_kwargs1["low"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["close"] = pf_kwargs1["close"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["bm_close"] = pf_kwargs1["bm_close"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["cash_deposits"] = pf_kwargs1["cash_deposits"].loc["2020-01-01":"2020-01-04"]
        pf_kwargs1["cash_earnings"] = pf_kwargs1["cash_earnings"].loc["2020-01-01":"2020-01-04"]

        pf_kwargs2 = dict(pf_shared_kwargs)
        pf_kwargs2["cash_earnings"] = pf_kwargs["cash_deposits"].shift(1).fillna(0) // 5
        del pf_kwargs2["init_position"]
        del pf_kwargs2["init_price"]
        pf_kwargs2["size"] = reindex_second_obj(pf_kwargs2["size"])
        pf_kwargs2["open"] = reindex_second_obj(pf_kwargs2["open"])
        pf_kwargs2["high"] = reindex_second_obj(pf_kwargs2["high"])
        pf_kwargs2["low"] = reindex_second_obj(pf_kwargs2["low"])
        pf_kwargs2["close"] = reindex_second_obj(pf_kwargs2["close"])
        pf_kwargs2["bm_close"] = reindex_second_obj(pf_kwargs2["bm_close"])
        pf_kwargs2["cash_deposits"] = reindex_second_obj(pf_kwargs2["cash_deposits"])
        pf_kwargs2["cash_earnings"] = reindex_second_obj(pf_kwargs2["cash_earnings"])

        pf1 = vbt.Portfolio.from_orders(**pf_kwargs1)
        pf2 = vbt.Portfolio.from_orders(**pf_kwargs2)
        new_pf = vbt.Portfolio.row_stack(pf1, pf2, combine_init_cash=True)
        assert_index_equal(new_pf.wrapper.index, pf1.wrapper.index.append(pf2.wrapper.index))
        assert_index_equal(new_pf.wrapper.columns, pf2.wrapper.columns)
        assert_index_equal(
            new_pf.wrapper.grouper.group_by,
            pd.Index(["first", "first", "second"], dtype="object", name="group"),
        )
        np.testing.assert_array_equal(
            new_pf._open,
            np.row_stack((pf1._open, pf2._open)),
        )
        np.testing.assert_array_equal(
            new_pf._high,
            np.row_stack((pf1._high, pf2._high)),
        )
        np.testing.assert_array_equal(
            new_pf._low,
            np.row_stack((pf1._low, pf2._low)),
        )
        np.testing.assert_array_equal(
            new_pf._close,
            np.row_stack((pf1._close, pf2._close)),
        )
        assert_records_close(
            new_pf.orders.values,
            vbt.Orders.row_stack(pf1.orders, pf2.orders).values,
        )
        assert_records_close(
            new_pf.logs.values,
            vbt.Logs.row_stack(pf1.logs, pf2.logs).values,
        )
        assert new_pf.cash_sharing
        np.testing.assert_array_equal(new_pf._init_cash, pf1._init_cash + pf2._init_cash)
        np.testing.assert_array_equal(new_pf._init_position, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(new_pf._init_price, np.array([np.nan, np.nan, np.nan]))
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            np.row_stack((pf1._cash_deposits, pf2._cash_deposits)),
        )
        np.testing.assert_array_equal(
            new_pf._cash_earnings,
            np.row_stack((pf1._cash_earnings, pf2._cash_earnings)),
        )
        np.testing.assert_array_equal(
            new_pf._call_seq,
            np.row_stack((pf1._call_seq, pf2._call_seq)),
        )
        np.testing.assert_array_equal(
            new_pf._bm_close,
            np.row_stack((pf1._bm_close, pf2._bm_close)),
        )
        in_outputs1 = dict(
            arr_2d_cs=np.arange(8).reshape((4, 2)),
            arr_2d_pcg=np.arange(8).reshape((4, 2)),
            arr_2d_pg=np.arange(8).reshape((4, 2)),
            arr_2d_pc=np.arange(12).reshape((4, 3)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr_2d_cs=np.arange(10).reshape((5, 2)),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_cs,
            np.row_stack((in_outputs1.arr_2d_cs, in_outputs2.arr_2d_cs)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pcg,
            np.row_stack((in_outputs1.arr_2d_pcg, in_outputs2.arr_2d_pcg)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pg,
            np.row_stack((in_outputs1.arr_2d_pg, in_outputs2.arr_2d_pg)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pc,
            np.row_stack((in_outputs1.arr_2d_pc, in_outputs2.arr_2d_pc)),
        )
        in_outputs1 = dict(
            arr=np.arange(8).reshape((4, 2)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr=np.arange(10).reshape((5, 2)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.row_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr, np.row_stack((in_outputs1.arr, in_outputs2.arr))
        )
        in_outputs1 = dict(
            arr=np.arange(12).reshape((4, 3)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr=np.arange(15).reshape((5, 3)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        with pytest.raises(Exception):
            vbt.Portfolio.row_stack(
                pf1.replace(in_outputs=in_outputs1),
                pf2.replace(in_outputs=in_outputs2),
            )

        index1 = pd.date_range("2020", periods=100)
        index2 = pd.date_range(index1[-1], periods=101, inclusive="right")
        close1 = pd.DataFrame({"a": pd.Series(np.arange(1, 101), index=index1)})
        close2 = pd.DataFrame(
            {
                "a": pd.Series(np.arange(101, 201), index=index2),
                "b": pd.Series(np.arange(101, 201), index=index2),
            }
        )
        sim_range_pf1 = vbt.Portfolio.from_orders(close1, size=1, sim_start=10)
        sim_range_pf2 = vbt.Portfolio.from_orders(close2, size=1, sim_end=[20, -30])
        new_sim_range_pf = vbt.Portfolio.row_stack(sim_range_pf1, sim_range_pf2)
        np.testing.assert_array_equal(new_sim_range_pf._sim_start, np.array([10, 10]))
        np.testing.assert_array_equal(new_sim_range_pf._sim_end, np.array([120, 170]))

    def test_column_stack(self):
        pf_kwargs1 = dict(pf_kwargs)
        pf_kwargs1["size"] = pf_kwargs1["size"]["a"]
        pf_kwargs1["open"] = pf_kwargs1["open"]["a"]
        pf_kwargs1["high"] = pf_kwargs1["high"]["a"]
        pf_kwargs1["low"] = pf_kwargs1["low"]["a"]
        pf_kwargs1["close"] = pf_kwargs1["close"]["a"]
        pf_kwargs1["bm_close"] = pf_kwargs1["bm_close"]["a"]
        pf_kwargs1["cash_deposits"] = pf_kwargs1["cash_deposits"]["a"]
        pf_kwargs1["cash_earnings"] = pf_kwargs1["cash_deposits"].shift(1).fillna(0) // 5
        pf_kwargs1["direction"] = pf_kwargs1["direction"][0][0]
        pf_kwargs1["init_position"] = pf_kwargs1["init_position"][0]
        pf_kwargs1["init_price"] = pf_kwargs1["init_price"][0]
        pf_kwargs1["init_cash"] = "auto"

        def reindex_second_obj(df):
            df = df.copy()
            df.index = pd.date_range("2020-01-03", "2020-01-07")
            return df.add_prefix("_")

        def column_stack_arrs(arrs, pf1, pf2, fill_value=np.nan):
            df1 = pd.DataFrame(arrs[0], index=pf1.wrapper.index)
            df2 = pd.DataFrame(arrs[1], index=pf2.wrapper.index)
            df1 = df1.reindex(df1.index.union(df2.index), fill_value=fill_value)
            df2 = df2.reindex(df1.index.union(df2.index), fill_value=fill_value)
            out = np.column_stack((df1.values, df2.values))
            return out

        pf_kwargs2 = dict(pf_kwargs)
        pf_kwargs2["size"] = reindex_second_obj(pf_kwargs2["size"])
        pf_kwargs2["open"] = reindex_second_obj(pf_kwargs2["open"])
        pf_kwargs2["high"] = reindex_second_obj(pf_kwargs2["high"])
        pf_kwargs2["low"] = reindex_second_obj(pf_kwargs2["low"])
        pf_kwargs2["close"] = reindex_second_obj(pf_kwargs2["close"])
        pf_kwargs2["bm_close"] = reindex_second_obj(pf_kwargs2["bm_close"])
        pf_kwargs2["cash_deposits"] = reindex_second_obj(pf_kwargs2["cash_deposits"])
        pf_kwargs2["cash_earnings"] = pf_kwargs2["cash_deposits"].shift(1).fillna(0) // 5
        pf_kwargs2["init_cash"] = "auto"

        pf1 = vbt.Portfolio.from_orders(**pf_kwargs1)
        pf2 = vbt.Portfolio.from_orders(**pf_kwargs2)
        new_pf = vbt.Portfolio.column_stack(pf1, pf2)
        assert_index_equal(new_pf.wrapper.index, pf1.wrapper.index.union(pf2.wrapper.index))
        assert_index_equal(new_pf.wrapper.columns, pf1.wrapper.columns.append(pf2.wrapper.columns))
        assert new_pf.wrapper.grouper.group_by is None
        np.testing.assert_array_equal(
            new_pf._open,
            column_stack_arrs((pf1._open, pf2._open), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._high,
            column_stack_arrs((pf1._high, pf2._high), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._low,
            column_stack_arrs((pf1._low, pf2._low), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._close,
            column_stack_arrs((pf1._close, pf2._close), pf1, pf2),
        )
        assert_records_close(
            new_pf.orders.values,
            vbt.Orders.column_stack(pf1.orders, pf2.orders).values,
        )
        assert_records_close(
            new_pf.logs.values,
            vbt.Logs.column_stack(pf1.logs, pf2.logs).values,
        )
        assert not new_pf.cash_sharing
        assert new_pf._init_cash == InitCashMode.Auto
        np.testing.assert_array_equal(
            new_pf._init_position, np.concatenate((pf1._init_position, pf2._init_position))
        )
        np.testing.assert_array_equal(
            new_pf._init_price, np.concatenate((pf1._init_price, pf2._init_price))
        )
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            column_stack_arrs((pf1._cash_deposits, pf2._cash_deposits), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._cash_earnings,
            column_stack_arrs((pf1._cash_earnings, pf2._cash_earnings), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._call_seq,
            column_stack_arrs((pf1._call_seq, pf2._call_seq), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._bm_close,
            column_stack_arrs((pf1._bm_close, pf2._bm_close), pf1, pf2),
        )
        new_pf = vbt.Portfolio.column_stack(pf1.replace(init_cash=100.0), pf2)
        np.testing.assert_array_equal(
            new_pf._init_cash,
            np.array([100.0, 0.30402, 0.29998, 1.4241200000000003]),
        )
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(init_cash=100.0), pf2.replace(init_cash=50.0)
        )
        np.testing.assert_array_equal(
            new_pf._init_cash,
            np.array([100.0, 50.0, 50.0, 50.0]),
        )
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(init_position=0.0), pf2.replace(init_position=0.0)
        )
        np.testing.assert_array_equal(
            new_pf._init_position,
            np.array([0.0]),
        )
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(init_price=np.nan), pf2.replace(init_price=np.nan)
        )
        np.testing.assert_array_equal(
            new_pf._init_price,
            np.array([np.nan]),
        )
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(cash_deposits=np.array([[0.0]]), cash_earnings=np.array([[0.0]])),
            pf2.replace(cash_deposits=np.array([[0.0]]), cash_earnings=np.array([[0.0]])),
        )
        np.testing.assert_array_equal(new_pf._cash_deposits, np.array([[0.0]]))
        np.testing.assert_array_equal(new_pf._cash_earnings, np.array([[0.0]]))
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(pf1.replace(call_seq=None), pf2)
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(pf1, pf2.replace(call_seq=None))
        new_pf = vbt.Portfolio.column_stack(pf1.replace(call_seq=None), pf2.replace(call_seq=None))
        assert new_pf._call_seq is None
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(pf1.replace(bm_close=None), pf2)
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(pf1, pf2.replace(bm_close=None))
        new_pf = vbt.Portfolio.column_stack(pf1.replace(bm_close=None), pf2.replace(bm_close=None))
        assert new_pf._bm_close is None
        in_outputs1 = dict(
            cash_flow=np.arange(5).reshape((5, 1)),
            total_return=np.arange(1),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            cash_flow=np.arange(15).reshape((5, 3)),
            total_return=np.arange(3),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.cash_flow,
            column_stack_arrs((in_outputs1.cash_flow, in_outputs2.cash_flow), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.total_return,
            np.concatenate((in_outputs1.total_return, in_outputs2.total_return)),
        )
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(
                pf1.replace(in_outputs=None),
                pf2.replace(in_outputs=in_outputs2),
            )
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(
                pf1.replace(in_outputs=in_outputs2),
                pf2.replace(in_outputs=None),
            )
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(in_outputs=None),
            pf2.replace(in_outputs=None),
        )
        assert new_pf.in_outputs is None
        in_outputs1 = dict(cash_flow1=np.arange(5).reshape((5, 1)))
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(cash_flow2=np.arange(15).reshape((5, 3)))
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(
                pf1.replace(in_outputs=in_outputs1),
                pf2.replace(in_outputs=in_outputs2),
            )
        in_outputs1 = dict(total_return=np.arange(5).reshape((5, 1)))
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(total_return=np.arange(15).reshape((5, 3)))
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(
                pf1.replace(in_outputs=in_outputs1),
                pf2.replace(in_outputs=in_outputs2),
            )

        pf_kwargs1 = dict(pf_grouped_kwargs)
        pf_kwargs1["cash_earnings"] = pf_kwargs1["cash_deposits"].shift(1).fillna(0) // 5
        pf_kwargs1["init_cash"] = "auto"
        pf_kwargs1["group_by"] = pd.Index(["first", "first", "second"], name="group")

        pf_kwargs2 = dict(pf_grouped_kwargs)
        pf_kwargs2["size"] = reindex_second_obj(pf_kwargs2["size"])
        pf_kwargs2["open"] = reindex_second_obj(pf_kwargs2["open"])
        pf_kwargs2["high"] = reindex_second_obj(pf_kwargs2["high"])
        pf_kwargs2["low"] = reindex_second_obj(pf_kwargs2["low"])
        pf_kwargs2["close"] = reindex_second_obj(pf_kwargs2["close"])
        pf_kwargs2["bm_close"] = reindex_second_obj(pf_kwargs2["bm_close"])
        pf_kwargs2["cash_deposits"] = reindex_second_obj(pf_kwargs2["cash_deposits"])
        pf_kwargs2["cash_earnings"] = pf_kwargs2["cash_deposits"].shift(1).fillna(0) // 5
        pf_kwargs2["init_cash"] = "auto"
        pf_kwargs2["group_by"] = pd.Index(["third", "fourth", "fourth"], name="group")

        pf1 = vbt.Portfolio.from_orders(**pf_kwargs1)
        pf2 = vbt.Portfolio.from_orders(**pf_kwargs2)
        new_pf = vbt.Portfolio.column_stack(pf1, pf2)
        assert_index_equal(new_pf.wrapper.index, pf1.wrapper.index.union(pf2.wrapper.index))
        assert_index_equal(new_pf.wrapper.columns, pf1.wrapper.columns.append(pf2.wrapper.columns))
        assert_index_equal(
            new_pf.wrapper.grouper.group_by,
            pd.Index(["first", "first", "second", "third", "fourth", "fourth"], name="group"),
        )
        np.testing.assert_array_equal(
            new_pf._open,
            column_stack_arrs((pf1._open, pf2._open), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._high,
            column_stack_arrs((pf1._high, pf2._high), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._low,
            column_stack_arrs((pf1._low, pf2._low), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._close,
            column_stack_arrs((pf1._close, pf2._close), pf1, pf2),
        )
        assert_records_close(
            new_pf.orders.values,
            vbt.Orders.column_stack(pf1.orders, pf2.orders).values,
        )
        assert_records_close(
            new_pf.logs.values,
            vbt.Logs.column_stack(pf1.logs, pf2.logs).values,
        )
        assert not new_pf.cash_sharing
        assert new_pf._init_cash == InitCashMode.Auto
        np.testing.assert_array_equal(
            new_pf._init_position, np.concatenate((pf1._init_position, pf2._init_position))
        )
        np.testing.assert_array_equal(
            new_pf._init_price, np.concatenate((pf1._init_price, pf2._init_price))
        )
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            column_stack_arrs((pf1._cash_deposits, pf2._cash_deposits), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._cash_earnings,
            column_stack_arrs((pf1._cash_earnings, pf2._cash_earnings), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._call_seq,
            column_stack_arrs((pf1._call_seq, pf2._call_seq), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._bm_close,
            column_stack_arrs((pf1._bm_close, pf2._bm_close), pf1, pf2),
        )
        in_outputs1 = dict(
            arr_2d_cs=np.arange(15).reshape((5, 3)),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d_cs=np.arange(3),
            arr_1d_pcg=np.arange(2),
            arr_1d_pg=np.arange(2),
            arr_1d_pc=np.arange(3),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr_2d_cs=np.arange(15).reshape((5, 3)),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d_cs=np.arange(3),
            arr_1d_pcg=np.arange(2),
            arr_1d_pg=np.arange(2),
            arr_1d_pc=np.arange(3),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_cs,
            column_stack_arrs((in_outputs1.arr_2d_cs, in_outputs2.arr_2d_cs), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pcg,
            column_stack_arrs((in_outputs1.arr_2d_pcg, in_outputs2.arr_2d_pcg), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pg,
            column_stack_arrs((in_outputs1.arr_2d_pg, in_outputs2.arr_2d_pg), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pc,
            column_stack_arrs((in_outputs1.arr_2d_pc, in_outputs2.arr_2d_pc), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_1d_cs,
            np.concatenate((in_outputs1.arr_1d_cs, in_outputs2.arr_1d_cs)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_1d_pcg,
            np.concatenate((in_outputs1.arr_1d_pcg, in_outputs2.arr_1d_pcg)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_1d_pg,
            np.concatenate((in_outputs1.arr_1d_pg, in_outputs2.arr_1d_pg)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_1d_pc,
            np.concatenate((in_outputs1.arr_1d_pc, in_outputs2.arr_1d_pc)),
        )
        in_outputs1 = dict(
            arr=np.arange(10).reshape((5, 2)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr=np.arange(10).reshape((5, 2)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr, column_stack_arrs((in_outputs1.arr, in_outputs2.arr), pf1, pf2)
        )
        in_outputs1 = dict(
            arr=np.arange(15).reshape((5, 3)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr=np.arange(15).reshape((5, 3)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(
                pf1.replace(in_outputs=in_outputs1),
                pf2.replace(in_outputs=in_outputs2),
            )

        pf_kwargs1 = dict(pf_shared_kwargs)
        pf_kwargs1["cash_earnings"] = pf_kwargs["cash_deposits"].shift(1).fillna(0) // 5
        pf_kwargs1["init_cash"] = "auto"
        pf_kwargs1["group_by"] = pd.Index(["first", "first", "second"], name="group")

        pf_kwargs2 = dict(pf_shared_kwargs)
        pf_kwargs2["size"] = reindex_second_obj(pf_kwargs2["size"])
        pf_kwargs2["open"] = reindex_second_obj(pf_kwargs2["open"])
        pf_kwargs2["high"] = reindex_second_obj(pf_kwargs2["high"])
        pf_kwargs2["low"] = reindex_second_obj(pf_kwargs2["low"])
        pf_kwargs2["close"] = reindex_second_obj(pf_kwargs2["close"])
        pf_kwargs2["bm_close"] = reindex_second_obj(pf_kwargs2["bm_close"])
        pf_kwargs2["cash_deposits"] = reindex_second_obj(pf_kwargs2["cash_deposits"])
        pf_kwargs2["cash_earnings"] = (
            reindex_second_obj(pf_kwargs["cash_deposits"]).shift(1).fillna(0) // 5
        )
        pf_kwargs2["init_cash"] = "auto"
        pf_kwargs2["group_by"] = pd.Index(["third", "fourth", "fourth"], name="group")

        pf1 = vbt.Portfolio.from_orders(**pf_kwargs1)
        pf2 = vbt.Portfolio.from_orders(**pf_kwargs2)
        new_pf = vbt.Portfolio.column_stack(pf1, pf2)
        assert_index_equal(new_pf.wrapper.index, pf1.wrapper.index.union(pf2.wrapper.index))
        assert_index_equal(new_pf.wrapper.columns, pf1.wrapper.columns.append(pf2.wrapper.columns))
        assert_index_equal(
            new_pf.wrapper.grouper.group_by,
            pd.Index(["first", "first", "second", "third", "fourth", "fourth"], name="group"),
        )
        np.testing.assert_array_equal(
            new_pf._open,
            column_stack_arrs((pf1._open, pf2._open), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._high,
            column_stack_arrs((pf1._high, pf2._high), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._low,
            column_stack_arrs((pf1._low, pf2._low), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf._close,
            column_stack_arrs((pf1._close, pf2._close), pf1, pf2),
        )
        assert_records_close(
            new_pf.orders.values,
            vbt.Orders.column_stack(pf1.orders, pf2.orders).values,
        )
        assert_records_close(
            new_pf.logs.values,
            vbt.Logs.column_stack(pf1.logs, pf2.logs).values,
        )
        assert new_pf.cash_sharing
        assert new_pf._init_cash == InitCashMode.Auto
        np.testing.assert_array_equal(
            new_pf._init_position, np.concatenate((pf1._init_position, pf2._init_position))
        )
        np.testing.assert_array_equal(
            new_pf._init_price, np.concatenate((pf1._init_price, pf2._init_price))
        )
        np.testing.assert_array_equal(
            new_pf._cash_deposits,
            column_stack_arrs((pf1._cash_deposits, pf2._cash_deposits), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._cash_earnings,
            column_stack_arrs((pf1._cash_earnings, pf2._cash_earnings), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._call_seq,
            column_stack_arrs((pf1._call_seq, pf2._call_seq), pf1, pf2, fill_value=0),
        )
        np.testing.assert_array_equal(
            new_pf._bm_close,
            column_stack_arrs((pf1._bm_close, pf2._bm_close), pf1, pf2),
        )
        in_outputs1 = dict(
            arr_2d_cs=np.arange(10).reshape((5, 2)),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d_cs=np.arange(2),
            arr_1d_pcg=np.arange(2),
            arr_1d_pg=np.arange(2),
            arr_1d_pc=np.arange(3),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr_2d_cs=np.arange(10).reshape((5, 2)),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d_cs=np.arange(2),
            arr_1d_pcg=np.arange(2),
            arr_1d_pg=np.arange(2),
            arr_1d_pc=np.arange(3),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_cs,
            column_stack_arrs((in_outputs1.arr_2d_cs, in_outputs2.arr_2d_cs), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pcg,
            column_stack_arrs((in_outputs1.arr_2d_pcg, in_outputs2.arr_2d_pcg), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pg,
            column_stack_arrs((in_outputs1.arr_2d_pg, in_outputs2.arr_2d_pg), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_2d_pc,
            column_stack_arrs((in_outputs1.arr_2d_pc, in_outputs2.arr_2d_pc), pf1, pf2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_1d_cs,
            np.concatenate((in_outputs1.arr_1d_cs, in_outputs2.arr_1d_cs)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_1d_pcg,
            np.concatenate((in_outputs1.arr_1d_pcg, in_outputs2.arr_1d_pcg)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_1d_pg,
            np.concatenate((in_outputs1.arr_1d_pg, in_outputs2.arr_1d_pg)),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr_1d_pc,
            np.concatenate((in_outputs1.arr_1d_pc, in_outputs2.arr_1d_pc)),
        )
        in_outputs1 = dict(
            arr=np.arange(10).reshape((5, 2)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr=np.arange(10).reshape((5, 2)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        new_pf = vbt.Portfolio.column_stack(
            pf1.replace(in_outputs=in_outputs1),
            pf2.replace(in_outputs=in_outputs2),
        )
        np.testing.assert_array_equal(
            new_pf.in_outputs.arr, column_stack_arrs((in_outputs1.arr, in_outputs2.arr), pf1, pf2)
        )
        in_outputs1 = dict(
            arr=np.arange(15).reshape((5, 3)),
        )
        in_outputs1 = namedtuple("InOutputs", in_outputs1)(**in_outputs1)
        in_outputs2 = dict(
            arr=np.arange(15).reshape((5, 3)),
        )
        in_outputs2 = namedtuple("InOutputs", in_outputs2)(**in_outputs2)
        with pytest.raises(Exception):
            vbt.Portfolio.column_stack(
                pf1.replace(in_outputs=in_outputs1),
                pf2.replace(in_outputs=in_outputs2),
            )

        index1 = pd.date_range("2020", periods=100)
        index2 = index1[20:80]
        close1 = pd.DataFrame({"a": pd.Series(np.arange(len(index1)), index=index1)})
        close2 = pd.DataFrame(
            {
                "b": pd.Series(np.arange(20, 80), index=index2),
                "c": pd.Series(np.arange(20, 80), index=index2),
            }
        )
        sim_range_pf1 = vbt.Portfolio.from_orders(close1, size=1, sim_start=5, sim_end=-15)
        sim_range_pf2 = vbt.Portfolio.from_orders(close2, size=1, sim_start=10, sim_end=[20, -30])
        new_sim_range_pf = vbt.Portfolio.column_stack(sim_range_pf1, sim_range_pf2)
        np.testing.assert_array_equal(new_sim_range_pf._sim_start, np.array([5, 30, 30]))
        np.testing.assert_array_equal(new_sim_range_pf._sim_end, np.array([85, 40, 50]))

    def test_config(self, tmp_path):
        pf2 = pf.copy()
        pf2._metrics = pf2._metrics.copy()
        pf2.metrics["hello"] = "world"
        pf2._subplots = pf2.subplots.copy()
        pf2.subplots["hello"] = "world"
        assert vbt.Portfolio.loads(pf2["a"].dumps()) == pf2["a"]
        assert vbt.Portfolio.loads(pf2.dumps()) == pf2
        pf2.save(tmp_path / "pf")
        assert vbt.Portfolio.load(tmp_path / "pf") == pf2
        pf2.save(tmp_path / "pf", file_format="ini")
        assert vbt.Portfolio.load(tmp_path / "pf", file_format="ini") == pf2

    def test_wrapper(self):
        assert_index_equal(pf.wrapper.index, close_na.index)
        assert_index_equal(pf.wrapper.columns, close_na.columns)
        assert pf.wrapper.ndim == 2
        assert pf.wrapper.grouper.group_by is None
        assert pf.wrapper.grouper.allow_enable
        assert pf.wrapper.grouper.allow_disable
        assert pf.wrapper.grouper.allow_modify
        assert_index_equal(pf_grouped.wrapper.index, close_na.index)
        assert_index_equal(pf_grouped.wrapper.columns, close_na.columns)
        assert pf_grouped.wrapper.ndim == 2
        assert_index_equal(pf_grouped.wrapper.grouper.group_by, group_by)
        assert pf_grouped.wrapper.grouper.allow_enable
        assert pf_grouped.wrapper.grouper.allow_disable
        assert pf_grouped.wrapper.grouper.allow_modify
        assert_index_equal(pf_shared.wrapper.index, close_na.index)
        assert_index_equal(pf_shared.wrapper.columns, close_na.columns)
        assert pf_shared.wrapper.ndim == 2
        assert_index_equal(pf_shared.wrapper.grouper.group_by, group_by)
        assert not pf_shared.wrapper.grouper.allow_enable
        assert pf_shared.wrapper.grouper.allow_disable
        assert not pf_shared.wrapper.grouper.allow_modify

    def test_indexing(self):
        assert pf["a"].wrapper == pf.wrapper["a"]
        assert pf["a"].orders == pf.orders["a"]
        assert pf["a"].logs == pf.logs["a"]
        assert pf["a"].init_cash == pf.init_cash["a"]
        assert pf["a"].init_position == pf.init_position["a"]
        assert pf["a"].init_price == pf.init_price["a"]
        assert_series_equal(pf["a"].call_seq, pf.call_seq["a"])
        assert_series_equal(pf["a"].cash_deposits, pf.cash_deposits["a"])

        assert pf["c"].wrapper == pf.wrapper["c"]
        assert pf["c"].orders == pf.orders["c"]
        assert pf["c"].logs == pf.logs["c"]
        assert pf["c"].init_cash == pf.init_cash["c"]
        assert pf["c"].init_position == pf.init_position["c"]
        assert pf["c"].init_price == pf.init_price["c"]
        assert_series_equal(pf["c"].call_seq, pf.call_seq["c"])
        assert_series_equal(pf["c"].cash_deposits, pf.cash_deposits["c"])

        assert pf[["c"]].wrapper == pf.wrapper[["c"]]
        assert pf[["c"]].orders == pf.orders[["c"]]
        assert pf[["c"]].logs == pf.logs[["c"]]
        assert_series_equal(pf[["c"]].init_cash, pf.init_cash[["c"]])
        assert_series_equal(pf[["c"]].init_position, pf.init_position[["c"]])
        assert_series_equal(pf[["c"]].init_price, pf.init_price[["c"]])
        assert_frame_equal(pf[["c"]].call_seq, pf.call_seq[["c"]])
        assert_frame_equal(pf[["c"]].cash_deposits, pf.cash_deposits[["c"]])

        assert pf_grouped["first"].wrapper == pf_grouped.wrapper["first"]
        assert pf_grouped["first"].orders == pf_grouped.orders["first"]
        assert pf_grouped["first"].logs == pf_grouped.logs["first"]
        assert pf_grouped["first"].init_cash == pf_grouped.init_cash["first"]
        assert_series_equal(pf_grouped["first"].init_position, pf_grouped.init_position[["a", "b"]])
        assert_series_equal(pf_grouped["first"].init_price, pf_grouped.init_price[["a", "b"]])
        assert_frame_equal(pf_grouped["first"].call_seq, pf_grouped.call_seq[["a", "b"]])
        assert_series_equal(pf_grouped["first"].cash_deposits, pf_grouped.cash_deposits["first"])

        assert pf_grouped[["first"]].wrapper == pf_grouped.wrapper[["first"]]
        assert pf_grouped[["first"]].orders == pf_grouped.orders[["first"]]
        assert pf_grouped[["first"]].logs == pf_grouped.logs[["first"]]
        assert_series_equal(pf_grouped[["first"]].init_cash, pf_grouped.init_cash[["first"]])
        assert_series_equal(
            pf_grouped[["first"]].init_position, pf_grouped.init_position[["a", "b"]]
        )
        assert_series_equal(pf_grouped[["first"]].init_price, pf_grouped.init_price[["a", "b"]])
        assert_frame_equal(pf_grouped[["first"]].call_seq, pf_grouped.call_seq[["a", "b"]])
        assert_frame_equal(pf_grouped[["first"]].cash_deposits, pf_grouped.cash_deposits[["first"]])

        assert pf_grouped["second"].wrapper == pf_grouped.wrapper["second"]
        assert pf_grouped["second"].orders == pf_grouped.orders["second"]
        assert pf_grouped["second"].logs == pf_grouped.logs["second"]
        assert pf_grouped["second"].init_cash == pf_grouped.init_cash["second"]
        assert pf_grouped["second"].init_position == pf_grouped.init_position["c"]
        assert pf_grouped["second"].init_price == pf_grouped.init_price["c"]
        assert_series_equal(pf_grouped["second"].call_seq, pf_grouped.call_seq["c"])
        assert_series_equal(pf_grouped["second"].cash_deposits, pf_grouped.cash_deposits["second"])

        assert pf_grouped[["second"]].orders == pf_grouped.orders[["second"]]
        assert pf_grouped[["second"]].wrapper == pf_grouped.wrapper[["second"]]
        assert pf_grouped[["second"]].orders == pf_grouped.orders[["second"]]
        assert pf_grouped[["second"]].logs == pf_grouped.logs[["second"]]
        assert_series_equal(pf_grouped[["second"]].init_cash, pf_grouped.init_cash[["second"]])
        assert_series_equal(pf_grouped[["second"]].init_position, pf_grouped.init_position[["c"]])
        assert_series_equal(pf_grouped[["second"]].init_price, pf_grouped.init_price[["c"]])
        assert_frame_equal(pf_grouped[["second"]].call_seq, pf_grouped.call_seq[["c"]])
        assert_frame_equal(
            pf_grouped[["second"]].cash_deposits, pf_grouped.cash_deposits[["second"]]
        )

        assert pf_shared["first"].wrapper == pf_shared.wrapper["first"]
        assert pf_shared["first"].orders == pf_shared.orders["first"]
        assert pf_shared["first"].logs == pf_shared.logs["first"]
        assert pf_shared["first"].init_cash == pf_shared.init_cash["first"]
        assert_series_equal(pf_shared["first"].init_position, pf_shared.init_position[["a", "b"]])
        assert_series_equal(pf_shared["first"].init_price, pf_shared.init_price[["a", "b"]])
        assert_frame_equal(pf_shared["first"].call_seq, pf_shared.call_seq[["a", "b"]])
        assert_series_equal(pf_shared["first"].cash_deposits, pf_shared.cash_deposits["first"])

        assert pf_shared[["first"]].orders == pf_shared.orders[["first"]]
        assert pf_shared[["first"]].wrapper == pf_shared.wrapper[["first"]]
        assert pf_shared[["first"]].orders == pf_shared.orders[["first"]]
        assert pf_shared[["first"]].logs == pf_shared.logs[["first"]]
        assert_series_equal(pf_shared[["first"]].init_cash, pf_shared.init_cash[["first"]])
        assert_series_equal(pf_shared[["first"]].init_position, pf_shared.init_position[["a", "b"]])
        assert_series_equal(pf_shared[["first"]].init_price, pf_shared.init_price[["a", "b"]])
        assert_frame_equal(pf_shared[["first"]].call_seq, pf_shared.call_seq[["a", "b"]])
        assert_frame_equal(pf_shared[["first"]].cash_deposits, pf_shared.cash_deposits[["first"]])

        assert pf_shared["second"].wrapper == pf_shared.wrapper["second"]
        assert pf_shared["second"].orders == pf_shared.orders["second"]
        assert pf_shared["second"].logs == pf_shared.logs["second"]
        assert pf_shared["second"].init_cash == pf_shared.init_cash["second"]
        assert pf_shared["second"].init_position == pf_shared.init_position["c"]
        assert pf_shared["second"].init_price == pf_shared.init_price["c"]
        assert_series_equal(pf_shared["second"].call_seq, pf_shared.call_seq["c"])
        assert_series_equal(pf_shared["second"].cash_deposits, pf_shared.cash_deposits["second"])

        assert pf_shared[["second"]].wrapper == pf_shared.wrapper[["second"]]
        assert pf_shared[["second"]].orders == pf_shared.orders[["second"]]
        assert pf_shared[["second"]].logs == pf_shared.logs[["second"]]
        assert_series_equal(pf_shared[["second"]].init_cash, pf_shared.init_cash[["second"]])
        assert_series_equal(pf_shared[["second"]].init_position, pf_shared.init_position[["c"]])
        assert_series_equal(pf_shared[["second"]].init_price, pf_shared.init_price[["c"]])
        assert_frame_equal(pf_shared[["second"]].call_seq, pf_shared.call_seq[["c"]])
        assert_frame_equal(pf_shared[["second"]].cash_deposits, pf_shared.cash_deposits[["second"]])

        assert pf.loc[:"2020-01-03"].wrapper == pf.wrapper.loc[:"2020-01-03"]
        assert pf.loc[:"2020-01-03"].orders == pf.orders.loc[:"2020-01-03"]
        assert pf.loc[:"2020-01-03"].logs == pf.logs.loc[:"2020-01-03"]
        assert_series_equal(pf.loc[:"2020-01-03"].init_cash, pf.init_cash)
        assert_series_equal(pf.loc[:"2020-01-03"].init_position, pf.init_position)
        assert_series_equal(pf.loc[:"2020-01-03"].init_price, pf.init_price)
        assert_frame_equal(pf.loc[:"2020-01-03"].call_seq, pf.call_seq.loc[:"2020-01-03"])
        assert_frame_equal(pf.loc[:"2020-01-03"].cash_deposits, pf.cash_deposits.loc[:"2020-01-03"])
        assert pf.loc["2020-01-03":].wrapper == pf.wrapper.loc["2020-01-03":]
        assert pf.loc["2020-01-03":].orders == pf.orders.loc["2020-01-03":]
        assert pf.loc["2020-01-03":].logs == pf.logs.loc["2020-01-03":]
        assert_series_equal(
            pf.loc["2020-01-03":].init_cash,
            pf.cash.loc["2020-01-02"].rename("init_cash"),
        )
        assert_series_equal(
            pf.loc["2020-01-03":].init_position,
            pf.assets.loc["2020-01-02"].rename("init_position"),
        )
        assert_series_equal(
            pf.loc["2020-01-03":].init_price,
            pf.filled_close.loc["2020-01-02"].rename("init_price"),
        )
        assert_frame_equal(pf.loc["2020-01-03":].call_seq, pf.call_seq.loc["2020-01-03":])
        assert_frame_equal(pf.loc["2020-01-03":].cash_deposits, pf.cash_deposits.loc["2020-01-03":])

        assert pf_grouped.loc[:"2020-01-03"].wrapper == pf_grouped.wrapper.loc[:"2020-01-03"]
        assert pf_grouped.loc[:"2020-01-03"].orders == pf_grouped.orders.loc[:"2020-01-03"]
        assert pf_grouped.loc[:"2020-01-03"].logs == pf_grouped.logs.loc[:"2020-01-03"]
        assert_series_equal(pf_grouped.loc[:"2020-01-03"].init_cash, pf_grouped.init_cash)
        assert_series_equal(pf_grouped.loc[:"2020-01-03"].init_position, pf_grouped.init_position)
        assert_series_equal(pf_grouped.loc[:"2020-01-03"].init_price, pf_grouped.init_price)
        assert_frame_equal(
            pf_grouped.loc[:"2020-01-03"].call_seq, pf_grouped.call_seq.loc[:"2020-01-03"]
        )
        assert_frame_equal(
            pf_grouped.loc[:"2020-01-03"].cash_deposits, pf_grouped.cash_deposits.loc[:"2020-01-03"]
        )
        assert pf_grouped.loc["2020-01-03":].wrapper == pf_grouped.wrapper.loc["2020-01-03":]
        assert pf_grouped.loc["2020-01-03":].orders == pf_grouped.orders.loc["2020-01-03":]
        assert pf_grouped.loc["2020-01-03":].logs == pf_grouped.logs.loc["2020-01-03":]
        assert_series_equal(
            pf_grouped.loc["2020-01-03":].init_cash,
            pf_grouped.cash.loc["2020-01-02"].rename("init_cash"),
        )
        assert_series_equal(
            pf_grouped.loc["2020-01-03":].init_position,
            pf_grouped.assets.loc["2020-01-02"].rename("init_position"),
        )
        assert_series_equal(
            pf_grouped.loc["2020-01-03":].init_price,
            pf_grouped.filled_close.loc["2020-01-02"].rename("init_price"),
        )
        assert_frame_equal(
            pf_grouped.loc["2020-01-03":].call_seq, pf_grouped.call_seq.loc["2020-01-03":]
        )
        assert_frame_equal(
            pf_grouped.loc["2020-01-03":].cash_deposits, pf_grouped.cash_deposits.loc["2020-01-03":]
        )

        assert pf_shared.loc[:"2020-01-03"].wrapper == pf_shared.wrapper.loc[:"2020-01-03"]
        assert pf_shared.loc[:"2020-01-03"].orders == pf_shared.orders.loc[:"2020-01-03"]
        assert pf_shared.loc[:"2020-01-03"].logs == pf_shared.logs.loc[:"2020-01-03"]
        assert_series_equal(pf_shared.loc[:"2020-01-03"].init_cash, pf_shared.init_cash)
        assert_series_equal(pf_shared.loc[:"2020-01-03"].init_position, pf_shared.init_position)
        assert_series_equal(pf_shared.loc[:"2020-01-03"].init_price, pf_shared.init_price)
        assert_frame_equal(
            pf_shared.loc[:"2020-01-03"].call_seq, pf_shared.call_seq.loc[:"2020-01-03"]
        )
        assert_frame_equal(
            pf_shared.loc[:"2020-01-03"].cash_deposits, pf_shared.cash_deposits.loc[:"2020-01-03"]
        )
        assert pf_shared.loc["2020-01-03":].wrapper == pf_shared.wrapper.loc["2020-01-03":]
        assert pf_shared.loc["2020-01-03":].orders == pf_shared.orders.loc["2020-01-03":]
        assert pf_shared.loc["2020-01-03":].logs == pf_shared.logs.loc["2020-01-03":]
        assert_series_equal(
            pf_shared.loc["2020-01-03":].init_cash,
            pf_shared.cash.loc["2020-01-02"].rename("init_cash"),
        )
        assert_series_equal(
            pf_shared.loc["2020-01-03":].init_position,
            pf_shared.assets.loc["2020-01-02"].rename("init_position"),
        )
        assert_series_equal(
            pf_shared.loc["2020-01-03":].init_price,
            pf_shared.filled_close.loc["2020-01-02"].rename("init_price"),
        )
        assert_frame_equal(
            pf_shared.loc["2020-01-03":].call_seq, pf_shared.call_seq.loc["2020-01-03":]
        )
        assert_frame_equal(
            pf_shared.loc["2020-01-03":].cash_deposits, pf_shared.cash_deposits.loc["2020-01-03":]
        )

        def _apply_loc(x):
            return x.loc["2020-01-02":"2020-01-04", ["c"]]

        def _apply_prev_loc(x):
            return x.loc["2020-01-01", ["c"]]

        assert _apply_loc(pf).wrapper == _apply_loc(pf.wrapper)
        assert _apply_loc(pf).orders == _apply_loc(pf.orders)
        assert _apply_loc(pf).logs == _apply_loc(pf.logs)
        assert_series_equal(
            _apply_loc(pf).init_cash,
            _apply_prev_loc(pf.cash).rename("init_cash"),
        )
        assert_series_equal(
            _apply_loc(pf).init_position,
            _apply_prev_loc(pf.assets).rename("init_position"),
        )
        assert_series_equal(
            _apply_loc(pf).init_price,
            _apply_prev_loc(pf.filled_close).rename("init_price"),
        )
        assert_frame_equal(_apply_loc(pf).call_seq, _apply_loc(pf.call_seq))
        assert_frame_equal(_apply_loc(pf).cash_deposits, _apply_loc(pf.cash_deposits))

        sim_range_pf = vbt.Portfolio.from_orders(
            pd.DataFrame(
                {
                    "a": pd.Series(np.arange(1, 101), index=pd.date_range("2020", periods=100)),
                    "b": pd.Series(np.arange(1, 101), index=pd.date_range("2020", periods=100)),
                }
            ),
            size=1,
            sim_start=[10, 20],
            sim_end=[-10, -20],
        )
        np.testing.assert_array_equal(sim_range_pf.iloc[5:95]._sim_start, np.array([5, 15]))
        np.testing.assert_array_equal(sim_range_pf.iloc[5:95]._sim_end, np.array([85, 75]))
        np.testing.assert_array_equal(sim_range_pf.iloc[10:90]._sim_start, np.array([0, 10]))
        np.testing.assert_array_equal(sim_range_pf.iloc[10:90]._sim_end, np.array([80, 70]))
        np.testing.assert_array_equal(sim_range_pf.iloc[15:85]._sim_start, np.array([0, 5]))
        np.testing.assert_array_equal(sim_range_pf.iloc[15:85]._sim_end, np.array([70, 65]))
        assert sim_range_pf.iloc[20:80]._sim_start is None
        assert sim_range_pf.iloc[20:80]._sim_end is None

    def test_regroup(self):
        assert pf.regroup(None) == pf
        assert pf.regroup(False) == pf
        assert pf.regroup(group_by) != pf
        assert_index_equal(pf.regroup(group_by).wrapper.grouper.group_by, group_by)
        assert pf_grouped.regroup(None) == pf_grouped
        assert pf_grouped.regroup(False) != pf_grouped
        assert pf_grouped.regroup(False).wrapper.grouper.group_by is None
        assert pf_grouped.regroup(group_by) == pf_grouped
        assert pf_shared.regroup(None) == pf_shared
        with pytest.raises(Exception):
            pf_shared.regroup(False)
        assert pf_shared.regroup(group_by) == pf_shared

    def test_cash_sharing(self):
        assert not pf.cash_sharing
        assert not pf_grouped.cash_sharing
        assert pf_shared.cash_sharing

    def test_call_seq(self):
        assert_frame_equal(
            pf.call_seq,
            pd.DataFrame(
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_grouped.call_seq,
            pd.DataFrame(
                np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_shared.call_seq,
            pd.DataFrame(
                np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )

    def test_in_outputs(self):
        def _apply_loc(x):
            return x.loc["2020-01-02":"2020-01-04"]

        in_outputs = dict(
            init_cash=pf.init_cash.values,
            init_position_value=pf.init_position_value.values,
            open=pf.open.values,
            high=pf.high.values,
            low=pf.low.values,
            close=pf.close.values,
            cash_flow=pf.cash_flow.values,
            cash=pf.cash.values,
            bm_returns=pf.bm_returns.values,
            orders=pf.orders.values,
            logs=pf.logs.values,
            entry_trades=pf.entry_trades.values,
            exit_trades=pf.exit_trades.values,
            positions=pf.positions.values,
            trades=pf.trades.values,
            drawdowns=pf.drawdowns.values,
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf2 = vbt.Portfolio.from_holding(close_na).replace(in_outputs=in_outputs)

        np.testing.assert_array_equal(pf2.init_cash.values, pf.init_cash.values)
        np.testing.assert_array_equal(pf2.init_position_value.values, pf.init_position_value.values)
        np.testing.assert_array_equal(pf2.open.values, pf.open.values)
        np.testing.assert_array_equal(pf2.high.values, pf.high.values)
        np.testing.assert_array_equal(pf2.low.values, pf.low.values)
        np.testing.assert_array_equal(pf2.close.values, pf.close.values)
        np.testing.assert_array_equal(pf2.cash_flow.values, pf.cash_flow.values)
        np.testing.assert_array_equal(pf2.cash.values, pf.cash.values)
        np.testing.assert_array_equal(pf2.bm_returns.values, pf.bm_returns.values)
        assert_records_close(pf2.orders.values, pf.orders.values)
        assert_records_close(pf2.logs.values, pf.logs.values)
        assert_records_close(pf2.entry_trades.values, pf.entry_trades.values)
        assert_records_close(pf2.exit_trades.values, pf.exit_trades.values)
        assert_records_close(pf2.positions.values, pf.positions.values)
        assert_records_close(pf2.trades.values, pf.trades.values)
        assert_records_close(pf2.drawdowns.values, pf.drawdowns.values)

        assert pf2["b"].init_cash == pf.init_cash["b"]
        assert pf2["b"].init_position_value == pf.init_position_value["b"]
        np.testing.assert_array_equal(pf2["b"].open.values, pf.open["b"].values)
        np.testing.assert_array_equal(pf2["b"].high.values, pf.high["b"].values)
        np.testing.assert_array_equal(pf2["b"].low.values, pf.low["b"].values)
        np.testing.assert_array_equal(pf2["b"].close.values, pf.close["b"].values)
        np.testing.assert_array_equal(pf2["b"].cash_flow.values, pf.cash_flow["b"].values)
        np.testing.assert_array_equal(pf2["b"].cash.values, pf.cash["b"].values)
        np.testing.assert_array_equal(pf2["b"].bm_returns.values, pf.bm_returns["b"].values)
        assert_records_close(pf2["b"].orders.values, pf.orders["b"].values)
        assert_records_close(pf2["b"].logs.values, pf.logs["b"].values)
        assert_records_close(pf2["b"].entry_trades.values, pf.entry_trades["b"].values)
        assert_records_close(pf2["b"].exit_trades.values, pf.exit_trades["b"].values)
        assert_records_close(pf2["b"].positions.values, pf.positions["b"].values)
        assert_records_close(pf2["b"].trades.values, pf.trades["b"].values)
        assert_records_close(pf2["b"].drawdowns.values, pf.drawdowns["b"].values)

        np.testing.assert_array_equal(_apply_loc(pf2).init_cash.values, pf.init_cash.values)
        np.testing.assert_array_equal(
            _apply_loc(pf2).init_position_value.values, pf.init_position_value.values
        )
        np.testing.assert_array_equal(_apply_loc(pf2).open.values, _apply_loc(pf.open).values)
        np.testing.assert_array_equal(_apply_loc(pf2).high.values, _apply_loc(pf.high).values)
        np.testing.assert_array_equal(_apply_loc(pf2).low.values, _apply_loc(pf.low).values)
        np.testing.assert_array_equal(_apply_loc(pf2).close.values, _apply_loc(pf.close).values)
        np.testing.assert_array_equal(
            _apply_loc(pf2).cash_flow.values, _apply_loc(pf.cash_flow).values
        )
        np.testing.assert_array_equal(_apply_loc(pf2).cash.values, _apply_loc(pf.cash).values)
        np.testing.assert_array_equal(
            _apply_loc(pf2).bm_returns.values, _apply_loc(pf.bm_returns).values
        )
        assert_records_close(_apply_loc(pf2).orders.values, _apply_loc(pf.orders).values)
        assert_records_close(_apply_loc(pf2).logs.values, _apply_loc(pf.logs).values)
        assert_records_close(
            _apply_loc(pf2).entry_trades.values, _apply_loc(pf.entry_trades).values
        )
        assert_records_close(_apply_loc(pf2).exit_trades.values, _apply_loc(pf.exit_trades).values)
        assert_records_close(_apply_loc(pf2).positions.values, _apply_loc(pf.positions).values)
        assert_records_close(_apply_loc(pf2).trades.values, _apply_loc(pf.trades).values)
        assert_records_close(_apply_loc(pf2).drawdowns.values, _apply_loc(pf.drawdowns).values)

        pf2 = pf2.replace(call_seq=None)
        np.testing.assert_allclose(
            pf2.resample("2d").init_cash.values,
            pf.init_cash.values,
        )
        np.testing.assert_allclose(
            pf2.resample("2d").init_position_value.values,
            pf.init_position_value.values,
        )
        np.testing.assert_allclose(
            pf2.resample("2d").open.values,
            pf.open.resample("2d").first().values,
        )
        np.testing.assert_allclose(
            pf2.resample("2d").high.values,
            pf.high.resample("2d").max().values,
        )
        np.testing.assert_allclose(
            pf2.resample("2d").low.values,
            pf.low.resample("2d").min().values,
        )
        np.testing.assert_allclose(
            pf2.resample("2d").close.values,
            pf.close.resample("2d").last().values,
        )
        np.testing.assert_allclose(
            pf2.resample("2d").cash_flow.values,
            pf.cash_flow.resample("2d").sum().fillna(0.0).values,
        )
        np.testing.assert_allclose(
            pf2.resample("2d").cash.values,
            pf.cash.resample("2d").last().values,
        )
        np.testing.assert_allclose(
            pf2.resample("2d").bm_returns.values,
            pf.bm_returns.resample("2d").apply(lambda x: (1 + x).prod() - 1).values,
        )
        assert_records_close(
            pf2.resample("2d").orders.values,
            pf.orders.resample("2d").values,
        )
        assert_records_close(
            pf2.resample("2d").logs.values,
            pf.logs.resample("2d").values,
        )
        assert_records_close(
            pf2.resample("2d").entry_trades.values,
            pf.entry_trades.resample("2d").values,
        )
        assert_records_close(
            pf2.resample("2d").exit_trades.values,
            pf.exit_trades.resample("2d").values,
        )
        assert_records_close(
            pf2.resample("2d").positions.values,
            pf.positions.resample("2d").values,
        )
        assert_records_close(
            pf2.resample("2d").trades.values,
            pf.trades.resample("2d").values,
        )
        assert_records_close(
            pf2.resample("2d").drawdowns.values,
            pf.drawdowns.resample("2d").values,
        )

        in_outputs = dict(
            init_cash=pf_grouped.init_cash.values,
            init_position_value=pf_grouped.init_position_value.values,
            open=pf_grouped.open.values,
            high=pf_grouped.high.values,
            low=pf_grouped.low.values,
            close=pf_grouped.close.values,
            cash_flow=pf_grouped.cash_flow.values,
            cash_pg=pf_grouped.cash.values,
            bm_returns=pf_grouped.bm_returns.values,
            orders=pf_grouped.orders.values,
            logs=pf_grouped.logs.values,
            entry_trades=pf_grouped.entry_trades.values,
            exit_trades=pf_grouped.exit_trades.values,
            positions=pf_grouped.positions.values,
            trades=pf_grouped.trades.values,
            drawdowns=pf_grouped.drawdowns.values,
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf_grouped2 = vbt.Portfolio.from_holding(close_na, group_by=group_by).replace(
            in_outputs=in_outputs
        )

        np.testing.assert_array_equal(pf_grouped2.init_cash.values, pf_grouped.init_cash.values)
        np.testing.assert_array_equal(
            pf_grouped2.init_position_value.values, pf_grouped.init_position_value.values
        )
        np.testing.assert_array_equal(pf_grouped2.open.values, pf_grouped.open.values)
        np.testing.assert_array_equal(pf_grouped2.high.values, pf_grouped.high.values)
        np.testing.assert_array_equal(pf_grouped2.low.values, pf_grouped.low.values)
        np.testing.assert_array_equal(pf_grouped2.close.values, pf_grouped.close.values)
        np.testing.assert_array_equal(pf_grouped2.cash_flow.values, pf_grouped.cash_flow.values)
        np.testing.assert_array_equal(pf_grouped2.cash.values, pf_grouped.cash.values)
        np.testing.assert_array_equal(pf_grouped2.bm_returns.values, pf_grouped.bm_returns.values)
        assert_records_close(pf_grouped2.orders.values, pf_grouped.orders.values)
        assert_records_close(pf_grouped2.logs.values, pf_grouped.logs.values)
        assert_records_close(pf_grouped2.entry_trades.values, pf_grouped.entry_trades.values)
        assert_records_close(pf_grouped2.exit_trades.values, pf_grouped.exit_trades.values)
        assert_records_close(pf_grouped2.positions.values, pf_grouped.positions.values)
        assert_records_close(pf_grouped2.trades.values, pf_grouped.trades.values)
        assert_records_close(pf_grouped2.drawdowns.values, pf_grouped.drawdowns.values)

        assert pf_grouped2["second"].init_cash == pf_grouped.init_cash["second"]
        assert pf_grouped2["second"].init_position_value == pf_grouped.init_position_value["second"]
        np.testing.assert_array_equal(
            pf_grouped2["second"].open.values, pf_grouped.open["c"].values
        )
        np.testing.assert_array_equal(
            pf_grouped2["second"].high.values, pf_grouped.high["c"].values
        )
        np.testing.assert_array_equal(pf_grouped2["second"].low.values, pf_grouped.low["c"].values)
        np.testing.assert_array_equal(
            pf_grouped2["second"].close.values, pf_grouped.close["c"].values
        )
        np.testing.assert_array_equal(
            pf_grouped2["second"].cash_flow.values, pf_grouped.cash_flow["second"].values
        )
        np.testing.assert_array_equal(
            pf_grouped2["second"].cash.values, pf_grouped.cash["second"].values
        )
        np.testing.assert_array_equal(
            pf_grouped2["second"].bm_returns.values, pf_grouped.bm_returns["second"].values
        )
        assert_records_close(
            pf_grouped2["second"].orders.values, pf_grouped.orders["second"].values
        )
        assert_records_close(pf_grouped2["second"].logs.values, pf_grouped.logs["second"].values)
        assert_records_close(
            pf_grouped2["second"].entry_trades.values, pf_grouped.entry_trades["second"].values
        )
        assert_records_close(
            pf_grouped2["second"].exit_trades.values, pf_grouped.exit_trades["second"].values
        )
        assert_records_close(
            pf_grouped2["second"].positions.values, pf_grouped.positions["second"].values
        )
        assert_records_close(
            pf_grouped2["second"].trades.values, pf_grouped.trades["second"].values
        )
        assert_records_close(
            pf_grouped2["second"].drawdowns.values, pf_grouped.drawdowns["second"].values
        )

        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).init_cash.values, pf_grouped.init_cash.values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).init_position_value.values,
            pf_grouped.init_position_value.values,
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).open.values, _apply_loc(pf_grouped.open).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).high.values, _apply_loc(pf_grouped.high).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).low.values, _apply_loc(pf_grouped.low).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).close.values, _apply_loc(pf_grouped.close).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).cash_flow.values, _apply_loc(pf_grouped.cash_flow).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).cash.values, _apply_loc(pf_grouped.cash).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_grouped2).bm_returns.values, _apply_loc(pf_grouped.bm_returns).values
        )
        assert_records_close(
            _apply_loc(pf_grouped2).orders.values, _apply_loc(pf_grouped.orders).values
        )
        assert_records_close(
            _apply_loc(pf_grouped2).logs.values, _apply_loc(pf_grouped.logs).values
        )
        assert_records_close(
            _apply_loc(pf_grouped2).entry_trades.values, _apply_loc(pf_grouped.entry_trades).values
        )
        assert_records_close(
            _apply_loc(pf_grouped2).exit_trades.values, _apply_loc(pf_grouped.exit_trades).values
        )
        assert_records_close(
            _apply_loc(pf_grouped2).positions.values, _apply_loc(pf_grouped.positions).values
        )
        assert_records_close(
            _apply_loc(pf_grouped2).trades.values, _apply_loc(pf_grouped.trades).values
        )
        assert_records_close(
            _apply_loc(pf_grouped2).drawdowns.values, _apply_loc(pf_grouped.drawdowns).values
        )

        pf_grouped2 = pf_grouped2.replace(call_seq=None)
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").init_cash.values,
            pf_grouped.init_cash.values,
        )
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").init_position_value.values,
            pf_grouped.init_position_value.values,
        )
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").open.values,
            pf_grouped.open.resample("2d").first().values,
        )
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").high.values,
            pf_grouped.high.resample("2d").max().values,
        )
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").low.values,
            pf_grouped.low.resample("2d").min().values,
        )
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").close.values,
            pf_grouped.close.resample("2d").last().values,
        )
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").cash_flow.values,
            pf_grouped.cash_flow.resample("2d").sum().fillna(0.0).values,
        )
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").cash.values,
            pf_grouped.cash.resample("2d").last().values,
        )
        np.testing.assert_allclose(
            pf_grouped2.resample("2d").bm_returns.values,
            pf_grouped.bm_returns.resample("2d").apply(lambda x: (1 + x).prod() - 1).values,
        )
        assert_records_close(
            pf_grouped2.resample("2d").orders.values,
            pf_grouped.orders.resample("2d").values,
        )
        assert_records_close(
            pf_grouped2.resample("2d").logs.values,
            pf_grouped.logs.resample("2d").values,
        )
        assert_records_close(
            pf_grouped2.resample("2d").entry_trades.values,
            pf_grouped.entry_trades.resample("2d").values,
        )
        assert_records_close(
            pf_grouped2.resample("2d").exit_trades.values,
            pf_grouped.exit_trades.resample("2d").values,
        )
        assert_records_close(
            pf_grouped2.resample("2d").positions.values,
            pf_grouped.positions.resample("2d").values,
        )
        assert_records_close(
            pf_grouped2.resample("2d").trades.values,
            pf_grouped.trades.resample("2d").values,
        )
        assert_records_close(
            pf_grouped2.resample("2d").drawdowns.values,
            pf_grouped.drawdowns.resample("2d").values,
        )

        in_outputs = dict(
            init_cash=pf_shared.init_cash.values,
            init_position_value=pf_shared.init_position_value.values,
            open=pf_shared.open.values,
            high=pf_shared.high.values,
            low=pf_shared.low.values,
            close=pf_shared.close.values,
            cash_flow=pf_shared.cash_flow.values,
            cash=pf_shared.cash.values,
            bm_returns=pf_shared.bm_returns.values,
            orders=pf_shared.orders.values,
            logs=pf_shared.logs.values,
            entry_trades=pf_shared.entry_trades.values,
            exit_trades=pf_shared.exit_trades.values,
            positions=pf_shared.positions.values,
            trades=pf_shared.trades.values,
            drawdowns=pf_shared.drawdowns.values,
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf_shared2 = vbt.Portfolio.from_holding(
            close_na,
            group_by=group_by,
            cash_sharing=True,
        ).replace(in_outputs=in_outputs)

        np.testing.assert_array_equal(pf_shared2.init_cash.values, pf_shared.init_cash.values)
        np.testing.assert_array_equal(
            pf_shared2.init_position_value.values, pf_shared.init_position_value.values
        )
        np.testing.assert_array_equal(pf_shared2.open.values, pf_shared.open.values)
        np.testing.assert_array_equal(pf_shared2.high.values, pf_shared.high.values)
        np.testing.assert_array_equal(pf_shared2.low.values, pf_shared.low.values)
        np.testing.assert_array_equal(pf_shared2.close.values, pf_shared.close.values)
        np.testing.assert_array_equal(pf_shared2.cash_flow.values, pf_shared.cash_flow.values)
        np.testing.assert_array_equal(pf_shared2.cash.values, pf_shared.cash.values)
        np.testing.assert_array_equal(pf_shared2.bm_returns.values, pf_shared.bm_returns.values)
        assert_records_close(pf_shared2.orders.values, pf_shared.orders.values)
        assert_records_close(pf_shared2.logs.values, pf_shared.logs.values)
        assert_records_close(pf_shared2.entry_trades.values, pf_shared.entry_trades.values)
        assert_records_close(pf_shared2.exit_trades.values, pf_shared.exit_trades.values)
        assert_records_close(pf_shared2.positions.values, pf_shared.positions.values)
        assert_records_close(pf_shared2.trades.values, pf_shared.trades.values)
        assert_records_close(pf_shared2.drawdowns.values, pf_shared.drawdowns.values)

        assert pf_shared2["second"].init_cash == pf_shared.init_cash["second"]
        assert pf_shared2["second"].init_position_value == pf_shared.init_position_value["second"]
        np.testing.assert_array_equal(pf_shared2["second"].open.values, pf_shared.open["c"].values)
        np.testing.assert_array_equal(pf_shared2["second"].high.values, pf_shared.high["c"].values)
        np.testing.assert_array_equal(pf_shared2["second"].low.values, pf_shared.low["c"].values)
        np.testing.assert_array_equal(
            pf_shared2["second"].close.values, pf_shared.close["c"].values
        )
        np.testing.assert_array_equal(
            pf_shared2["second"].cash_flow.values, pf_shared.cash_flow["second"].values
        )
        np.testing.assert_array_equal(
            pf_shared2["second"].cash.values, pf_shared.cash["second"].values
        )
        np.testing.assert_array_equal(
            pf_shared2["second"].bm_returns.values, pf_shared.bm_returns["second"].values
        )
        assert_records_close(pf_shared2["second"].orders.values, pf_shared.orders["second"].values)
        assert_records_close(pf_shared2["second"].logs.values, pf_shared.logs["second"].values)
        assert_records_close(
            pf_shared2["second"].entry_trades.values, pf_shared.entry_trades["second"].values
        )
        assert_records_close(
            pf_shared2["second"].exit_trades.values, pf_shared.exit_trades["second"].values
        )
        assert_records_close(
            pf_shared2["second"].positions.values, pf_shared.positions["second"].values
        )
        assert_records_close(pf_shared2["second"].trades.values, pf_shared.trades["second"].values)
        assert_records_close(
            pf_shared2["second"].drawdowns.values, pf_shared.drawdowns["second"].values
        )

        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).init_cash.values, pf_shared.init_cash.values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).init_position_value.values, pf_shared.init_position_value.values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).open.values, _apply_loc(pf_shared.open).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).high.values, _apply_loc(pf_shared.high).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).low.values, _apply_loc(pf_shared.low).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).close.values, _apply_loc(pf_shared.close).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).cash_flow.values, _apply_loc(pf_shared.cash_flow).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).cash.values, _apply_loc(pf_shared.cash).values
        )
        np.testing.assert_array_equal(
            _apply_loc(pf_shared2).bm_returns.values, _apply_loc(pf_shared.bm_returns).values
        )
        assert_records_close(
            _apply_loc(pf_shared2).orders.values, _apply_loc(pf_shared.orders).values
        )
        assert_records_close(_apply_loc(pf_shared2).logs.values, _apply_loc(pf_shared.logs).values)
        assert_records_close(
            _apply_loc(pf_shared2).entry_trades.values, _apply_loc(pf_shared.entry_trades).values
        )
        assert_records_close(
            _apply_loc(pf_shared2).exit_trades.values, _apply_loc(pf_shared.exit_trades).values
        )
        assert_records_close(
            _apply_loc(pf_shared2).positions.values, _apply_loc(pf_shared.positions).values
        )
        assert_records_close(
            _apply_loc(pf_shared2).trades.values, _apply_loc(pf_shared.trades).values
        )
        assert_records_close(
            _apply_loc(pf_shared2).drawdowns.values, _apply_loc(pf_shared.drawdowns).values
        )

        pf_shared2 = pf_shared2.replace(call_seq=None)
        np.testing.assert_allclose(
            pf_shared2.resample("2d").init_cash.values,
            pf_shared.init_cash.values,
        )
        np.testing.assert_allclose(
            pf_shared2.resample("2d").init_position_value.values,
            pf_shared.init_position_value.values,
        )
        np.testing.assert_allclose(
            pf_shared2.resample("2d").open.values,
            pf_shared.open.resample("2d").first().values,
        )
        np.testing.assert_allclose(
            pf_shared2.resample("2d").high.values,
            pf_shared.high.resample("2d").max().values,
        )
        np.testing.assert_allclose(
            pf_shared2.resample("2d").low.values,
            pf_shared.low.resample("2d").min().values,
        )
        np.testing.assert_allclose(
            pf_shared2.resample("2d").close.values,
            pf_shared.close.resample("2d").last().values,
        )
        np.testing.assert_allclose(
            pf_shared2.resample("2d").cash_flow.values,
            pf_shared.cash_flow.resample("2d").sum().fillna(0.0).values,
        )
        np.testing.assert_allclose(
            pf_shared2.resample("2d").cash.values,
            pf_shared.cash.resample("2d").last().values,
        )
        np.testing.assert_allclose(
            pf_shared2.resample("2d").bm_returns.values,
            pf_shared.bm_returns.resample("2d").apply(lambda x: (1 + x).prod() - 1).values,
        )
        assert_records_close(
            pf_shared2.resample("2d").orders.values,
            pf_shared.orders.resample("2d").values,
        )
        assert_records_close(
            pf_shared2.resample("2d").logs.values,
            pf_shared.logs.resample("2d").values,
        )
        assert_records_close(
            pf_shared2.resample("2d").entry_trades.values,
            pf_shared.entry_trades.resample("2d").values,
        )
        assert_records_close(
            pf_shared2.resample("2d").exit_trades.values,
            pf_shared.exit_trades.resample("2d").values,
        )
        assert_records_close(
            pf_shared2.resample("2d").positions.values,
            pf_shared.positions.resample("2d").values,
        )
        assert_records_close(
            pf_shared2.resample("2d").trades.values,
            pf_shared.trades.resample("2d").values,
        )
        assert_records_close(
            pf_shared2.resample("2d").drawdowns.values,
            pf_shared.drawdowns.resample("2d").values,
        )

    def test_custom_in_outputs(self):
        in_outputs = dict(
            arr_1d_cs=np.arange(3),
            arr_2d_cs=np.arange(15).reshape((5, 3)),
            arr_1d_pcg=np.arange(3),
            arr_2d_pcg=np.arange(15).reshape((5, 3)),
            arr_1d_pg=np.arange(3),
            arr_2d_pg=np.arange(15).reshape((5, 3)),
            arr_1d_pc=np.arange(3),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d=np.arange(3),
            arr_2d=np.arange(15).reshape((5, 3)),
            arr_records=np.concatenate(
                (
                    np.full(5, 0, dtype=np.dtype([("col", int_)])),
                    np.full(5, 1, dtype=np.dtype([("col", int_)])),
                    np.full(5, 2, dtype=np.dtype([("col", int_)])),
                )
            ),
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf2 = pf.replace(in_outputs=in_outputs)

        with pytest.raises(AttributeError):
            pf2.get_in_output("my_arr")

        assert_series_equal(
            pf2.get_in_output("arr_1d_cs"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_cs, name_or_index="arr"),
        )
        assert_frame_equal(
            pf2.get_in_output("arr_2d_cs"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_cs)
        )
        assert_series_equal(
            pf2.get_in_output("arr_1d_pcg"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        assert_frame_equal(
            pf2.get_in_output("arr_2d_pcg"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_pcg)
        )
        assert_series_equal(
            pf2.get_in_output("arr_1d_pg"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_pg, name_or_index="arr"),
        )
        assert_frame_equal(
            pf2.get_in_output("arr_2d_pg"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_pg)
        )
        assert_series_equal(
            pf2.get_in_output("arr_1d_pc"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_pc, name_or_index="arr"),
        )
        assert_frame_equal(
            pf2.get_in_output("arr_2d_pc"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_pc)
        )
        assert_series_equal(
            pf2.get_in_output("arr_1d"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        assert_frame_equal(pf2.get_in_output("arr_2d"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_pcg))
        assert_records_close(pf2.get_in_output("arr_records"), pf2.in_outputs.arr_records)
        with pytest.raises(NotImplementedError):
            pf2.get_in_output("arr_records", force_wrapping=True)

        assert pf2["a"].in_outputs.arr_1d_cs == in_outputs.arr_1d_cs[0]
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_2d_cs, in_outputs.arr_2d_cs[:, 0])
        assert pf2["a"].in_outputs.arr_1d_pcg == in_outputs.arr_1d_pcg[0]
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_2d_pcg, in_outputs.arr_2d_pcg[:, 0])
        assert pf2["a"].in_outputs.arr_1d_pg == in_outputs.arr_1d_pg[0]
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_2d_pg, in_outputs.arr_2d_pg[:, 0])
        assert pf2["a"].in_outputs.arr_1d_pc == in_outputs.arr_1d_pc[0]
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_2d_pc, in_outputs.arr_2d_pc[:, 0])
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_records, in_outputs.arr_records[:5])

        in_outputs = dict(
            arr_1d_cs=np.arange(3),
            arr_2d_cs=np.arange(15).reshape((5, 3)),
            arr_1d_pcg=np.arange(2),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_1d_pg=np.arange(2),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_1d_pc=np.arange(3),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d=np.arange(2),
            arr_2d=np.arange(10).reshape((5, 2)),
            arr_records=np.concatenate(
                (
                    np.full(5, 0, dtype=np.dtype([("col", int_)])),
                    np.full(5, 1, dtype=np.dtype([("col", int_)])),
                    np.full(5, 2, dtype=np.dtype([("col", int_)])),
                )
            ),
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf_grouped2 = pf_grouped.replace(in_outputs=in_outputs)

        assert_series_equal(
            pf_grouped2.get_in_output("arr_1d_cs"),
            pf_grouped2.wrapper.wrap_reduced(
                pf_grouped2.in_outputs.arr_1d_cs, name_or_index="arr", group_by=False
            ),
        )
        assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d_cs"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_cs, group_by=False),
        )
        assert_series_equal(
            pf_grouped2.get_in_output("arr_1d_pcg"),
            pf_grouped2.wrapper.wrap_reduced(
                pf_grouped2.in_outputs.arr_1d_pcg, name_or_index="arr"
            ),
        )
        assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d_pcg"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_pcg),
        )
        assert_series_equal(
            pf_grouped2.get_in_output("arr_1d_pg"),
            pf_grouped2.wrapper.wrap_reduced(pf_grouped2.in_outputs.arr_1d_pg, name_or_index="arr"),
        )
        assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d_pg"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_pg),
        )
        assert_series_equal(
            pf_grouped2.get_in_output("arr_1d_pc"),
            pf_grouped2.wrapper.wrap_reduced(
                pf_grouped2.in_outputs.arr_1d_pc, name_or_index="arr", group_by=False
            ),
        )
        assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d_pc"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_pc, group_by=False),
        )
        assert_series_equal(
            pf_grouped2.get_in_output("arr_1d"),
            pf_grouped2.wrapper.wrap_reduced(
                pf_grouped2.in_outputs.arr_1d_pcg, name_or_index="arr"
            ),
        )
        assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_pcg),
        )
        assert_records_close(
            pf_grouped2.get_in_output("arr_records"), pf_grouped2.in_outputs.arr_records
        )
        with pytest.raises(NotImplementedError):
            pf_grouped2.get_in_output("arr_records", force_wrapping=True)

        np.testing.assert_array_equal(
            pf_grouped2["first"].in_outputs.arr_1d_cs, in_outputs.arr_1d_cs[:2]
        )
        np.testing.assert_array_equal(
            pf_grouped2["first"].in_outputs.arr_2d_cs, in_outputs.arr_2d_cs[:, :2]
        )
        assert pf_grouped2["first"].in_outputs.arr_1d_pcg == in_outputs.arr_1d_pcg[0]
        np.testing.assert_array_equal(
            pf_grouped2["first"].in_outputs.arr_2d_pcg, in_outputs.arr_2d_pcg[:, 0]
        )
        assert pf_grouped2["first"].in_outputs.arr_1d_pg == in_outputs.arr_1d_pg[0]
        np.testing.assert_array_equal(
            pf_grouped2["first"].in_outputs.arr_2d_pg, in_outputs.arr_2d_pg[:, 0]
        )
        np.testing.assert_array_equal(
            pf_grouped2["first"].in_outputs.arr_1d_pc, in_outputs.arr_1d_pc[:2]
        )
        np.testing.assert_array_equal(
            pf_grouped2["first"].in_outputs.arr_2d_pc, in_outputs.arr_2d_pc[:, :2]
        )
        np.testing.assert_array_equal(
            pf_grouped2["first"].in_outputs.arr_records, in_outputs.arr_records[:10]
        )

        in_outputs = dict(
            arr_1d_cs=np.arange(2),
            arr_2d_cs=np.arange(10).reshape((5, 2)),
            arr_1d_pcg=np.arange(2),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_1d_pg=np.arange(2),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_1d_pc=np.arange(3),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d=np.arange(2),
            arr_2d=np.arange(10).reshape((5, 2)),
            arr_records=np.concatenate(
                (
                    np.full(5, 0, dtype=np.dtype([("col", int_)])),
                    np.full(5, 1, dtype=np.dtype([("col", int_)])),
                    np.full(5, 2, dtype=np.dtype([("col", int_)])),
                )
            ),
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf_shared2 = pf_shared.replace(in_outputs=in_outputs)

        assert_series_equal(
            pf_shared2.get_in_output("arr_1d_cs"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_cs, name_or_index="arr"),
        )
        assert_frame_equal(
            pf_shared2.get_in_output("arr_2d_cs"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_cs),
        )
        assert_series_equal(
            pf_shared2.get_in_output("arr_1d_pcg"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        assert_frame_equal(
            pf_shared2.get_in_output("arr_2d_pcg"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_pcg),
        )
        assert_series_equal(
            pf_shared2.get_in_output("arr_1d_pg"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_pg, name_or_index="arr"),
        )
        assert_frame_equal(
            pf_shared2.get_in_output("arr_2d_pg"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_pg),
        )
        assert_series_equal(
            pf_shared2.get_in_output("arr_1d_pc"),
            pf_shared2.wrapper.wrap_reduced(
                pf_shared2.in_outputs.arr_1d_pc, name_or_index="arr", group_by=False
            ),
        )
        assert_frame_equal(
            pf_shared2.get_in_output("arr_2d_pc"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_pc, group_by=False),
        )
        assert_series_equal(
            pf_shared2.get_in_output("arr_1d"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        assert_frame_equal(
            pf_shared2.get_in_output("arr_2d"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_pcg),
        )
        assert_records_close(
            pf_shared2.get_in_output("arr_records"), pf_shared2.in_outputs.arr_records
        )
        with pytest.raises(NotImplementedError):
            pf_shared2.get_in_output("arr_records", force_wrapping=True)

        assert pf_shared2["first"].in_outputs.arr_1d_cs == in_outputs.arr_1d_cs[0]
        np.testing.assert_array_equal(
            pf_shared2["first"].in_outputs.arr_2d_cs, in_outputs.arr_2d_cs[:, 0]
        )
        assert pf_shared2["first"].in_outputs.arr_1d_pcg == in_outputs.arr_1d_pcg[0]
        np.testing.assert_array_equal(
            pf_shared2["first"].in_outputs.arr_2d_pcg, in_outputs.arr_2d_pcg[:, 0]
        )
        assert pf_shared2["first"].in_outputs.arr_1d_pg == in_outputs.arr_1d_pg[0]
        np.testing.assert_array_equal(
            pf_shared2["first"].in_outputs.arr_2d_pg, in_outputs.arr_2d_pg[:, 0]
        )
        np.testing.assert_array_equal(
            pf_shared2["first"].in_outputs.arr_1d_pc, in_outputs.arr_1d_pc[:2]
        )
        np.testing.assert_array_equal(
            pf_shared2["first"].in_outputs.arr_2d_pc, in_outputs.arr_2d_pc[:, :2]
        )
        np.testing.assert_array_equal(
            pf_shared2["first"].in_outputs.arr_records, in_outputs.arr_records[:10]
        )

    def test_close(self):
        assert_frame_equal(pf.close, close_na)
        assert_frame_equal(pf_grouped.close, close_na)
        assert_frame_equal(pf_shared.close, close_na)

    def test_get_filled_close(self):
        assert_frame_equal(pf.filled_close, close_na.ffill().bfill())
        assert_frame_equal(
            pf.filled_close,
            vbt.Portfolio.get_filled_close(close=pf.close, wrapper=pf.wrapper),
        )
        assert_frame_equal(
            pf.get_filled_close(jitted=dict(parallel=True)),
            pf.get_filled_close(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_filled_close(chunked=True), pf.get_filled_close(chunked=False))

    def test_bm_close(self):
        assert_frame_equal(pf.bm_close, bm_close_na)
        assert_frame_equal(pf_grouped.bm_close, bm_close_na)
        assert_frame_equal(pf_shared.bm_close, bm_close_na)
        assert pf.replace(bm_close=None).bm_close is None
        assert not pf.replace(bm_close=False).bm_close

    def test_get_filled_bm_close(self):
        assert_frame_equal(pf.filled_bm_close, bm_close_na.ffill().bfill())
        assert_frame_equal(
            pf.filled_bm_close,
            vbt.Portfolio.get_filled_bm_close(bm_close=pf.bm_close, wrapper=pf.wrapper),
        )
        assert_frame_equal(
            pf.get_filled_bm_close(jitted=dict(parallel=True)),
            pf.get_filled_bm_close(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf.get_filled_bm_close(chunked=True), pf.get_filled_bm_close(chunked=False)
        )
        assert pf.replace(bm_close=None).filled_bm_close is None
        assert not pf.replace(bm_close=False).filled_bm_close

    def test_views(self):
        assert_records_close(pf.long_view.entry_trades.values, pf.entry_trades.long_view.values)
        assert_records_close(pf.short_view.entry_trades.values, pf.entry_trades.short_view.values)
        assert_records_close(pf.long_view.exit_trades.values, pf.exit_trades.long_view.values)
        assert_records_close(pf.short_view.exit_trades.values, pf.exit_trades.short_view.values)
        assert_records_close(pf.long_view.positions.values, pf.positions.long_view.values)
        assert_records_close(pf.short_view.positions.values, pf.positions.short_view.values)
        np.testing.assert_allclose(
            [pf.order_records["size"].sum()],
            [pf.long_view.order_records["size"].sum() + pf.short_view.order_records["size"].sum()],
        )
        np.testing.assert_allclose(
            [pf.order_records["fees"].sum()],
            [pf.long_view.order_records["fees"].sum() + pf.short_view.order_records["fees"].sum()],
        )

    def test_orders(self):
        result = np.array(
            [
                (0, 0, 1, 0.1, 2.02, 0.10202, 0),
                (1, 0, 2, 1.0, 2.9699999999999998, 0.1297, 1),
                (2, 0, 3, 0.1, 3.96, 0.10396000000000001, 1),
                (3, 0, 4, 1.0, 5.05, 0.1505, 0),
                (0, 1, 1, 0.1, 1.98, 0.10198, 1),
                (1, 1, 3, 0.1, 4.04, 0.10404000000000001, 0),
                (2, 1, 4, 1.0, 4.95, 0.14950000000000002, 1),
                (0, 2, 0, 1.0, 1.01, 0.1101, 0),
                (1, 2, 1, 0.1, 2.02, 0.10202, 0),
                (2, 2, 2, 1.0, 2.9699999999999998, 0.1297, 1),
                (3, 2, 3, 0.1, 3.96, 0.10396000000000001, 1),
            ],
            dtype=order_dt,
        )
        assert_records_close(pf.orders.values, result)
        assert_records_close(pf_grouped.orders.values, result)
        assert_records_close(pf_shared.orders.values, result)
        result2 = pd.Series(np.array([4, 3, 4]), index=close_na.columns).rename("count")
        assert_series_equal(pf.orders.count(), result2)
        assert_series_equal(pf_grouped.get_orders(group_by=False).count(), result2)
        assert_series_equal(pf_shared.get_orders(group_by=False).count(), result2)
        result3 = pd.Series(np.array([7, 4]), index=group_columns).rename("count")
        assert_series_equal(pf.get_orders(group_by=group_by).count(), result3)
        assert_series_equal(pf_grouped.orders.count(), result3)
        assert_series_equal(pf_shared.orders.count(), result3)

        sim_range_result = result[[0, 1, 2, 4, 5, 8, 9, 10]]
        assert_records_close(pf.get_orders(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(pf_grouped.get_orders(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(pf_shared.get_orders(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(
            pf.get_orders(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_grouped.get_orders(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_shared.get_orders(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf.get_orders(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf.replace(sim_start=1, sim_end=4).orders.values,
        )
        assert_records_close(
            pf_grouped.get_orders(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_grouped.replace(sim_start=1, sim_end=4).orders.values,
        )
        assert_records_close(
            pf_shared.get_orders(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_shared.replace(sim_start=1, sim_end=4).orders.values,
        )
        result = np.array(
            [
                (0, 0, 1, 0.05, 2.02, 0.05101, 0),
                (1, 0, 2, 0.5, 2.9699999999999998, 0.06485, 1),
                (2, 0, 3, 0.05, 3.96, 0.051980000000000005, 1),
                (3, 0, 4, 0.5, 5.05, 0.07525, 0),
                (0, 1, 1, 0.1, 1.98, 0.10198, 1),
                (1, 1, 3, 0.1, 4.04, 0.10404000000000001, 0),
                (2, 1, 4, 1.0, 4.95, 0.14950000000000002, 1),
                (0, 2, 0, 1.5, 1.01, 0.16515000000000002, 0),
                (1, 2, 1, 0.15000000000000002, 2.02, 0.15303, 0),
                (2, 2, 2, 1.5, 2.9699999999999998, 0.19455, 1),
                (3, 2, 3, 0.15000000000000002, 3.96, 0.15594000000000002, 1),
            ],
            dtype=order_dt,
        )
        assert_records_close(
            pf.get_orders(weights=[0.5, 1.0, 1.5]).values,
            result,
        )
        assert_records_close(
            pf_grouped.get_orders(weights=[0.5, 1.0, 1.5]).values,
            result,
        )
        assert_records_close(
            pf_shared.get_orders(weights=[0.5, 1.0, 1.5]).values,
            result,
        )

    def test_logs(self):
        result = np.array(
            [
                (
                    0,
                    0,
                    0,
                    0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    100.0,
                    1.0,
                    0.0,
                    0.0,
                    100.0,
                    np.nan,
                    np.nan,
                    0.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    100.0,
                    1.0,
                    0.0,
                    0.0,
                    100.0,
                    np.nan,
                    np.nan,
                    -1,
                ),
                (
                    1,
                    0,
                    0,
                    1,
                    1.0,
                    2.5,
                    0.5,
                    2.0,
                    100.0,
                    1.0,
                    0.0,
                    0.0,
                    100.0,
                    2.0,
                    102.0,
                    0.1,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    0.1,
                    2.02,
                    0.10202,
                    0,
                    0,
                    -1,
                    99.69598,
                    1.1,
                    0.0,
                    0.0,
                    99.69598,
                    2.0,
                    102.0,
                    0,
                ),
                (
                    2,
                    0,
                    0,
                    2,
                    2.0,
                    3.5,
                    1.5,
                    3.0,
                    199.69598000000002,
                    1.1,
                    0.0,
                    0.0,
                    199.69598000000002,
                    3.0,
                    202.99598000000003,
                    -1.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    2.9699999999999998,
                    0.1297,
                    1,
                    0,
                    -1,
                    202.53628000000003,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    202.53628000000003,
                    3.0,
                    202.99598000000003,
                    1,
                ),
                (
                    3,
                    0,
                    0,
                    3,
                    3.0,
                    4.5,
                    2.5,
                    4.0,
                    202.53628000000003,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    202.53628000000003,
                    4.0,
                    202.93628000000004,
                    -0.1,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    0.1,
                    3.96,
                    0.10396000000000001,
                    1,
                    0,
                    -1,
                    202.82832000000002,
                    0.0,
                    0.0,
                    0.0,
                    202.82832000000002,
                    4.0,
                    202.93628000000004,
                    2,
                ),
                (
                    4,
                    0,
                    0,
                    4,
                    4.0,
                    5.5,
                    3.5,
                    5.0,
                    202.82832000000002,
                    0.0,
                    0.0,
                    0.0,
                    202.82832000000002,
                    5.0,
                    202.82832000000002,
                    1.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    0.1505,
                    0,
                    0,
                    -1,
                    197.62782,
                    1.0,
                    0.0,
                    0.0,
                    197.62782,
                    5.0,
                    202.82832000000002,
                    3,
                ),
                (
                    0,
                    1,
                    1,
                    0,
                    0.5,
                    1.5,
                    0.25,
                    1.0,
                    100.0,
                    -1.0,
                    0.0,
                    0.0,
                    100.0,
                    1.0,
                    99.0,
                    0.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    5,
                    100.0,
                    -1.0,
                    0.0,
                    0.0,
                    100.0,
                    1.0,
                    99.0,
                    -1,
                ),
                (
                    1,
                    1,
                    1,
                    1,
                    1.0,
                    2.5,
                    0.5,
                    2.0,
                    100.0,
                    -1.0,
                    0.0,
                    0.0,
                    100.0,
                    2.0,
                    98.0,
                    0.1,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    0.1,
                    1.98,
                    0.10198,
                    1,
                    0,
                    -1,
                    100.09602,
                    -1.1,
                    0.198,
                    0.198,
                    99.70002,
                    2.0,
                    98.0,
                    0,
                ),
                (
                    2,
                    1,
                    1,
                    2,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    200.09602,
                    -1.1,
                    0.198,
                    0.198,
                    199.70002,
                    2.0,
                    197.89602000000002,
                    -1.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    200.09602,
                    -1.1,
                    0.198,
                    0.198,
                    199.70002,
                    2.0,
                    197.89602000000002,
                    -1,
                ),
                (
                    3,
                    1,
                    1,
                    3,
                    3.0,
                    4.5,
                    2.5,
                    4.0,
                    200.09602,
                    -1.1,
                    0.198,
                    0.198,
                    199.70002,
                    4.0,
                    195.69602,
                    -0.1,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    0.1,
                    4.04,
                    0.10404000000000001,
                    0,
                    0,
                    -1,
                    199.58798000000002,
                    -1.0,
                    0.18,
                    0.18,
                    199.22798,
                    4.0,
                    195.69602,
                    1,
                ),
                (
                    4,
                    1,
                    1,
                    4,
                    4.0,
                    5.5,
                    3.5,
                    5.0,
                    199.58798000000002,
                    -1.0,
                    0.18,
                    0.18,
                    199.22798,
                    5.0,
                    194.58798000000002,
                    1.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    4.95,
                    0.14950000000000002,
                    1,
                    0,
                    -1,
                    204.38848000000002,
                    -2.0,
                    5.13,
                    5.13,
                    194.12848,
                    5.0,
                    194.58798000000002,
                    2,
                ),
                (
                    0,
                    2,
                    2,
                    0,
                    0.5,
                    1.5,
                    0.25,
                    1.0,
                    100.0,
                    0.0,
                    0.0,
                    0.0,
                    100.0,
                    1.0,
                    100.0,
                    1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    0.1101,
                    0,
                    0,
                    -1,
                    98.8799,
                    1.0,
                    0.0,
                    0.0,
                    98.8799,
                    1.0,
                    100.0,
                    0,
                ),
                (
                    1,
                    2,
                    2,
                    1,
                    1.0,
                    2.5,
                    0.5,
                    2.0,
                    98.8799,
                    1.0,
                    0.0,
                    0.0,
                    98.8799,
                    2.0,
                    100.8799,
                    0.1,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    0.1,
                    2.02,
                    0.10202,
                    0,
                    0,
                    -1,
                    98.57588000000001,
                    1.1,
                    0.0,
                    0.0,
                    98.57588000000001,
                    2.0,
                    100.8799,
                    1,
                ),
                (
                    2,
                    2,
                    2,
                    2,
                    2.0,
                    3.5,
                    1.5,
                    3.0,
                    98.57588000000001,
                    1.1,
                    0.0,
                    0.0,
                    98.57588000000001,
                    3.0,
                    101.87588000000001,
                    -1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    2.9699999999999998,
                    0.1297,
                    1,
                    0,
                    -1,
                    101.41618000000001,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    101.41618000000001,
                    3.0,
                    101.87588000000001,
                    2,
                ),
                (
                    3,
                    2,
                    2,
                    3,
                    3.0,
                    4.5,
                    2.5,
                    4.0,
                    101.41618000000001,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    101.41618000000001,
                    4.0,
                    101.81618000000002,
                    -0.1,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    0.1,
                    3.96,
                    0.10396000000000001,
                    1,
                    0,
                    -1,
                    101.70822000000001,
                    0.0,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.81618000000002,
                    3,
                ),
                (
                    4,
                    2,
                    2,
                    4,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    101.70822000000001,
                    0.0,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.70822000000001,
                    1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    101.70822000000001,
                    0.0,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.70822000000001,
                    -1,
                ),
            ],
            dtype=log_dt,
        )
        assert_records_close(pf.logs.values, result)
        assert_records_close(pf_grouped.logs.values, result)
        result_shared = np.array(
            [
                (
                    0,
                    0,
                    0,
                    0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    200.0,
                    1.0,
                    0.0,
                    0.0,
                    200.0,
                    np.nan,
                    np.nan,
                    0.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    200.0,
                    1.0,
                    0.0,
                    0.0,
                    200.0,
                    np.nan,
                    np.nan,
                    -1,
                ),
                (
                    1,
                    0,
                    0,
                    1,
                    1.0,
                    2.5,
                    0.5,
                    2.0,
                    200.09602,
                    1.0,
                    0.0,
                    0.0,
                    199.70002,
                    2.0,
                    200.0,
                    0.1,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    0.1,
                    2.02,
                    0.10202,
                    0,
                    0,
                    -1,
                    199.792,
                    1.1,
                    0.0,
                    0.0,
                    199.396,
                    2.0,
                    200.0,
                    0,
                ),
                (
                    2,
                    0,
                    0,
                    2,
                    2.0,
                    3.5,
                    1.5,
                    3.0,
                    399.79200000000003,
                    1.1,
                    0.0,
                    0.0,
                    399.39599999999996,
                    3.0,
                    400.89200000000005,
                    -1.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    2.9699999999999998,
                    0.1297,
                    1,
                    0,
                    -1,
                    402.63230000000004,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    402.23629999999997,
                    3.0,
                    400.89200000000005,
                    1,
                ),
                (
                    3,
                    0,
                    0,
                    3,
                    3.0,
                    4.5,
                    2.5,
                    4.0,
                    402.12426000000005,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    401.76426,
                    4.0,
                    398.63230000000004,
                    -0.1,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    0.1,
                    3.96,
                    0.10396000000000001,
                    1,
                    0,
                    -1,
                    402.41630000000004,
                    0.0,
                    0.0,
                    0.0,
                    402.05629999999996,
                    4.0,
                    398.63230000000004,
                    2,
                ),
                (
                    4,
                    0,
                    0,
                    4,
                    4.0,
                    5.5,
                    3.5,
                    5.0,
                    407.21680000000003,
                    0.0,
                    0.0,
                    0.0,
                    396.9568,
                    5.0,
                    397.41630000000004,
                    1.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
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
                    0.1505,
                    0,
                    0,
                    -1,
                    402.01630000000006,
                    1.0,
                    0.0,
                    0.0,
                    391.7563,
                    5.0,
                    397.41630000000004,
                    3,
                ),
                (
                    0,
                    0,
                    1,
                    0,
                    0.5,
                    1.5,
                    0.25,
                    1.0,
                    200.0,
                    -1.0,
                    0.0,
                    0.0,
                    200.0,
                    1.0,
                    np.nan,
                    0.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    5,
                    200.0,
                    -1.0,
                    0.0,
                    0.0,
                    200.0,
                    1.0,
                    np.nan,
                    -1,
                ),
                (
                    1,
                    0,
                    1,
                    1,
                    1.0,
                    2.5,
                    0.5,
                    2.0,
                    200.0,
                    -1.0,
                    0.0,
                    0.0,
                    200.0,
                    2.0,
                    200.0,
                    0.1,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    0.1,
                    1.98,
                    0.10198,
                    1,
                    0,
                    -1,
                    200.09602,
                    -1.1,
                    0.198,
                    0.198,
                    199.70002,
                    2.0,
                    200.0,
                    0,
                ),
                (
                    2,
                    0,
                    1,
                    2,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    399.79200000000003,
                    -1.1,
                    0.198,
                    0.198,
                    399.39599999999996,
                    2.0,
                    400.89200000000005,
                    -1.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    399.79200000000003,
                    -1.1,
                    0.198,
                    0.198,
                    399.39599999999996,
                    2.0,
                    400.89200000000005,
                    -1,
                ),
                (
                    3,
                    0,
                    1,
                    3,
                    3.0,
                    4.5,
                    2.5,
                    4.0,
                    402.63230000000004,
                    -1.1,
                    0.198,
                    0.198,
                    402.23629999999997,
                    4.0,
                    398.63230000000004,
                    -0.1,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    0.1,
                    4.04,
                    0.10404000000000001,
                    0,
                    0,
                    -1,
                    402.12426000000005,
                    -1.0,
                    0.18,
                    0.18,
                    401.76426,
                    4.0,
                    398.63230000000004,
                    1,
                ),
                (
                    4,
                    0,
                    1,
                    4,
                    4.0,
                    5.5,
                    3.5,
                    5.0,
                    402.41630000000004,
                    -1.0,
                    0.18,
                    0.18,
                    402.05629999999996,
                    5.0,
                    397.41630000000004,
                    1.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
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
                    4.95,
                    0.14950000000000002,
                    1,
                    0,
                    -1,
                    407.21680000000003,
                    -2.0,
                    5.13,
                    5.13,
                    396.9568,
                    5.0,
                    397.41630000000004,
                    2,
                ),
                (
                    0,
                    1,
                    2,
                    0,
                    0.5,
                    1.5,
                    0.25,
                    1.0,
                    100.0,
                    0.0,
                    0.0,
                    0.0,
                    100.0,
                    1.0,
                    100.0,
                    1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    0.1101,
                    0,
                    0,
                    -1,
                    98.8799,
                    1.0,
                    0.0,
                    0.0,
                    98.8799,
                    1.0,
                    100.0,
                    0,
                ),
                (
                    1,
                    1,
                    2,
                    1,
                    1.0,
                    2.5,
                    0.5,
                    2.0,
                    98.8799,
                    1.0,
                    0.0,
                    0.0,
                    98.8799,
                    2.0,
                    100.8799,
                    0.1,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    0.1,
                    2.02,
                    0.10202,
                    0,
                    0,
                    -1,
                    98.57588000000001,
                    1.1,
                    0.0,
                    0.0,
                    98.57588000000001,
                    2.0,
                    100.8799,
                    1,
                ),
                (
                    2,
                    1,
                    2,
                    2,
                    2.0,
                    3.5,
                    1.5,
                    3.0,
                    98.57588000000001,
                    1.1,
                    0.0,
                    0.0,
                    98.57588000000001,
                    3.0,
                    101.87588000000001,
                    -1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    2.9699999999999998,
                    0.1297,
                    1,
                    0,
                    -1,
                    101.41618000000001,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    101.41618000000001,
                    3.0,
                    101.87588000000001,
                    2,
                ),
                (
                    3,
                    1,
                    2,
                    3,
                    3.0,
                    4.5,
                    2.5,
                    4.0,
                    101.41618000000001,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    101.41618000000001,
                    4.0,
                    101.81618000000002,
                    -0.1,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    0.1,
                    3.96,
                    0.10396000000000001,
                    1,
                    0,
                    -1,
                    101.70822000000001,
                    0.0,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.81618000000002,
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
                    np.nan,
                    101.70822000000001,
                    0.0,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.70822000000001,
                    1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    101.70822000000001,
                    0.0,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.70822000000001,
                    -1,
                ),
            ],
            dtype=log_dt,
        )
        assert_records_close(pf_shared.logs.values, result_shared)
        result2 = pd.Series(np.array([5, 5, 5]), index=close_na.columns).rename("count")
        assert_series_equal(pf.logs.count(), result2)
        assert_series_equal(pf_grouped.get_logs(group_by=False).count(), result2)
        assert_series_equal(pf_shared.get_logs(group_by=False).count(), result2)
        result3 = pd.Series(
            np.array([10, 5]),
            index=group_columns,
        ).rename("count")
        assert_series_equal(pf.get_logs(group_by=group_by).count(), result3)
        assert_series_equal(pf_grouped.logs.count(), result3)
        assert_series_equal(pf_shared.logs.count(), result3)

        sim_range_result = result[[1, 2, 3, 6, 7, 8, 11, 12, 13]]
        sim_range_result_shared = result_shared[[1, 2, 3, 6, 7, 8, 11, 12, 13]]
        assert_records_close(pf.get_logs(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(pf_grouped.get_logs(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(
            pf_shared.get_logs(sim_start=1, sim_end=4).values, sim_range_result_shared
        )
        assert_records_close(
            pf.get_logs(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_grouped.get_logs(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_shared.get_logs(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result_shared,
        )
        assert_records_close(
            pf.get_logs(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf.replace(sim_start=1, sim_end=4).logs.values,
        )
        assert_records_close(
            pf_grouped.get_logs(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_grouped.replace(sim_start=1, sim_end=4).logs.values,
        )
        assert_records_close(
            pf_shared.get_logs(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_shared.replace(sim_start=1, sim_end=4).logs.values,
        )

    def test_entry_trades(self):
        result = np.array(
            [
                (
                    0,
                    0,
                    1.0,
                    -1,
                    -1,
                    2.0,
                    0.0,
                    2,
                    3,
                    3.0599999999999996,
                    0.21241818181818184,
                    0.8475818181818178,
                    0.4237909090909089,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    0,
                    0.1,
                    0,
                    1,
                    2.02,
                    0.10202,
                    2,
                    3,
                    3.0599999999999996,
                    0.021241818181818185,
                    -0.019261818181818203,
                    -0.09535553555355546,
                    0,
                    1,
                    0,
                ),
                (
                    2,
                    0,
                    1.0,
                    3,
                    4,
                    5.05,
                    0.1505,
                    -1,
                    4,
                    5.0,
                    0.0,
                    -0.20049999999999982,
                    -0.03970297029702967,
                    0,
                    0,
                    1,
                ),
                (
                    0,
                    1,
                    -1.0,
                    -1,
                    -1,
                    1.0,
                    0.0,
                    -1,
                    4,
                    4.954285714285714,
                    -0.049542857142857145,
                    4.003828571428571,
                    -4.003828571428571,
                    1,
                    0,
                    0,
                ),
                (
                    1,
                    1,
                    0.1,
                    0,
                    1,
                    1.98,
                    0.10198,
                    -1,
                    4,
                    4.954285714285714,
                    0.004954285714285714,
                    -0.4043628571428571,
                    -2.0422366522366517,
                    1,
                    0,
                    0,
                ),
                (
                    2,
                    1,
                    1.0,
                    2,
                    4,
                    4.95,
                    0.14950000000000002,
                    -1,
                    4,
                    4.954285714285714,
                    0.049542857142857145,
                    -0.20332857142857072,
                    -0.04107647907647893,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    1.0,
                    0,
                    0,
                    1.01,
                    0.1101,
                    3,
                    3,
                    3.0599999999999996,
                    0.21241818181818184,
                    1.727481818181818,
                    1.71037803780378,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    2,
                    0.1,
                    1,
                    1,
                    2.02,
                    0.10202,
                    3,
                    3,
                    3.0599999999999996,
                    0.021241818181818185,
                    -0.019261818181818203,
                    -0.09535553555355546,
                    0,
                    1,
                    0,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.entry_trades.values, result)
        assert_records_close(pf_grouped.entry_trades.values, result)
        assert_records_close(pf_shared.entry_trades.values, result)
        result2 = pd.Series(np.array([3, 3, 2]), index=close_na.columns).rename("count")
        assert_series_equal(pf.entry_trades.count(), result2)
        assert_series_equal(pf_grouped.get_entry_trades(group_by=False).count(), result2)
        assert_series_equal(pf_shared.get_entry_trades(group_by=False).count(), result2)
        result3 = pd.Series(np.array([6, 2]), index=group_columns).rename("count")
        assert_series_equal(pf.get_entry_trades(group_by=group_by).count(), result3)
        assert_series_equal(pf_grouped.entry_trades.count(), result3)
        assert_series_equal(pf_shared.entry_trades.count(), result3)

        sim_range_result = np.array(
            [
                (
                    0,
                    0,
                    1.0,
                    -1,
                    -1,
                    2.0,
                    0.0,
                    2,
                    3,
                    3.0599999999999996,
                    0.21241818181818184,
                    0.8475818181818178,
                    0.4237909090909089,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    0,
                    0.1,
                    0,
                    1,
                    2.02,
                    0.10202,
                    2,
                    3,
                    3.0599999999999996,
                    0.021241818181818185,
                    -0.019261818181818203,
                    -0.09535553555355546,
                    0,
                    1,
                    0,
                ),
                (
                    0,
                    1,
                    -1.0,
                    -1,
                    -1,
                    1.0,
                    0.0,
                    -1,
                    3,
                    4.003636363636363,
                    -0.09458181818181818,
                    3.098218181818181,
                    -3.098218181818181,
                    1,
                    0,
                    0,
                ),
                (
                    1,
                    1,
                    0.1,
                    0,
                    1,
                    1.98,
                    0.10198,
                    -1,
                    3,
                    4.003636363636363,
                    0.009458181818181819,
                    -0.3138018181818181,
                    -1.58485766758494,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    0.1,
                    1,
                    1,
                    2.02,
                    0.10202,
                    2,
                    2,
                    2.9699999999999998,
                    0.012970000000000002,
                    -0.01999000000000003,
                    -0.09896039603960409,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    2,
                    0.9,
                    2,
                    2,
                    2.9699999999999998,
                    0.11673000000000001,
                    -1,
                    3,
                    4.0,
                    0.0,
                    -1.04373,
                    -0.39047138047138047,
                    1,
                    0,
                    1,
                ),
                (
                    2,
                    2,
                    0.1,
                    3,
                    3,
                    3.96,
                    0.10396000000000001,
                    -1,
                    3,
                    4.0,
                    0.0,
                    -0.10796000000000001,
                    -0.27262626262626266,
                    1,
                    0,
                    1,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.get_entry_trades(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(
            pf_grouped.get_entry_trades(sim_start=1, sim_end=4).values, sim_range_result
        )
        assert_records_close(
            pf_shared.get_entry_trades(sim_start=1, sim_end=4).values, sim_range_result
        )
        assert_records_close(
            pf.get_entry_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_grouped.get_entry_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_shared.get_entry_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf.get_entry_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf.replace(sim_start=1, sim_end=4).entry_trades.values,
        )
        assert_records_close(
            pf_grouped.get_entry_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_grouped.replace(sim_start=1, sim_end=4).entry_trades.values,
        )
        assert_records_close(
            pf_shared.get_entry_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_shared.replace(sim_start=1, sim_end=4).entry_trades.values,
        )

    def test_exit_trades(self):
        result = np.array(
            [
                (
                    0,
                    0,
                    1.0,
                    -1,
                    -1,
                    2.0018181818181815,
                    0.09274545454545455,
                    1,
                    2,
                    2.9699999999999998,
                    0.1297,
                    0.7457363636363636,
                    0.37252951861943695,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    0,
                    0.10000000000000009,
                    -1,
                    -1,
                    2.0018181818181815,
                    0.009274545454545462,
                    2,
                    3,
                    3.96,
                    0.10396000000000001,
                    0.08258363636363657,
                    0.41254314259763925,
                    0,
                    1,
                    0,
                ),
                (
                    2,
                    0,
                    1.0,
                    3,
                    4,
                    5.05,
                    0.1505,
                    -1,
                    4,
                    5.0,
                    0.0,
                    -0.20049999999999982,
                    -0.03970297029702967,
                    0,
                    0,
                    1,
                ),
                (
                    0,
                    1,
                    0.1,
                    -1,
                    -1,
                    1.0890909090909089,
                    0.009270909090909092,
                    1,
                    3,
                    4.04,
                    0.10404000000000001,
                    -0.40840181818181825,
                    -3.749933222036729,
                    1,
                    1,
                    0,
                ),
                (
                    1,
                    1,
                    2.0,
                    -1,
                    -1,
                    3.0195454545454545,
                    0.24220909090909093,
                    -1,
                    4,
                    5.0,
                    0.0,
                    -4.203118181818182,
                    -0.6959852476290832,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    1.0,
                    0,
                    0,
                    1.1018181818181818,
                    0.19283636363636364,
                    2,
                    2,
                    2.9699999999999998,
                    0.1297,
                    1.5456454545454543,
                    1.4028135313531351,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    2,
                    0.10000000000000009,
                    0,
                    0,
                    1.1018181818181818,
                    0.019283636363636378,
                    3,
                    3,
                    3.96,
                    0.10396000000000001,
                    0.1625745454545457,
                    1.4755115511551162,
                    0,
                    1,
                    0,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.exit_trades.values, result)
        assert_records_close(pf_grouped.exit_trades.values, result)
        assert_records_close(pf_shared.exit_trades.values, result)
        result2 = pd.Series(np.array([3, 2, 2]), index=close_na.columns).rename("count")
        assert_series_equal(pf.exit_trades.count(), result2)
        assert_series_equal(pf_grouped.get_exit_trades(group_by=False).count(), result2)
        assert_series_equal(pf_shared.get_exit_trades(group_by=False).count(), result2)
        result3 = pd.Series(np.array([5, 2]), index=group_columns).rename("count")
        assert_series_equal(pf.get_exit_trades(group_by=group_by).count(), result3)
        assert_series_equal(pf_grouped.exit_trades.count(), result3)
        assert_series_equal(pf_shared.exit_trades.count(), result3)

        sim_range_result = np.array(
            [
                (
                    0,
                    0,
                    1.0,
                    -1,
                    -1,
                    2.0018181818181815,
                    0.09274545454545455,
                    1,
                    2,
                    2.9699999999999998,
                    0.1297,
                    0.7457363636363636,
                    0.37252951861943695,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    0,
                    0.10000000000000009,
                    -1,
                    -1,
                    2.0018181818181815,
                    0.009274545454545462,
                    2,
                    3,
                    3.96,
                    0.10396000000000001,
                    0.08258363636363657,
                    0.41254314259763925,
                    0,
                    1,
                    0,
                ),
                (
                    0,
                    1,
                    0.1,
                    -1,
                    -1,
                    1.0890909090909089,
                    0.009270909090909092,
                    1,
                    3,
                    4.04,
                    0.10404000000000001,
                    -0.40840181818181825,
                    -3.749933222036729,
                    1,
                    1,
                    0,
                ),
                (
                    1,
                    1,
                    1.0,
                    -1,
                    -1,
                    1.089090909090909,
                    0.09270909090909091,
                    -1,
                    3,
                    4.0,
                    0.0,
                    -3.003618181818182,
                    -2.7579131886477466,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    0.1,
                    1,
                    1,
                    2.02,
                    0.10202,
                    2,
                    2,
                    2.9699999999999998,
                    0.012970000000000002,
                    -0.01999000000000003,
                    -0.09896039603960409,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    2,
                    1.0,
                    2,
                    2,
                    3.069,
                    0.22069,
                    -1,
                    3,
                    4.0,
                    0.0,
                    -1.15169,
                    -0.37526555881394597,
                    1,
                    0,
                    1,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.get_exit_trades(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(
            pf_grouped.get_exit_trades(sim_start=1, sim_end=4).values, sim_range_result
        )
        assert_records_close(
            pf_shared.get_exit_trades(sim_start=1, sim_end=4).values, sim_range_result
        )
        assert_records_close(
            pf.get_exit_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_grouped.get_exit_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_shared.get_exit_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf.get_exit_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf.replace(sim_start=1, sim_end=4).exit_trades.values,
        )
        assert_records_close(
            pf_grouped.get_exit_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_grouped.replace(sim_start=1, sim_end=4).exit_trades.values,
        )
        assert_records_close(
            pf_shared.get_exit_trades(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_shared.replace(sim_start=1, sim_end=4).exit_trades.values,
        )

    def test_trade_history(self):
        result = pd.DataFrame(
            [
                [
                    0,
                    "a",
                    pd.Timestamp("2020-01-02 00:00:00"),
                    "Buy",
                    0.1,
                    2.02,
                    0.10202,
                    -0.019261818181818203,
                    -0.09535553555355546,
                    "Long",
                    "Closed",
                    1,
                    -1,
                    0,
                ],
                [
                    1,
                    "a",
                    pd.Timestamp("2020-01-03 00:00:00"),
                    "Sell",
                    1.0,
                    2.9699999999999998,
                    0.1297,
                    0.7457363636363636,
                    0.37252951861943695,
                    "Long",
                    "Closed",
                    -1,
                    0,
                    0,
                ],
                [
                    2,
                    "a",
                    pd.Timestamp("2020-01-04 00:00:00"),
                    "Sell",
                    0.10000000000000009,
                    3.96,
                    0.10396000000000001,
                    0.08258363636363657,
                    0.41254314259763925,
                    "Long",
                    "Closed",
                    -1,
                    1,
                    0,
                ],
                [
                    3,
                    "a",
                    pd.Timestamp("2020-01-05 00:00:00"),
                    "Buy",
                    1.0,
                    5.05,
                    0.1505,
                    -0.20049999999999982,
                    -0.03970297029702967,
                    "Long",
                    "Open",
                    2,
                    -1,
                    1,
                ],
                [
                    0,
                    "b",
                    pd.Timestamp("2020-01-02 00:00:00"),
                    "Sell",
                    0.1,
                    1.98,
                    0.10198,
                    -0.4043628571428571,
                    -2.0422366522366517,
                    "Short",
                    "Open",
                    1,
                    -1,
                    0,
                ],
                [
                    1,
                    "b",
                    pd.Timestamp("2020-01-04 00:00:00"),
                    "Buy",
                    0.1,
                    4.04,
                    0.10404000000000001,
                    -0.40840181818181825,
                    -3.749933222036729,
                    "Short",
                    "Closed",
                    -1,
                    0,
                    0,
                ],
                [
                    2,
                    "b",
                    pd.Timestamp("2020-01-05 00:00:00"),
                    "Sell",
                    1.0,
                    4.95,
                    0.14950000000000002,
                    -0.20332857142857072,
                    -0.04107647907647893,
                    "Short",
                    "Open",
                    2,
                    -1,
                    0,
                ],
                [
                    0,
                    "c",
                    pd.Timestamp("2020-01-01 00:00:00"),
                    "Buy",
                    1.0,
                    1.01,
                    0.1101,
                    1.727481818181818,
                    1.71037803780378,
                    "Long",
                    "Closed",
                    0,
                    -1,
                    0,
                ],
                [
                    1,
                    "c",
                    pd.Timestamp("2020-01-02 00:00:00"),
                    "Buy",
                    0.1,
                    2.02,
                    0.10202,
                    -0.019261818181818203,
                    -0.09535553555355546,
                    "Long",
                    "Closed",
                    1,
                    -1,
                    0,
                ],
                [
                    2,
                    "c",
                    pd.Timestamp("2020-01-03 00:00:00"),
                    "Sell",
                    1.0,
                    2.9699999999999998,
                    0.1297,
                    1.5456454545454543,
                    1.4028135313531351,
                    "Long",
                    "Closed",
                    -1,
                    0,
                    0,
                ],
                [
                    3,
                    "c",
                    pd.Timestamp("2020-01-04 00:00:00"),
                    "Sell",
                    0.10000000000000009,
                    3.96,
                    0.10396000000000001,
                    0.1625745454545457,
                    1.4755115511551162,
                    "Long",
                    "Closed",
                    -1,
                    1,
                    0,
                ],
            ],
            columns=pd.Index(
                [
                    "Order Id",
                    "Column",
                    "Index",
                    "Side",
                    "Size",
                    "Price",
                    "Fees",
                    "PnL",
                    "Return",
                    "Direction",
                    "Status",
                    "Entry Trade Id",
                    "Exit Trade Id",
                    "Position Id",
                ],
                dtype="object",
            ),
        )
        assert_frame_equal(pf.trade_history, result)
        assert_frame_equal(pf_grouped.trade_history, result)
        assert_frame_equal(pf_shared.trade_history, result)

        sim_range_result = pd.DataFrame(
            [
                [
                    0,
                    "a",
                    pd.Timestamp("2020-01-02 00:00:00"),
                    "Buy",
                    0.1,
                    2.02,
                    0.10202,
                    -0.019261818181818203,
                    -0.09535553555355546,
                    "Long",
                    "Closed",
                    1,
                    -1,
                    0,
                ],
                [
                    1,
                    "a",
                    pd.Timestamp("2020-01-03 00:00:00"),
                    "Sell",
                    1.0,
                    2.9699999999999998,
                    0.1297,
                    0.7457363636363636,
                    0.37252951861943695,
                    "Long",
                    "Closed",
                    -1,
                    0,
                    0,
                ],
                [
                    2,
                    "a",
                    pd.Timestamp("2020-01-04 00:00:00"),
                    "Sell",
                    0.10000000000000009,
                    3.96,
                    0.10396000000000001,
                    0.08258363636363657,
                    0.41254314259763925,
                    "Long",
                    "Closed",
                    -1,
                    1,
                    0,
                ],
                [
                    0,
                    "b",
                    pd.Timestamp("2020-01-02 00:00:00"),
                    "Sell",
                    0.1,
                    1.98,
                    0.10198,
                    -0.3138018181818181,
                    -1.58485766758494,
                    "Short",
                    "Open",
                    1,
                    -1,
                    0,
                ],
                [
                    1,
                    "b",
                    pd.Timestamp("2020-01-04 00:00:00"),
                    "Buy",
                    0.1,
                    4.04,
                    0.10404000000000001,
                    -0.40840181818181825,
                    -3.749933222036729,
                    "Short",
                    "Closed",
                    -1,
                    0,
                    0,
                ],
                [
                    1,
                    "c",
                    pd.Timestamp("2020-01-02 00:00:00"),
                    "Buy",
                    0.1,
                    2.02,
                    0.10202,
                    -0.01999000000000003,
                    -0.09896039603960409,
                    "Long",
                    "Closed",
                    0,
                    -1,
                    0,
                ],
                [
                    2,
                    "c",
                    pd.Timestamp("2020-01-03 00:00:00"),
                    "Sell",
                    0.1,
                    2.9699999999999998,
                    0.012970000000000002,
                    -0.01999000000000003,
                    -0.09896039603960409,
                    "Long",
                    "Closed",
                    -1,
                    0,
                    0,
                ],
                [
                    2,
                    "c",
                    pd.Timestamp("2020-01-03 00:00:00"),
                    "Sell",
                    0.9,
                    2.9699999999999998,
                    0.11673000000000001,
                    -1.04373,
                    -0.39047138047138047,
                    "Short",
                    "Open",
                    1,
                    -1,
                    1,
                ],
                [
                    3,
                    "c",
                    pd.Timestamp("2020-01-04 00:00:00"),
                    "Sell",
                    0.1,
                    3.96,
                    0.10396000000000001,
                    -0.10796000000000001,
                    -0.27262626262626266,
                    "Short",
                    "Open",
                    2,
                    -1,
                    1,
                ],
            ],
            columns=pd.Index(
                [
                    "Order Id",
                    "Column",
                    "Index",
                    "Side",
                    "Size",
                    "Price",
                    "Fees",
                    "PnL",
                    "Return",
                    "Direction",
                    "Status",
                    "Entry Trade Id",
                    "Exit Trade Id",
                    "Position Id",
                ],
                dtype="object",
            ),
        )
        assert_frame_equal(pf.get_trade_history(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(pf_grouped.get_trade_history(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(pf_shared.get_trade_history(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(
            pf.get_trade_history(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf_grouped.get_trade_history(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf_shared.get_trade_history(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf.get_trade_history(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).trade_history,
        )
        assert_frame_equal(
            pf_grouped.get_trade_history(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).trade_history,
        )
        assert_frame_equal(
            pf_shared.get_trade_history(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).trade_history,
        )

    def test_signals(self):
        keys = ["long_entries", "long_exits", "short_entries", "short_exits"]
        assert_frame_equal(
            pd.concat(pf.get_signals(), keys=keys, axis=1),
            pd.DataFrame(
                [
                    [
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
                        False,
                    ],
                    [
                        True,
                        False,
                        True,
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
                    [
                        False,
                        False,
                        False,
                        True,
                        False,
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
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                    ],
                    [
                        True,
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
                ],
                index=close_na.index,
                columns=pd.MultiIndex.from_product((keys, close_na.columns)),
            ),
        )
        assert_frame_equal(
            pd.concat(pf_grouped.get_signals(), keys=keys, axis=1),
            pd.DataFrame(
                [
                    [False, True, False, False, False, False, False, False],
                    [True, True, False, False, True, False, False, False],
                    [False, False, True, True, False, False, True, False],
                    [False, False, True, True, True, False, True, False],
                    [True, False, True, False, True, False, False, False],
                ],
                index=close_na.index,
                columns=pd.MultiIndex.from_product((keys, group_columns)),
            ),
        )
        assert_frame_equal(
            pd.concat(pf_shared.get_signals(), keys=keys, axis=1),
            pd.DataFrame(
                [
                    [False, True, False, False, False, False, False, False],
                    [True, True, False, False, True, False, False, False],
                    [False, False, True, True, False, False, True, False],
                    [False, False, True, True, True, False, True, False],
                    [True, False, True, False, True, False, False, False],
                ],
                index=close_na.index,
                columns=pd.MultiIndex.from_product((keys, group_columns)),
            ),
        )

    def test_positions(self):
        result = np.array(
            [
                (
                    0,
                    0,
                    1.1,
                    -1,
                    -1,
                    2.0018181818181815,
                    0.10202000000000001,
                    2,
                    3,
                    3.06,
                    0.23366000000000003,
                    0.8283200000000006,
                    0.37616712079927367,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    0,
                    1.0,
                    3,
                    4,
                    5.05,
                    0.1505,
                    -1,
                    4,
                    5.0,
                    0.0,
                    -0.20049999999999982,
                    -0.03970297029702967,
                    0,
                    0,
                    1,
                ),
                (
                    0,
                    1,
                    2.1,
                    -1,
                    -1,
                    2.9276190476190473,
                    0.25148000000000004,
                    -1,
                    4,
                    4.954285714285714,
                    0.10404000000000001,
                    -4.6115200000000005,
                    -0.7500845803513339,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    1.1,
                    0,
                    0,
                    1.1018181818181818,
                    0.21212000000000003,
                    3,
                    3,
                    3.06,
                    0.23366000000000003,
                    1.7082200000000003,
                    1.4094224422442245,
                    0,
                    1,
                    0,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.positions.values, result)
        assert_records_close(pf_grouped.positions.values, result)
        assert_records_close(pf_shared.positions.values, result)
        result2 = pd.Series(np.array([2, 1, 1]), index=close_na.columns).rename("count")
        assert_series_equal(pf.positions.count(), result2)
        assert_series_equal(pf_grouped.get_positions(group_by=False).count(), result2)
        assert_series_equal(pf_shared.get_positions(group_by=False).count(), result2)
        result3 = pd.Series(np.array([3, 1]), index=group_columns).rename("count")
        assert_series_equal(pf.get_positions(group_by=group_by).count(), result3)
        assert_series_equal(pf_grouped.positions.count(), result3)
        assert_series_equal(pf_shared.positions.count(), result3)

        sim_range_result = np.array(
            [
                (
                    0,
                    0,
                    1.1,
                    -1,
                    -1,
                    2.0018181818181815,
                    0.10202000000000001,
                    2,
                    3,
                    3.06,
                    0.23366000000000003,
                    0.8283200000000006,
                    0.37616712079927367,
                    0,
                    1,
                    0,
                ),
                (
                    0,
                    1,
                    1.1,
                    -1,
                    -1,
                    1.0890909090909089,
                    0.10198,
                    -1,
                    3,
                    4.003636363636363,
                    0.10404000000000001,
                    -3.41202,
                    -2.848096828046745,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    0.1,
                    1,
                    1,
                    2.02,
                    0.10202,
                    2,
                    2,
                    2.9699999999999998,
                    0.012970000000000002,
                    -0.01999000000000003,
                    -0.09896039603960409,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    2,
                    1.0,
                    2,
                    2,
                    3.069,
                    0.22069,
                    -1,
                    3,
                    4.0,
                    0.0,
                    -1.15169,
                    -0.37526555881394597,
                    1,
                    0,
                    1,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.get_positions(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(
            pf_grouped.get_positions(sim_start=1, sim_end=4).values, sim_range_result
        )
        assert_records_close(
            pf_shared.get_positions(sim_start=1, sim_end=4).values, sim_range_result
        )
        assert_records_close(
            pf.get_positions(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_grouped.get_positions(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_shared.get_positions(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf.get_positions(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf.replace(sim_start=1, sim_end=4).positions.values,
        )
        assert_records_close(
            pf_grouped.get_positions(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_grouped.replace(sim_start=1, sim_end=4).positions.values,
        )
        assert_records_close(
            pf_shared.get_positions(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_shared.replace(sim_start=1, sim_end=4).positions.values,
        )

    def test_drawdowns(self):
        result = np.array(
            [
                (0, 0, 0, 1, 2, 102.0, 101.89598000000001, 202.83628000000004, 1),
                (1, 0, 2, 4, 4, 202.83628000000004, 202.62782, 202.62782, 0),
                (0, 1, 0, 1, 2, 99.0, 97.89602, 197.89602000000002, 1),
                (1, 1, 2, 4, 4, 197.89602000000002, 194.38848000000002, 194.38848000000002, 0),
                (0, 2, 2, 3, 4, 101.71618000000001, 101.70822000000001, 101.70822000000001, 0),
            ],
            dtype=drawdown_dt,
        )
        assert_records_close(pf.drawdowns.values, result)
        result_grouped = np.array(
            [
                (0, 0, 0, 1, 2, 201.0, 199.792, 400.73230000000007, 1),
                (1, 0, 2, 4, 4, 400.73230000000007, 397.01630000000006, 397.01630000000006, 0),
                (0, 1, 2, 3, 4, 101.71618000000001, 101.70822000000001, 101.70822000000001, 0),
            ],
            dtype=drawdown_dt,
        )
        assert_records_close(pf_grouped.drawdowns.values, result_grouped)
        assert_records_close(pf_shared.drawdowns.values, result_grouped)
        result2 = pd.Series(np.array([2, 2, 1]), index=close_na.columns).rename("count")
        assert_series_equal(pf.drawdowns.count(), result2)
        assert_series_equal(pf_grouped.get_drawdowns(group_by=False).count(), result2)
        assert_series_equal(pf_shared.get_drawdowns(group_by=False).count(), result2)
        result3 = pd.Series(np.array([2, 1]), index=group_columns).rename("count")
        assert_series_equal(pf.get_drawdowns(group_by=group_by).count(), result3)
        assert_series_equal(pf_grouped.drawdowns.count(), result3)
        assert_series_equal(pf_shared.drawdowns.count(), result3)

        sim_range_result = np.array(
            [
                (0, 0, 2, 3, 3, 202.83628000000004, 202.82832000000002, 202.82832000000002, 0),
                (0, 1, 2, 3, 3, 197.89602000000002, 195.58798000000002, 195.58798000000002, 0),
                (0, 2, 2, 3, 3, 101.71618000000001, 101.70822000000001, 101.70822000000001, 0),
            ],
            dtype=drawdown_dt,
        )
        sim_range_result_grouped = np.array(
            [
                (0, 0, 2, 3, 3, 400.73230000000007, 398.41630000000004, 398.41630000000004, 0),
                (0, 1, 2, 3, 3, 101.71618000000001, 101.70822000000001, 101.70822000000001, 0),
            ],
            dtype=drawdown_dt,
        )
        assert_records_close(pf.get_drawdowns(sim_start=1, sim_end=4).values, sim_range_result)
        assert_records_close(
            pf_grouped.get_drawdowns(sim_start=1, sim_end=4).values, sim_range_result_grouped
        )
        assert_records_close(
            pf_shared.get_drawdowns(sim_start=1, sim_end=4).values, sim_range_result_grouped
        )
        sim_range_result = np.array(
            [
                (0, 0, 2, 3, 3, 202.83628000000004, 202.82832000000002, 202.82832000000002, 0),
                (0, 1, 2, 3, 3, 197.89602000000002, 195.58798000000002, 195.58798000000002, 0),
                (0, 2, 1, 3, 3, 99.89598000000001, 98.82832, 98.82832, 0),
            ],
            dtype=drawdown_dt,
        )
        sim_range_result_grouped = np.array(
            [
                (0, 0, 2, 3, 3, 400.73230000000007, 398.41630000000004, 398.41630000000004, 0),
                (0, 1, 1, 3, 3, 99.89598000000001, 98.82832, 98.82832, 0),
            ],
            dtype=drawdown_dt,
        )
        assert_records_close(
            pf.get_drawdowns(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result,
        )
        assert_records_close(
            pf_grouped.get_drawdowns(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result_grouped,
        )
        assert_records_close(
            pf_shared.get_drawdowns(sim_start=1, sim_end=4, rec_sim_range=True).values,
            sim_range_result_grouped,
        )
        assert_records_close(
            pf.get_drawdowns(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf.replace(sim_start=1, sim_end=4).drawdowns.values,
        )
        assert_records_close(
            pf_grouped.get_drawdowns(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_grouped.replace(sim_start=1, sim_end=4).drawdowns.values,
        )
        assert_records_close(
            pf_shared.get_drawdowns(sim_start=1, sim_end=4, rec_sim_range=True).values,
            pf_shared.replace(sim_start=1, sim_end=4).drawdowns.values,
        )

    def test_init_position(self):
        result = pd.Series(np.array([1.0, -1.0, 0.0]), index=close_na.columns).rename(
            "init_position"
        )
        assert_series_equal(pf.init_position, result)
        assert_series_equal(pf_grouped.init_position, result)
        assert_series_equal(pf_shared.init_position, result)
        result = pd.Series(np.array([0.5, -1.0, 0.0]), index=close_na.columns).rename(
            "init_position"
        )
        assert_series_equal(pf.get_init_position(weights=[0.5, 1.0, 1.5]), result)
        assert_series_equal(pf_grouped.get_init_position(weights=[0.5, 1.0, 1.5]), result)
        assert_series_equal(pf_shared.get_init_position(weights=[0.5, 1.0, 1.5]), result)

    def test_asset_flow(self):
        assert_frame_equal(
            pf.get_asset_flow(direction="longonly"),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [0.1, 0.0, 0.1],
                        [-1, 0.0, -1.0],
                        [-0.1, 0.0, -0.1],
                        [1.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.get_asset_flow(direction="shortonly"),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.1, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, -0.1, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.1, -0.1, 0.1],
                    [-1, 0.0, -1.0],
                    [-0.1, 0.1, -0.1],
                    [1.0, -1.0, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.asset_flow, result)
        assert_frame_equal(pf_grouped.asset_flow, result)
        assert_frame_equal(pf_shared.asset_flow, result)
        assert_frame_equal(
            pf.asset_flow,
            vbt.Portfolio.get_asset_flow(orders=pf.orders, init_position=pf.init_position),
        )
        assert_frame_equal(
            pf_grouped.asset_flow,
            vbt.Portfolio.get_asset_flow(
                orders=pf_grouped.orders, init_position=pf_grouped.init_position
            ),
        )
        assert_frame_equal(
            pf_shared.asset_flow,
            vbt.Portfolio.get_asset_flow(
                orders=pf_shared.orders, init_position=pf_shared.init_position
            ),
        )
        assert_frame_equal(
            pf.get_asset_flow(jitted=dict(parallel=True)),
            pf.get_asset_flow(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_asset_flow(chunked=False), pf.get_asset_flow(chunked=True))

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.1, -0.1, 0.1],
                [-1.0, 0.0, -1.0],
                [-0.1, 0.1, -0.1],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_asset_flow(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(pf_grouped.get_asset_flow(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(pf_shared.get_asset_flow(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(
            pf.get_asset_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf_grouped.get_asset_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf_shared.get_asset_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf.get_asset_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).asset_flow,
        )
        assert_frame_equal(
            pf_grouped.get_asset_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).asset_flow,
        )
        assert_frame_equal(
            pf_shared.get_asset_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).asset_flow,
        )

    def test_assets(self):
        assert_frame_equal(
            pf.get_assets(direction="longonly"),
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 0.0, 1.0],
                        [1.1, 0.0, 1.1],
                        [0.1, 0.0, 0.1],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.get_assets(direction="shortonly"),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 1.1, 0.0],
                        [0.0, 1.1, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 2.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [1.0, -1.0, 1.0],
                    [1.1, -1.1, 1.1],
                    [0.1, -1.1, 0.1],
                    [0.0, -1.0, 0.0],
                    [1.0, -2.0, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.assets, result)
        assert_frame_equal(pf_grouped.assets, result)
        assert_frame_equal(pf_shared.assets, result)
        assert_frame_equal(
            pf.assets,
            vbt.Portfolio.get_assets(
                asset_flow=pf.asset_flow, init_position=pf.init_position, wrapper=pf.wrapper
            ),
        )
        assert_frame_equal(
            pf_grouped.assets,
            vbt.Portfolio.get_assets(
                asset_flow=pf_grouped.asset_flow,
                init_position=pf_grouped.init_position,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.assets,
            vbt.Portfolio.get_assets(
                asset_flow=pf_shared.asset_flow,
                init_position=pf_shared.init_position,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_assets(jitted=dict(parallel=True)),
            pf.get_assets(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_assets(chunked=False), pf.get_assets(chunked=True))

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [1.1, -1.1, 0.1],
                [0.10000000000000009, -1.1, -0.9],
                [0.0, -1.0, -1.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_assets(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(pf_grouped.get_assets(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(pf_shared.get_assets(sim_start=1, sim_end=4), sim_range_result)
        assert_frame_equal(
            pf.get_assets(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf_grouped.get_assets(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf_shared.get_assets(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf.get_assets(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).assets,
        )
        assert_frame_equal(
            pf_grouped.get_assets(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).assets,
        )
        assert_frame_equal(
            pf_shared.get_assets(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).assets,
        )

    def test_position_mask(self):
        assert_frame_equal(
            pf.get_position_mask(direction="longonly"),
            pd.DataFrame(
                np.array(
                    [
                        [True, False, True],
                        [True, False, True],
                        [True, False, True],
                        [False, False, False],
                        [True, False, False],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.get_position_mask(direction="shortonly"),
            pd.DataFrame(
                np.array(
                    [
                        [False, True, False],
                        [False, True, False],
                        [False, True, False],
                        [False, True, False],
                        [False, True, False],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [False, True, False],
                [True, True, False],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.position_mask, result)
        assert_frame_equal(pf_grouped.get_position_mask(group_by=False), result)
        assert_frame_equal(pf_shared.get_position_mask(group_by=False), result)
        result = pd.DataFrame(
            np.array([[True, True], [True, True], [True, True], [True, False], [True, False]]),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_position_mask(group_by=group_by), result)
        assert_frame_equal(pf_grouped.position_mask, result)
        assert_frame_equal(pf_shared.position_mask, result)
        assert_frame_equal(
            pf.position_mask,
            vbt.Portfolio.get_position_mask(assets=pf.assets, wrapper=pf.wrapper),
        )
        assert_frame_equal(
            pf_grouped.position_mask,
            vbt.Portfolio.get_position_mask(assets=pf_grouped.assets, wrapper=pf_grouped.wrapper),
        )
        assert_frame_equal(
            pf_shared.position_mask,
            vbt.Portfolio.get_position_mask(assets=pf_shared.assets, wrapper=pf_shared.wrapper),
        )
        assert_frame_equal(
            pf.get_position_mask(jitted=dict(parallel=True)),
            pf.get_position_mask(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf.get_position_mask(chunked=True),
            pf.get_position_mask(chunked=False),
        )
        assert_frame_equal(
            pf_grouped.get_position_mask(jitted=dict(parallel=True)),
            pf_grouped.get_position_mask(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_grouped.get_position_mask(chunked=True),
            pf_grouped.get_position_mask(chunked=False),
        )

        sim_range_result = pd.DataFrame(
            [
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [False, True, False],
                [False, False, False],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_position_mask(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [[False, False], [True, True], [True, True], [True, False], [False, False]],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_position_mask(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_position_mask(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [False, True, True],
                [False, False, False],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_position_mask(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [[False, False], [True, True], [True, True], [True, True], [False, False]],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_position_mask(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_position_mask(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_position_mask(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).position_mask,
        )
        assert_frame_equal(
            pf_grouped.get_position_mask(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).position_mask,
        )
        assert_frame_equal(
            pf_shared.get_position_mask(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).position_mask,
        )

    def test_position_coverage(self):
        assert_series_equal(
            pf.get_position_coverage(direction="longonly"),
            pd.Series(np.array([0.8, 0.0, 0.6]), index=close_na.columns).rename(
                "position_coverage"
            ),
        )
        assert_series_equal(
            pf.get_position_coverage(direction="shortonly"),
            pd.Series(np.array([0.0, 1.0, 0.0]), index=close_na.columns).rename(
                "position_coverage"
            ),
        )
        result = pd.Series(np.array([0.8, 1.0, 0.6]), index=close_na.columns).rename(
            "position_coverage"
        )
        assert_series_equal(pf.position_coverage, result)
        assert_series_equal(pf_grouped.get_position_coverage(group_by=False), result)
        assert_series_equal(pf_shared.get_position_coverage(group_by=False), result)
        result = pd.Series(np.array([1.0, 0.6]), group_columns).rename("position_coverage")
        assert_series_equal(pf.get_position_coverage(group_by=group_by), result)
        assert_series_equal(pf_grouped.position_coverage, result)
        assert_series_equal(pf_shared.position_coverage, result)
        result = pd.Series(np.array([0.9, 0.6]), group_columns).rename("position_coverage")
        assert_series_equal(
            pf.get_position_coverage(granular_groups=True, group_by=group_by), result
        )
        assert_series_equal(pf_grouped.get_position_coverage(granular_groups=True), result)
        assert_series_equal(pf_shared.get_position_coverage(granular_groups=True), result)
        assert_series_equal(
            pf.position_coverage,
            vbt.Portfolio.get_position_coverage(assets=pf.assets, wrapper=pf.wrapper),
        )
        assert_series_equal(
            pf_grouped.position_coverage,
            vbt.Portfolio.get_position_coverage(
                assets=pf_grouped.assets, wrapper=pf_grouped.wrapper
            ),
        )
        assert_series_equal(
            pf_shared.position_coverage,
            vbt.Portfolio.get_position_coverage(assets=pf_shared.assets, wrapper=pf_shared.wrapper),
        )
        assert_series_equal(
            pf.get_position_coverage(jitted=dict(parallel=True)),
            pf.get_position_coverage(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            pf.get_position_coverage(chunked=True),
            pf.get_position_coverage(chunked=False),
        )
        assert_series_equal(
            pf_grouped.get_position_coverage(jitted=dict(parallel=True)),
            pf_grouped.get_position_coverage(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            pf_grouped.get_position_coverage(chunked=True),
            pf_grouped.get_position_coverage(chunked=False),
        )

        sim_range_result = pd.Series(
            [0.6666666666666666, 1.0, 0.6666666666666666], index=close_na.columns
        ).rename("position_coverage")
        assert_series_equal(pf.get_position_coverage(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.Series([1.0, 0.6666666666666666], index=group_columns).rename(
            "position_coverage"
        )
        assert_series_equal(
            pf_grouped.get_position_coverage(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf_shared.get_position_coverage(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.Series([0.6666666666666666, 1.0, 1.0], index=close_na.columns).rename(
            "position_coverage"
        )
        assert_series_equal(
            pf.get_position_coverage(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.Series([1.0, 1.0], index=group_columns).rename(
            "position_coverage"
        )
        assert_series_equal(
            pf_grouped.get_position_coverage(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf_shared.get_position_coverage(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf.get_position_coverage(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).position_coverage,
        )
        assert_series_equal(
            pf_grouped.get_position_coverage(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).position_coverage,
        )
        assert_series_equal(
            pf_shared.get_position_coverage(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).position_coverage,
        )

    def test_cash_flow(self):
        assert_frame_equal(
            pf.get_cash_flow(free=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, -1.1201],
                        [-0.30402, -0.29998, -0.3040200000000002],
                        [-2.5057000000000005, 0.0, 2.8402999999999996],
                        [-0.4999599999999999, -0.11204000000000003, 0.29204000000000035],
                        [0.9375, -5.0995, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, -1.1201],
                    [-0.30402, 0.09602000000000001, -0.30402],
                    [2.8402999999999996, 0.0, 2.8402999999999996],
                    [0.29204, -0.50804, 0.29204],
                    [-5.2005, 4.8005, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.cash_flow, result)
        assert_frame_equal(pf_grouped.get_cash_flow(group_by=False), result)
        assert_frame_equal(pf_shared.get_cash_flow(group_by=False), result)
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, -1.1201],
                    [-0.20800000000000002, -0.30402],
                    [2.8402999999999996, 2.8402999999999996],
                    [-0.21600000000000003, 0.29204],
                    [-0.39999999999999947, 0.0],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_cash_flow(group_by=group_by), result)
        assert_frame_equal(pf_grouped.cash_flow, result)
        assert_frame_equal(pf_shared.cash_flow, result)
        assert_frame_equal(
            pf.cash_flow, vbt.Portfolio.get_cash_flow(orders=pf.orders, wrapper=pf.wrapper)
        )
        assert_frame_equal(
            pf_grouped.cash_flow,
            vbt.Portfolio.get_cash_flow(orders=pf_grouped.orders, wrapper=pf_grouped.wrapper),
        )
        assert_frame_equal(
            pf_shared.cash_flow,
            vbt.Portfolio.get_cash_flow(orders=pf_shared.orders, wrapper=pf_shared.wrapper),
        )
        assert_frame_equal(
            pf.get_cash_flow(jitted=dict(parallel=True)),
            pf.get_cash_flow(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_cash_flow(chunked=False), pf.get_cash_flow(chunked=True))
        assert_frame_equal(
            pf_grouped.get_cash_flow(jitted=dict(parallel=True)),
            pf_grouped.get_cash_flow(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_grouped.get_cash_flow(chunked=False), pf_grouped.get_cash_flow(chunked=True)
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [-0.30402, 0.09602000000000001, -0.30402],
                [2.8402999999999996, 0.0, 2.8402999999999996],
                [0.29204, -0.50804, 0.29204],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_cash_flow(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [-0.20800000000000002, -0.30402],
                [2.8402999999999996, 2.8402999999999996],
                [-0.21600000000000003, 0.29204],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_cash_flow(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_cash_flow(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf.get_cash_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf_grouped.get_cash_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_cash_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_cash_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).cash_flow,
        )
        assert_frame_equal(
            pf_grouped.get_cash_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).cash_flow,
        )
        assert_frame_equal(
            pf_shared.get_cash_flow(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).cash_flow,
        )

    def test_init_cash(self):
        assert_series_equal(
            pf.init_cash,
            pd.Series(np.array([100.0, 100.0, 100.0]), index=close_na.columns).rename("init_cash"),
        )
        assert_series_equal(
            pf_grouped.get_init_cash(group_by=False),
            pd.Series(np.array([100.0, 100.0, 100.0]), index=close_na.columns).rename("init_cash"),
        )
        assert_series_equal(
            pf_shared.get_init_cash(group_by=False),
            pd.Series(np.array([200.0, 200.0, 100.0]), index=close_na.columns).rename("init_cash"),
        )
        assert_series_equal(
            pf_shared.get_init_cash(group_by=False, split_shared=True),
            pd.Series(np.array([100.0, 100.0, 100.0]), index=close_na.columns).rename("init_cash"),
        )
        result = pd.Series(
            np.array([200.0, 100.0]),
            group_columns,
        ).rename("init_cash")
        assert_series_equal(pf.get_init_cash(group_by=group_by), result)
        assert_series_equal(pf_grouped.init_cash, result)
        assert_series_equal(pf_shared.init_cash, result)
        assert_series_equal(
            vbt.Portfolio.from_orders(
                close_na, 1000.0, init_cash=InitCashMode.Auto, group_by=None
            ).init_cash,
            pd.Series(np.array([14000.0, 12000.0, 10000.0]), index=close_na.columns).rename(
                "init_cash"
            ),
        )
        assert_series_equal(
            vbt.Portfolio.from_orders(
                close_na, 1000.0, init_cash=InitCashMode.Auto, group_by=group_by
            ).init_cash,
            pd.Series(
                np.array([26000.0, 10000.0]),
                index=group_columns,
            ).rename("init_cash"),
        )
        assert_series_equal(
            vbt.Portfolio.from_orders(
                close_na,
                1000.0,
                init_cash=InitCashMode.Auto,
                group_by=group_by,
                cash_sharing=True,
            ).init_cash,
            pd.Series(
                np.array([26000.0, 10000.0]),
                index=group_columns,
            ).rename("init_cash"),
        )
        assert_series_equal(
            vbt.Portfolio.from_orders(
                close_na, 1000.0, init_cash=InitCashMode.AutoAlign, group_by=None
            ).init_cash,
            pd.Series(np.array([14000.0, 14000.0, 14000.0]), index=close_na.columns).rename(
                "init_cash"
            ),
        )
        assert_series_equal(
            vbt.Portfolio.from_orders(
                close_na, 1000.0, init_cash=InitCashMode.AutoAlign, group_by=group_by
            ).init_cash,
            pd.Series(
                np.array([26000.0, 26000.0]),
                index=group_columns,
            ).rename("init_cash"),
        )
        assert_series_equal(
            vbt.Portfolio.from_orders(
                close_na,
                1000.0,
                init_cash=InitCashMode.AutoAlign,
                group_by=group_by,
                cash_sharing=True,
            ).init_cash,
            pd.Series(
                np.array([26000.0, 26000.0]),
                index=group_columns,
            ).rename("init_cash"),
        )
        assert_series_equal(
            pf.init_cash,
            vbt.Portfolio.get_init_cash(
                init_cash_raw=pf._init_cash,
                cash_sharing=pf.cash_sharing,
                free_cash_flow=pf.free_cash_flow,
                wrapper=pf.wrapper,
            ),
        )
        assert_series_equal(
            pf_grouped.init_cash,
            vbt.Portfolio.get_init_cash(
                init_cash_raw=pf_grouped._init_cash,
                cash_sharing=pf_grouped.cash_sharing,
                free_cash_flow=pf_grouped.free_cash_flow,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_series_equal(
            pf_shared.init_cash,
            vbt.Portfolio.get_init_cash(
                init_cash_raw=pf_shared._init_cash,
                cash_sharing=pf_shared.cash_sharing,
                free_cash_flow=pf_shared.free_cash_flow,
                wrapper=pf_shared.wrapper,
            ),
        )
        pf2 = vbt.Portfolio.from_orders(
            close_na,
            1000.0,
            init_cash=InitCashMode.AutoAlign,
            group_by=group_by,
            cash_sharing=True,
        )
        assert_series_equal(
            pf2.init_cash,
            type(pf2).get_init_cash(
                init_cash_raw=pf2._init_cash,
                cash_sharing=pf2.cash_sharing,
                free_cash_flow=pf2.free_cash_flow,
                wrapper=pf2.wrapper,
            ),
        )
        assert_series_equal(
            vbt.Portfolio.from_orders(close_na, 1000.0, init_cash=InitCashMode.Auto).get_init_cash(
                jitted=dict(parallel=True)
            ),
            vbt.Portfolio.from_orders(close_na, 1000.0, init_cash=InitCashMode.Auto).get_init_cash(
                jitted=dict(parallel=False)
            ),
        )
        assert_series_equal(
            vbt.Portfolio.from_orders(close_na, 1000.0, init_cash=InitCashMode.Auto).get_init_cash(
                chunked=True
            ),
            vbt.Portfolio.from_orders(close_na, 1000.0, init_cash=InitCashMode.Auto).get_init_cash(
                chunked=False
            ),
        )
        assert_series_equal(
            pf.get_init_cash(weights=[0.5, 1.0, 1.5]),
            pd.Series(np.array([50.0, 100.0, 150.0]), index=close_na.columns).rename("init_cash"),
        )
        assert_series_equal(
            pf_grouped.get_init_cash(weights=[0.5, 1.0, 1.5], group_by=False),
            pd.Series(np.array([50.0, 100.0, 150.0]), index=close_na.columns).rename("init_cash"),
        )
        assert_series_equal(
            pf_shared.get_init_cash(weights=[0.5, 1.0, 1.5], group_by=False),
            pd.Series(np.array([100.0, 200.0, 150.0]), index=close_na.columns).rename("init_cash"),
        )
        assert_series_equal(
            pf_shared.get_init_cash(weights=[0.5, 1.0, 1.5], group_by=False, split_shared=True),
            pd.Series(np.array([50.0, 100.0, 150.0]), index=close_na.columns).rename("init_cash"),
        )
        result = pd.Series(
            np.array([150.0, 150.0]),
            group_columns,
        ).rename("init_cash")
        assert_series_equal(pf.get_init_cash(weights=[0.5, 1.0, 1.5], group_by=group_by), result)
        assert_series_equal(pf_grouped.get_init_cash(weights=[0.5, 1.0, 1.5]), result)
        assert_series_equal(pf_shared.get_init_cash(weights=[0.5, 1.0, 1.5]), result)

    def test_cash_deposits(self):
        assert_frame_equal(
            pf.cash_deposits,
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [100.0, 100.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_grouped.get_cash_deposits(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [100.0, 100.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_shared.get_cash_deposits(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [200.0, 200.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_shared.get_cash_deposits(group_by=False, split_shared=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [100.0, 100.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array([[0.0, 0.0], [0.0, 0.0], [200.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_cash_deposits(group_by=group_by), result)
        assert_frame_equal(pf_grouped.cash_deposits, result)
        assert_frame_equal(pf_shared.cash_deposits, result)
        assert_frame_equal(
            pf.cash_deposits,
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=pf._cash_deposits,
                cash_sharing=pf.cash_sharing,
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.cash_deposits,
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=pf_grouped._cash_deposits,
                cash_sharing=pf_grouped.cash_sharing,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.cash_deposits,
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=pf_shared._cash_deposits,
                cash_sharing=pf_shared.cash_sharing,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_cash_deposits(jitted=dict(parallel=True)),
            pf.get_cash_deposits(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_cash_deposits(chunked=True), pf.get_cash_deposits(chunked=False))
        assert_frame_equal(
            pf_grouped.get_cash_deposits(jitted=dict(parallel=True)),
            pf_grouped.get_cash_deposits(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_grouped.get_cash_deposits(chunked=True),
            pf_grouped.get_cash_deposits(chunked=False),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=1,
                keep_flex=True,
                cash_sharing=True,
                wrapper=pf_grouped.wrapper,
            ),
            np.array([[1]]),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=1,
                keep_flex=True,
                cash_sharing=False,
                wrapper=pf_grouped.wrapper,
            ),
            np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=1,
                keep_flex=True,
                cash_sharing=False,
                wrapper=pf.wrapper,
            ),
            np.array([[1]]),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=1, keep_flex=True, cash_sharing=True, wrapper=pf.wrapper
            ),
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 0.0],
                [100.0, 100.0, 0.0],
                [0.0, 0.0, 0.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_cash_deposits(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [[np.nan, np.nan], [0.0, 0.0], [200.0, 0.0], [0.0, 0.0], [np.nan, np.nan]],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_cash_deposits(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_cash_deposits(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf.get_cash_deposits(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            pf_grouped.get_cash_deposits(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_cash_deposits(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_cash_deposits(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).cash_deposits,
        )
        assert_frame_equal(
            pf_grouped.get_cash_deposits(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).cash_deposits,
        )
        assert_frame_equal(
            pf_shared.get_cash_deposits(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).cash_deposits,
        )
        assert_frame_equal(
            pf.get_cash_deposits(weights=[0.5, 1.0, 1.5]),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [50.0, 100.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_grouped.get_cash_deposits(weights=[0.5, 1.0, 1.5], group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [50.0, 100.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_shared.get_cash_deposits(weights=[0.5, 1.0, 1.5], group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [100.0, 200.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_shared.get_cash_deposits(weights=[0.5, 1.0, 1.5], group_by=False, split_shared=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [50.0, 100.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array([[0.0, 0.0], [0.0, 0.0], [150.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_cash_deposits(weights=[0.5, 1.0, 1.5], group_by=group_by), result)
        assert_frame_equal(pf_grouped.get_cash_deposits(weights=[0.5, 1.0, 1.5]), result)
        assert_frame_equal(pf_shared.get_cash_deposits(weights=[0.5, 1.0, 1.5]), result)

    def test_cash_earnings(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.cash_earnings, result)
        assert_frame_equal(pf_grouped.get_cash_earnings(group_by=False), result)
        assert_frame_equal(pf_shared.get_cash_earnings(group_by=False), result)
        result = pd.DataFrame(
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_cash_earnings(group_by=group_by), result)
        assert_frame_equal(pf_grouped.cash_earnings, result)
        assert_frame_equal(pf_shared.cash_earnings, result)
        assert_frame_equal(
            pf.cash_earnings,
            vbt.Portfolio.get_cash_earnings(
                cash_earnings_raw=pf._cash_earnings, wrapper=pf.wrapper
            ),
        )
        assert_frame_equal(
            pf_grouped.cash_earnings,
            vbt.Portfolio.get_cash_earnings(
                cash_earnings_raw=pf_grouped._cash_earnings, wrapper=pf_grouped.wrapper
            ),
        )
        assert_frame_equal(
            pf_shared.cash_earnings,
            vbt.Portfolio.get_cash_earnings(
                cash_earnings_raw=pf_shared._cash_earnings, wrapper=pf_shared.wrapper
            ),
        )
        assert_frame_equal(
            pf.get_cash_earnings(jitted=dict(parallel=True)),
            pf.get_cash_earnings(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf.get_cash_earnings(chunked=True),
            pf.get_cash_earnings(chunked=False),
        )
        assert_frame_equal(
            pf_grouped.get_cash_earnings(jitted=dict(parallel=True)),
            pf_grouped.get_cash_earnings(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_grouped.get_cash_earnings(chunked=True),
            pf_grouped.get_cash_earnings(chunked=False),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_earnings(
                cash_earnings_raw=1, keep_flex=True, wrapper=pf.wrapper
            ),
            np.array([[1]]),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_earnings(
                cash_earnings_raw=1, keep_flex=True, wrapper=pf_grouped.wrapper
            ),
            np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]),
        )

        new_pf = pf.replace(cash_earnings=1)
        new_pf_grouped = pf_grouped.replace(cash_earnings=1)
        new_pf_shared = pf_shared.replace(cash_earnings=1)
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(new_pf.get_cash_earnings(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [[np.nan, np.nan], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [np.nan, np.nan]],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            new_pf_grouped.get_cash_earnings(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            new_pf_shared.get_cash_earnings(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            new_pf.get_cash_earnings(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_frame_equal(
            new_pf_grouped.get_cash_earnings(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            new_pf_shared.get_cash_earnings(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            new_pf.get_cash_earnings(sim_start=1, sim_end=4, rec_sim_range=True),
            new_pf.replace(sim_start=1, sim_end=4).cash_earnings,
        )
        assert_frame_equal(
            new_pf_grouped.get_cash_earnings(sim_start=1, sim_end=4, rec_sim_range=True),
            new_pf_grouped.replace(sim_start=1, sim_end=4).cash_earnings,
        )
        assert_frame_equal(
            new_pf_shared.get_cash_earnings(sim_start=1, sim_end=4, rec_sim_range=True),
            new_pf_shared.replace(sim_start=1, sim_end=4).cash_earnings,
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.5, 1.0, 1.5],
                    [0.5, 1.0, 1.5],
                    [0.5, 1.0, 1.5],
                    [0.5, 1.0, 1.5],
                    [0.5, 1.0, 1.5],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_cash_earnings(cash_earnings_raw=1, weights=[0.5, 1.0, 1.5]),
            result,
        )
        assert_frame_equal(
            pf_grouped.get_cash_earnings(
                cash_earnings_raw=1, weights=[0.5, 1.0, 1.5], group_by=False
            ),
            result,
        )
        assert_frame_equal(
            pf_shared.get_cash_earnings(
                cash_earnings_raw=1, weights=[0.5, 1.0, 1.5], group_by=False
            ),
            result,
        )
        result = pd.DataFrame(
            np.array([[1.5, 1.5], [1.5, 1.5], [1.5, 1.5], [1.5, 1.5], [1.5, 1.5]]),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf.get_cash_earnings(cash_earnings_raw=1, weights=[0.5, 1.0, 1.5], group_by=group_by),
            result,
        )
        assert_frame_equal(
            pf_grouped.get_cash_earnings(cash_earnings_raw=1, weights=[0.5, 1.0, 1.5]),
            result,
        )
        assert_frame_equal(
            pf_shared.get_cash_earnings(cash_earnings_raw=1, weights=[0.5, 1.0, 1.5]),
            result,
        )

    def test_cash(self):
        assert_frame_equal(
            pf.get_cash(free=True),
            pd.DataFrame(
                np.array(
                    [
                        [100.0, 100.0, 98.8799],
                        [99.69598, 99.70002, 98.57588000000001],
                        [197.19028000000003, 199.70002, 101.41618000000001],
                        [196.69032000000004, 199.58798, 101.70822000000001],
                        [197.62782000000004, 194.48847999999998, 101.70822000000001],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [100.0, 100.0, 98.8799],
                    [99.69598, 100.09602, 98.57588000000001],
                    [202.53628000000003, 200.09602, 101.41618000000001],
                    [202.82832000000002, 199.58798000000002, 101.70822000000001],
                    [197.62782, 204.38848000000002, 101.70822000000001],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.cash, result)
        assert_frame_equal(pf_grouped.get_cash(group_by=False), result)
        assert_frame_equal(
            pf_shared.get_cash(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [200.0, 200.0, 98.8799],
                        [199.69598, 200.09602, 98.57588000000001],
                        [402.53628, 400.09602, 101.41618000000001],
                        [402.82831999999996, 399.58798, 101.70822000000001],
                        [397.62782, 404.38848, 101.70822000000001],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [200.0, 98.8799],
                    [199.792, 98.57588000000001],
                    [402.63230000000004, 101.41618000000001],
                    [402.41630000000004, 101.70822000000001],
                    [402.01630000000006, 101.70822000000001],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_cash(group_by=group_by), result)
        assert_frame_equal(pf_grouped.cash, result)
        assert_frame_equal(pf_shared.cash, result)
        assert_frame_equal(
            pf.cash,
            vbt.Portfolio.get_cash(
                init_cash=pf.init_cash,
                cash_deposits=pf.cash_deposits,
                cash_flow=pf.cash_flow,
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.cash,
            vbt.Portfolio.get_cash(
                init_cash=pf_grouped.init_cash,
                cash_deposits=pf_grouped.cash_deposits,
                cash_flow=pf_grouped.cash_flow,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_cash(jitted=dict(parallel=True)), pf.get_cash(jitted=dict(parallel=False))
        )
        assert_frame_equal(pf.get_cash(chunked=True), pf.get_cash(chunked=False))
        assert_frame_equal(
            pf_grouped.get_cash(jitted=dict(parallel=True)),
            pf_grouped.get_cash(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf_grouped.get_cash(chunked=True), pf_grouped.get_cash(chunked=False))

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [99.69598, 100.09602, 99.69598],
                [202.53628000000003, 200.09602, 102.53628],
                [202.82832000000002, 199.58798000000002, 102.82832],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_cash(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [199.792, 99.69598],
                [402.63230000000004, 102.53628],
                [402.41630000000004, 102.82832],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf_grouped.get_cash(sim_start=1, sim_end=4), sim_range_result_grouped)
        assert_frame_equal(pf_shared.get_cash(sim_start=1, sim_end=4), sim_range_result_grouped)
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [99.69598, 100.09602, 99.69598],
                [202.53628000000003, 200.09602, 102.53628],
                [202.82832000000002, 199.58798000000002, 102.82832],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_cash(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [199.792, 99.69598],
                [402.63230000000004, 102.53628],
                [402.41630000000004, 102.82832],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_cash(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_cash(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_cash(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).cash,
        )
        assert_frame_equal(
            pf_grouped.get_cash(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).cash,
        )
        assert_frame_equal(
            pf_shared.get_cash(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).cash,
        )

    def test_init_position_value(self):
        result = pd.Series(np.array([2.0, -1.0, 0.0]), index=close_na.columns).rename(
            "init_position_value"
        )
        assert_series_equal(pf.init_position_value, result)
        result = pd.Series(np.array([1.0, 0.0]), index=group_columns).rename("init_position_value")
        assert_series_equal(pf_grouped.init_position_value, result)
        assert_series_equal(pf_shared.init_position_value, result)
        assert_series_equal(
            pf.init_position_value,
            vbt.Portfolio.get_init_position_value(
                init_price=pf.init_price,
                init_position=pf.init_position,
                wrapper=pf.wrapper,
            ),
        )
        assert_series_equal(
            pf_grouped.init_position_value,
            vbt.Portfolio.get_init_position_value(
                init_price=pf_grouped.init_price,
                init_position=pf_grouped.init_position,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_series_equal(
            pf_shared.init_position_value,
            vbt.Portfolio.get_init_position_value(
                init_price=pf_shared.init_price,
                init_position=pf_shared.init_position,
                wrapper=pf_shared.wrapper,
            ),
        )

    def test_init_value(self):
        assert_series_equal(
            pf.init_value,
            pd.Series(np.array([102.0, 99.0, 100.0]), index=close_na.columns).rename("init_value"),
        )
        assert_series_equal(
            pf_grouped.get_init_value(group_by=False),
            pd.Series(np.array([102.0, 99.0, 100.0]), index=close_na.columns).rename("init_value"),
        )
        assert_series_equal(
            pf_shared.get_init_value(group_by=False),
            pd.Series(np.array([202.0, 199.0, 100.0]), index=close_na.columns).rename("init_value"),
        )
        result = pd.Series(
            np.array([201.0, 100.0]),
            group_columns,
        ).rename("init_value")
        assert_series_equal(pf.get_init_value(group_by=group_by), result)
        assert_series_equal(pf_grouped.init_value, result)
        assert_series_equal(pf_shared.init_value, result)
        assert_series_equal(
            pf.get_init_value(jitted=dict(parallel=True)),
            pf.get_init_value(jitted=dict(parallel=False)),
        )
        assert_series_equal(pf.get_init_value(chunked=True), pf.get_init_value(chunked=False))
        assert_series_equal(
            pf.init_value,
            vbt.Portfolio.get_init_value(
                init_position_value=pf.init_position_value,
                init_cash=pf.init_cash,
                split_shared=False,
                wrapper=pf.wrapper,
            ),
        )
        assert_series_equal(
            pf.init_value,
            vbt.Portfolio.get_init_value(
                init_position_value=pf.init_position_value,
                init_cash=pf.init_cash,
                split_shared=True,
                wrapper=pf.wrapper,
            ),
        )
        assert_series_equal(
            pf_grouped.init_value,
            vbt.Portfolio.get_init_value(
                init_position_value=pf_grouped.init_position_value,
                init_cash=pf_grouped.init_cash,
                split_shared=False,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_series_equal(
            pf_shared.init_value,
            vbt.Portfolio.get_init_value(
                init_position_value=pf_shared.init_position_value,
                init_cash=pf_shared.init_cash,
                split_shared=False,
                wrapper=pf_shared.wrapper,
            ),
        )

    def test_input_value(self):
        assert_series_equal(
            pf.input_value,
            pd.Series(np.array([202.0, 199.0, 100.0]), index=close_na.columns).rename(
                "input_value"
            ),
        )
        assert_series_equal(
            pf_grouped.get_input_value(group_by=False),
            pd.Series(np.array([202.0, 199.0, 100.0]), index=close_na.columns).rename(
                "input_value"
            ),
        )
        assert_series_equal(
            pf_shared.get_input_value(group_by=False),
            pd.Series(np.array([402.0, 399.0, 100.0]), index=close_na.columns).rename(
                "input_value"
            ),
        )
        result = pd.Series(
            np.array([401.0, 100.0]),
            group_columns,
        ).rename("input_value")
        assert_series_equal(pf.get_input_value(group_by=group_by), result)
        assert_series_equal(pf_grouped.input_value, result)
        assert_series_equal(pf_shared.input_value, result)
        assert_series_equal(
            pf.get_input_value(jitted=dict(parallel=True)),
            pf.get_input_value(jitted=dict(parallel=False)),
        )
        assert_series_equal(pf.get_input_value(chunked=True), pf.get_input_value(chunked=False))
        assert_series_equal(
            pf.input_value,
            vbt.Portfolio.get_input_value(
                total_cash_deposits=pf.total_cash_deposits,
                init_value=pf.init_value,
                wrapper=pf.wrapper,
            ),
        )
        assert_series_equal(
            pf_grouped.input_value,
            vbt.Portfolio.get_input_value(
                total_cash_deposits=pf_grouped.total_cash_deposits,
                init_value=pf_grouped.init_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_series_equal(
            pf_shared.input_value,
            vbt.Portfolio.get_input_value(
                total_cash_deposits=pf_shared.total_cash_deposits,
                init_value=pf_shared.init_value,
                wrapper=pf_shared.wrapper,
            ),
        )

        sim_range_result = pd.Series([202.0, 199.0, 100.0], index=close_na.columns).rename(
            "input_value"
        )
        assert_series_equal(pf.get_input_value(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.Series([401.0, 100.0], index=group_columns).rename(
            "input_value"
        )
        assert_series_equal(
            pf_grouped.get_input_value(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf_shared.get_input_value(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf.get_input_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_series_equal(
            pf_grouped.get_input_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf_shared.get_input_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf.get_input_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).input_value,
        )
        assert_series_equal(
            pf_grouped.get_input_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).input_value,
        )
        assert_series_equal(
            pf_shared.get_input_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).input_value,
        )

    def test_asset_value(self):
        assert_frame_equal(
            pf.get_asset_value(direction="longonly"),
            pd.DataFrame(
                np.array(
                    [
                        [2.0, 0.0, 1.0],
                        [2.2, 0.0, 2.2],
                        [0.30000000000000027, 0.0, 0.30000000000000027],
                        [0.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.get_asset_value(direction="shortonly"),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 2.2, 0.0],
                        [0.0, 2.2, 0.0],
                        [0.0, 4.0, 0.0],
                        [0.0, 10.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [2.0, -1.0, 1.0],
                    [2.2, -2.2, 2.2],
                    [0.30000000000000027, -2.2, 0.30000000000000027],
                    [0.0, -4.0, 0.0],
                    [5.0, -10.0, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.asset_value, result)
        assert_frame_equal(pf_grouped.get_asset_value(group_by=False), result)
        assert_frame_equal(pf_shared.get_asset_value(group_by=False), result)
        result = pd.DataFrame(
            np.array(
                [[1.0, 1.0], [0.0, 2.2], [-1.9, 0.30000000000000027], [-4.0, 0.0], [-5.0, 0.0]]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_asset_value(group_by=group_by), result)
        assert_frame_equal(pf_grouped.asset_value, result)
        assert_frame_equal(pf_shared.asset_value, result)
        assert_frame_equal(
            pf.asset_value,
            vbt.Portfolio.get_asset_value(
                close=pf.filled_close, assets=pf.assets, wrapper=pf.wrapper
            ),
        )
        assert_frame_equal(
            pf_grouped.asset_value,
            vbt.Portfolio.get_asset_value(
                close=pf_grouped.filled_close,
                assets=pf_grouped.assets,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.asset_value,
            vbt.Portfolio.get_asset_value(
                close=pf_shared.filled_close,
                assets=pf_shared.assets,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_asset_value(jitted=dict(parallel=True)),
            pf.get_asset_value(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf.get_asset_value(chunked=True),
            pf.get_asset_value(chunked=False),
        )
        assert_frame_equal(
            pf_grouped.get_asset_value(jitted=dict(parallel=True)),
            pf_grouped.get_asset_value(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_grouped.get_asset_value(chunked=True),
            pf_grouped.get_asset_value(chunked=False),
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [2.2, -2.2, 2.2],
                [0.30000000000000027, -2.2, 0.30000000000000027],
                [0.0, -4.0, 0.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_asset_value(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [0.0, 2.2],
                [-1.9, 0.30000000000000027],
                [-4.0, 0.0],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_asset_value(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_asset_value(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [2.2, -2.2, 0.2],
                [0.30000000000000027, -2.2, -2.7],
                [0.0, -4.0, -4.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_asset_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [[np.nan, np.nan], [0.0, 0.2], [-1.9, -2.7], [-4.0, -4.0], [np.nan, np.nan]],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_asset_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_asset_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_asset_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).asset_value,
        )
        assert_frame_equal(
            pf_grouped.get_asset_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).asset_value,
        )
        assert_frame_equal(
            pf_shared.get_asset_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).asset_value,
        )

    def test_gross_exposure(self):
        assert_frame_equal(
            pf.get_gross_exposure(direction="longonly"),
            pd.DataFrame(
                np.array(
                    [
                        [0.0196078431372549, 0.0, 0.010012024441354066],
                        [0.021590645676110087, 0.0, 0.021830620581035857],
                        [0.0014790253499028883, 0.0, 0.002949383274126105],
                        [0.0, 0.0, 0.0],
                        [0.024675782427111933, 0.0, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.get_gross_exposure(direction="shortonly"),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.010101010101010102, 0.0],
                        [0.0, 0.022472823716428926, 0.0],
                        [0.0, 0.01111694919382411, 0.0],
                        [0.0, 0.020451154513687394, 0.0],
                        [0.0, 0.05144337771456415, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0196078431372549, 0.010101010101010102, 0.010012024441354066],
                    [0.021590645676110087, 0.022472823716428926, 0.021830620581035857],
                    [0.0014790253499028883, 0.01111694919382411, 0.002949383274126105],
                    [0.0, 0.020451154513687394, 0.0],
                    [0.024675782427111933, 0.05144337771456415, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.gross_exposure, result)
        assert_frame_equal(pf_grouped.get_gross_exposure(group_by=False), result)
        assert_frame_equal(
            pf_shared.get_gross_exposure(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.009900990099009901, 0.005025125628140704, 0.010012024441354066],
                        [0.010896700370160913, 0.01111694919382411, 0.021830620581035857],
                        [0.0007447194180226276, 0.005529082698540187, 0.002949383274126105],
                        [0.0, 0.010111530689077055, 0.0],
                        [0.01241841659128274, 0.0253557101870724, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.014925373134328358, 0.010012024441354066],
                    [0.022022903819972774, 0.021830620581035857],
                    [0.00623857872200469, 0.002949383274126105],
                    [0.010039749879711245, 0.0],
                    [0.03778182407120312, 0.0],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_gross_exposure(group_by=group_by), result)
        assert_frame_equal(pf_grouped.gross_exposure, result)
        assert_frame_equal(pf_shared.gross_exposure, result)
        assert_frame_equal(
            pf.gross_exposure,
            vbt.Portfolio.get_gross_exposure(
                asset_value=pf.long_asset_value + pf.short_asset_value,
                value=pf.get_value(),
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.gross_exposure,
            vbt.Portfolio.get_gross_exposure(
                asset_value=pf_grouped.long_asset_value + pf_grouped.short_asset_value,
                value=pf_grouped.get_value(),
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.gross_exposure,
            vbt.Portfolio.get_gross_exposure(
                asset_value=pf_shared.long_asset_value + pf_shared.short_asset_value,
                value=pf_shared.get_value(),
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_gross_exposure(jitted=dict(parallel=True)),
            pf.get_gross_exposure(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf.get_gross_exposure(chunked=True),
            pf.get_gross_exposure(chunked=False),
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.021590645676110087, 0.022472823716428926, 0.021830620581035857],
                [0.0014790253499028883, 0.01111694919382411, 0.002949383274126105],
                [0.0, 0.020451154513687394, 0.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_gross_exposure(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [0.022022903819972774, 0.021830620581035857],
                [0.00623857872200469, 0.002949383274126105],
                [0.010039749879711245, 0.0],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_gross_exposure(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_gross_exposure(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.021590645676110087, 0.022472823716428926, 0.00200208256628545],
                [0.0014790253499028883, 0.01111694919382411, 0.02704427689012451],
                [0.0, 0.020451154513687394, 0.04047422843978325],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_gross_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [0.022022903819972774, 0.00200208256628545],
                [0.00623857872200469, 0.02704427689012451],
                [0.010039749879711245, 0.04047422843978325],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_gross_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_gross_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_gross_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).gross_exposure,
        )
        assert_frame_equal(
            pf_grouped.get_gross_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).gross_exposure,
        )
        assert_frame_equal(
            pf_shared.get_gross_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).gross_exposure,
        )

    def test_net_exposure(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0196078431372549, -0.010101010101010102, 0.010012024441354066],
                    [0.021590645676110087, -0.022472823716428926, 0.021830620581035857],
                    [0.0014790253499028883, -0.01111694919382411, 0.002949383274126105],
                    [0.0, -0.020451154513687394, 0.0],
                    [0.024675782427111933, -0.05144337771456415, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.net_exposure, result)
        assert_frame_equal(pf_grouped.get_net_exposure(group_by=False), result)
        assert_frame_equal(
            pf_shared.get_net_exposure(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.009900990099009901, -0.005025125628140704, 0.010012024441354066],
                        [0.010896700370160913, -0.01111694919382411, 0.021830620581035857],
                        [0.0007447194180226276, -0.005529082698540187, 0.002949383274126105],
                        [0.0, -0.010111530689077055, 0.0],
                        [0.01241841659128274, -0.0253557101870724, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.004975124378109453, 0.010012024441354066],
                    [0.0, 0.021830620581035857],
                    [-0.004741319828723563, 0.002949383274126105],
                    [-0.010039749879711245, 0.0],
                    [-0.012593941357067706, 0.0],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_net_exposure(group_by=group_by), result)
        assert_frame_equal(pf_grouped.net_exposure, result)
        assert_frame_equal(pf_shared.net_exposure, result)
        assert_frame_equal(
            pf.net_exposure,
            vbt.Portfolio.get_net_exposure(
                long_exposure=pf.get_gross_exposure(direction="longonly"),
                short_exposure=pf.get_gross_exposure(direction="shortonly"),
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.net_exposure,
            vbt.Portfolio.get_net_exposure(
                long_exposure=pf_grouped.get_gross_exposure(direction="longonly"),
                short_exposure=pf_grouped.get_gross_exposure(direction="shortonly"),
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.net_exposure,
            vbt.Portfolio.get_net_exposure(
                long_exposure=pf_shared.get_gross_exposure(direction="longonly"),
                short_exposure=pf_shared.get_gross_exposure(direction="shortonly"),
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_net_exposure(jitted=dict(parallel=True)),
            pf.get_net_exposure(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_net_exposure(chunked=True), pf.get_net_exposure(chunked=False))

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.021590645676110087, -0.022472823716428926, 0.021830620581035857],
                [0.0014790253499028883, -0.01111694919382411, 0.002949383274126105],
                [0.0, -0.020451154513687394, 0.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_net_exposure(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [0.0, 0.021830620581035857],
                [-0.004741319828723563, 0.002949383274126105],
                [-0.010039749879711245, 0.0],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_net_exposure(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_net_exposure(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.021590645676110087, -0.022472823716428926, 0.00200208256628545],
                [0.0014790253499028883, -0.01111694919382411, -0.02704427689012451],
                [0.0, -0.020451154513687394, -0.04047422843978325],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_net_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [0.0, 0.00200208256628545],
                [-0.004741319828723563, -0.02704427689012451],
                [-0.010039749879711245, -0.04047422843978325],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_net_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_net_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_net_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).net_exposure,
        )
        assert_frame_equal(
            pf_grouped.get_net_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).net_exposure,
        )
        assert_frame_equal(
            pf_shared.get_net_exposure(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).net_exposure,
        )

    def test_value(self):
        result = pd.DataFrame(
            np.array(
                [
                    [102.0, 99.0, 99.8799],
                    [101.89598000000001, 97.89602, 100.77588000000002],
                    [202.83628000000004, 197.89602000000002, 101.71618000000001],
                    [202.82832000000002, 195.58798000000002, 101.70822000000001],
                    [202.62782, 194.38848000000002, 101.70822000000001],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.value, result)
        assert_frame_equal(pf_grouped.get_value(group_by=False), result)
        assert_frame_equal(
            pf_shared.get_value(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [202.0, 199.0, 99.8799],
                        [201.89597999999998, 197.89602000000002, 100.77588000000002],
                        [402.83628, 397.89602, 101.71618000000001],
                        [402.82831999999996, 395.58798, 101.70822000000001],
                        [402.62782, 394.38848, 101.70822000000001],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [201.0, 99.8799],
                    [199.792, 100.77588000000002],
                    [400.73230000000007, 101.71618000000001],
                    [398.41630000000004, 101.70822000000001],
                    [397.01630000000006, 101.70822000000001],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_value(group_by=group_by), result)
        assert_frame_equal(pf_grouped.value, result)
        assert_frame_equal(pf_shared.value, result)
        assert_frame_equal(
            pf.value,
            vbt.Portfolio.get_value(cash=pf.cash, asset_value=pf.asset_value, wrapper=pf.wrapper),
        )
        assert_frame_equal(
            pf_grouped.value,
            vbt.Portfolio.get_value(
                cash=pf_grouped.cash,
                asset_value=pf_grouped.asset_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_value(jitted=dict(parallel=True)),
            pf.get_value(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_value(chunked=True), pf.get_value(chunked=False))

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [101.89598000000001, 97.89602, 100.77588000000002],
                [202.83628000000004, 197.89602000000002, 101.71618000000001],
                [202.82832000000002, 195.58798000000002, 101.70822000000001],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_value(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [199.792, 100.77588000000002],
                [400.73230000000007, 101.71618000000001],
                [398.41630000000004, 101.70822000000001],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf_grouped.get_value(sim_start=1, sim_end=4), sim_range_result_grouped)
        assert_frame_equal(pf_shared.get_value(sim_start=1, sim_end=4), sim_range_result_grouped)
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [101.89598000000001, 97.89602, 99.89598000000001],
                [202.83628000000004, 197.89602000000002, 99.83628],
                [202.82832000000002, 195.58798000000002, 98.82832],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [199.792, 99.89598000000001],
                [400.73230000000007, 99.83628],
                [398.41630000000004, 98.82832],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).value,
        )
        assert_frame_equal(
            pf_grouped.get_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).value,
        )
        assert_frame_equal(
            pf_shared.get_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).value,
        )

    def test_allocations(self):
        assert_frame_equal(
            pf.get_allocations(),
            pd.DataFrame(
                [
                    [0.0196078431372549, -0.010101010101010102, 0.010012024441354066],
                    [0.021590645676110087, -0.022472823716428926, 0.021830620581035857],
                    [0.0014790253499028883, -0.01111694919382411, 0.002949383274126105],
                    [0.0, -0.020451154513687394, 0.0],
                    [0.024675782427111933, -0.05144337771456415, 0.0],
                ],
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.get_allocations(direction="longonly"),
            pd.DataFrame(
                [
                    [0.0196078431372549, 0.0, 0.010012024441354066],
                    [0.021590645676110087, 0.0, 0.021830620581035857],
                    [0.0014790253499028883, 0.0, 0.002949383274126105],
                    [0.0, 0.0, 0.0],
                    [0.024675782427111933, 0.0, 0.0],
                ],
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.get_allocations(direction="shortonly"),
            pd.DataFrame(
                [
                    [0.0, 0.010101010101010102, 0.0],
                    [0.0, 0.022472823716428926, 0.0],
                    [0.0, 0.01111694919382411, 0.0],
                    [0.0, 0.020451154513687394, 0.0],
                    [0.0, 0.05144337771456415, 0.0],
                ],
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_grouped.allocations,
            pd.DataFrame(
                [
                    [0.009950248756218905, -0.004975124378109453, 0.010012024441354066],
                    [0.011011451909986387, -0.011011451909986387, 0.021830620581035857],
                    [0.0007486294466405633, -0.0054899492753641265, 0.002949383274126105],
                    [0.0, -0.010039749879711245, 0.0],
                    [0.012593941357067706, -0.025187882714135412, 0.0],
                ],
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_grouped.get_allocations(direction="longonly"),
            pd.DataFrame(
                [
                    [0.009950248756218905, 0.0, 0.010012024441354066],
                    [0.011011451909986387, 0.0, 0.021830620581035857],
                    [0.0007486294466405633, 0.0, 0.002949383274126105],
                    [0.0, 0.0, 0.0],
                    [0.012593941357067706, 0.0, 0.0],
                ],
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_grouped.get_allocations(direction="shortonly"),
            pd.DataFrame(
                [
                    [0.0, 0.004975124378109453, 0.0],
                    [0.0, 0.011011451909986387, 0.0],
                    [0.0, 0.0054899492753641265, 0.0],
                    [0.0, 0.010039749879711245, 0.0],
                    [0.0, 0.025187882714135412, 0.0],
                ],
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf_grouped.allocations,
            vbt.Portfolio.get_allocations(
                asset_value=pf_grouped.get_asset_value(group_by=False),
                value=pf_grouped.value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.allocations,
            vbt.Portfolio.get_allocations(
                asset_value=pf_grouped.get_asset_value(group_by=False),
                value=pf_shared.value,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.get_allocations(jitted=dict(parallel=True)),
            pf_shared.get_allocations(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_shared.get_allocations(chunked=False),
            pf_shared.get_allocations(chunked=True),
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.021590645676110087, -0.022472823716428926, 0.021830620581035857],
                [0.0014790253499028883, -0.01111694919382411, 0.002949383274126105],
                [0.0, -0.020451154513687394, 0.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_allocations(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.011011451909986387, -0.011011451909986387, 0.021830620581035857],
                [0.0007486294466405633, -0.0054899492753641265, 0.002949383274126105],
                [0.0, -0.010039749879711245, 0.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf_grouped.get_allocations(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_allocations(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.021590645676110087, -0.022472823716428926, 0.00200208256628545],
                [0.0014790253499028883, -0.01111694919382411, -0.02704427689012451],
                [0.0, -0.020451154513687394, -0.04047422843978325],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_allocations(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.011011451909986387, -0.011011451909986387, 0.00200208256628545],
                [0.0007486294466405633, -0.0054899492753641265, -0.02704427689012451],
                [0.0, -0.010039749879711245, -0.04047422843978325],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf_grouped.get_allocations(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_allocations(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_allocations(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).allocations,
        )
        assert_frame_equal(
            pf_grouped.get_allocations(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).allocations,
        )
        assert_frame_equal(
            pf_shared.get_allocations(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).allocations,
        )

    def test_total_profit(self):
        assert_series_equal(
            pf.total_profit, (pf.value.iloc[-1] - pf.input_value).rename("total_profit")
        )
        assert_series_equal(
            pf_grouped.get_total_profit(group_by=False),
            (
                pf_grouped.get_value(group_by=False).iloc[-1]
                - pf_grouped.get_input_value(group_by=False)
            ).rename("total_profit"),
        )
        assert_series_equal(
            pf_shared.get_total_profit(group_by=False),
            (
                pf_shared.get_value(group_by=False).iloc[-1]
                - pf_shared.get_input_value(group_by=False)
            ).rename("total_profit"),
        )
        assert_series_equal(
            pf.get_total_profit(group_by=group_by),
            (
                pf.get_value(group_by=group_by).iloc[-1] - pf.get_input_value(group_by=group_by)
            ).rename("total_profit"),
        )
        assert_series_equal(
            pf_grouped.total_profit,
            (pf_grouped.value.iloc[-1] - pf_grouped.input_value).rename("total_profit"),
        )
        assert_series_equal(
            pf_shared.total_profit,
            (pf_shared.value.iloc[-1] - pf_shared.input_value).rename("total_profit"),
        )
        assert_series_equal(
            pf.total_profit,
            vbt.Portfolio.get_total_profit(
                close=pf.filled_close,
                orders=pf.orders,
                init_position=pf.init_position,
                init_price=pf.init_price,
                wrapper=pf.wrapper,
            ),
        )
        assert_series_equal(
            pf_grouped.total_profit,
            vbt.Portfolio.get_total_profit(
                close=pf_grouped.filled_close,
                orders=pf_grouped.orders,
                init_position=pf.init_position,
                init_price=pf.init_price,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_series_equal(
            pf_shared.total_profit,
            vbt.Portfolio.get_total_profit(
                close=pf_shared.filled_close,
                orders=pf_shared.orders,
                init_position=pf.init_position,
                init_price=pf.init_price,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_series_equal(
            pf.get_total_profit(jitted=dict(parallel=True)),
            pf.get_total_profit(jitted=dict(parallel=False)),
        )
        assert_series_equal(pf.get_total_profit(chunked=True), pf.get_total_profit(chunked=False))

        sim_range_result = pd.Series(
            [0.8283199999999997, -3.41202, -1.1716800000000003], index=close_na.columns
        ).rename("total_profit")
        assert_series_equal(pf.get_total_profit(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.Series(
            [-2.5837000000000003, -1.1716800000000003], index=group_columns
        ).rename("total_profit")
        assert_series_equal(
            pf_grouped.get_total_profit(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf_shared.get_total_profit(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf.get_total_profit(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_series_equal(
            pf_grouped.get_total_profit(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf_shared.get_total_profit(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf.get_total_profit(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).total_profit,
        )
        assert_series_equal(
            pf_grouped.get_total_profit(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).total_profit,
        )
        assert_series_equal(
            pf_shared.get_total_profit(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).total_profit,
        )

    def test_final_value(self):
        assert_series_equal(pf.final_value, pf.value.iloc[-1].rename("final_value"))
        assert_series_equal(
            pf_grouped.get_final_value(group_by=False),
            pf_grouped.get_value(group_by=False).iloc[-1].rename("final_value"),
        )
        assert_series_equal(
            pf_shared.get_final_value(group_by=False),
            pf_shared.get_value(group_by=False).iloc[-1].rename("final_value"),
        )
        assert_series_equal(
            pf.get_final_value(group_by=group_by),
            pf.get_value(group_by=group_by).iloc[-1].rename("final_value"),
        )
        assert_series_equal(pf_grouped.final_value, pf_grouped.value.iloc[-1].rename("final_value"))
        assert_series_equal(pf_shared.final_value, pf_shared.value.iloc[-1].rename("final_value"))
        assert_series_equal(
            pf.final_value,
            vbt.Portfolio.get_final_value(
                input_value=pf.input_value, total_profit=pf.total_profit, wrapper=pf.wrapper
            ),
        )
        assert_series_equal(
            pf_grouped.final_value,
            vbt.Portfolio.get_final_value(
                input_value=pf_grouped.input_value,
                total_profit=pf_grouped.total_profit,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_series_equal(
            pf_shared.final_value,
            vbt.Portfolio.get_final_value(
                input_value=pf_shared.input_value,
                total_profit=pf_shared.total_profit,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_series_equal(
            pf.get_final_value(jitted=dict(parallel=True)),
            pf.get_final_value(jitted=dict(parallel=False)),
        )
        assert_series_equal(pf.get_final_value(chunked=True), pf.get_final_value(chunked=False))

        sim_range_result = pd.Series(
            [202.82832, 195.58798, 98.82832], index=close_na.columns
        ).rename("final_value")
        assert_series_equal(pf.get_final_value(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.Series([398.4163, 98.82832], index=group_columns).rename(
            "final_value"
        )
        assert_series_equal(
            pf_grouped.get_final_value(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf_shared.get_final_value(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf.get_final_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_series_equal(
            pf_grouped.get_final_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf_shared.get_final_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf.get_final_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).final_value,
        )
        assert_series_equal(
            pf_grouped.get_final_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).final_value,
        )
        assert_series_equal(
            pf_shared.get_final_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).final_value,
        )

    def test_total_return(self):
        assert_series_equal(
            pf.total_return,
            ((pf.value.iloc[-1] - pf.input_value) / pf.input_value).rename("total_return"),
        )
        assert_series_equal(
            pf_grouped.get_total_return(group_by=False),
            (
                (
                    pf_grouped.get_value(group_by=False).iloc[-1]
                    - pf_grouped.get_input_value(group_by=False)
                )
                / pf_grouped.get_input_value(group_by=False)
            ).rename("total_return"),
        )
        assert_series_equal(
            pf_shared.get_total_return(group_by=False),
            (
                (
                    pf_shared.get_value(group_by=False).iloc[-1]
                    - pf_shared.get_input_value(group_by=False)
                )
                / pf_shared.get_input_value(group_by=False)
            ).rename("total_return"),
        )
        assert_series_equal(
            pf.get_total_return(group_by=group_by),
            (
                (pf.get_value(group_by=group_by).iloc[-1] - pf.get_input_value(group_by=group_by))
                / pf.get_input_value(group_by=group_by)
            ).rename("total_return"),
        )
        assert_series_equal(
            pf_grouped.total_return,
            ((pf_grouped.value.iloc[-1] - pf_grouped.input_value) / pf_grouped.input_value).rename(
                "total_return"
            ),
        )
        assert_series_equal(
            pf_shared.total_return,
            ((pf_shared.value.iloc[-1] - pf_shared.input_value) / pf_shared.input_value).rename(
                "total_return"
            ),
        )
        assert_series_equal(
            pf.total_return,
            vbt.Portfolio.get_total_return(
                input_value=pf.input_value,
                total_profit=pf.total_profit,
                wrapper=pf.wrapper,
            ),
        )
        assert_series_equal(
            pf_grouped.total_return,
            vbt.Portfolio.get_total_return(
                input_value=pf_grouped.input_value,
                total_profit=pf_grouped.total_profit,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_series_equal(
            pf_shared.total_return,
            vbt.Portfolio.get_total_return(
                input_value=pf_shared.input_value,
                total_profit=pf_shared.total_profit,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_series_equal(
            pf.get_total_return(jitted=dict(parallel=True)),
            pf.get_total_return(jitted=dict(parallel=False)),
        )
        assert_series_equal(pf.get_total_return(chunked=True), pf.get_total_return(chunked=False))

        sim_range_result = pd.Series(
            [0.004100594059405939, -0.017145829145728642, -0.011716800000000003],
            index=close_na.columns,
        ).rename("total_return")
        assert_series_equal(pf.get_total_return(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.Series(
            [-0.006443142144638405, -0.011716800000000003], index=group_columns
        ).rename("total_return")
        assert_series_equal(
            pf_grouped.get_total_return(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf_shared.get_total_return(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf.get_total_return(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        assert_series_equal(
            pf_grouped.get_total_return(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf_shared.get_total_return(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf.get_total_return(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).total_return,
        )
        assert_series_equal(
            pf_grouped.get_total_return(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).total_return,
        )
        assert_series_equal(
            pf_shared.get_total_return(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).total_return,
        )

    def test_returns(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, -0.0012009999999999365],
                    [-0.0010198039215685425, -0.011151313131313203, 0.008970573658964502],
                    [0.00922803824056686, 0.0, 0.009330605696521761],
                    [-3.9243472617548996e-05, -0.01166289246241539, -7.825696954011722e-05],
                    [-0.0009885207351715245, -0.006132789959792009, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.returns, result)
        assert_frame_equal(pf_grouped.get_returns(group_by=False), result)
        assert_frame_equal(
            pf_shared.get_returns(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, -0.0012009999999999365],
                        [-0.0005149504950496028, -0.005547638190954667, 0.008970573658964502],
                        [0.0046573487991192685, 0.0, 0.009330605696521761],
                        [-1.9759888558263675e-05, -0.005800610923426691, -7.825696954011722e-05],
                        [-0.0004977306461471647, -0.003032195265386983, 0.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.get_returns(log_returns=True),
            np.log(result + 1),
        )
        assert_frame_equal(
            pf.get_returns(daily_returns=True),
            pf.returns.resample("D").apply(lambda x: (x + 1).prod() - 1),
        )
        assert_frame_equal(
            pf.get_returns(log_returns=True, daily_returns=True),
            pf.get_returns(log_returns=True).resample("D").apply(lambda x: x.cumsum()).fillna(0),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, -0.0012009999999999365],
                    [-0.006009950248756211, 0.008970573658964502],
                    [0.0047063946504367765, 0.009330605696521761],
                    [-0.005779419328065221, -7.825696954011722e-05],
                    [-0.003513912457898879, 0.0],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_returns(group_by=group_by), result)
        assert_frame_equal(pf_grouped.returns, result)
        assert_frame_equal(pf_shared.returns, result)
        assert_frame_equal(
            pf.returns,
            vbt.Portfolio.get_returns(
                init_value=pf.init_value,
                cash_deposits=pf.cash_deposits,
                value=pf.value,
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.returns,
            vbt.Portfolio.get_returns(
                init_value=pf_grouped.init_value,
                cash_deposits=pf_grouped.cash_deposits,
                value=pf_grouped.value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.returns,
            vbt.Portfolio.get_returns(
                init_value=pf_shared.init_value,
                cash_deposits=pf_shared.cash_deposits,
                value=pf_shared.value,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_returns(jitted=dict(parallel=True)),
            pf.get_returns(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_returns(chunked=True), pf.get_returns(chunked=False))
        assert_frame_equal(
            pf_grouped.get_returns(jitted=dict(parallel=True)),
            pf_grouped.get_returns(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_grouped.get_returns(chunked=True), pf_grouped.get_returns(chunked=False)
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [-0.0010198039215685425, -0.011151313131313203, 0.00775880000000015],
                [0.00922803824056686, 0.0, 0.009330605696521761],
                [-3.9243472617548996e-05, -0.01166289246241539, -7.825696954011722e-05],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_returns(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [-0.006009950248756211, 0.00775880000000015],
                [0.0047063946504367765, 0.009330605696521761],
                [-0.005779419328065221, -7.825696954011722e-05],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf_grouped.get_returns(sim_start=1, sim_end=4), sim_range_result_grouped)
        assert_frame_equal(pf_shared.get_returns(sim_start=1, sim_end=4), sim_range_result_grouped)
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [-0.0010198039215685425, -0.011151313131313203, -0.0010401999999999135],
                [0.00922803824056686, 0.0, -0.0005976216460362721],
                [-3.9243472617548996e-05, -0.01166289246241539, -0.01009612938302586],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [-0.006009950248756211, -0.0010401999999999135],
                [0.0047063946504367765, -0.0005976216460362721],
                [-0.005779419328065221, -0.01009612938302586],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).returns,
        )
        assert_frame_equal(
            pf_grouped.get_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).returns,
        )
        assert_frame_equal(
            pf_shared.get_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).returns,
        )

    def test_asset_pnl(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, -0.1201000000000001],
                    [-0.10401999999999978, -1.10398, 0.8959800000000002],
                    [0.9402999999999997, 0.0, 0.9402999999999997],
                    [-0.007960000000000245, -2.30804, -0.007960000000000245],
                    [-0.2004999999999999, -1.1994999999999996, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.asset_pnl, result)
        assert_frame_equal(pf_grouped.get_asset_pnl(group_by=False), result)
        assert_frame_equal(pf_shared.get_asset_pnl(group_by=False), result)
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, -0.1201000000000001],
                    [-1.208, 0.8959800000000002],
                    [0.9402999999999997, 0.9402999999999997],
                    [-2.3160000000000003, -0.007960000000000245],
                    [-1.3999999999999995, 0.0],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_asset_pnl(group_by=group_by), result)
        assert_frame_equal(pf_grouped.asset_pnl, result)
        assert_frame_equal(pf_shared.asset_pnl, result)
        assert_frame_equal(
            pf.asset_pnl,
            vbt.Portfolio.get_asset_pnl(
                init_position_value=pf.init_position_value,
                cash_flow=pf.cash_flow,
                asset_value=pf.asset_value,
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.asset_pnl,
            vbt.Portfolio.get_asset_pnl(
                init_position_value=pf_grouped.init_position_value,
                cash_flow=pf_grouped.cash_flow,
                asset_value=pf_grouped.asset_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.asset_pnl,
            vbt.Portfolio.get_asset_pnl(
                init_position_value=pf_shared.init_position_value,
                cash_flow=pf_shared.cash_flow,
                asset_value=pf_shared.asset_value,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_asset_pnl(jitted=dict(parallel=True)),
            pf.get_asset_pnl(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_asset_pnl(chunked=True), pf.get_asset_pnl(chunked=False))
        size = pd.Series([0.0, 0.5, -0.5, -0.5, 0.5, 1.0, -2.0, 2.0])
        pf2 = vbt.Portfolio.from_orders(1, size, fees=0.0)
        assert_series_equal(pf2.asset_pnl, pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        pf3 = vbt.Portfolio.from_orders(1, size, fees=0.01)
        assert_series_equal(
            pf3.asset_pnl,
            pd.Series(
                [
                    0.0,
                    -0.0050000000000000044,
                    -0.0050000000000000044,
                    -0.0050000000000000044,
                    -0.0050000000000000044,
                    -0.010000000000000009,
                    -0.020000000000000018,
                    -0.020000000000000018,
                ]
            ),
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [-0.10401999999999978, -1.1039800000000002, 1.8959800000000002],
                [0.9402999999999997, 0.0, 0.9402999999999997],
                [-0.007960000000000245, -2.30804, -0.007960000000000245],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_asset_pnl(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [-1.208, 1.8959800000000002],
                [0.9402999999999997, 0.9402999999999997],
                [-2.3160000000000003, -0.007960000000000245],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_asset_pnl(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_asset_pnl(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [-0.10401999999999978, -1.1039800000000002, -0.10402],
                [0.9402999999999997, 0.0, -0.059700000000000586],
                [-0.007960000000000245, -2.30804, -1.0079599999999997],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_asset_pnl(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [-1.208, -0.10402],
                [0.9402999999999997, -0.059700000000000586],
                [-2.3160000000000003, -1.0079599999999997],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_asset_pnl(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_asset_pnl(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_asset_pnl(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).asset_pnl,
        )
        assert_frame_equal(
            pf_grouped.get_asset_pnl(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).asset_pnl,
        )
        assert_frame_equal(
            pf_shared.get_asset_pnl(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).asset_pnl,
        )

    def test_asset_returns(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, -0.1201000000000001],
                    [-0.05200999999999989, -1.10398, 0.8959800000000002],
                    [0.42740909090909074, 0.0, 0.42740909090909074],
                    [-0.026533333333334127, -1.0491090909090908, -0.026533333333334127],
                    [-0.04009999999999998, -0.2998749999999999, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.asset_returns, result)
        assert_frame_equal(pf_grouped.get_asset_returns(group_by=False), result)
        assert_frame_equal(pf_shared.get_asset_returns(group_by=False), result)
        assert_frame_equal(
            pf.get_asset_returns(log_returns=True),
            np.log1p(result),
        )
        assert_frame_equal(
            pf.get_asset_returns(daily_returns=True),
            pf.asset_returns.resample("D").apply(lambda x: (x + 1).prod() - 1),
        )
        assert_frame_equal(
            pf.get_asset_returns(log_returns=True, daily_returns=True),
            pf.get_asset_returns(log_returns=True)
            .resample("D")
            .apply(lambda x: x.cumsum())
            .fillna(0),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [-0.0, -0.1201000000000001],
                    [-1.208, 0.8959800000000002],
                    [0.4948947368421051, 0.42740909090909074],
                    [-1.2189473684210528, -0.026533333333334127],
                    [-0.34999999999999987, 0.0],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_asset_returns(group_by=group_by), result)
        assert_frame_equal(pf_grouped.asset_returns, result)
        assert_frame_equal(pf_shared.asset_returns, result)
        assert_frame_equal(
            pf.asset_returns,
            vbt.Portfolio.get_asset_returns(
                init_position_value=pf.init_position_value,
                cash_flow=pf.cash_flow,
                asset_value=pf.asset_value,
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.asset_returns,
            vbt.Portfolio.get_asset_returns(
                init_position_value=pf_grouped.init_position_value,
                cash_flow=pf_grouped.cash_flow,
                asset_value=pf_grouped.asset_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.asset_returns,
            vbt.Portfolio.get_asset_returns(
                init_position_value=pf_shared.init_position_value,
                cash_flow=pf_shared.cash_flow,
                asset_value=pf_shared.asset_value,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_asset_returns(jitted=dict(parallel=True)),
            pf.get_asset_returns(jitted=dict(parallel=False)),
        )
        assert_frame_equal(pf.get_asset_returns(chunked=True), pf.get_asset_returns(chunked=False))
        size = pd.Series([0.0, 0.5, -0.5, -0.5, 0.5, 1.0, -2.0, 2.0])
        pf2 = vbt.Portfolio.from_orders(1, size, fees=0.0)
        assert_series_equal(pf2.asset_returns, pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        pf3 = vbt.Portfolio.from_orders(1, size, fees=0.01)
        assert_series_equal(
            pf3.asset_returns,
            pd.Series(
                [
                    0.0,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.020000000000000018,
                    -0.020000000000000018,
                ]
            ),
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [-0.05200999999999989, -1.10398, 0.861809090909091],
                [0.42740909090909074, -0.0, 0.42740909090909074],
                [-0.026533333333334127, -1.0491090909090908, -0.026533333333334127],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_asset_returns(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [-1.208, 0.861809090909091],
                [0.4948947368421051, 0.42740909090909074],
                [-1.2189473684210528, -0.026533333333334127],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_asset_returns(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_asset_returns(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [-0.05200999999999989, -1.10398, -0.5201],
                [0.42740909090909074, -0.0, -0.29850000000000293],
                [-0.026533333333334127, -1.0491090909090908, -0.3733185185185184],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_asset_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [-1.208, -0.5201],
                [0.4948947368421051, -0.29850000000000293],
                [-1.2189473684210528, -0.3733185185185184],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_asset_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_asset_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_asset_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).asset_returns,
        )
        assert_frame_equal(
            pf_grouped.get_asset_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).asset_returns,
        )
        assert_frame_equal(
            pf_shared.get_asset_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).asset_returns,
        )

    def test_market_value(self):
        result = pd.DataFrame(
            np.array(
                [
                    [102.0, 99.0, 100.0],
                    [102.0, 198.0, 200.0],
                    [253.0, 298.0, 300.0],
                    [337.3333333333333, 596.0, 400.0],
                    [421.66666666666663, 745.0, 400.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.market_value, result)
        assert_frame_equal(pf_grouped.get_market_value(group_by=False), result)
        assert_frame_equal(
            pf_shared.get_market_value(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [202.0, 199.0, 100.0],
                        [202.0, 398.0, 200.0],
                        [503.0, 598.0, 300.0],
                        [670.6666666666666, 1196.0, 400.0],
                        [838.3333333333333, 1495.0, 400.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [201.0, 100.0],
                    [300.0, 200.0],
                    [551.0, 300.0],
                    [933.3333333333333, 400.0],
                    [1166.6666666666665, 400.0],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_market_value(group_by=group_by), result)
        assert_frame_equal(pf_grouped.market_value, result)
        assert_frame_equal(pf_shared.market_value, result)
        assert_frame_equal(
            pf.market_value,
            vbt.Portfolio.get_market_value(
                close=pf.filled_close,
                init_value=pf.get_init_value(group_by=False),
                cash_deposits=pf.get_cash_deposits(group_by=False),
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.market_value,
            vbt.Portfolio.get_market_value(
                close=pf_grouped.filled_close,
                init_value=pf_grouped.get_init_value(group_by=False),
                cash_deposits=pf_grouped.get_cash_deposits(group_by=False),
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.market_value,
            vbt.Portfolio.get_market_value(
                close=pf_shared.filled_close,
                init_value=pf_shared.get_init_value(group_by=False, split_shared=True),
                cash_deposits=pf_shared.get_cash_deposits(group_by=False, split_shared=True),
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_market_value(jitted=dict(parallel=True)),
            pf.get_market_value(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf.get_market_value(chunked=True),
            pf.get_market_value(chunked=False),
        )
        assert_frame_equal(
            pf_grouped.get_market_value(jitted=dict(parallel=True)),
            pf_grouped.get_market_value(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_grouped.get_market_value(chunked=True),
            pf_grouped.get_market_value(chunked=False),
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [102.0, 99.0, 100.0],
                [253.0, 199.0, 150.0],
                [337.3333333333333, 398.0, 200.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_market_value(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [201.0, 100.0],
                [452.0, 150.0],
                [735.3333333333333, 200.0],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_market_value(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_market_value(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [102.0, 99.0, 100.0],
                [253.0, 199.0, 150.0],
                [337.3333333333333, 398.0, 200.0],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_market_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [201.0, 100.0],
                [452.0, 150.0],
                [735.3333333333333, 200.0],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_market_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_market_value(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_market_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).market_value,
        )
        assert_frame_equal(
            pf_grouped.get_market_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).market_value,
        )
        assert_frame_equal(
            pf_shared.get_market_value(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).market_value,
        )

    def test_market_returns(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.5, 0.0, 0.5],
                    [0.33333333333333326, 1.0, 0.3333333333333333],
                    [0.24999999999999994, 0.25, 0.0],
                ]
            ),
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.market_returns, result)
        assert_frame_equal(pf_grouped.get_market_returns(group_by=False), result)
        assert_frame_equal(pf_shared.get_market_returns(group_by=False), result)
        assert_frame_equal(
            pf.get_market_returns(log_returns=True),
            np.log(result + 1),
        )
        assert_frame_equal(
            pf.get_market_returns(daily_returns=True),
            pf.market_returns.resample("D").apply(lambda x: (x + 1).prod() - 1),
        )
        assert_frame_equal(
            pf.get_market_returns(log_returns=True, daily_returns=True),
            pf.get_market_returns(log_returns=True)
            .resample("D")
            .apply(lambda x: x.cumsum())
            .fillna(0),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0],
                    [0.4925373134328358, 1.0],
                    [0.17, 0.5],
                    [0.6938898971566847, 0.3333333333333333],
                    [0.24999999999999994, 0.0],
                ]
            ),
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(pf.get_market_returns(group_by=group_by), result)
        assert_frame_equal(pf_grouped.market_returns, result)
        assert_frame_equal(pf_shared.market_returns, result)
        assert_frame_equal(
            pf.market_returns,
            vbt.Portfolio.get_market_returns(
                init_value=pf.init_value,
                cash_deposits=pf.cash_deposits,
                market_value=pf.market_value,
                wrapper=pf.wrapper,
            ),
        )
        assert_frame_equal(
            pf_grouped.market_returns,
            vbt.Portfolio.get_market_returns(
                init_value=pf_grouped.init_value,
                cash_deposits=pf_grouped.cash_deposits,
                market_value=pf_grouped.market_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_frame_equal(
            pf_shared.market_returns,
            vbt.Portfolio.get_market_returns(
                init_value=pf_shared.init_value,
                cash_deposits=pf_shared.cash_deposits,
                market_value=pf_shared.market_value,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_frame_equal(
            pf.get_market_returns(jitted=dict(parallel=True)),
            pf.get_market_returns(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf.get_market_returns(chunked=True),
            pf.get_market_returns(chunked=False),
        )

        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.0, 1.0, 1.0],
                [0.5, 0.0, 0.5],
                [0.33333333333333326, 1.0, 0.3333333333333333],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(pf.get_market_returns(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [0.4925373134328358, 1.0],
                [0.17, 0.5],
                [0.6938898971566847, 0.3333333333333333],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_market_returns(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_frame_equal(
            pf_shared.get_market_returns(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.33333333333333326, 1.0, 0.3333333333333333],
                [np.nan, np.nan, np.nan],
            ],
            index=close_na.index,
            columns=close_na.columns,
        )
        assert_frame_equal(
            pf.get_market_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.DataFrame(
            [
                [np.nan, np.nan],
                [0.0, 0.0],
                [0.2537313432835821, 0.5],
                [0.6268436578171089, 0.3333333333333333],
                [np.nan, np.nan],
            ],
            index=close_na.index,
            columns=group_columns,
        )
        assert_frame_equal(
            pf_grouped.get_market_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf_shared.get_market_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_frame_equal(
            pf.get_market_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).market_returns,
        )
        assert_frame_equal(
            pf_grouped.get_market_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).market_returns,
        )
        assert_frame_equal(
            pf_shared.get_market_returns(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).market_returns,
        )

    def test_total_market_return(self):
        assert_series_equal(
            pf.total_market_return,
            ((pf.market_value.iloc[-1] - pf.input_value) / pf.input_value).rename(
                "total_market_return"
            ),
        )
        assert_series_equal(
            pf_grouped.get_total_market_return(group_by=False),
            (
                (
                    pf_grouped.get_market_value(group_by=False).iloc[-1]
                    - pf_grouped.get_input_value(group_by=False)
                )
                / pf_grouped.get_input_value(group_by=False)
            ).rename("total_market_return"),
        )
        assert_series_equal(
            pf_shared.get_total_market_return(group_by=False),
            (
                (
                    pf_shared.get_market_value(group_by=False).iloc[-1]
                    - pf_shared.get_input_value(group_by=False)
                )
                / pf_shared.get_input_value(group_by=False)
            ).rename("total_market_return"),
        )
        assert_series_equal(
            pf.get_total_market_return(group_by=group_by),
            (
                (
                    pf.get_market_value(group_by=group_by).iloc[-1]
                    - pf.get_input_value(group_by=group_by)
                )
                / pf.get_input_value(group_by=group_by)
            ).rename("total_market_return"),
        )
        assert_series_equal(
            pf_grouped.total_market_return,
            (
                (pf_grouped.market_value.iloc[-1] - pf_grouped.input_value) / pf_grouped.input_value
            ).rename("total_market_return"),
        )
        assert_series_equal(
            pf_shared.total_market_return,
            (
                (pf_shared.market_value.iloc[-1] - pf_shared.input_value) / pf_shared.input_value
            ).rename("total_market_return"),
        )
        assert_series_equal(
            pf.total_market_return,
            vbt.Portfolio.get_total_market_return(
                input_value=pf.input_value,
                market_value=pf.market_value,
                wrapper=pf.wrapper,
            ),
        )
        assert_series_equal(
            pf_grouped.total_market_return,
            vbt.Portfolio.get_total_market_return(
                input_value=pf_grouped.input_value,
                market_value=pf_grouped.market_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        assert_series_equal(
            pf_shared.total_market_return,
            vbt.Portfolio.get_total_market_return(
                input_value=pf_shared.input_value,
                market_value=pf_shared.market_value,
                wrapper=pf_shared.wrapper,
            ),
        )
        assert_series_equal(
            pf.get_total_market_return(jitted=dict(parallel=True)),
            pf.get_total_market_return(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            pf.get_total_market_return(chunked=True),
            pf.get_total_market_return(chunked=False),
        )

        sim_range_result = pd.Series(
            [0.6699669966996699, 1.9949748743718594, 3.0], index=close_na.columns
        ).rename("total_market_return")
        assert_series_equal(pf.get_total_market_return(sim_start=1, sim_end=4), sim_range_result)
        sim_range_result_grouped = pd.Series([1.3275145469659184, 3.0], index=group_columns).rename(
            "total_market_return"
        )
        assert_series_equal(
            pf_grouped.get_total_market_return(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        assert_series_equal(
            pf_shared.get_total_market_return(sim_start=1, sim_end=4), sim_range_result_grouped
        )
        sim_range_result = pd.Series([0.6699669966996699, 1.0, 1.0], index=close_na.columns).rename(
            "total_market_return"
        )
        assert_series_equal(
            pf.get_total_market_return(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result,
        )
        sim_range_result_grouped = pd.Series([0.8337489609310056, 1.0], index=group_columns).rename(
            "total_market_return"
        )
        assert_series_equal(
            pf_grouped.get_total_market_return(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf_shared.get_total_market_return(sim_start=1, sim_end=4, rec_sim_range=True),
            sim_range_result_grouped,
        )
        assert_series_equal(
            pf.get_total_market_return(sim_start=1, sim_end=4, rec_sim_range=True),
            pf.replace(sim_start=1, sim_end=4).total_market_return,
        )
        assert_series_equal(
            pf_grouped.get_total_market_return(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_grouped.replace(sim_start=1, sim_end=4).total_market_return,
        )
        assert_series_equal(
            pf_shared.get_total_market_return(sim_start=1, sim_end=4, rec_sim_range=True),
            pf_shared.replace(sim_start=1, sim_end=4).total_market_return,
        )

    def test_bm_value(self):
        assert_frame_equal(
            pf.bm_value,
            pd.DataFrame(
                np.array(
                    [
                        [102.0, 99.0, 100.0],
                        [81.60000000000001, 79.2, 100.0],
                        [161.2, 179.2, 75.0],
                        [107.46666666666665, 89.6, 50.0],
                        [107.46666666666665, 44.8, 25.0],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.replace(bm_close=None).bm_value,
            pf.market_value,
        )
        assert pf.replace(bm_close=False).bm_value is None

    def test_bm_returns(self):
        assert_frame_equal(
            pf.bm_returns,
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [-0.19999999999999993, -0.19999999999999998, 0.0],
                        [-0.2500000000000002, 0.0, -0.25],
                        [-0.33333333333333337, -0.5, -0.3333333333333333],
                        [0.0, -0.5, -0.5],
                    ]
                ),
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_frame_equal(
            pf.replace(bm_close=None).bm_returns,
            pf.market_returns,
        )
        assert pf.replace(bm_close=False).bm_returns is None

    def test_return_methods(self):
        assert_frame_equal(pf_shared.cumulative_returns, pf_shared.cumulative_returns)
        assert_frame_equal(
            pf_shared.cumulative_returns,
            pd.DataFrame(
                np.array(
                    [
                        [0.0, -0.0012009999999998966],
                        [-0.0060099502487561685, 0.007758800000000177],
                        [-0.0013318407960194456, 0.017161800000000005],
                        [-0.0071035628576462395, 0.017082199999999936],
                        [-0.010592514017524146, 0.017082199999999936],
                    ]
                )
                + 1,
                index=close_na.index,
                columns=group_columns,
            ),
        )
        assert_frame_equal(
            pf_shared.get_cumulative_returns(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, -0.0012009999999998966],
                        [-0.0005149504950495709, -0.005547638190954718, 0.007758800000000177],
                        [0.0041400000000000325, -0.005547638190954718, 0.017161800000000005],
                        [0.004120158305503052, -0.011316069423691677, 0.017082199999999936],
                        [0.003620376930300262, -0.014313952156949417, 0.017082199999999936],
                    ]
                )
                + 1,
                index=close_na.index,
                columns=close_na.columns,
            ),
        )
        assert_series_equal(pf_shared.sharpe_ratio, pf_shared.sharpe_ratio)
        assert_series_equal(
            pf_shared.sharpe_ratio,
            pd.Series(
                np.array([-8.966972200385989, 12.345065267401496]),
                index=group_columns,
            ).rename("sharpe_ratio"),
        )
        assert_series_equal(
            pf_shared.get_sharpe_ratio(risk_free=0.01),
            pd.Series(
                np.array([-51.276434758632554, -23.91718815937344]),
                index=group_columns,
            ).rename("sharpe_ratio"),
        )
        assert_series_equal(
            pf_shared.get_sharpe_ratio(year_freq="365D"),
            pd.Series(
                np.array([-8.966972200385989, 12.345065267401496]),
                index=group_columns,
            ).rename("sharpe_ratio"),
        )
        assert_series_equal(
            pf_shared.get_sharpe_ratio(group_by=False),
            pd.Series(
                np.array([6.260933805237826, -19.34902167642263, 12.345065267401496]),
                index=close_na.columns,
            ).rename("sharpe_ratio"),
        )
        assert_series_equal(
            pf_shared.get_information_ratio(group_by=False),
            pd.Series(
                np.array([1.0384749617059628, 0.9525372423693426, 1.0199058062359245]),
                index=close_na.columns,
            ).rename("information_ratio"),
        )
        with pytest.raises(Exception):
            pf_shared.get_information_ratio(pf_shared.get_market_returns(group_by=False) * 2)
        assert_frame_equal(
            pf_shared.get_cumulative_returns(jitted=dict(parallel=True)),
            pf_shared.get_cumulative_returns(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            pf_shared.get_cumulative_returns(chunked=True),
            pf_shared.get_cumulative_returns(chunked=False),
        )

    def test_qs_methods(self):
        if qs_available:
            isclose(pf_shared.qs.sharpe(column="first"), pf_shared.sharpe_ratio["first"])

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Start Value",
                "Min Value",
                "Max Value",
                "End Value",
                "Total Cash Deposits",
                "Total Return [%]",
                "Benchmark Return [%]",
                "Position Coverage [%]",
                "Max Gross Exposure [%]",
                "Max Drawdown [%]",
                "Max Drawdown Duration",
                "Total Orders",
                "Total Fees Paid",
                "Total Trades",
                "Win Rate [%]",
                "Best Trade [%]",
                "Worst Trade [%]",
                "Avg Winning Trade [%]",
                "Avg Losing Trade [%]",
                "Avg Winning Trade Duration",
                "Avg Losing Trade Duration",
                "Profit Factor",
                "Expectancy",
                "Sharpe Ratio",
                "Calmar Ratio",
                "Omega Ratio",
                "Sortino Ratio",
            ],
            dtype="object",
        )
        assert_series_equal(
            pf.stats(),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.33333333333333,
                        99.89063333333333,
                        167.48282666666668,
                        166.24150666666665,
                        66.66666666666667,
                        -0.09944158449010732,
                        -71.66666666666667,
                        80.0,
                        3.2649926907570648,
                        0.6276712912251518,
                        pd.Timedelta("3 days 00:00:00"),
                        3.6666666666666665,
                        0.42916000000000004,
                        2.3333333333333335,
                        66.66666666666667,
                        -62.06261760946578,
                        -65.81967240213856,
                        91.58494359313319,
                        -374.9933222036729,
                        pd.Timedelta("3 days 00:00:00"),
                        pd.Timedelta("4 days 00:00:00"),
                        np.inf,
                        0.2866227272727273,
                        -0.25595098630477686,
                        889.6944375349927,
                        6.270976459353577,
                        49.897006624719126,
                    ]
                ),
                index=stats_index,
                name="agg_stats",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.replace(bm_close=None).stats(),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.33333333333333,
                        99.89063333333333,
                        167.48282666666668,
                        166.24150666666665,
                        66.66666666666667,
                        -0.09944158449010732,
                        283.3333333333333,
                        80.0,
                        3.2649926907570648,
                        0.6276712912251518,
                        pd.Timedelta("3 days 00:00:00"),
                        3.6666666666666665,
                        0.42916000000000004,
                        2.3333333333333335,
                        66.66666666666667,
                        -62.06261760946578,
                        -65.81967240213856,
                        91.58494359313319,
                        -374.9933222036729,
                        pd.Timedelta("3 days 00:00:00"),
                        pd.Timedelta("4 days 00:00:00"),
                        np.inf,
                        0.2866227272727273,
                        -0.25595098630477686,
                        889.6944375349927,
                        6.270976459353577,
                        49.897006624719126,
                    ]
                ),
                index=stats_index,
                name="agg_stats",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.replace(bm_close=False).stats(),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.33333333333333,
                        99.89063333333333,
                        167.48282666666668,
                        166.24150666666665,
                        66.66666666666667,
                        -0.09944158449010732,
                        80.0,
                        3.2649926907570648,
                        0.6276712912251518,
                        pd.Timedelta("3 days 00:00:00"),
                        3.6666666666666665,
                        0.42916000000000004,
                        2.3333333333333335,
                        66.66666666666667,
                        -62.06261760946578,
                        -65.81967240213856,
                        91.58494359313319,
                        -374.9933222036729,
                        pd.Timedelta("3 days 00:00:00"),
                        pd.Timedelta("4 days 00:00:00"),
                        np.inf,
                        0.2866227272727273,
                        -0.25595098630477686,
                        889.6944375349927,
                        6.270976459353577,
                        49.897006624719126,
                    ]
                ),
                index=pd.Index(
                    [
                        "Start Index",
                        "End Index",
                        "Total Duration",
                        "Start Value",
                        "Min Value",
                        "Max Value",
                        "End Value",
                        "Total Cash Deposits",
                        "Total Return [%]",
                        "Position Coverage [%]",
                        "Max Gross Exposure [%]",
                        "Max Drawdown [%]",
                        "Max Drawdown Duration",
                        "Total Orders",
                        "Total Fees Paid",
                        "Total Trades",
                        "Win Rate [%]",
                        "Best Trade [%]",
                        "Worst Trade [%]",
                        "Avg Winning Trade [%]",
                        "Avg Losing Trade [%]",
                        "Avg Winning Trade Duration",
                        "Avg Losing Trade Duration",
                        "Profit Factor",
                        "Expectancy",
                        "Sharpe Ratio",
                        "Calmar Ratio",
                        "Omega Ratio",
                        "Sortino Ratio",
                    ],
                    dtype="object",
                ),
                name="agg_stats",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.stats(column="a"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        102.0,
                        101.89598000000001,
                        202.83628000000004,
                        202.62781999999999,
                        100.0,
                        0.3108019801980197,
                        -60.00000000000001,
                        80.0,
                        2.467578242711193,
                        0.10277254148026709,
                        pd.Timedelta("3 days 00:00:00"),
                        4,
                        0.48618000000000006,
                        3,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", settings=dict(freq="10 days", year_freq="200 days")),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("50 days 00:00:00"),
                        102.0,
                        101.89598000000001,
                        202.83628000000004,
                        202.62781999999999,
                        100.0,
                        0.3108019801980197,
                        -60.00000000000001,
                        80.0,
                        2.467578242711193,
                        0.10277254148026709,
                        pd.Timedelta("30 days 00:00:00"),
                        4,
                        0.48618000000000006,
                        3,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("35 days 00:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        1.4651010643478568,
                        28.178717025259605,
                        4.506828421607624,
                        10.1075418640978,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", settings=dict(trade_type="positions")),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        102.0,
                        101.89598000000001,
                        202.83628000000004,
                        202.62781999999999,
                        100.0,
                        0.3108019801980197,
                        -60.00000000000001,
                        80.0,
                        2.467578242711193,
                        0.10277254148026709,
                        pd.Timedelta("3 days 00:00:00"),
                        4,
                        0.48618000000000006,
                        3,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", settings=dict(required_return=0.1, risk_free=0.01)),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        102.0,
                        101.89598000000001,
                        202.83628000000004,
                        202.62781999999999,
                        100.0,
                        0.3108019801980197,
                        -60.00000000000001,
                        80.0,
                        2.467578242711193,
                        0.10277254148026709,
                        pd.Timedelta("3 days 00:00:00"),
                        4,
                        0.48618000000000006,
                        3,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        -37.32398741973627,
                        665.2843559613844,
                        0.0,
                        -19.089875288486446,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", settings=dict(use_asset_returns=True)),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        102.0,
                        101.89598000000001,
                        202.83628000000004,
                        202.62781999999999,
                        100.0,
                        0.3108019801980197,
                        -60.00000000000001,
                        80.0,
                        2.467578242711193,
                        0.10277254148026709,
                        pd.Timedelta("3 days 00:00:00"),
                        4,
                        0.48618000000000006,
                        3,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        5.746061520593739,
                        418742834.6664331,
                        3.602470352954977,
                        37.244793636996356,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", settings=dict(log_returns=True)),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        102.0,
                        101.89598000000001,
                        202.83628000000004,
                        202.62781999999999,
                        100.0,
                        0.3108019801980197,
                        -67.10029548637564,
                        80.0,
                        2.467578242711193,
                        0.10277254148026709,
                        pd.Timedelta("3 days 00:00:00"),
                        4,
                        0.48618000000000006,
                        3,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        6.24774839454674,
                        665.2843559613844,
                        4.483949018487763,
                        42.89733832279388,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", settings=dict(daily_returns=True)),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        102.0,
                        101.89598000000001,
                        202.83628000000004,
                        202.62781999999999,
                        100.0,
                        0.3108019801980197,
                        -60.00000000000001,
                        80.0,
                        2.467578242711193,
                        0.10277254148026709,
                        pd.Timedelta("3 days 00:00:00"),
                        4,
                        0.48618000000000006,
                        3,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        6.2589144905284515,
                        665.2843559613844,
                        4.506828421607747,
                        43.17943777140366,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", settings=dict(incl_open=True)),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        102.0,
                        101.89598000000001,
                        202.83628000000004,
                        202.62781999999999,
                        100.0,
                        0.3108019801980197,
                        -60.00000000000001,
                        80.0,
                        2.467578242711193,
                        0.10277254148026709,
                        pd.Timedelta("3 days 00:00:00"),
                        4,
                        0.48618000000000006,
                        3,
                        66.66666666666666,
                        41.25431425976392,
                        -3.9702970297029667,
                        39.25363306085381,
                        -3.9702970297029667,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.Timedelta("1 days 00:00:00"),
                        4.131271820448882,
                        0.20927333333333345,
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf_grouped.stats(column="first"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        201.0,
                        199.792,
                        400.73230000000007,
                        397.0163,
                        200.0,
                        -0.9934413965087281,
                        -68.75458196678184,
                        100.0,
                        3.778182407120312,
                        0.9273023412387791,
                        pd.Timedelta("3 days 00:00:00"),
                        7,
                        0.8417000000000001,
                        5,
                        66.66666666666666,
                        41.25431425976392,
                        -374.9933222036729,
                        39.25363306085381,
                        -374.9933222036729,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.Timedelta("4 days 00:00:00"),
                        2.0281986101032405,
                        0.1399727272727273,
                        -8.966972200385989,
                        -51.016263121272715,
                        0.307541522123087,
                        -10.006463484493487,
                    ]
                ),
                index=stats_index,
                name="first",
                dtype=object,
            ),
        )
        max_winning_streak = (
            "max_winning_streak",
            dict(
                title="Max Winning Streak",
                calc_func=lambda trades: trades.winning_streak.max(),
                resolve_trades=True,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", metrics=max_winning_streak),
            pd.Series([2.0], index=["Max Winning Streak"], name="a", dtype=object),
        )
        max_winning_streak = (
            "max_winning_streak",
            dict(
                title="Max Winning Streak",
                calc_func=lambda self, group_by: self.get_trades(
                    group_by=group_by
                ).winning_streak.max(),
            ),
        )
        assert_series_equal(
            pf.stats(column="a", metrics=max_winning_streak),
            pd.Series([2.0], index=["Max Winning Streak"], name="a", dtype=object),
        )
        max_winning_streak = (
            "max_winning_streak",
            dict(
                title="Max Winning Streak",
                calc_func=lambda self, settings: self.get_trades(
                    group_by=settings["group_by"]
                ).winning_streak.max(),
                resolve_calc_func=False,
            ),
        )
        assert_series_equal(
            pf.stats(column="a", metrics=max_winning_streak),
            pd.Series([2.0], index=["Max Winning Streak"], name="a", dtype=object),
        )
        vbt.settings.portfolio.stats["settings"]["my_arg"] = 100
        my_arg_metric = ("my_arg_metric", dict(title="My Arg", calc_func=lambda my_arg: my_arg))
        assert_series_equal(
            pf.stats(my_arg_metric, column="a"),
            pd.Series([100], index=["My Arg"], name="a", dtype=object),
        )
        vbt.settings.portfolio.stats.reset()
        assert_series_equal(
            pf.stats(my_arg_metric, column="a", settings=dict(my_arg=200)),
            pd.Series([200], index=["My Arg"], name="a", dtype=object),
        )
        my_arg_metric = (
            "my_arg_metric",
            dict(title="My Arg", my_arg=300, calc_func=lambda my_arg: my_arg),
        )
        assert_series_equal(
            pf.stats(my_arg_metric, column="a", settings=dict(my_arg=200)),
            pd.Series([300], index=["My Arg"], name="a", dtype=object),
        )
        assert_series_equal(
            pf.stats(
                my_arg_metric,
                column="a",
                settings=dict(my_arg=200),
                metric_settings=dict(my_arg_metric=dict(my_arg=400)),
            ),
            pd.Series([400], index=["My Arg"], name="a", dtype=object),
        )
        trade_min_pnl_cnt = (
            "trade_min_pnl_cnt",
            dict(
                title=vbt.Sub("Trades with PnL over $$${min_pnl}"),
                calc_func=lambda trades, min_pnl: trades.apply_mask(
                    trades.pnl.values >= min_pnl
                ).count(),
                resolve_trades=True,
            ),
        )
        assert_series_equal(
            pf.stats(
                metrics=trade_min_pnl_cnt,
                column="a",
                metric_settings=dict(trade_min_pnl_cnt=dict(min_pnl=0)),
            ),
            pd.Series([2], index=["Trades with PnL over $0"], name="a", dtype=object),
        )
        assert_series_equal(
            pf.stats(
                metrics=[trade_min_pnl_cnt, trade_min_pnl_cnt, trade_min_pnl_cnt],
                column="a",
                metric_settings=dict(
                    trade_min_pnl_cnt_0=dict(min_pnl=0),
                    trade_min_pnl_cnt_1=dict(min_pnl=10),
                    trade_min_pnl_cnt_2=dict(min_pnl=20),
                ),
            ),
            pd.Series(
                [2, 0, 0],
                index=[
                    "Trades with PnL over $0",
                    "Trades with PnL over $10",
                    "Trades with PnL over $20",
                ],
                name="a",
                dtype=object,
            ),
        )
        assert_frame_equal(
            pf.stats(
                metrics="total_trades", agg_func=None, settings=dict(trades_type="entry_trades")
            ),
            pd.DataFrame([3, 3, 2], index=close_na.columns, columns=["Total Trades"]),
        )
        assert_frame_equal(
            pf.stats(
                metrics="total_trades", agg_func=None, settings=dict(trades_type="exit_trades")
            ),
            pd.DataFrame([3, 2, 2], index=close_na.columns, columns=["Total Trades"]),
        )
        assert_frame_equal(
            pf.stats(metrics="total_trades", agg_func=None, settings=dict(trades_type="positions")),
            pd.DataFrame([2, 1, 1], index=close_na.columns, columns=["Total Trades"]),
        )
        assert_series_equal(pf["c"].stats(), pf.stats(column="c"))
        assert_series_equal(pf["c"].stats(), pf_grouped.stats(column="c", group_by=False))
        assert_series_equal(pf_grouped["second"].stats(), pf_grouped.stats(column="second"))
        assert_series_equal(
            pf_grouped["second"].stats(), pf.stats(column="second", group_by=group_by)
        )
        assert_series_equal(
            pf.replace(wrapper=pf.wrapper.replace(freq="10d")).stats(),
            pf.stats(settings=dict(freq="10d")),
        )
        stats_df = pf.stats(agg_func=None)
        assert stats_df.shape == (3, 30)
        assert_index_equal(stats_df.index, pf.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)
        assert_series_equal(
            pf.replace(sim_start=1, sim_end=4).stats(),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-02 00:00:00"),
                        pd.Timestamp("2020-01-04 00:00:00"),
                        pd.Timedelta("3 days 00:00:00"),
                        100.33333333333333,
                        99.54010666666666,
                        166.87609333333336,
                        165.74820666666668,
                        66.66666666666667,
                        -0.8254011695440902,
                        -50.0,
                        88.88888888888887,
                        2.817923261077409,
                        0.7463284432878199,
                        pd.Timedelta("2 days 08:00:00"),
                        2.6666666666666665,
                        0.29246,
                        2.0,
                        33.333333333333336,
                        -114.54501584928978,
                        -115.87880331522986,
                        39.25363306085381,
                        -192.44468090381665,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.Timedelta("2 days 12:00:00"),
                        np.inf,
                        -0.0047439393939394,
                        -8.927839596805114,
                        516.1769753863151,
                        2.904509055725145,
                        19.988554313695975,
                    ]
                ),
                index=stats_index,
                name="agg_stats",
                dtype=object,
            ),
        )
        assert_series_equal(
            pf.replace(sim_start=1, sim_end=4).stats(),
            pf.stats(settings=dict(sim_start=1, sim_end=4, rec_sim_range=True)),
        )

    def test_returns_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Return [%]",
                "Benchmark Return [%]",
                "Annualized Return [%]",
                "Annualized Volatility [%]",
                "Max Drawdown [%]",
                "Max Drawdown Duration",
                "Sharpe Ratio",
                "Calmar Ratio",
                "Omega Ratio",
                "Sortino Ratio",
                "Skew",
                "Kurtosis",
                "Tail Ratio",
                "Common Sense Ratio",
                "Value at Risk",
                "Alpha",
                "Beta",
            ],
            dtype="object",
        )
        assert_series_equal(
            pf.returns_stats(column="a"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        0.7162671975297075,
                        -60.00000000000001,
                        68.3729640692142,
                        8.374843895239454,
                        0.10277254148026353,
                        pd.Timedelta("3 days 00:00:00"),
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                        2.1657940859079745,
                        4.749360549470598,
                        7.283755486189502,
                        32.8266362411993,
                        -0.001013547284289139,
                        -0.07963807950336177,
                        -0.010617659789479695,
                    ]
                ),
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            pf.replace(bm_close=None).returns_stats(column="a"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        0.7162671975297075,
                        150.0,
                        68.3729640692142,
                        8.374843895239454,
                        0.10277254148026353,
                        pd.Timedelta("3 days 00:00:00"),
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                        2.1657940859079745,
                        4.749360549470598,
                        7.283755486189502,
                        32.8266362411993,
                        -0.001013547284289139,
                        -0.4768429263982791,
                        0.014813148996296632,
                    ]
                ),
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            pf.replace(bm_close=False).returns_stats(column="a"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        0.7162671975297075,
                        68.3729640692142,
                        8.374843895239454,
                        0.10277254148026353,
                        pd.Timedelta("3 days 00:00:00"),
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                        2.1657940859079745,
                        4.749360549470598,
                        7.283755486189502,
                        32.8266362411993,
                        -0.001013547284289139,
                    ]
                ),
                index=pd.Index(
                    [
                        "Start Index",
                        "End Index",
                        "Total Duration",
                        "Total Return [%]",
                        "Annualized Return [%]",
                        "Annualized Volatility [%]",
                        "Max Drawdown [%]",
                        "Max Drawdown Duration",
                        "Sharpe Ratio",
                        "Calmar Ratio",
                        "Omega Ratio",
                        "Sortino Ratio",
                        "Skew",
                        "Kurtosis",
                        "Tail Ratio",
                        "Common Sense Ratio",
                        "Value at Risk",
                    ],
                    dtype="object",
                ),
                name="a",
            ),
        )

    def test_plots(self):
        pf.plot(column="a", subplots="all")
        pf.replace(bm_close=None).plot(column="a", subplots="all")
        pf.replace(bm_close=False).plot(column="a", subplots="all")
        pf.plot(column="a", subplots="all")
        pf_grouped.plot(column="first", subplots="all")
        pf_grouped.plot(column="a", subplots="all", group_by=False)
        pf_shared.plot(column="a", subplots="all", group_by=False)
        with pytest.raises(Exception):
            pf.plot(subplots="all")
        pf.plot(subplots="all", per_column=True)
        with pytest.raises(Exception):
            pf_grouped.plot(subplots="all")
        pf_grouped.plot(subplots="all", per_column=True)
        pf.plot(column="a", subplots="all", settings=dict(sim_start=1, sim_end=4))
        pf.replace(bm_close=None).plot(
            column="a", subplots="all", settings=dict(sim_start=1, sim_end=4)
        )
        pf.replace(bm_close=False).plot(
            column="a", subplots="all", settings=dict(sim_start=1, sim_end=4)
        )
        pf.plot(column="a", subplots="all", settings=dict(sim_start=1, sim_end=4))
        pf_grouped.plot(column="first", subplots="all", settings=dict(sim_start=1, sim_end=4))
        pf_grouped.plot(
            column="a", subplots="all", group_by=False, settings=dict(sim_start=1, sim_end=4)
        )
        pf_shared.plot(
            column="a", subplots="all", group_by=False, settings=dict(sim_start=1, sim_end=4)
        )

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3D"])
    def test_resample(self, test_freq):
        assert_frame_equal(
            pf.replace(call_seq=None).resample(test_freq).open,
            pf.open.resample(test_freq).first(),
        )
        assert_frame_equal(
            pf.replace(call_seq=None).resample(test_freq).high,
            pf.high.resample(test_freq).max(),
        )
        assert_frame_equal(
            pf.replace(call_seq=None).resample(test_freq).low,
            pf.low.resample(test_freq).min(),
        )
        assert_frame_equal(
            pf.replace(call_seq=None).resample(test_freq).close,
            pf.close.resample(test_freq).last(),
        )
        assert_frame_equal(
            pf.replace(call_seq=None).resample(test_freq).returns,
            pf.returns_acc.resample(test_freq).obj,
        )
        assert_series_equal(
            pf.replace(call_seq=None).resample(test_freq).total_return,
            pf.total_return,
        )
        assert_frame_equal(
            pf_filled.replace(call_seq=None).resample(test_freq).returns,
            pf_filled.replace(call_seq=None).returns_acc.resample(test_freq).obj,
        )
        assert_frame_equal(
            pf_grouped.replace(call_seq=None).resample(test_freq).returns,
            pf_grouped.returns_acc.resample(test_freq).obj,
        )
        assert_series_equal(
            pf_grouped.replace(call_seq=None).resample(test_freq).total_return,
            pf_grouped.total_return,
        )
        assert_frame_equal(
            pf_grouped_filled.replace(call_seq=None).resample(test_freq).returns,
            pf_grouped_filled.replace(call_seq=None).returns_acc.resample(test_freq).obj,
        )
        assert_frame_equal(
            pf_shared.replace(call_seq=None).resample(test_freq).returns,
            pf_shared.returns_acc.resample(test_freq).obj,
        )
        assert_series_equal(
            pf_shared.replace(call_seq=None).resample(test_freq).total_return,
            pf_shared.total_return,
        )
        assert_frame_equal(
            pf_shared_filled.replace(call_seq=None).resample(test_freq).returns,
            pf_shared_filled.replace(call_seq=None).returns_acc.resample(test_freq).obj,
        )
        with pytest.raises(Exception):
            pf.resample(test_freq).stats()

        sim_range_pf = vbt.Portfolio.from_orders(
            pd.DataFrame(
                {
                    "a": pd.Series(np.arange(1, 101), index=pd.date_range("2020", periods=100)),
                    "b": pd.Series(np.arange(1, 101), index=pd.date_range("2020", periods=100)),
                }
            ),
            size=1,
            sim_start=[10, 20],
            sim_end=[-10, -20],
            attach_call_seq=False,
        )
        new_sim_range_pf = sim_range_pf.resample(test_freq)
        delta_factor = pd.Timedelta("1D") / pd.Timedelta(test_freq)
        np.testing.assert_array_equal(
            new_sim_range_pf._sim_start, np.floor(sim_range_pf._sim_start * delta_factor)
        )
        np.testing.assert_array_equal(
            new_sim_range_pf._sim_end, np.ceil(sim_range_pf._sim_end * delta_factor)
        )
