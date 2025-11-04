import os
from datetime import datetime

import pytest
import vectorbtpro as vbt

from tests.utils import *

qs_available = True
try:
    import quantstats as qs
except:
    qs_available = False

day_dt = np.timedelta64(86400000000000)

ts = pd.DataFrame(
    {
        "a": [101, 102, 103, 104, 105],
        "b": [105, 104, 103, 102, 101],
        "c": [101, 102, 103, 102, 101],
    },
    index=pd.date_range("2020", periods=5),
)
rets = ts.pct_change()

bm_ts = pd.DataFrame(
    {
        "a": [101, 102, 103, 102, 101],
        "b": [101, 102, 103, 104, 105],
        "c": [105, 104, 103, 102, 101],
    },
    index=ts.index,
)
bm_returns = bm_ts.pct_change()
log_bm_returns = np.log(bm_returns + 1)

ret_acc = rets.vbt.returns(bm_returns=bm_returns)
log_ret_acc = np.log(rets + 1).vbt.returns(bm_returns=log_bm_returns, log_returns=True)


def prepare_sim_range_obj(df):
    return pd.concat(
        (
            df["a"].iloc[sim_start[0] : sim_end[0]].reset_index(drop=True),
            df["b"].iloc[sim_start[1] : sim_end[1]].reset_index(drop=True),
            df["c"].iloc[sim_start[2] : sim_end[2]].reset_index(drop=True),
        ),
        axis=1,
    )


sim_start = [0, 1, 2]
sim_end = [3, 4, 5]
sim_range_rets = prepare_sim_range_obj(rets)
sim_range_bm_returns = prepare_sim_range_obj(bm_returns)
sim_range_ret_acc = sim_range_rets.vbt.returns(bm_returns=sim_range_bm_returns, freq="daily")


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.returns.defaults = dict(
        start_value=0.0,
        window=rets.shape[0],
        minp=1,
        ddof=1,
        risk_free=0.001,
        levy_alpha=2.0,
        required_return=0.01,
        cutoff=0.05,
        periods="dt_periods",
    )


def teardown_module():
    vbt.settings.reset()


# ############# accessors ############# #


class TestAccessors:
    def test_row_stack(self):
        rets2 = rets * 2
        rets2.index += pd.Timedelta(days=len(rets2.index))
        bm_returns2 = bm_returns * 2
        bm_returns2.index += pd.Timedelta(days=len(bm_returns2.index))
        ret_acc2 = ret_acc.replace(
            wrapper=ret_acc.wrapper.replace(index=rets2.index),
            obj=rets2,
            bm_returns=bm_returns2,
        )
        acc = vbt.ReturnsAccessor.row_stack(ret_acc, ret_acc2)
        assert isinstance(acc, vbt.ReturnsDFAccessor)
        assert_frame_equal(acc.obj, pd.concat((rets, rets2)))
        assert_frame_equal(acc.bm_returns, pd.concat((bm_returns, bm_returns2)))

    def test_column_stack(self):
        rets2 = (rets * 2).add_prefix("_")
        bm_returns2 = (bm_returns * 2).add_prefix("_")
        ret_acc2 = ret_acc.replace(
            wrapper=ret_acc.wrapper.replace(index=rets2.index, columns=rets2.columns),
            obj=rets2,
            bm_returns=bm_returns2,
        )
        acc = vbt.ReturnsAccessor.column_stack(ret_acc, ret_acc2)
        assert isinstance(acc, vbt.ReturnsDFAccessor)
        assert_frame_equal(acc.obj, pd.concat((rets, rets2), axis=1))
        assert_frame_equal(acc.bm_returns, pd.concat((bm_returns, bm_returns2), axis=1))
        with pytest.raises(Exception):
            vbt.ReturnsAccessor.column_stack(ret_acc, ret_acc2.replace(bm_returns=None))
        with pytest.raises(Exception):
            vbt.ReturnsAccessor.column_stack(ret_acc.replace(bm_returns=None), ret_acc2)
        acc = vbt.ReturnsAccessor.column_stack(
            ret_acc.replace(bm_returns=None), ret_acc2.replace(bm_returns=None)
        )
        assert acc.bm_returns is None

    def test_indexing(self):
        assert ret_acc["a"].total() == ret_acc["a"].total()

    def test_bm_returns(self):
        assert_frame_equal(bm_returns, ret_acc.bm_returns)
        assert_series_equal(bm_returns["a"], ret_acc["a"].bm_returns)

    def test_freq(self):
        assert ret_acc.wrapper.freq == day_dt
        assert ret_acc["a"].wrapper.freq == day_dt
        assert ret_acc(freq="2D").wrapper.freq == day_dt * 2
        assert ret_acc["a"](freq="2D").wrapper.freq == day_dt * 2
        assert pd.Series([1, 2, 3]).vbt.returns.wrapper.freq is None
        assert pd.Series([1, 2, 3]).vbt.returns(freq="3D").wrapper.freq == day_dt * 3
        assert (
            pd.Series([1, 2, 3]).vbt.returns(freq=np.timedelta64(4, "D")).wrapper.freq == day_dt * 4
        )

    def test_ann_factor(self):
        assert ret_acc["a"](year_freq="365 days").ann_factor == 365
        assert ret_acc(year_freq="365 days").ann_factor == 365
        with pytest.raises(Exception):
            assert pd.Series([1, 2, 3]).vbt.returns(freq=None).ann_factor

        returns = pd.Series(np.nan, index=pd.date_range("2021-01-01", "2024-01-01", freq="B"))
        assert returns.vbt.returns(year_freq="yearly").ann_factor == 365.2425
        assert returns.vbt.returns(year_freq="auto").ann_factor == 260.9047184170472
        assert returns.vbt.returns(year_freq="auto_round").ann_factor == 261.0
        assert returns.vbt.returns(year_freq="index_min").ann_factor == 260.0

    def test_from_value(self):
        assert_series_equal(
            pd.Series.vbt.returns.from_value(ts["a"]).obj, ts["a"].pct_change().fillna(0.0)
        )
        assert_frame_equal(pd.DataFrame.vbt.returns.from_value(ts).obj, ts.pct_change().fillna(0.0))
        assert_frame_equal(
            pd.DataFrame.vbt.returns.from_value(ts, log_returns=True).obj,
            np.log(ts.pct_change() + 1).fillna(0.0),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.returns.from_value(ts, jitted=dict(parallel=True)).obj,
            pd.DataFrame.vbt.returns.from_value(ts, jitted=dict(parallel=False)).obj,
        )
        assert_frame_equal(
            pd.DataFrame.vbt.returns.from_value(ts, chunked=True).obj,
            pd.DataFrame.vbt.returns.from_value(ts, chunked=False).obj,
        )
        assert pd.Series.vbt.returns.from_value(
            ts["a"], year_freq="365 days"
        ).year_freq == pd.to_timedelta("365 days")
        assert pd.DataFrame.vbt.returns.from_value(
            ts, year_freq="365 days"
        ).year_freq == pd.to_timedelta("365 days")
        assert_frame_equal(
            pd.DataFrame.vbt.returns.from_value(ts, sim_start=sim_start, sim_end=sim_end).obj,
            pd.DataFrame(
                [
                    [0.0, np.nan, np.nan],
                    [0.009900990099009901, 0.0, np.nan],
                    [0.00980392156862745, -0.009615384615384616, 0.0],
                    [np.nan, -0.009708737864077669, -0.009708737864077669],
                    [np.nan, np.nan, -0.00980392156862745],
                ],
                index=ts.index,
                columns=ts.columns,
            ),
        )

    def test_daily(self):
        ret_12h = pd.DataFrame(
            {
                "a": [0.1, 0.1, 0.1, 0.1, 0.1],
                "b": [-0.1, -0.1, -0.1, -0.1, -0.1],
                "c": [0.1, -0.1, 0.1, -0.1, 0.1],
            },
            index=pd.DatetimeIndex(
                [
                    datetime(2020, 1, 1, 0),
                    datetime(2020, 1, 1, 12),
                    datetime(2020, 1, 2, 0),
                    datetime(2020, 1, 2, 12),
                    datetime(2020, 1, 3, 0),
                ]
            ),
        )
        assert_series_equal(
            ret_12h["a"].vbt.returns.daily(),
            pd.Series(
                np.array([0.21, 0.21, 0.1]),
                index=pd.DatetimeIndex(
                    ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]", freq="D"
                ),
                name=ret_12h["a"].name,
            ),
        )
        assert_frame_equal(
            ret_12h.vbt.returns.daily(),
            pd.DataFrame(
                np.array([[0.21, -0.19, -0.01], [0.21, -0.19, -0.01], [0.1, -0.1, 0.1]]),
                index=pd.DatetimeIndex(
                    ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]", freq="D"
                ),
                columns=ret_12h.columns,
            ),
        )
        assert_frame_equal(
            np.log(ret_12h + 1).vbt.returns(log_returns=True).daily(),
            pd.DataFrame(
                np.array(
                    [
                        [0.19062035960864987, -0.21072103131565256, -0.010050335853501347],
                        [0.19062035960864987, -0.21072103131565256, -0.010050335853501347],
                        [0.09531017980432493, -0.10536051565782628, 0.09531017980432493],
                    ]
                ),
                index=pd.DatetimeIndex(
                    ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]", freq="D"
                ),
                columns=ret_12h.columns,
            ),
        )
        assert_frame_equal(
            ret_12h.vbt.returns.daily(jitted=dict(parallel=True)),
            ret_12h.vbt.returns.daily(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_12h.vbt.returns.daily(chunked=True),
            ret_12h.vbt.returns.daily(chunked=False),
        )

    def test_annual(self):
        assert_series_equal(
            ret_acc["a"].annual(),
            pd.Series(
                np.array([0.03960396039603964]),
                index=pd.DatetimeIndex(["2020-01-01"], dtype="datetime64[ns]", freq="365D"),
                name=rets["a"].name,
            ),
        )
        assert_frame_equal(
            ret_acc.annual(),
            pd.DataFrame(
                np.array([[0.03960396039603964, -0.03809523809523796, 0.0]]),
                index=pd.DatetimeIndex(["2020-01-01"], dtype="datetime64[ns]", freq="365D"),
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            log_ret_acc.annual(),
            pd.DataFrame(
                np.array([[0.03883983331626386, -0.03883983331626381, 1.5612511283791264e-17]]),
                index=pd.DatetimeIndex(["2020-01-01"], dtype="datetime64[ns]", freq="365D"),
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.annual(jitted=dict(parallel=True)),
            ret_acc.annual(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.annual(chunked=True),
            ret_acc.annual(chunked=False),
        )

    def test_mirror(self):
        assert_series_equal(
            ret_acc["a"].mirror(),
            pd.Series(
                [
                    np.nan,
                    -0.009803921568627416,
                    -0.009708737864077666,
                    -0.009615384615384581,
                    -0.00952380952380949,
                ],
                index=rets.index,
                name="a",
            ),
        )
        assert_frame_equal(
            ret_acc.mirror(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [-0.009803921568627416, 0.009615384615384581, -0.009803921568627416],
                    [-0.009708737864077666, 0.009708737864077666, -0.009708737864077666],
                    [-0.009615384615384581, 0.009803921568627416, 0.009803921568627416],
                    [-0.00952380952380949, 0.00990099009900991, 0.00990099009900991],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.mirror(jitted=dict(parallel=True)),
            ret_acc.mirror(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.mirror(chunked=True),
            ret_acc.mirror(chunked=False),
        )
        assert_frame_equal(
            prepare_sim_range_obj(ret_acc.mirror(sim_start=sim_start, sim_end=sim_end)),
            sim_range_ret_acc.mirror(),
        )

    def test_cumulative(self):
        assert_series_equal(
            ret_acc["a"].cumulative(),
            pd.Series(
                [
                    0.0,
                    0.00990099009900991,
                    0.01980198019801982,
                    0.02970297029702973,
                    0.03960396039603964,
                ],
                index=rets.index,
                name="a",
            ),
        )
        assert_frame_equal(
            ret_acc.cumulative(),
            pd.DataFrame(
                [
                    [0.0, 0.0, 0.0],
                    [0.00990099009900991, -0.00952380952380949, 0.00990099009900991],
                    [0.01980198019801982, -0.01904761904761898, 0.01980198019801982],
                    [0.02970297029702973, -0.02857142857142847, 0.00990099009900991],
                    [0.03960396039603964, -0.03809523809523796, 0.0],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.cumulative(start_value=2),
            (ret_acc.cumulative() + 1) * 2,
        )
        assert_frame_equal(
            log_ret_acc.cumulative(),
            log_ret_acc.obj.cumsum().fillna(0.0),
        )
        assert_frame_equal(
            log_ret_acc.cumulative(start_value=2),
            np.exp(log_ret_acc.obj.cumsum().fillna(0.0)) * 2,
        )
        assert_frame_equal(
            ret_acc.cumulative(jitted=dict(parallel=True)),
            ret_acc.cumulative(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.cumulative(chunked=True),
            ret_acc.cumulative(chunked=False),
        )
        assert_frame_equal(
            prepare_sim_range_obj(ret_acc.cumulative(sim_start=sim_start, sim_end=sim_end)),
            sim_range_ret_acc.cumulative(),
        )

    def test_total(self):
        assert isclose(ret_acc["a"].total(), 0.03960396039603964)
        assert_series_equal(
            ret_acc.total(),
            ret_acc.cumulative().iloc[-1].rename("total_return"),
        )
        assert_series_equal(
            log_ret_acc.total(),
            log_ret_acc.cumulative().iloc[-1].rename("total_return"),
        )
        assert_series_equal(
            ret_acc.total(jitted=dict(parallel=True)),
            ret_acc.total(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.total(chunked=True),
            ret_acc.total(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_total(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [0.00990099009900991, -0.00952380952380949, 0.00990099009900991],
                    [0.01980198019801982, -0.01904761904761898, 0.01980198019801982],
                    [0.02970297029702973, -0.02857142857142847, 0.00990099009900991],
                    [0.03960396039603964, -0.03809523809523796, 0.0],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_total(jitted=dict(parallel=True)),
            ret_acc.rolling_total(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_total(chunked=True),
            ret_acc.rolling_total(chunked=False),
        )
        assert_series_equal(
            ret_acc.total(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.total(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(ret_acc.rolling_total(sim_start=sim_start, sim_end=sim_end)),
            sim_range_ret_acc.rolling_total(),
        )

    def test_annualized(self):
        assert isclose(ret_acc["a"].annualized(), 16.03564361105591)
        assert_series_equal(
            ret_acc.annualized(),
            pd.Series(
                [16.03564361105591, -0.94129954683068, 0.0],
                index=rets.columns,
                name="annualized_return",
            ),
        )
        assert_series_equal(
            log_ret_acc.annualized(),
            ret_acc.annualized(),
        )
        assert_series_equal(
            ret_acc.annualized(jitted=dict(parallel=True)),
            ret_acc.annualized(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.annualized(chunked=True),
            ret_acc.annualized(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_annualized(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [5.037826528125304, -0.8256036698501655, 5.037826528125304],
                    [9.866637897725285, -0.9036546070678746, 9.866637897725285],
                    [13.453294546738768, -0.9290026509302941, 1.4571989191201644],
                    [16.03564361105591, -0.94129954683068, 0.0],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            log_ret_acc.rolling_annualized(),
            ret_acc.rolling_annualized(),
        )
        assert_frame_equal(
            ret_acc.rolling_annualized(jitted=dict(parallel=True)),
            ret_acc.rolling_annualized(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_annualized(chunked=True),
            ret_acc.rolling_annualized(chunked=False),
        )
        assert_series_equal(
            ret_acc.annualized(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.annualized(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(ret_acc.rolling_annualized(sim_start=sim_start, sim_end=sim_end)),
            sim_range_ret_acc.rolling_annualized(),
        )

    def test_annualized_volatility(self):
        assert isclose(ret_acc["a"].annualized_volatility(), 0.0023481420320083587)
        assert_series_equal(
            ret_acc.annualized_volatility(),
            pd.Series(
                [0.0023481420320083587, 0.002302976178562684, 0.21629262958084472],
                index=rets.columns,
                name="annualized_volatility",
            ),
        )
        assert_series_equal(
            ret_acc.annualized_volatility(jitted=dict(parallel=True)),
            ret_acc.annualized_volatility(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.annualized_volatility(chunked=True),
            ret_acc.annualized_volatility(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_annualized_volatility(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [0.001311323634839096, 0.0012371113631965429, 0.001311323634839096],
                    [0.0018365163148920949, 0.0017665527106770073, 0.21576707228562383],
                    [0.0023481420320083587, 0.002302976178562684, 0.21629262958084472],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_annualized_volatility(jitted=dict(parallel=True)),
            ret_acc.rolling_annualized_volatility(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_annualized_volatility(chunked=True),
            ret_acc.rolling_annualized_volatility(chunked=False),
        )
        assert_series_equal(
            ret_acc.annualized_volatility(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.annualized_volatility(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_annualized_volatility(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_annualized_volatility(),
        )

    def test_calmar_ratio(self):
        assert isclose(ret_acc["a"].calmar_ratio(), np.nan)
        assert_series_equal(
            ret_acc.calmar_ratio(),
            pd.Series([np.nan, -24.709113104305438, 0.0], index=rets.columns, name="calmar_ratio"),
        )
        assert_series_equal(
            log_ret_acc.calmar_ratio(),
            ret_acc.calmar_ratio(),
        )
        assert_series_equal(
            ret_acc.calmar_ratio(jitted=dict(parallel=True)),
            ret_acc.calmar_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.calmar_ratio(chunked=True),
            ret_acc.calmar_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_calmar_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, -86.68838533426769, np.nan],
                    [np.nan, -47.44186687106358, np.nan],
                    [np.nan, -32.51509278256041, 150.09148866937701],
                    [np.nan, -24.709113104305438, 0.0],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            log_ret_acc.rolling_calmar_ratio(),
            ret_acc.rolling_calmar_ratio(),
        )
        assert_frame_equal(
            ret_acc.rolling_calmar_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_calmar_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_calmar_ratio(chunked=True),
            ret_acc.rolling_calmar_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.calmar_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.calmar_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_calmar_ratio(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_calmar_ratio(),
        )

    def test_omega_ratio(self):
        assert isclose(ret_acc["a"].omega_ratio(), np.inf)
        assert_series_equal(
            ret_acc.omega_ratio(),
            pd.Series([np.inf, 0.0, 0.8183910222656902], index=rets.columns, name="omega_ratio"),
        )
        assert_series_equal(
            ret_acc.omega_ratio(jitted=dict(parallel=True)),
            ret_acc.omega_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.omega_ratio(chunked=True),
            ret_acc.omega_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_omega_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.inf, 0.0, np.inf],
                    [np.inf, 0.0, np.inf],
                    [np.inf, 0.0, 1.6440377723169823],
                    [np.inf, 0.0, 0.8183910222656902],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_omega_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_omega_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_omega_ratio(chunked=True),
            ret_acc.rolling_omega_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.omega_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.omega_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_omega_ratio(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_omega_ratio(),
        )

    def test_sharpe_ratio(self):
        assert isclose(ret_acc["a"].sharpe_ratio(), 1361.2461777659016)
        assert_series_equal(
            ret_acc.sharpe_ratio(),
            pd.Series(
                [1361.2461777659016, -1689.979112534647, -1.6064208208840622],
                index=rets.columns,
                name="sharpe_ratio",
            ),
        )
        assert_series_equal(
            ret_acc.sharpe_ratio(jitted=dict(parallel=True)),
            ret_acc.sharpe_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.sharpe_ratio(chunked=True),
            ret_acc.sharpe_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_sharpe_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [2464.0342730803322, -3118.4766749167784, 2464.0342730803322],
                    [1749.8677615690785, -2193.4424620591994, 3.945000212882215],
                    [1361.2461777659016, -1689.979112534647, -1.6064208208840622],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_sharpe_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_sharpe_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_sharpe_ratio(chunked=True),
            ret_acc.rolling_sharpe_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.sharpe_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.sharpe_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_sharpe_ratio(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_sharpe_ratio(),
        )

    def test_sharpe_ratio_std(self):
        assert_series_equal(
            ret_acc.sharpe_ratio_std(),
            pd.Series(
                [np.nan, np.nan, 0.49885271955248023], index=rets.columns, name="sharpe_ratio_std"
            ),
        )

    def test_prob_sharpe_ratio(self):
        assert_series_equal(
            ret_acc.prob_sharpe_ratio(),
            pd.Series([np.nan, np.nan, 1.0], index=rets.columns, name="prob_sharpe_ratio"),
        )

    def test_deflated_sharpe_ratio(self):
        assert_series_equal(
            ret_acc.deflated_sharpe_ratio(),
            pd.Series([np.nan, np.nan, 0.0], index=rets.columns, name="deflated_sharpe_ratio"),
        )

    def test_downside_risk(self):
        assert isclose(ret_acc["a"].downside_risk(), 0.0050638301522095246)
        assert_series_equal(
            ret_acc.downside_risk(),
            pd.Series(
                [0.0050638301522095246, 0.3756656824931905, 0.2669023399120462],
                index=rets.columns,
                name="downside_risk",
            ),
        )
        assert_series_equal(
            ret_acc.downside_risk(jitted=dict(parallel=True)),
            ret_acc.downside_risk(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.downside_risk(chunked=True),
            ret_acc.downside_risk(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_downside_risk(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [0.001891581502429816, 0.3730018572172636, 0.001891581502429816],
                    [0.002967418472882974, 0.373877650411858, 0.002967418472882974],
                    [0.0040239044223846, 0.3747654964757437, 0.2174060319892523],
                    [0.0050638301522095246, 0.3756656824931905, 0.2669023399120462],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_downside_risk(jitted=dict(parallel=True)),
            ret_acc.rolling_downside_risk(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_downside_risk(chunked=True),
            ret_acc.rolling_downside_risk(chunked=False),
        )
        assert_series_equal(
            ret_acc.downside_risk(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.downside_risk(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_downside_risk(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_downside_risk(),
        )

    def test_sortino_ratio(self):
        assert isclose(ret_acc["a"].sortino_ratio(), -17.496762611302145)
        assert_series_equal(
            ret_acc.sortino_ratio(),
            pd.Series(
                [-17.496762611302145, -19.104703923989362, -13.609685792786445],
                index=rets.columns,
                name="sortino_ratio",
            ),
        )
        assert_series_equal(
            ret_acc.sortino_ratio(jitted=dict(parallel=True)),
            ret_acc.sortino_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.sortino_ratio(chunked=True),
            ret_acc.sortino_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_sortino_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [-19.104973174542803, -19.1049731745428, -19.104973174542803],
                    [-18.148306734731282, -19.104920881294735, -18.148306734731282],
                    [-17.72887710200627, -19.1048316731264, -11.194716317809245],
                    [-17.496762611302145, -19.104703923989362, -13.609685792786445],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_sortino_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_sortino_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_sortino_ratio(chunked=True),
            ret_acc.rolling_sortino_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.sortino_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.sortino_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_sortino_ratio(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_sortino_ratio(),
        )

    def test_information_ratio(self):
        assert isclose(ret_acc["a"].information_ratio(), 0.8660254018606264)
        assert_series_equal(
            ret_acc.information_ratio(),
            pd.Series(
                [0.8660254018606264, -6123.473107502151, 0.8660253864635988],
                index=rets.columns,
                name="information_ratio",
            ),
        )
        assert_series_equal(
            ret_acc.information_ratio(jitted=dict(parallel=True)),
            ret_acc.information_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.information_ratio(chunked=True),
            ret_acc.information_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_information_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, -4999.952049730064, 4999.952049730064],
                    [0.5773502691896257, -5095.1669527186, 1.1547005037375726],
                    [0.8660254018606264, -6123.473107502151, 0.8660253864635988],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_information_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_information_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_information_ratio(chunked=True),
            ret_acc.rolling_information_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.information_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.information_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_information_ratio(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_information_ratio(),
        )

    def test_beta(self):
        assert isclose(ret_acc["a"].beta(), 0.00973323097108315)
        assert_series_equal(
            ret_acc.beta(),
            pd.Series(
                [0.00973323097108315, 0.9806173576309167, 84.20285394440484],
                index=rets.columns,
                name="beta",
            ),
        )
        assert_series_equal(
            ret_acc.beta(jitted=dict(parallel=True)),
            ret_acc.beta(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.beta(chunked=True),
            ret_acc.beta(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_beta(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [1.0, 0.943406593406166, 1.0599883517768343],
                    [0.007365428386426863, 0.9618437196674668, 106.37283149922057],
                    [0.00973323097108315, 0.9806173576309167, 84.20285394440484],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_beta(jitted=dict(parallel=True)),
            ret_acc.rolling_beta(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_beta(chunked=True),
            ret_acc.rolling_beta(chunked=False),
        )
        assert_series_equal(
            ret_acc.beta(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.beta(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(ret_acc.rolling_beta(sim_start=sim_start, sim_end=sim_end)),
            sim_range_ret_acc.rolling_beta(),
        )

    def test_alpha(self):
        assert isclose(ret_acc["a"].alpha(), 23.18752973476371)
        assert_series_equal(
            ret_acc.alpha(),
            pd.Series(
                [23.18752973476371, -0.9991707078650568, 3.063517449349351e101],
                index=rets.columns,
                name="alpha",
            ),
        )
        assert_series_equal(
            ret_acc.alpha(jitted=dict(parallel=True)),
            ret_acc.alpha(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.alpha(chunked=True),
            ret_acc.alpha(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_alpha(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, -np.nan, np.nan],
                    [1.7343904090694195e-11, -0.9990625620245503, 1404.3412272567655],
                    [23.37068500428681, -0.9991178949779492, 9.470372618136523e119],
                    [23.18752973476371, -0.9991707078650568, 3.063517449349351e101],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_alpha(jitted=dict(parallel=True)),
            ret_acc.rolling_alpha(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_alpha(chunked=True),
            ret_acc.rolling_alpha(chunked=False),
        )
        assert_series_equal(
            ret_acc.alpha(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.alpha(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(ret_acc.rolling_alpha(sim_start=sim_start, sim_end=sim_end)),
            sim_range_ret_acc.rolling_alpha(),
        )

    def test_tail_ratio(self):
        assert isclose(ret_acc["a"].tail_ratio(), 1.0266935164903142)
        assert_series_equal(
            ret_acc.tail_ratio(),
            pd.Series(
                [1.0266935164903142, 0.9742484787939328, 1.0098865501523446],
                index=rets.columns,
                name="tail_ratio",
            ),
        )
        assert_series_equal(
            ret_acc.tail_ratio(jitted=dict(parallel=True)),
            ret_acc.tail_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.tail_ratio(chunked=True),
            ret_acc.tail_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_tail_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [1.0, 1.0, 1.0],
                    [1.0089064819396378, 0.991424487851358, 1.0089064819396378],
                    [1.0178043269557133, 0.9828406434758667, 1.2750652979408377],
                    [1.026693516490314, 0.9742484787939328, 1.0098865501523446],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_tail_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_tail_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_tail_ratio(chunked=True),
            ret_acc.rolling_tail_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.tail_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.tail_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(ret_acc.rolling_tail_ratio(sim_start=sim_start, sim_end=sim_end)),
            sim_range_ret_acc.rolling_tail_ratio(),
        )

    def test_value_at_risk(self):
        assert isclose(ret_acc["a"].value_at_risk(), 0.009629387602688543)
        assert_series_equal(
            ret_acc.value_at_risk(),
            pd.Series(
                [0.009629387602688543, -0.009789644012944954, -0.009789644012944954],
                index=rets.columns,
                name="value_at_risk",
            ),
        )
        assert_series_equal(
            ret_acc.value_at_risk(jitted=dict(parallel=True)),
            ret_acc.value_at_risk(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.value_at_risk(chunked=True),
            ret_acc.value_at_risk(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_value_at_risk(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [0.00990099009900991, -0.00952380952380949, 0.00990099009900991],
                    [0.00980877499514654, -0.009610805860805826, 0.00980877499514654],
                    [0.009718256234532641, -0.009699402539208358, -0.007757471920807157],
                    [0.009629387602688545, -0.009789644012944954, -0.009789644012944954],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_value_at_risk(jitted=dict(parallel=True)),
            ret_acc.rolling_value_at_risk(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_value_at_risk(chunked=True),
            ret_acc.rolling_value_at_risk(chunked=False),
        )
        assert_series_equal(
            ret_acc.value_at_risk(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.value_at_risk(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_value_at_risk(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_value_at_risk(),
        )

    def test_cond_value_at_risk(self):
        assert isclose(ret_acc["a"].cond_value_at_risk(), 0.009615384615384581)
        assert_series_equal(
            ret_acc.cond_value_at_risk(),
            pd.Series(
                [0.009615384615384581, -0.009803921568627416, -0.009803921568627416],
                index=rets.columns,
                name="cond_value_at_risk",
            ),
        )
        assert_series_equal(
            ret_acc.cond_value_at_risk(jitted=dict(parallel=True)),
            ret_acc.cond_value_at_risk(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.cond_value_at_risk(chunked=True),
            ret_acc.cond_value_at_risk(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_cond_value_at_risk(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [0.00990099009900991, -0.00952380952380949, 0.00990099009900991],
                    [0.009803921568627416, -0.009615384615384581, 0.009803921568627416],
                    [0.009708737864077666, -0.009708737864077666, -0.009708737864077666],
                    [0.009615384615384581, -0.009803921568627416, -0.009803921568627416],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.rolling_cond_value_at_risk(jitted=dict(parallel=True)),
            ret_acc.rolling_cond_value_at_risk(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_cond_value_at_risk(chunked=True),
            ret_acc.rolling_cond_value_at_risk(chunked=False),
        )
        assert_series_equal(
            ret_acc.cond_value_at_risk(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.cond_value_at_risk(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_cond_value_at_risk(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_cond_value_at_risk(),
        )

    def test_capture_ratio(self):
        assert isclose(ret_acc["a"].capture_ratio(), np.inf)
        assert_series_equal(
            ret_acc.capture_ratio(),
            pd.Series(
                [np.inf, -0.05870045316931919, -0.0], index=rets.columns, name="capture_ratio"
            ),
        )
        assert_series_equal(
            log_ret_acc.capture_ratio(),
            ret_acc.capture_ratio(),
        )
        assert_series_equal(
            ret_acc.capture_ratio(jitted=dict(parallel=True)),
            ret_acc.capture_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.capture_ratio(chunked=True),
            ret_acc.capture_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_capture_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [1.0, -0.16388092468864593, -6.101991442261386],
                    [1.0, -0.09158688262758773, -10.918594140453699],
                    [9.232297917748712, -0.06905391446703253, -1.5685627136380502],
                    [np.inf, -0.05870045316931919, -0.0],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            log_ret_acc.rolling_capture_ratio(),
            ret_acc.rolling_capture_ratio(),
        )
        assert_frame_equal(
            ret_acc.rolling_capture_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_capture_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_capture_ratio(chunked=True),
            ret_acc.rolling_capture_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.capture_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.capture_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_capture_ratio(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_capture_ratio(),
        )

    def test_up_capture_ratio(self):
        assert isclose(ret_acc["a"].up_capture_ratio(), 5.035323109391956)
        assert_series_equal(
            ret_acc.up_capture_ratio(),
            pd.Series(
                [5.035323109391956, np.nan, np.nan], index=rets.columns, name="up_capture_ratio"
            ),
        )
        assert_series_equal(
            log_ret_acc.up_capture_ratio(),
            ret_acc.up_capture_ratio(),
        )
        assert_series_equal(
            ret_acc.up_capture_ratio(jitted=dict(parallel=True)),
            ret_acc.up_capture_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.up_capture_ratio(chunked=True),
            ret_acc.up_capture_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_up_capture_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan],
                    [1.0, np.nan, np.nan],
                    [2.6987011457489407, np.nan, np.nan],
                    [5.035323109391956, np.nan, np.nan],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            log_ret_acc.rolling_up_capture_ratio(),
            ret_acc.rolling_up_capture_ratio(),
        )
        assert_frame_equal(
            ret_acc.rolling_up_capture_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_up_capture_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_up_capture_ratio(chunked=True),
            ret_acc.rolling_up_capture_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.up_capture_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.up_capture_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_up_capture_ratio(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_up_capture_ratio(),
        )

    def test_down_capture_ratio(self):
        assert isclose(ret_acc["a"].down_capture_ratio(), np.nan)
        assert_series_equal(
            ret_acc.down_capture_ratio(),
            pd.Series(
                [np.nan, np.nan, 0.8084889429645409], index=rets.columns, name="down_capture_ratio"
            ),
        )
        assert_series_equal(
            log_ret_acc.down_capture_ratio(),
            ret_acc.down_capture_ratio(),
        )
        assert_series_equal(
            ret_acc.down_capture_ratio(jitted=dict(parallel=True)),
            ret_acc.down_capture_ratio(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.down_capture_ratio(chunked=True),
            ret_acc.down_capture_ratio(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_down_capture_ratio(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 0.6344947384595495],
                    [np.nan, np.nan, 0.8084889429645409],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            log_ret_acc.rolling_down_capture_ratio(),
            ret_acc.rolling_down_capture_ratio(),
        )
        assert_frame_equal(
            ret_acc.rolling_down_capture_ratio(jitted=dict(parallel=True)),
            ret_acc.rolling_down_capture_ratio(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_down_capture_ratio(chunked=True),
            ret_acc.rolling_down_capture_ratio(chunked=False),
        )
        assert_series_equal(
            ret_acc.down_capture_ratio(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.down_capture_ratio(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_down_capture_ratio(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_down_capture_ratio(),
        )

    def test_drawdown(self):
        assert_series_equal(
            ret_acc["a"].drawdown(),
            pd.Series(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0]), index=rets["a"].index, name=rets["a"].name
            ),
        )
        assert_frame_equal(
            ret_acc.drawdown(),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, -0.00952380952380949, 0.0],
                        [0.0, -0.01904761904761898, 0.0],
                        [0.0, -0.02857142857142847, -0.009708737864077666],
                        [0.0, -0.03809523809523796, -0.01941747572815533],
                    ]
                ),
                index=pd.DatetimeIndex(
                    ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                    dtype="datetime64[ns]",
                    freq="D",
                ),
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            ret_acc.drawdown(jitted=dict(parallel=True)),
            ret_acc.drawdown(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.drawdown(chunked=True),
            ret_acc.drawdown(chunked=False),
        )
        assert_frame_equal(
            prepare_sim_range_obj(ret_acc.drawdown(sim_start=sim_start, sim_end=sim_end)),
            sim_range_ret_acc.drawdown(),
        )

    def test_max_drawdown(self):
        assert isclose(
            ret_acc["a"].max_drawdown(), ret_acc["a"].drawdowns.get_max_drawdown(fill_value=0.0)
        )
        assert_series_equal(
            ret_acc.max_drawdown(), ret_acc.drawdowns.get_max_drawdown(fill_value=0.0)
        )
        assert_series_equal(
            log_ret_acc.max_drawdown(),
            ret_acc.max_drawdown(),
        )
        assert_series_equal(
            ret_acc.max_drawdown(jitted=dict(parallel=True)),
            ret_acc.max_drawdown(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ret_acc.max_drawdown(chunked=True),
            ret_acc.max_drawdown(chunked=False),
        )
        assert_frame_equal(
            ret_acc.rolling_max_drawdown(),
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [0.0, -0.00952380952380949, 0.0],
                    [0.0, -0.01904761904761898, 0.0],
                    [0.0, -0.02857142857142847, -0.009708737864077666],
                    [0.0, -0.03809523809523796, -0.01941747572815533],
                ],
                index=rets.index,
                columns=rets.columns,
            ),
        )
        assert_frame_equal(
            log_ret_acc.rolling_max_drawdown(),
            ret_acc.rolling_max_drawdown(),
        )
        assert_frame_equal(
            ret_acc.rolling_max_drawdown(jitted=dict(parallel=True)),
            ret_acc.rolling_max_drawdown(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ret_acc.rolling_max_drawdown(chunked=True),
            ret_acc.rolling_max_drawdown(chunked=False),
        )
        assert_series_equal(
            ret_acc.max_drawdown(sim_start=sim_start, sim_end=sim_end),
            sim_range_ret_acc.max_drawdown(),
        )
        assert_frame_equal(
            prepare_sim_range_obj(
                ret_acc.rolling_max_drawdown(sim_start=sim_start, sim_end=sim_end)
            ),
            sim_range_ret_acc.rolling_max_drawdown(),
        )

    def test_drawdowns(self):
        assert type(ret_acc["a"].drawdowns) is vbt.Drawdowns
        assert ret_acc["a"].drawdowns.wrapper.freq == rets["a"].vbt.wrapper.freq
        assert ret_acc["a"].drawdowns.wrapper.ndim == rets["a"].ndim
        assert ret_acc.drawdowns.wrapper.ndim == rets.ndim
        assert isclose(
            ret_acc["a"].drawdowns.get_max_drawdown(fill_value=0.0), ret_acc["a"].max_drawdown()
        )
        assert_series_equal(
            ret_acc.drawdowns.get_max_drawdown(fill_value=0.0), ret_acc.max_drawdown()
        )
        assert_series_equal(
            ret_acc.get_drawdowns(sim_start=sim_start, sim_end=sim_end).get_max_drawdown(),
            sim_range_ret_acc.drawdowns.get_max_drawdown(),
        )

    def test_stats(self):
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
            ret_acc.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    0.05029074336005598,
                    0.05029074336005598,
                    503.14480214084097,
                    7.364791593047191,
                    2.875635691169666,
                    pd.Timedelta("4 days 00:00:00"),
                    -110.11311852987644,
                    -12.354556552152719,
                    np.inf,
                    -16.737050776025985,
                    0.00011764488982174458,
                    -1.9997596731600706,
                    1.0036095151455307,
                    np.inf,
                    -0.0033166334744004554,
                    1.0211724831164504e101,
                    28.397734844335613,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            ret_acc.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    3.960396039603964,
                    0.0,
                    1603.564361105591,
                    0.23481420320083587,
                    np.nan,
                    pd.NaT,
                    1361.2461777659016,
                    np.nan,
                    np.inf,
                    -17.496762611302145,
                    0.036274312675500846,
                    0.0,
                    1.0266935164903142,
                    np.inf,
                    0.009629387602688543,
                    23.18752973476371,
                    0.00973323097108315,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            ret_acc.stats(column="a", settings=dict(freq="10 days", year_freq="200 days")),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("50 days 00:00:00"),
                    3.960396039603964,
                    0.0,
                    117.44954624097473,
                    0.054965847441142295,
                    np.nan,
                    pd.NaT,
                    318.643628524138,
                    np.nan,
                    np.inf,
                    -4.095682336490336,
                    0.036274312675500846,
                    0.0,
                    1.0266935164903142,
                    np.inf,
                    0.009629387602688543,
                    0.19072983032602941,
                    0.00973323097108315,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            ret_acc.stats(column="a", settings=dict(bm_returns=None)),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    3.960396039603964,
                    1603.564361105591,
                    0.23481420320083587,
                    np.nan,
                    pd.NaT,
                    1361.2461777659016,
                    np.nan,
                    np.inf,
                    -17.496762611302145,
                    0.036274312675500846,
                    0.0,
                    1.0266935164903142,
                    np.inf,
                    0.009629387602688543,
                ],
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
        assert_series_equal(
            ret_acc.stats(column="a", settings=dict()),
            ret_acc().stats(column="a"),
        )
        assert_series_equal(
            ret_acc.stats(column="a", settings=dict(bm_returns=None)),
            ret_acc(bm_returns=None).stats(column="a"),
        )
        assert_series_equal(ret_acc["c"].stats(), ret_acc.stats(column="c"))
        assert_series_equal(ret_acc["c"].stats(), ret_acc.stats(column="c", group_by=False))
        assert_series_equal(ret_acc(freq="10d").stats(), ret_acc.stats(settings=dict(freq="10d")))
        assert_series_equal(
            ret_acc(freq="d", year_freq="400d").stats(),
            ret_acc.stats(settings=dict(freq="d", year_freq="400d")),
        )
        stats_df = ret_acc.stats(agg_func=None)
        assert stats_df.shape == (3, 20)
        assert_index_equal(stats_df.index, ret_acc.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)
        assert_series_equal(
            ret_acc.replace(sim_start=sim_start, sim_end=sim_end).stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-02 00:00:00"),
                    pd.Timestamp("2020-01-04 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    -0.6191123314012023,
                    0.6886265549631935,
                    273.2541148409902,
                    7.294482567980012,
                    1.9324122479462216,
                    pd.Timedelta("3 days 00:00:00"),
                    87.80842141139925,
                    -34.969658476787785,
                    np.inf,
                    -17.643087229802163,
                    0.00011764488982174458,
                    -1.9997596731600706,
                    0.93116546551751,
                    np.inf,
                    -0.0032283435807447525,
                    3.8022060744258976e117,
                    35.2084282383429,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            ret_acc.stats(settings=dict(sim_start=sim_start, sim_end=sim_end)),
            ret_acc.replace(sim_start=sim_start, sim_end=sim_end).stats(),
        )

    def test_qs(self):
        if qs_available:
            for c in rets.columns:
                assert ret_acc.qs.sharpe(column=c) == qs.stats.sharpe(
                    rets[c].dropna(), periods=365, rf=0.001
                )
                assert ret_acc(freq="h", year_freq="252d").qs.sharpe(column=c) == qs.stats.sharpe(
                    rets[c].dropna(), periods=252 * 24, rf=0.001
                )
                assert ret_acc(freq="h", year_freq="252d").qs.sharpe(
                    column=c, periods=252, periods_per_year=252, rf=0
                ) == qs.stats.sharpe(rets[c].dropna())

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        assert_frame_equal(
            ts.vbt.to_returns().vbt.returns.resample(test_freq).obj,
            (1 + ts.vbt.to_returns()).resample(test_freq).apply(lambda x: x.prod() - 1),
        )
