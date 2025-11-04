import os
from collections import namedtuple
from datetime import datetime
from itertools import product

import pytest
import vectorbtpro as vbt
from numba import njit
from vectorbtpro._dtypes import *

from tests.utils import *

ta_available = True
try:
    pass
except:
    ta_available = False

pandas_ta_available = True
try:
    pass
except:
    pandas_ta_available = False

talib_available = True
try:
    pass
except:
    talib_available = False

technical_available = True
try:
    pass
except:
    technical_available = False

smc_available = True
try:
    pass
except:
    smc_available = False

seed = 42


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


# ############# factory ############# #

ts = pd.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [5.0, 4.0, 3.0, 2.0, 1.0],
        "c": [1.0, 2.0, 3.0, 2.0, 1.0],
    },
    index=pd.date_range("2020", periods=5),
)


class TestFactory:
    def test_row_stack(self):
        ts1 = ts.copy().iloc[:4]
        ts2 = ts.copy()
        ts2.index = pd.date_range("2020-01-05", "2020-01-09")
        F = vbt.IndicatorFactory(
            input_names=["ts"],
            param_names=["p"],
            in_output_names=["inout"],
            output_names=["out"],
        )

        def apply_func(ts, inout, p):
            inout[:] = p
            return ts * p

        I = F.with_apply_func(apply_func)
        indicator1 = I.run(ts1, [0, 1])
        indicator2 = I.run(ts2, [0, 1])
        new_indicator = I.row_stack((indicator1, indicator2))
        assert_frame_equal(
            new_indicator.ts,
            pd.concat((indicator1.ts, indicator2.ts), axis=0),
        )
        assert_frame_equal(
            new_indicator.inout,
            pd.concat((indicator1.inout, indicator2.inout), axis=0),
        )
        assert_frame_equal(
            new_indicator.out,
            pd.concat((indicator1.out, indicator2.out), axis=0),
        )

    def test_column_stack(self):
        F = vbt.IndicatorFactory(
            input_names=["ts"],
            param_names=["p"],
            in_output_names=["inout"],
            output_names=["out"],
        )

        def apply_func(ts, inout, p):
            inout[:] = p
            return ts * p

        I = F.with_apply_func(apply_func)
        indicator1 = I.run(ts, 0)
        indicator2 = I.run(ts, [1, 2])
        new_indicator = I.column_stack((indicator1, indicator2))
        np.testing.assert_array_equal(
            new_indicator._input_mapper,
            np.concatenate((indicator1._input_mapper, indicator2._input_mapper)),
        )
        assert_frame_equal(
            new_indicator.ts,
            pd.concat((indicator1.ts, indicator2.ts), axis=1),
        )
        assert_frame_equal(
            new_indicator.inout,
            pd.concat((indicator1.inout, indicator2.inout), axis=1),
        )
        assert_frame_equal(
            new_indicator.out,
            pd.concat((indicator1.out, indicator2.out), axis=1),
        )
        assert new_indicator.p_list == indicator1.p_list + indicator2.p_list
        assert_index_equal(
            new_indicator._p_mapper,
            indicator1._p_mapper.append(indicator2._p_mapper),
        )

    def test_config(self, tmp_path):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def apply_func(ts, p, a, b=10):
            return ts * p + a + b

        I = F.with_apply_func(apply_func, var_args=True)
        I._rec_id = "123456789"
        vbt.RecInfo(I._rec_id, I).register()
        indicator = I.run(ts, [0, 1], 10, b=100)
        assert I.loads(indicator.dumps()).equals(indicator)
        indicator.save(tmp_path / "indicator")
        assert I.load(tmp_path / "indicator").equals(indicator)
        indicator.save(tmp_path / "indicator", file_format="ini")
        assert I.load(tmp_path / "indicator", file_format="ini").equals(indicator)

    def test_with_custom_func(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def apply_func(i, ts, p, a, b=10, per_column=False):
            if per_column:
                return ts[:, i : i + 1] * p[i] + a + b
            return ts * p[i] + a + b

        @njit
        def apply_func_nb(i, ts, p, a, b, per_column):
            if per_column:
                return ts[:, i : i + 1] * p[i] + a + b
            return ts * p[i] + a + b  # numba doesn't support **kwargs

        def custom_func(ts, p, *args, **kwargs):
            return vbt.base.combining.apply_and_concat(len(p), apply_func, ts, p, *args, **kwargs)

        @njit
        def custom_func_nb(ts, p, *args):
            return vbt.base.combining.apply_and_concat_one_nb(len(p), apply_func_nb, ts, p, *args)

        target = pd.DataFrame(
            np.array(
                [
                    [110.0, 110.0, 110.0, 111.0, 115.0, 111.0],
                    [110.0, 110.0, 110.0, 112.0, 114.0, 112.0],
                    [110.0, 110.0, 110.0, 113.0, 113.0, 113.0],
                    [110.0, 110.0, 110.0, 114.0, 112.0, 112.0],
                    [110.0, 110.0, 110.0, 115.0, 111.0, 111.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_custom_func(custom_func, var_args=True, per_column=False)
            .run(ts, [0, 1], 10, b=100)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_custom_func(custom_func_nb, var_args=True, per_column=False)
            .run(ts, [0, 1], 10, 100)
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [
                    [110.0, 115.0, 112.0],
                    [110.0, 114.0, 114.0],
                    [110.0, 113.0, 116.0],
                    [110.0, 112.0, 114.0],
                    [110.0, 111.0, 112.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (1, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(
            F.with_custom_func(custom_func, var_args=True)
            .run(ts, [0, 1, 2], 10, b=100, per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_custom_func(custom_func_nb, var_args=True)
            .run(ts, [0, 1, 2], 10, 100, per_column=True)
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [[110.0, 111.0], [110.0, 112.0], [110.0, 113.0], [110.0, 114.0], [110.0, 115.0]]
            ),
            index=ts.index,
            columns=pd.Index([0, 1], dtype="int64", name="custom_p"),
        )
        assert_frame_equal(
            F.with_custom_func(custom_func, var_args=True, per_column=False)
            .run(ts["a"], [0, 1], 10, b=100)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_custom_func(custom_func_nb, var_args=True, per_column=False)
            .run(ts["a"], [0, 1], 10, 100)
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array([[110.0], [110.0], [110.0], [110.0], [110.0]]),
            index=ts.index,
            columns=pd.Index([0], dtype="int64", name="custom_p"),
        )
        assert_frame_equal(
            F.with_custom_func(custom_func, var_args=True)
            .run(ts[["a"]], 0, 10, b=100, per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_custom_func(custom_func_nb, var_args=True)
            .run(ts[["a"]], 0, 10, 100, per_column=True)
            .out,
            target,
        )
        target = pd.Series(np.array([110.0, 110.0, 110.0, 110.0, 110.0]), index=ts.index)
        assert_series_equal(
            F.with_custom_func(custom_func, var_args=True, per_column=False)
            .run(ts["a"], 0, 10, b=100)
            .out,
            target,
        )
        assert_series_equal(
            F.with_custom_func(custom_func_nb, var_args=True, per_column=False)
            .run(ts["a"], 0, 10, 100)
            .out,
            target,
        )
        assert_series_equal(
            F.with_custom_func(custom_func, var_args=True)
            .run(ts["a"], 0, 10, b=100, per_column=True)
            .out,
            target,
        )
        assert_series_equal(
            F.with_custom_func(custom_func_nb, var_args=True)
            .run(ts["a"], 0, 10, 100, per_column=True)
            .out,
            target,
        )

    def test_with_apply_func(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def apply_func(ts, p, a, b=10):
            return ts * p + a + b

        @njit
        def apply_func_nb(ts, p, a, b):
            return ts * p + a + b  # numba doesn't support **kwargs

        target = pd.DataFrame(
            np.array(
                [
                    [110.0, 110.0, 110.0, 111.0, 115.0, 111.0],
                    [110.0, 110.0, 110.0, 112.0, 114.0, 112.0],
                    [110.0, 110.0, 110.0, 113.0, 113.0, 113.0],
                    [110.0, 110.0, 110.0, 114.0, 112.0, 112.0],
                    [110.0, 110.0, 110.0, 115.0, 111.0, 111.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, var_args=True).run(ts, [0, 1], 10, b=100).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, var_args=True).run(ts, [0, 1], 10, 100).out,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [
                    [110.0, 115.0, 112.0],
                    [110.0, 114.0, 114.0],
                    [110.0, 113.0, 116.0],
                    [110.0, 112.0, 114.0],
                    [110.0, 111.0, 112.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (1, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, var_args=True)
            .run(ts, [0, 1, 2], 10, b=100, per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, var_args=True)
            .run(ts, [0, 1, 2], 10, 100, per_column=True)
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [[110.0, 111.0], [110.0, 112.0], [110.0, 113.0], [110.0, 114.0], [110.0, 115.0]]
            ),
            index=ts.index,
            columns=pd.Index([0, 1], dtype="int64", name="custom_p"),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, var_args=True).run(ts["a"], [0, 1], 10, b=100).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, var_args=True)
            .run(ts["a"], [0, 1], 10, 100)
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array([[110.0], [110.0], [110.0], [110.0], [110.0]]),
            index=ts.index,
            columns=pd.Index([0], dtype="int64", name="custom_p"),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, var_args=True)
            .run(ts[["a"]], 0, 10, b=100, per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, var_args=True)
            .run(ts[["a"]], 0, 10, 100, per_column=True)
            .out,
            target,
        )
        target = pd.Series(np.array([110.0, 110.0, 110.0, 110.0, 110.0]), index=ts.index)
        assert_series_equal(
            F.with_apply_func(apply_func, var_args=True).run(ts["a"], 0, 10, b=100).out,
            target,
        )
        assert_series_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, var_args=True)
            .run(ts["a"], 0, 10, 100)
            .out,
            target,
        )
        assert_series_equal(
            F.with_apply_func(apply_func, var_args=True)
            .run(ts["a"], 0, 10, b=100, per_column=True)
            .out,
            target,
        )
        assert_series_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, var_args=True)
            .run(ts["a"], 0, 10, 100, per_column=True)
            .out,
            target,
        )

    def test_templates(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def apply_func(ts, p, a, b=10):
            return ts * p + a + b

        target = pd.DataFrame(
            np.array(
                [
                    [110.0, 110.0, 110.0, 111.0, 115.0, 111.0],
                    [110.0, 110.0, 110.0, 112.0, 114.0, 112.0],
                    [110.0, 110.0, 110.0, 113.0, 113.0, 113.0],
                    [110.0, 110.0, 110.0, 114.0, 112.0, 112.0],
                    [110.0, 110.0, 110.0, 115.0, 111.0, 111.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, var_args=True)
            .run(ts, [0, 1], vbt.RepEval("10"), b=vbt.Rep("b"), template_context=dict(b=100))
            .out,
            target,
        )

    def test_no_inputs(self):
        F = vbt.IndicatorFactory(param_names=["p"], output_names=["out"])

        def apply_func(p):
            return np.full((3, 3), p)

        @njit
        def apply_func_nb(p):
            return np.full((3, 3), p)

        target = pd.DataFrame(
            np.array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]),
            index=pd.RangeIndex(start=0, stop=3, step=1),
            columns=pd.MultiIndex.from_tuples(
                [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(F.with_apply_func(apply_func).run([0, 1]).out, target)
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True).run([0, 1]).out, target
        )
        with pytest.raises(Exception):
            F.with_apply_func(apply_func).run([0, 1], per_column=True)

    def test_input_shape(self):
        F = vbt.IndicatorFactory(param_names=["p"], output_names=["out"])

        def apply_func(input_shape, p):
            return np.full(input_shape, p)

        @njit
        def apply_func_nb(input_shape, p):
            return np.full(input_shape, p)

        target = pd.Series(np.array([0, 0, 0, 0, 0]), index=pd.RangeIndex(start=0, stop=5, step=1))
        assert_series_equal(
            F.with_apply_func(apply_func, require_input_shape=True).run(5, 0).out, target
        )
        assert_series_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, require_input_shape=True)
            .run(5, 0)
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
            index=pd.RangeIndex(start=0, stop=5, step=1),
            columns=pd.Index([0, 1], dtype="int64", name="custom_p"),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, require_input_shape=True).run(5, [0, 1]).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, require_input_shape=True)
            .run(5, [0, 1])
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, require_input_shape=True)
            .run((5, 3), [0, 1], input_index=ts.index, input_columns=ts.columns)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, require_input_shape=True)
            .run((5, 3), [0, 1], input_index=ts.index, input_columns=ts.columns)
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (1, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, require_input_shape=True)
            .run((5, 3), [0, 1, 2], input_index=ts.index, input_columns=ts.columns, per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, require_input_shape=True)
            .run((5, 3), [0, 1, 2], input_index=ts.index, input_columns=ts.columns, per_column=True)
            .out,
            target,
        )

    def test_multiple_inputs(self):
        F = vbt.IndicatorFactory(
            input_names=["ts1", "ts2"], param_names=["p"], output_names=["out"]
        )

        def apply_func(ts1, ts2, p):
            return ts1 * ts2 * p

        @njit
        def apply_func_nb(ts1, ts2, p):
            return ts1 * ts2 * p

        target = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 25.0, 1.0],
                    [0.0, 0.0, 0.0, 4.0, 16.0, 4.0],
                    [0.0, 0.0, 0.0, 9.0, 9.0, 9.0],
                    [0.0, 0.0, 0.0, 16.0, 4.0, 4.0],
                    [0.0, 0.0, 0.0, 25.0, 1.0, 1.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(F.with_apply_func(apply_func).run(ts, ts, [0, 1]).out, target)
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True).run(ts, ts, [0, 1]).out,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [
                    [0.0, 25.0, 2.0],
                    [0.0, 16.0, 8.0],
                    [0.0, 9.0, 18.0],
                    [0.0, 4.0, 8.0],
                    [0.0, 1.0, 2.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (1, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, ts, [0, 1, 2], per_column=True).out, target
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts, ts, [0, 1, 2], per_column=True)
            .out,
            target,
        )

    def test_takes_1d(self):
        F = vbt.IndicatorFactory(
            input_names=["ts1", "ts2"],
            param_names=["p1", "p2"],
            in_output_names=["in_out1", "in_out2"],
            output_names=["out1", "out2"],
        )

        def apply_func(ts1, ts2, in_out1, in_out2, p1, p2):
            in_out1[::2] = ts1[::2] * ts2[::2] * (p1 + p2)
            in_out2[::2] = ts1[::2] * ts2[::2] * (p1 + p2)
            return ts1 * p1, ts2 * p2

        @njit
        def apply_func_nb(ts1, ts2, in_out1, in_out2, p1, p2):
            in_out1[::2] = ts1[::2] * ts2[::2] * (p1 + p2)
            in_out2[::2] = ts1[::2] * ts2[::2] * (p1 + p2)
            return ts1 * p1, ts2 * p2

        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out1,
            F.with_apply_func(apply_func, takes_1d=False)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out2,
            F.with_apply_func(apply_func, takes_1d=False)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out1,
            F.with_apply_func(apply_func, takes_1d=False)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out2,
            F.with_apply_func(apply_func, takes_1d=False)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, takes_1d=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out1,
            F.with_apply_func(apply_func_nb, takes_1d=False)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, takes_1d=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out2,
            F.with_apply_func(apply_func_nb, takes_1d=False)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, takes_1d=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out1,
            F.with_apply_func(apply_func_nb, takes_1d=False)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, takes_1d=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out2,
            F.with_apply_func(apply_func_nb, takes_1d=False)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out2,
        )

        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out1,
            F.with_apply_func(apply_func, takes_1d=False)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out2,
            F.with_apply_func(apply_func, takes_1d=False)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out1,
            F.with_apply_func(apply_func, takes_1d=False)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out2,
            F.with_apply_func(apply_func, takes_1d=False)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, takes_1d=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out1,
            F.with_apply_func(apply_func_nb, takes_1d=False)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, takes_1d=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out2,
            F.with_apply_func(apply_func_nb, takes_1d=False)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, takes_1d=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out1,
            F.with_apply_func(apply_func_nb, takes_1d=False)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, takes_1d=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out2,
            F.with_apply_func(apply_func_nb, takes_1d=False)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out2,
        )

        def apply_func(ts1, ts2, in_out1, in_out2, p1, p2):
            in_out1.iloc[::2] = ts1[::2] * ts2[::2] * (p1 + p2)
            in_out2.iloc[::2] = ts1[::2] * ts2[::2] * (p1 + p2)
            return ts1 * p1, ts2 * p2

        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True, keep_pd=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out1,
            F.with_apply_func(apply_func, takes_1d=False, keep_pd=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True, keep_pd=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out2,
            F.with_apply_func(apply_func, takes_1d=False, keep_pd=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True, keep_pd=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out1,
            F.with_apply_func(apply_func, takes_1d=False, keep_pd=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True, keep_pd=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out2,
            F.with_apply_func(apply_func, takes_1d=False, keep_pd=True)
            .run(ts, 0, [1, 2], 3, in_out1=4.0, in_out2=5.0)
            .in_out2,
        )

        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True, keep_pd=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out1,
            F.with_apply_func(apply_func, takes_1d=False, keep_pd=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True, keep_pd=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out2,
            F.with_apply_func(apply_func, takes_1d=False, keep_pd=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True, keep_pd=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out1,
            F.with_apply_func(apply_func, takes_1d=False, keep_pd=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out1,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, takes_1d=True, keep_pd=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out2,
            F.with_apply_func(apply_func, takes_1d=False, keep_pd=True)
            .run(ts, 0, [1, 2, 3], 3, in_out1=4.0, in_out2=5.0, per_column=True)
            .in_out2,
        )

    def test_no_params(self):
        F = vbt.IndicatorFactory(input_names=["ts"], output_names=["out"])

        def apply_func(ts):
            return ts * 2

        @njit
        def apply_func_nb(ts):
            return ts * 2

        assert_frame_equal(F.with_apply_func(apply_func).run(ts).out, ts * 2)
        assert_frame_equal(F.with_apply_func(apply_func_nb, jitted_loop=True).run(ts).out, ts * 2)

    def test_no_inputs_and_params(self):
        F = vbt.IndicatorFactory(output_names=["out"])

        def apply_func():
            return np.full((3, 3), 1)

        @njit
        def apply_func_nb():
            return np.full((3, 3), 1)

        assert_frame_equal(
            F.with_apply_func(apply_func).run().out, pd.DataFrame(np.full((3, 3), 1))
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True).run().out,
            pd.DataFrame(np.full((3, 3), 1)),
        )
        with pytest.raises(Exception):
            F.with_apply_func(apply_func).run(per_column=True)

    def test_multiple_params(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p1", "p2"], output_names=["out"])

        def apply_func(ts, p1, p2):
            return ts * (p1 + p2)

        @njit
        def apply_func_nb(ts, p1, p2):
            return ts * (p1 + p2)

        target = pd.DataFrame(
            np.array(
                [
                    [2.0, 10.0, 2.0, 3.0, 15.0, 3.0],
                    [4.0, 8.0, 4.0, 6.0, 12.0, 6.0],
                    [6.0, 6.0, 6.0, 9.0, 9.0, 9.0],
                    [8.0, 4.0, 4.0, 12.0, 6.0, 6.0],
                    [10.0, 2.0, 2.0, 15.0, 3.0, 3.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, 2, "a"), (0, 2, "b"), (0, 2, "c"), (1, 2, "a"), (1, 2, "b"), (1, 2, "c")],
                names=["custom_p1", "custom_p2", None],
            ),
        )
        assert_frame_equal(F.with_apply_func(apply_func).run(ts, np.array([0, 1]), 2).out, target)
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True).run(ts, np.array([0, 1]), 2).out,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [
                    [2.0, 15.0, 4.0],
                    [4.0, 12.0, 8.0],
                    [6.0, 9.0, 12.0],
                    [8.0, 6.0, 8.0],
                    [10.0, 3.0, 4.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, 2, "a"), (1, 2, "b"), (2, 2, "c")],
                names=["custom_p1", "custom_p2", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, np.array([0, 1, 2]), 2, per_column=True).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts, np.array([0, 1, 2]), 2, per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, np.array([0, 1, 2]), [2], per_column=True).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func)
            .run(ts, np.array([0, 1, 2]), np.array([2]), per_column=True)
            .out,
            target,
        )
        with pytest.raises(Exception):
            F.with_apply_func(apply_func).run(ts, np.array([0, 1]), 2, per_column=True)
        with pytest.raises(Exception):
            F.with_apply_func(apply_func).run(ts, np.array([0, 1, 2, 3]), 2, per_column=True)

    def test_param_settings(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def apply_func(ts, p):
            return ts * p

        @njit
        def apply_func_nb(ts, p):
            return ts * p

        target = pd.DataFrame(
            np.array(
                [
                    [0.0, 5.0, 2.0],
                    [0.0, 4.0, 4.0],
                    [0.0, 3.0, 6.0],
                    [0.0, 2.0, 4.0],
                    [0.0, 1.0, 2.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [
                    ("array", "a"),
                    ("array", "b"),
                    ("array", "c"),
                ],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func)
            .run(ts, np.array([0, 1, 2]), param_settings={"p": {"is_array_like": True}})
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts, np.array([0, 1, 2]), param_settings={"p": {"is_array_like": True}})
            .out,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [
                    [0.0, 5.0, 2.0],
                    [0.0, 4.0, 4.0],
                    [0.0, 3.0, 6.0],
                    [0.0, 2.0, 4.0],
                    [0.0, 1.0, 2.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [
                    (0, "a"),
                    (1, "b"),
                    (2, "c"),
                ],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func)
            .run(
                ts,
                np.array([0, 1, 2]),
                param_settings={"p": {"is_array_like": True, "bc_to_input": 1, "per_column": True}},
            )
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(
                ts,
                np.array([0, 1, 2]),
                param_settings={"p": {"is_array_like": True, "bc_to_input": 1, "per_column": True}},
            )
            .out,
            target,
        )

        def apply_func2(ts, p):
            return ts * np.expand_dims(p, 1)

        @njit
        def apply_func2_nb(ts, p):
            return ts * np.expand_dims(p, 1)

        target = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 4.0, 2.0],
                    [6.0, 6.0, 6.0],
                    [12.0, 6.0, 6.0],
                    [20.0, 4.0, 4.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [
                    ("array", "a"),
                    ("array", "b"),
                    ("array", "c"),
                ],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func2)
            .run(
                ts,
                np.array([0, 1, 2, 3, 4]),
                param_settings={"p": {"is_array_like": True, "bc_to_input": 0}},
            )
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func2_nb)
            .run(
                ts,
                np.array([0, 1, 2, 3, 4]),
                param_settings={"p": {"is_array_like": True, "bc_to_input": 0}},
            )
            .out,
            target,
        )

        def apply_func3(ts, p):
            return ts * (p[0] + p[1])

        @njit
        def apply_func3_nb(ts, p):
            return ts * (p[0] + p[1])

        target = pd.DataFrame(
            np.array(
                [
                    [1.0, 5.0, 1.0],
                    [2.0, 4.0, 2.0],
                    [3.0, 3.0, 3.0],
                    [4.0, 2.0, 2.0],
                    [5.0, 1.0, 1.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [
                    ("tuple", "a"),
                    ("tuple", "b"),
                    ("tuple", "c"),
                ],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func3)
            .run(ts, (0, 1), param_settings={"p": {"is_tuple": True}})
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func3_nb)
            .run(ts, (0, 1), param_settings={"p": {"is_tuple": True}})
            .out,
            target,
        )

    def test_param_product(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p1", "p2"], output_names=["out"])

        def apply_func(ts, p1, p2):
            return ts * (p1 + p2)

        @njit
        def apply_func_nb(ts, p1, p2):
            return ts * (p1 + p2)

        target = pd.DataFrame(
            np.array(
                [
                    [2.0, 10.0, 2.0, 3.0, 15.0, 3.0, 3.0, 15.0, 3.0, 4.0, 20.0, 4.0],
                    [4.0, 8.0, 4.0, 6.0, 12.0, 6.0, 6.0, 12.0, 6.0, 8.0, 16.0, 8.0],
                    [6.0, 6.0, 6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 12.0, 12.0, 12.0],
                    [8.0, 4.0, 4.0, 12.0, 6.0, 6.0, 12.0, 6.0, 6.0, 16.0, 8.0, 8.0],
                    [10.0, 2.0, 2.0, 15.0, 3.0, 3.0, 15.0, 3.0, 3.0, 20.0, 4.0, 4.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [
                    (0, 2, "a"),
                    (0, 2, "b"),
                    (0, 2, "c"),
                    (0, 3, "a"),
                    (0, 3, "b"),
                    (0, 3, "c"),
                    (1, 2, "a"),
                    (1, 2, "b"),
                    (1, 2, "c"),
                    (1, 3, "a"),
                    (1, 3, "b"),
                    (1, 3, "c"),
                ],
                names=["custom_p1", "custom_p2", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, [0, 1], [2, 3], param_product=True).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts, [0, 1], [2, 3], param_product=True)
            .out,
            target,
        )

    def test_default(self):
        F = vbt.IndicatorFactory(
            input_names=["ts1", "ts2"],
            param_names=["p1", "p2"],
            in_output_names=["in_out1", "in_out2"],
            output_names=["out"],
        )

        def apply_func(ts1, ts2, in_out1, in_out2, p1, p2):
            in_out1[::2] = ts1[::2] * ts2[::2] * (p1 + p2)
            in_out2[::2] = ts1[::2] * ts2[::2] * (p1 + p2)
            return ts1 * ts2 * (p1 + p2)

        # default inputs
        assert_frame_equal(
            F.with_apply_func(apply_func, ts2=0).run(ts, [1, 2], 3).out,
            F.with_apply_func(apply_func).run(ts, 0, [1, 2], 3).out,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, ts2=vbt.Ref("ts1")).run(ts, [1, 2], 3).out,
            F.with_apply_func(apply_func).run(ts, ts, [1, 2], 3).out,
        )
        # default params
        assert_frame_equal(
            F.with_apply_func(apply_func, p2=0, hide_default=False).run(ts, ts, [1, 2]).out,
            F.with_apply_func(apply_func, hide_default=False).run(ts, ts, [1, 2], 0).out,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, p2=vbt.Ref("p1"), hide_default=False)
            .run(ts, ts, [1, 2])
            .out,
            F.with_apply_func(apply_func, hide_default=False).run(ts, ts, [1, 2], [1, 2]).out,
        )
        with pytest.raises(Exception):
            assert_frame_equal(
                F.with_apply_func(apply_func, in_out1=1, in_out2=2).run(ts, ts, [1, 2], 3).in_out1,
                F.with_apply_func(apply_func, in_out1=1, in_out2=2).run(ts, ts, [1, 2], 3).in_out2,
            )
        assert_frame_equal(
            F.with_apply_func(apply_func, in_out1=1, in_out2=vbt.Ref("in_out1"))
            .run(ts, ts, [1, 2], 3)
            .in_out2,
            F.with_apply_func(apply_func, in_out1=1, in_out2=1).run(ts, ts, [1, 2], 3).in_out2,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, in_out1=1, in_out2=vbt.Ref("ts1"))
            .run(ts, ts, [1, 2], 3)
            .in_out2,
            F.with_apply_func(apply_func, in_out1=1, in_out2=ts).run(ts, ts, [1, 2], 3).in_out2,
        )

    def test_hide_params(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p1", "p2"], output_names=["out"])

        assert F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2, hide_params=[]).run(
            ts,
            [0, 1],
            2,
        ).out.columns.names == ["custom_p1", "custom_p2", None]
        assert F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2, hide_params=["p2"]).run(
            ts,
            [0, 1],
            2,
        ).out.columns.names == ["custom_p1", None]

    def test_hide_default(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p1", "p2"], output_names=["out"])

        assert F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2, p2=2, hide_default=False).run(
            ts,
            [0, 1],
        ).out.columns.names == ["custom_p1", "custom_p2", None]
        assert F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2, p2=2, hide_default=True).run(
            ts,
            [0, 1],
        ).out.columns.names == ["custom_p1", None]

    def test_multiple_outputs(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["o1", "o2"])

        def apply_func(ts, p):
            return (ts * p, (ts * p) ** 2)

        @njit
        def apply_func_nb(ts, p):
            return (ts * p, (ts * p) ** 2)

        target = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 5.0, 1.0],
                    [0.0, 0.0, 0.0, 2.0, 4.0, 2.0],
                    [0.0, 0.0, 0.0, 3.0, 3.0, 3.0],
                    [0.0, 0.0, 0.0, 4.0, 2.0, 2.0],
                    [0.0, 0.0, 0.0, 5.0, 1.0, 1.0],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(F.with_apply_func(apply_func).run(ts, [0, 1]).o1, target)
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True).run(ts, [0, 1]).o1, target
        )
        assert_frame_equal(
            F.with_apply_func(apply_func)
            .run(ts.vbt.tile(2), [0, 0, 0, 1, 1, 1], per_column=True)
            .o1,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts.vbt.tile(2), [0, 0, 0, 1, 1, 1], per_column=True)
            .o1,
            target,
        )
        target = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 25.0, 1.0],
                    [0.0, 0.0, 0.0, 4.0, 16.0, 4.0],
                    [0.0, 0.0, 0.0, 9.0, 9.0, 9.0],
                    [0.0, 0.0, 0.0, 16.0, 4.0, 4.0],
                    [0.0, 0.0, 0.0, 25.0, 1.0, 1.0],
                ]
            ),
            index=target.index,
            columns=target.columns,
        )
        assert_frame_equal(F.with_apply_func(apply_func).run(ts, [0, 1]).o2, target)
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True).run(ts, [0, 1]).o2, target
        )
        assert_frame_equal(
            F.with_apply_func(apply_func)
            .run(ts.vbt.tile(2), [0, 0, 0, 1, 1, 1], per_column=True)
            .o2,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts.vbt.tile(2), [0, 0, 0, 1, 1, 1], per_column=True)
            .o2,
            target,
        )

    def test_in_outputs(self):
        F = vbt.IndicatorFactory(
            input_names=["ts"],
            param_names=["p"],
            output_names=["out"],
            in_output_names=["in_out"],
        )

        def apply_func(ts, in_out, p):
            in_out[:, 0] = p
            return ts * p

        @njit
        def apply_func_nb(ts, in_out, p):
            in_out[:, 0] = p
            return ts * p

        target = pd.DataFrame(
            np.array(
                [
                    [0, -1, -1, 1, -1, -1],
                    [0, -1, -1, 1, -1, -1],
                    [0, -1, -1, 1, -1, -1],
                    [0, -1, -1, 1, -1, -1],
                    [0, -1, -1, 1, -1, -1],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert F.with_apply_func(apply_func).run(ts, [0, 1])._in_out.dtype == float_
        assert (
            F.with_apply_func(apply_func, in_output_settings={"in_out": {"dtype": int_}})
            .run(ts, [0, 1])
            ._in_out.dtype
            == int_
        )
        assert_frame_equal(F.with_apply_func(apply_func, in_out=-1).run(ts, [0, 1]).in_out, target)
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, in_out=-1).run(ts, [0, 1]).in_out,
            target,
        )
        assert_frame_equal(F.with_apply_func(apply_func).run(ts, [0, 1], in_out=-1).in_out, target)
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True).run(ts, [0, 1], in_out=-1).in_out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func)
            .run(ts, [0, 1], in_out=np.full(ts.shape, -1, dtype=int))
            .in_out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts, [0, 1], in_out=np.full(ts.shape, -1, dtype=int))
            .in_out,
            target,
        )
        target = pd.DataFrame(
            np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (1, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, [0, 1, 2], in_out=-1, per_column=True).in_out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts, [0, 1, 2], in_out=-1, per_column=True)
            .in_out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func)
            .run(ts, [0, 1, 2], in_out=np.full(ts.shape, -1, dtype=int), per_column=True)
            .in_out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True)
            .run(ts, [0, 1, 2], in_out=np.full(ts.shape, -1, dtype=int), per_column=True)
            .in_out,
            target,
        )

    def test_no_outputs(self):
        F = vbt.IndicatorFactory(param_names=["p"], in_output_names=["in_out"])

        def apply_func(in_out, p):
            in_out[:] = p

        @njit
        def apply_func_nb(in_out, p):
            in_out[:] = p

        target = pd.DataFrame(
            np.array(
                [
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, in_output_settings=dict(in_out=dict(dtype=int_)))
            .run([0, 1], input_shape=ts.shape, input_index=ts.index, input_columns=ts.columns)
            .in_out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(
                apply_func_nb, jitted_loop=True, in_output_settings=dict(in_out=dict(dtype=int_))
            )
            .run([0, 1], input_shape=ts.shape, input_index=ts.index, input_columns=ts.columns)
            .in_out,
            target,
        )

    def test_kwargs_as_args(self):
        F = vbt.IndicatorFactory(input_names=["ts"], output_names=["out"])

        def apply_func(ts, kw):
            return ts * kw

        @njit
        def apply_func_nb(ts, kw):
            return ts * kw

        assert_frame_equal(
            F.with_apply_func(apply_func, kwargs_as_args=["kw"]).run(ts, kw=2).out, ts * 2
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, jitted_loop=True, kwargs_as_args=["kw"])
            .run(ts, kw=2)
            .out,
            ts * 2,
        )

    def test_cache(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def cache_func(ts, ps, per_column=False):
            np.random.seed(seed)
            cache = dict()
            for i, p in enumerate(ps):
                if per_column:
                    cache[p] = ts[:, i : i + 1] * p + np.random.uniform(0, 1)
                else:
                    cache[p] = ts * p + np.random.uniform(0, 1)
            return cache

        @njit
        def cache_func_nb(ts, ps, per_column=False):
            np.random.seed(seed)
            cache = dict()
            for i, p in enumerate(ps):
                if per_column:
                    cache[p] = np.ascontiguousarray(ts[:, i : i + 1] * p + np.random.uniform(0, 1))
                else:
                    cache[p] = np.ascontiguousarray(ts * p + np.random.uniform(0, 1))
            return cache

        def apply_func(ts, p, c):
            return c[p]

        @njit
        def apply_func_nb(ts, p, c):
            return c[p]

        target = pd.DataFrame(
            np.array(
                [
                    [
                        0.3745401188473625,
                        0.3745401188473625,
                        0.3745401188473625,
                        1.9507143064099162,
                        5.950714306409916,
                        1.9507143064099162,
                    ],
                    [
                        0.3745401188473625,
                        0.3745401188473625,
                        0.3745401188473625,
                        2.950714306409916,
                        4.950714306409916,
                        2.950714306409916,
                    ],
                    [
                        0.3745401188473625,
                        0.3745401188473625,
                        0.3745401188473625,
                        3.950714306409916,
                        3.950714306409916,
                        3.950714306409916,
                    ],
                    [
                        0.3745401188473625,
                        0.3745401188473625,
                        0.3745401188473625,
                        4.950714306409916,
                        2.950714306409916,
                        2.950714306409916,
                    ],
                    [
                        0.3745401188473625,
                        0.3745401188473625,
                        0.3745401188473625,
                        5.950714306409916,
                        1.9507143064099162,
                        1.9507143064099162,
                    ],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
                names=["custom_p", None],
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, cache_func=cache_func).run(ts, [0, 1]).out, target
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, cache_func=cache_func_nb).run(ts, [0, 1]).out,
            target,
        )
        # return_cache
        target1 = np.array(
            [
                [0.3745401188473625, 0.3745401188473625, 0.3745401188473625],
                [0.3745401188473625, 0.3745401188473625, 0.3745401188473625],
                [0.3745401188473625, 0.3745401188473625, 0.3745401188473625],
                [0.3745401188473625, 0.3745401188473625, 0.3745401188473625],
                [0.3745401188473625, 0.3745401188473625, 0.3745401188473625],
            ]
        )
        target2 = np.array(
            [
                [1.9507143064099162, 5.950714306409916, 1.9507143064099162],
                [2.950714306409916, 4.950714306409916, 2.950714306409916],
                [3.950714306409916, 3.950714306409916, 3.950714306409916],
                [4.950714306409916, 2.950714306409916, 2.950714306409916],
                [5.950714306409916, 1.9507143064099162, 1.9507143064099162],
            ]
        )
        cache = F.with_apply_func(apply_func, cache_func=cache_func).run(
            ts, [0, 1], return_cache=True
        )
        np.testing.assert_array_equal(cache[0], target1)
        np.testing.assert_array_equal(cache[1], target2)
        cache = F.with_apply_func(apply_func_nb, cache_func=cache_func_nb).run(
            ts, [0, 1], return_cache=True
        )
        np.testing.assert_array_equal(cache[0], target1)
        np.testing.assert_array_equal(cache[1], target2)
        # use_cache
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, [0, 1], use_cache=cache).out, target
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb).run(ts, [0, 1], use_cache=cache).out, target
        )

        # per_column
        target = pd.DataFrame(
            np.array(
                [
                    [0.3745401188473625, 5.950714306409916, 2.731993941811405],
                    [0.3745401188473625, 4.950714306409916, 4.731993941811405],
                    [0.3745401188473625, 3.950714306409916, 6.731993941811405],
                    [0.3745401188473625, 2.950714306409916, 4.731993941811405],
                    [0.3745401188473625, 1.9507143064099162, 2.731993941811405],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(0, "a"), (1, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, cache_func=cache_func, cache_pass_per_column=True)
            .run(ts, [0, 1, 2], per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, cache_func=cache_func_nb, cache_pass_per_column=True)
            .run(ts, [0, 1, 2], per_column=True)
            .out,
            target,
        )

    def test_raw(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def apply_func(ts, p, a, b=10):
            return ts * p + a + b

        @njit
        def apply_func_nb(ts, p, a, b):
            return ts * p + a + b

        target = np.array(
            [
                [110.0, 110.0, 110.0, 111.0, 115.0, 111.0],
                [110.0, 110.0, 110.0, 112.0, 114.0, 112.0],
                [110.0, 110.0, 110.0, 113.0, 113.0, 113.0],
                [110.0, 110.0, 110.0, 114.0, 112.0, 112.0],
                [110.0, 110.0, 110.0, 115.0, 111.0, 111.0],
            ]
        )
        np.testing.assert_array_equal(
            F.with_apply_func(apply_func, var_args=True).run(
                ts, [0, 1], 10, b=100, return_raw=True
            )[0][0],
            target,
        )
        np.testing.assert_array_equal(
            F.with_apply_func(apply_func_nb, var_args=True).run(
                ts, [0, 1], 10, 100, return_raw=True
            )[0][0],
            target,
        )
        np.testing.assert_array_equal(
            F.with_apply_func(apply_func, var_args=True).run(
                ts, [0, 1], 10, b=100, return_raw=True
            )[1],
            [(0,), (1,)],
        )
        np.testing.assert_array_equal(
            F.with_apply_func(apply_func_nb, var_args=True).run(
                ts, [0, 1], 10, 100, return_raw=True
            )[1],
            [(0,), (1,)],
        )
        assert (
            F.with_apply_func(apply_func, var_args=True).run(
                ts, [0, 1], 10, b=100, return_raw=True
            )[2]
            == 3
        )
        assert (
            F.with_apply_func(apply_func_nb, var_args=True).run(
                ts, [0, 1], 10, 100, return_raw=True
            )[2]
            == 3
        )
        assert (
            F.with_apply_func(apply_func, var_args=True).run(
                ts, [0, 1], 10, b=100, return_raw=True
            )[3]
            == []
        )
        assert (
            F.with_apply_func(apply_func_nb, var_args=True).run(
                ts, [0, 1], 10, 100, return_raw=True
            )[3]
            == []
        )
        raw_results = F.with_apply_func(apply_func, var_args=True).run(
            ts, [0, 1, 2], 10, b=100, return_raw=True
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, var_args=True)
            .run(ts, [0, 1], 10, b=100, use_raw=raw_results)
            .out,
            F.with_apply_func(apply_func_nb, var_args=True).run(ts, [0, 1], 10, 100).out,
        )

        # per_column
        target = np.array(
            [
                [110.0, 115.0, 112.0],
                [110.0, 114.0, 114.0],
                [110.0, 113.0, 116.0],
                [110.0, 112.0, 114.0],
                [110.0, 111.0, 112.0],
            ]
        )
        np.testing.assert_array_equal(
            F.with_apply_func(apply_func, var_args=True).run(
                ts,
                [0, 1, 2],
                10,
                b=100,
                return_raw=True,
                per_column=True,
            )[0][0],
            target,
        )
        np.testing.assert_array_equal(
            F.with_apply_func(apply_func_nb, var_args=True).run(
                ts,
                [0, 1, 2],
                10,
                100,
                return_raw=True,
                per_column=True,
            )[0][0],
            target,
        )
        np.testing.assert_array_equal(
            F.with_apply_func(apply_func, var_args=True).run(
                ts,
                [0, 1, 2],
                10,
                b=100,
                return_raw=True,
                per_column=True,
            )[1],
            [(0,), (1,), (2,)],
        )
        np.testing.assert_array_equal(
            F.with_apply_func(apply_func_nb, var_args=True).run(
                ts,
                [0, 1, 2],
                10,
                100,
                return_raw=True,
                per_column=True,
            )[1],
            [(0,), (1,), (2,)],
        )
        assert (
            F.with_apply_func(apply_func, var_args=True).run(
                ts, [0, 1, 2], 10, b=100, return_raw=True
            )[2]
            == 3
        )
        assert (
            F.with_apply_func(apply_func_nb, var_args=True).run(
                ts, [0, 1, 2], 10, 100, return_raw=True
            )[2]
            == 3
        )
        assert (
            F.with_apply_func(apply_func, var_args=True).run(
                ts, [0, 1, 2], 10, b=100, return_raw=True
            )[3]
            == []
        )
        assert (
            F.with_apply_func(apply_func_nb, var_args=True).run(
                ts, [0, 1, 2], 10, 100, return_raw=True
            )[3]
            == []
        )
        raw_results = F.with_apply_func(apply_func, var_args=True).run(
            ts, [0, 1, 2], 10, b=100, return_raw=True
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, var_args=True)
            .run(ts, [0, 0, 0], 10, b=100, use_raw=raw_results)
            .out,
            F.with_apply_func(apply_func_nb, var_args=True).run(ts, [0, 0, 0], 10, 100).out,
        )

    @pytest.mark.parametrize(
        "test_to_2d,test_keep_pd", [(False, False), (False, True), (True, False), (True, True)]
    )
    def test_to_2d_and_keep_pd(self, test_to_2d, test_keep_pd):
        F = vbt.IndicatorFactory(
            input_names=["ts"], in_output_names=["in_out"], output_names=["out"]
        )

        def custom_func(_ts, _in_out):
            if test_to_2d:
                assert _ts.ndim == 2
                for __in_out in _in_out:
                    assert __in_out.ndim == 2
                if test_keep_pd:
                    assert_frame_equal(_ts, ts[["a"]].vbt.wrapper.wrap(_ts.values))
                    for __in_out in _in_out:
                        assert_frame_equal(__in_out, ts[["a"]].vbt.wrapper.wrap(__in_out.values))
            else:
                assert _ts.ndim == 1
                for __in_out in _in_out:
                    assert __in_out.ndim == 1
                if test_keep_pd:
                    assert_series_equal(_ts, ts["a"].vbt.wrapper.wrap(_ts.values))
                    for __in_out in _in_out:
                        assert_series_equal(__in_out, ts["a"].vbt.wrapper.wrap(__in_out.values))
            return _ts

        def apply_func(_ts, _in_out):
            if test_to_2d:
                assert _ts.ndim == 2
                assert _in_out.ndim == 2
                if test_keep_pd:
                    assert_frame_equal(_ts, ts[["a"]].vbt.wrapper.wrap(_ts.values))
                    assert_frame_equal(_in_out, ts[["a"]].vbt.wrapper.wrap(_in_out.values))
            else:
                assert _ts.ndim == 1
                assert _in_out.ndim == 1
                if test_keep_pd:
                    assert_series_equal(_ts, ts["a"].vbt.wrapper.wrap(_ts.values))
                    assert_series_equal(_in_out, ts["a"].vbt.wrapper.wrap(_in_out.values))
            return _ts

        F.with_custom_func(custom_func, to_2d=test_to_2d, keep_pd=test_keep_pd, var_args=True).run(
            ts["a"]
        )
        F.with_apply_func(apply_func, to_2d=test_to_2d, keep_pd=test_keep_pd, var_args=True).run(
            ts["a"]
        )

        def custom_func(_ts, _in_out, col=None):
            if test_to_2d:
                assert _ts.ndim == 2
                for __in_out in _in_out:
                    assert __in_out.ndim == 2
                if test_keep_pd:
                    assert_frame_equal(_ts, ts.iloc[:, [col]].vbt.wrapper.wrap(_ts.values))
                    for __in_out in _in_out:
                        assert_frame_equal(
                            __in_out, ts.iloc[:, [col]].vbt.wrapper.wrap(__in_out.values)
                        )
            else:
                assert _ts.ndim == 1
                for __in_out in _in_out:
                    assert __in_out.ndim == 1
                if test_keep_pd:
                    assert_series_equal(_ts, ts.iloc[:, col].vbt.wrapper.wrap(_ts.values))
                    for __in_out in _in_out:
                        assert_series_equal(
                            __in_out, ts.iloc[:, col].vbt.wrapper.wrap(__in_out.values)
                        )
            return _ts

        def apply_func(col, _ts, _in_out):
            if test_to_2d:
                assert _ts.ndim == 2
                assert _in_out.ndim == 2
                if test_keep_pd:
                    assert_frame_equal(_ts, ts.iloc[:, [col]].vbt.wrapper.wrap(_ts.values))
                    assert_frame_equal(_in_out, ts.iloc[:, [col]].vbt.wrapper.wrap(_in_out.values))
            else:
                assert _ts.ndim == 1
                assert _in_out.ndim == 1
                if test_keep_pd:
                    assert_series_equal(_ts, ts.iloc[:, col].vbt.wrapper.wrap(_ts.values))
                    assert_series_equal(_in_out, ts.iloc[:, col].vbt.wrapper.wrap(_in_out.values))
            return _ts

    def test_pass_packed(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def custom_func(input_list, in_output_list, param_list, per_column=False):
            return input_list[0] * param_list[0][0]

        @njit
        def custom_func_nb(input_list, in_output_list, param_list, per_column=False):
            return input_list[0] * param_list[0][0]

        target = pd.DataFrame(
            ts.values * 2,
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(2, "a"), (2, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(F.with_custom_func(custom_func, pass_packed=True).run(ts, 2).out, target)
        assert_frame_equal(
            F.with_custom_func(custom_func_nb, pass_packed=True).run(ts, 2).out, target
        )
        assert_frame_equal(
            F.with_custom_func(custom_func, pass_packed=True).run(ts, 2, per_column=True).out,
            target,
        )
        assert_frame_equal(
            F.with_custom_func(custom_func_nb, pass_packed=True).run(ts, 2, per_column=True).out,
            target,
        )

        def apply_func(input_tuple, in_output_tuple, param_tuple):
            return input_tuple[0] * param_tuple[0]

        @njit
        def apply_func_nb(input_tuple, in_output_tuple, param_tuple):
            return input_tuple[0] * param_tuple[0]

        target = pd.DataFrame(
            ts.values * 2,
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(2, "a"), (2, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(F.with_apply_func(apply_func, pass_packed=True).run(ts, 2).out, target)
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, pass_packed=True).run(ts, 2).out, target
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, pass_packed=True).run(ts, 2, per_column=True).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, pass_packed=True).run(ts, 2, per_column=True).out,
            target,
        )

    def test_select_params(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])

        def apply_func(i, ts, p):
            return ts[i] * p[i]

        @njit
        def apply_func_nb(i, ts, p):
            return ts[i] * p[i]

        target = pd.DataFrame(
            ts.values * 2,
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(2, "a"), (2, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, pass_packed=False, select_params=False).run(ts, 2).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, pass_packed=False, select_params=False).run(ts, 2).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, pass_packed=False, select_params=False)
            .run(ts, 2, per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, pass_packed=False, select_params=False)
            .run(ts, 2, per_column=True)
            .out,
            target,
        )

        def apply_func(i, input_tuple, in_output_tuple, param_tuple):
            return input_tuple[0][i] * param_tuple[0][i]

        @njit
        def apply_func_nb(i, input_tuple, in_output_tuple, param_tuple):
            return input_tuple[0][i] * param_tuple[0][i]

        target = pd.DataFrame(
            ts.values * 2,
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [(2, "a"), (2, "b"), (2, "c")], names=["custom_p", None]
            ),
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, pass_packed=True, select_params=False).run(ts, 2).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, pass_packed=True, select_params=False).run(ts, 2).out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func, pass_packed=True, select_params=False)
            .run(ts, 2, per_column=True)
            .out,
            target,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func_nb, pass_packed=True, select_params=False)
            .run(ts, 2, per_column=True)
            .out,
            target,
        )

    def test_other(self):
        F = vbt.IndicatorFactory(input_names=["ts"], output_names=["o1", "o2"])

        def custom_func(ts):
            return ts, ts + 1, ts + 2

        @njit
        def custom_func_nb(ts):
            return ts, ts + 1, ts + 2

        obj, other = F.with_custom_func(custom_func).run(ts)
        np.testing.assert_array_equal(other, ts + 2)
        obj, other = F.with_custom_func(custom_func_nb).run(ts)
        np.testing.assert_array_equal(other, ts + 2)

    def test_run_unique(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p1", "p2"], output_names=["out"])

        def apply_func(ts, p1, p2):
            return ts * (p1 + p2)

        assert_series_equal(
            F.with_apply_func(apply_func).run(ts["a"], 2, 3, run_unique=True).out,
            F.with_apply_func(apply_func).run(ts["a"], 2, 3, run_unique=False).out,
        )
        raw = F.with_apply_func(apply_func).run(
            ts["a"], [2, 2, 2], [3, 3, 3], run_unique=True, return_raw=True
        )
        np.testing.assert_array_equal(raw[0][0], np.array([[5.0], [10.0], [15.0], [20.0], [25.0]]))
        assert raw[1] == [(2, 3)]
        assert raw[2] == 1
        assert raw[3] == []
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts["a"], [2, 2, 2], [3, 3, 3], run_unique=True).out,
            F.with_apply_func(apply_func).run(ts["a"], [2, 2, 2], [3, 3, 3], run_unique=False).out,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, 2, 3, run_unique=True).out,
            F.with_apply_func(apply_func).run(ts, 2, 3, run_unique=False).out,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, [2, 2, 2], [3, 3, 3], run_unique=True).out,
            F.with_apply_func(apply_func).run(ts, [2, 2, 2], [3, 3, 3], run_unique=False).out,
        )
        assert_frame_equal(
            F.with_apply_func(apply_func).run(ts, [2, 3, 4], [4, 3, 2], run_unique=True).out,
            F.with_apply_func(apply_func).run(ts, [2, 3, 4], [4, 3, 2], run_unique=False).out,
        )

    def test_run_combs(self):
        # itertools.combinations
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p1", "p2"], output_names=["out"])

        ind1 = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(
            ts,
            [2, 2, 3],
            [10, 10, 11],
            short_name="custom_1",
        )
        ind2 = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(
            ts,
            [3, 4, 4],
            [11, 12, 12],
            short_name="custom_2",
        )
        ind1_1, ind2_1 = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run_combs(
            ts,
            [2, 3, 4],
            [10, 11, 12],
            r=2,
            run_unique=False,
        )
        ind1_2, ind2_2 = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run_combs(
            ts,
            [2, 3, 4],
            [10, 11, 12],
            r=2,
            run_unique=True,
        )
        assert_frame_equal(ind1.out, ind1_1.out)
        assert_frame_equal(ind2.out, ind2_1.out)
        assert_frame_equal(ind1.out, ind1_2.out)
        assert_frame_equal(ind2.out, ind2_2.out)
        # itertools.product
        ind3 = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(
            ts,
            [2, 2, 2, 3, 3, 3, 4, 4, 4],
            [10, 10, 10, 11, 11, 11, 12, 12, 12],
            short_name="custom_1",
        )
        ind4 = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(
            ts,
            [2, 3, 4, 2, 3, 4, 2, 3, 4],
            [10, 11, 12, 10, 11, 12, 10, 11, 12],
            short_name="custom_2",
        )
        ind3_1, ind4_1 = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run_combs(
            ts,
            [2, 3, 4],
            [10, 11, 12],
            r=2,
            comb_func=product,
            run_unique=False,
        )
        ind3_2, ind4_2 = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run_combs(
            ts,
            [2, 3, 4],
            [10, 11, 12],
            r=2,
            comb_func=product,
            run_unique=True,
        )
        assert_frame_equal(ind3.out, ind3_1.out)
        assert_frame_equal(ind4.out, ind4_1.out)
        assert_frame_equal(ind3.out, ind3_2.out)
        assert_frame_equal(ind4.out, ind4_2.out)

    def test_skipna(self):
        F = vbt.IndicatorFactory(
            input_names=["in1", "in2"], param_names=["p"], output_names=["o1", "o2"]
        )

        def apply_func(in1, in2, p):
            return in1 * p, in2 * p

        Ind = F.with_apply_func(apply_func)
        in1 = np.arange(0, 20).reshape((10, 2)).astype(float_)
        in1[1:3, 0] = np.nan
        in1[2:4, 1] = np.nan
        in2 = np.arange(20, 40).reshape((10, 2)).astype(float_)
        in2[5:7, 0] = np.nan
        in2[6:8, 1] = np.nan
        ind_1d = Ind.run(in1[:, 0], in2[:, 0], [1, 2, 3], skipna=True)
        ind_12d = Ind.run(in1[:, [0]], in2[:, [0]], [1, 2, 3], skipna=True)
        np.testing.assert_array_equal(ind_12d.o1.values, ind_1d.o1.values)
        np.testing.assert_array_equal(ind_12d.o2.values, ind_1d.o2.values)
        ind_2d = Ind.run(in1, in2, [1, 2, 3], skipna=True, split_columns=True)
        np.testing.assert_array_equal(ind_2d.o1.values[:, [0, 2, 4]], ind_12d.o1.values)
        np.testing.assert_array_equal(ind_2d.o2.values[:, [0, 2, 4]], ind_12d.o2.values)
        np.testing.assert_array_equal(
            ind_2d.o1.values,
            np.array(
                [
                    [0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
                    [np.nan, 3.0, np.nan, 6.0, np.nan, 9.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [6.0, np.nan, 12.0, np.nan, 18.0, np.nan],
                    [8.0, 9.0, 16.0, 18.0, 24.0, 27.0],
                    [np.nan, 11.0, np.nan, 22.0, np.nan, 33.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [14.0, np.nan, 28.0, np.nan, 42.0, np.nan],
                    [16.0, 17.0, 32.0, 34.0, 48.0, 51.0],
                    [18.0, 19.0, 36.0, 38.0, 54.0, 57.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            ind_2d.o2.values,
            np.array(
                [
                    [20.0, 21.0, 40.0, 42.0, 60.0, 63.0],
                    [np.nan, 23.0, np.nan, 46.0, np.nan, 69.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [26.0, np.nan, 52.0, np.nan, 78.0, np.nan],
                    [28.0, 29.0, 56.0, 58.0, 84.0, 87.0],
                    [np.nan, 31.0, np.nan, 62.0, np.nan, 93.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [34.0, np.nan, 68.0, np.nan, 102.0, np.nan],
                    [36.0, 37.0, 72.0, 74.0, 108.0, 111.0],
                    [38.0, 39.0, 76.0, 78.0, 114.0, 117.0],
                ]
            ),
        )

    def test_wrapper(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p1", "p2"], output_names=["out"])

        obj = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(ts["a"], 0, 1)
        assert obj.wrapper.ndim == 1
        assert_index_equal(obj.wrapper.index, ts.index)
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples([(0, 1)], names=["custom_p1", "custom_p2"]),
        )
        obj = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(ts["a"], [0, 1], 2)
        assert obj.wrapper.ndim == 2
        assert_index_equal(obj.wrapper.index, ts.index)
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (0, 2),
                    (1, 2),
                ],
                names=["custom_p1", "custom_p2"],
            ),
        )
        obj = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(ts, 0, 1)
        assert obj.wrapper.ndim == 2
        assert_index_equal(obj.wrapper.index, ts.index)
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(0, 1, "a"), (0, 1, "b"), (0, 1, "c")], names=["custom_p1", "custom_p2", None]
            ),
        )
        obj = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(ts, [1, 2], 3)
        assert obj.wrapper.ndim == 2
        assert_index_equal(obj.wrapper.index, ts.index)
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(1, 3, "a"), (1, 3, "b"), (1, 3, "c"), (2, 3, "a"), (2, 3, "b"), (2, 3, "c")],
                names=["custom_p1", "custom_p2", None],
            ),
        )
        obj = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(ts["a"], 0, 1, per_column=True)
        assert obj.wrapper.ndim == 1
        assert_index_equal(obj.wrapper.index, ts.index)
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples([(0, 1)], names=["custom_p1", "custom_p2"]),
        )
        obj = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(
            ts[["a"]], 0, 1, per_column=True
        )
        assert obj.wrapper.ndim == 2
        assert_index_equal(obj.wrapper.index, ts.index)
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples([(0, 1)], names=["custom_p1", "custom_p2"]),
        )
        obj = F.with_apply_func(lambda ts, p1, p2: ts * p1 * p2).run(ts, 0, 1, per_column=True)
        assert obj.wrapper.ndim == 2
        assert_index_equal(obj.wrapper.index, ts.index)
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(0, 1, "a"), (0, 1, "b"), (0, 1, "c")], names=["custom_p1", "custom_p2", None]
            ),
        )

    @pytest.mark.parametrize(
        "test_config",
        [
            lambda F, shape, *args, **kwargs: F.with_apply_func(
                lambda input_shape, p1, p2: np.empty(input_shape) * p1 * p2,
                require_input_shape=True,
            ).run(shape, *args, **kwargs),
            lambda F, shape, *args, **kwargs: F.with_apply_func(
                lambda p1, p2: np.full(shape, p1 + p2)
            ).run(
                *args,
                **kwargs,
            ),
        ],
    )
    def test_no_inputs_wrapper(self, test_config):
        F = vbt.IndicatorFactory(param_names=["p1", "p2"], output_names=["out"])

        obj = test_config(F, (5,), 0, 1)
        assert obj.wrapper.ndim == 1
        assert_index_equal(obj.wrapper.index, pd.RangeIndex(start=0, stop=5, step=1))
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (0, 1),
                ],
                names=["custom_p1", "custom_p2"],
            ),
        )
        obj = test_config(F, (5,), [0, 1], 2)
        assert obj.wrapper.ndim == 2
        assert_index_equal(obj.wrapper.index, pd.RangeIndex(start=0, stop=5, step=1))
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples([(0, 2), (1, 2)], names=["custom_p1", "custom_p2"]),
        )
        obj = test_config(F, (5, 3), [0, 1], 2)
        assert obj.wrapper.ndim == 2
        assert_index_equal(obj.wrapper.index, pd.RangeIndex(start=0, stop=5, step=1))
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2)],
                names=["custom_p1", "custom_p2", None],
            ),
        )
        obj = test_config(F, ts.shape, [0, 1], 2, input_index=ts.index, input_columns=ts.columns)
        assert obj.wrapper.ndim == ts.ndim
        assert_index_equal(obj.wrapper.index, ts.index)
        assert_index_equal(
            obj.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(0, 2, "a"), (0, 2, "b"), (0, 2, "c"), (1, 2, "a"), (1, 2, "b"), (1, 2, "c")],
                names=["custom_p1", "custom_p2", None],
            ),
        )

    def test_mappers(self):
        F = vbt.IndicatorFactory(input_names=["ts"], param_names=["p1", "p2"], output_names=["out"])

        obj = F.with_apply_func(lambda ts, p1, p2: ts * (p1 + p2)).run(ts, 0, 2)
        np.testing.assert_array_equal(obj._input_mapper, np.array([0, 1, 2]))
        np.testing.assert_array_equal(obj._p1_mapper, np.array([0, 0, 0]))
        np.testing.assert_array_equal(obj._p2_mapper, np.array([2, 2, 2]))
        assert_index_equal(
            obj._tuple_mapper,
            pd.MultiIndex.from_tuples([(0, 2), (0, 2), (0, 2)], names=["custom_p1", "custom_p2"]),
        )
        obj = F.with_apply_func(lambda ts, p1, p2: ts * (p1 + p2)).run(ts, [0, 1], [1, 2])
        np.testing.assert_array_equal(obj._input_mapper, np.array([0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(obj._p1_mapper, np.array([0, 0, 0, 1, 1, 1]))
        np.testing.assert_array_equal(obj._p2_mapper, np.array([1, 1, 1, 2, 2, 2]))
        assert_index_equal(
            obj._tuple_mapper,
            pd.MultiIndex.from_tuples(
                [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (1, 2)], names=["custom_p1", "custom_p2"]
            ),
        )
        obj = F.with_apply_func(lambda ts, p1, p2: ts * (p1 + p2)).run(
            ts, [0, 1, 2], 3, per_column=True
        )
        np.testing.assert_array_equal(obj._input_mapper, np.array([0, 1, 2]))
        np.testing.assert_array_equal(obj._p1_mapper, np.array([0, 1, 2]))
        np.testing.assert_array_equal(obj._p2_mapper, np.array([3, 3, 3]))
        assert_index_equal(
            obj._tuple_mapper,
            pd.MultiIndex.from_tuples([(0, 3), (1, 3), (2, 3)], names=["custom_p1", "custom_p2"]),
        )

    def test_properties(self):
        I = vbt.IndicatorFactory(
            input_names=["ts1", "ts2"],
            param_names=["p1", "p2"],
            output_names=["o1", "o2"],
            in_output_names=["in_o1", "in_o2"],
            output_flags={"o1": "Hello"},
        ).with_apply_func(lambda ts1, ts2, p1, p2, in_o1, in_o2: (ts1, ts2))
        obj = I.run(ts, ts, [0, 1], 2)

        # Class properties
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ("p1", "p2")
        assert I.output_names == ("o1", "o2")
        assert I.in_output_names == ("in_o1", "in_o2")
        assert I.output_flags == {"o1": "Hello"}

        # Instance properties
        assert obj.input_names == ("ts1", "ts2")
        assert obj.param_names == ("p1", "p2")
        assert obj.output_names == ("o1", "o2")
        assert obj.in_output_names == ("in_o1", "in_o2")
        assert obj.output_flags == {"o1": "Hello"}
        assert obj.short_name == "custom"
        assert obj.level_names == ("custom_p1", "custom_p2")
        assert obj.p1_list == [0, 1]
        assert obj.p2_list == [2, 2]

    @pytest.mark.parametrize(
        "test_attr", ["ts1", "ts2", "o1", "o2", "in_o1", "in_o2", "co1", "co2"]
    )
    def test_indexing(self, test_attr):
        obj = (
            vbt.IndicatorFactory(
                input_names=["ts1", "ts2"],
                param_names=["p1", "p2"],
                output_names=["o1", "o2"],
                in_output_names=["in_o1", "in_o2"],
                lazy_outputs={
                    "co1": lambda self: self.ts1 + self.ts2,
                    "co2": lambda self: self.o1 + self.o2,
                },
            )
            .with_apply_func(lambda ts1, ts2, p1, p2, in_o1, in_o2: (ts1, ts2))
            .run(ts, ts + 1, [1, 2], 3)
        )

        assert_frame_equal(
            getattr(obj.iloc[np.arange(3), np.arange(3)], test_attr),
            getattr(obj, test_attr).iloc[np.arange(3), np.arange(3)],
        )
        assert_series_equal(
            getattr(obj.loc[:, (1, 3, "a")], test_attr),
            getattr(obj, test_attr).loc[:, (1, 3, "a")],
        )
        assert_frame_equal(
            getattr(obj.loc[:, (1, 3)], test_attr), getattr(obj, test_attr).loc[:, (1, 3)]
        )
        assert_frame_equal(getattr(obj[(1, 3)], test_attr), getattr(obj, test_attr)[(1, 3)])
        assert_frame_equal(
            getattr(obj.xs(1, axis=1, level=0), test_attr),
            getattr(obj, test_attr).xs(1, axis=1, level=0),
        )
        assert_frame_equal(
            getattr(obj.p1_loc[2], test_attr),
            getattr(obj, test_attr).xs(2, level="custom_p1", axis=1),
        )
        assert_frame_equal(
            getattr(obj.p1_loc[1:2], test_attr),
            pd.concat(
                (
                    getattr(obj, test_attr).xs(1, level="custom_p1", drop_level=False, axis=1),
                    getattr(obj, test_attr).xs(2, level="custom_p1", drop_level=False, axis=1),
                ),
                axis=1,
            ),
        )
        assert_frame_equal(
            getattr(obj.p1_loc[[1, 1, 1]], test_attr),
            pd.concat(
                (
                    getattr(obj, test_attr).xs(1, level="custom_p1", drop_level=False, axis=1),
                    getattr(obj, test_attr).xs(1, level="custom_p1", drop_level=False, axis=1),
                    getattr(obj, test_attr).xs(1, level="custom_p1", drop_level=False, axis=1),
                ),
                axis=1,
            ),
        )
        assert_frame_equal(
            getattr(obj.tuple_loc[(1, 3)], test_attr),
            getattr(obj, test_attr).xs((1, 3), level=("custom_p1", "custom_p2"), axis=1),
        )
        assert_frame_equal(
            getattr(obj.tuple_loc[(1, 3) : (2, 3)], test_attr),
            pd.concat(
                (
                    getattr(obj, test_attr).xs(
                        (1, 3), level=("custom_p1", "custom_p2"), drop_level=False, axis=1
                    ),
                    getattr(obj, test_attr).xs(
                        (2, 3), level=("custom_p1", "custom_p2"), drop_level=False, axis=1
                    ),
                ),
                axis=1,
            ),
        )
        assert_frame_equal(
            getattr(obj.tuple_loc[[(1, 3), (1, 3), (1, 3)]], test_attr),
            pd.concat(
                (
                    getattr(obj, test_attr).xs(
                        (1, 3), level=("custom_p1", "custom_p2"), drop_level=False, axis=1
                    ),
                    getattr(obj, test_attr).xs(
                        (1, 3), level=("custom_p1", "custom_p2"), drop_level=False, axis=1
                    ),
                    getattr(obj, test_attr).xs(
                        (1, 3), level=("custom_p1", "custom_p2"), drop_level=False, axis=1
                    ),
                ),
                axis=1,
            ),
        )

    def test_numeric_attr(self):
        obj = (
            vbt.IndicatorFactory(input_names=["ts"], param_names=["p"], output_names=["out"])
            .with_apply_func(lambda ts, p: ts * p)
            .run(ts, 1)
        )

        assert_frame_equal(obj.out_above(2), obj.out > 2)
        target = pd.DataFrame(
            np.array(
                [
                    [False, True, False, False, True, False],
                    [False, True, False, False, True, False],
                    [True, True, True, False, False, False],
                    [True, False, False, True, False, False],
                    [True, False, False, True, False, False],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [
                    (2, 1, "a"),
                    (2, 1, "b"),
                    (2, 1, "c"),
                    (3, 1, "a"),
                    (3, 1, "b"),
                    (3, 1, "c"),
                ],
                names=["custom_out_above", "custom_p", None],
            ),
        )
        assert_frame_equal(obj.out_above([2, 3]), target)
        columns = target.columns.set_names("my_above", level=0)
        assert_frame_equal(
            obj.out_above([2, 3], level_name="my_above"),
            pd.DataFrame(target.values, index=target.index, columns=columns),
        )
        assert_frame_equal(
            obj.out_crossed_above(2),
            pd.DataFrame(
                np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [True, False, True],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                index=ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (1, "a"),
                        (1, "b"),
                        (1, "c"),
                    ],
                    names=["custom_p", None],
                ),
            ),
        )
        assert_series_equal(
            obj.out_stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    5.0,
                    2.6,
                    1.3329792289008184,
                    1.0,
                    2.6666666666666665,
                    4.333333333333333,
                ],
                index=pd.Index(
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
                    ],
                    dtype="object",
                ),
                name="agg_stats",
            ),
        )

    def test_unpacking(self):
        obj = (
            vbt.IndicatorFactory(
                input_names=["ts1", "ts2"],
                param_names=["p1", "p2"],
                output_names=["o1", "o2"],
                in_output_names=["in_o1", "in_o2"],
                lazy_outputs={
                    "co1": lambda self: self.ts1 + self.ts2,
                    "co2": lambda self: self.o1 + self.o2,
                },
            )
            .with_apply_func(lambda ts1, ts2, p1, p2, in_o1, in_o2: (ts1, ts2))
            .run(ts, ts + 1, [1, 2], 3)
        )
        assert_frame_equal(
            obj.unpack()[0],
            obj.o1,
        )
        assert_frame_equal(
            obj.unpack()[1],
            obj.o2,
        )
        assert_frame_equal(
            obj.to_dict()["o1"],
            obj.o1,
        )
        assert_frame_equal(
            obj.to_dict()["o2"],
            obj.o2,
        )
        assert_frame_equal(
            obj.to_dict()["in_o1"],
            obj.in_o1,
        )
        assert_frame_equal(
            obj.to_dict()["in_o2"],
            obj.in_o2,
        )
        assert_frame_equal(
            obj.to_dict()["co1"],
            obj.co1,
        )
        assert_frame_equal(
            obj.to_dict()["co2"],
            obj.co2,
        )
        assert_frame_equal(
            obj.to_frame(),
            pd.concat(
                (
                    obj.o1,
                    obj.o2,
                    obj.in_o1,
                    obj.in_o2,
                    obj.co1,
                    obj.co2,
                ),
                axis=1,
                keys=pd.Index(["o1", "o2", "in_o1", "in_o2", "co1", "co2"], name="output"),
            ),
        )

    def test_boolean_attr(self):
        obj = (
            vbt.IndicatorFactory(
                input_names=["ts"],
                param_names=["p"],
                output_names=["out"],
                attr_settings=dict(out=dict(dtype=np.bool_)),
            )
            .with_apply_func(lambda ts, p: ts > p)
            .run(ts, 2)
        )

        assert_frame_equal(obj.out_and(True), obj.out)
        target = pd.DataFrame(
            np.array(
                [
                    [False, False, False, False, True, False],
                    [False, False, False, False, True, False],
                    [False, False, False, True, True, True],
                    [False, False, False, True, False, False],
                    [False, False, False, True, False, False],
                ]
            ),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples(
                [
                    (False, 2, "a"),
                    (False, 2, "b"),
                    (False, 2, "c"),
                    (True, 2, "a"),
                    (True, 2, "b"),
                    (True, 2, "c"),
                ],
                names=["custom_out_and", "custom_p", None],
            ),
        )
        assert_frame_equal(obj.out_and([False, True]), target)
        columns = target.columns.set_names("my_and", level=0)
        assert_frame_equal(
            obj.out_and([False, True], level_name="my_and"),
            pd.DataFrame(target.values, index=target.index, columns=columns),
        )
        assert_series_equal(
            obj.out_stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    2.3333333333333335,
                    46.666666666666664,
                    pd.Timestamp("2020-01-02 08:00:00"),
                    pd.Timestamp("2020-01-03 16:00:00"),
                    0.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    1.0,
                    55.55555555555555,
                    pd.Timedelta("2 days 08:00:00"),
                    pd.Timedelta("2 days 08:00:00"),
                    pd.Timedelta("2 days 08:00:00"),
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                ],
                index=pd.Index(
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
                ),
                name="agg_stats",
            ),
        )

    def test_mapping_attr(self):
        TestEnum = namedtuple("TestEnum", ["Hello", "World"])(0, 1)
        obj = (
            vbt.IndicatorFactory(output_names=["out"], attr_settings=dict(out=dict(dtype=TestEnum)))
            .with_apply_func(lambda: np.array([[0, 1], [1, -1]]))
            .run()
        )

        assert_frame_equal(obj.out_readable, pd.DataFrame([["Hello", "World"], ["World", None]]))
        assert_series_equal(
            obj.out_stats(),
            pd.Series(
                [0.0, 1.0, 2.0, 0.5, 0.5, 1.0],
                index=pd.Index(
                    [
                        "Start Index",
                        "End Index",
                        "Total Duration",
                        "Value Counts: None",
                        "Value Counts: Hello",
                        "Value Counts: World",
                    ],
                    dtype="object",
                ),
                name="agg_stats",
                dtype=object,
            ),
        )

    def test_stats(self):
        @njit
        def apply_func_nb(ts):
            return ts**2, ts**3

        MyInd = vbt.IndicatorFactory(
            input_names=["ts"],
            output_names=["out1", "out2"],
            metrics=dict(
                sum_diff=dict(
                    calc_func=lambda self, const: self.out2.sum() * self.out1.sum() + const
                )
            ),
            stats_defaults=dict(settings=dict(const=1000)),
        ).with_apply_func(apply_func_nb)

        myind = MyInd.run(ts)
        assert_series_equal(
            myind.stats(),
            pd.Series([9535.0], index=["sum_diff"], name="agg_stats", dtype=object),
        )

    def test_dir(self):
        TestEnum = namedtuple("TestEnum", ["Hello", "World"])(0, 1)
        F = vbt.IndicatorFactory(
            input_names=["ts"],
            output_names=["o1", "o2"],
            in_output_names=["in_out"],
            param_names=["p1", "p2"],
            attr_settings={
                "ts": {"dtype": None},
                "o1": {"dtype": float_},
                "o2": {"dtype": np.bool_},
                "in_out": {"dtype": TestEnum},
            },
        )
        ind = F.with_apply_func(lambda ts, in_out, p1, p2: (ts + in_out, ts + in_out)).run(
            ts, 100, 200
        )
        test_attr_set = {k for k in dir(ind) if not k.startswith("__")}
        assert "p1_list" in test_attr_set
        assert "p2_list" in test_attr_set
        assert "ts" in test_attr_set
        assert "in_out" in test_attr_set
        assert "o1" in test_attr_set
        assert "o2" in test_attr_set
        assert "ts_above" in test_attr_set
        assert "ts_below" in test_attr_set
        assert "ts_equal" in test_attr_set
        assert "ts_crossed_above" in test_attr_set
        assert "ts_crossed_below" in test_attr_set
        assert "ts_stats" in test_attr_set
        assert "in_out_readable" in test_attr_set
        assert "in_out_stats" in test_attr_set
        assert "o1_above" in test_attr_set
        assert "o1_below" in test_attr_set
        assert "o1_equal" in test_attr_set
        assert "o1_crossed_above" in test_attr_set
        assert "o1_crossed_below" in test_attr_set
        assert "o1_stats" in test_attr_set
        assert "o2_and" in test_attr_set
        assert "o2_or" in test_attr_set
        assert "o2_xor" in test_attr_set
        assert "o2_stats" in test_attr_set
        assert "apply_func" in test_attr_set
        assert "custom_func" in test_attr_set
        assert "run" in test_attr_set
        assert "run_combs" in test_attr_set

    def test_from_expr(self):
        I = vbt.IndicatorFactory.from_expr("RollMean: rolling_mean(@in_ts, @p_window)", window=2)
        assert I.__name__ == "RollMean"
        assert I.short_name == "rollmean"
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr("RollMean:rolling_mean(@in_ts, @p_window)", window=2)
        assert I.__name__ == "RollMean"
        assert I.short_name == "rollmean"
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr(
            """
        RollMean:
        rolling_mean(@in_ts, @p_window)
        """,
            window=2,
        )
        assert I.__name__ == "RollMean"
        assert I.short_name == "rollmean"
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr(
            "RollMean[rm]: rolling_mean(@in_ts, @p_window)", window=2
        )
        assert I.__name__ == "RollMean"
        assert I.short_name == "rm"
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))

        I = vbt.IndicatorFactory.from_expr("rolling_mean(@in_ts, @p_window)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr("rolling_mean(@in_ts, window)")
        assert I.input_names == ("ts",)
        assert I.param_names == ()
        assert_frame_equal(I.run(ts, window=2).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr(
            "rolling_mean(@in_ts, window)",
            func_mapping=dict(rolling_mean=dict(func=vbt.nb.rolling_min_nb)),
        )
        assert I.input_names == ("ts",)
        assert I.param_names == ()
        assert_frame_equal(I.run(ts, window=2).out, ts.vbt.rolling_min(2))
        I = vbt.IndicatorFactory.from_expr(
            "hello",
            magnet_inputs=["ts"],
            res_func_mapping=dict(
                hello=dict(func=lambda context: context["ts"], magnet_inputs=["ts"])
            ),
        )
        assert I.input_names == ("ts",)
        assert I.param_names == ()
        assert_frame_equal(I.run(ts).out, ts)
        np.random.seed(42)
        I = vbt.IndicatorFactory.from_expr(
            "ts * rand, ts * rand",
            magnet_inputs=["ts"],
            res_func_mapping=dict(rand=dict(func=lambda: np.random.uniform())),
        )
        assert I.input_names == ("ts",)
        assert I.param_names == ()
        ind = I.run(ts)
        assert_frame_equal(ind.out1, ind.out2)
        I = vbt.IndicatorFactory.from_expr("rolling_mean_nb(@in_ts, @p_window)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr("(rolling_mean(@in_ts, @p_window))", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr(
            "(rolling_mean(@in_ts, @p_window) - 1) * (3 - 1)", window=2
        )
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, (ts.vbt.rolling_mean(2) - 1) * (3 - 1))
        I = vbt.IndicatorFactory.from_expr("(rolling_mean(@in_ts, @p_window),)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr("rolling_mean(@in_ts, @p_window),", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr(
            "rolling_mean(@in_ts1, @p_window1),rolling_mean(@in_ts2, @p_window2)",
            window1=2,
            window2=3,
        )
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ("window1", "window2")
        assert_frame_equal(I.run(ts, ts * 2).out1, ts.vbt.rolling_mean(2))
        assert_frame_equal(I.run(ts, ts * 2).out2, (ts * 2).vbt.rolling_mean(3))
        I = vbt.IndicatorFactory.from_expr("ts_mean(@in_ts, @p_window)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr("returns")
        assert I.input_names == ("close",)
        assert I.param_names == ()
        assert_frame_equal(I.run(ts).out, ts.vbt.to_returns())
        I = vbt.IndicatorFactory.from_expr("random_func(@in_ts)", random_func=lambda x: x)
        assert I.input_names == ("ts",)
        assert I.param_names == ()
        assert_frame_equal(I.run(ts).out, ts)
        I = vbt.IndicatorFactory.from_expr("vbt.nb.rolling_mean_nb(@in_ts, @p_window)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr(
            "random_func()",
            random_func=lambda context: context["close"],
            factory_kwargs=dict(input_names=["close"]),
        )
        assert I.input_names == ("close",)
        assert I.param_names == ()
        assert_frame_equal(I.run(ts).out, ts)
        I = vbt.IndicatorFactory.from_expr("(close + low + high + open + volume) / 5")
        assert I.input_names == ("open", "high", "low", "close", "volume")
        assert I.param_names == ()
        assert_frame_equal(
            I.run(ts, ts * 2, ts * 3, ts * 4, ts * 5).out,
            (ts + ts * 2 + ts * 3 + ts * 4 + ts * 5) / 5,
        )
        I = vbt.IndicatorFactory.from_expr("@in_ts1 + @in_ts2", use_pd_eval=True)
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ()
        assert_frame_equal(I.run(ts, ts * 2).out, ts + ts * 2)

        I = vbt.IndicatorFactory.from_expr("@out_o:rolling_mean(@in_ts, @p_window)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).o, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr("@out_o :rolling_mean(@in_ts, @p_window)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).o, ts.vbt.rolling_mean(2))
        I = vbt.IndicatorFactory.from_expr(
            "@out_o1:rolling_mean(@in_ts1, @p_window1),@out_o2:rolling_mean(@in_ts2, @p_window2)",
            window1=2,
            window2=3,
        )
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ("window1", "window2")
        assert_frame_equal(I.run(ts, ts * 2).o1, ts.vbt.rolling_mean(2))
        assert_frame_equal(I.run(ts, ts * 2).o2, (ts * 2).vbt.rolling_mean(3))
        I = vbt.IndicatorFactory.from_expr(
            """
        @out_o1 = rolling_mean(@in_ts1, @p_window1)
        @out_o2 = rolling_mean(@in_ts2, @p_window2)
        @out_o1, @out_o2
        """,
            window1=2,
            window2=3,
        )
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ("window1", "window2")
        assert_frame_equal(I.run(ts, ts * 2).o1, ts.vbt.rolling_mean(2))
        assert_frame_equal(I.run(ts, ts * 2).o2, (ts * 2).vbt.rolling_mean(3))
        I = vbt.IndicatorFactory.from_expr(
            """
        o1 = rolling_mean(@in_ts1, @p_window1)
        o2 = rolling_mean(@in_ts2, @p_window2)
        o1, o2
        """,
            window1=2,
            window2=3,
        )
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ("window1", "window2")
        assert_frame_equal(I.run(ts, ts * 2).o1, ts.vbt.rolling_mean(2))
        assert_frame_equal(I.run(ts, ts * 2).o2, (ts * 2).vbt.rolling_mean(3))

        I = vbt.IndicatorFactory.from_expr("@talib_sma(@in_ts, @p_window)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_frame_equal(I.run(ts).out, vbt.IF.from_talib("SMA", timeperiod=2).run(ts).real)
        I = vbt.IndicatorFactory.from_expr("@talib_sma(@in_ts, @p_window, skipna=True)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_series_equal(
            I.run(pd.Series([np.nan, 1, np.nan, 2, np.nan, 3])).out,
            vbt.IF.from_talib("SMA", timeperiod=2, skipna=True)
            .run(pd.Series([np.nan, 1, np.nan, 2, np.nan, 3]))
            .real,
        )
        I = vbt.IndicatorFactory.from_expr("@talib_sma(@in_ts, @p_window)", window=2)
        assert I.input_names == ("ts",)
        assert I.param_names == ("window",)
        assert_series_equal(
            I.run(ts["a"], to_2d=False).out,
            vbt.IF.from_talib("SMA", timeperiod=2).run(ts["a"]).real,
        )
        I = vbt.IndicatorFactory.from_expr(
            "@talib_macd(@in_ts, @p_fastperiod, @p_slowperiod, @p_signalperiod)[1]",
            fastperiod=2,
            slowperiod=3,
            signalperiod=4,
        )
        assert I.input_names == ("ts",)
        assert I.param_names == ("fastperiod", "slowperiod", "signalperiod")
        assert_frame_equal(
            I.run(ts).out,
            vbt.IF.from_talib("MACD", fastperiod=2, slowperiod=3, signalperiod=4)
            .run(ts)
            .macdsignal,
        )
        I = vbt.IndicatorFactory.from_expr(
            "@talib_macd(@in_ts, @p_fastperiod, @p_slowperiod, @p_signalperiod)[1]",
            fastperiod=2,
            slowperiod=3,
            signalperiod=4,
        )
        assert I.input_names == ("ts",)
        assert I.param_names == ("fastperiod", "slowperiod", "signalperiod")
        assert_series_equal(
            I.run(ts["a"], to_2d=False).out,
            vbt.IF.from_talib("MACD", fastperiod=2, slowperiod=3, signalperiod=4)
            .run(ts["a"])
            .macdsignal,
        )

        I = vbt.IndicatorFactory.from_expr(
            """
        @settings({
            'factory_kwargs': {
                'input_names': ['ts1', 'ts2'],
                'param_names': ['window1', 'window2'],
                'output_names': ['o3', 'o4']
            },
            'window1': 2,
            'window2': 3
        })
        o1 = rolling_mean(ts1, window1)
        o2 = rolling_mean(ts2, window2)
        o1, o2
        """
        )
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ("window1", "window2")
        assert_frame_equal(I.run(ts, ts * 2).o3, ts.vbt.rolling_mean(2))
        assert_frame_equal(I.run(ts, ts * 2).o4, (ts * 2).vbt.rolling_mean(3))
        I = vbt.IndicatorFactory.from_expr(
            """
        @settings({
            'factory_kwargs': {
                'input_names': ['ts1', 'ts2'],
                'param_names': ['window1', 'window2'],
                'output_names': ['o3', 'o4']
            },
            'window1': 2,
            'window2': 3
        })
        o1 = rolling_mean(ts1, window1)
        o2 = rolling_mean(ts2, window2)
        o1, o2
        """,
            factory_kwargs=dict(output_names=["o1", "o2"]),
        )
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ("window1", "window2")
        assert_frame_equal(I.run(ts, ts * 2).o1, ts.vbt.rolling_mean(2))
        assert_frame_equal(I.run(ts, ts * 2).o2, (ts * 2).vbt.rolling_mean(3))
        I = vbt.IndicatorFactory.from_expr(
            """
        @settings({
            'factory_kwargs': {
                'input_names': ['ts1', 'ts2'],
                'param_names': ['window1', 'window2'],
                'output_names': ['o3', 'o4']
            },
            'window1': 2,
            'window2': 3
        })
        @settings({
            'factory_kwargs': {
                'output_names': ['o1', 'o2']
            },
            'window2': 4
        })
        o1 = rolling_mean(ts1, window1)
        o2 = rolling_mean(ts2, window2)
        o1, o2
        """
        )
        assert I.input_names == ("ts1", "ts2")
        assert I.param_names == ("window1", "window2")
        assert_frame_equal(I.run(ts, ts * 2).o1, ts.vbt.rolling_mean(2))
        assert_frame_equal(I.run(ts, ts * 2).o2, (ts * 2).vbt.rolling_mean(4))

        I = vbt.IndicatorFactory.from_expr("@res_talib_sma", sma_timeperiod=2)
        assert I.input_names == ("close",)
        assert I.param_names == ("sma_timeperiod", "sma_timeframe")
        assert_frame_equal(I.run(ts).out, vbt.IF.from_talib("SMA", timeperiod=2).run(ts).real)
        I = vbt.IndicatorFactory.from_expr(
            "@res_talib_sma * @res_talib_ema", sma_timeperiod=2, ema_timeperiod=3
        )
        assert I.input_names == ("close",)
        assert I.param_names == (
            "sma_timeperiod",
            "sma_timeframe",
            "ema_timeperiod",
            "ema_timeframe",
        )
        assert_frame_equal(
            I.run(ts).out,
            vbt.IF.from_talib("SMA", timeperiod=2).run(ts).real
            * vbt.IF.from_talib("EMA", timeperiod=3).run(ts).real,
        )
        I = vbt.IndicatorFactory.from_expr(
            "@res_talib_sma.real.cumsum()",
            sma_timeperiod=2,
            sma_kwargs=dict(return_raw=False),
        )
        assert I.input_names == ("close",)
        assert I.param_names == ("sma_timeperiod", "sma_timeframe")
        assert_frame_equal(
            I.run(ts).out, vbt.IF.from_talib("SMA", timeperiod=2).run(ts).real.cumsum()
        )

        with pytest.raises(Exception):
            vbt.IndicatorFactory.from_expr(
                "rolling_mean(@in_ts, @p_window)", parse_annotations=False
            )

    def test_from_wqa101(self):
        index = pd.date_range("2020", periods=100)
        columns = pd.MultiIndex.from_tuples(
            [("A", 0, 0, 0), ("B", 0, 0, 0)],
            names=["symbol", "sector", "subindustry", "industry"],
        )
        np.random.seed(42)
        data_dct = {
            "open": pd.DataFrame(np.random.uniform(size=(100, 2)), index=index, columns=columns),
            "high": pd.DataFrame(np.random.uniform(size=(100, 2)), index=index, columns=columns),
            "low": pd.DataFrame(np.random.uniform(size=(100, 2)), index=index, columns=columns),
            "close": pd.DataFrame(np.random.uniform(size=(100, 2)), index=index, columns=columns),
            "volume": pd.DataFrame(np.random.uniform(size=(100, 2)), index=index, columns=columns),
        }
        for i in range(1, 102):
            WQA = vbt.IndicatorFactory.from_wqa101(i)
            wqa = WQA.run(*[data_dct[input_name] for input_name in WQA.input_names])
            assert wqa.out.shape == data_dct["open"].shape

    def test_list_talib_indicators(self):
        if talib_available:
            assert len(vbt.IndicatorFactory.list_talib_indicators()) > 0

    def test_from_talib(self):
        if talib_available:
            SMA = vbt.talib("SMA")
            assert_frame_equal(SMA.run(ts, vbt.Default(2)).real, ts.rolling(2).mean())
            assert_series_equal(
                SMA.run(pd.Series([1, np.nan, 2, np.nan]), vbt.Default(2)).real,
                pd.Series([1, np.nan, 2, np.nan]).rolling(2).mean(),
            )
            assert_series_equal(
                SMA.run(pd.Series([1, np.nan, 2, np.nan, 3]), vbt.Default(2), skipna=True).real,
                pd.Series([np.nan, np.nan, 1.5, np.nan, 2.5]),
            )
            # with params
            target = pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [2.5, 5.5, 2.5],
                        [3.5, 4.5, 3.5],
                        [4.5, 3.5, 3.5],
                        [5.5, 2.5, 2.5],
                    ],
                ),
                index=ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [(2, 2, 2, "a"), (2, 2, 2, "b"), (2, 2, 2, "c")],
                    names=["bbands_timeperiod", "bbands_nbdevup", "bbands_nbdevdn", None],
                ),
            )
            BBANDS = vbt.talib("BBANDS")
            assert_frame_equal(BBANDS.run(ts, timeperiod=2, nbdevup=2, nbdevdn=2).upperband, target)
            assert_frame_equal(
                BBANDS.run(ts, timeperiod=2, nbdevup=2, nbdevdn=2).middleband, target - 1
            )
            assert_frame_equal(
                BBANDS.run(ts, timeperiod=2, nbdevup=2, nbdevdn=2).lowerband, target - 2
            )
            target = pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [2.5, 5.5, 2.5, 2.5, 5.5, 2.5],
                        [3.5, 4.5, 3.5, 3.5, 4.5, 3.5],
                        [4.5, 3.5, 3.5, 4.5, 3.5, 3.5],
                        [5.5, 2.5, 2.5, 5.5, 2.5, 2.5],
                    ]
                ),
                index=ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [
                        (2, 2, 2, "a"),
                        (2, 2, 2, "b"),
                        (2, 2, 2, "c"),
                        (2, 2, 2, "a"),
                        (2, 2, 2, "b"),
                        (2, 2, 2, "c"),
                    ],
                    names=["bbands_timeperiod", "bbands_nbdevup", "bbands_nbdevdn", None],
                ),
            )
            BBANDS = vbt.talib("BBANDS")
            assert_frame_equal(
                BBANDS.run(ts, timeperiod=[2, 2], nbdevup=2, nbdevdn=2).upperband, target
            )
            assert_frame_equal(
                BBANDS.run(ts, timeperiod=[2, 2], nbdevup=2, nbdevdn=2).middleband,
                target - 1,
            )
            assert_frame_equal(
                BBANDS.run(ts, timeperiod=[2, 2], nbdevup=2, nbdevdn=2).lowerband, target - 2
            )
            # without params
            OBV = vbt.talib("OBV")
            assert_frame_equal(
                OBV.run(ts, ts * 2).real,
                pd.DataFrame(
                    np.array(
                        [
                            [2.0, 10.0, 2.0],
                            [6.0, 2.0, 6.0],
                            [12.0, -4.0, 12.0],
                            [20.0, -8.0, 8.0],
                            [30.0, -10.0, 6.0],
                        ],
                    ),
                    index=ts.index,
                    columns=ts.columns,
                ),
            )
            # multiple timeframes
            assert_frame_equal(
                SMA.run(ts, vbt.Default(2), skipna=True, timeframe=["1d", "2d"]).real,
                pd.DataFrame(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [1.5, 4.5, 1.5, np.nan, np.nan, np.nan],
                        [2.5, 3.5, 2.5, np.nan, np.nan, np.nan],
                        [3.5, 2.5, 2.5, 3.0, 3.0, 2.0],
                        [4.5, 1.5, 1.5, 3.0, 3.0, 2.0],
                    ],
                    index=ts.index,
                    columns=pd.MultiIndex.from_tuples(
                        [
                            ("1d", "a"),
                            ("1d", "b"),
                            ("1d", "c"),
                            ("2d", "a"),
                            ("2d", "b"),
                            ("2d", "c"),
                        ],
                        names=["sma_timeframe", None],
                    ),
                ),
            )

    def test_list_pandas_ta_indicators(self):
        if pandas_ta_available:
            assert len(vbt.IndicatorFactory.list_pandas_ta_indicators()) > 0

    def test_from_pandas_ta(self):
        if pandas_ta_available:
            assert_frame_equal(
                vbt.pandas_ta("SMA").run(ts, 2).sma,
                pd.DataFrame(
                    ts.rolling(2).mean().values,
                    index=ts.index,
                    columns=pd.MultiIndex.from_tuples(
                        [(2, "a"), (2, "b"), (2, "c")], names=["sma_length", None]
                    ),
                ),
            )
            assert_frame_equal(
                vbt.pandas_ta("SMA").run(ts["a"], [2, 3, 4]).sma,
                pd.DataFrame(
                    np.column_stack(
                        (
                            ts["a"].rolling(2).mean().values,
                            ts["a"].rolling(3).mean().values,
                            ts["a"].rolling(4).mean().values,
                        )
                    ),
                    index=ts.index,
                    columns=pd.Index([2, 3, 4], dtype="int64", name="sma_length"),
                ),
            )

    def test_list_ta_indicators(self):
        if ta_available:
            assert len(vbt.IndicatorFactory.list_ta_indicators()) > 0

    def test_from_ta(self):
        if ta_available:
            assert_frame_equal(
                vbt.ta("SMAIndicator").run(ts, 2).sma_indicator,
                pd.DataFrame(
                    ts.rolling(2).mean().values,
                    index=ts.index,
                    columns=pd.MultiIndex.from_tuples(
                        [(2, "a"), (2, "b"), (2, "c")],
                        names=["smaindicator_window", None],
                    ),
                ),
            )
            assert_frame_equal(
                vbt.ta("SMAIndicator").run(ts["a"], [2, 3, 4]).sma_indicator,
                pd.DataFrame(
                    np.column_stack(
                        (
                            ts["a"].rolling(2).mean().values,
                            ts["a"].rolling(3).mean().values,
                            ts["a"].rolling(4).mean().values,
                        )
                    ),
                    index=ts.index,
                    columns=pd.Index([2, 3, 4], dtype="int64", name="smaindicator_window"),
                ),
            )
            target = pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [2.5, 5.5, 2.5],
                        [3.5, 4.5, 3.5],
                        [4.5, 3.5, 3.5],
                        [5.5, 2.5, 2.5],
                    ],
                ),
                index=ts.index,
                columns=pd.MultiIndex.from_tuples(
                    [(2, 2, "a"), (2, 2, "b"), (2, 2, "c")],
                    names=["bollingerbands_window", "bollingerbands_window_dev", None],
                ),
            )
            BollingerBands = vbt.ta("BollingerBands")
            assert_frame_equal(
                BollingerBands.run(ts, window=2, window_dev=2).bollinger_hband, target
            )
            assert_frame_equal(
                BollingerBands.run(ts, window=2, window_dev=2).bollinger_mavg, target - 1
            )
            assert_frame_equal(
                BollingerBands.run(ts, window=2, window_dev=2).bollinger_lband, target - 2
            )

    def test_list_technical_indicators(self):
        if technical_available:
            assert len(vbt.IndicatorFactory.list_technical_indicators()) > 0

    def test_from_technical(self):
        if technical_available:
            assert_frame_equal(
                vbt.technical("ROLLING_MIN").run(ts, 2).rolling_min,
                pd.DataFrame(
                    ts.rolling(2).min().values,
                    index=ts.index,
                    columns=pd.MultiIndex.from_tuples(
                        [(2, "a"), (2, "b"), (2, "c")], names=["rolling_min_window", None]
                    ),
                ),
            )
            assert_frame_equal(
                vbt.technical("ROLLING_MIN").run(ts["a"], [2, 3, 4]).rolling_min,
                pd.DataFrame(
                    np.column_stack(
                        (
                            ts["a"].rolling(2).min().values,
                            ts["a"].rolling(3).min().values,
                            ts["a"].rolling(4).min().values,
                        )
                    ),
                    index=ts.index,
                    columns=pd.Index([2, 3, 4], dtype="int64", name="rolling_min_window"),
                ),
            )

    def test_from_custom_techcon(self):
        if technical_available:
            from technical.consensus.consensus import Consensus

            class CustomConsensus(Consensus):
                def __init__(self, dataframe):
                    super().__init__(dataframe)

                    self.evaluate_sma(period=2)
                    self.evaluate_sma(period=3)

            consensus = vbt.IF.from_custom_techcon(CustomConsensus).run(
                open, high, low, close, volume
            )
            assert_frame_equal(
                consensus.buy,
                pd.DataFrame(
                    [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0], [100.0, 0.0], [100.0, 0.0]],
                    index=close.index,
                    columns=close.columns,
                ),
            )
            assert_frame_equal(
                consensus.sell,
                pd.DataFrame(
                    [[0.0, 0.0], [0.0, 50.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]],
                    index=close.index,
                    columns=close.columns,
                ),
            )
            assert_frame_equal(
                consensus.buy_agreement,
                pd.DataFrame(
                    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
                    index=close.index,
                    columns=close.columns,
                ),
            )
            assert_frame_equal(
                consensus.sell_agreement,
                pd.DataFrame(
                    [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
                    index=close.index,
                    columns=close.columns,
                ),
            )
            assert_frame_equal(
                consensus.buy_disagreement,
                pd.DataFrame(
                    [[2.0, 2.0], [1.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
                    index=close.index,
                    columns=close.columns,
                ),
            )
            assert_frame_equal(
                consensus.sell_disagreement,
                pd.DataFrame(
                    [[2.0, 2.0], [2.0, 1.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
                    index=close.index,
                    columns=close.columns,
                ),
            )

    def test_list_smc_indicators(self):
        if smc_available:
            assert len(vbt.IndicatorFactory.list_smc_indicators()) > 0

    def test_from_smc(self):
        if smc_available:
            assert_frame_equal(
                vbt.smc("swing_highs_lows")
                .run(open, high, low, close, volume, swing_length=1)
                .high_low,
                pd.DataFrame(
                    [[1.0, -1.0], [-1.0, 1.0], [np.nan, np.nan], [np.nan, np.nan], [1.0, -1.0]],
                    index=close.index,
                    columns=pd.MultiIndex.from_tuples(
                        [(1, "a"), (1, "b")],
                        names=["swing_highs_lows_swing_length", None],
                    ),
                ),
            )
            assert_frame_equal(
                vbt.smc("swing_highs_lows")
                .run(open, high, low, close, volume, swing_length=1)
                .level,
                pd.DataFrame(
                    [[2.5, 4.5], [1.5, 5.5], [np.nan, np.nan], [np.nan, np.nan], [6.5, 0.5]],
                    index=close.index,
                    columns=pd.MultiIndex.from_tuples(
                        [(1, "a"), (1, "b")],
                        names=["swing_highs_lows_swing_length", None],
                    ),
                ),
            )
            for ind_name in vbt.IndicatorFactory.list_smc_indicators():
                if ind_name != "SESSIONS":
                    vbt.smc(ind_name).run(open, high, low, close, volume)

    def test_custom_indicators(self):
        custom_ma = lambda x: x
        vbt.IF.register_custom_indicator(custom_ma, "MA")
        assert vbt.IF.get_indicator("MA") == custom_ma
        custom1_ma = lambda x: x
        vbt.IF.register_custom_indicator(custom1_ma, "MA", location="custom1")
        assert vbt.IF.get_indicator("MA", location="custom1") == custom1_ma
        custom2_ma = lambda x: x
        vbt.IF.register_custom_indicator(custom2_ma, "custom2:MA")
        assert vbt.IF.get_indicator("custom2:MA") == custom2_ma
        vbt.IF.register_custom_indicator("vbt:MA", location="CUSTOM3")
        assert vbt.IF.get_indicator("MA", location="custom3") == vbt.MA
        with pytest.raises(Exception):
            vbt.IF.register_custom_indicator(vbt.MA, "MA", location="custom3")
        with pytest.raises(Exception):
            vbt.IF.register_custom_indicator(vbt.MA, "ma", location="custom3")
        with pytest.raises(Exception):
            vbt.IF.register_custom_indicator(vbt.MA, "MA", location="CUSTOM3")
        with pytest.raises(Exception):
            vbt.IF.register_custom_indicator(vbt.MA, "MA", location="talib")
        with pytest.raises(Exception):
            vbt.IF.register_custom_indicator(vbt.MA, "MA MA", location="custom4")
        with pytest.raises(Exception):
            vbt.IF.register_custom_indicator(vbt.MA, "MA", location="MA MA")
        assert vbt.IF.list_custom_locations() == ["custom", "custom1", "custom2", "CUSTOM3"]
        assert vbt.IF.list_custom_indicators() == [
            "custom:MA",
            "custom1:MA",
            "custom2:MA",
            "CUSTOM3:MA",
        ]
        vbt.IF.deregister_custom_indicator("ma", location="custom")
        assert vbt.IF.list_custom_locations() == ["custom1", "custom2", "CUSTOM3"]
        vbt.IF.deregister_custom_indicator("custom1:ma")
        assert vbt.IF.list_custom_locations() == ["custom2", "CUSTOM3"]
        vbt.IF.deregister_custom_indicator(location="custom2")
        assert vbt.IF.list_custom_locations() == ["CUSTOM3"]
        vbt.IF.deregister_custom_indicator("ma")
        assert vbt.IF.list_custom_locations() == []
        vbt.IF.register_custom_indicator(vbt.MA, "custom1:MA")
        vbt.IF.register_custom_indicator(vbt.MA, "custom2:MA")
        vbt.IF.deregister_custom_indicator()
        assert vbt.IF.list_custom_locations() == []


# ############# custom ############# #

ohlcv_df = pd.DataFrame(
    {
        "open": [1, 2, 3, 4, 5],
        "high": [2.5, 3.5, 4.5, 5.5, 6.5],
        "low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "close": [2, 3, 4, 5, 6],
        "volume": [1, 2, 3, 2, 1],
    },
    index=pd.Index(
        [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 5),
        ],
    ),
)
open = pd.DataFrame(
    {
        "a": [1, 2, 3, 4, 5],
        "b": [6, 5, 4, 3, 2],
    },
    index=pd.date_range("2023", periods=5),
)
high = pd.DataFrame(
    {
        "a": [2.5, 3.5, 4.5, 5.5, 6.5],
        "b": [6.5, 5.5, 4.5, 3.5, 2.5],
    },
    index=pd.date_range("2023", periods=5),
)
low = pd.DataFrame(
    {
        "a": [0.5, 1.5, 2.5, 3.5, 4.5],
        "b": [4.5, 3.5, 2.5, 1.5, 0.5],
    },
    index=pd.date_range("2023", periods=5),
)
close = pd.DataFrame(
    {
        "a": [2, 3, 4, 5, 6],
        "b": [5, 4, 3, 2, 1],
    },
    index=pd.date_range("2023", periods=5),
)
volume = pd.DataFrame(
    {
        "a": [1, 2, 3, 2, 1],
        "b": [3, 2, 1, 2, 3],
    },
    index=pd.date_range("2023", periods=5),
)


class TestBasic:
    def test_MA(self):
        ma = vbt.MA.run(close, window=3, wtype="simple", hide_params=True)
        assert_frame_equal(
            ma.ma,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [3.0, 4.0],
                    [4.0, 3.0],
                    [5.0, 2.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        np.testing.assert_array_equal(
            vbt.ind_nb.ma_nb(close.values, window=np.array([2, 3]), wtype=np.array([0, 1])),
            np.array(
                [
                    [np.nan, np.nan],
                    [2.5, np.nan],
                    [3.5, 3.6666666666666665],
                    [4.5, 2.6666666666666665],
                    [5.5, 1.6666666666666667],
                ]
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.ma_nb, dict(n_chunks=2))
        np.testing.assert_array_equal(
            chunked_func(close.values, window=np.array([2, 3]), wtype=np.array([0, 1])),
            vbt.ind_nb.ma_nb(close.values, window=np.array([2, 3]), wtype=np.array([0, 1])),
        )

    def test_MSD(self):
        msd = vbt.MSD.run(close, window=3, wtype="simple", hide_params=True)
        assert_frame_equal(
            msd.msd,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.816496580927726, 0.816496580927726],
                    [0.816496580927726, 0.816496580927726],
                    [0.816496580927726, 0.816496580927726],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        np.testing.assert_array_equal(
            vbt.ind_nb.msd_nb(close.values, window=np.array([2, 3]), wtype=np.array([0, 2])),
            np.array(
                [
                    [np.nan, np.nan],
                    [0.5, np.nan],
                    [0.5, 1.0488088481701516],
                    [0.5, 1.300183137283433],
                    [0.5, 1.46929354773366],
                ]
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.msd_nb, dict(n_chunks=2))
        np.testing.assert_array_equal(
            chunked_func(close.values, window=np.array([2, 3]), wtype=np.array([0, 2])),
            vbt.ind_nb.msd_nb(close.values, window=np.array([2, 3]), wtype=np.array([0, 2])),
        )

    def test_BBANDS(self):
        bbands = vbt.BBANDS.run(close, window=3, wtype="simple", alpha=2.0, hide_params=True)
        assert_frame_equal(
            bbands.upper,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [4.6329931618554525, 5.6329931618554525],
                    [5.6329931618554525, 4.6329931618554525],
                    [6.6329931618554525, 3.632993161855452],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            bbands.middle,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [3.0, 4.0],
                    [4.0, 3.0],
                    [5.0, 2.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            bbands.lower,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [1.367006838144548, 2.367006838144548],
                    [2.367006838144548, 1.367006838144548],
                    [3.367006838144548, 0.36700683814454793],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            bbands.percent_b,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.8061862178478971, 0.1938137821521027],
                    [0.8061862178478971, 0.1938137821521027],
                    [0.8061862178478971, 0.19381378215210274],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            bbands.bandwidth,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [1.088662107903635, 0.8164965809277261],
                    [0.8164965809277261, 1.088662107903635],
                    [0.6531972647421809, 1.632993161855452],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )

        outputs = vbt.ind_nb.bbands_nb(close.values, window=np.array([2, 3]))
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [np.nan, np.nan],
                    [3.5, np.nan],
                    [4.5, 5.6329931618554525],
                    [5.5, 4.6329931618554525],
                    [6.5, 3.632993161855452],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [np.nan, np.nan],
                    [2.5, np.nan],
                    [3.5, 4.0],
                    [4.5, 3.0],
                    [5.5, 2.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[2],
            np.array(
                [
                    [np.nan, np.nan],
                    [1.5, np.nan],
                    [2.5, 2.367006838144548],
                    [3.5, 1.367006838144548],
                    [4.5, 0.36700683814454793],
                ]
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.bbands_nb, dict(n_chunks=2))
        for i in range(3):
            np.testing.assert_array_equal(
                chunked_func(
                    close.values,
                    window=np.array([2, 3]),
                    wtype=np.array([0, 2]),
                    alpha=np.array([1.0, 2.0]),
                )[i],
                vbt.ind_nb.bbands_nb(
                    close.values,
                    window=np.array([2, 3]),
                    wtype=np.array([0, 2]),
                    alpha=np.array([1.0, 2.0]),
                )[i],
            )

    def test_RSI(self):
        rsi = vbt.RSI.run(close, window=3, wtype="simple", hide_params=True)
        assert_frame_equal(
            rsi.rsi,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [100.0, 0.0],
                    [100.0, 0.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        np.testing.assert_array_equal(
            vbt.ind_nb.rsi_nb(close.values, window=np.array([2, 3]), wtype=np.array([0, 1])),
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [100.0, np.nan],
                    [100.0, 0.0],
                    [100.0, 0.0],
                ]
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.rsi_nb, dict(n_chunks=2))
        np.testing.assert_array_equal(
            chunked_func(close.values, window=np.array([2, 3]), wtype=np.array([0, 1])),
            vbt.ind_nb.rsi_nb(close.values, window=np.array([2, 3]), wtype=np.array([0, 1])),
        )

    def test_STOCH(self):
        stoch = vbt.STOCH.run(
            high,
            low,
            close,
            fast_k_window=3,
            slow_k_window=2,
            slow_d_window=2,
            wtype="simple",
            hide_params=True,
        )
        assert_frame_equal(
            stoch.fast_k,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [87.5, 12.5],
                    [87.5, 12.5],
                    [87.5, 12.5],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            stoch.slow_k,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [87.5, 12.5],
                    [87.5, 12.5],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            stoch.slow_d,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [87.5, 12.5],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        outputs = vbt.ind_nb.stoch_nb(
            high.values,
            low.values,
            close.values,
            fast_k_window=np.array([2, 3]),
            slow_k_window=np.array([1, 2]),
            slow_d_window=np.array([1, 2]),
            wtype=np.array([0, 1]),
        )
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [np.nan, np.nan],
                    [83.33333333333333, np.nan],
                    [83.33333333333333, 12.5],
                    [83.33333333333333, 12.5],
                    [83.33333333333333, 12.5],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [np.nan, np.nan],
                    [83.33333333333333, np.nan],
                    [83.33333333333333, np.nan],
                    [83.33333333333333, 12.5],
                    [83.33333333333333, 12.5],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[2],
            np.array(
                [
                    [np.nan, np.nan],
                    [83.33333333333333, np.nan],
                    [83.33333333333333, np.nan],
                    [83.33333333333333, np.nan],
                    [83.33333333333333, 12.5],
                ]
            ),
        )
        for i in range(3):
            chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.stoch_nb, dict(n_chunks=2))
            np.testing.assert_array_equal(
                chunked_func(
                    high.values,
                    low.values,
                    close.values,
                    fast_k_window=3,
                    slow_k_window=2,
                    slow_d_window=2,
                    wtype=[0, 1],
                )[i],
                vbt.ind_nb.stoch_nb(
                    high.values,
                    low.values,
                    close.values,
                    fast_k_window=3,
                    slow_k_window=2,
                    slow_d_window=2,
                    slow_k_wtype=[0, 1],
                    slow_d_wtype=[0, 1],
                )[i],
            )

    def test_MACD(self):
        macd = vbt.MACD.run(
            close,
            fast_window=2,
            slow_window=3,
            signal_window=2,
            wtype="exp",
            hide_params=True,
        )
        assert_frame_equal(
            macd.macd,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.30555555555555536, -0.30555555555555536],
                    [0.39351851851851816, -0.39351851851851816],
                    [0.4436728395061724, -0.44367283950617264],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            macd.signal,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.3641975308641972, -0.3641975308641972],
                    [0.41718106995884735, -0.41718106995884746],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            macd.hist,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.02932098765432095, -0.02932098765432095],
                    [0.026491769547325072, -0.026491769547325184],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        outputs = vbt.ind_nb.macd_nb(
            close.values,
            fast_window=np.array([2, 3]),
            slow_window=np.array([3, 4]),
            signal_window=np.array([1, 2]),
            wtype=np.array([0, 2]),
        )
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.5, np.nan],
                    [0.5, -0.30100000000000016],
                    [0.5, -0.3681000000000001],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.5, np.nan],
                    [0.5, np.nan],
                    [0.5, -0.34573333333333345],
                ]
            ),
        )
        for i in range(2):
            chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.macd_nb, dict(n_chunks=2))
            np.testing.assert_array_equal(
                chunked_func(
                    close.values,
                    fast_window=2,
                    slow_window=3,
                    signal_window=2,
                    wtype=np.array([0, 2]),
                )[i],
                vbt.ind_nb.macd_nb(
                    close.values,
                    fast_window=2,
                    slow_window=3,
                    signal_window=2,
                    macd_wtype=np.array([0, 2]),
                    signal_wtype=np.array([0, 2]),
                )[i],
            )

    def test_ATR(self):
        atr = vbt.ATR.run(
            high,
            low,
            close,
            window=3,
            wtype="wilder",
            hide_params=True,
        )
        assert_frame_equal(
            atr.tr,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            atr.atr,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [2.0, 2.0],
                    [2.0, 2.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        outputs = vbt.ind_nb.atr_nb(
            high.values,
            low.values,
            close.values,
            window=np.array([2, 3]),
            wtype=np.array([0, 1]),
        )
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [np.nan, np.nan],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [2.0, np.nan],
                    [2.0, 2.0],
                    [2.0, 2.0],
                ]
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.atr_nb, dict(n_chunks=2))
        for i in range(2):
            np.testing.assert_array_equal(
                chunked_func(
                    high.values,
                    low.values,
                    close.values,
                    window=np.array([2, 3]),
                    wtype=np.array([0, 1]),
                )[i],
                vbt.ind_nb.atr_nb(
                    high.values,
                    low.values,
                    close.values,
                    window=np.array([2, 3]),
                    wtype=np.array([0, 1]),
                )[i],
            )

    def test_ADX(self):
        adx = vbt.ADX.run(
            high,
            low,
            close,
            window=2,
            wtype="wilder",
            hide_params=True,
        )
        assert_frame_equal(
            adx.plus_di,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [37.5, 0.0],
                    [43.75, 0.0],
                    [46.875, 0.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            adx.minus_di,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.0, 37.5],
                    [0.0, 43.75],
                    [0.0, 46.875],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            adx.dx,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [100.0, 100.0],
                    [100.0, 100.0],
                    [100.0, 100.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            adx.adx,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [100.0, 100.0],
                    [100.0, 100.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        outputs = vbt.ind_nb.adx_nb(
            high.values,
            low.values,
            close.values,
            window=np.array([2, 3]),
            wtype=np.array([0, 1]),
        )
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [50.0, np.nan],
                    [50.0, 0.0],
                    [50.0, 0.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.0, np.nan],
                    [0.0, 50.0],
                    [0.0, 50.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[2],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [100.0, np.nan],
                    [100.0, 100.0],
                    [100.0, 100.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[3],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [100.0, np.nan],
                    [100.0, np.nan],
                ]
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.adx_nb, dict(n_chunks=2))
        for i in range(4):
            np.testing.assert_array_equal(
                chunked_func(
                    high.values,
                    low.values,
                    close.values,
                    window=np.array([2, 3]),
                    wtype=np.array([0, 1]),
                )[i],
                vbt.ind_nb.adx_nb(
                    high.values,
                    low.values,
                    close.values,
                    window=np.array([2, 3]),
                    wtype=np.array([0, 1]),
                )[i],
            )

    def test_OBV(self):
        obv = vbt.OBV.run(close, volume)
        assert_frame_equal(
            obv.obv,
            pd.DataFrame(
                [
                    [1.0, 3.0],
                    [3.0, 1.0],
                    [6.0, 0.0],
                    [8.0, -2.0],
                    [9.0, -5.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.obv_nb, dict(n_chunks=2))
        np.testing.assert_array_equal(
            chunked_func(close.values, volume.values),
            vbt.ind_nb.obv_nb(close.values, volume.values),
        )

    def test_OLS(self):
        ols = vbt.OLS.run(
            np.arange(len(close)), close, window=2, with_zscore=True, hide_params=True
        )
        assert_frame_equal(
            ols.slope,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, -1.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            ols.intercept,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [2.0, 5.0],
                    [2.0, 5.0],
                    [2.0, 5.0],
                    [2.0, 5.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            ols.zscore,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            ols.pred,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [3.0, 4.0],
                    [4.0, 3.0],
                    [5.0, 2.0],
                    [6.0, 1.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            ols.error,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            ols.angle,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [45.0, -45.0],
                    [45.0, -45.0],
                    [45.0, -45.0],
                    [45.0, -45.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        outputs = vbt.ind_nb.ols_nb(
            np.column_stack((np.arange(len(close)), np.arange(len(close)))),
            close.values,
            window=np.array([2, 3]),
            with_zscore=True,
        )
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [np.nan, np.nan],
                    [1.0, np.nan],
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, -1.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [np.nan, np.nan],
                    [2.0, np.nan],
                    [2.0, 5.0],
                    [2.0, 5.0],
                    [2.0, 5.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[2],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                ]
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.ols_nb, dict(n_chunks=2))
        for i in range(3):
            np.testing.assert_array_equal(
                chunked_func(
                    np.column_stack((np.arange(len(close)), np.arange(len(close)))),
                    close.values,
                    window=np.array([2, 3]),
                    with_zscore=True,
                )[i],
                vbt.ind_nb.ols_nb(
                    np.column_stack((np.arange(len(close)), np.arange(len(close)))),
                    close.values,
                    window=np.array([2, 3]),
                    with_zscore=True,
                )[i],
            )

    def test_VWAP(self):
        vwap = vbt.VWAP.run(high, low, close, volume)
        assert_frame_equal(
            vwap.vwap,
            pd.DataFrame(
                [
                    [1.6666666666666667, 5.333333333333333],
                    [2.6666666666666665, 4.333333333333333],
                    [3.6666666666666665, 3.3333333333333335],
                    [4.666666666666667, 2.3333333333333335],
                    [5.666666666666667, 1.3333333333333333],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.vwap_nb, dict(n_chunks=2))
        np.testing.assert_array_equal(
            chunked_func(
                high.values,
                low.values,
                close.values,
                volume.values,
                np.array([close.shape[0]]),
            ),
            vbt.ind_nb.vwap_nb(
                high.values,
                low.values,
                close.values,
                volume.values,
                np.array([close.shape[0]]),
            ),
        )

    def test_PATSIM(self):
        patsim = vbt.PATSIM.run(
            close,
            np.array([1, 2]),
            hide_params=True,
        )
        assert_frame_equal(
            patsim.similarity,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )

    def test_PIVOTINFO(self):
        pivot_info = vbt.PIVOTINFO.run(high, low, up_th=0.5, down_th=0.5, hide_params=True)
        assert_frame_equal(
            pivot_info.conf_pivot,
            pd.DataFrame(
                [
                    [0, 0],
                    [-1, 0],
                    [-1, 1],
                    [-1, 1],
                    [-1, -1],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            pivot_info.conf_idx,
            pd.DataFrame(
                [
                    [-1, -1],
                    [0, -1],
                    [0, 0],
                    [0, 0],
                    [0, 3],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            pivot_info.last_pivot,
            pd.DataFrame(
                [
                    [0, 0],
                    [1, 0],
                    [1, -1],
                    [1, -1],
                    [1, 1],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            pivot_info.last_idx,
            pd.DataFrame(
                [
                    [-1, -1],
                    [1, -1],
                    [2, 2],
                    [3, 3],
                    [4, 4],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            pivot_info.conf_value,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [0.5, np.nan],
                    [0.5, 6.5],
                    [0.5, 6.5],
                    [0.5, 1.5],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            pivot_info.last_value,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [3.5, np.nan],
                    [4.5, 2.5],
                    [5.5, 1.5],
                    [6.5, 2.5],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            pivot_info.pivots,
            pd.DataFrame(
                [
                    [-1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            pivot_info.modes,
            pd.DataFrame(
                [
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [-1, -1],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        outputs = vbt.ind_nb.pivot_info_nb(
            high.values,
            low.values,
            up_th=high.values * 0.1,
            down_th=low.values * 0.1,
        )
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [0, 0],
                    [0, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, -1],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [-1, -1],
                    [-1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 3],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[2],
            np.array(
                [
                    [0, 0],
                    [0, -1],
                    [1, -1],
                    [1, -1],
                    [1, 1],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[3],
            np.array(
                [
                    [-1, -1],
                    [-1, 1],
                    [2, 2],
                    [3, 3],
                    [4, 4],
                ]
            ),
        )
        for i in range(4):
            chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.pivot_info_nb, dict(n_chunks=2))
            np.testing.assert_array_equal(
                chunked_func(
                    high.values,
                    low.values,
                    up_th=high.values * 0.1,
                    down_th=low.values * 0.1,
                )[i],
                vbt.ind_nb.pivot_info_nb(
                    high.values,
                    low.values,
                    up_th=high.values * 0.1,
                    down_th=low.values * 0.1,
                )[i],
            )

    def test_SUPERTREND(self):
        supertrend = vbt.SUPERTREND.run(high, low, close, period=3, multiplier=2, hide_params=True)
        assert_frame_equal(
            supertrend.trend,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.5, -1.5],
                    [1.5, -1.5],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            supertrend.direction,
            pd.DataFrame(
                [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            supertrend.long,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [0.5, -1.5],
                    [1.5, -1.5],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        assert_frame_equal(
            supertrend.short,
            pd.DataFrame(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                ],
                index=close.index,
                columns=close.columns,
            ),
        )
        outputs = vbt.ind_nb.supertrend_nb(
            high.values,
            low.values,
            close.values,
            period=np.array([2, 3]),
            multiplier=np.array([2, 3]),
        )
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [-0.5, np.nan],
                    [0.5, -3.5],
                    [1.5, -3.5],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[2],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [-0.5, np.nan],
                    [0.5, -3.5],
                    [1.5, -3.5],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[3],
            np.array(
                [
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                ]
            ),
        )
        for i in range(4):
            chunked_func = vbt.ch_reg.resolve_option(vbt.ind_nb.supertrend_nb, dict(n_chunks=2))
            np.testing.assert_array_equal(
                chunked_func(
                    high.values,
                    low.values,
                    close.values,
                    period=np.array([2, 3]),
                    multiplier=np.array([2, 3]),
                )[i],
                vbt.ind_nb.supertrend_nb(
                    high.values,
                    low.values,
                    close.values,
                    period=np.array([2, 3]),
                    multiplier=np.array([2, 3]),
                )[i],
            )

    def test_SIGDET(self):
        sr = pd.Series([10, 10, 11, 12, 11, 10, 9, 8, 9, 10, 10])
        sigdet = vbt.SIGDET.run(sr, lag=3, factor=1, influence=1, hide_params=True)
        assert_series_equal(
            sigdet.signal,
            pd.Series(
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    -1,
                    -1,
                    -1,
                    0,
                    1,
                    1,
                ]
            ),
        )
        assert_series_equal(
            sigdet.upper_band,
            pd.Series(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    11.471404520791031,
                    12.14982991426106,
                    11.471404520791031,
                    10.816496580927726,
                    9.816496580927726,
                    9.483163247594392,
                    9.471404520791031,
                    10.483163247594392,
                ]
            ),
        )
        assert_series_equal(
            sigdet.lower_band,
            pd.Series(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    10.528595479208969,
                    10.516836752405608,
                    10.528595479208969,
                    9.183503419072274,
                    8.183503419072274,
                    7.85017008573894,
                    8.528595479208969,
                    8.85017008573894,
                ]
            ),
        )
        outputs = vbt.ind_nb.signal_detection_nb(
            sr.values[:, None],
            lag=np.array([2, 3]),
            factor=sr.values[:, None],
            influence=sr.values[:, None],
        )
        np.testing.assert_array_equal(
            outputs[0],
            np.array(
                [
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[1],
            np.array(
                [
                    [np.nan],
                    [np.nan],
                    [15.5],
                    [82.5],
                    [61.0],
                    [15.5],
                    [14.0],
                    [12.5],
                    [13.0],
                    [14.5],
                    [15.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            outputs[2],
            np.array(
                [
                    [np.nan],
                    [np.nan],
                    [15.5],
                    [-49.5],
                    [-38.0],
                    [5.5],
                    [5.0],
                    [4.5],
                    [4.0],
                    [4.5],
                    [5.0],
                ]
            ),
        )
        for i in range(3):
            chunked_func = vbt.ch_reg.resolve_option(
                vbt.ind_nb.signal_detection_nb, dict(n_chunks=2)
            )
            np.testing.assert_array_equal(
                chunked_func(
                    sr.values[:, None],
                    lag=np.array([2, 3]),
                    factor=sr.values[:, None],
                    influence=sr.values[:, None],
                )[i],
                vbt.ind_nb.signal_detection_nb(
                    sr.values[:, None],
                    lag=np.array([2, 3]),
                    up_factor=sr.values[:, None],
                    down_factor=sr.values[:, None],
                    mean_influence=sr.values[:, None],
                    std_influence=sr.values[:, None],
                )[i],
            )

    def test_HURST(self):
        close = vbt.RandomData.pull(start="2020", periods=200, seed=42).get()
        hurst = vbt.HURST.run(close, method=["standard", "logrs", "rs", "dma", "dsod"])
        assert_series_equal(
            hurst.hurst.iloc[-1].rename(None),
            pd.Series(
                [
                    0.4764129065260935,
                    0.6132611955700931,
                    0.6145327842667652,
                    0.3569452731447068,
                    0.47570178547480757,
                ],
                index=pd.Index(["standard", "logrs", "rs", "dma", "dsod"], name="hurst_method"),
            ),
        )
