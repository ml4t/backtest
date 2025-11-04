import os

import pytest
import vectorbtpro as vbt
from numba import njit

from tests.utils import *

day_dt = np.timedelta64(86400000000000)

# Initialize global variables
a1 = np.array([1])
a2 = np.array([1, 2, 3])
a3 = np.array([[1, 2, 3]])
a4 = np.array([[1], [2], [3]])
a5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
sr_none = pd.Series([1])
sr1 = pd.Series([1], index=pd.Index(["x1"], name="i1"), name="a1")
sr2 = pd.Series([1, 2, 3], index=pd.Index(["x2", "y2", "z2"], name="i2"), name="a2")
df_none = pd.DataFrame([[1]])
df1 = pd.DataFrame([[1]], index=pd.Index(["x3"], name="i3"), columns=pd.Index(["a3"], name="c3"))
df2 = pd.DataFrame(
    [[1], [2], [3]],
    index=pd.Index(["x4", "y4", "z4"], name="i4"),
    columns=pd.Index(["a4"], name="c4"),
)
df3 = pd.DataFrame(
    [[1, 2, 3]], index=pd.Index(["x5"], name="i5"), columns=pd.Index(["a5", "b5", "c5"], name="c5")
)
df4 = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    index=pd.Index(["x6", "y6", "z6"], name="i6"),
    columns=pd.Index(["a6", "b6", "c6"], name="c6"),
)
multi_i = pd.MultiIndex.from_arrays([["x7", "y7", "z7"], ["x8", "y8", "z8"]], names=["i7", "i8"])
multi_c = pd.MultiIndex.from_arrays([["a7", "b7", "c7"], ["a8", "b8", "c8"]], names=["c7", "c8"])
df5 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=multi_i, columns=multi_c)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.broadcasting["index_from"] = "stack"
    vbt.settings.broadcasting["columns_from"] = "stack"


def teardown_module():
    vbt.settings.reset()


# ############# accessors ############# #


class TestAccessors:
    def test_indexing(self):
        assert_series_equal(df4.vbt["a6"].obj, df4["a6"].vbt.obj)

    def test_freq(self):
        ts = pd.Series(
            [1, 2, 3],
            index=pd.date_range("2020", periods=3),
        )
        assert ts.vbt.wrapper.freq == day_dt
        assert ts.vbt(freq="2D").wrapper.freq == day_dt * 2
        assert pd.Series([1, 2, 3]).vbt.wrapper.freq is None
        assert pd.Series([1, 2, 3]).vbt(freq="3D").wrapper.freq == day_dt * 3
        assert pd.Series([1, 2, 3]).vbt(freq=np.timedelta64(4, "D")).wrapper.freq == day_dt * 4

    def test_props(self):
        assert sr1.vbt.is_series()
        assert not sr1.vbt.is_frame()
        assert not df1.vbt.is_series()
        assert df2.vbt.is_frame()

    def test_wrapper(self):
        assert_index_equal(sr2.vbt.wrapper.index, sr2.index)
        assert_index_equal(sr2.vbt.wrapper.columns, sr2.to_frame().columns)
        assert sr2.vbt.wrapper.ndim == sr2.ndim
        assert sr2.vbt.wrapper.name == sr2.name
        assert pd.Series([1, 2, 3]).vbt.wrapper.name is None
        assert sr2.vbt.wrapper.shape == sr2.shape
        assert sr2.vbt.wrapper.shape_2d == (sr2.shape[0], 1)
        assert_index_equal(df4.vbt.wrapper.index, df4.index)
        assert_index_equal(df4.vbt.wrapper.columns, df4.columns)
        assert df4.vbt.wrapper.ndim == df4.ndim
        assert df4.vbt.wrapper.name is None
        assert df4.vbt.wrapper.shape == df4.shape
        assert df4.vbt.wrapper.shape_2d == df4.shape
        assert_series_equal(sr2.vbt.wrapper.wrap(a2), sr2)
        assert_series_equal(sr2.vbt.wrapper.wrap(df2), sr2)
        assert_series_equal(
            sr2.vbt.wrapper.wrap(df2.values, index=df2.index, columns=df2.columns),
            pd.Series(df2.values[:, 0], index=df2.index, name=df2.columns[0]),
        )
        assert_frame_equal(
            sr2.vbt.wrapper.wrap(df4.values, columns=df4.columns),
            pd.DataFrame(df4.values, index=sr2.index, columns=df4.columns),
        )
        assert_frame_equal(df2.vbt.wrapper.wrap(a2), df2)
        assert_frame_equal(df2.vbt.wrapper.wrap(sr2), df2)
        assert_frame_equal(
            df2.vbt.wrapper.wrap(df4.values, columns=df4.columns),
            pd.DataFrame(df4.values, index=df2.index, columns=df4.columns),
        )

    def test_row_stack(self):
        acc = vbt.BaseAccessor.row_stack(
            vbt.BaseAccessor(pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"]))),
            vbt.BaseAccessor(pd.Series([3, 4, 5], index=pd.Index(["d", "e", "f"]))),
        )
        target_obj = pd.DataFrame(
            [0, 1, 2, 3, 4, 5],
            index=pd.Index(["a", "b", "c", "d", "e", "f"]),
        )
        assert isinstance(acc, vbt.BaseSRAccessor)
        assert_index_equal(
            acc.wrapper.index,
            target_obj.index,
        )
        assert acc.wrapper.name is None
        assert target_obj.vbt.wrapper.ndim == target_obj.ndim
        acc = vbt.BaseAccessor.row_stack(
            vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr")),
            vbt.BaseAccessor(
                pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c"))
            ),
        )
        target_obj = pd.DataFrame(
            [[0, 0], [1, 1], [2, 2], [0, 1], [2, 3], [4, 5]],
            index=pd.RangeIndex(start=0, stop=6, step=1),
            columns=pd.Index(["a", "b"], dtype="object", name="c"),
        )
        assert isinstance(acc, vbt.BaseDFAccessor)
        assert_index_equal(
            acc.wrapper.index,
            target_obj.index,
        )
        assert_index_equal(
            acc.wrapper.columns,
            target_obj.columns,
        )
        assert target_obj.vbt.wrapper.ndim == target_obj.ndim
        acc = vbt.BaseAccessor.row_stack(
            vbt.BaseAccessor(
                pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False
            ),
            vbt.BaseAccessor(
                pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                some_arg=2,
                check_expected_keys_=False,
            ),
        )
        assert acc.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.BaseAccessor.row_stack(
                vbt.BaseAccessor(
                    pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False
                ),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c"))
                ),
            )
        with pytest.raises(Exception):
            vbt.BaseAccessor.row_stack(
                vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr")),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                    some_arg=2,
                    check_expected_keys_=False,
                ),
            )
        with pytest.raises(Exception):
            vbt.BaseAccessor.row_stack(
                vbt.BaseAccessor(
                    pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False
                ),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                    some_arg=3,
                    check_expected_keys_=False,
                ),
            )

    def test_column_stack(self):
        acc = vbt.BaseAccessor.column_stack(
            vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr")),
            vbt.BaseAccessor(
                pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c"))
            ),
        )
        target_obj = pd.DataFrame(
            [[0, 0, 1], [1, 2, 3], [2, 4, 5]],
            index=pd.RangeIndex(start=0, stop=3, step=1),
            columns=pd.Index(["sr", "a", "b"], dtype="object"),
        )
        assert isinstance(acc, vbt.BaseDFAccessor)
        assert_index_equal(
            acc.wrapper.index,
            target_obj.index,
        )
        assert_index_equal(
            acc.wrapper.columns,
            target_obj.columns,
        )
        acc = vbt.BaseAccessor.column_stack(
            vbt.BaseAccessor(
                pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False
            ),
            vbt.BaseAccessor(
                pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                some_arg=2,
                check_expected_keys_=False,
            ),
        )
        assert acc.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.BaseAccessor.column_stack(
                vbt.BaseAccessor(
                    pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False
                ),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c"))
                ),
            )
        with pytest.raises(Exception):
            vbt.BaseAccessor.column_stack(
                vbt.BaseAccessor(pd.Series([0, 1, 2], name="sr")),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                    some_arg=2,
                    check_expected_keys_=False,
                ),
            )
        with pytest.raises(Exception):
            vbt.BaseAccessor.column_stack(
                vbt.BaseAccessor(
                    pd.Series([0, 1, 2], name="sr"), some_arg=2, check_expected_keys_=False
                ),
                vbt.BaseAccessor(
                    pd.DataFrame([[0, 1], [2, 3], [4, 5]], columns=pd.Index(["a", "b"], name="c")),
                    some_arg=3,
                    check_expected_keys_=False,
                ),
            )

    def test_empty(self):
        assert_series_equal(
            pd.Series.vbt.empty(5, index=np.arange(10, 15), name="a", fill_value=5),
            pd.Series(np.full(5, 5), index=np.arange(10, 15), name="a"),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.empty(
                (5, 3), index=np.arange(10, 15), columns=["a", "b", "c"], fill_value=5
            ),
            pd.DataFrame(np.full((5, 3), 5), index=np.arange(10, 15), columns=["a", "b", "c"]),
        )
        assert_series_equal(
            pd.Series.vbt.empty_like(sr2, fill_value=5),
            pd.Series(np.full(sr2.shape, 5), index=sr2.index, name=sr2.name),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.empty_like(df4, fill_value=5),
            pd.DataFrame(np.full(df4.shape, 5), index=df4.index, columns=df4.columns),
        )

    def test_apply_func_on_index(self):
        assert_frame_equal(
            df1.vbt.apply_to_index(lambda idx: idx + "_yo", axis=0),
            pd.DataFrame(
                np.array([1]),
                index=pd.Index(["x3_yo"], dtype="object", name="i3"),
                columns=pd.Index(["a3"], dtype="object", name="c3"),
            ),
        )
        assert_frame_equal(
            df1.vbt.apply_to_index(lambda idx: idx + "_yo", axis=1),
            pd.DataFrame(
                np.array([1]),
                index=pd.Index(["x3"], dtype="object", name="i3"),
                columns=pd.Index(["a3_yo"], dtype="object", name="c3"),
            ),
        )

    def test_add_levels(self):
        assert_frame_equal(
            df5.vbt.add_levels([1, 2, 3], on_top=True),
            pd.DataFrame(
                df5.values,
                index=df5.index,
                columns=pd.MultiIndex.from_tuples(
                    [(1, "a7", "a8"), (2, "b7", "b8"), (3, "c7", "c8")],
                    names=[None, "c7", "c8"],
                ),
            ),
        )
        assert_frame_equal(
            df5.vbt.add_levels([1, 2, 3], on_top=False),
            pd.DataFrame(
                df5.values,
                index=df5.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a7", "a8", 1), ("b7", "b8", 2), ("c7", "c8", 3)],
                    names=["c7", "c8", None],
                ),
            ),
        )

    def test_drop_levels(self):
        assert_frame_equal(
            df5.vbt.drop_levels("c7"),
            pd.DataFrame(
                df5.values,
                index=df5.index,
                columns=pd.Index(["a8", "b8", "c8"], dtype="object", name="c8"),
            ),
        )

    def test_rename_levels(self):
        assert_frame_equal(
            df5.vbt.rename_levels({"c8": "c9"}),
            pd.DataFrame(
                df5.values,
                index=df5.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a7", "a8"), ("b7", "b8"), ("c7", "c8")], names=["c7", "c9"]
                ),
            ),
        )

    def test_select_levels(self):
        assert_frame_equal(
            df5.vbt.select_levels("c8"),
            pd.DataFrame(
                df5.values,
                index=df5.index,
                columns=pd.Index(["a8", "b8", "c8"], dtype="object", name="c8"),
            ),
        )

    def test_drop_redundant_levels(self):
        assert_frame_equal(
            df5.vbt.add_levels(pd.RangeIndex(start=0, step=1, stop=3)).vbt.drop_redundant_levels(),
            df5,
        )

    def test_drop_duplicate_levels(self):
        assert_frame_equal(
            df5.vbt.add_levels(df5.columns.get_level_values(0)).vbt.drop_duplicate_levels(),
            df5,
        )

    def test_set(self):
        ts_index = pd.date_range("2020-01-01", "2020-01-05")
        df = pd.DataFrame(0, index=ts_index, columns=["a", "b", "c"])
        sr = pd.Series(0, index=ts_index)

        target_sr = sr.copy()
        target_sr.iloc[::2] = 100
        assert_series_equal(sr.vbt.set(100, every=2), target_sr)
        target_df = df.copy()
        target_df.iloc[::2] = 100
        assert_frame_equal(df.vbt.set(100, every=2), target_df)
        target_df = df.copy()
        target_df.iloc[::2, 1] = 100
        assert_frame_equal(df.vbt.set(100, columns="b", every=2), target_df)
        target_df = df.copy()
        target_df.iloc[::2, [1, 2]] = 100
        assert_frame_equal(df.vbt.set(100, columns=["b", "c"], every=2), target_df)

        target_sr = sr.copy()
        target_sr.iloc[0] = 100
        target_sr.iloc[2] = 200
        target_sr.iloc[4] = 300
        assert_series_equal(sr.vbt.set([100, 200, 300], every=2), target_sr)
        target_df = df.copy()
        target_df.iloc[0] = 100
        target_df.iloc[2] = 200
        target_df.iloc[4] = 300
        assert_frame_equal(df.vbt.set([100, 200, 300], every=2), target_df)
        target_df = df.copy()
        target_df.iloc[0, 1] = 100
        target_df.iloc[2, 1] = 200
        target_df.iloc[4, 1] = 300
        assert_frame_equal(df.vbt.set([100, 200, 300], columns="b", every=2), target_df)
        target_df = df.copy()
        target_df.iloc[0, [1, 2]] = 100
        target_df.iloc[2, [1, 2]] = 200
        target_df.iloc[4, [1, 2]] = 300
        assert_frame_equal(df.vbt.set([100, 200, 300], columns=["b", "c"], every=2), target_df)

        target_sr = sr.copy()
        target_sr.iloc[0] = 100
        target_sr.iloc[2] = 200
        target_sr.iloc[4] = 300
        assert_series_equal(
            sr.vbt.set(lambda i: [100, 200, 300][i], vbt.Rep("i"), every=2),
            target_sr,
        )
        target_df = df.copy()
        target_df.iloc[0] = 100
        target_df.iloc[2] = 200
        target_df.iloc[4] = 300
        assert_frame_equal(
            df.vbt.set(lambda i: [100, 200, 300][i], vbt.Rep("i"), every=2),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, 1] = 100
        target_df.iloc[2, 1] = 200
        target_df.iloc[4, 1] = 300
        assert_frame_equal(
            df.vbt.set(lambda i: [100, 200, 300][i], vbt.Rep("i"), columns="b", every=2),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, [1, 2]] = 100
        target_df.iloc[2, [1, 2]] = 200
        target_df.iloc[4, [1, 2]] = 300
        assert_frame_equal(
            df.vbt.set(lambda i: [100, 200, 300][i], vbt.Rep("i"), columns=["b", "c"], every=2),
            target_df,
        )

    def test_set_between(self):
        ts_index = pd.date_range("2020-01-01", "2020-01-05")
        df = pd.DataFrame(0, index=ts_index, columns=["a", "b", "c"])
        sr = pd.Series(0, index=ts_index)

        target_sr = sr.copy()
        target_sr.iloc[::2] = 100
        assert_series_equal(sr.vbt.set_between(100, start=[0, 2, 4], end=[1, 3, 5]), target_sr)
        target_df = df.copy()
        target_df.iloc[::2] = 100
        assert_frame_equal(df.vbt.set_between(100, start=[0, 2, 4], end=[1, 3, 5]), target_df)
        target_df = df.copy()
        target_df.iloc[::2, 1] = 100
        assert_frame_equal(
            df.vbt.set_between(100, columns="b", start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[::2, [1, 2]] = 100
        assert_frame_equal(
            df.vbt.set_between(100, columns=["b", "c"], start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )

        target_sr = sr.copy()
        target_sr.iloc[0] = 100
        target_sr.iloc[2] = 200
        target_sr.iloc[4] = 300
        assert_series_equal(
            sr.vbt.set_between([100, 200, 300], start=[0, 2, 4], end=[1, 3, 5]), target_sr
        )
        target_df = df.copy()
        target_df.iloc[0] = 100
        target_df.iloc[2] = 200
        target_df.iloc[4] = 300
        assert_frame_equal(
            df.vbt.set_between([100, 200, 300], start=[0, 2, 4], end=[1, 3, 5]), target_df
        )
        target_df = df.copy()
        target_df.iloc[0, 1] = 100
        target_df.iloc[2, 1] = 200
        target_df.iloc[4, 1] = 300
        assert_frame_equal(
            df.vbt.set_between([100, 200, 300], columns="b", start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, [1, 2]] = 100
        target_df.iloc[2, [1, 2]] = 200
        target_df.iloc[4, [1, 2]] = 300
        assert_frame_equal(
            df.vbt.set_between([100, 200, 300], columns=["b", "c"], start=[0, 2, 4], end=[1, 3, 5]),
            target_df,
        )

        target_sr = sr.copy()
        target_sr.iloc[0] = 100
        target_sr.iloc[2] = 200
        target_sr.iloc[4] = 300
        assert_series_equal(
            sr.vbt.set_between(
                lambda i: [100, 200, 300][i], vbt.Rep("i"), start=[0, 2, 4], end=[1, 3, 5]
            ),
            target_sr,
        )
        target_df = df.copy()
        target_df.iloc[0] = 100
        target_df.iloc[2] = 200
        target_df.iloc[4] = 300
        assert_frame_equal(
            df.vbt.set_between(
                lambda i: [100, 200, 300][i], vbt.Rep("i"), start=[0, 2, 4], end=[1, 3, 5]
            ),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, 1] = 100
        target_df.iloc[2, 1] = 200
        target_df.iloc[4, 1] = 300
        assert_frame_equal(
            df.vbt.set_between(
                lambda i: [100, 200, 300][i],
                vbt.Rep("i"),
                columns="b",
                start=[0, 2, 4],
                end=[1, 3, 5],
            ),
            target_df,
        )
        target_df = df.copy()
        target_df.iloc[0, [1, 2]] = 100
        target_df.iloc[2, [1, 2]] = 200
        target_df.iloc[4, [1, 2]] = 300
        assert_frame_equal(
            df.vbt.set_between(
                lambda i: [100, 200, 300][i],
                vbt.Rep("i"),
                columns=["b", "c"],
                start=[0, 2, 4],
                end=[1, 3, 5],
            ),
            target_df,
        )

    def test_to_array(self):
        np.testing.assert_array_equal(sr2.vbt.to_1d_array(), sr2.values)
        np.testing.assert_array_equal(sr2.vbt.to_2d_array(), sr2.to_frame().values)
        np.testing.assert_array_equal(df2.vbt.to_1d_array(), df2.iloc[:, 0].values)
        np.testing.assert_array_equal(df2.vbt.to_2d_array(), df2.values)

    def test_tile(self):
        assert_frame_equal(
            df4.vbt.tile(2, keys=["a", "b"], axis=0),
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.MultiIndex.from_tuples(
                    [("a", "x6"), ("a", "y6"), ("a", "z6"), ("b", "x6"), ("b", "y6"), ("b", "z6")],
                    names=[None, "i6"],
                ),
                columns=df4.columns,
            ),
        )
        assert_frame_equal(
            df4.vbt.tile(2, keys=["a", "b"], axis=1),
            pd.DataFrame(
                np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9]]),
                index=df4.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a", "a6"), ("a", "b6"), ("a", "c6"), ("b", "a6"), ("b", "b6"), ("b", "c6")],
                    names=[None, "c6"],
                ),
            ),
        )

    def test_repeat(self):
        assert_frame_equal(
            df4.vbt.repeat(2, keys=["a", "b"], axis=0),
            pd.DataFrame(
                np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [7, 8, 9], [7, 8, 9]]),
                index=pd.MultiIndex.from_tuples(
                    [("x6", "a"), ("x6", "b"), ("y6", "a"), ("y6", "b"), ("z6", "a"), ("z6", "b")],
                    names=["i6", None],
                ),
                columns=df4.columns,
            ),
        )
        assert_frame_equal(
            df4.vbt.repeat(2, keys=["a", "b"], axis=1),
            pd.DataFrame(
                np.array([[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6], [7, 7, 8, 8, 9, 9]]),
                index=df4.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a6", "a"), ("a6", "b"), ("b6", "a"), ("b6", "b"), ("c6", "a"), ("c6", "b")],
                    names=["c6", None],
                ),
            ),
        )

    def test_align_to(self):
        multi_c1 = pd.MultiIndex.from_arrays([["a8", "b8"]], names=["c8"])
        multi_c2 = pd.MultiIndex.from_arrays(
            [["a7", "a7", "c7", "c7"], ["a8", "b8", "a8", "b8"]], names=["c7", "c8"]
        )
        df10 = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=multi_c1)
        df20 = pd.DataFrame([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], columns=multi_c2)
        assert_frame_equal(
            df10.vbt.align_to(df20),
            pd.DataFrame(
                np.array([[1, 2, 1, 2], [4, 5, 4, 5], [7, 8, 7, 8]]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=multi_c2,
            ),
        )

    def test_align(self):
        index1 = pd.MultiIndex.from_arrays([pd.Index(["a1", "a2"], name="a")])
        index2 = pd.MultiIndex.from_arrays([pd.Index(["b1", "b2"], name="b")])
        index3 = pd.MultiIndex.from_arrays(
            [
                pd.Index(["b2", "b2", "b1", "b1"], name="b"),
                pd.Index(["a2", "a1", "a2", "a1"], name="a"),
            ]
        )
        index4 = pd.MultiIndex.from_arrays(
            [
                pd.Index(["e1", "e1", "e1", "e1", "e2", "e2", "e2", "e2"], name="e"),
                pd.Index(["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2"], name="b"),
                pd.Index(["a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2"], name="a"),
            ]
        )
        df1 = pd.DataFrame([range(len(index1))], columns=index1)
        df2 = pd.DataFrame([range(len(index2))], columns=index2)
        df3 = pd.DataFrame([range(len(index3))], columns=index3)
        df4 = pd.DataFrame([range(len(index4))], columns=index4)
        new_df1, new_df2, new_df3, new_df4 = df1.vbt.align(df2, df3, df4)
        new_columns = pd.MultiIndex.from_tuples(
            [
                ("e1", "b1", "a1"),
                ("e1", "b1", "a2"),
                ("e1", "b2", "a1"),
                ("e1", "b2", "a2"),
                ("e2", "b1", "a1"),
                ("e2", "b1", "a2"),
                ("e2", "b2", "a1"),
                ("e2", "b2", "a2"),
            ],
            names=["e", "b", "a"],
        )
        assert_index_equal(new_df1.columns, new_columns)
        assert_index_equal(new_df2.columns, new_columns)
        assert_index_equal(new_df3.columns, new_columns)
        assert_index_equal(new_df4.columns, new_columns)
        np.testing.assert_array_equal(
            new_df1.values,
            np.array([[0, 1, 0, 1, 0, 1, 0, 1]]),
        )
        np.testing.assert_array_equal(
            new_df2.values,
            np.array([[0, 0, 1, 1, 0, 0, 1, 1]]),
        )
        np.testing.assert_array_equal(
            new_df3.values,
            np.array([[3, 2, 1, 0, 3, 2, 1, 0]]),
        )
        np.testing.assert_array_equal(
            new_df4.values,
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
        )

    def test_cross(self):
        index1 = pd.MultiIndex.from_arrays([pd.Index(["a1", "a2"], name="a")])
        index2 = pd.MultiIndex.from_arrays([pd.Index(["b1", "b2"], name="b")])
        index3 = pd.MultiIndex.from_arrays(
            [
                pd.Index(["b2", "b2", "b1", "b1"], name="b"),
                pd.Index(["a2", "a1", "a2", "a1"], name="a"),
            ]
        )
        index4 = pd.MultiIndex.from_arrays(
            [
                pd.Index(["e1", "e1", "e1", "e1", "e2", "e2", "e2", "e2"], name="e"),
                pd.Index(["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2"], name="b"),
                pd.Index(["a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2"], name="a"),
            ]
        )
        index5 = pd.MultiIndex.from_arrays(
            [
                pd.Index(
                    ["c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2", "c3", "c3", "c3", "c3"],
                    name="c",
                ),
                pd.Index(
                    ["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2"],
                    name="b",
                ),
                pd.Index(
                    ["a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2"],
                    name="a",
                ),
            ]
        )
        df1 = pd.DataFrame([range(len(index1))], columns=index1)
        df2 = pd.DataFrame([range(len(index2))], columns=index2)
        df3 = pd.DataFrame([range(len(index3))], columns=index3)
        df4 = pd.DataFrame([range(len(index4))], columns=index4)
        df5 = pd.DataFrame([range(len(index5))], columns=index5)
        new_df1, new_df2, new_df3, new_df4, new_df5 = df1.vbt.x(df2, df3, df4, df5)
        new_columns = pd.MultiIndex.from_tuples(
            [
                ("e1", "c1", "b1", "a1"),
                ("e1", "c1", "b1", "a2"),
                ("e1", "c1", "b2", "a1"),
                ("e1", "c1", "b2", "a2"),
                ("e1", "c2", "b1", "a1"),
                ("e1", "c2", "b1", "a2"),
                ("e1", "c2", "b2", "a1"),
                ("e1", "c2", "b2", "a2"),
                ("e1", "c3", "b1", "a1"),
                ("e1", "c3", "b1", "a2"),
                ("e1", "c3", "b2", "a1"),
                ("e1", "c3", "b2", "a2"),
                ("e2", "c1", "b1", "a1"),
                ("e2", "c1", "b1", "a2"),
                ("e2", "c1", "b2", "a1"),
                ("e2", "c1", "b2", "a2"),
                ("e2", "c2", "b1", "a1"),
                ("e2", "c2", "b1", "a2"),
                ("e2", "c2", "b2", "a1"),
                ("e2", "c2", "b2", "a2"),
                ("e2", "c3", "b1", "a1"),
                ("e2", "c3", "b1", "a2"),
                ("e2", "c3", "b2", "a1"),
                ("e2", "c3", "b2", "a2"),
            ],
            names=["e", "c", "b", "a"],
        )
        assert_index_equal(new_df1.columns, new_columns)
        assert_index_equal(new_df2.columns, new_columns)
        assert_index_equal(new_df3.columns, new_columns)
        assert_index_equal(new_df4.columns, new_columns)
        assert_index_equal(new_df5.columns, new_columns)
        np.testing.assert_array_equal(
            new_df1.values,
            np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]),
        )
        np.testing.assert_array_equal(
            new_df2.values,
            np.array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]),
        )
        np.testing.assert_array_equal(
            new_df3.values,
            np.array([[3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]]),
        )
        np.testing.assert_array_equal(
            new_df4.values,
            np.array([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7]]),
        )
        np.testing.assert_array_equal(
            new_df5.values,
            np.array(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
            ),
        )

    def test_broadcast(self):
        a, b = pd.Series.vbt.broadcast(sr2, 10)
        b_target = pd.Series(np.full(sr2.shape, 10), index=sr2.index, name=sr2.name)
        assert_series_equal(a, sr2)
        assert_series_equal(b, b_target)
        a, b = sr2.vbt.broadcast(10)
        assert_series_equal(a, sr2)
        assert_series_equal(b, b_target)

    def test_broadcast_to(self):
        assert_frame_equal(sr2.vbt.broadcast_to(df2, align_index=False), df2)
        assert_frame_equal(sr2.vbt.broadcast_to(df2.vbt, align_index=False), df2)

    def test_broadcast_combs(self):
        new_index = pd.MultiIndex.from_tuples(
            [("x6", "x7", "x8"), ("y6", "y7", "y8"), ("z6", "z7", "z8")],
            names=["i6", "i7", "i8"],
        )
        new_columns = pd.MultiIndex.from_tuples(
            [
                ("a6", "a7", "a8"),
                ("a6", "b7", "b8"),
                ("a6", "c7", "c8"),
                ("b6", "a7", "a8"),
                ("b6", "b7", "b8"),
                ("b6", "c7", "c8"),
                ("c6", "a7", "a8"),
                ("c6", "b7", "b8"),
                ("c6", "c7", "c8"),
            ],
            names=["c6", "c7", "c8"],
        )
        assert_frame_equal(
            df4.vbt.broadcast_combs(df5, align_index=False)[0],
            pd.DataFrame(
                [
                    [1, 1, 1, 2, 2, 2, 3, 3, 3],
                    [4, 4, 4, 5, 5, 5, 6, 6, 6],
                    [7, 7, 7, 8, 8, 8, 9, 9, 9],
                ],
                index=new_index,
                columns=new_columns,
            ),
        )
        assert_frame_equal(
            df4.vbt.broadcast_combs(df5, align_index=False)[1],
            pd.DataFrame(
                [
                    [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    [4, 5, 6, 4, 5, 6, 4, 5, 6],
                    [7, 8, 9, 7, 8, 9, 7, 8, 9],
                ],
                index=new_index,
                columns=new_columns,
            ),
        )

    def test_apply(self):
        assert_series_equal(sr2.vbt.apply(lambda x: x**2), sr2**2)
        assert_series_equal(sr2.vbt.apply(lambda x: x**2, to_2d=True), sr2**2)
        assert_frame_equal(df4.vbt.apply(lambda x: x**2), df4**2)
        assert_frame_equal(
            sr2.vbt.apply(lambda x, y: x**y, vbt.Rep("y"), broadcast_named_args=dict(y=df4)),
            sr2.vbt**df4,
        )

    def test_concat(self):
        assert_frame_equal(
            pd.DataFrame.vbt.concat(
                pd.Series([1, 2, 3]),
                pd.Series([1, 2, 3]),
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame({0: pd.Series([1, 2, 3]), 1: pd.Series([1, 2, 3])}),
        )
        target = pd.DataFrame(
            np.array(
                [
                    [1, 1, 1, 10, 10, 10, 1, 2, 3],
                    [2, 2, 2, 10, 10, 10, 4, 5, 6],
                    [3, 3, 3, 10, 10, 10, 7, 8, 9],
                ]
            ),
            index=pd.MultiIndex.from_tuples(
                [("x2", "x6"), ("y2", "y6"), ("z2", "z6")], names=["i2", "i6"]
            ),
            columns=pd.MultiIndex.from_tuples(
                [
                    ("a", "a6"),
                    ("a", "b6"),
                    ("a", "c6"),
                    ("b", "a6"),
                    ("b", "b6"),
                    ("b", "c6"),
                    ("c", "a6"),
                    ("c", "b6"),
                    ("c", "c6"),
                ],
                names=[None, "c6"],
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.concat(
                sr2, 10, df4, keys=["a", "b", "c"], broadcast_kwargs=dict(align_index=False)
            ),
            target,
        )
        assert_frame_equal(
            sr2.vbt.concat(10, df4, keys=["a", "b", "c"], broadcast_kwargs=dict(align_index=False)),
            target,
        )

    def test_apply_and_concat(self):
        def apply_func(i, x, y, c, d=1):
            return x + y[i] + c + d

        @njit
        def apply_func_nb(i, x, y, c, d):
            return x + y[i] + c + d

        target = pd.DataFrame(
            np.array([[112, 113, 114], [113, 114, 115], [114, 115, 116]]),
            index=pd.Index(["x2", "y2", "z2"], dtype="object", name="i2"),
            columns=pd.Index(["a", "b", "c"], dtype="object"),
        )
        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func,
                np.array([1, 2, 3]),
                10,
                d=100,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            ),
            target,
        )
        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func_nb,
                np.array([1, 2, 3]),
                10,
                100,
                jitted_loop=True,
                n_outputs=1,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            ),
            target,
        )
        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func,
                np.array([1, 2, 3]),
                10,
                d=100,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                target.values,
                index=target.index,
                columns=pd.Index([0, 1, 2], dtype="int64", name="apply_idx"),
            ),
        )

        def apply_func2(i, x, y, c, d=1):
            return x + y + c + d

        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func2,
                np.array([[1], [2], [3]]),
                10,
                d=100,
                keys=["a", "b", "c"],
                to_2d=True,  # otherwise (3, 1) + (1, 3) = (3, 3) != (3, 1) -> error
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                np.array([[112, 112, 112], [114, 114, 114], [116, 116, 116]]),
                index=target.index,
                columns=target.columns,
            ),
        )
        target2 = pd.DataFrame(
            np.array([[112, 113, 114], [113, 114, 115], [114, 115, 116]]),
            index=pd.Index(["x4", "y4", "z4"], dtype="object", name="i4"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "a4"), ("b", "a4"), ("c", "a4")], names=[None, "c4"]
            ),
        )
        assert_frame_equal(
            df2.vbt.apply_and_concat(
                3,
                apply_func,
                np.array([1, 2, 3]),
                10,
                d=100,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            ),
            target2,
        )
        assert_frame_equal(
            df2.vbt.apply_and_concat(
                3,
                apply_func_nb,
                np.array([1, 2, 3]),
                10,
                100,
                jitted_loop=True,
                n_outputs=1,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            ),
            target2,
        )

        def apply_func3(i, x, y, c, d=1):
            return (x + y[i] + c + d, x + y[i] + c + d)

        assert_frame_equal(
            df2.vbt.apply_and_concat(
                3,
                apply_func3,
                np.array([1, 2, 3]),
                10,
                d=100,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            )[0],
            target2,
        )
        assert_frame_equal(
            df2.vbt.apply_and_concat(
                3,
                apply_func3,
                np.array([1, 2, 3]),
                10,
                d=100,
                keys=["a", "b", "c"],
                broadcast_kwargs=dict(align_index=False),
            )[1],
            target2,
        )

        def apply_func2(i, x, y, z):
            return x + y + z[i]

        assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3,
                apply_func2,
                vbt.Rep("y"),
                vbt.RepEval("np.arange(ntimes)"),
                broadcast_named_args=dict(y=df4),
                template_context=dict(np=np),
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                [
                    [2, 3, 4, 3, 4, 5, 4, 5, 6],
                    [6, 7, 8, 7, 8, 9, 8, 9, 10],
                    [10, 11, 12, 11, 12, 13, 12, 13, 14],
                ],
                index=pd.MultiIndex.from_tuples(
                    [("x2", "x6"), ("y2", "y6"), ("z2", "z6")], names=["i2", "i6"]
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0, "a6"),
                        (0, "b6"),
                        (0, "c6"),
                        (1, "a6"),
                        (1, "b6"),
                        (1, "c6"),
                        (2, "a6"),
                        (2, "b6"),
                        (2, "c6"),
                    ],
                    names=["apply_idx", "c6"],
                ),
            ),
        )

    def test_combine(self):
        def combine_func(x, y, a, b=1):
            return x + y + a + b

        @njit
        def combine_func_nb(x, y, a, b):
            return x + y + a + b

        assert_series_equal(
            sr2.vbt.combine(
                10,
                combine_func,
                100,
                b=1000,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.Series(
                np.array([1111, 1112, 1113]),
                index=pd.Index(["x2", "y2", "z2"], dtype="object", name="i2"),
                name=sr2.name,
            ),
        )
        assert_series_equal(
            sr2.vbt.combine(
                10,
                combine_func,
                100,
                1000,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.Series(
                np.array([1111, 1112, 1113]),
                index=pd.Index(["x2", "y2", "z2"], dtype="object", name="i2"),
                name=sr2.name,
            ),
        )

        @njit
        def combine_func2_nb(x, y):
            return x + y + np.array([[1], [2], [3]])

        assert_series_equal(
            sr2.vbt.combine(
                10,
                combine_func2_nb,
                to_2d=True,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.Series(
                np.array([12, 14, 16]),
                index=pd.Index(["x2", "y2", "z2"], dtype="object", name="i2"),
                name="a2",
            ),
        )

        @njit
        def combine_func3_nb(x, y):
            return x + y

        assert_frame_equal(
            df4.vbt.combine(
                sr2,
                combine_func3_nb,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                np.array([[2, 3, 4], [6, 7, 8], [10, 11, 12]]),
                index=pd.MultiIndex.from_tuples(
                    [("x6", "x2"), ("y6", "y2"), ("z6", "z2")], names=["i6", "i2"]
                ),
                columns=pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
            ),
        )
        assert_frame_equal(
            pd.DataFrame.vbt.combine(
                [df4, sr2],
                combine_func3_nb,
                broadcast_kwargs=dict(align_index=False),
            ),
            df4.vbt.combine(
                sr2,
                combine_func3_nb,
                broadcast_kwargs=dict(align_index=False),
            ),
        )

        target = pd.DataFrame(
            np.array([[232, 233, 234], [236, 237, 238], [240, 241, 242]]),
            index=pd.MultiIndex.from_tuples(
                [("x2", "x6"), ("y2", "y6"), ("z2", "z6")], names=["i2", "i6"]
            ),
            columns=pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                combine_func,
                10,
                b=100,
                concat=False,
                broadcast_kwargs=dict(align_index=False),
            ),
            target,
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                combine_func_nb,
                10,
                100,
                jitted_loop=True,
                concat=False,
                broadcast_kwargs=dict(align_index=False),
            ),
            target,
        )
        assert_frame_equal(
            df4.vbt.combine(
                [10, sr2],
                combine_func,
                10,
                b=100,
                concat=False,
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                target.values,
                index=pd.MultiIndex.from_tuples(
                    [("x6", "x2"), ("y6", "y2"), ("z6", "z2")], names=["i6", "i2"]
                ),
                columns=target.columns,
            ),
        )
        target2 = pd.DataFrame(
            np.array(
                [
                    [121, 121, 121, 112, 113, 114],
                    [122, 122, 122, 116, 117, 118],
                    [123, 123, 123, 120, 121, 122],
                ]
            ),
            index=pd.MultiIndex.from_tuples(
                [("x2", "x6"), ("y2", "y6"), ("z2", "z6")], names=["i2", "i6"]
            ),
            columns=pd.MultiIndex.from_tuples(
                [(0, "a6"), (0, "b6"), (0, "c6"), (1, "a6"), (1, "b6"), (1, "c6")],
                names=["combine_idx", "c6"],
            ),
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                combine_func,
                10,
                b=100,
                concat=True,
                broadcast_kwargs=dict(align_index=False),
            ),
            target2,
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                combine_func_nb,
                10,
                100,
                jitted_loop=True,
                concat=True,
                broadcast_kwargs=dict(align_index=False),
            ),
            target2,
        )
        assert_frame_equal(
            sr2.vbt.combine(
                [10, df4],
                lambda x, y, a, b=1: x + y + a + b,
                10,
                b=100,
                concat=True,
                keys=["a", "b"],
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                target2.values,
                index=target2.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a", "a6"), ("a", "b6"), ("a", "c6"), ("b", "a6"), ("b", "b6"), ("b", "c6")],
                    names=[None, "c6"],
                ),
            ),
        )

        assert_frame_equal(
            sr2.vbt.combine(
                [10, 20],
                lambda x, y, a, b=1: x + y + a + b,
                vbt.Rep("y"),
                b=100,
                concat=True,
                keys=["a", "b"],
                broadcast_named_args=dict(y=df4),
                broadcast_kwargs=dict(align_index=False),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [112, 113, 114, 122, 123, 124],
                        [116, 117, 118, 126, 127, 128],
                        [120, 121, 122, 130, 131, 132],
                    ],
                ),
                index=target2.index,
                columns=pd.MultiIndex.from_tuples(
                    [("a", "a6"), ("a", "b6"), ("a", "c6"), ("b", "a6"), ("b", "b6"), ("b", "c6")],
                    names=[None, "c6"],
                ),
            ),
        )
