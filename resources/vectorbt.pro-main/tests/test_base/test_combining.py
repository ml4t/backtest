import os

import pytest
import vectorbtpro as vbt
from numba import njit
from vectorbtpro.base import combining, merging

from tests.utils import *

# Initialize global variables
sr1 = pd.Series([1], index=pd.Index(["x1"], name="i1"), name="a1")
sr2 = pd.Series([1, 2, 3], index=pd.Index(["x2", "y2", "z2"], name="i2"), name="a2")
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


# ############# combining ############# #


class TestCombining:
    def test_apply_and_concat_none(self):
        def apply_func(i, x, a):
            x[i] = a[i]

        @njit
        def apply_func_nb(i, x, a):
            x[i] = a[i]

        # 1d
        target = pd.Series([10, 20, 30], index=sr2.index, name=sr2.name)
        sr2_copy = sr2.copy()
        combining.apply_and_concat(3, apply_func, sr2_copy.values, [10, 20, 30])
        assert_series_equal(sr2_copy, target)
        sr2_copy = sr2.copy()
        combining.apply_and_concat_none_nb(3, apply_func_nb, sr2_copy.values, (10, 20, 30))
        assert_series_equal(sr2_copy, target)
        sr2_copy = sr2.copy()
        combining.apply_and_concat(
            3, apply_func_nb, sr2_copy.values, [10, 20, 30], n_outputs=0, jitted_loop=True
        )
        assert_series_equal(sr2_copy, target)

    def test_apply_and_concat_one(self):
        def apply_func(i, x, a):
            return x + a[i]

        @njit
        def apply_func_nb(i, x, a):
            return x + a[i]

        # 1d
        target = np.array([[11, 21, 31], [12, 22, 32], [13, 23, 33]])
        np.testing.assert_array_equal(
            combining.apply_and_concat(3, apply_func, sr2.values, [10, 20, 30]), target
        )
        np.testing.assert_array_equal(
            combining.apply_and_concat_one_nb(3, apply_func_nb, sr2.values, (10, 20, 30)),
            target,
        )
        np.testing.assert_array_equal(
            combining.apply_and_concat(
                3, apply_func_nb, sr2.values, [10, 20, 30], n_outputs=1, jitted_loop=True
            ),
            combining.apply_and_concat_one_nb(3, apply_func_nb, sr2.values, (10, 20, 30)),
        )
        # 2d
        target2 = np.array(
            [
                [11, 12, 13, 21, 22, 23, 31, 32, 33],
                [14, 15, 16, 24, 25, 26, 34, 35, 36],
                [17, 18, 19, 27, 28, 29, 37, 38, 39],
            ]
        )
        np.testing.assert_array_equal(
            combining.apply_and_concat(3, apply_func, df4.values, [10, 20, 30]), target2
        )
        np.testing.assert_array_equal(
            combining.apply_and_concat_one_nb(3, apply_func_nb, df4.values, (10, 20, 30)),
            target2,
        )
        np.testing.assert_array_equal(
            combining.apply_and_concat(
                3, apply_func_nb, df4.values, [10, 20, 30], n_outputs=1, jitted_loop=True
            ),
            combining.apply_and_concat_one_nb(3, apply_func_nb, df4.values, (10, 20, 30)),
        )

    def test_apply_and_concat_multiple(self):
        def apply_func(i, x, a):
            return (x, x + a[i])

        @njit
        def apply_func_nb(i, x, a):
            return (x, x + a[i])

        # 1d
        target_a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        target_b = np.array([[11, 21, 31], [12, 22, 32], [13, 23, 33]])
        a, b = combining.apply_and_concat(3, apply_func, sr2.values, [10, 20, 30])
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combining.apply_and_concat_multiple_nb(3, apply_func_nb, sr2.values, (10, 20, 30))
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combining.apply_and_concat(
            3, apply_func_nb, sr2.values, [10, 20, 30], n_outputs=2, jitted_loop=True
        )
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        # 2d
        target_a = np.array(
            [[1, 2, 3, 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9, 7, 8, 9]]
        )
        target_b = np.array(
            [
                [11, 12, 13, 21, 22, 23, 31, 32, 33],
                [14, 15, 16, 24, 25, 26, 34, 35, 36],
                [17, 18, 19, 27, 28, 29, 37, 38, 39],
            ]
        )
        a, b = combining.apply_and_concat(3, apply_func, df4.values, [10, 20, 30])
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combining.apply_and_concat_multiple_nb(3, apply_func_nb, df4.values, (10, 20, 30))
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combining.apply_and_concat(
            3, apply_func_nb, df4.values, [10, 20, 30], n_outputs=2, jitted_loop=True
        )
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)

    def test_combine_and_concat(self):
        def combine_func(x, y, a):
            return x + y + a

        @njit
        def combine_func_nb(x, y, a):
            return x + y + a

        # 1d
        target = np.array([[103, 104], [106, 108], [109, 112]])
        np.testing.assert_array_equal(
            combining.combine_and_concat(
                sr2.values, (sr2.values * 2, sr2.values * 3), combine_func, 100
            ),
            target,
        )
        np.testing.assert_array_equal(
            combining.combine_and_concat_nb(
                sr2.values, (sr2.values * 2, sr2.values * 3), combine_func_nb, 100
            ),
            target,
        )
        # 2d
        target2 = np.array(
            [
                [103, 106, 109, 104, 108, 112],
                [112, 115, 118, 116, 120, 124],
                [121, 124, 127, 128, 132, 136],
            ],
        )
        np.testing.assert_array_equal(
            combining.combine_and_concat(
                df4.values, (df4.values * 2, df4.values * 3), combine_func, 100
            ),
            target2,
        )
        np.testing.assert_array_equal(
            combining.combine_and_concat_nb(
                df4.values, (df4.values * 2, df4.values * 3), combine_func_nb, 100
            ),
            target2,
        )

    def test_combine_multiple(self):
        def combine_func(x, y, a):
            return x + y + a

        @njit
        def combine_func_nb(x, y, a):
            return x + y + a

        # 1d
        target = np.array([206, 212, 218])
        np.testing.assert_array_equal(
            combining.combine_multiple(
                (sr2.values, sr2.values * 2, sr2.values * 3), combine_func, 100
            ),
            target,
        )
        np.testing.assert_array_equal(
            combining.combine_multiple_nb(
                (sr2.values, sr2.values * 2, sr2.values * 3), combine_func_nb, 100
            ),
            target,
        )
        # 2d
        target2 = np.array([[206, 212, 218], [224, 230, 236], [242, 248, 254]])
        np.testing.assert_array_equal(
            combining.combine_multiple(
                (df4.values, df4.values * 2, df4.values * 3), combine_func, 100
            ),
            target2,
        )
        np.testing.assert_array_equal(
            combining.combine_multiple_nb(
                (df4.values, df4.values * 2, df4.values * 3), combine_func_nb, 100
            ),
            target2,
        )


# ############# merging ############# #


class TestMerging:
    def test_concat_merge(self):
        np.testing.assert_array_equal(
            merging.concat_merge([0, 1, 2]),
            np.array([0, 1, 2]),
        )
        np.testing.assert_array_equal(
            merging.concat_merge(([0, 1, 2], [3, 4, 5])),
            np.array([0, 1, 2, 3, 4, 5]),
        )
        np.testing.assert_array_equal(
            merging.concat_merge((([0, 1, 2],), ([0, 1, 2],)))[0],
            np.array([0, 1, 2, 0, 1, 2]),
        )
        np.testing.assert_array_equal(
            merging.concat_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[0],
            np.array([0, 1, 2, 0, 1, 2]),
        )
        np.testing.assert_array_equal(
            merging.concat_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[1],
            np.array([3, 4, 5, 3, 4, 5]),
        )
        assert_series_equal(
            merging.concat_merge([0, 1, 2], keys=pd.Index(["a", "b", "c"], name="d")),
            pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"], name="d")),
        )
        assert_series_equal(
            merging.concat_merge(
                [0, 1, 2], wrap_kwargs=dict(index=pd.Index(["a", "b", "c"], name="d"), name="name")
            ),
            pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"], name="d"), name="name"),
        )
        assert_series_equal(
            merging.concat_merge(([0, 1, 2], [3, 4, 5]), keys=pd.Index(["k1", "k2"], name="key")),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", 0), ("k1", 1), ("k1", 2), ("k2", 0), ("k2", 1), ("k2", 2)],
                    names=["key", None],
                ),
            ),
        )
        assert_series_equal(
            merging.concat_merge(
                ([0, 1, 2], [3, 4, 5]),
                wrap_kwargs=dict(index=pd.Index(["a", "b", "c"], name="d"), name="name"),
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "a", "b", "c"], name="d"),
                name="name",
            ),
        )
        assert_series_equal(
            merging.concat_merge(
                ([0, 1, 2], [3, 4, 5]),
                wrap_kwargs=[
                    dict(index=pd.Index(["a", "b", "c"], name="d"), name="name"),
                    dict(index=pd.Index(["e", "f", "g"], name="h"), name="name"),
                ],
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        sr1 = pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"], name="d"), name="name")
        sr2 = pd.Series([3, 4, 5], index=pd.Index(["e", "f", "g"], name="h"), name="name")
        assert_series_equal(
            merging.concat_merge((sr1, sr2)),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        assert_series_equal(
            merging.concat_merge((sr1, sr2), keys=pd.Index(["k1", "k2"], name="key")),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")],
                    names=["key", None],
                ),
                name="name",
            ),
        )
        assert_series_equal(
            merging.concat_merge([dict(a=0, b=1, c=2), dict(d=3, e=4, f=5)], wrap=True),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "d", "e", "f"]),
            ),
        )

    def test_row_stack_merge(self):
        np.testing.assert_array_equal(
            merging.row_stack_merge(([0, 1, 2], [3, 4, 5])),
            np.array([[0], [1], [2], [3], [4], [5]]),
        )
        np.testing.assert_array_equal(
            merging.row_stack_merge((([0, 1, 2],), ([0, 1, 2],)))[0],
            np.array([[0], [1], [2], [0], [1], [2]]),
        )
        np.testing.assert_array_equal(
            merging.row_stack_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[0],
            np.array([[0], [1], [2], [0], [1], [2]]),
        )
        np.testing.assert_array_equal(
            merging.row_stack_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[1],
            np.array([[3], [4], [5], [3], [4], [5]]),
        )
        assert_series_equal(
            merging.row_stack_merge(
                ([0, 1, 2], [3, 4, 5]), keys=pd.Index(["k1", "k2"], name="key")
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", 0), ("k1", 1), ("k1", 2), ("k2", 0), ("k2", 1), ("k2", 2)],
                    names=["key", None],
                ),
            ),
        )
        assert_series_equal(
            merging.row_stack_merge(
                ([0, 1, 2], [3, 4, 5]),
                wrap_kwargs=dict(index=pd.Index(["a", "b", "c"], name="d"), name="name"),
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "a", "b", "c"], name="d"),
                name="name",
            ),
        )
        assert_series_equal(
            merging.row_stack_merge(
                ([0, 1, 2], [3, 4, 5]),
                wrap_kwargs=[
                    dict(index=pd.Index(["a", "b", "c"], name="d"), name="name"),
                    dict(index=pd.Index(["e", "f", "g"], name="h"), name="name"),
                ],
            ),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge(
                ([[0], [1], [2]], [[3], [4], [5]]),
                wrap_kwargs=[
                    dict(index=pd.Index(["a", "b", "c"], name="d"), columns=["name"]),
                    dict(index=pd.Index(["e", "f", "g"], name="h"), columns=["name"]),
                ],
            ),
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                columns=["name"],
            ),
        )
        sr1 = pd.Series([0, 1, 2], index=pd.Index(["a", "b", "c"], name="d"), name="name")
        sr2 = pd.Series([3, 4, 5], index=pd.Index(["e", "f", "g"], name="h"), name="name")
        assert_series_equal(
            merging.row_stack_merge((sr1, sr2)),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        assert_series_equal(
            merging.row_stack_merge((sr1, sr2), keys=pd.Index(["k1", "k2"], name="key")),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")],
                    names=["key", None],
                ),
                name="name",
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge((sr1.to_frame(), sr2.to_frame())),
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                columns=["name"],
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge(
                (sr1.to_frame(), sr2.to_frame()), keys=pd.Index(["k1", "k2"], name="key")
            ),
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")],
                    names=["key", None],
                ),
                columns=["name"],
            ),
        )
        assert_series_equal(
            merging.row_stack_merge([dict(a=0, b=1, c=2), dict(d=3, e=4, f=5)], wrap="sr"),
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "d", "e", "f"]),
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge(
                [dict(a=[0], b=[1], c=[2]), dict(a=[3], b=[4], c=[5])],
                wrap="df",
                ignore_index=True,
            ),
            pd.DataFrame(
                [[0, 1, 2], [3, 4, 5]],
                columns=pd.Index(["a", "b", "c"]),
            ),
        )
        assert_series_equal(
            merging.row_stack_merge((sr1.vbt, sr2.vbt)).obj,
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                name="name",
            ),
        )
        assert_series_equal(
            merging.row_stack_merge(
                (sr1.vbt, sr2.vbt), keys=pd.Index(["k1", "k2"], name="key")
            ).obj,
            pd.Series(
                [0, 1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")],
                    names=["key", None],
                ),
                name="name",
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge((sr1.to_frame().vbt, sr2.to_frame().vbt)).obj,
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.Index(["a", "b", "c", "e", "f", "g"]),
                columns=["name"],
            ),
        )
        assert_frame_equal(
            merging.row_stack_merge(
                (sr1.to_frame().vbt, sr2.to_frame().vbt), keys=pd.Index(["k1", "k2"], name="key")
            ).obj,
            pd.DataFrame(
                [[0], [1], [2], [3], [4], [5]],
                index=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")],
                    names=["key", None],
                ),
                columns=["name"],
            ),
        )

    def test_column_stack_merge(self):
        np.testing.assert_array_equal(
            merging.column_stack_merge(([0, 1, 2], [3, 4, 5])),
            np.array([[0, 3], [1, 4], [2, 5]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((([0, 1, 2],), ([0, 1, 2],)))[0],
            np.array([[0, 0], [1, 1], [2, 2]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[0],
            np.array([[0, 0], [1, 1], [2, 2]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((([0, 1, 2], [3, 4, 5]), ([0, 1, 2], [3, 4, 5])))[1],
            np.array([[3, 3], [4, 4], [5, 5]]),
        )
        assert_frame_equal(
            merging.column_stack_merge(
                ([[0, 1, 2]], [[3, 4, 5]]), keys=pd.Index(["k1", "k2"], name="key")
            ),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.MultiIndex.from_tuples(
                    [("k1", 0), ("k1", 1), ("k1", 2), ("k2", 0), ("k2", 1), ("k2", 2)],
                    names=["key", None],
                ),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge(
                ([[0, 1, 2]], [[3, 4, 5]]),
                wrap_kwargs=dict(columns=pd.Index(["a", "b", "c"]), index=pd.Index(["d"])),
            ),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "a", "b", "c"]),
                index=pd.Index(["d"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge(
                ([[0, 1, 2]], [[3, 4, 5]]),
                wrap_kwargs=[
                    dict(columns=pd.Index(["a", "b", "c"], name="d")),
                    dict(columns=pd.Index(["e", "f", "g"], name="h")),
                ],
            ),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "e", "f", "g"]),
            ),
        )
        df1 = pd.DataFrame(
            [[0, 1, 2]], columns=pd.Index(["a", "b", "c"], name="d"), index=pd.Index(["i"])
        )
        df2 = pd.DataFrame(
            [[3, 4, 5]], columns=pd.Index(["e", "f", "g"], name="h"), index=pd.Index(["i"])
        )
        assert_frame_equal(
            merging.column_stack_merge((df1, df2)),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "e", "f", "g"]),
                index=pd.Index(["i"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((df1, df2), keys=pd.Index(["k1", "k2"], name="key")),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")],
                    names=["key", None],
                ),
                index=pd.Index(["i"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge([dict(a=0, b=1, c=2), dict(a=3, b=4, c=5)], wrap="sr"),
            pd.DataFrame(
                [[0, 3], [1, 4], [2, 5]],
                index=pd.Index(["a", "b", "c"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge(
                [dict(a=[0], b=[1], c=[2]), dict(d=[3], e=[4], f=[5])], wrap="df"
            ),
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "d", "e", "f"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((df1.vbt, df2.vbt)).obj,
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.Index(["a", "b", "c", "e", "f", "g"]),
                index=pd.Index(["i"]),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge(
                (df1.vbt, df2.vbt), keys=pd.Index(["k1", "k2"], name="key")
            ).obj,
            pd.DataFrame(
                [[0, 1, 2, 3, 4, 5]],
                columns=pd.MultiIndex.from_tuples(
                    [("k1", "a"), ("k1", "b"), ("k1", "c"), ("k2", "e"), ("k2", "f"), ("k2", "g")],
                    names=["key", None],
                ),
                index=pd.Index(["i"]),
            ),
        )
        sr1 = pd.Series([0, 1, 2, 4, 5], index=pd.date_range("2020-01-01", periods=5))
        sr2 = pd.Series([6, 7, 8], index=pd.date_range("2020-01-04", periods=3))
        assert_frame_equal(
            merging.column_stack_merge((sr1, sr2), reset_index=False),
            pd.DataFrame(
                [
                    [0.0, np.nan],
                    [1.0, np.nan],
                    [2.0, np.nan],
                    [4.0, 6.0],
                    [5.0, 7.0],
                    [np.nan, 8.0],
                ],
                index=pd.date_range("2020-01-01", periods=6),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1, sr2), reset_index=True),
            pd.DataFrame([[0, 6.0], [1, 7.0], [2, 8.0], [4, np.nan], [5, np.nan]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1, sr2), reset_index="from_start"),
            pd.DataFrame([[0, 6.0], [1, 7.0], [2, 8.0], [4, np.nan], [5, np.nan]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1, sr2), reset_index="from_end"),
            pd.DataFrame([[0, np.nan], [1, np.nan], [2, 6.0], [4, 7.0], [5, 8.0]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1.vbt, sr2.vbt), reset_index=False).obj,
            pd.DataFrame(
                [
                    [0.0, np.nan],
                    [1.0, np.nan],
                    [2.0, np.nan],
                    [4.0, 6.0],
                    [5.0, 7.0],
                    [np.nan, 8.0],
                ],
                index=pd.date_range("2020-01-01", periods=6),
            ),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1.vbt, sr2.vbt), reset_index=True).obj,
            pd.DataFrame([[0.0, 6.0], [1.0, 7.0], [2.0, 8.0], [4.0, np.nan], [5.0, np.nan]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1.vbt, sr2.vbt), reset_index="from_start").obj,
            pd.DataFrame([[0.0, 6.0], [1.0, 7.0], [2.0, 8.0], [4.0, np.nan], [5.0, np.nan]]),
        )
        assert_frame_equal(
            merging.column_stack_merge((sr1.vbt, sr2.vbt), reset_index="from_end").obj,
            pd.DataFrame([[0.0, np.nan], [1.0, np.nan], [2.0, 6.0], [4.0, 7.0], [5.0, 8.0]]),
        )
        arr1 = np.array([0, 1, 2, 4, 5])
        arr2 = np.array([6, 7, 8])
        with pytest.raises(Exception):
            merging.column_stack_merge((arr1, arr2), reset_index=False)
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index=True),
            np.array([[0, 6.0], [1, 7.0], [2, 8.0], [4, np.nan], [5, np.nan]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index="from_start"),
            np.array([[0, 6.0], [1, 7.0], [2, 8.0], [4, np.nan], [5, np.nan]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index="from_end"),
            np.array([[0, np.nan], [1, np.nan], [2, 6.0], [4, 7.0], [5, 8.0]]),
        )
        arr1 = np.array([0, 1, 2])
        arr2 = np.array([6, 7, 8])
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index=False),
            np.array([[0, 6], [1, 7], [2, 8]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index=True),
            np.array([[0, 6], [1, 7], [2, 8]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index="from_start"),
            np.array([[0, 6], [1, 7], [2, 8]]),
        )
        np.testing.assert_array_equal(
            merging.column_stack_merge((arr1, arr2), reset_index="from_end"),
            np.array([[0, 6], [1, 7], [2, 8]]),
        )

    def test_mixed_merge(self):
        np.testing.assert_array_equal(
            merging.mixed_merge(
                [
                    ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
                    ([9, 10, 11], [12, 13, 14], [15, 16, 17]),
                ],
                merge_funcs=("concat", "row_stack", "column_stack"),
            )[0],
            np.array([0, 1, 2, 9, 10, 11]),
        )
        np.testing.assert_array_equal(
            merging.mixed_merge(
                [
                    ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
                    ([9, 10, 11], [12, 13, 14], [15, 16, 17]),
                ],
                merge_funcs=("concat", "row_stack", "column_stack"),
            )[1],
            np.array([[3], [4], [5], [12], [13], [14]]),
        )
        np.testing.assert_array_equal(
            merging.mixed_merge(
                [
                    ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
                    ([9, 10, 11], [12, 13, 14], [15, 16, 17]),
                ],
                merge_funcs=("concat", "row_stack", "column_stack"),
            )[2],
            np.array([[6, 15], [7, 16], [8, 17]]),
        )
