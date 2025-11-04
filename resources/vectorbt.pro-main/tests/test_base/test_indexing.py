import os

import pytest
import vectorbtpro as vbt
from vectorbtpro._dtypes import *
from vectorbtpro.base import flex_indexing, indexes, indexing, reshaping
from vectorbtpro.utils import checks

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


# ############# indexes ############# #


class TestIndexes:
    def test_get_index(self):
        assert_index_equal(indexes.get_index(sr1, 0), sr1.index)
        assert_index_equal(indexes.get_index(sr1, 1), pd.Index([sr1.name]))
        assert_index_equal(indexes.get_index(pd.Series([1, 2, 3]), 1), pd.Index([0]))  # empty
        assert_index_equal(indexes.get_index(df1, 0), df1.index)
        assert_index_equal(indexes.get_index(df1, 1), df1.columns)

    def test_index_from_values(self):
        assert_index_equal(
            indexes.index_from_values([0.1, 0.2], name="a"),
            pd.Index([0.1, 0.2], dtype="float64", name="a"),
        )
        assert_index_equal(
            indexes.index_from_values(
                np.tile(np.arange(1, 4)[:, None][:, None], (1, 3, 3)), name="b"
            ),
            pd.Index([1, 2, 3], dtype="int64", name="b"),
        )
        assert_index_equal(
            indexes.index_from_values(
                [
                    np.random.uniform(size=(3, 3)),
                    np.random.uniform(size=(3, 3)),
                    np.random.uniform(size=(3, 3)),
                ],
                name="c",
            ),
            pd.Index(["array_0", "array_1", "array_2"], dtype="object", name="c"),
        )
        rand_arr = np.random.uniform(size=(3, 3))
        assert_index_equal(
            indexes.index_from_values([rand_arr, rand_arr, rand_arr], name="c"),
            pd.Index(["array_0", "array_0", "array_0"], dtype="object", name="c"),
        )
        assert_index_equal(
            indexes.index_from_values(
                [
                    rand_arr,
                    np.random.uniform(size=(3, 3)),
                    rand_arr,
                    np.random.uniform(size=(3, 3)),
                ],
                name="c",
            ),
            pd.Index(["array_0", "array_1", "array_0", "array_2"], dtype="object", name="c"),
        )
        assert_index_equal(
            indexes.index_from_values([(1, 2), (3, 4), (5, 6)], name="c"),
            pd.Index(["tuple_0", "tuple_1", "tuple_2"], dtype="object", name="c"),
        )

        class A:
            pass

        class B:
            pass

        class C:
            pass

        assert_index_equal(
            indexes.index_from_values([A(), B(), B(), C()], name="c"),
            pd.Index(["A_0", "B_0", "B_1", "C_0"], dtype="object", name="c"),
        )
        a = A()
        b = B()
        c = C()
        assert_index_equal(
            indexes.index_from_values([a, b, b, c], name="c"),
            pd.Index(["A_0", "B_0", "B_0", "C_0"], dtype="object", name="c"),
        )

    def test_repeat_index(self):
        i = pd.Index([1, 2, 3], name="i")
        assert_index_equal(
            indexes.repeat_index(i, 3),
            pd.Index([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype="int64", name="i"),
        )
        assert_index_equal(
            indexes.repeat_index(multi_i, 3),
            pd.MultiIndex.from_tuples(
                [
                    ("x7", "x8"),
                    ("x7", "x8"),
                    ("x7", "x8"),
                    ("y7", "y8"),
                    ("y7", "y8"),
                    ("y7", "y8"),
                    ("z7", "z8"),
                    ("z7", "z8"),
                    ("z7", "z8"),
                ],
                names=["i7", "i8"],
            ),
        )
        assert_index_equal(
            indexes.repeat_index([0], 3), pd.Index([0, 1, 2], dtype="int64")
        )  # empty
        assert_index_equal(
            indexes.repeat_index(sr_none.index, 3),
            pd.RangeIndex(start=0, stop=3, step=1),  # simple range,
        )

    def test_tile_index(self):
        i = pd.Index([1, 2, 3], name="i")
        assert_index_equal(
            indexes.tile_index(i, 3),
            pd.Index([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype="int64", name="i"),
        )
        assert_index_equal(
            indexes.tile_index(multi_i, 3),
            pd.MultiIndex.from_tuples(
                [
                    ("x7", "x8"),
                    ("y7", "y8"),
                    ("z7", "z8"),
                    ("x7", "x8"),
                    ("y7", "y8"),
                    ("z7", "z8"),
                    ("x7", "x8"),
                    ("y7", "y8"),
                    ("z7", "z8"),
                ],
                names=["i7", "i8"],
            ),
        )
        assert_index_equal(indexes.tile_index([0], 3), pd.Index([0, 1, 2], dtype="int64"))  # empty
        assert_index_equal(
            indexes.tile_index(sr_none.index, 3),
            pd.RangeIndex(start=0, stop=3, step=1),  # simple range,
        )

    def test_stack_indexes(self):
        assert_index_equal(
            indexes.stack_indexes([sr2.index, df2.index, df5.index]),
            pd.MultiIndex.from_tuples(
                [("x2", "x4", "x7", "x8"), ("y2", "y4", "y7", "y8"), ("z2", "z4", "z7", "z8")],
                names=["i2", "i4", "i7", "i8"],
            ),
        )
        assert_index_equal(
            indexes.stack_indexes([sr2.index, df2.index, sr2.index], drop_duplicates=False),
            pd.MultiIndex.from_tuples(
                [("x2", "x4", "x2"), ("y2", "y4", "y2"), ("z2", "z4", "z2")],
                names=["i2", "i4", "i2"],
            ),
        )
        assert_index_equal(
            indexes.stack_indexes([sr2.index, df2.index, sr2.index], drop_duplicates=True),
            pd.MultiIndex.from_tuples(
                [("x4", "x2"), ("y4", "y2"), ("z4", "z2")], names=["i4", "i2"]
            ),
        )
        assert_index_equal(
            indexes.stack_indexes([pd.Index([1, 1]), pd.Index([2, 3])], drop_redundant=True),
            pd.Index([2, 3]),
        )

    def test_combine_indexes(self):
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1]), pd.Index([2, 3])], drop_redundant=False),
            pd.MultiIndex.from_tuples([(1, 2), (1, 3)]),
        )
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1]), pd.Index([2, 3])], drop_redundant=True),
            pd.Index([2, 3], dtype="int64"),
        )
        assert_index_equal(
            indexes.combine_indexes(
                [pd.Index([1], name="i"), pd.Index([2, 3])], drop_redundant=True
            ),
            pd.MultiIndex.from_tuples([(1, 2), (1, 3)], names=["i", None]),
        )
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1, 2]), pd.Index([3])], drop_redundant=False),
            pd.MultiIndex.from_tuples([(1, 3), (2, 3)]),
        )
        assert_index_equal(
            indexes.combine_indexes([pd.Index([1, 2]), pd.Index([3])], drop_redundant=True),
            pd.Index([1, 2], dtype="int64"),
        )
        assert_index_equal(
            indexes.combine_indexes(
                [pd.Index([1]), pd.Index([2, 3])], drop_redundant=(False, True)
            ),
            pd.Index([2, 3], dtype="int64"),
        )
        assert_index_equal(
            indexes.combine_indexes([df2.index, df5.index]),
            pd.MultiIndex.from_tuples(
                [
                    ("x4", "x7", "x8"),
                    ("x4", "y7", "y8"),
                    ("x4", "z7", "z8"),
                    ("y4", "x7", "x8"),
                    ("y4", "y7", "y8"),
                    ("y4", "z7", "z8"),
                    ("z4", "x7", "x8"),
                    ("z4", "y7", "y8"),
                    ("z4", "z7", "z8"),
                ],
                names=["i4", "i7", "i8"],
            ),
        )

    def test_drop_levels(self):
        assert_index_equal(
            indexes.drop_levels(multi_i, "i7"),
            pd.Index(["x8", "y8", "z8"], dtype="object", name="i8"),
        )
        assert_index_equal(
            indexes.drop_levels(multi_i, "i8"),
            pd.Index(["x7", "y7", "z7"], dtype="object", name="i7"),
        )
        assert_index_equal(indexes.drop_levels(multi_i, "i9", strict=False), multi_i)
        with pytest.raises(Exception):
            indexes.drop_levels(multi_i, "i9")
        assert_index_equal(
            indexes.drop_levels(multi_i, ["i7", "i8"], strict=False),  # won't do anything
            pd.MultiIndex.from_tuples(
                [("x7", "x8"), ("y7", "y8"), ("z7", "z8")], names=["i7", "i8"]
            ),
        )
        with pytest.raises(Exception):
            indexes.drop_levels(multi_i, ["i7", "i8"])

    def test_rename_levels(self):
        i = pd.Index([1, 2, 3], name="i")
        assert_index_equal(
            indexes.rename_levels(i, {"i": "f"}),
            pd.Index([1, 2, 3], dtype="int64", name="f"),
        )
        assert_index_equal(indexes.rename_levels(i, {"a": "b"}, strict=False), i)
        with pytest.raises(Exception):
            indexes.rename_levels(i, {"a": "b"}, strict=True)
        assert_index_equal(
            indexes.rename_levels(multi_i, {"i7": "f7", "i8": "f8"}),
            pd.MultiIndex.from_tuples(
                [("x7", "x8"), ("y7", "y8"), ("z7", "z8")], names=["f7", "f8"]
            ),
        )

    def test_select_levels(self):
        assert_index_equal(
            indexes.select_levels(multi_i, "i7"),
            pd.Index(["x7", "y7", "z7"], dtype="object", name="i7"),
        )
        assert_index_equal(
            indexes.select_levels(multi_i, ["i7"]),
            pd.MultiIndex.from_tuples([("x7",), ("y7",), ("z7",)], names=["i7"]),
        )
        assert_index_equal(
            indexes.select_levels(multi_i, ["i7", "i8"]),
            pd.MultiIndex.from_tuples(
                [("x7", "x8"), ("y7", "y8"), ("z7", "z8")], names=["i7", "i8"]
            ),
        )

    def test_drop_redundant_levels(self):
        assert_index_equal(
            indexes.drop_redundant_levels(pd.Index(["a", "a"])),
            pd.Index(["a", "a"], dtype="object"),
        )  # if one unnamed, leaves as-is
        assert_index_equal(
            indexes.drop_redundant_levels(pd.MultiIndex.from_arrays([["a", "a"], ["b", "b"]])),
            pd.MultiIndex.from_tuples([("a", "b"), ("a", "b")]),  # if all unnamed, leaves as-is
        )
        assert_index_equal(
            indexes.drop_redundant_levels(
                pd.MultiIndex.from_arrays([["a", "a"], ["b", "b"]], names=["hi", None])
            ),
            pd.Index(
                ["a", "a"], dtype="object", name="hi"
            ),  # removes level with single unnamed value
        )
        assert_index_equal(
            indexes.drop_redundant_levels(
                pd.MultiIndex.from_arrays([["a", "b"], ["a", "b"]], names=["hi", "hi2"])
            ),
            pd.MultiIndex.from_tuples([("a", "a"), ("b", "b")], names=["hi", "hi2"]),  # legit
        )
        assert_index_equal(  # ignores 0-to-n
            indexes.drop_redundant_levels(
                pd.MultiIndex.from_arrays([[0, 1], ["a", "b"]], names=[None, "hi2"])
            ),
            pd.Index(["a", "b"], dtype="object", name="hi2"),
        )
        assert_index_equal(  # legit
            indexes.drop_redundant_levels(
                pd.MultiIndex.from_arrays([[0, 2], ["a", "b"]], names=[None, "hi2"])
            ),
            pd.MultiIndex.from_tuples([(0, "a"), (2, "b")], names=[None, "hi2"]),
        )
        assert_index_equal(  # legit (w/ name)
            indexes.drop_redundant_levels(
                pd.MultiIndex.from_arrays([[0, 1], ["a", "b"]], names=["hi", "hi2"])
            ),
            pd.MultiIndex.from_tuples([(0, "a"), (1, "b")], names=["hi", "hi2"]),
        )

    def test_drop_duplicate_levels(self):
        assert_index_equal(
            indexes.drop_duplicate_levels(
                pd.MultiIndex.from_arrays([[1, 2, 3], [1, 2, 3]], names=["a", "a"])
            ),
            pd.Index([1, 2, 3], dtype="int64", name="a"),
        )
        assert_index_equal(
            indexes.drop_duplicate_levels(
                pd.MultiIndex.from_tuples(
                    [(0, 1, 2, 1), ("a", "b", "c", "b")], names=["x", "y", "z", "y"]
                ),
                keep="last",
            ),
            pd.MultiIndex.from_tuples([(0, 2, 1), ("a", "c", "b")], names=["x", "z", "y"]),
        )
        assert_index_equal(
            indexes.drop_duplicate_levels(
                pd.MultiIndex.from_tuples(
                    [(0, 1, 2, 1), ("a", "b", "c", "b")], names=["x", "y", "z", "y"]
                ),
                keep="first",
            ),
            pd.MultiIndex.from_tuples([(0, 1, 2), ("a", "b", "c")], names=["x", "y", "z"]),
        )

    def test_align_index_to(self):
        index1 = pd.Index(["c", "b", "a"], name="name1")
        assert indexes.align_index_to(index1, index1) == pd.IndexSlice[:]
        index2 = pd.Index(["a", "b", "c", "a", "b", "c"], name="name1")
        np.testing.assert_array_equal(
            indexes.align_index_to(index1, index2), np.array([2, 1, 0, 2, 1, 0])
        )
        with pytest.raises(Exception):
            indexes.align_index_to(pd.Index(["a"]), pd.Index(["a", "b", "c"]))
        index3 = pd.MultiIndex.from_tuples(
            [(0, "c"), (0, "b"), (0, "a"), (1, "c"), (1, "b"), (1, "a")],
            names=["name2", "name1"],
        )
        np.testing.assert_array_equal(
            indexes.align_index_to(index1, index3), np.array([0, 1, 2, 0, 1, 2])
        )
        with pytest.raises(Exception):
            indexes.align_index_to(pd.Index(["b", "a"], name="name1"), index3)
        with pytest.raises(Exception):
            indexes.align_index_to(pd.Index(["c", "b", "a", "a"], name="name1"), index3)
        index4 = pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")],
            names=["name2", "name1"],
        )
        np.testing.assert_array_equal(
            indexes.align_index_to(index1, index4), np.array([2, 1, 0, 2, 1, 0])
        )

    def test_align_indexes(self):
        index1 = pd.Index(["a", "b", "c"])
        index2 = pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")]
        )
        index3 = pd.MultiIndex.from_tuples(
            [
                (2, 0, "a"),
                (2, 0, "b"),
                (2, 0, "c"),
                (2, 1, "a"),
                (2, 1, "b"),
                (2, 1, "c"),
                (3, 0, "a"),
                (3, 0, "b"),
                (3, 0, "c"),
                (3, 1, "a"),
                (3, 1, "b"),
                (3, 1, "c"),
            ]
        )
        indices1, indices2, indices3 = indexes.align_indexes([index1, index2, index3])
        np.testing.assert_array_equal(indices1, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(indices2, np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]))
        assert indices3 == pd.IndexSlice[:]

    def test_pick_levels(self):
        index = indexes.stack_indexes([multi_i, multi_c])
        assert indexes.pick_levels(index, required_levels=[], optional_levels=[]) == ([], [])
        assert indexes.pick_levels(
            index, required_levels=["c8", "c7", "i8", "i7"], optional_levels=[]
        ) == (
            [3, 2, 1, 0],
            [],
        )
        assert indexes.pick_levels(
            index, required_levels=["c8", None, "i8", "i7"], optional_levels=[]
        ) == (
            [3, 2, 1, 0],
            [],
        )
        assert indexes.pick_levels(
            index, required_levels=[None, "c7", "i8", "i7"], optional_levels=[]
        ) == (
            [3, 2, 1, 0],
            [],
        )
        assert indexes.pick_levels(
            index, required_levels=[None, None, None, None], optional_levels=[]
        ) == (
            [0, 1, 2, 3],
            [],
        )
        assert indexes.pick_levels(
            index, required_levels=["c8", "c7", "i8"], optional_levels=["i7"]
        ) == (
            [3, 2, 1],
            [0],
        )
        assert indexes.pick_levels(
            index, required_levels=["c8", None, "i8"], optional_levels=["i7"]
        ) == (
            [3, 2, 1],
            [0],
        )
        assert indexes.pick_levels(
            index, required_levels=[None, "c7", "i8"], optional_levels=["i7"]
        ) == (
            [3, 2, 1],
            [0],
        )
        assert indexes.pick_levels(
            index, required_levels=[None, None, None, None], optional_levels=[None]
        ) == (
            [0, 1, 2, 3],
            [None],
        )
        with pytest.raises(Exception):
            indexes.pick_levels(index, required_levels=["i8", "i8", "i8", "i8"], optional_levels=[])
        with pytest.raises(Exception):
            indexes.pick_levels(
                index, required_levels=["c8", "c7", "i8", "i7"], optional_levels=["i7"]
            )

    def test_concat_indexes(self):
        assert_index_equal(
            indexes.concat_indexes(
                pd.RangeIndex(stop=2),
                pd.RangeIndex(stop=3),
            ),
            pd.RangeIndex(start=0, stop=5, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="append",
            ),
            pd.Index([4, 5, 6, 1, 2, 3], dtype="int64", name="name"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="union",
            ),
            pd.Index([1, 2, 3, 4, 5, 6], dtype="int64", name="name"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="pd_concat",
            ),
            pd.Index([4, 5, 6, 1, 2, 3], dtype="int64", name="name"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="reset",
            ),
            pd.RangeIndex(start=0, stop=6, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([3, 4, 5], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="factorize",
                verify_integrity=False,
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 0], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([3, 4, 5], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="factorize_each",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="append",
            ),
            pd.Index(["a", "b", "c", 1, 2, 3]),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="union",
            ),
            pd.Index(["a", "b", "c", 1, 2, 3]),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="pd_concat",
            ),
            pd.Index(["a", "b", "c", 1, 2, 3]),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="reset",
            ),
            pd.RangeIndex(start=0, stop=6, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="factorize",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index(["a", "b", "c"], name="name1"),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="factorize_each",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="append",
            ),
            pd.Index([("a", 4), ("b", 5), ("c", 6), 1, 2, 3]),
        )
        with pytest.raises(Exception):
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="union",
            )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="pd_concat",
            ),
            pd.MultiIndex.from_tuples(
                [("a", 4), ("b", 5), ("c", 6), (None, 1), (None, 2), (None, 3)],
                names=("name1", "name2"),
            ),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="reset",
            ),
            pd.RangeIndex(start=0, stop=6, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="factorize",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method="factorize_each",
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method=("append", "factorize_each"),
                axis=2,
            ),
            pd.Index([("a", 4), ("b", 5), ("c", 6), 1, 2, 3], dtype="object"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.MultiIndex.from_tuples([("a", 4), ("b", 5), ("c", 6)], names=("name1", "name2")),
                pd.Index([1, 2, 3], name="name2"),
                index_concat_method=("union", "factorize_each"),
                axis=2,
            ),
            pd.Index([0, 1, 2, 3, 4, 5], name="group_idx", dtype="int64"),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.RangeIndex(stop=2),
                pd.RangeIndex(stop=3),
                keys=pd.Index(["x", "y"], name="key"),
            ),
            pd.MultiIndex.from_tuples(
                [("x", 0), ("x", 1), ("y", 0), ("y", 1), ("y", 2)], names=["key", None]
            ),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="append",
                keys=pd.Index(["x", "y"], name="key"),
            ),
            pd.MultiIndex.from_tuples(
                [("x", 4), ("x", 5), ("x", 6), ("y", 1), ("y", 2), ("y", 3)], names=["key", "name"]
            ),
        )
        with pytest.raises(Exception):
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="union",
                keys=pd.Index(["x", "y"], name="key"),
            )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="pd_concat",
                keys=pd.Index(["x", "y"], name="key"),
            ),
            pd.MultiIndex.from_tuples(
                [("x", 4), ("x", 5), ("x", 6), ("y", 1), ("y", 2), ("y", 3)], names=["key", "name"]
            ),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([4, 5, 6], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="reset",
                keys=pd.Index(["x", "y"], name="key"),
            ),
            pd.RangeIndex(start=0, stop=6, step=1),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([3, 4, 5], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="factorize",
                keys=pd.Index(["x", "y"], name="key"),
                verify_integrity=False,
                axis=2,
            ),
            pd.MultiIndex.from_tuples(
                [("x", 0), ("x", 1), ("x", 2), ("y", 3), ("y", 4), ("y", 0)],
                names=["key", "group_idx"],
            ),
        )
        assert_index_equal(
            indexes.concat_indexes(
                pd.Index([3, 4, 5], name="name"),
                pd.Index([1, 2, 3], name="name"),
                index_concat_method="factorize_each",
                keys=pd.Index(["x", "y"], name="key"),
                axis=2,
            ),
            pd.MultiIndex.from_tuples(
                [("x", 0), ("x", 1), ("x", 2), ("y", 3), ("y", 4), ("y", 5)],
                names=["key", "group_idx"],
            ),
        )


# ############# indexing ############# #


called_dict = {}

PandasIndexer = indexing.PandasIndexer
ParamIndexer = indexing.build_param_indexer(["param1", "param2", "tuple"])


class H(PandasIndexer, ParamIndexer):
    def __init__(self, a, param1_mapper, param2_mapper, tuple_mapper, level_names):
        self.a = a

        self._param1_mapper = param1_mapper
        self._param2_mapper = param2_mapper
        self._tuple_mapper = tuple_mapper
        self._level_names = level_names

        PandasIndexer.__init__(self, calling="PandasIndexer")
        ParamIndexer.__init__(
            self,
            [param1_mapper, param2_mapper, tuple_mapper],
            level_names=[level_names[0], level_names[1], level_names],
            calling="ParamIndexer",
        )

    def indexing_func(self, pd_indexing_func, calling=None):
        # As soon as you call iloc etc., performs it on each dataframe and mapper and returns a new class instance
        called_dict[calling] = True
        param1_mapper = indexing.indexing_on_mapper(self._param1_mapper, self.a, pd_indexing_func)
        param2_mapper = indexing.indexing_on_mapper(self._param2_mapper, self.a, pd_indexing_func)
        tuple_mapper = indexing.indexing_on_mapper(self._tuple_mapper, self.a, pd_indexing_func)
        return H(
            pd_indexing_func(self.a), param1_mapper, param2_mapper, tuple_mapper, self._level_names
        )

    @classmethod
    def run(cls, a, params1, params2, level_names=("p1", "p2")):
        a = reshaping.to_2d(a)
        # Build column hierarchy
        params1_idx = pd.Index(params1, name=level_names[0])
        params2_idx = pd.Index(params2, name=level_names[1])
        params_idx = indexes.stack_indexes([params1_idx, params2_idx])
        new_columns = indexes.combine_indexes([params_idx, a.columns])

        # Build mappers
        param1_mapper = np.repeat(params1, len(a.columns))
        param1_mapper = pd.Series(param1_mapper, index=new_columns)

        param2_mapper = np.repeat(params2, len(a.columns))
        param2_mapper = pd.Series(param2_mapper, index=new_columns)

        tuple_mapper = list(zip(*list(map(lambda x: x.values, [param1_mapper, param2_mapper]))))
        tuple_mapper = pd.Series(tuple_mapper, index=new_columns)

        # Tile a to match the length of new_columns
        a = vbt.ArrayWrapper(a.index, new_columns, 2).wrap(reshaping.tile(a.values, 4, axis=1))
        return cls(a, param1_mapper, param2_mapper, tuple_mapper, level_names)


# Similate an indicator with two params
h = H.run(df4, [0.1, 0.1, 0.2, 0.2], [0.3, 0.4, 0.5, 0.6])


class TestIndexing:
    def test_kwargs(self):
        h[(0.1, 0.3, "a6")]
        assert called_dict["PandasIndexer"]
        h.param1_loc[0.1]
        assert called_dict["ParamIndexer"]

    def test_pandas_indexing(self):
        # __getitem__
        assert_series_equal(
            h[(0.1, 0.3, "a6")].a,
            pd.Series(
                np.array([1, 4, 7]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                name=(0.1, 0.3, "a6"),
            ),
        )
        # loc
        assert_frame_equal(
            h.loc[:, (0.1, 0.3, "a6") : (0.1, 0.3, "c6")].a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, 0.3, "a6"), (0.1, 0.3, "b6"), (0.1, 0.3, "c6")],
                    names=["p1", "p2", "c6"],
                ),
            ),
        )
        # iloc
        assert_frame_equal(
            h.iloc[-2:, -2:].a,
            pd.DataFrame(
                np.array([[5, 6], [8, 9]]),
                index=pd.Index(["y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [(0.2, 0.6, "b6"), (0.2, 0.6, "c6")], names=["p1", "p2", "c6"]
                ),
            ),
        )
        # xs
        assert_frame_equal(
            h.xs((0.1, 0.3), level=("p1", "p2"), axis=1).a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
            ),
        )

    def test_param_indexing(self):
        # param1
        assert_frame_equal(
            h.param1_loc[0.1].a,
            pd.DataFrame(
                np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [(0.3, "a6"), (0.3, "b6"), (0.3, "c6"), (0.4, "a6"), (0.4, "b6"), (0.4, "c6")],
                    names=["p2", "c6"],
                ),
            ),
        )
        # param2
        assert_frame_equal(
            h.param2_loc[0.3].a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, "a6"), (0.1, "b6"), (0.1, "c6")], names=["p1", "c6"]
                ),
            ),
        )
        # tuple
        assert_frame_equal(
            h.tuple_loc[(0.1, 0.3)].a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
            ),
        )
        assert_frame_equal(
            h.tuple_loc[(0.1, 0.3) : (0.1, 0.3)].a,
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [(0.1, 0.3, "a6"), (0.1, 0.3, "b6"), (0.1, 0.3, "c6")],
                    names=["p1", "p2", "c6"],
                ),
            ),
        )
        assert_frame_equal(
            h.tuple_loc[[(0.1, 0.3), (0.1, 0.3)]].a,
            pd.DataFrame(
                np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [7, 8, 9, 7, 8, 9]]),
                index=pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
                columns=pd.MultiIndex.from_tuples(
                    [
                        (0.1, 0.3, "a6"),
                        (0.1, 0.3, "b6"),
                        (0.1, 0.3, "c6"),
                        (0.1, 0.3, "a6"),
                        (0.1, 0.3, "b6"),
                        (0.1, 0.3, "c6"),
                    ],
                    names=["p1", "p2", "c6"],
                ),
            ),
        )

    @pytest.mark.parametrize(
        "test_inputs",
        [
            (0, a1, a2, sr_none, sr1, sr2),
            (0, a1, a2, a3, a4, a5, sr_none, sr1, sr2, df_none, df1, df2, df3, df4),
        ],
    )
    def test_flex(self, test_inputs):
        raw_args = reshaping.broadcast(
            *test_inputs,
            keep_flex=True,
            align_index=False,
        )
        bc_args = reshaping.broadcast(
            *test_inputs,
            keep_flex=False,
            align_index=False,
        )
        for r in range(len(test_inputs)):
            raw_arg = raw_args[r]
            bc_arg = np.array(bc_args[r])
            bc_arg_2d = reshaping.to_2d(bc_arg)
            for col in range(bc_arg_2d.shape[1]):
                for i in range(bc_arg_2d.shape[0]):
                    assert bc_arg_2d[i, col] == flex_indexing.flex_select_nb(raw_arg, i, col)

    def test_get_index_points(self):
        index = pd.date_range("2020-01-01", "2020-01-03", freq="3h", tz="+0400")
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every=2), np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every=2, start=5, end=10), np.array([5, 7, 9])
        )

        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h"),
            np.array([0, 2, 4, 5, 7, 9, 10, 12, 14, 15]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", add_delta="1h"),
            np.array([1, 2, 4, 6, 7, 9, 11, 12, 14, 16]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start=5),
            np.array([5, 7, 9, 10, 12, 14, 15]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start=5, end=12),
            np.array([5, 7, 9, 10, 12]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start=index[5]),
            np.array([5, 7, 9, 10, 12, 14, 15]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start=index[5], end=index[12]),
            np.array([5, 7, 9, 10, 12]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, every="5h", start="2020-01-01 15:00:00"),
            np.array([5, 7, 9, 10, 12, 14, 15]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index, every="5h", start="2020-01-01 15:00:00", end="2020-01-02 12:00:00"
            ),
            np.array([5, 7, 9, 10, 12]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, at_time="12:00"), np.array([4, 12])
        )

        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
            ),
            np.arange(len(index)),
        )
        np.testing.assert_array_equal(indexing.get_index_points(index, start=5), np.array([5]))
        np.testing.assert_array_equal(
            indexing.get_index_points(index, start=index[0]), np.array([0])
        )

        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5],
                end=index[10],
                kind="labels",
            ),
            np.array([5]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5] + pd.Timedelta(nanoseconds=1),
                end=index[10],
                kind="labels",
            ),
            np.array([6]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5] - pd.Timedelta(nanoseconds=1),
                end=index[10],
                kind="labels",
            ),
            np.array([5]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5],
                end=index[10] - pd.Timedelta(nanoseconds=1),
                kind="labels",
            ),
            np.array([5]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(
                index,
                start=index[5],
                end=index[10] + pd.Timedelta(nanoseconds=1),
                kind="labels",
            ),
            np.array([5]),
        )

        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[5, 10]), np.array([5, 10])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=index[[5, 10]], at_time="12:00"), np.array([4, 12])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[index[5], index[10]]), np.array([5, 10])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[index[5], index[10]], start=index[7]),
            np.array([10]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[index[5], index[10]], end=index[7]), np.array([5])
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[index[5], index[10]], end=index[10]),
            np.array([5, 10]),
        )
        np.testing.assert_array_equal(
            indexing.get_index_points(index, on=[index[5], index[10]], end=10), np.array([5])
        )

    def test_get_index_ranges(self):
        index = pd.date_range("2020-01-01", "2020-01-03", freq="3h", tz="+0400")
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, closed_start=False, closed_end=False)
            ),
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, closed_start=True, closed_end=False)
            ),
            np.array([[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, closed_start=False, closed_end=True)
            ),
            np.array([[1, 3], [3, 5], [5, 7], [7, 9], [9, 11], [11, 13], [13, 15], [15, 17]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, closed_start=True, closed_end=True)
            ),
            np.array([[0, 3], [2, 5], [4, 7], [6, 9], [8, 11], [10, 13], [12, 15], [14, 17]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index, every=2, start=5, end=10, closed_start=False, closed_end=False
                )
            ),
            np.array([[6, 7], [8, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index, every=2, start=5, end=10, closed_start=True, closed_end=False
                )
            ),
            np.array([[5, 7], [7, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index, every=2, start=5, end=10, closed_start=False, closed_end=True
                )
            ),
            np.array([[6, 8], [8, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index, every=2, start=5, end=10, closed_start=True, closed_end=True
                )
            ),
            np.array([[5, 8], [7, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every=4, lookback_period=1)),
            np.array([[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=4, lookback_period=1, closed_end=True)
            ),
            np.array([[0, 2], [4, 6], [8, 10], [12, 14]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, start=5, end=10, lookback_period=1)
            ),
            np.array([[5, 6], [7, 8], [9, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every=2, start=5, end=10, fixed_start=True)
            ),
            np.array([[5, 7], [5, 9]]),
        )
        with pytest.raises(Exception):
            indexing.get_index_ranges(
                index, every=2, start=5, end=10, fixed_start=True, lookback_period=1
            )

        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every="5h", closed_start=False, closed_end=False)
            ),
            np.array(
                [[1, 2], [2, 4], [4, 5], [6, 7], [7, 9], [9, 10], [11, 12], [12, 14], [14, 15]]
            ),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every="5h", closed_start=True, closed_end=False)
            ),
            np.array(
                [[0, 2], [2, 4], [4, 5], [5, 7], [7, 9], [9, 10], [10, 12], [12, 14], [14, 15]]
            ),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every="5h", closed_start=False, closed_end=True)
            ),
            np.array(
                [[1, 2], [2, 4], [4, 6], [6, 7], [7, 9], [9, 11], [11, 12], [12, 14], [14, 16]]
            ),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every="5h", closed_start=True, closed_end=True)
            ),
            np.array(
                [[0, 2], [2, 4], [4, 6], [5, 7], [7, 9], [9, 11], [10, 12], [12, 14], [14, 16]]
            ),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index, every="5h", add_start_delta="1h", add_end_delta="1h"
                )
            ),
            np.array(
                [[1, 2], [2, 4], [4, 6], [6, 7], [7, 9], [9, 11], [11, 12], [12, 14], [14, 16]]
            ),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", start=5)),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12], [12, 14], [14, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", start=5, end=12)),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", start=index[5])),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12], [12, 14], [14, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every="5h", start=index[5], end=index[12])
            ),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, every="5h", start="2020-01-01 15:00:00")
            ),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12], [12, 14], [14, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index, every="5h", start="2020-01-01 15:00:00", end="2020-01-02 12:00:00"
                )
            ),
            np.array([[5, 7], [7, 9], [9, 10], [10, 12]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index, every="5h", start="2020-01-01 15:00:00", fixed_start=True
                )
            ),
            np.array([[5, 7], [5, 9], [5, 10], [5, 12], [5, 14], [5, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", lookback_period="12h")),
            np.array([[0, 4], [2, 6], [4, 8], [5, 9], [7, 11], [9, 13], [10, 14], [12, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, every="5h", lookback_period=4)),
            np.array([[0, 4], [2, 6], [4, 8], [5, 9], [7, 11], [9, 13], [10, 14], [12, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start_time="12:00")),
            np.array([[4, 8], [12, 16]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start_time="12:00", end_time="15:00")),
            np.array([[4, 5], [12, 13]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end_time="15:00")),
            np.array([[0, 5], [8, 13], [16, 17]]),
        )
        assert (
            len(
                np.column_stack(
                    indexing.get_index_ranges(index, start_time="15:00", end_time="15:00")
                )
            )
            == 0
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start_time="15:00", end_time="15:01")),
            np.array([[5, 6], [13, 14]], dtype=int_),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start_time="15:00", end_time="14:59")),
            np.array([[5, 13], [13, 17]], dtype=int_),
        )

        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                )
            ),
            np.array([[0, 17]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=5)), np.array([[5, 17]])
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end=10)), np.array([[0, 10]])
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=5, end=10)), np.array([[5, 10]])
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=index[0], end=index[10])),
            np.array([[0, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=5, end=[10, 15])),
            np.array([[5, 10], [5, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, start=[5, 7], end=10)),
            np.array([[5, 10], [7, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, start=index[5], end=[index[10], index[15]])
            ),
            np.array([[5, 10], [5, 15]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, start=[index[5], index[7]], end=index[10])
            ),
            np.array([[5, 10], [7, 10]]),
        )
        with pytest.raises(Exception):
            indexing.get_index_ranges(index, start=0, end=index[10])
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(index, start=index[0], end=index[10], kind="labels")
            ),
            np.array([[0, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end=10, lookback_period=3)),
            np.array([[7, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end=index[10], lookback_period=3)),
            np.array([[7, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(indexing.get_index_ranges(index, end=index[10], lookback_period="9h")),
            np.array([[7, 10]]),
        )

        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[9],
                    kind="labels",
                )
            ),
            np.array([[5, 8]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5] + pd.Timedelta(nanoseconds=1),
                    end=index[10],
                    kind="labels",
                )
            ),
            np.array([[6, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5] - pd.Timedelta(nanoseconds=1),
                    end=index[10],
                    kind="labels",
                )
            ),
            np.array([[5, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10] - pd.Timedelta(nanoseconds=1),
                    kind="labels",
                )
            ),
            np.array([[5, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10] + pd.Timedelta(nanoseconds=1),
                    kind="labels",
                )
            ),
            np.array([[5, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10],
                    kind="labels",
                    closed_start=False,
                )
            ),
            np.array([[6, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5] + pd.Timedelta(nanoseconds=1),
                    end=index[10],
                    kind="labels",
                    closed_start=False,
                )
            ),
            np.array([[6, 9]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10],
                    kind="labels",
                    closed_end=True,
                )
            ),
            np.array([[5, 10]]),
        )
        np.testing.assert_array_equal(
            np.column_stack(
                indexing.get_index_ranges(
                    index,
                    start=index[5],
                    end=index[10] + pd.Timedelta(nanoseconds=1),
                    kind="labels",
                    closed_end=True,
                )
            ),
            np.array([[5, 10]]),
        )

    def test_get_idxs(self):
        i = pd.Index([1, 2, 3, 4, 5, 6], name="i")
        c = pd.Index([1, 2, 3, 4], name="c")
        dti = pd.date_range("2020-01-01", "2020-01-03", freq="1h", tz="Europe/Berlin")
        mi = pd.MultiIndex.from_arrays([[1, 1, 2, 2, 3, 3], [4, 5, 4, 5, 4, 5]], names=["i1", "i2"])
        mc = pd.MultiIndex.from_arrays([[1, 1, 2, 2], [3, 4, 3, 4]], names=["c1", "c2"])

        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(0), i, c)
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx((0, 2)), i, c)
        np.testing.assert_array_equal(row_idxs, np.array([0, 2]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(((0, 2),)), i, c)
        np.testing.assert_array_equal(row_idxs, np.array([[0, 2]]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 2)), i, c)
        assert row_idxs == slice(0, 2, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 2, 2)), i, c)
        assert row_idxs == slice(0, 2, 2)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(None, None, None)), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(0), dti, c)
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx((0, 2)), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([0, 2]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(((0, 2),)), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([[0, 2]]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 2)), dti, c)
        assert row_idxs == slice(0, 2, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 2, 2)), dti, c)
        assert row_idxs == slice(0, 2, 2)
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(0), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == 0
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((0, 2)), i, c)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(0, 2)), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 2, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(0, 2, 2)), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 2, 2)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(None, None, None)), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(1, kind="labels"), i, c)
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx((1, 3), kind="labels"), i, c)
        np.testing.assert_array_equal(row_idxs, np.array([0, 2]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(1, 3), kind="labels"), i, c)
        assert row_idxs == slice(0, 3, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(1, 3, 2), kind="labels"), i, c)
        assert row_idxs == slice(0, 3, 2)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(0, 10), kind="labels"), i, c)
        assert row_idxs == slice(0, 6, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx("2020-01-01 17:00"), dti, c)
        assert row_idxs == 17
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            indexing.rowidx(("2020-01-01", "2020-01-02 17:00")), dti, c
        )
        np.testing.assert_array_equal(row_idxs, np.concatenate((np.arange(24), np.array([41]))))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx((1, 4)), mi, mc)
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.rowidx((1, 3)), mi, mc)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice((1, 4), (2, 4))), mi, mc)
        assert row_idxs == slice(0, 3, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice((1, 4), (2, 4), 2)), mi, mc)
        assert row_idxs == slice(0, 3, 2)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rowidx(slice(None, None, None)), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(1, kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == 0
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((1, 3), kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(1, 3), kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 3, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(1, 3, 2), kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 3, 2)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(0, 10), kind="labels"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 4, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((1, 3), kind="labels"), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == 0
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx((1, 2)), mi, mc)
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx((2, 5)), mi, mc)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(((1, 3), (2, 3))), mi, mc)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice((1, 3), (2, 3))), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 3, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice((1, 3), (2, 3), 2)), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 3, 2)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice((0, 0), (10, 10))), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 4, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(slice(None, None, None)), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(1, level="c"), i, c)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == 0
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((1, 2), level="c"), i, c)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 1]))
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx((1, 2, 0), level="c"), i, c)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(1, level="c1"), mi, mc)
        assert row_idxs == slice(None, None, None)
        assert col_idxs == slice(0, 2, None)
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx(0, level="c1"), mi, mc)
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx(3, level="c2"), mi, mc)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))
        row_idxs, col_idxs = indexing.get_idxs(indexing.colidx((3, 4), level="c2"), mi, mc)
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2, 1, 3]))
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx((3, 4, 0), level="c2"), mi, mc)
        with pytest.raises(Exception):
            indexing.get_idxs(indexing.colidx(slice(0, 10), level="c2"), mi, mc)
        row_idxs, col_idxs = indexing.get_idxs(
            indexing.colidx(((1, 3), (2, 3)), level=("c1", "c2")), mi, mc
        )
        assert row_idxs == slice(None, None, None)
        np.testing.assert_array_equal(col_idxs, np.array([0, 2]))

        row_idxs, col_idxs = indexing.get_idxs(indexing.pointidx(on="2020"), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([0]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.pointidx(on=(1, 3, 5)), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([1, 3, 5]))
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(
            indexing.rangeidx(start="2020-01-01 12:00", end="2020-01-01 17:00"), dti, c
        )
        np.testing.assert_array_equal(row_idxs, np.array([[12, 17]]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(indexing.rangeidx(start=(0, 4), end=(2, 6)), dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([[0, 2], [4, 6]]))
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(
            indexing.idx(
                indexing.rowidx(1),
                indexing.colidx(2),
            ),
            i,
            c,
        )
        assert row_idxs == 1
        assert col_idxs == 2
        with pytest.raises(Exception):
            indexing.get_idxs(
                indexing.idx(
                    indexing.colidx(1),
                    indexing.colidx(2),
                ),
                i,
                c,
            )
        with pytest.raises(Exception):
            indexing.get_idxs(
                indexing.idx(
                    indexing.rowidx(1),
                    indexing.rowidx(2),
                ),
                i,
                c,
            )

        row_idxs, col_idxs = indexing.get_idxs(1, i, c, kind="labels")
        assert row_idxs == 0
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs((1, 3), i, c, kind="labels")
        np.testing.assert_array_equal(row_idxs, np.array([0, 2]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(slice(1, 2), i, c, kind="labels")
        assert row_idxs == slice(0, 2, None)
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs("2020-01-01 17:00", dti, c)
        assert row_idxs == 17
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(("2020-01-01", "2020-01-02 17:00"), dti, c)
        np.testing.assert_array_equal(row_idxs, np.concatenate((np.arange(24), np.array([41]))))
        assert col_idxs == slice(None, None, None)

        row_idxs, col_idxs = indexing.get_idxs(vbt.DTC(year=2020), dti, c)
        np.testing.assert_array_equal(row_idxs, np.arange(len(dti)))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs("2020", dti, c)
        np.testing.assert_array_equal(row_idxs, np.arange(len(dti)))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs("2020-01", dti, c)
        np.testing.assert_array_equal(row_idxs, np.arange(len(dti)))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs("2020-01-01", dti, c)
        np.testing.assert_array_equal(row_idxs, np.arange(24))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs("2020-01-01 17:00", dti, c)
        np.testing.assert_array_equal(row_idxs, np.array([17]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(slice("2020-01-01", "2020-01-02"), dti, c)
        np.testing.assert_array_equal(row_idxs, np.arange(24))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("2020-01-01", "2020-01-02"), dti, c, closed_end=True
        )
        np.testing.assert_array_equal(row_idxs, np.arange(48))
        row_idxs, col_idxs = indexing.get_idxs(
            slice("2020-01-01", "2020-01-02"), dti, c, closed_start=False, closed_end=True
        )
        np.testing.assert_array_equal(row_idxs, np.arange(24, 48))
        with pytest.raises(Exception):
            indexing.get_idxs(slice("2020-01-01", "2020-01-02"), dti, c, closed_start=False)

        row_idxs, col_idxs = indexing.get_idxs(
            slice("12:00", "14:00"), dti, c, closed_start=False, closed_end=False
        )
        np.testing.assert_array_equal(row_idxs, np.array([13, 37]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("12:00", "14:00"), dti, c, closed_start=False, closed_end=True
        )
        np.testing.assert_array_equal(row_idxs, np.array([13, 14, 37, 38]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("12:00", "14:00"), dti, c, closed_start=True, closed_end=False
        )
        np.testing.assert_array_equal(row_idxs, np.array([12, 13, 36, 37]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("12:00", "14:00"), dti, c, closed_start=True, closed_end=True
        )
        np.testing.assert_array_equal(row_idxs, np.array([12, 13, 14, 36, 37, 38]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("23:00", "01:00"), dti, c, closed_start=False, closed_end=False
        )
        np.testing.assert_array_equal(row_idxs, np.array([0, 24, 48]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("23:00", "01:00"), dti, c, closed_start=False, closed_end=True
        )
        np.testing.assert_array_equal(row_idxs, np.array([0, 1, 24, 25, 48]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("23:00", "01:00"), dti, c, closed_start=True, closed_end=False
        )
        np.testing.assert_array_equal(row_idxs, np.array([0, 23, 24, 47, 48]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("23:00", "01:00"), dti, c, closed_start=True, closed_end=True
        )
        np.testing.assert_array_equal(row_idxs, np.array([0, 1, 23, 24, 25, 47, 48]))
        assert col_idxs == slice(None, None, None)

        dti = pd.date_range("2020-01-01", "2020-01-03", freq="15min", tz="Europe/Berlin")
        row_idxs, col_idxs = indexing.get_idxs(
            slice("14:30", "15:45"), dti, c, closed_start=False, closed_end=False
        )
        np.testing.assert_array_equal(row_idxs, np.array([59, 60, 61, 62, 155, 156, 157, 158]))
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("14:30", "15:45"), dti, c, closed_start=False, closed_end=True
        )
        np.testing.assert_array_equal(
            row_idxs, np.array([59, 60, 61, 62, 63, 155, 156, 157, 158, 159])
        )
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("14:30", "15:45"), dti, c, closed_start=True, closed_end=False
        )
        np.testing.assert_array_equal(
            row_idxs, np.array([58, 59, 60, 61, 62, 154, 155, 156, 157, 158])
        )
        assert col_idxs == slice(None, None, None)
        row_idxs, col_idxs = indexing.get_idxs(
            slice("14:30", "15:45"), dti, c, closed_start=True, closed_end=True
        )
        np.testing.assert_array_equal(
            row_idxs, np.array([58, 59, 60, 61, 62, 63, 154, 155, 156, 157, 158, 159])
        )
        assert col_idxs == slice(None, None, None)

    def test_idx_setter_factory(self):
        idx_setter = indexing.IdxDict({0: 0, indexing.idx(1): 1}).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [(0, 0), (indexing.Idxr(1), 1)],
        )
        sr = pd.Series([3, 2, 1], index=["x", "y", "z"])
        idx_setter = indexing.IdxSeries(sr).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [(indexing.Idxr(sr.index), sr.values)],
        )
        idx_setter = indexing.IdxSeries(sr, split=True).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [
                (indexing.Idxr(sr.index[0]), sr.values[0].item()),
                (indexing.Idxr(sr.index[1]), sr.values[1].item()),
                (indexing.Idxr(sr.index[2]), sr.values[2].item()),
            ],
        )
        df = pd.DataFrame([[6, 5], [4, 3], [2, 1]], index=["x", "y", "z"], columns=["a", "b"])
        idx_setter = indexing.IdxFrame(df).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [(indexing.Idxr(indexing.RowIdxr(df.index), indexing.ColIdxr(df.columns)), df.values)],
        )
        idx_setter = indexing.IdxFrame(df, split="columns").get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [
                (
                    indexing.Idxr(indexing.RowIdxr(df.index), indexing.ColIdxr(df.columns[0])),
                    df.values[:, 0],
                ),
                (
                    indexing.Idxr(indexing.RowIdxr(df.index), indexing.ColIdxr(df.columns[1])),
                    df.values[:, 1],
                ),
            ],
        )
        idx_setter = indexing.IdxFrame(df, split="rows").get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[0]), indexing.ColIdxr(df.columns)),
                    df.values[0],
                ),
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[1]), indexing.ColIdxr(df.columns)),
                    df.values[1],
                ),
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[2]), indexing.ColIdxr(df.columns)),
                    df.values[2],
                ),
            ],
        )
        idx_setter = indexing.IdxFrame(df, split=True).get()
        assert checks.is_deep_equal(
            idx_setter.idx_items,
            [
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[0]), indexing.ColIdxr(df.columns[0])),
                    df.values[0, 0].item(),
                ),
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[1]), indexing.ColIdxr(df.columns[0])),
                    df.values[1, 0].item(),
                ),
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[2]), indexing.ColIdxr(df.columns[0])),
                    df.values[2, 0].item(),
                ),
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[0]), indexing.ColIdxr(df.columns[1])),
                    df.values[0, 1].item(),
                ),
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[1]), indexing.ColIdxr(df.columns[1])),
                    df.values[1, 1].item(),
                ),
                (
                    indexing.Idxr(indexing.RowIdxr(df.index[2]), indexing.ColIdxr(df.columns[1])),
                    df.values[2, 1].item(),
                ),
            ],
        )
        records = pd.Series([3, 2, 1], index=["x", "y", "z"])
        idx_setters = indexing.IdxRecords(records).get()
        assert len(idx_setters) == 1
        assert checks.is_deep_equal(
            idx_setters["_1"].idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(records.index[0]), None), records.values[0].item()),
                (indexing.Idxr(indexing.RowIdxr(records.index[1]), None), records.values[1].item()),
                (indexing.Idxr(indexing.RowIdxr(records.index[2]), None), records.values[2].item()),
            ],
        )
        records = pd.DataFrame([[9, 8, 7], [6, 5, 4], [3, 2, 1]], columns=["row", "col", "X"])
        idx_setters = indexing.IdxRecords(records).get()
        assert len(idx_setters) == 1
        assert checks.is_deep_equal(
            idx_setters["X"].idx_items,
            [
                (
                    indexing.Idxr(
                        indexing.RowIdxr(records["row"].values[0].item()),
                        indexing.ColIdxr(records["col"].values[0].item()),
                    ),
                    records["X"].values[0].item(),
                ),
                (
                    indexing.Idxr(
                        indexing.RowIdxr(records["row"].values[1].item()),
                        indexing.ColIdxr(records["col"].values[1].item()),
                    ),
                    records["X"].values[1].item(),
                ),
                (
                    indexing.Idxr(
                        indexing.RowIdxr(records["row"].values[2].item()),
                        indexing.ColIdxr(records["col"].values[2].item()),
                    ),
                    records["X"].values[2].item(),
                ),
            ],
        )
        records1 = pd.DataFrame([[9, 8, 7], [6, 5, 4], [3, 2, 1]], columns=["row", "col", "X"])
        idx_setters1 = indexing.IdxRecords(records1).get()
        records2 = pd.DataFrame([[8, 7], [5, 4], [2, 1]], index=[9, 6, 3], columns=["col", "X"])
        idx_setters2 = indexing.IdxRecords(records2).get()
        assert checks.is_deep_equal(idx_setters1, idx_setters2)
        records3 = pd.DataFrame(
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]], index=[20, 30, 40], columns=["row", "col", "X"]
        )
        with pytest.raises(Exception):
            indexing.IdxRecords(records3).get()
        idx_setters3 = indexing.IdxRecords(records3, row_field="row").get()
        assert checks.is_deep_equal(idx_setters1, idx_setters3)
        records = [
            dict(row=1, X=2, Y=3),
            dict(col=4, X=5),
            dict(row=6, col=7, Y=8),
            dict(Z=9),
        ]
        idx_setters = indexing.IdxRecords(records).get()
        assert len(idx_setters) == 3
        assert checks.is_deep_equal(
            idx_setters["X"].idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(1), None), 2),
                (indexing.Idxr(None, indexing.ColIdxr(4)), 5),
            ],
        )
        assert checks.is_deep_equal(
            idx_setters["Y"].idx_items,
            [
                (indexing.Idxr(indexing.RowIdxr(1), None), 3),
                (indexing.Idxr(indexing.RowIdxr(6), indexing.ColIdxr(7)), 8),
            ],
        )
        assert checks.is_deep_equal(
            idx_setters["Z"].idx_items,
            [
                ("_def", 9),
            ],
        )


# ############# flex_indexing ############# #


class TestFlexIndexing:
    def test_flex_select_nb(self):
        arr_1d = np.array([1, 2, 3])
        arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert flex_indexing.flex_select_1d_nb(arr_1d, 0) == arr_1d[0]
        assert flex_indexing.flex_select_1d_nb(arr_1d, 1) == arr_1d[1]
        with pytest.raises(Exception):
            flex_indexing.flex_select_nb(arr_1d, 100)
        assert (
            flex_indexing.flex_select_1d_pr_nb(arr_1d, 100, rotate_rows=True)
            == arr_1d[100 % arr_1d.shape[0]]
        )
        assert (
            flex_indexing.flex_select_1d_pc_nb(arr_1d, 100, rotate_cols=True)
            == arr_1d[100 % arr_1d.shape[0]]
        )
        assert flex_indexing.flex_select_nb(arr_2d, 0, 0) == arr_2d[0, 0]
        assert flex_indexing.flex_select_nb(arr_2d, 1, 0) == arr_2d[1, 0]
        assert flex_indexing.flex_select_nb(arr_2d, 0, 1) == arr_2d[0, 1]
        assert (
            flex_indexing.flex_select_nb(arr_2d, 100, 0, rotate_rows=True)
            == arr_2d[100 % arr_2d.shape[0], 0]
        )
        with pytest.raises(Exception):
            flex_indexing.flex_select_nb(arr_2d, 100, 0, rotate_rows=False)
        assert (
            flex_indexing.flex_select_nb(arr_2d, 0, 100, rotate_cols=True)
            == arr_2d[0, 100 % arr_2d.shape[1]]
        )
        with pytest.raises(Exception):
            flex_indexing.flex_select_nb(arr_2d, 0, 100, rotate_cols=False)
