import os

import pytest
import vectorbtpro as vbt
from vectorbtpro.base import indexing, wrapping

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


# ############# wrapping ############# #


sr2_wrapper = vbt.ArrayWrapper.from_obj(sr2)
df2_wrapper = vbt.ArrayWrapper.from_obj(df2)
df4_wrapper = vbt.ArrayWrapper.from_obj(df4)

sr2_wrapper_co = sr2_wrapper.replace(column_only_select=True)
df4_wrapper_co = df4_wrapper.replace(column_only_select=True)

sr2_grouped_wrapper = sr2_wrapper.replace(group_by=np.array(["g1"]), group_select=True)
df4_grouped_wrapper = df4_wrapper.replace(group_by=np.array(["g1", "g1", "g2"]), group_select=True)

sr2_grouped_wrapper_co = sr2_grouped_wrapper.replace(column_only_select=True, group_select=True)
df4_grouped_wrapper_co = df4_grouped_wrapper.replace(column_only_select=True, group_select=True)

sr2_grouped_wrapper_conog = sr2_grouped_wrapper.replace(column_only_select=True, group_select=False)
df4_grouped_wrapper_conog = df4_grouped_wrapper.replace(column_only_select=True, group_select=False)


class TestArrayWrapper:
    def test_row_stack(self):
        assert vbt.ArrayWrapper.row_stack(
            (
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
            )
        ) == vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
        )
        assert vbt.ArrayWrapper.row_stack(
            (
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
            )
        ) == vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2).row_stack(
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2)
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.RangeIndex(start=0, stop=3), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.RangeIndex(start=3, stop=6), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.RangeIndex(start=0, stop=6, step=1))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(
                pd.RangeIndex(start=0, stop=3, name="i"), pd.Index(["a", "b"], name="c"), 2
            ),
            vbt.ArrayWrapper(
                pd.RangeIndex(start=3, stop=6, name="i"), pd.Index(["a", "b"], name="c"), 2
            ),
        )
        assert_index_equal(wrapper.index, pd.RangeIndex(start=0, stop=6, step=1, name="i"))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(
                pd.RangeIndex(start=0, stop=3, name="i1"), pd.Index(["a", "b"], name="c"), 2
            ),
            vbt.ArrayWrapper(
                pd.RangeIndex(start=3, stop=6, name="i2"), pd.Index(["a", "b"], name="c"), 2
            ),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 3, 4, 5], dtype="int64"))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.RangeIndex(start=0, stop=3), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.RangeIndex(start=4, stop=7), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 4, 5, 6], dtype="int64"))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 4, 5, 6]))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([2, 3, 4]), pd.Index(["a", "b"], name="c"), 2),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([2, 3, 4]), pd.Index(["a", "b"], name="c"), 2),
            verify_integrity=False,
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 2, 3, 4]))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([2, 1, 0]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([3, 4, 5]), pd.Index(["a", "b"], name="c"), 2),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([2, 1, 0]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([3, 4, 5]), pd.Index(["a", "b"], name="c"), 2),
            verify_integrity=False,
        )
        assert_index_equal(wrapper.index, pd.Index([2, 1, 0, 3, 4, 5]))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index(["x", "y", "z"]), pd.Index(["a", "b"], name="c"), 2),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index(["x", "y", "z"]), pd.Index(["a", "b"], name="c"), 2),
            verify_integrity=False,
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, "x", "y", "z"]))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index(["x", "y", "z"]), pd.Index(["a", "b"], name="c"), 2),
            index=np.arange(6),
        )
        assert_index_equal(wrapper.index, pd.Index(np.arange(6)))
        index1 = pd.date_range("2020-01-01", "2020-01-05", inclusive="left")
        index2 = pd.date_range("2020-01-05", "2020-01-10", inclusive="left")
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq == pd.Timedelta(days=1)
        index1 = pd.date_range("2020-01-01", "2020-01-05", inclusive="left")
        index2 = pd.date_range("2020-01-05", "2020-01-06", inclusive="left")
        index3 = pd.date_range("2020-01-07", "2020-01-10", inclusive="left")
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2.append(index3), pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2).append(index3))
        assert wrapper.freq == pd.Timedelta(days=1)
        with pytest.raises(Exception):
            index1 = pd.date_range("2020-01-01", "2020-01-05", freq="1d", inclusive="left")
            index2 = pd.date_range("2020-01-05", "2020-01-10", freq="1h", inclusive="left")
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(index2, pd.Index(["a", "b"], name="c"), 2),
            )
        index1 = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
        index2 = pd.DatetimeIndex(["2020-01-06", "2020-01-10"])
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["a", "b"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq is None
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["a", "b"], name="c"), 2),
            freq="3d",
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq == pd.Timedelta(days=3)
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]),
                pd.MultiIndex.from_tuples([("a", "a"), ("b", "b")], names=["c1", "c2"]),
                2,
            ),
        )
        assert_index_equal(
            wrapper.columns, pd.MultiIndex.from_tuples([("a", "a"), ("b", "b")], names=["c1", "c2"])
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]),
                pd.MultiIndex.from_tuples([("a", "a"), ("b", "b")], names=["c1", "c2"]),
                2,
            ),
            clean_index_kwargs=dict(drop_duplicates=False),
        )
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples([("a", "a", "a"), ("b", "b", "b")], names=["c1", "c1", "c2"]),
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c2"), 2),
            columns=["a2", "b2"],
        )
        assert_index_equal(wrapper.columns, pd.Index(["a2", "b2"]))
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a2"], name="c2"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 4, 5, 6]))
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples([("a", "a2"), ("b", "a2")], names=["c1", "c2"]),
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(
                pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=False
            ),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=False
            ),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index(["a", "b"], name="c", dtype="object"),
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=True),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=True),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index(["group", "group"], dtype="object", name="group"),
        )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(
                pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=[0, 1]
            ),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=[0, 1]
            ),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1], dtype="int64", name="group"),
        )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(
                    pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=True
                ),
                vbt.ArrayWrapper(
                    pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=False
                ),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(
                    pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, group_by=[0, 1]
                ),
                vbt.ArrayWrapper(
                    pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, group_by=[1, 0]
                ),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(
                pd.Index([0, 1, 2]),
                pd.Index(["a", "b"], name="c"),
                2,
                some_arg=2,
                check_expected_keys_=False,
            ),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]),
                pd.Index(["a", "b"], name="c"),
                2,
                some_arg=2,
                check_expected_keys_=False,
            ),
        )
        assert wrapper.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(
                    pd.Index([0, 1, 2]),
                    pd.Index(["a", "b"], name="c"),
                    2,
                    some_arg=2,
                    check_expected_keys_=False,
                ),
                vbt.ArrayWrapper(
                    pd.Index([4, 5, 6]),
                    pd.Index(["a", "b"], name="c"),
                    2,
                    some_arg=3,
                    check_expected_keys_=False,
                ),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(
                    pd.Index([0, 1, 2]),
                    pd.Index(["a", "b"], name="c"),
                    2,
                    some_arg=2,
                    check_expected_keys_=False,
                ),
                vbt.ArrayWrapper(pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(
                    pd.Index([4, 5, 6]),
                    pd.Index(["a", "b"], name="c"),
                    2,
                    some_arg=2,
                    check_expected_keys_=False,
                ),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(
                    pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, column_only_select=True
                ),
                vbt.ArrayWrapper(
                    pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, column_only_select=False
                ),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.row_stack(
                vbt.ArrayWrapper(
                    pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, allow_enable=True
                ),
                vbt.ArrayWrapper(
                    pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, allow_enable=False
                ),
            )
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(
                pd.Index([0, 1, 2]), pd.Index(["a", "b"], name="c"), 2, allow_enable=True
            ),
            vbt.ArrayWrapper(
                pd.Index([4, 5, 6]), pd.Index(["a", "b"], name="c"), 2, allow_enable=False
            ),
            allow_enable=False,
        )
        assert not wrapper.grouper.allow_enable
        columns = pd.Index(["a", "b"], name="c")
        grouper = vbt.Grouper(columns, group_by=True)
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), columns, 2, grouper=grouper),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), columns, 2, grouper=grouper),
        )
        assert wrapper.grouper == grouper
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), columns, 2, group_by=True, allow_enable=True),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), columns, 2, group_by=False, allow_enable=False),
            grouper=grouper,
        )
        assert wrapper.grouper == grouper
        wrapper = vbt.ArrayWrapper.row_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2]), columns, 2),
            vbt.ArrayWrapper(pd.Index([4, 5, 6]), columns, 2),
            grouper=grouper,
            group_by=False,
        )
        assert not wrapper.grouper.is_grouped()

    def test_column_stack(self):
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2], name="i"))
        wrapper = vbt.ArrayWrapper(
            pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2
        ).column_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["c", "d"], name="c"), 2)
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2], name="i"))
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([1, 2, 3], name="i"), pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 3], name="i"))
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(pd.Index([2, 1, 0], name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2], name="i"))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(pd.Index([0, 0, 1], name="i"), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["c", "d"], name="c"), 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(pd.Index([2, 1, 0], name="i"), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([2, 1, 0], name="i"), pd.Index(["c", "d"], name="c"), 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(
                    pd.Index(["x", "y", "z"], name="i"), pd.Index(["c", "d"], name="c"), 2
                ),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(pd.Index([1, 2, 3], name="i"), pd.Index(["c", "d"], name="c"), 2),
                union_index=False,
            )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(pd.Index([0, 1, 2], name="i"), pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(pd.Index([5, 6, 7], name="i"), pd.Index(["c", "d"], name="c"), 2),
            index=[0, 1, 2, 3, 4, 5],
        )
        assert_index_equal(wrapper.index, pd.Index([0, 1, 2, 3, 4, 5]))
        index1 = pd.date_range("2020-01-01", "2020-01-05", inclusive="left")
        index2 = pd.date_range("2020-01-05", "2020-01-10", inclusive="left")
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq == pd.Timedelta(days=1)
        index1 = pd.date_range("2020-01-01", "2020-01-05", inclusive="left")
        index2 = pd.date_range("2020-01-05", "2020-01-06", inclusive="left")
        index3 = pd.date_range("2020-01-07", "2020-01-10", inclusive="left")
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2.append(index3), pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2).append(index3))
        assert wrapper.freq == pd.Timedelta(days=1)
        with pytest.raises(Exception):
            index1 = pd.date_range("2020-01-01", "2020-01-05", freq="1d", inclusive="left")
            index2 = pd.date_range("2020-01-05", "2020-01-10", freq="1h", inclusive="left")
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
                vbt.ArrayWrapper(index2, pd.Index(["c", "d"], name="c"), 2),
            )
        index1 = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
        index2 = pd.DatetimeIndex(["2020-01-06", "2020-01-10"])
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq is None
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index1, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index2, pd.Index(["c", "d"], name="c"), 2),
            freq="3d",
        )
        assert_index_equal(wrapper.index, index1.append(index2))
        assert wrapper.freq == pd.Timedelta(days=3)
        index = pd.Index([0, 1, 2], name="i")
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c"), 2),
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c", "d"], name="c"))
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c1"), 2),
                vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c2"), 2),
                normalize_columns=False,
            )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c2"), 2),
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c", "d"]))
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c"), 2),
            keys=["o1", "o2"],
        )
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [("o1", "a"), ("o1", "b"), ("o2", "c"), ("o2", "d")], names=[None, "c"]
            ),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c2"), 2),
            keys=pd.Index(["k1", "k2"], name="k"),
        )
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [("k1", "a"), ("k1", "b"), ("k2", "c"), ("k2", "d")], names=["k", None]
            ),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, pd.Index(["a", "b"], name="c1"), 2),
            vbt.ArrayWrapper(index, pd.Index(["c", "d"], name="c2"), 2),
            columns=["a2", "b2", "c2", "d2"],
        )
        assert_index_equal(wrapper.columns, pd.Index(["a2", "b2", "c2", "d2"]))
        columns1 = pd.Index(["a", "b"], name="c")
        columns2 = pd.Index(["c", "d"], name="c")
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=pd.Index([2, 3], name="g")),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1, 2, 3], name="g", dtype="int64"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=False),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1, "c", "d"], dtype="object"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=pd.Index([0, 1], name="g")),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1, 2, 3], name="group_idx", dtype="int64"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g1")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=pd.Index([2, 3], name="g2")),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 1, 2, 3], dtype="int64"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=True),
            vbt.ArrayWrapper(index, columns2, 2, group_by=True),
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.Index([0, 0, 1, 1], name="group_idx", dtype="int64"),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=True),
            vbt.ArrayWrapper(index, columns2, 2, group_by=True),
            keys=["o1", "o2"],
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.MultiIndex.from_tuples(
                [("o1", "group"), ("o1", "group"), ("o2", "group"), ("o2", "group")],
                names=(None, "group"),
            ),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=True),
            vbt.ArrayWrapper(index, columns2, 2, group_by=False),
            keys=["o1", "o2"],
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.MultiIndex.from_tuples(
                [("o1", "group"), ("o1", "group"), ("o2", "c"), ("o2", "d")], names=(None, None)
            ),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=True),
            vbt.ArrayWrapper(index, columns2, 2, group_by=False),
            keys=["o1", "o2"],
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.MultiIndex.from_tuples(
                [("o1", "group"), ("o1", "group"), ("o2", "c"), ("o2", "d")], names=(None, None)
            ),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_by=pd.Index([0, 1], name="g")),
            vbt.ArrayWrapper(index, columns2, 2, group_by=pd.Index([2, 3], name="g")),
            keys=["o1", "o2"],
        )
        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
        assert_index_equal(
            wrapper_grouped_index[wrapper_groups],
            pd.MultiIndex.from_tuples(
                [("o1", 0), ("o1", 1), ("o2", 2), ("o2", 3)], names=(None, "g")
            ),
        )
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, column_only_select=False),
            vbt.ArrayWrapper(index, columns2, 2, column_only_select=False),
        )
        assert not wrapper.column_only_select
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, column_only_select=True),
            vbt.ArrayWrapper(index, columns2, 2, column_only_select=False),
        )
        assert wrapper.column_only_select
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_select=True),
            vbt.ArrayWrapper(index, columns2, 2, group_select=True),
        )
        assert wrapper.group_select
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, group_select=True),
            vbt.ArrayWrapper(index, columns2, 2, group_select=False),
        )
        assert not wrapper.group_select
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_enable=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_enable=True),
        )
        assert wrapper.grouper.allow_enable
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_enable=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_enable=False),
        )
        assert not wrapper.grouper.allow_enable
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_disable=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_disable=True),
        )
        assert wrapper.grouper.allow_disable
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_disable=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_disable=False),
        )
        assert not wrapper.grouper.allow_disable
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_modify=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_modify=True),
        )
        assert wrapper.grouper.allow_modify
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_modify=True),
            vbt.ArrayWrapper(index, columns2, 2, allow_modify=False),
        )
        assert not wrapper.grouper.allow_modify
        columns = pd.Index(["a", "b", "c", "d"], name="c2")
        grouper = vbt.Grouper(columns, group_by=True)
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, columns1, 2),
                vbt.ArrayWrapper(index, columns2, 2),
                grouper=grouper,
            )
        columns = pd.Index(["a", "b", "c", "d"], name="c")
        grouper = vbt.Grouper(columns, group_by=True)
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, allow_modify=False),
            vbt.ArrayWrapper(index, columns2, 2, allow_modify=False),
            grouper=grouper,
        )
        assert wrapper.grouper == grouper
        assert wrapper.grouper.allow_modify
        wrapper = vbt.ArrayWrapper.column_stack(
            vbt.ArrayWrapper(index, columns1, 2, some_arg=2, check_expected_keys_=False),
            vbt.ArrayWrapper(index, columns2, 2, some_arg=2, check_expected_keys_=False),
        )
        assert wrapper.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, columns1, 2, some_arg=2, check_expected_keys_=False),
                vbt.ArrayWrapper(index, columns2, 2, some_arg=3, check_expected_keys_=False),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, columns1, 2, some_arg=2, check_expected_keys_=False),
                vbt.ArrayWrapper(index, columns2, 2),
            )
        with pytest.raises(Exception):
            vbt.ArrayWrapper.column_stack(
                vbt.ArrayWrapper(index, columns1, 2),
                vbt.ArrayWrapper(index, columns2, 2, some_arg=2, check_expected_keys_=False),
            )

    def test_config(self, tmp_path):
        assert vbt.ArrayWrapper.loads(sr2_wrapper.dumps()) == sr2_wrapper
        assert vbt.ArrayWrapper.loads(sr2_wrapper_co.dumps()) == sr2_wrapper_co
        assert vbt.ArrayWrapper.loads(sr2_grouped_wrapper.dumps()) == sr2_grouped_wrapper
        assert vbt.ArrayWrapper.loads(sr2_grouped_wrapper_co.dumps()) == sr2_grouped_wrapper_co
        sr2_grouped_wrapper_co.save(tmp_path / "sr2_grouped_wrapper_co")
        assert vbt.ArrayWrapper.load(tmp_path / "sr2_grouped_wrapper_co") == sr2_grouped_wrapper_co
        sr2_grouped_wrapper_co.save(tmp_path / "sr2_grouped_wrapper_co", file_format="ini")
        assert (
            vbt.ArrayWrapper.load(tmp_path / "sr2_grouped_wrapper_co", file_format="ini")
            == sr2_grouped_wrapper_co
        )

    def test_indexing_func_meta(self):
        # not grouped
        wrapper_meta = sr2_wrapper.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[0, :2])
        assert wrapper_meta["row_idxs"] == slice(0, 1, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == slice(0, 2, None)
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[:2, 0])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[:2, [0]])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(0, 1, None)
        assert wrapper_meta["group_idxs"] == slice(0, 1, None)
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[:2, :2])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == slice(0, 2, None)
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[[0, 2], [0, 2]])
        np.testing.assert_array_equal(wrapper_meta["row_idxs"], np.array([0, 2]))
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 2]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 2]))
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[[0, 0], [0, 0]])
        np.testing.assert_array_equal(wrapper_meta["row_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        wrapper_meta = df4_wrapper.indexing_func_meta(lambda x: x.iloc[0, 0])
        assert wrapper_meta["row_idxs"] == slice(0, 1, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0

        # not grouped, column only
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[0])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 1, None)
        assert wrapper_meta["group_idxs"] == slice(0, 1, None)
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == slice(0, 2, None)
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0, 2]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 2]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 2]))
        wrapper_meta = df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0, 0]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        with pytest.raises(Exception):
            sr2_wrapper_co.indexing_func_meta(lambda x: x.iloc[:2])
        with pytest.raises(Exception):
            df4_wrapper_co.indexing_func_meta(lambda x: x.iloc[:, :2])

        # grouped
        wrapper_meta = sr2_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2, 0])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2, 1])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == 2
        assert wrapper_meta["group_idxs"] == 1
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2, [1]])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(2, 3, None)
        assert wrapper_meta["group_idxs"] == slice(1, 2, None)
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[:2, :2])
        assert wrapper_meta["row_idxs"] == slice(0, 2, None)
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[[0, 2], [0, 1]])
        np.testing.assert_array_equal(wrapper_meta["row_idxs"], np.array([0, 2]))
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[[0, 0], [0, 0]])
        np.testing.assert_array_equal(wrapper_meta["row_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        wrapper_meta = df4_grouped_wrapper.indexing_func_meta(lambda x: x.iloc[0, :2])
        assert wrapper_meta["row_idxs"] == slice(0, 1, None)
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)

        # grouped, column only
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[0])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[1])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == 2
        assert wrapper_meta["group_idxs"] == 1
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[[1]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(2, 3, None)
        assert wrapper_meta["group_idxs"] == slice(1, 2, None)
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0, 1]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(None, None, None)
        assert wrapper_meta["group_idxs"] == slice(None, None, None)
        wrapper_meta = df4_grouped_wrapper_co.indexing_func_meta(lambda x: x.iloc[[0, 0]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))

        # grouped, column only, no group select
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[0])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == 0
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[1])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == 1
        assert wrapper_meta["group_idxs"] == 0
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[[1]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(1, 2, None)
        assert wrapper_meta["group_idxs"] == slice(0, 1, None)
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[:2])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[[0, 1]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        assert wrapper_meta["col_idxs"] == slice(0, 2, None)
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))
        wrapper_meta = df4_grouped_wrapper_conog.indexing_func_meta(lambda x: x.iloc[[0, 0]])
        assert wrapper_meta["row_idxs"] == slice(None, None, None)
        np.testing.assert_array_equal(wrapper_meta["col_idxs"], np.array([0, 0]))
        np.testing.assert_array_equal(wrapper_meta["group_idxs"], np.array([0, 0]))

    def test_indexing(self):
        # not grouped
        assert_index_equal(
            sr2_wrapper.iloc[:2].index, pd.Index(["x2", "y2"], dtype="object", name="i2")
        )
        assert_index_equal(sr2_wrapper.iloc[:2].columns, pd.Index(["a2"], dtype="object"))
        assert sr2_wrapper.iloc[:2].ndim == 1
        assert_index_equal(
            df4_wrapper.iloc[0, :2].index, pd.Index(["x6"], dtype="object", name="i6")
        )
        assert_index_equal(
            df4_wrapper.iloc[0, :2].columns, pd.Index(["a6", "b6"], dtype="object", name="c6")
        )
        assert df4_wrapper.iloc[0, :2].ndim == 2
        assert_index_equal(
            df4_wrapper.iloc[:2, 0].index, pd.Index(["x6", "y6"], dtype="object", name="i6")
        )
        assert_index_equal(
            df4_wrapper.iloc[:2, 0].columns, pd.Index(["a6"], dtype="object", name="c6")
        )
        assert df4_wrapper.iloc[:2, 0].ndim == 1
        assert_index_equal(
            df4_wrapper.iloc[:2, [0]].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_wrapper.iloc[:2, [0]].columns, pd.Index(["a6"], dtype="object", name="c6")
        )
        assert df4_wrapper.iloc[:2, [0]].ndim == 2
        assert_index_equal(
            df4_wrapper.iloc[:2, :2].index, pd.Index(["x6", "y6"], dtype="object", name="i6")
        )
        assert_index_equal(
            df4_wrapper.iloc[:2, :2].columns,
            pd.Index(["a6", "b6"], dtype="object", name="c6"),
        )
        assert df4_wrapper.iloc[:2, :2].ndim == 2

        # not grouped, column only
        assert_index_equal(
            df4_wrapper_co.iloc[0].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_wrapper_co.iloc[0].columns, pd.Index(["a6"], dtype="object", name="c6")
        )
        assert df4_wrapper_co.iloc[0].ndim == 1
        assert_index_equal(
            df4_wrapper_co.iloc[[0]].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_wrapper_co.iloc[[0]].columns, pd.Index(["a6"], dtype="object", name="c6")
        )
        assert df4_wrapper_co.iloc[[0]].ndim == 2
        assert_index_equal(
            df4_wrapper_co.iloc[:2].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_wrapper_co.iloc[:2].columns,
            pd.Index(["a6", "b6"], dtype="object", name="c6"),
        )
        assert df4_wrapper_co.iloc[:2].ndim == 2

        # grouped
        assert_index_equal(
            sr2_grouped_wrapper.iloc[:2].index,
            pd.Index(["x2", "y2"], dtype="object", name="i2"),
        )
        assert_index_equal(sr2_grouped_wrapper.iloc[:2].columns, pd.Index(["a2"], dtype="object"))
        assert sr2_grouped_wrapper.iloc[:2].ndim == 1
        assert sr2_grouped_wrapper.iloc[:2].grouped_ndim == 1
        assert_index_equal(
            sr2_grouped_wrapper.iloc[:2].grouper.group_by,
            pd.Index(["g1"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 0].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 0].columns,
            pd.Index(["a6", "b6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper.iloc[:2, 0].ndim == 2
        assert df4_grouped_wrapper.iloc[:2, 0].grouped_ndim == 1
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 0].grouper.group_by,
            pd.Index(["g1", "g1"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 1].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 1].columns,
            pd.Index(["c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper.iloc[:2, 1].ndim == 1
        assert df4_grouped_wrapper.iloc[:2, 1].grouped_ndim == 1
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, 1].grouper.group_by,
            pd.Index(["g2"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, [1]].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, [1]].columns,
            pd.Index(["c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper.iloc[:2, [1]].ndim == 2
        assert df4_grouped_wrapper.iloc[:2, [1]].grouped_ndim == 2
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, [1]].grouper.group_by,
            pd.Index(["g2"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, :2].index,
            pd.Index(["x6", "y6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, :2].columns,
            pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper.iloc[:2, :2].ndim == 2
        assert df4_grouped_wrapper.iloc[:2, :2].grouped_ndim == 2
        assert_index_equal(
            df4_grouped_wrapper.iloc[:2, :2].grouper.group_by,
            pd.Index(["g1", "g1", "g2"], dtype="object", name="group"),
        )

        # grouped, column only
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[0].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[0].columns,
            pd.Index(["a6", "b6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper_co.iloc[0].ndim == 2
        assert df4_grouped_wrapper_co.iloc[0].grouped_ndim == 1
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[0].grouper.group_by,
            pd.Index(["g1", "g1"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[1].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[1].columns,
            pd.Index(["c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper_co.iloc[1].ndim == 1
        assert df4_grouped_wrapper_co.iloc[1].grouped_ndim == 1
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[1].grouper.group_by,
            pd.Index(["g2"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[[1]].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[[1]].columns,
            pd.Index(["c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper_co.iloc[[1]].ndim == 2
        assert df4_grouped_wrapper_co.iloc[[1]].grouped_ndim == 2
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[[1]].grouper.group_by,
            pd.Index(["g2"], dtype="object", name="group"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[:2].index,
            pd.Index(["x6", "y6", "z6"], dtype="object", name="i6"),
        )
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[:2].columns,
            pd.Index(["a6", "b6", "c6"], dtype="object", name="c6"),
        )
        assert df4_grouped_wrapper_co.iloc[:2].ndim == 2
        assert df4_grouped_wrapper_co.iloc[:2].grouped_ndim == 2
        assert_index_equal(
            df4_grouped_wrapper_co.iloc[:2].grouper.group_by,
            pd.Index(["g1", "g1", "g2"], dtype="object", name="group"),
        )

    def test_from_obj(self):
        assert vbt.ArrayWrapper.from_obj(sr2) == sr2_wrapper
        assert vbt.ArrayWrapper.from_obj(df4) == df4_wrapper
        assert vbt.ArrayWrapper.from_obj(sr2, column_only_select=True) == sr2_wrapper_co
        assert vbt.ArrayWrapper.from_obj(df4, column_only_select=True) == df4_wrapper_co

    def test_from_shape(self):
        assert vbt.ArrayWrapper.from_shape((3,)) == vbt.ArrayWrapper(
            pd.RangeIndex(start=0, stop=3, step=1),
            pd.RangeIndex(start=0, stop=1, step=1),
            1,
        )
        assert vbt.ArrayWrapper.from_shape((3, 3)) == vbt.ArrayWrapper.from_obj(
            pd.DataFrame(np.empty((3, 3))),
        )

    def test_columns(self):
        assert_index_equal(df4_wrapper.columns, df4.columns)
        assert_index_equal(df4_grouped_wrapper.columns, df4.columns)
        assert_index_equal(
            df4_grouped_wrapper.get_columns(), pd.Index(["g1", "g2"], dtype="object", name="group")
        )

    def test_name(self):
        assert sr2_wrapper.name == "a2"
        assert df4_wrapper.name is None
        assert vbt.ArrayWrapper.from_obj(pd.Series([0])).name is None
        assert sr2_grouped_wrapper.name == "a2"
        assert sr2_grouped_wrapper.get_name() == "g1"
        assert df4_grouped_wrapper.name is None
        assert df4_grouped_wrapper.get_name() is None

    def test_ndim(self):
        assert sr2_wrapper.ndim == 1
        assert df4_wrapper.ndim == 2
        assert sr2_grouped_wrapper.ndim == 1
        assert sr2_grouped_wrapper.get_ndim() == 1
        assert df4_grouped_wrapper.ndim == 2
        assert df4_grouped_wrapper.get_ndim() == 2
        assert df4_grouped_wrapper["g1"].ndim == 2
        assert df4_grouped_wrapper["g1"].get_ndim() == 1
        assert df4_grouped_wrapper["g2"].ndim == 1
        assert df4_grouped_wrapper["g2"].get_ndim() == 1

    def test_shape(self):
        assert sr2_wrapper.shape == (3,)
        assert df4_wrapper.shape == (3, 3)
        assert sr2_grouped_wrapper.shape == (3,)
        assert sr2_grouped_wrapper.get_shape() == (3,)
        assert df4_grouped_wrapper.shape == (3, 3)
        assert df4_grouped_wrapper.get_shape() == (3, 2)

    def test_shape_2d(self):
        assert sr2_wrapper.shape_2d == (3, 1)
        assert df4_wrapper.shape_2d == (3, 3)
        assert sr2_grouped_wrapper.shape_2d == (3, 1)
        assert sr2_grouped_wrapper.get_shape_2d() == (3, 1)
        assert df4_grouped_wrapper.shape_2d == (3, 3)
        assert df4_grouped_wrapper.get_shape_2d() == (3, 2)

    def test_freq(self):
        assert sr2_wrapper.freq is None
        assert sr2_wrapper.replace(freq="1D").freq == day_dt
        assert sr2_wrapper.replace(index=pd.date_range("2020", periods=3)).freq == day_dt

    def test_periods(self):
        test_sr = pd.Series([1, 2], index=pd.date_range("2020", periods=2))
        assert test_sr.vbt.wrapper.periods == 2

    def test_dt_periods(self):
        assert sr2_wrapper.dt_periods == 3
        assert sr2_wrapper.replace(freq="1D").dt_periods == 3
        test_sr = pd.Series([1, 2], index=["2020-01-01", "2021-01-01"])
        assert test_sr.vbt.wrapper.dt_periods == 2
        assert test_sr.vbt(freq="1D").wrapper.dt_periods == 367

    def test_to_timedelta(self):
        sr = pd.Series([1, 2, np.nan], index=["x", "y", "z"], name="name")
        assert_series_equal(
            vbt.ArrayWrapper.from_obj(sr, freq="1 days").arr_to_timedelta(sr),
            pd.Series(
                np.array([86400000000000, 172800000000000, "NaT"], dtype="timedelta64[ns]"),
                index=sr.index,
                name=sr.name,
            ),
        )
        df = sr.to_frame()
        assert_frame_equal(
            vbt.ArrayWrapper.from_obj(df, freq="1 days").arr_to_timedelta(df),
            pd.DataFrame(
                np.array([86400000000000, 172800000000000, "NaT"], dtype="timedelta64[ns]"),
                index=df.index,
                columns=df.columns,
            ),
        )

    def test_wrap(self):
        assert_series_equal(
            vbt.ArrayWrapper(index=sr1.index, columns=[0], ndim=1).wrap(a1),  # empty
            pd.Series(a1, index=sr1.index, name=None),
        )
        assert_series_equal(
            vbt.ArrayWrapper(index=sr1.index, columns=[sr1.name], ndim=1).wrap(a1),
            pd.Series(a1, index=sr1.index, name=sr1.name),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=sr1.index, columns=[sr1.name], ndim=2).wrap(a1),
            pd.DataFrame(a1, index=sr1.index, columns=[sr1.name]),
        )
        assert_series_equal(
            vbt.ArrayWrapper(index=sr2.index, columns=[sr2.name], ndim=1).wrap(a2),
            pd.Series(a2, index=sr2.index, name=sr2.name),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=sr2.index, columns=[sr2.name], ndim=2).wrap(a2),
            pd.DataFrame(a2, index=sr2.index, columns=[sr2.name]),
        )
        assert_series_equal(
            vbt.ArrayWrapper(index=df2.index, columns=df2.columns, ndim=1).wrap(a2),
            pd.Series(a2, index=df2.index, name=df2.columns[0]),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=df2.index, columns=df2.columns, ndim=2).wrap(a2),
            pd.DataFrame(a2, index=df2.index, columns=df2.columns),
        )
        assert_frame_equal(
            vbt.ArrayWrapper.from_obj(df2).wrap(a2, index=df4.index),
            pd.DataFrame(a2, index=df4.index, columns=df2.columns),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=df4.index, columns=df4.columns, ndim=2).wrap(
                np.array([[0, 0, np.nan], [1, np.nan, 1], [2, 2, np.nan]]),
                fillna=-1,
            ),
            pd.DataFrame(
                [[0.0, 0.0, -1.0], [1.0, -1.0, 1.0], [2.0, 2.0, -1.0]],
                index=df4.index,
                columns=df4.columns,
            ),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=df4.index, columns=df4.columns, ndim=2).wrap(
                np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                to_index=True,
            ),
            pd.DataFrame(
                [["x6", "x6", "x6"], ["y6", "y6", "y6"], ["z6", "z6", "z6"]],
                index=df4.index,
                columns=df4.columns,
            ),
        )
        assert_frame_equal(
            vbt.ArrayWrapper(index=df4.index, columns=df4.columns, ndim=2, freq="d").wrap(
                np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                to_timedelta=True,
            ),
            pd.DataFrame(
                [
                    [pd.Timedelta(days=0), pd.Timedelta(days=0), pd.Timedelta(days=0)],
                    [pd.Timedelta(days=1), pd.Timedelta(days=1), pd.Timedelta(days=1)],
                    [pd.Timedelta(days=2), pd.Timedelta(days=2), pd.Timedelta(days=2)],
                ],
                index=df4.index,
                columns=df4.columns,
            ),
        )

    def test_wrap_reduced(self):
        # sr to value
        assert sr2_wrapper.wrap_reduced(0) == 0
        assert sr2_wrapper.wrap_reduced(np.array([0])) == 0  # result of computation on 2d
        # sr to array
        assert_series_equal(
            sr2_wrapper.wrap_reduced(np.array([0, 1])),
            pd.Series(np.array([0, 1]), name=sr2.name),
        )
        assert_series_equal(
            sr2_wrapper.wrap_reduced(np.array([0, 1]), name_or_index=["x", "y"]),
            pd.Series(np.array([0, 1]), index=["x", "y"], name=sr2.name),
        )
        assert_series_equal(
            sr2_wrapper.wrap_reduced(np.array([0, 1]), name_or_index=["x", "y"], columns=[0]),
            pd.Series(np.array([0, 1]), index=["x", "y"], name=None),
        )
        # df to value
        assert df2_wrapper.wrap_reduced(0) == 0
        assert df4_wrapper.wrap_reduced(0) == 0
        # df to value per column
        assert_series_equal(
            df4_wrapper.wrap_reduced(np.array([0, 1, 2]), name_or_index="test"),
            pd.Series(np.array([0, 1, 2]), index=df4.columns, name="test"),
        )
        assert_series_equal(
            df4_wrapper.wrap_reduced(
                np.array([0, 1, 2]), columns=["m", "n", "l"], name_or_index="test"
            ),
            pd.Series(np.array([0, 1, 2]), index=["m", "n", "l"], name="test"),
        )
        # df to array per column
        assert_frame_equal(
            df4_wrapper.wrap_reduced(np.array([[0, 1, 2], [3, 4, 5]]), name_or_index=["x", "y"]),
            pd.DataFrame(np.array([[0, 1, 2], [3, 4, 5]]), index=["x", "y"], columns=df4.columns),
        )
        assert_frame_equal(
            df4_wrapper.wrap_reduced(
                np.array([[0, 1, 2], [3, 4, 5]]),
                name_or_index=["x", "y"],
                columns=["m", "n", "l"],
            ),
            pd.DataFrame(
                np.array([[0, 1, 2], [3, 4, 5]]), index=["x", "y"], columns=["m", "n", "l"]
            ),
        )

    def test_grouped_wrapping(self):
        assert_frame_equal(
            df4_grouped_wrapper_co.wrap(np.array([[1, 2], [3, 4], [5, 6]])),
            pd.DataFrame(
                np.array([[1, 2], [3, 4], [5, 6]]),
                index=df4.index,
                columns=pd.Index(["g1", "g2"], dtype="object", name="group"),
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.wrap_reduced(np.array([1, 2])),
            pd.Series(np.array([1, 2]), index=pd.Index(["g1", "g2"], dtype="object", name="group")),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.wrap(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), group_by=False
            ),
            pd.DataFrame(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index=df4.index, columns=df4.columns
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.wrap_reduced(np.array([1, 2, 3]), group_by=False),
            pd.Series(np.array([1, 2, 3]), index=df4.columns),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[0].wrap(np.array([1, 2, 3])),
            pd.Series(np.array([1, 2, 3]), index=df4.index, name="g1"),
        )
        assert df4_grouped_wrapper_co.iloc[0].wrap_reduced(np.array([1])) == 1
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[0].wrap(np.array([[1], [2], [3]])),
            pd.Series(np.array([1, 2, 3]), index=df4.index, name="g1"),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[0].wrap(np.array([[1, 2], [3, 4], [5, 6]]), group_by=False),
            pd.DataFrame(
                np.array([[1, 2], [3, 4], [5, 6]]), index=df4.index, columns=df4.columns[:2]
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[0].wrap_reduced(np.array([1, 2]), group_by=False),
            pd.Series(np.array([1, 2]), index=df4.columns[:2]),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[[0]].wrap(np.array([1, 2, 3])),
            pd.DataFrame(
                np.array([[1], [2], [3]]),
                index=df4.index,
                columns=pd.Index(["g1"], dtype="object", name="group"),
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[[0]].wrap_reduced(np.array([1])),
            pd.Series(np.array([1]), index=pd.Index(["g1"], dtype="object", name="group")),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[[0]].wrap(
                np.array([[1, 2], [3, 4], [5, 6]]), group_by=False
            ),
            pd.DataFrame(
                np.array([[1, 2], [3, 4], [5, 6]]), index=df4.index, columns=df4.columns[:2]
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[[0]].wrap_reduced(np.array([1, 2]), group_by=False),
            pd.Series(np.array([1, 2]), index=df4.columns[:2]),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[1].wrap(np.array([1, 2, 3])),
            pd.Series(np.array([1, 2, 3]), index=df4.index, name="g2"),
        )
        assert df4_grouped_wrapper_co.iloc[1].wrap_reduced(np.array([1])) == 1
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[1].wrap(np.array([1, 2, 3]), group_by=False),
            pd.Series(np.array([1, 2, 3]), index=df4.index, name=df4.columns[2]),
        )
        assert df4_grouped_wrapper_co.iloc[1].wrap_reduced(np.array([1]), group_by=False) == 1
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[[1]].wrap(np.array([1, 2, 3])),
            pd.DataFrame(
                np.array([[1], [2], [3]]),
                index=df4.index,
                columns=pd.Index(["g2"], dtype="object", name="group"),
            ),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[[1]].wrap_reduced(np.array([1])),
            pd.Series(np.array([1]), index=pd.Index(["g2"], dtype="object", name="group")),
        )
        assert_frame_equal(
            df4_grouped_wrapper_co.iloc[[1]].wrap(np.array([1, 2, 3]), group_by=False),
            pd.DataFrame(np.array([[1], [2], [3]]), index=df4.index, columns=df4.columns[2:]),
        )
        assert_series_equal(
            df4_grouped_wrapper_co.iloc[[1]].wrap_reduced(np.array([1]), group_by=False),
            pd.Series(np.array([1]), index=df4.columns[2:]),
        )

    def test_dummy(self):
        assert_index_equal(sr2_wrapper.dummy().index, sr2_wrapper.index)
        assert_index_equal(sr2_wrapper.dummy().to_frame().columns, sr2_wrapper.columns)
        assert_index_equal(df4_wrapper.dummy().index, df4_wrapper.index)
        assert_index_equal(df4_wrapper.dummy().columns, df4_wrapper.columns)
        assert_index_equal(sr2_grouped_wrapper.dummy().index, sr2_grouped_wrapper.index)
        assert_index_equal(
            sr2_grouped_wrapper.dummy().to_frame().columns.rename("group"),
            sr2_grouped_wrapper.get_columns(),
        )
        assert_index_equal(df4_grouped_wrapper.dummy().index, df4_grouped_wrapper.index)
        assert_index_equal(df4_grouped_wrapper.dummy().columns, df4_grouped_wrapper.get_columns())

    def test_fill(self):
        assert_series_equal(sr2_wrapper.fill(0), sr2 * 0)
        assert_frame_equal(df4_wrapper.fill(0), df4 * 0)
        assert_series_equal(
            sr2_grouped_wrapper.fill(0),
            pd.Series(0, index=sr2.index, name="g1"),
        )
        assert_frame_equal(
            df4_grouped_wrapper.fill(0),
            pd.DataFrame(0, index=df4.index, columns=pd.Index(["g1", "g2"], name="group")),
        )

    def test_fill_reduced(self):
        assert sr2_wrapper.fill_reduced(0) == 0
        assert_series_equal(df4_wrapper.fill_reduced(0), pd.Series(0, index=df4.columns))
        assert sr2_grouped_wrapper.fill_reduced(0) == 0
        assert_series_equal(
            df4_grouped_wrapper.fill_reduced(0),
            pd.Series(0, index=pd.Index(["g1", "g2"], name="group")),
        )

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        ts = pd.Series(np.arange(5), index=pd.date_range("2020-01-01", "2020-01-05"))
        assert_index_equal(
            ts.vbt.wrapper.resample(test_freq).index,
            ts.resample(test_freq).last().index,
        )
        assert (
            ts.vbt.wrapper.resample(test_freq).freq
            == ts.resample(test_freq).last().vbt.wrapper.freq
        )

    def test_fill_and_set(self):
        i = pd.date_range("2020-01-01", "2020-01-05")
        c = pd.Index(["a", "b", "c"], name="c")
        sr = pd.Series(np.nan, index=i, name=c[0])
        sr_wrapper = wrapping.ArrayWrapper.from_obj(sr)
        df = pd.DataFrame(np.nan, index=i, columns=c)
        df_wrapper = wrapping.ArrayWrapper.from_obj(df)

        obj = sr_wrapper.fill_and_set(
            indexing.index_dict(
                {
                    vbt.rowidx(0): 100,
                    "_def": 0,
                }
            )
        )
        assert_series_equal(
            obj,
            pd.Series([100, 0, 0, 0, 0], index=i, name=c[0]),
        )
        obj = df_wrapper.fill_and_set(
            indexing.index_dict(
                {
                    vbt.idx(1, 1): 100,
                    "_def": 1,
                }
            )
        )
        assert_frame_equal(
            obj,
            pd.DataFrame(
                [
                    [1, 1, 1],
                    [1, 100, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                index=i,
                columns=c,
            ),
        )

        def _sr_assert_flex_index_dct(index_dct, target_arr):
            arr = sr_wrapper.fill_and_set(indexing.index_dict(index_dct), keep_flex=True)
            np.testing.assert_array_equal(arr, target_arr)

        _sr_assert_flex_index_dct({indexing.hslice(None, None, None): 0}, np.array([0.0]))
        _sr_assert_flex_index_dct({0: 0}, np.array([0.0, np.nan, np.nan, np.nan, np.nan]))
        _sr_assert_flex_index_dct({(0, 2): 0}, np.array([0.0, np.nan, 0.0, np.nan, np.nan]))
        _sr_assert_flex_index_dct({(0, 2): [0, 1]}, np.array([0.0, np.nan, 1.0, np.nan, np.nan]))
        _sr_assert_flex_index_dct(
            {(0, 2): vbt.RepEval("np.arange(len(row_idxs))")},
            np.array([0.0, np.nan, 1.0, np.nan, np.nan]),
        )
        _sr_assert_flex_index_dct(
            {indexing.idx(0, indexing.hslice(None, None, None)): 0},
            np.array([0, np.nan, np.nan, np.nan, np.nan]),
        )

        def _df_assert_flex_index_dct(index_dct, target_arr):
            arr = df_wrapper.fill_and_set(indexing.index_dict(index_dct), keep_flex=True)
            np.testing.assert_array_equal(arr, target_arr)

        _df_assert_flex_index_dct(
            {indexing.rowidx(indexing.hslice(None, None, None)): 0}, np.array([0.0])[:, None]
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx(0): 0}, np.array([0.0, np.nan, np.nan, np.nan, np.nan])[:, None]
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): 0}, np.array([0.0, np.nan, 0.0, np.nan, np.nan])[:, None]
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): [0, 1]}, np.array([0.0, np.nan, 1.0, np.nan, np.nan])[:, None]
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): vbt.RepEval("np.arange(len(row_idxs))")},
            np.array([0.0, np.nan, 1.0, np.nan, np.nan])[:, None],
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx(0): [0, 1, 2]},
            np.array(
                [
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): [[0, 1, 2]]},
            np.array(
                [
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): [[0, 1, 2], [0, 1, 2]]},
            np.array(
                [
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [0, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.rowidx((0, 2)): 0}, np.array([0.0, np.nan, 0.0, np.nan, np.nan])[:, None]
        )
        _df_assert_flex_index_dct(
            {indexing.colidx(indexing.hslice(None, None, None)): 0}, np.array([0.0])[None]
        )
        _df_assert_flex_index_dct({indexing.colidx(0): 0}, np.array([0.0, np.nan, np.nan])[None])
        _df_assert_flex_index_dct({indexing.colidx((0, 2)): 0}, np.array([0.0, np.nan, 0.0])[None])
        _df_assert_flex_index_dct(
            {indexing.colidx((0, 2)): [0, 1]}, np.array([0.0, np.nan, 1.0])[None]
        )
        _df_assert_flex_index_dct(
            {indexing.colidx((0, 2)): vbt.RepEval("np.arange(len(col_idxs))")},
            np.array([0.0, np.nan, 1.0])[None],
        )
        _df_assert_flex_index_dct(
            {indexing.colidx(0): [0, 1, 2, 3, 4]},
            np.array(
                [
                    [0, np.nan, np.nan],
                    [1, np.nan, np.nan],
                    [2, np.nan, np.nan],
                    [3, np.nan, np.nan],
                    [4, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.colidx((0, 2)): [[0], [1], [2], [3], [4]]},
            np.array(
                [
                    [0, np.nan, 0],
                    [1, np.nan, 1],
                    [2, np.nan, 2],
                    [3, np.nan, 3],
                    [4, np.nan, 4],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.colidx((0, 2)): [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]},
            np.array(
                [
                    [0, np.nan, 0],
                    [1, np.nan, 1],
                    [2, np.nan, 2],
                    [3, np.nan, 3],
                    [4, np.nan, 4],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx(indexing.hslice(None, None, None), indexing.hslice(None, None, None)): 0},
            np.array([0.0])[None],
        )
        _df_assert_flex_index_dct(
            {indexing.idx(0, 0): 0},
            np.array(
                [
                    [0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), 0): 0},
            np.array(
                [
                    [0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx(0, (0, 2)): 0},
            np.array(
                [
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): 0},
            np.array(
                [
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): [0, 1]},
            np.array(
                [
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [1, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): [[0], [1]]},
            np.array(
                [
                    [0, np.nan, 0],
                    [np.nan, np.nan, np.nan],
                    [1, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): [[0, 1]]},
            np.array(
                [
                    [0, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [0, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )
        _df_assert_flex_index_dct(
            {indexing.idx((0, 2), (0, 2)): [[0, 1], [2, 3]]},
            np.array(
                [
                    [0, np.nan, 1],
                    [np.nan, np.nan, np.nan],
                    [2, np.nan, 3],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )


sr2_wrapping = wrapping.Wrapping(sr2_wrapper)
df4_wrapping = wrapping.Wrapping(df4_wrapper)

sr2_grouped_wrapping = wrapping.Wrapping(sr2_grouped_wrapper)
df4_grouped_wrapping = wrapping.Wrapping(df4_grouped_wrapper)


class TestWrapping:
    def test_regroup(self):
        assert df4_wrapping.regroup(None) == df4_wrapping
        assert df4_wrapping.regroup(False) == df4_wrapping
        assert df4_grouped_wrapping.regroup(None) == df4_grouped_wrapping
        assert (
            df4_grouped_wrapping.regroup(df4_grouped_wrapper.grouper.group_by)
            == df4_grouped_wrapping
        )
        assert_index_equal(
            df4_wrapping.regroup(df4_grouped_wrapper.grouper.group_by).wrapper.grouper.group_by,
            df4_grouped_wrapper.grouper.group_by,
        )
        assert df4_grouped_wrapping.regroup(False).wrapper.grouper.group_by is None

    def test_select_col(self):
        assert sr2_wrapping.select_col() == sr2_wrapping
        assert sr2_grouped_wrapping.select_col() == sr2_grouped_wrapping
        assert_index_equal(
            df4_wrapping.select_col(column="a6").wrapper.get_columns(),
            pd.Index(["a6"], dtype="object", name="c6"),
        )
        assert_index_equal(
            df4_grouped_wrapping.select_col(column="g1").wrapper.get_columns(),
            pd.Index(["g1"], dtype="object", name="group"),
        )
        with pytest.raises(Exception):
            df4_wrapping.select_col()
        with pytest.raises(Exception):
            df4_grouped_wrapping.select_col()
