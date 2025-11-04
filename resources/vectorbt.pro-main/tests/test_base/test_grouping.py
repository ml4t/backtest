import os

import pytest
import vectorbtpro as vbt
from vectorbtpro._dtypes import *
from vectorbtpro.base import grouping

from tests.utils import *

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


# ############# grouping ############# #


grouped_index = pd.MultiIndex.from_arrays(
    [[1, 1, 1, 1, 0, 0, 0, 0], [3, 3, 2, 2, 1, 1, 0, 0], [7, 6, 5, 4, 3, 2, 1, 0]],
    names=["first", "second", "third"],
)


class TestGrouper:
    def test_group_by_to_index(self):
        assert not vbt.Grouper.group_by_to_index(grouped_index, group_by=False)
        assert vbt.Grouper.group_by_to_index(grouped_index, group_by=None) is None
        assert_index_equal(
            vbt.Grouper.group_by_to_index(grouped_index, group_by=True),
            pd.Index(["group"] * len(grouped_index), name="group"),
        )
        assert_index_equal(
            vbt.Grouper.group_by_to_index(grouped_index, group_by=0),
            pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
        )
        assert_index_equal(
            vbt.Grouper.group_by_to_index(grouped_index, group_by="first"),
            pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
        )
        assert_index_equal(
            vbt.Grouper.group_by_to_index(grouped_index, group_by=[0, 1]),
            pd.MultiIndex.from_tuples(
                [(1, 3), (1, 3), (1, 2), (1, 2), (0, 1), (0, 1), (0, 0), (0, 0)],
                names=["first", "second"],
            ),
        )
        assert_index_equal(
            vbt.Grouper.group_by_to_index(grouped_index, group_by=["first", "second"]),
            pd.MultiIndex.from_tuples(
                [(1, 3), (1, 3), (1, 2), (1, 2), (0, 1), (0, 1), (0, 0), (0, 0)],
                names=["first", "second"],
            ),
        )
        assert_index_equal(
            vbt.Grouper.group_by_to_index(
                grouped_index, group_by=np.array([3, 2, 1, 1, 1, 0, 0, 0])
            ),
            pd.Index([3, 2, 1, 1, 1, 0, 0, 0], dtype="int64", name="group"),
        )
        assert_index_equal(
            vbt.Grouper.group_by_to_index(
                grouped_index,
                group_by=pd.Index([3, 2, 1, 1, 1, 0, 0, 0], name="fourth"),
            ),
            pd.Index([3, 2, 1, 1, 1, 0, 0, 0], dtype="int64", name="fourth"),
        )

    def test_get_groups_and_index(self):
        a, b = vbt.Grouper.group_by_to_groups_and_index(grouped_index, group_by=None)
        np.testing.assert_array_equal(a, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        assert_index_equal(b, grouped_index)
        a, b = vbt.Grouper.group_by_to_groups_and_index(grouped_index, group_by=0)
        np.testing.assert_array_equal(a, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        assert_index_equal(b, pd.Index([1, 0], dtype="int64", name="first"))
        a, b = vbt.Grouper.group_by_to_groups_and_index(grouped_index, group_by=[0, 1])
        np.testing.assert_array_equal(a, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
        assert_index_equal(
            b,
            pd.MultiIndex.from_tuples([(1, 3), (1, 2), (0, 1), (0, 0)], names=["first", "second"]),
        )

    def test_get_group_lens_nb(self):
        np.testing.assert_array_equal(
            grouping.nb.get_group_lens_nb(np.array([0, 0, 0, 0, 1, 1, 1, 1])),
            np.array([4, 4]),
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_lens_nb(np.array([0, 1])), np.array([1, 1])
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_lens_nb(np.array([0, 0])), np.array([2])
        )
        np.testing.assert_array_equal(grouping.nb.get_group_lens_nb(np.array([0])), np.array([1]))
        np.testing.assert_array_equal(grouping.nb.get_group_lens_nb(np.array([])), np.array([]))
        with pytest.raises(Exception):
            grouping.nb.get_group_lens_nb(np.array([1, 1, 0, 0]))
        with pytest.raises(Exception):
            grouping.nb.get_group_lens_nb(np.array([0, 1, 0, 1]))

    def test_get_group_map_nb(self):
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([0, 1, 0, 1, 0, 1, 0, 1]), 2)[0],
            np.array([0, 2, 4, 6, 1, 3, 5, 7]),
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([0, 1, 0, 1, 0, 1, 0, 1]), 2)[1],
            np.array([4, 4]),
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([1, 0]), 2)[0], np.array([1, 0])
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([1, 0]), 2)[1], np.array([1, 1])
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([0, 0]), 1)[0], np.array([0, 1])
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([0, 0]), 1)[1], np.array([2])
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([0]), 1)[0], np.array([0])
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([0]), 1)[1], np.array([1])
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([], dtype=int_), 0)[0], np.array([])
        )
        np.testing.assert_array_equal(
            grouping.nb.get_group_map_nb(np.array([], dtype=int_), 0)[1], np.array([])
        )

    def test_is_grouped(self):
        assert vbt.Grouper(grouped_index, group_by=0).is_grouped()
        assert vbt.Grouper(grouped_index, group_by=0).is_grouped(group_by=True)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouped(group_by=1)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouped(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouped()
        assert vbt.Grouper(grouped_index).is_grouped(group_by=0)
        assert vbt.Grouper(grouped_index).is_grouped(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouped(group_by=False)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouped(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_grouping_enabled(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled()
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled(group_by=True)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled(group_by=1)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouping_enabled()
        assert vbt.Grouper(grouped_index).is_grouping_enabled(group_by=0)
        assert vbt.Grouper(grouped_index).is_grouping_enabled(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouping_enabled(group_by=False)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_enabled(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_grouping_disabled(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled()
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled(group_by=True)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled(group_by=1)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouping_disabled()
        assert not vbt.Grouper(grouped_index).is_grouping_disabled(group_by=0)
        assert not vbt.Grouper(grouped_index).is_grouping_disabled(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouping_disabled(group_by=False)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_disabled(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_grouping_modified(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_modified()
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_modified(group_by=True)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_modified(group_by=1)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_modified(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouping_modified()
        assert vbt.Grouper(grouped_index).is_grouping_modified(group_by=0)
        assert vbt.Grouper(grouped_index).is_grouping_modified(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouping_modified(group_by=False)
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_modified(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_grouping_changed(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_grouping_changed()
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_changed(group_by=True)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_changed(group_by=1)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_changed(group_by=False)
        assert not vbt.Grouper(grouped_index).is_grouping_changed()
        assert vbt.Grouper(grouped_index).is_grouping_changed(group_by=0)
        assert vbt.Grouper(grouped_index).is_grouping_changed(group_by=True)
        assert not vbt.Grouper(grouped_index).is_grouping_changed(group_by=False)
        assert vbt.Grouper(grouped_index, group_by=0).is_grouping_changed(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_is_group_count_changed(self):
        assert not vbt.Grouper(grouped_index, group_by=0).is_group_count_changed()
        assert vbt.Grouper(grouped_index, group_by=0).is_group_count_changed(group_by=True)
        assert vbt.Grouper(grouped_index, group_by=0).is_group_count_changed(group_by=1)
        assert vbt.Grouper(grouped_index, group_by=0).is_group_count_changed(group_by=False)
        assert not vbt.Grouper(grouped_index).is_group_count_changed()
        assert vbt.Grouper(grouped_index).is_group_count_changed(group_by=0)
        assert vbt.Grouper(grouped_index).is_group_count_changed(group_by=True)
        assert not vbt.Grouper(grouped_index).is_group_count_changed(group_by=False)
        assert not vbt.Grouper(grouped_index, group_by=0).is_group_count_changed(
            group_by=grouped_index.get_level_values(0) + 1
        )  # only labels

    def test_check_group_by(self):
        vbt.Grouper(grouped_index, group_by=None, allow_enable=True).check_group_by(group_by=0)
        with pytest.raises(Exception):
            vbt.Grouper(grouped_index, group_by=None, allow_enable=False).check_group_by(group_by=0)
        vbt.Grouper(grouped_index, group_by=0, allow_disable=True).check_group_by(group_by=False)
        with pytest.raises(Exception):
            vbt.Grouper(grouped_index, group_by=0, allow_disable=False).check_group_by(
                group_by=False
            )
        vbt.Grouper(grouped_index, group_by=0, allow_modify=True).check_group_by(group_by=1)
        vbt.Grouper(grouped_index, group_by=0, allow_modify=False).check_group_by(
            group_by=np.array([2, 2, 2, 2, 3, 3, 3, 3]),
        )
        with pytest.raises(Exception):
            vbt.Grouper(grouped_index, group_by=0, allow_modify=False).check_group_by(group_by=1)

    def test_resolve_group_by(self):
        assert vbt.Grouper(grouped_index, group_by=None).resolve_group_by() is None  # default
        assert_index_equal(
            vbt.Grouper(grouped_index, group_by=None).resolve_group_by(group_by=0),  # overrides
            pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
        )
        assert_index_equal(
            vbt.Grouper(grouped_index, group_by=0).resolve_group_by(),  # default
            pd.Index([1, 1, 1, 1, 0, 0, 0, 0], dtype="int64", name="first"),
        )
        assert_index_equal(
            vbt.Grouper(grouped_index, group_by=0).resolve_group_by(group_by=1),  # overrides
            pd.Index([3, 3, 2, 2, 1, 1, 0, 0], dtype="int64", name="second"),
        )

    def test_get_groups(self):
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_groups(),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        )
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_groups(group_by=0),
            np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        )

    def test_get_index(self):
        assert_index_equal(
            vbt.Grouper(grouped_index).get_index(),
            vbt.Grouper(grouped_index).index,
        )
        assert_index_equal(
            vbt.Grouper(grouped_index).get_index(group_by=0),
            pd.Index([1, 0], dtype="int64", name="first"),
        )

    def test_get_group_lens(self):
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_lens(),
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),
        )
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_lens(group_by=0),
            np.array([4, 4]),
        )

    def test_get_group_start_idxs(self):
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_start_idxs(),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        )
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_start_idxs(group_by=0),
            np.array([0, 4]),
        )

    def test_get_group_end_idxs(self):
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_end_idxs(),
            np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        )
        np.testing.assert_array_equal(
            vbt.Grouper(grouped_index).get_group_end_idxs(group_by=0),
            np.array([4, 8]),
        )

    def test_iter_group_idxs(self):
        np.testing.assert_array_equal(
            np.concatenate(tuple(vbt.Grouper(grouped_index).iter_group_idxs())),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        )
        np.testing.assert_array_equal(
            np.concatenate(tuple(vbt.Grouper(grouped_index).iter_group_idxs(group_by=0))),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        )

    def test_eq(self):
        assert vbt.Grouper(grouped_index) == vbt.Grouper(grouped_index)
        assert vbt.Grouper(grouped_index, group_by=0) == vbt.Grouper(grouped_index, group_by=0)
        assert vbt.Grouper(grouped_index) != 0
        assert vbt.Grouper(grouped_index) != vbt.Grouper(grouped_index, group_by=0)
        assert vbt.Grouper(grouped_index) != vbt.Grouper(pd.Index([0]))
        assert vbt.Grouper(grouped_index) != vbt.Grouper(grouped_index, allow_enable=False)
        assert vbt.Grouper(grouped_index) != vbt.Grouper(grouped_index, allow_disable=False)
        assert vbt.Grouper(grouped_index) != vbt.Grouper(grouped_index, allow_modify=False)
