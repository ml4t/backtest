import os

import pytest
import vectorbtpro as vbt
from numba import njit
from vectorbtpro._dtypes import *
from vectorbtpro.generic.enums import drawdown_dt, pattern_range_dt, range_dt
from vectorbtpro.portfolio.enums import log_dt, order_dt, trade_dt
from vectorbtpro.records.base import Records

from tests.utils import *

day_dt = np.timedelta64(86400000000000)

example_dt = np.dtype(
    [
        ("id", np.int64),
        ("col", np.int64),
        ("idx", np.int64),
        ("some_field1", np.float64),
        ("some_field2", np.float64),
    ],
    align=True,
)

records_arr = np.asarray(
    [
        (0, 0, 0, 10, 21),
        (1, 0, 1, 11, 20),
        (2, 0, 2, 12, 19),
        (0, 1, 0, 13, 18),
        (1, 1, 1, 14, 17),
        (2, 1, 2, 13, 18),
        (0, 2, 0, 12, 19),
        (1, 2, 1, 11, 20),
        (2, 2, 2, 10, 21),
    ],
    dtype=example_dt,
)
records_nosort_arr = np.concatenate((records_arr[0::3], records_arr[1::3], records_arr[2::3]))

group_by = pd.Index(["g1", "g1", "g2", "g2"])

wrapper = vbt.ArrayWrapper(
    index=["x", "y", "z"], columns=["a", "b", "c", "d"], ndim=2, freq="1 days"
)
wrapper_grouped = wrapper.replace(group_by=group_by)

records = vbt.records.Records(wrapper, records_arr)
records_grouped = vbt.records.Records(wrapper_grouped, records_arr)
records_nosort = records.replace(records_arr=records_nosort_arr)
records_nosort_grouped = vbt.records.Records(wrapper_grouped, records_nosort_arr)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.chunking["n_chunks"] = 2


def teardown_module():
    vbt.settings.reset()


# ############# col_mapper ############# #


class TestColumnMapper:
    def test_col_arr(self):
        np.testing.assert_array_equal(records.col_mapper["a"].col_arr, np.array([0, 0, 0]))
        np.testing.assert_array_equal(
            records.col_mapper.col_arr, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        )

    def test_get_col_arr(self):
        np.testing.assert_array_equal(records.col_mapper.get_col_arr(), records.col_mapper.col_arr)
        np.testing.assert_array_equal(
            records_grouped.col_mapper["g1"].get_col_arr(), np.array([0, 0, 0, 0, 0, 0])
        )
        np.testing.assert_array_equal(
            records_grouped.col_mapper.get_col_arr(), np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
        )

    def test_col_lens(self):
        np.testing.assert_array_equal(records.col_mapper["a"].col_lens, np.array([3]))
        np.testing.assert_array_equal(records.col_mapper.col_lens, np.array([3, 3, 3, 0]))

    def test_get_col_lens(self):
        np.testing.assert_array_equal(records.col_mapper.get_col_lens(), np.array([3, 3, 3, 0]))
        np.testing.assert_array_equal(
            records_grouped.col_mapper["g1"].get_col_lens(), np.array([6])
        )
        np.testing.assert_array_equal(records_grouped.col_mapper.get_col_lens(), np.array([6, 3]))

    def test_col_map(self):
        np.testing.assert_array_equal(records.col_mapper["a"].col_map[0], np.array([0, 1, 2]))
        np.testing.assert_array_equal(records.col_mapper["a"].col_map[1], np.array([3]))
        np.testing.assert_array_equal(
            records.col_mapper.col_map[0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        )
        np.testing.assert_array_equal(records.col_mapper.col_map[1], np.array([3, 3, 3, 0]))

    def test_get_col_map(self):
        np.testing.assert_array_equal(
            records.col_mapper.get_col_map()[0], records.col_mapper.col_map[0]
        )
        np.testing.assert_array_equal(
            records.col_mapper.get_col_map()[1], records.col_mapper.col_map[1]
        )
        np.testing.assert_array_equal(
            records_grouped.col_mapper["g1"].get_col_map()[0], np.array([0, 1, 2, 3, 4, 5])
        )
        np.testing.assert_array_equal(
            records_grouped.col_mapper["g1"].get_col_map()[1], np.array([6])
        )
        np.testing.assert_array_equal(
            records_grouped.col_mapper.get_col_map()[0],
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        )
        np.testing.assert_array_equal(records_grouped.col_mapper.get_col_map()[1], np.array([6, 3]))

    def test_is_sorted(self):
        assert records.col_mapper.is_sorted()
        assert not records_nosort.col_mapper.is_sorted()


# ############# mapped_array ############# #

mapped_array = records.map_field("some_field1")
mapped_array_grouped = records_grouped.map_field("some_field1")
mapped_array_nosort = records_nosort.map_field("some_field1")
mapped_array_nosort_grouped = records_nosort_grouped.map_field("some_field1")
mapping = {x: "test_" + str(x) for x in pd.unique(mapped_array.values)}
mp_mapped_array = mapped_array.replace(mapping=mapping)
mp_mapped_array_grouped = mapped_array_grouped.replace(mapping=mapping)


class TestMappedArray:
    def test_row_stack(self):
        df1 = pd.DataFrame(
            {
                "a": [True, True, False, True],
            },
            index=pd.date_range("2020-01-01", "2020-01-04"),
        )
        df2 = pd.DataFrame(
            {
                "a": [True, True, False, True, False, True],
                "b": [False, False, True, False, True, False],
            },
            index=pd.date_range("2020-01-05", "2020-01-10"),
        )
        mapped_array1 = vbt.MappedArray(
            df1.vbt.wrapper,
            np.array([], dtype=int_),
            np.array([], dtype=int_),
            np.array([], dtype=int_),
        )
        mapped_array2 = vbt.MappedArray(
            df2.vbt.wrapper,
            np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))),
            np.concatenate((np.full(df2["a"].sum(), 0), np.full(df2["b"].sum(), 1))),
            np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))),
        )
        new_mapped_array = vbt.MappedArray.row_stack(mapped_array1, mapped_array2)
        assert_index_equal(new_mapped_array.wrapper.index, df1.index.append(df2.index))
        assert_index_equal(new_mapped_array.wrapper.columns, df2.columns)
        np.testing.assert_array_equal(new_mapped_array.mapped_arr, np.array([0, 1, 3, 5, 2, 4]))
        np.testing.assert_array_equal(new_mapped_array.col_arr, np.array([0, 0, 0, 0, 1, 1]))
        np.testing.assert_array_equal(new_mapped_array.idx_arr, np.array([4, 5, 7, 9, 6, 8]))
        np.testing.assert_array_equal(new_mapped_array.id_arr, np.array([0, 1, 2, 3, 0, 1]))
        mapped_array1 = vbt.MappedArray(
            df1.vbt.wrapper,
            np.flatnonzero(df1["a"]),
            np.full(df1["a"].sum(), 0),
            np.flatnonzero(df1["a"]),
        )
        new_mapped_array = vbt.MappedArray.row_stack(mapped_array1, mapped_array2)
        assert_index_equal(new_mapped_array.wrapper.index, df1.index.append(df2.index))
        assert_index_equal(new_mapped_array.wrapper.columns, df2.columns)
        np.testing.assert_array_equal(
            new_mapped_array.mapped_arr, np.array([0, 1, 3, 0, 1, 3, 5, 0, 1, 3, 2, 4])
        )
        np.testing.assert_array_equal(
            new_mapped_array.col_arr, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        )
        np.testing.assert_array_equal(
            new_mapped_array.idx_arr, np.array([0, 1, 3, 4, 5, 7, 9, 0, 1, 3, 6, 8])
        )
        np.testing.assert_array_equal(
            new_mapped_array.id_arr, np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4])
        )

        df1 = pd.DataFrame(
            {
                "a": [True, True, False, True],
                "b": [False, False, True, False],
            },
            index=pd.date_range("2020-01-01", "2020-01-04"),
        )
        df2 = pd.DataFrame(
            {
                "a": [True, True, False, True, False, True],
                "b": [False, False, True, False, True, False],
            },
            index=pd.date_range("2020-01-05", "2020-01-10"),
        )
        mapped_array1 = vbt.MappedArray(
            df1.vbt.wrapper,
            np.concatenate((np.flatnonzero(df1["a"]), np.flatnonzero(df1["b"]))),
            np.concatenate((np.full(df1["a"].sum(), 0), np.full(df1["b"].sum(), 1))),
            np.concatenate((np.flatnonzero(df1["a"]), np.flatnonzero(df1["b"]))),
        )
        mapped_array2 = vbt.MappedArray(
            df2.vbt.wrapper,
            np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))),
            np.concatenate((np.full(df2["a"].sum(), 0), np.full(df2["b"].sum(), 1))),
            np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))),
        )
        new_mapped_array = vbt.MappedArray.row_stack(mapped_array1, mapped_array2)
        assert_index_equal(new_mapped_array.wrapper.index, df1.index.append(df2.index))
        assert_index_equal(new_mapped_array.wrapper.columns, df1.columns)
        np.testing.assert_array_equal(
            new_mapped_array.mapped_arr, np.array([0, 1, 3, 0, 1, 3, 5, 2, 2, 4])
        )
        np.testing.assert_array_equal(
            new_mapped_array.col_arr, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        )
        np.testing.assert_array_equal(
            new_mapped_array.idx_arr, np.array([0, 1, 3, 4, 5, 7, 9, 2, 6, 8])
        )
        np.testing.assert_array_equal(
            new_mapped_array.id_arr, np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
        )
        new_mapped_array = vbt.MappedArray.row_stack(
            mapped_array1.replace(some_arg=2, check_expected_keys_=False),
            mapped_array2.replace(some_arg=2, check_expected_keys_=False),
        )
        assert new_mapped_array.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.MappedArray.row_stack(
                mapped_array1.replace(some_arg=2, check_expected_keys_=False),
                mapped_array2.replace(some_arg=3, check_expected_keys_=False),
            )
        with pytest.raises(Exception):
            vbt.MappedArray.row_stack(
                mapped_array1.replace(some_arg=2, check_expected_keys_=False),
                mapped_array2,
            )
        with pytest.raises(Exception):
            vbt.MappedArray.row_stack(
                mapped_array1,
                mapped_array2.replace(some_arg=2, check_expected_keys_=False),
            )

    def test_column_stack(self):
        df1 = pd.DataFrame(
            {
                "a": [True, True, False, True, True],
            },
            index=pd.date_range("2020-01-01", "2020-01-05"),
        )
        df2 = pd.DataFrame(
            {
                "b": [True, False, True, False, True],
                "c": [False, True, False, True, False],
            },
            index=pd.date_range("2020-01-03", "2020-01-07"),
        )
        mapped_array1 = vbt.MappedArray(
            df1.vbt.wrapper,
            np.array([], dtype=int_),
            np.array([], dtype=int_),
            np.array([], dtype=int_),
        )
        mapped_array2 = vbt.MappedArray(
            df2.vbt.wrapper,
            np.concatenate((np.flatnonzero(df2["b"]), np.flatnonzero(df2["c"]))),
            np.concatenate((np.full(df2["b"].sum(), 0), np.full(df2["c"].sum(), 1))),
            np.concatenate((np.flatnonzero(df2["b"]), np.flatnonzero(df2["c"]))),
        )
        mapped_array = vbt.MappedArray.column_stack(mapped_array1, mapped_array2)
        assert_index_equal(mapped_array.wrapper.index, df1.index.union(df2.index))
        assert_index_equal(mapped_array.wrapper.columns, df1.columns.append(df2.columns))
        np.testing.assert_array_equal(mapped_array.mapped_arr, np.array([0, 2, 4, 1, 3]))
        np.testing.assert_array_equal(mapped_array.col_arr, np.array([1, 1, 1, 2, 2]))
        np.testing.assert_array_equal(mapped_array.idx_arr, np.array([2, 4, 6, 3, 5]))
        np.testing.assert_array_equal(mapped_array.id_arr, np.array([0, 1, 2, 0, 1]))
        mapped_array1 = vbt.MappedArray(
            df1.vbt.wrapper,
            np.flatnonzero(df1["a"]),
            np.full(df1["a"].sum(), 0),
            np.flatnonzero(df1["a"]),
        )
        mapped_array = vbt.MappedArray.column_stack(mapped_array1, mapped_array2)
        assert_index_equal(mapped_array.wrapper.index, df1.index.union(df2.index))
        assert_index_equal(mapped_array.wrapper.columns, df1.columns.append(df2.columns))
        np.testing.assert_array_equal(
            mapped_array.mapped_arr, np.array([0, 1, 3, 4, 0, 2, 4, 1, 3])
        )
        np.testing.assert_array_equal(mapped_array.col_arr, np.array([0, 0, 0, 0, 1, 1, 1, 2, 2]))
        np.testing.assert_array_equal(mapped_array.idx_arr, np.array([0, 1, 3, 4, 2, 4, 6, 3, 5]))
        np.testing.assert_array_equal(mapped_array.id_arr, np.array([0, 1, 2, 3, 0, 1, 2, 0, 1]))
        mapped_array = vbt.MappedArray.column_stack(
            mapped_array1.replace(some_arg=2, check_expected_keys_=False),
            mapped_array2.replace(some_arg=2, check_expected_keys_=False),
        )
        assert mapped_array.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.MappedArray.column_stack(
                mapped_array1.replace(some_arg=2, check_expected_keys_=False),
                mapped_array2.replace(some_arg=3, check_expected_keys_=False),
            )
        with pytest.raises(Exception):
            vbt.MappedArray.column_stack(
                mapped_array1.replace(some_arg=2, check_expected_keys_=False),
                mapped_array2,
            )
        with pytest.raises(Exception):
            vbt.MappedArray.column_stack(
                mapped_array1,
                mapped_array2.replace(some_arg=2, check_expected_keys_=False),
            )

    def test_config(self, tmp_path):
        assert vbt.MappedArray.loads(mapped_array.dumps()) == mapped_array
        mapped_array.save(tmp_path / "mapped_array")
        assert vbt.MappedArray.load(tmp_path / "mapped_array") == mapped_array
        mapped_array.save(tmp_path / "mapped_array", file_format="ini")
        assert vbt.MappedArray.load(tmp_path / "mapped_array", file_format="ini") == mapped_array

    def test_mapped_arr(self):
        np.testing.assert_array_equal(mapped_array["a"].values, np.array([10.0, 11.0, 12.0]))
        np.testing.assert_array_equal(
            mapped_array.values,
            np.array([10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0]),
        )

    def test_id_arr(self):
        np.testing.assert_array_equal(mapped_array["a"].id_arr, np.array([0, 1, 2]))
        np.testing.assert_array_equal(mapped_array.id_arr, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(
            vbt.MappedArray(mapped_array.wrapper, mapped_array.values, mapped_array.col_arr).id_arr,
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        )

    def test_col_arr(self):
        np.testing.assert_array_equal(mapped_array["a"].col_arr, np.array([0, 0, 0]))
        np.testing.assert_array_equal(mapped_array.col_arr, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))

    def test_idx_arr(self):
        np.testing.assert_array_equal(mapped_array["a"].idx_arr, np.array([0, 1, 2]))
        np.testing.assert_array_equal(mapped_array.idx_arr, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))

    def test_is_sorted(self):
        assert mapped_array.is_sorted()
        assert mapped_array.is_sorted(incl_id=True)
        assert not mapped_array_nosort.is_sorted()
        assert not mapped_array_nosort.is_sorted(incl_id=True)

    def test_sort(self):
        assert mapped_array.sort().is_sorted()
        assert mapped_array.sort().is_sorted(incl_id=True)
        assert mapped_array.sort(incl_id=True).is_sorted(incl_id=True)
        assert mapped_array_nosort.sort().is_sorted()
        assert mapped_array_nosort.sort().is_sorted(incl_id=True)
        assert mapped_array_nosort.sort(incl_id=True).is_sorted(incl_id=True)

    def test_apply_mask(self):
        mask_a = mapped_array["a"].values >= mapped_array["a"].values.mean()
        np.testing.assert_array_equal(mapped_array["a"].apply_mask(mask_a).id_arr, np.array([1, 2]))
        mask = mapped_array.values >= mapped_array.values.mean()
        filtered = mapped_array.apply_mask(mask)
        np.testing.assert_array_equal(filtered.id_arr, np.array([2, 0, 1, 2, 0]))
        np.testing.assert_array_equal(filtered.col_arr, mapped_array.col_arr[mask])
        np.testing.assert_array_equal(filtered.idx_arr, mapped_array.idx_arr[mask])
        assert mapped_array_grouped.apply_mask(mask).wrapper == mapped_array_grouped.wrapper
        assert (
            mapped_array_grouped.apply_mask(mask, group_by=False).wrapper.grouper.group_by is None
        )

    def test_top_n_mask(self):
        np.testing.assert_array_equal(
            mapped_array.top_n_mask(1),
            np.array([False, False, True, False, True, False, True, False, False]),
        )
        np.testing.assert_array_equal(
            mapped_array.top_n_mask(1, jitted=dict(parallel=True)),
            mapped_array.top_n_mask(1, jitted=dict(parallel=False)),
        )
        np.testing.assert_array_equal(
            mapped_array.top_n_mask(1, chunked=True),
            mapped_array.top_n_mask(1, chunked=False),
        )

    def test_bottom_n_mask(self):
        np.testing.assert_array_equal(
            mapped_array.bottom_n_mask(1),
            np.array([True, False, False, True, False, False, False, False, True]),
        )
        np.testing.assert_array_equal(
            mapped_array.bottom_n_mask(1, jitted=dict(parallel=True)),
            mapped_array.bottom_n_mask(1, jitted=dict(parallel=False)),
        )
        np.testing.assert_array_equal(
            mapped_array.bottom_n_mask(1, chunked=True),
            mapped_array.bottom_n_mask(1, chunked=False),
        )

    def test_top_n(self):
        np.testing.assert_array_equal(mapped_array.top_n(1).id_arr, np.array([2, 1, 0]))

    def test_bottom_n(self):
        np.testing.assert_array_equal(mapped_array.bottom_n(1).id_arr, np.array([0, 0, 2]))

    def test_has_conflicts(self):
        assert not mapped_array.has_conflicts()
        mapped_array2 = vbt.MappedArray(
            wrapper,
            records_arr["some_field1"].tolist() + [1],
            records_arr["col"].tolist() + [2],
            idx_arr=records_arr["idx"].tolist() + [2],
        )
        assert mapped_array2.has_conflicts()

    def test_coverage_map(self):
        assert_frame_equal(
            mapped_array.coverage_map(),
            pd.DataFrame(
                [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
                index=wrapper.index,
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array.coverage_map(group_by=group_by),
            pd.DataFrame(
                [[2, 1], [2, 1], [2, 1]],
                index=wrapper.index,
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )

    def test_to_pd(self):
        target = pd.DataFrame(
            np.array(
                [[10.0, 13.0, 12.0, np.nan], [11.0, 14.0, 11.0, np.nan], [12.0, 13.0, 10.0, np.nan]]
            ),
            index=wrapper.index,
            columns=wrapper.columns,
        )
        assert_series_equal(mapped_array["a"].to_pd(), target["a"])
        assert_frame_equal(mapped_array.to_pd(), target)
        assert_frame_equal(mapped_array.to_pd(fill_value=0.0), target.fillna(0.0))
        mapped_array2 = vbt.MappedArray(
            wrapper,
            records_arr["some_field1"].tolist() + [1],
            records_arr["col"].tolist() + [2],
            idx_arr=records_arr["idx"].tolist() + [2],
        )
        assert_frame_equal(
            mapped_array2.to_pd(),
            pd.DataFrame(
                np.array(
                    [
                        [10.0, 13.0, 12.0, np.nan],
                        [11.0, 14.0, 11.0, np.nan],
                        [12.0, 13.0, 1.0, np.nan],
                    ]
                ),
                index=wrapper.index,
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array2.to_pd(group_by=group_by),
            pd.DataFrame(
                np.array([[13.0, 12.0], [14.0, 11.0], [13.0, 1.0]]),
                index=wrapper.index,
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )
        assert_frame_equal(
            mapped_array2.to_pd(repeat_index=True),
            pd.DataFrame(
                np.array(
                    [
                        [10.0, 13.0, 12.0, np.nan],
                        [11.0, 14.0, 11.0, np.nan],
                        [12.0, 13.0, 10.0, np.nan],
                        [np.nan, np.nan, 1.0, np.nan],
                    ]
                ),
                index=np.concatenate((wrapper.index, wrapper.index[[-1]])),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array2.to_pd(repeat_index=True, group_by=group_by),
            pd.DataFrame(
                np.array(
                    [
                        [10.0, 12.0],
                        [13.0, np.nan],
                        [11.0, 11.0],
                        [14.0, np.nan],
                        [12.0, 10.0],
                        [13.0, 1.0],
                    ]
                ),
                index=np.repeat(wrapper.index, 2),
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )
        assert_series_equal(
            mapped_array["a"].to_pd(ignore_index=True),
            pd.Series(np.array([10.0, 11.0, 12.0]), name="a"),
        )
        assert_frame_equal(
            mapped_array.to_pd(ignore_index=True),
            pd.DataFrame(
                np.array(
                    [
                        [10.0, 13.0, 12.0, np.nan],
                        [11.0, 14.0, 11.0, np.nan],
                        [12.0, 13.0, 10.0, np.nan],
                    ]
                ),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array.to_pd(fill_value=0, ignore_index=True),
            pd.DataFrame(
                np.array(
                    [[10.0, 13.0, 12.0, 0.0], [11.0, 14.0, 11.0, 0.0], [12.0, 13.0, 10.0, 0.0]]
                ),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array_grouped.to_pd(ignore_index=True),
            pd.DataFrame(
                np.array(
                    [
                        [10.0, 12.0],
                        [11.0, 11.0],
                        [12.0, 10.0],
                        [13.0, np.nan],
                        [14.0, np.nan],
                        [13.0, np.nan],
                    ]
                ),
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )

    def test_apply(self):
        @njit
        def cumsum_apply_nb(a):
            return np.cumsum(a)

        @njit
        def cumsum_apply_meta_nb(ridxs, col, a):
            return np.cumsum(a[ridxs])

        np.testing.assert_array_equal(
            mapped_array["a"].apply(cumsum_apply_nb).values, np.array([10.0, 21.0, 33.0])
        )
        np.testing.assert_array_equal(
            mapped_array.apply(cumsum_apply_nb).values,
            np.array([10.0, 21.0, 33.0, 13.0, 27.0, 40.0, 12.0, 23.0, 33.0]),
        )
        np.testing.assert_array_equal(
            mapped_array_grouped.apply(cumsum_apply_nb, apply_per_group=False).values,
            np.array([10.0, 21.0, 33.0, 13.0, 27.0, 40.0, 12.0, 23.0, 33.0]),
        )
        np.testing.assert_array_equal(
            mapped_array_grouped.apply(cumsum_apply_nb, apply_per_group=True).values,
            np.array([10.0, 21.0, 33.0, 46.0, 60.0, 73.0, 12.0, 23.0, 33.0]),
        )
        assert (
            mapped_array_grouped.apply(cumsum_apply_nb).wrapper
            == mapped_array.apply(cumsum_apply_nb, group_by=group_by).wrapper
        )
        assert mapped_array.apply(cumsum_apply_nb, group_by=False).wrapper.grouper.group_by is None
        np.testing.assert_array_equal(
            mapped_array.apply(cumsum_apply_nb, jitted=dict(parallel=True)).values,
            mapped_array.apply(cumsum_apply_nb, jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            mapped_array.apply(cumsum_apply_nb, chunked=True).values,
            mapped_array.apply(cumsum_apply_nb, chunked=False).values,
        )
        np.testing.assert_array_equal(
            type(mapped_array)
            .apply(cumsum_apply_meta_nb, mapped_array.values, col_mapper=mapped_array.col_mapper)
            .values,
            mapped_array.apply(cumsum_apply_nb).values,
        )
        np.testing.assert_array_equal(
            type(mapped_array)
            .apply(
                cumsum_apply_meta_nb,
                mapped_array.values,
                col_mapper=mapped_array.col_mapper,
                jitted=dict(parallel=True),
            )
            .values,
            type(mapped_array)
            .apply(
                cumsum_apply_meta_nb,
                mapped_array.values,
                col_mapper=mapped_array.col_mapper,
                jitted=dict(parallel=False),
            )
            .values,
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0, mapper=vbt.GroupIdxsMapper(arg_query="col_map")),
                )
            )
        )
        np.testing.assert_array_equal(
            type(mapped_array)
            .apply(
                cumsum_apply_meta_nb,
                mapped_array.values,
                col_mapper=mapped_array.col_mapper,
                chunked=chunked,
            )
            .values,
            type(mapped_array)
            .apply(
                cumsum_apply_meta_nb,
                mapped_array.values,
                col_mapper=mapped_array.col_mapper,
                chunked=False,
            )
            .values,
        )

    def test_reduce_segments(self):
        @njit
        def sum_reduce_nb(a):
            return np.sum(a)

        np.testing.assert_array_equal(
            mapped_array["a"].reduce_segments(np.array([0, 1, 2]), sum_reduce_nb).values,
            np.array([10.0, 11.0, 12.0]),
        )
        np.testing.assert_array_equal(
            mapped_array["a"].reduce_segments(np.array([0, 0, 2]), sum_reduce_nb).values,
            np.array([21.0, 12.0]),
        )
        np.testing.assert_array_equal(
            mapped_array["a"].reduce_segments(np.array([0, 0, 0]), sum_reduce_nb).values,
            np.array([33.0]),
        )
        with pytest.raises(Exception):
            mapped_array["a"].reduce_segments(np.array([2, 1, 0]), sum_reduce_nb)
        np.testing.assert_array_equal(
            mapped_array.reduce_segments(np.arange(9), sum_reduce_nb).values,
            np.array([10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0]),
        )
        np.testing.assert_array_equal(
            mapped_array.reduce_segments(
                np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]), sum_reduce_nb
            ).values,
            np.array([10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0]),
        )
        np.testing.assert_array_equal(
            mapped_array.reduce_segments(
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), sum_reduce_nb
            ).values,
            np.array([33.0, 40.0, 33.0]),
        )
        mapped_array2 = vbt.MappedArray(
            wrapper,
            np.repeat(records_arr["some_field1"], 2),
            np.repeat(records_arr["col"], 2),
            idx_arr=np.repeat(records_arr["idx"], 2),
        )
        result = mapped_array2.reduce_segments(
            (mapped_array2.idx_arr, mapped_array2.col_arr), sum_reduce_nb
        )
        np.testing.assert_array_equal(result.values, mapped_array.values * 2)
        np.testing.assert_array_equal(result.col_arr, mapped_array.col_arr)
        np.testing.assert_array_equal(result.idx_arr, mapped_array.idx_arr)
        np.testing.assert_array_equal(result.id_arr, np.array([1, 3, 5, 1, 3, 5, 1, 3, 5]))
        np.testing.assert_array_equal(
            mapped_array_grouped.reduce_segments(
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                sum_reduce_nb,
                apply_per_group=False,
            ).values,
            np.array([33.0, 40.0, 33.0]),
        )
        np.testing.assert_array_equal(
            mapped_array_grouped.reduce_segments(
                np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                sum_reduce_nb,
                apply_per_group=True,
            ).values,
            np.array([33.0, 40.0, 33.0]),
        )
        assert (
            mapped_array_grouped.reduce_segments(np.arange(9), sum_reduce_nb).wrapper
            == mapped_array.reduce_segments(np.arange(9), sum_reduce_nb, group_by=group_by).wrapper
        )
        assert (
            mapped_array.reduce_segments(
                np.arange(9), sum_reduce_nb, group_by=False
            ).wrapper.grouper.group_by
            is None
        )
        np.testing.assert_array_equal(
            mapped_array.reduce_segments(np.arange(9), sum_reduce_nb, chunked=True).values,
            mapped_array.reduce_segments(np.arange(9), sum_reduce_nb, chunked=False).values,
        )

    def test_reduce(self):
        @njit
        def mean_reduce_nb(a):
            return np.mean(a)

        @njit
        def mean_reduce_meta_nb(ridxs, col, records):
            return np.mean(records[ridxs])

        assert mapped_array["a"].reduce(mean_reduce_nb) == 11.0
        assert_series_equal(
            mapped_array.reduce(mean_reduce_nb),
            pd.Series(
                np.array([11.0, 13.333333333333334, 11.0, np.nan]), index=wrapper.columns
            ).rename("reduce"),
        )
        assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, fill_value=0.0),
            pd.Series(
                np.array([11.0, 13.333333333333334, 11.0, 0.0]), index=wrapper.columns
            ).rename("reduce"),
        )
        assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, fill_value=0.0, wrap_kwargs=dict(dtype=int_)),
            pd.Series(np.array([11, 13, 11, 0]), index=wrapper.columns).rename("reduce"),
        )
        assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, wrap_kwargs=dict(to_timedelta=True)),
            pd.Series(
                np.array([11.0, 13.333333333333334, 11.0, np.nan]), index=wrapper.columns
            ).rename("reduce")
            * day_dt,
        )
        assert_series_equal(
            mapped_array_grouped.reduce(mean_reduce_nb),
            pd.Series(
                [12.166666666666666, 11.0], index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("reduce"),
        )
        assert mapped_array_grouped["g1"].reduce(mean_reduce_nb) == 12.166666666666666
        assert_series_equal(
            mapped_array_grouped[["g1"]].reduce(mean_reduce_nb),
            pd.Series([12.166666666666666], index=pd.Index(["g1"], dtype="object")).rename(
                "reduce"
            ),
        )
        assert_series_equal(
            mapped_array.reduce(mean_reduce_nb),
            mapped_array_grouped.reduce(mean_reduce_nb, group_by=False),
        )
        assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, group_by=group_by),
            mapped_array_grouped.reduce(mean_reduce_nb),
        )
        assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, jitted=dict(parallel=True)),
            mapped_array.reduce(mean_reduce_nb, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, chunked=True),
            mapped_array.reduce(mean_reduce_nb, chunked=False),
        )
        assert_series_equal(
            type(mapped_array).reduce(
                mean_reduce_meta_nb, mapped_array.values, col_mapper=mapped_array.col_mapper
            ),
            mapped_array.reduce(mean_reduce_nb),
        )
        assert_series_equal(
            type(mapped_array).reduce(
                mean_reduce_meta_nb,
                mapped_array.values,
                col_mapper=mapped_array.col_mapper,
                jitted=dict(parallel=True),
            ),
            type(mapped_array).reduce(
                mean_reduce_meta_nb,
                mapped_array.values,
                col_mapper=mapped_array.col_mapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0, mapper=vbt.GroupIdxsMapper(arg_query="col_map")),
                )
            )
        )
        assert_series_equal(
            type(mapped_array).reduce(
                mean_reduce_meta_nb,
                mapped_array.values,
                col_mapper=mapped_array.col_mapper,
                chunked=chunked,
            ),
            type(mapped_array).reduce(
                mean_reduce_meta_nb,
                mapped_array.values,
                col_mapper=mapped_array.col_mapper,
                chunked=False,
            ),
        )

    def test_reduce_to_idx(self):
        @njit
        def argmin_reduce_nb(a):
            return np.argmin(a)

        @njit
        def argmin_reduce_meta_nb(ridxs, col, records):
            return np.argmin(records[ridxs])

        assert mapped_array["a"].reduce(argmin_reduce_nb, returns_idx=True) == "x"
        assert_series_equal(
            mapped_array.reduce(argmin_reduce_nb, returns_idx=True),
            pd.Series(
                np.array(["x", "x", "z", np.nan], dtype=object), index=wrapper.columns
            ).rename("reduce"),
        )
        assert_series_equal(
            mapped_array.reduce(argmin_reduce_nb, returns_idx=True, to_index=False),
            pd.Series(np.array([0, 0, 2, -1], dtype=int), index=wrapper.columns).rename("reduce"),
        )
        assert_series_equal(
            mapped_array_grouped.reduce(argmin_reduce_nb, returns_idx=True, to_index=False),
            pd.Series(
                np.array([0, 2], dtype=int), index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("reduce"),
        )
        assert_series_equal(
            mapped_array.reduce(argmin_reduce_nb, returns_idx=True, jitted=dict(parallel=True)),
            mapped_array.reduce(argmin_reduce_nb, returns_idx=True, jitted=dict(parallel=False)),
        )
        assert_series_equal(
            mapped_array.reduce(argmin_reduce_nb, returns_idx=True, chunked=True),
            mapped_array.reduce(argmin_reduce_nb, returns_idx=True, chunked=False),
        )
        assert_series_equal(
            type(mapped_array).reduce(
                argmin_reduce_meta_nb,
                mapped_array.values,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
            ),
            mapped_array.reduce(argmin_reduce_nb, returns_idx=True),
        )
        assert_series_equal(
            type(mapped_array).reduce(
                argmin_reduce_meta_nb,
                mapped_array.values,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
                jitted=dict(parallel=True),
            ),
            type(mapped_array).reduce(
                argmin_reduce_meta_nb,
                mapped_array.values,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0, mapper=vbt.GroupIdxsMapper(arg_query="col_map")),
                )
            )
        )
        assert_series_equal(
            type(mapped_array).reduce(
                argmin_reduce_meta_nb,
                mapped_array.values,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
                chunked=chunked,
            ),
            type(mapped_array).reduce(
                argmin_reduce_meta_nb,
                mapped_array.values,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
                chunked=False,
            ),
        )

    def test_reduce_to_array(self):
        @njit
        def min_max_reduce_nb(a):
            return np.array([np.min(a), np.max(a)])

        @njit
        def min_max_reduce_meta_nb(ridxs, col, records):
            return np.array([np.min(records[ridxs]), np.max(records[ridxs])])

        assert_series_equal(
            mapped_array["a"].reduce(
                min_max_reduce_nb,
                returns_array=True,
                wrap_kwargs=dict(name_or_index=["min", "max"]),
            ),
            pd.Series([10.0, 12.0], index=pd.Index(["min", "max"], dtype="object"), name="a"),
        )
        assert_frame_equal(
            mapped_array.reduce(
                min_max_reduce_nb,
                returns_array=True,
                wrap_kwargs=dict(name_or_index=["min", "max"]),
            ),
            pd.DataFrame(
                np.array([[10.0, 13.0, 10.0, np.nan], [12.0, 14.0, 12.0, np.nan]]),
                index=pd.Index(["min", "max"], dtype="object"),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, returns_array=True, fill_value=0.0),
            pd.DataFrame(
                np.array([[10.0, 13.0, 10.0, 0.0], [12.0, 14.0, 12.0, 0.0]]),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array.reduce(
                min_max_reduce_nb, returns_array=True, wrap_kwargs=dict(to_timedelta=True)
            ),
            pd.DataFrame(
                np.array([[10.0, 13.0, 10.0, np.nan], [12.0, 14.0, 12.0, np.nan]]),
                columns=wrapper.columns,
            )
            * day_dt,
        )
        assert_frame_equal(
            mapped_array_grouped.reduce(min_max_reduce_nb, returns_array=True),
            pd.DataFrame(
                np.array([[10.0, 10.0], [14.0, 12.0]]),
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )
        assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, returns_array=True),
            mapped_array_grouped.reduce(min_max_reduce_nb, returns_array=True, group_by=False),
        )
        assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, returns_array=True, group_by=group_by),
            mapped_array_grouped.reduce(min_max_reduce_nb, returns_array=True),
        )
        assert_series_equal(
            mapped_array_grouped["g1"].reduce(min_max_reduce_nb, returns_array=True),
            pd.Series([10.0, 14.0], name="g1"),
        )
        assert_frame_equal(
            mapped_array_grouped[["g1"]].reduce(min_max_reduce_nb, returns_array=True),
            pd.DataFrame([[10.0], [14.0]], columns=pd.Index(["g1"], dtype="object")),
        )
        assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, returns_array=True, jitted=dict(parallel=True)),
            mapped_array.reduce(min_max_reduce_nb, returns_array=True, jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, returns_array=True, chunked=True),
            mapped_array.reduce(min_max_reduce_nb, returns_array=True, chunked=False),
        )
        assert_frame_equal(
            type(mapped_array).reduce(
                min_max_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                col_mapper=mapped_array.col_mapper,
            ),
            mapped_array.reduce(min_max_reduce_nb, returns_array=True),
        )
        assert_frame_equal(
            type(mapped_array).reduce(
                min_max_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                col_mapper=mapped_array.col_mapper,
                jitted=dict(parallel=True),
            ),
            type(mapped_array).reduce(
                min_max_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                col_mapper=mapped_array.col_mapper,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0, mapper=vbt.GroupIdxsMapper(arg_query="col_map")),
                )
            )
        )
        assert_frame_equal(
            type(mapped_array).reduce(
                min_max_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                col_mapper=mapped_array.col_mapper,
                chunked=chunked,
            ),
            type(mapped_array).reduce(
                min_max_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                col_mapper=mapped_array.col_mapper,
                chunked=False,
            ),
        )

    def test_reduce_to_idx_array(self):
        @njit
        def idxmin_idxmax_reduce_nb(a):
            return np.array([np.argmin(a), np.argmax(a)])

        @njit
        def idxmin_idxmax_reduce_meta_nb(ridxs, col, records):
            return np.array([np.argmin(records[ridxs]), np.argmax(records[ridxs])])

        assert_series_equal(
            mapped_array["a"].reduce(
                idxmin_idxmax_reduce_nb,
                returns_array=True,
                returns_idx=True,
                wrap_kwargs=dict(name_or_index=["min", "max"]),
            ),
            pd.Series(
                np.array(["x", "z"], dtype=object),
                index=pd.Index(["min", "max"], dtype="object"),
                name="a",
            ),
        )
        assert_frame_equal(
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb,
                returns_array=True,
                returns_idx=True,
                wrap_kwargs=dict(name_or_index=["min", "max"]),
            ),
            pd.DataFrame(
                {"a": ["x", "z"], "b": ["x", "y"], "c": ["z", "x"], "d": [np.nan, np.nan]},
                index=pd.Index(["min", "max"], dtype="object"),
            ),
        )
        assert_frame_equal(
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb, returns_array=True, returns_idx=True, to_index=False
            ),
            pd.DataFrame(np.array([[0, 0, 2, -1], [2, 1, 0, -1]]), columns=wrapper.columns),
        )
        assert_frame_equal(
            mapped_array_grouped.reduce(
                idxmin_idxmax_reduce_nb, returns_array=True, returns_idx=True, to_index=False
            ),
            pd.DataFrame(
                np.array([[0, 2], [1, 0]]), columns=pd.Index(["g1", "g2"], dtype="object")
            ),
        )
        assert_frame_equal(
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb,
                returns_array=True,
                returns_idx=True,
                jitted=dict(parallel=True),
            ),
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb,
                returns_array=True,
                returns_idx=True,
                jitted=dict(parallel=False),
            ),
        )
        assert_frame_equal(
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb, returns_array=True, returns_idx=True, chunked=True
            ),
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb, returns_array=True, returns_idx=True, chunked=False
            ),
        )
        assert_frame_equal(
            type(mapped_array).reduce(
                idxmin_idxmax_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
            ),
            mapped_array.reduce(idxmin_idxmax_reduce_nb, returns_array=True, returns_idx=True),
        )
        assert_frame_equal(
            type(mapped_array).reduce(
                idxmin_idxmax_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
                jitted=dict(parallel=True),
            ),
            type(mapped_array).reduce(
                idxmin_idxmax_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
                jitted=dict(parallel=False),
            ),
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0, mapper=vbt.GroupIdxsMapper(arg_query="col_map")),
                )
            )
        )
        assert_frame_equal(
            type(mapped_array).reduce(
                idxmin_idxmax_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
                chunked=chunked,
            ),
            type(mapped_array).reduce(
                idxmin_idxmax_reduce_meta_nb,
                mapped_array.values,
                returns_array=True,
                returns_idx=True,
                col_mapper=mapped_array.col_mapper,
                idx_arr=mapped_array.idx_arr,
                chunked=False,
            ),
        )

    def test_nth(self):
        assert mapped_array["a"].nth(0) == 10.0
        assert_series_equal(
            mapped_array.nth(0),
            pd.Series(np.array([10.0, 13.0, 12.0, np.nan]), index=wrapper.columns).rename("nth"),
        )
        assert mapped_array["a"].nth(-1) == 12.0
        assert_series_equal(
            mapped_array.nth(-1),
            pd.Series(np.array([12.0, 13.0, 10.0, np.nan]), index=wrapper.columns).rename("nth"),
        )
        with pytest.raises(Exception):
            mapped_array.nth(10)
        assert_series_equal(
            mapped_array_grouped.nth(0),
            pd.Series(np.array([10.0, 12.0]), index=pd.Index(["g1", "g2"], dtype="object")).rename(
                "nth"
            ),
        )

    def test_nth_index(self):
        assert mapped_array["a"].nth(0) == 10.0
        assert_series_equal(
            mapped_array.nth_index(0),
            pd.Series(
                np.array(["x", "x", "x", np.nan], dtype="object"), index=wrapper.columns
            ).rename("nth_index"),
        )
        assert mapped_array["a"].nth(-1) == 12.0
        assert_series_equal(
            mapped_array.nth_index(-1),
            pd.Series(
                np.array(["z", "z", "z", np.nan], dtype="object"), index=wrapper.columns
            ).rename("nth_index"),
        )
        with pytest.raises(Exception):
            mapped_array.nth_index(10)
        assert_series_equal(
            mapped_array_grouped.nth_index(0),
            pd.Series(
                np.array(["x", "x"], dtype="object"), index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("nth_index"),
        )

    def test_min(self):
        assert mapped_array["a"].min() == mapped_array["a"].to_pd().min()
        assert_series_equal(mapped_array.min(), mapped_array.to_pd().min().rename("min"))
        assert_series_equal(
            mapped_array_grouped.min(),
            pd.Series([10.0, 10.0], index=pd.Index(["g1", "g2"], dtype="object")).rename("min"),
        )

    def test_max(self):
        assert mapped_array["a"].max() == mapped_array["a"].to_pd().max()
        assert_series_equal(mapped_array.max(), mapped_array.to_pd().max().rename("max"))
        assert_series_equal(
            mapped_array_grouped.max(),
            pd.Series([14.0, 12.0], index=pd.Index(["g1", "g2"], dtype="object")).rename("max"),
        )

    def test_mean(self):
        assert mapped_array["a"].mean() == mapped_array["a"].to_pd().mean()
        assert_series_equal(mapped_array.mean(), mapped_array.to_pd().mean().rename("mean"))
        assert_series_equal(
            mapped_array_grouped.mean(),
            pd.Series(
                [12.166666666666666, 11.0], index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("mean"),
        )

    def test_median(self):
        assert mapped_array["a"].median() == mapped_array["a"].to_pd().median()
        assert_series_equal(mapped_array.median(), mapped_array.to_pd().median().rename("median"))
        assert_series_equal(
            mapped_array_grouped.median(),
            pd.Series([12.5, 11.0], index=pd.Index(["g1", "g2"], dtype="object")).rename("median"),
        )

    def test_std(self):
        assert mapped_array["a"].std() == mapped_array["a"].to_pd().std()
        assert_series_equal(mapped_array.std(), mapped_array.to_pd().std().rename("std"))
        assert_series_equal(
            mapped_array.std(ddof=0), mapped_array.to_pd().std(ddof=0).rename("std")
        )
        assert_series_equal(
            mapped_array_grouped.std(),
            pd.Series(
                [1.4719601443879746, 1.0], index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("std"),
        )

    def test_sum(self):
        assert mapped_array["a"].sum() == mapped_array["a"].to_pd().sum()
        assert_series_equal(mapped_array.sum(), mapped_array.to_pd().sum().rename("sum"))
        assert_series_equal(
            mapped_array_grouped.sum(),
            pd.Series([73.0, 33.0], index=pd.Index(["g1", "g2"], dtype="object")).rename("sum"),
        )

    def test_count(self):
        assert mapped_array["a"].count() == mapped_array["a"].to_pd().count()
        assert_series_equal(mapped_array.count(), mapped_array.to_pd().count().rename("count"))
        assert_series_equal(
            mapped_array_grouped.count(),
            pd.Series([6, 3], index=pd.Index(["g1", "g2"], dtype="object")).rename("count"),
        )

    def test_idxmin(self):
        assert mapped_array["a"].idxmin() == mapped_array["a"].to_pd().idxmin()
        assert_series_equal(mapped_array.idxmin(), mapped_array.to_pd().idxmin().rename("idxmin"))
        assert_series_equal(
            mapped_array_grouped.idxmin(),
            pd.Series(
                np.array(["x", "z"], dtype=object), index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("idxmin"),
        )

    def test_idxmax(self):
        assert mapped_array["a"].idxmax() == mapped_array["a"].to_pd().idxmax()
        assert_series_equal(mapped_array.idxmax(), mapped_array.to_pd().idxmax().rename("idxmax"))
        assert_series_equal(
            mapped_array_grouped.idxmax(),
            pd.Series(
                np.array(["y", "x"], dtype=object), index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("idxmax"),
        )

    def test_describe(self):
        assert_series_equal(mapped_array["a"].describe(), mapped_array["a"].to_pd().describe())
        assert_frame_equal(
            mapped_array.describe(percentiles=None),
            mapped_array.to_pd().describe(percentiles=None),
        )
        assert_frame_equal(
            mapped_array.describe(percentiles=[]),
            mapped_array.to_pd().describe(percentiles=[]),
        )
        assert_frame_equal(
            mapped_array.describe(percentiles=np.arange(0, 1, 0.1)),
            mapped_array.to_pd().describe(percentiles=np.arange(0, 1, 0.1)),
        )
        assert_frame_equal(
            mapped_array_grouped.describe(),
            pd.DataFrame(
                np.array(
                    [
                        [6.0, 3.0],
                        [12.166666666666666, 11.0],
                        [1.4719601443879746, 1.0],
                        [10.0, 10.0],
                        [11.25, 10.5],
                        [12.5, 11.0],
                        [13.0, 11.5],
                        [14.0, 12.0],
                    ]
                ),
                columns=pd.Index(["g1", "g2"], dtype="object"),
                index=mapped_array.describe().index,
            ),
        )

    def test_value_counts(self):
        assert_series_equal(
            mapped_array["a"].value_counts(),
            pd.Series(
                np.array([1, 1, 1]), index=pd.Index([10.0, 11.0, 12.0], dtype="float64"), name="a"
            ),
        )
        assert_series_equal(
            mapped_array["a"].value_counts(mapping=mapping),
            pd.Series(
                np.array([1, 1, 1]),
                index=pd.Index(["test_10.0", "test_11.0", "test_12.0"], dtype="object"),
                name="a",
            ),
        )
        assert_frame_equal(
            mapped_array.value_counts(),
            pd.DataFrame(
                np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [0, 2, 0, 0], [0, 1, 0, 0]]),
                index=pd.Index([10.0, 11.0, 12.0, 13.0, 14.0], dtype="float64"),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array.value_counts(jitted=dict(parallel=True)),
            mapped_array.value_counts(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            mapped_array.value_counts(chunked=True), mapped_array.value_counts(chunked=False)
        )
        assert_frame_equal(
            mapped_array.value_counts(axis=0),
            pd.DataFrame(
                np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0]]),
                index=pd.Index([10.0, 11.0, 12.0, 13.0, 14.0], dtype="float64"),
                columns=wrapper.index,
            ),
        )
        assert_series_equal(
            mapped_array.value_counts(axis=-1),
            pd.Series(
                np.array([2, 2, 2, 2, 1]),
                index=pd.Index([10.0, 11.0, 12.0, 13.0, 14.0], dtype="float64"),
                name="value_counts",
            ),
        )
        assert_frame_equal(
            mapped_array_grouped.value_counts(),
            pd.DataFrame(
                np.array([[1, 1], [1, 1], [1, 1], [2, 0], [1, 0]]),
                index=pd.Index([10.0, 11.0, 12.0, 13.0, 14.0], dtype="float64"),
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )
        mapped_array2 = mapped_array.replace(mapped_arr=[4, 4, 3, 2, np.nan, 4, 3, 2, 1])
        assert_frame_equal(
            mapped_array2.value_counts(sort_uniques=False),
            pd.DataFrame(
                np.array([[2, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
                index=pd.Index([4.0, 3.0, 2.0, None, 1.0], dtype="float64"),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array2.value_counts(sort_uniques=True),
            pd.DataFrame(
                np.array([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [2, 1, 0, 0], [0, 1, 0, 0]]),
                index=pd.Index([1.0, 2.0, 3.0, 4.0, None], dtype="float64"),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array2.value_counts(sort=True),
            pd.DataFrame(
                np.array([[2, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]]),
                index=pd.Index([4.0, 2.0, 3.0, 1.0, np.nan], dtype="float64"),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array2.value_counts(sort=True, ascending=True),
            pd.DataFrame(
                np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [2, 1, 0, 0]]),
                index=pd.Index([1.0, np.nan, 2.0, 3.0, 4.0], dtype="float64"),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array2.value_counts(sort=True, normalize=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.2222222222222222, 0.1111111111111111, 0.0, 0.0],
                        [0.0, 0.1111111111111111, 0.1111111111111111, 0.0],
                        [0.1111111111111111, 0.0, 0.1111111111111111, 0.0],
                        [0.0, 0.0, 0.1111111111111111, 0.0],
                        [0.0, 0.1111111111111111, 0.0, 0.0],
                    ]
                ),
                index=pd.Index([4.0, 2.0, 3.0, 1.0, np.nan], dtype="float64"),
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            mapped_array2.value_counts(sort=True, normalize=True, dropna=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.25, 0.125, 0.0, 0.0],
                        [0.0, 0.125, 0.125, 0.0],
                        [0.125, 0.0, 0.125, 0.0],
                        [0.0, 0.0, 0.125, 0.0],
                    ]
                ),
                index=pd.Index([4.0, 2.0, 3.0, 1.0], dtype="float64"),
                columns=wrapper.columns,
            ),
        )

    @pytest.mark.parametrize(
        "test_nosort",
        [False, True],
    )
    def test_indexing(self, test_nosort):
        if test_nosort:
            ma = mapped_array_nosort
            ma_grouped = mapped_array_nosort_grouped
        else:
            ma = mapped_array
            ma_grouped = mapped_array_grouped
        np.testing.assert_array_equal(ma["a"].id_arr, np.array([0, 1, 2]))
        np.testing.assert_array_equal(ma["a"].col_arr, np.array([0, 0, 0]))
        assert_index_equal(ma["a"].wrapper.columns, pd.Index(["a"], dtype="object"))
        np.testing.assert_array_equal(ma["b"].id_arr, np.array([0, 1, 2]))
        np.testing.assert_array_equal(ma["b"].col_arr, np.array([0, 0, 0]))
        assert_index_equal(ma["b"].wrapper.columns, pd.Index(["b"], dtype="object"))
        np.testing.assert_array_equal(ma[["a", "a"]].id_arr, np.array([0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(ma[["a", "a"]].col_arr, np.array([0, 0, 0, 1, 1, 1]))
        assert_index_equal(ma[["a", "a"]].wrapper.columns, pd.Index(["a", "a"], dtype="object"))
        np.testing.assert_array_equal(ma[["a", "b"]].id_arr, np.array([0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(ma[["a", "b"]].col_arr, np.array([0, 0, 0, 1, 1, 1]))
        assert_index_equal(ma[["a", "b"]].wrapper.columns, pd.Index(["a", "b"], dtype="object"))
        assert_index_equal(ma_grouped["g1"].wrapper.columns, pd.Index(["a", "b"], dtype="object"))
        assert ma_grouped["g1"].wrapper.ndim == 2
        assert ma_grouped["g1"].wrapper.grouped_ndim == 1
        assert_index_equal(
            ma_grouped["g1"].wrapper.grouper.group_by, pd.Index(["g1", "g1"], dtype="object")
        )
        assert_index_equal(ma_grouped["g2"].wrapper.columns, pd.Index(["c", "d"], dtype="object"))
        assert ma_grouped["g2"].wrapper.ndim == 2
        assert ma_grouped["g2"].wrapper.grouped_ndim == 1
        assert_index_equal(
            ma_grouped["g2"].wrapper.grouper.group_by, pd.Index(["g2", "g2"], dtype="object")
        )
        assert_index_equal(ma_grouped[["g1"]].wrapper.columns, pd.Index(["a", "b"], dtype="object"))
        assert ma_grouped[["g1"]].wrapper.ndim == 2
        assert ma_grouped[["g1"]].wrapper.grouped_ndim == 2
        assert_index_equal(
            ma_grouped[["g1"]].wrapper.grouper.group_by,
            pd.Index(["g1", "g1"], dtype="object"),
        )
        assert_index_equal(
            ma_grouped[["g1", "g2"]].wrapper.columns,
            pd.Index(["a", "b", "c", "d"], dtype="object"),
        )
        assert ma_grouped[["g1", "g2"]].wrapper.ndim == 2
        assert ma_grouped[["g1", "g2"]].wrapper.grouped_ndim == 2
        assert_index_equal(
            ma_grouped[["g1", "g2"]].wrapper.grouper.group_by,
            pd.Index(["g1", "g1", "g2", "g2"], dtype="object"),
        )

        np.testing.assert_array_equal(ma.loc[["y"]].id_arr, np.array([1, 1, 1]))
        np.testing.assert_array_equal(ma.loc[["y"]].col_arr, np.array([0, 1, 2]))
        np.testing.assert_array_equal(ma.loc[["y"]].idx_arr, np.array([0, 0, 0]))
        assert_index_equal(ma.loc[["y"]].wrapper.index, ma.wrapper.index[[1]])
        assert ma.replace(idx_arr=None).loc[["y"]].idx_arr is None
        if test_nosort:
            np.testing.assert_array_equal(ma.loc["x":"y"].id_arr, np.array([0, 0, 0, 1, 1, 1]))
            np.testing.assert_array_equal(ma.loc["x":"y"].col_arr, np.array([0, 1, 2, 0, 1, 2]))
            np.testing.assert_array_equal(ma.loc["x":"y"].idx_arr, np.array([0, 0, 0, 1, 1, 1]))
            assert_index_equal(ma.loc["x":"y"].wrapper.index, ma.wrapper.index[:2])
            np.testing.assert_array_equal(ma.loc["y":"z"].id_arr, np.array([1, 1, 1, 2, 2, 2]))
            np.testing.assert_array_equal(ma.loc["y":"z"].col_arr, np.array([0, 1, 2, 0, 1, 2]))
            np.testing.assert_array_equal(ma.loc["y":"z"].idx_arr, np.array([0, 0, 0, 1, 1, 1]))
            assert_index_equal(ma.loc["y":"z"].wrapper.index, ma.wrapper.index[1:])
        else:
            np.testing.assert_array_equal(ma.loc["x":"y"].id_arr, np.array([0, 1, 0, 1, 0, 1]))
            np.testing.assert_array_equal(ma.loc["x":"y"].col_arr, np.array([0, 0, 1, 1, 2, 2]))
            np.testing.assert_array_equal(ma.loc["x":"y"].idx_arr, np.array([0, 1, 0, 1, 0, 1]))
            assert_index_equal(ma.loc["x":"y"].wrapper.index, ma.wrapper.index[:2])
            np.testing.assert_array_equal(ma.loc["y":"z"].id_arr, np.array([1, 2, 1, 2, 1, 2]))
            np.testing.assert_array_equal(ma.loc["y":"z"].col_arr, np.array([0, 0, 1, 1, 2, 2]))
            np.testing.assert_array_equal(ma.loc["y":"z"].idx_arr, np.array([0, 1, 0, 1, 0, 1]))
            assert_index_equal(ma.loc["y":"z"].wrapper.index, ma.wrapper.index[1:])
        with pytest.raises(Exception):
            ma.loc[["x", "z"]]

    def test_magic(self):
        a = vbt.MappedArray(
            wrapper,
            records_arr["some_field1"],
            records_arr["col"],
            id_arr=records_arr["id"],
            idx_arr=records_arr["idx"],
        )
        a_inv = vbt.MappedArray(
            wrapper,
            records_arr["some_field1"][::-1],
            records_arr["col"][::-1],
            id_arr=records_arr["id"][::-1],
            idx_arr=records_arr["idx"][::-1],
        )
        b = records_arr["some_field2"]
        a_bool = vbt.MappedArray(
            wrapper,
            records_arr["some_field1"] > np.mean(records_arr["some_field1"]),
            records_arr["col"],
            id_arr=records_arr["id"],
            idx_arr=records_arr["idx"],
        )
        b_bool = records_arr["some_field2"] > np.mean(records_arr["some_field2"])
        assert a**a == a**2
        with pytest.raises(Exception):
            a * a_inv

        # binary ops
        # comparison ops
        np.testing.assert_array_equal((a == b).values, a.values == b)
        np.testing.assert_array_equal((a != b).values, a.values != b)
        np.testing.assert_array_equal((a < b).values, a.values < b)
        np.testing.assert_array_equal((a > b).values, a.values > b)
        np.testing.assert_array_equal((a <= b).values, a.values <= b)
        np.testing.assert_array_equal((a >= b).values, a.values >= b)
        # arithmetic ops
        np.testing.assert_array_equal((a + b).values, a.values + b)
        np.testing.assert_array_equal((a - b).values, a.values - b)
        np.testing.assert_array_equal((a * b).values, a.values * b)
        np.testing.assert_array_equal((a**b).values, a.values**b)
        np.testing.assert_array_equal((a % b).values, a.values % b)
        np.testing.assert_array_equal((a // b).values, a.values // b)
        np.testing.assert_array_equal((a / b).values, a.values / b)
        # __r*__ is only called if the left object does not have an __*__ method
        np.testing.assert_array_equal((10 + a).values, 10 + a.values)
        np.testing.assert_array_equal((10 - a).values, 10 - a.values)
        np.testing.assert_array_equal((10 * a).values, 10 * a.values)
        np.testing.assert_array_equal((10**a).values, 10**a.values)
        np.testing.assert_array_equal((10 % a).values, 10 % a.values)
        np.testing.assert_array_equal((10 // a).values, 10 // a.values)
        np.testing.assert_array_equal((10 / a).values, 10 / a.values)
        # mask ops
        np.testing.assert_array_equal((a_bool & b_bool).values, a_bool.values & b_bool)
        np.testing.assert_array_equal((a_bool | b_bool).values, a_bool.values | b_bool)
        np.testing.assert_array_equal((a_bool ^ b_bool).values, a_bool.values ^ b_bool)
        np.testing.assert_array_equal((True & a_bool).values, True & a_bool.values)
        np.testing.assert_array_equal((True | a_bool).values, True | a_bool.values)
        np.testing.assert_array_equal((True ^ a_bool).values, True ^ a_bool.values)
        # unary ops
        np.testing.assert_array_equal((-a).values, -a.values)
        np.testing.assert_array_equal((+a).values, +a.values)
        np.testing.assert_array_equal((abs(-a)).values, abs(-a.values))

    def test_stats(self):
        stats_index = pd.Index(
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
                "Min Index",
                "Max Index",
            ],
            dtype="object",
        )
        assert_series_equal(
            mapped_array.stats(),
            pd.Series(
                [
                    "x",
                    "z",
                    pd.Timedelta("3 days 00:00:00"),
                    2.25,
                    11.777777777777779,
                    0.859116756396542,
                    11.0,
                    11.666666666666666,
                    12.666666666666666,
                ],
                index=stats_index[:-2],
                name="agg_stats",
            ),
        )
        assert_series_equal(
            mapped_array.stats(column="a"),
            pd.Series(
                [
                    "x",
                    "z",
                    pd.Timedelta("3 days 00:00:00"),
                    3,
                    11.0,
                    1.0,
                    10.0,
                    11.0,
                    12.0,
                    "x",
                    "z",
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            mapped_array.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    "x",
                    "z",
                    pd.Timedelta("3 days 00:00:00"),
                    6,
                    12.166666666666666,
                    1.4719601443879746,
                    10.0,
                    12.5,
                    14.0,
                    "x",
                    "y",
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(mapped_array["c"].stats(), mapped_array.stats(column="c"))
        assert_series_equal(
            mapped_array["c"].stats(), mapped_array.stats(column="c", group_by=False)
        )
        assert_series_equal(
            mapped_array_grouped["g2"].stats(), mapped_array_grouped.stats(column="g2")
        )
        assert_series_equal(
            mapped_array_grouped["g2"].stats(),
            mapped_array.stats(column="g2", group_by=group_by),
        )
        stats_df = mapped_array.stats(agg_func=None)
        assert stats_df.shape == (4, 11)
        assert_index_equal(stats_df.index, mapped_array.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_stats_mapping(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Count",
                "Value Counts: test_10.0",
                "Value Counts: test_11.0",
                "Value Counts: test_12.0",
                "Value Counts: test_13.0",
                "Value Counts: test_14.0",
            ],
            dtype="object",
        )
        assert_series_equal(
            mp_mapped_array.stats(),
            pd.Series(
                ["x", "z", pd.Timedelta("3 days 00:00:00"), 2.25, 0.5, 0.5, 0.5, 0.5, 0.25],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            mp_mapped_array.stats(column="a"),
            pd.Series(
                ["x", "z", pd.Timedelta("3 days 00:00:00"), 3, 1, 1, 1, 0, 0],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            mp_mapped_array.stats(column="g1", group_by=group_by),
            pd.Series(
                ["x", "z", pd.Timedelta("3 days 00:00:00"), 6, 1, 1, 1, 2, 1],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(
            mp_mapped_array.stats(), mapped_array.stats(settings=dict(mapping=mapping))
        )
        assert_series_equal(
            mp_mapped_array["c"].stats(settings=dict(incl_all_keys=True)),
            mp_mapped_array.stats(column="c"),
        )
        assert_series_equal(
            mp_mapped_array["c"].stats(settings=dict(incl_all_keys=True)),
            mp_mapped_array.stats(column="c", group_by=False),
        )
        assert_series_equal(
            mp_mapped_array_grouped["g2"].stats(settings=dict(incl_all_keys=True)),
            mp_mapped_array_grouped.stats(column="g2"),
        )
        assert_series_equal(
            mp_mapped_array_grouped["g2"].stats(settings=dict(incl_all_keys=True)),
            mp_mapped_array.stats(column="g2", group_by=group_by),
        )
        stats_df = mp_mapped_array.stats(agg_func=None)
        assert stats_df.shape == (4, 9)
        assert_index_equal(stats_df.index, mp_mapped_array.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_resample(self):
        dt_mapped_array = mapped_array.replace(
            wrapper=mapped_array.wrapper.replace(
                index=pd.date_range("2020-01-01", periods=len(mapped_array.wrapper.index))
            )
        )
        np.testing.assert_array_equal(
            dt_mapped_array.resample("1h").idx_arr,
            np.array([0, 24, 48, 0, 24, 48, 0, 24, 48]),
        )
        np.testing.assert_array_equal(
            dt_mapped_array.resample("10h").idx_arr,
            np.array([0, 2, 4, 0, 2, 4, 0, 2, 4]),
        )
        np.testing.assert_array_equal(
            dt_mapped_array.resample("3d").idx_arr,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )


# ############# base ############# #


class TestRecords:
    def test_row_stack(self):
        df1 = pd.DataFrame(
            {
                "a": [True, True, False, True],
            },
            index=pd.date_range("2020-01-01", "2020-01-04"),
        )
        df2 = pd.DataFrame(
            {
                "a": [True, True, False, True, False, True],
                "b": [False, False, True, False, True, False],
            },
            index=pd.date_range("2020-01-05", "2020-01-10"),
        )
        records_arr1 = np.array(
            list(
                zip(
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                )
            ),
            dtype=example_dt,
        )
        records_arr2 = np.array(
            list(
                zip(
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    ),
                    np.concatenate((np.full(df2["a"].sum(), 0), np.full(df2["b"].sum(), 1))),
                    np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))),
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    ),
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    )[::-1],
                )
            ),
            dtype=example_dt,
        )
        records1 = vbt.Records(df1.vbt.wrapper, records_arr1)
        records2 = vbt.Records(df2.vbt.wrapper, records_arr2)
        new_records = vbt.Records.row_stack(records1, records2)
        assert_index_equal(new_records.wrapper.index, df1.index.append(df2.index))
        assert_index_equal(new_records.wrapper.columns, df2.columns)
        np.testing.assert_array_equal(
            new_records.id_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).id_arr,
        )
        np.testing.assert_array_equal(
            new_records.col_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).col_arr,
        )
        np.testing.assert_array_equal(
            new_records.idx_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).idx_arr,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field1").values,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).values,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field2").values,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field2"), records2.map_field("some_field2")
            ).values,
        )
        records_arr1 = np.array(
            list(
                zip(
                    np.arange(len(np.flatnonzero(df1["a"]))),
                    np.full(df1["a"].sum(), 0),
                    np.flatnonzero(df1["a"]),
                    np.arange(len(np.flatnonzero(df1["a"]))),
                    np.arange(len(np.flatnonzero(df1["a"])))[::-1],
                )
            ),
            dtype=example_dt,
        )
        records1 = vbt.Records(df1.vbt.wrapper, records_arr1)
        records2 = vbt.Records(df2.vbt.wrapper, records_arr2)
        new_records = vbt.Records.row_stack(records1, records2)
        assert_index_equal(new_records.wrapper.index, df1.index.append(df2.index))
        assert_index_equal(new_records.wrapper.columns, df2.columns)
        np.testing.assert_array_equal(
            new_records.id_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).id_arr,
        )
        np.testing.assert_array_equal(
            new_records.col_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).col_arr,
        )
        np.testing.assert_array_equal(
            new_records.idx_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).idx_arr,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field1").values,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).values,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field2").values,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field2"), records2.map_field("some_field2")
            ).values,
        )

        df1 = pd.DataFrame(
            {
                "a": [True, True, False, True],
                "b": [False, False, True, False],
            },
            index=pd.date_range("2020-01-01", "2020-01-04"),
        )
        df2 = pd.DataFrame(
            {
                "a": [True, True, False, True, False, True],
                "b": [False, False, True, False, True, False],
            },
            index=pd.date_range("2020-01-05", "2020-01-10"),
        )
        records_arr1 = np.array(
            list(
                zip(
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    ),
                    np.concatenate((np.full(df2["a"].sum(), 0), np.full(df2["b"].sum(), 1))),
                    np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))),
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    ),
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    )[::-1],
                )
            ),
            dtype=example_dt,
        )
        records_arr2 = np.array(
            list(
                zip(
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    ),
                    np.concatenate((np.full(df2["a"].sum(), 0), np.full(df2["b"].sum(), 1))),
                    np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))),
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    ),
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["a"]), np.flatnonzero(df2["b"]))))
                    )[::-1],
                )
            ),
            dtype=example_dt,
        )
        records1 = vbt.Records(df1.vbt.wrapper, records_arr1)
        records2 = vbt.Records(df2.vbt.wrapper, records_arr2)
        new_records = vbt.Records.row_stack(records1, records2)
        assert_index_equal(new_records.wrapper.index, df1.index.append(df2.index))
        assert_index_equal(new_records.wrapper.columns, df2.columns)
        np.testing.assert_array_equal(
            new_records.id_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).id_arr,
        )
        np.testing.assert_array_equal(
            new_records.col_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).col_arr,
        )
        np.testing.assert_array_equal(
            new_records.idx_arr,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).idx_arr,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field1").values,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).values,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field2").values,
            vbt.MappedArray.row_stack(
                records1.map_field("some_field2"), records2.map_field("some_field2")
            ).values,
        )
        new_records = vbt.Records.row_stack(
            records1.replace(some_arg=2, check_expected_keys_=False),
            records2.replace(some_arg=2, check_expected_keys_=False),
        )
        assert new_records.config["some_arg"] == 2
        with pytest.raises(Exception):
            vbt.Records.row_stack(
                records1.replace(some_arg=2, check_expected_keys_=False),
                records2.replace(some_arg=3, check_expected_keys_=False),
            )
        with pytest.raises(Exception):
            vbt.Records.row_stack(
                records1.replace(some_arg=2, check_expected_keys_=False),
                records2,
            )
        with pytest.raises(Exception):
            vbt.Records.row_stack(
                records1,
                records2.replace(some_arg=2, check_expected_keys_=False),
            )

    def test_column_stack(self):
        df1 = pd.DataFrame(
            {
                "a": [True, True, False, True, True],
            },
            index=pd.date_range("2020-01-01", "2020-01-05"),
        )
        df2 = pd.DataFrame(
            {
                "b": [True, False, True, False, True],
                "c": [False, True, False, True, False],
            },
            index=pd.date_range("2020-01-03", "2020-01-07"),
        )
        records_arr1 = np.array(
            list(
                zip(
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                )
            ),
            dtype=example_dt,
        )
        records_arr2 = np.array(
            list(
                zip(
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["b"]), np.flatnonzero(df2["c"]))))
                    ),
                    np.concatenate((np.full(df2["b"].sum(), 0), np.full(df2["c"].sum(), 1))),
                    np.concatenate((np.flatnonzero(df2["b"]), np.flatnonzero(df2["c"]))),
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["b"]), np.flatnonzero(df2["c"]))))
                    ),
                    np.arange(
                        len(np.concatenate((np.flatnonzero(df2["b"]), np.flatnonzero(df2["c"]))))
                    )[::-1],
                )
            ),
            dtype=example_dt,
        )
        records1 = vbt.Records(df1.vbt.wrapper, records_arr1)
        records2 = vbt.Records(df2.vbt.wrapper, records_arr2)
        new_records = vbt.Records.column_stack(records1, records2)
        assert_index_equal(new_records.wrapper.index, df1.index.union(df2.index))
        assert_index_equal(new_records.wrapper.columns, df1.columns.append(df2.columns))
        np.testing.assert_array_equal(
            new_records.id_arr,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).id_arr,
        )
        np.testing.assert_array_equal(
            new_records.col_arr,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).col_arr,
        )
        np.testing.assert_array_equal(
            new_records.idx_arr,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).idx_arr,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field1").values,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).values,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field2").values,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field2"), records2.map_field("some_field2")
            ).values,
        )
        records_arr1 = np.array(
            list(
                zip(
                    np.arange(len(np.flatnonzero(df1["a"]))),
                    np.full(df1["a"].sum(), 0),
                    np.flatnonzero(df1["a"]),
                    np.arange(len(np.flatnonzero(df1["a"]))),
                    np.arange(len(np.flatnonzero(df1["a"])))[::-1],
                )
            ),
            dtype=example_dt,
        )
        records1 = vbt.Records(df1.vbt.wrapper, records_arr1)
        records2 = vbt.Records(df2.vbt.wrapper, records_arr2)
        new_records = vbt.Records.column_stack(records1, records2)
        assert_index_equal(new_records.wrapper.index, df1.index.union(df2.index))
        assert_index_equal(new_records.wrapper.columns, df1.columns.append(df2.columns))
        np.testing.assert_array_equal(
            new_records.id_arr,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).id_arr,
        )
        np.testing.assert_array_equal(
            new_records.col_arr,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).col_arr,
        )
        np.testing.assert_array_equal(
            new_records.idx_arr,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).idx_arr,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field1").values,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field1"), records2.map_field("some_field1")
            ).values,
        )
        np.testing.assert_array_equal(
            new_records.map_field("some_field2").values,
            vbt.MappedArray.column_stack(
                records1.map_field("some_field2"), records2.map_field("some_field2")
            ).values,
        )

    def test_config(self, tmp_path):
        assert vbt.Records.loads(records["a"].dumps()) == records["a"]
        assert vbt.Records.loads(records.dumps()) == records
        records.save(tmp_path / "records")
        assert vbt.Records.load(tmp_path / "records") == records
        records.save(tmp_path / "records", file_format="ini")
        assert vbt.Records.load(tmp_path / "records", file_format="ini") == records

    def test_field_config(self):
        records2 = vbt.records.Records(wrapper, records_arr)
        records2.field_config["settings"]["id"]["title"] = "My id"
        assert Records.field_config["settings"]["id"]["title"] == "Id"
        records2_copy = records2.copy()
        records2_copy.field_config["settings"]["id"]["title"] = "My id 2"
        assert records2.field_config["settings"]["id"]["title"] == "My id"
        assert Records.field_config["settings"]["id"]["title"] == "Id"

    def test_metrics_config(self):
        records2 = vbt.records.Records(wrapper, records_arr)
        records2.metrics["hello"] = "world"
        assert "hello" not in Records.metrics
        records2_copy = records2.copy()
        records2_copy.metrics["hello"] = "world2"
        assert records2.metrics["hello"] == "world"
        assert "hello" not in Records.metrics

    def test_subplots_config(self):
        records2 = vbt.records.Records(wrapper, records_arr)
        records2.subplots["hello"] = "world"
        assert "hello" not in Records.subplots
        records2_copy = records2.copy()
        records2_copy.subplots["hello"] = "world2"
        assert records2.subplots["hello"] == "world"
        assert "hello" not in Records.subplots

    def test_records(self):
        assert_frame_equal(records.records, pd.DataFrame.from_records(records_arr))

    def test_recarray(self):
        np.testing.assert_array_equal(
            records["a"].recarray.some_field1, records["a"].values["some_field1"]
        )
        np.testing.assert_array_equal(records.recarray.some_field1, records.values["some_field1"])

    def test_records_readable(self):
        assert_frame_equal(
            records.records_readable,
            pd.DataFrame(
                [
                    [0, "a", "x", 10.0, 21.0],
                    [1, "a", "y", 11.0, 20.0],
                    [2, "a", "z", 12.0, 19.0],
                    [0, "b", "x", 13.0, 18.0],
                    [1, "b", "y", 14.0, 17.0],
                    [2, "b", "z", 13.0, 18.0],
                    [0, "c", "x", 12.0, 19.0],
                    [1, "c", "y", 11.0, 20.0],
                    [2, "c", "z", 10.0, 21.0],
                ],
                columns=pd.Index(
                    ["Id", "Column", "Index", "some_field1", "some_field2"], dtype="object"
                ),
            ),
        )

    def test_is_sorted(self):
        assert records.is_sorted()
        assert records.is_sorted(incl_id=True)
        assert not records_nosort.is_sorted()
        assert not records_nosort.is_sorted(incl_id=True)

    def test_sort(self):
        assert records.sort().is_sorted()
        assert records.sort().is_sorted(incl_id=True)
        assert records.sort(incl_id=True).is_sorted(incl_id=True)
        assert records_nosort.sort().is_sorted()
        assert records_nosort.sort().is_sorted(incl_id=True)
        assert records_nosort.sort(incl_id=True).is_sorted(incl_id=True)

    def test_apply_mask(self):
        mask_a = records["a"].values["some_field1"] >= records["a"].values["some_field1"].mean()
        assert_records_close(
            records["a"].apply_mask(mask_a).values,
            np.array([(1, 0, 1, 11.0, 20.0), (2, 0, 2, 12.0, 19.0)], dtype=example_dt),
        )
        mask = records.values["some_field1"] >= records.values["some_field1"].mean()
        filtered = records.apply_mask(mask)
        assert_records_close(
            filtered.values,
            np.array(
                [
                    (2, 0, 2, 12.0, 19.0),
                    (0, 1, 0, 13.0, 18.0),
                    (1, 1, 1, 14.0, 17.0),
                    (2, 1, 2, 13.0, 18.0),
                    (0, 2, 0, 12.0, 19.0),
                ],
                dtype=example_dt,
            ),
        )
        assert records_grouped.apply_mask(mask).wrapper == records_grouped.wrapper

    def test_map_field(self):
        np.testing.assert_array_equal(
            records["a"].map_field("some_field1").values, np.array([10.0, 11.0, 12.0])
        )
        np.testing.assert_array_equal(
            records.map_field("some_field1").values,
            np.array([10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0]),
        )
        assert (
            records_grouped.map_field("some_field1").wrapper
            == records.map_field("some_field1", group_by=group_by).wrapper
        )
        assert (
            records_grouped.map_field("some_field1", group_by=False).wrapper.grouper.group_by
            is None
        )

    def test_map(self):
        @njit
        def map_func_nb(record):
            return record["some_field1"] + record["some_field2"]

        @njit
        def map_func_meta_nb(ridx, records):
            return records[ridx]["some_field1"] + records[ridx]["some_field2"]

        np.testing.assert_array_equal(
            records["a"].map(map_func_nb).values, np.array([31.0, 31.0, 31.0])
        )
        np.testing.assert_array_equal(
            records.map(map_func_nb).values,
            np.array([31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0]),
        )
        assert (
            records_grouped.map(map_func_nb).wrapper
            == records.map(map_func_nb, group_by=group_by).wrapper
        )
        assert records_grouped.map(map_func_nb, group_by=False).wrapper.grouper.group_by is None
        np.testing.assert_array_equal(
            records.map(map_func_nb, jitted=dict(parallel=True)).values,
            records.map(map_func_nb, jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            records.map(map_func_nb, chunked=True).values,
            records.map(map_func_nb, chunked=False).values,
        )
        np.testing.assert_array_equal(
            type(records)
            .map(map_func_meta_nb, records.values, col_mapper=records.col_mapper)
            .values,
            records.map(map_func_nb).values,
        )
        np.testing.assert_array_equal(
            type(records)
            .map(
                map_func_meta_nb,
                records.values,
                col_mapper=records.col_mapper,
                jitted=dict(parallel=True),
            )
            .values,
            type(records)
            .map(
                map_func_meta_nb,
                records.values,
                col_mapper=records.col_mapper,
                jitted=dict(parallel=False),
            )
            .values,
        )
        count_chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0),
                )
            )
        )
        np.testing.assert_array_equal(
            type(records)
            .map(
                map_func_meta_nb,
                records.values,
                col_mapper=records.col_mapper,
                chunked=count_chunked,
            )
            .values,
            type(records)
            .map(map_func_meta_nb, records.values, col_mapper=records.col_mapper, chunked=False)
            .values,
        )

    def test_map_array(self):
        arr = records_arr["some_field1"] + records_arr["some_field2"]
        np.testing.assert_array_equal(
            records["a"].map_array(arr[:3]).values, np.array([31.0, 31.0, 31.0])
        )
        np.testing.assert_array_equal(
            records.map_array(arr).values,
            np.array([31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0]),
        )
        assert (
            records_grouped.map_array(arr).wrapper
            == records.map_array(arr, group_by=group_by).wrapper
        )
        assert records_grouped.map_array(arr, group_by=False).wrapper.grouper.group_by is None

    def test_apply(self):
        @njit
        def cumsum_apply_nb(records):
            return np.cumsum(records["some_field1"])

        @njit
        def cumsum_apply_meta_nb(ridxs, col, records):
            return np.cumsum(records[ridxs]["some_field1"])

        np.testing.assert_array_equal(
            records["a"].apply(cumsum_apply_nb).values, np.array([10.0, 21.0, 33.0])
        )
        np.testing.assert_array_equal(
            records.apply(cumsum_apply_nb).values,
            np.array([10.0, 21.0, 33.0, 13.0, 27.0, 40.0, 12.0, 23.0, 33.0]),
        )
        np.testing.assert_array_equal(
            records_grouped.apply(cumsum_apply_nb, apply_per_group=False).values,
            np.array([10.0, 21.0, 33.0, 13.0, 27.0, 40.0, 12.0, 23.0, 33.0]),
        )
        np.testing.assert_array_equal(
            records_grouped.apply(cumsum_apply_nb, apply_per_group=True).values,
            np.array([10.0, 21.0, 33.0, 46.0, 60.0, 73.0, 12.0, 23.0, 33.0]),
        )
        assert (
            records_grouped.apply(cumsum_apply_nb).wrapper
            == records.apply(cumsum_apply_nb, group_by=group_by).wrapper
        )
        assert (
            records_grouped.apply(cumsum_apply_nb, group_by=False).wrapper.grouper.group_by is None
        )
        np.testing.assert_array_equal(
            records.apply(cumsum_apply_nb, jitted=dict(parallel=True)).values,
            records.apply(cumsum_apply_nb, jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            records.apply(cumsum_apply_nb, chunked=True).values,
            records.apply(cumsum_apply_nb, chunked=False).values,
        )
        np.testing.assert_array_equal(
            type(records)
            .apply(cumsum_apply_meta_nb, records.values, col_mapper=records.col_mapper)
            .values,
            records.apply(cumsum_apply_nb).values,
        )
        np.testing.assert_array_equal(
            type(records)
            .apply(
                cumsum_apply_meta_nb,
                records.values,
                col_mapper=records.col_mapper,
                jitted=dict(parallel=True),
            )
            .values,
            type(records)
            .apply(
                cumsum_apply_meta_nb,
                records.values,
                col_mapper=records.col_mapper,
                jitted=dict(parallel=False),
            )
            .values,
        )
        chunked = dict(
            arg_take_spec=dict(
                args=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0, mapper=vbt.GroupIdxsMapper(arg_query="col_map")),
                )
            )
        )
        np.testing.assert_array_equal(
            type(records)
            .apply(
                cumsum_apply_meta_nb, records.values, col_mapper=records.col_mapper, chunked=chunked
            )
            .values,
            type(records)
            .apply(
                cumsum_apply_meta_nb, records.values, col_mapper=records.col_mapper, chunked=False
            )
            .values,
        )

    def test_count(self):
        assert records["a"].count() == 3
        assert_series_equal(
            records.count(),
            pd.Series(np.array([3, 3, 3, 0]), index=wrapper.columns).rename("count"),
        )
        assert records_grouped["g1"].count() == 6
        assert_series_equal(
            records_grouped.count(),
            pd.Series(np.array([6, 3]), index=pd.Index(["g1", "g2"], dtype="object")).rename(
                "count"
            ),
        )

    def test_has_conflicts(self):
        assert not records.has_conflicts()

    def test_coverage_map(self):
        assert_frame_equal(
            records.coverage_map(),
            pd.DataFrame(
                [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
                index=wrapper.index,
                columns=wrapper.columns,
            ),
        )
        assert_frame_equal(
            records.coverage_map(group_by=group_by),
            pd.DataFrame(
                [[2, 1], [2, 1], [2, 1]],
                index=wrapper.index,
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )

    @pytest.mark.parametrize(
        "test_nosort",
        [False, True],
    )
    def test_indexing(self, test_nosort):
        if test_nosort:
            r = records_nosort
            r_grouped = records_nosort_grouped
        else:
            r = records
            r_grouped = records_grouped
        assert_records_close(
            r["a"].values,
            np.array(
                [(0, 0, 0, 10.0, 21.0), (1, 0, 1, 11.0, 20.0), (2, 0, 2, 12.0, 19.0)],
                dtype=example_dt,
            ),
        )
        assert_index_equal(r["a"].wrapper.columns, pd.Index(["a"], dtype="object"))
        assert_index_equal(r["b"].wrapper.columns, pd.Index(["b"], dtype="object"))
        assert_records_close(
            r[["a", "a"]].values,
            np.array(
                [
                    (0, 0, 0, 10.0, 21.0),
                    (1, 0, 1, 11.0, 20.0),
                    (2, 0, 2, 12.0, 19.0),
                    (0, 1, 0, 10.0, 21.0),
                    (1, 1, 1, 11.0, 20.0),
                    (2, 1, 2, 12.0, 19.0),
                ],
                dtype=example_dt,
            ),
        )
        assert_index_equal(r[["a", "a"]].wrapper.columns, pd.Index(["a", "a"], dtype="object"))
        assert_records_close(
            r[["a", "b"]].values,
            np.array(
                [
                    (0, 0, 0, 10.0, 21.0),
                    (1, 0, 1, 11.0, 20.0),
                    (2, 0, 2, 12.0, 19.0),
                    (0, 1, 0, 13.0, 18.0),
                    (1, 1, 1, 14.0, 17.0),
                    (2, 1, 2, 13.0, 18.0),
                ],
                dtype=example_dt,
            ),
        )
        assert_index_equal(r[["a", "b"]].wrapper.columns, pd.Index(["a", "b"], dtype="object"))
        assert_index_equal(r_grouped["g1"].wrapper.columns, pd.Index(["a", "b"], dtype="object"))
        assert r_grouped["g1"].wrapper.ndim == 2
        assert r_grouped["g1"].wrapper.grouped_ndim == 1
        assert_index_equal(
            r_grouped["g1"].wrapper.grouper.group_by, pd.Index(["g1", "g1"], dtype="object")
        )
        assert_index_equal(r_grouped["g2"].wrapper.columns, pd.Index(["c", "d"], dtype="object"))
        assert r_grouped["g2"].wrapper.ndim == 2
        assert r_grouped["g2"].wrapper.grouped_ndim == 1
        assert_index_equal(
            r_grouped["g2"].wrapper.grouper.group_by, pd.Index(["g2", "g2"], dtype="object")
        )
        assert_index_equal(r_grouped[["g1"]].wrapper.columns, pd.Index(["a", "b"], dtype="object"))
        assert r_grouped[["g1"]].wrapper.ndim == 2
        assert r_grouped[["g1"]].wrapper.grouped_ndim == 2
        assert_index_equal(
            r_grouped[["g1"]].wrapper.grouper.group_by,
            pd.Index(["g1", "g1"], dtype="object"),
        )
        assert_index_equal(
            r_grouped[["g1", "g2"]].wrapper.columns,
            pd.Index(["a", "b", "c", "d"], dtype="object"),
        )
        assert r_grouped[["g1", "g2"]].wrapper.ndim == 2
        assert r_grouped[["g1", "g2"]].wrapper.grouped_ndim == 2
        assert_index_equal(
            r_grouped[["g1", "g2"]].wrapper.grouper.group_by,
            pd.Index(["g1", "g1", "g2", "g2"], dtype="object"),
        )

        assert_records_close(
            r.loc[["y"]].values,
            np.array(
                [
                    (1, 0, 0, 11, 20),
                    (1, 1, 0, 14, 17),
                    (1, 2, 0, 11, 20),
                ],
                dtype=example_dt,
            ),
        )
        if test_nosort:
            assert_records_close(
                r.loc["x":"y"].values,
                np.array(
                    [
                        (0, 0, 0, 10.0, 21.0),
                        (0, 1, 0, 13.0, 18.0),
                        (0, 2, 0, 12.0, 19.0),
                        (1, 0, 1, 11.0, 20.0),
                        (1, 1, 1, 14.0, 17.0),
                        (1, 2, 1, 11.0, 20.0),
                    ],
                    dtype=example_dt,
                ),
            )
            assert_records_close(
                r.loc["y":"z"].values,
                np.array(
                    [
                        (1, 0, 0, 11.0, 20.0),
                        (1, 1, 0, 14.0, 17.0),
                        (1, 2, 0, 11.0, 20.0),
                        (2, 0, 1, 12.0, 19.0),
                        (2, 1, 1, 13.0, 18.0),
                        (2, 2, 1, 10.0, 21.0),
                    ],
                    dtype=example_dt,
                ),
            )
        else:
            assert_records_close(
                r.loc["x":"y"].values,
                np.array(
                    [
                        (0, 0, 0, 10.0, 21.0),
                        (1, 0, 1, 11.0, 20.0),
                        (0, 1, 0, 13.0, 18.0),
                        (1, 1, 1, 14.0, 17.0),
                        (0, 2, 0, 12.0, 19.0),
                        (1, 2, 1, 11.0, 20.0),
                    ],
                    dtype=example_dt,
                ),
            )
            assert_records_close(
                r.loc["y":"z"].values,
                np.array(
                    [
                        (1, 0, 0, 11.0, 20.0),
                        (2, 0, 1, 12.0, 19.0),
                        (1, 1, 0, 14.0, 17.0),
                        (2, 1, 1, 13.0, 18.0),
                        (1, 2, 0, 11.0, 20.0),
                        (2, 2, 1, 10.0, 21.0),
                    ],
                    dtype=example_dt,
                ),
            )
        with pytest.raises(Exception):
            r.loc[["x", "z"]]

    def test_filtering(self):
        filtered_records = vbt.Records(wrapper, records_arr[[0, -1]])
        assert_records_close(
            filtered_records.values,
            np.array([(0, 0, 0, 10.0, 21.0), (2, 2, 2, 10.0, 21.0)], dtype=example_dt),
        )
        # a
        assert_records_close(
            filtered_records["a"].values, np.array([(0, 0, 0, 10.0, 21.0)], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records["a"].map_field("some_field1").id_arr, np.array([0])
        )
        assert filtered_records["a"].map_field("some_field1").min() == 10.0
        assert filtered_records["a"].count() == 1.0
        # b
        assert_records_close(filtered_records["b"].values, np.array([], dtype=example_dt))
        np.testing.assert_array_equal(
            filtered_records["b"].map_field("some_field1").id_arr, np.array([])
        )
        assert np.isnan(filtered_records["b"].map_field("some_field1").min())
        assert filtered_records["b"].count() == 0.0
        # c
        assert_records_close(
            filtered_records["c"].values, np.array([(2, 0, 2, 10.0, 21.0)], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records["c"].map_field("some_field1").id_arr, np.array([2])
        )
        assert filtered_records["c"].map_field("some_field1").min() == 10.0
        assert filtered_records["c"].count() == 1.0
        # d
        assert_records_close(filtered_records["d"].values, np.array([], dtype=example_dt))
        np.testing.assert_array_equal(
            filtered_records["d"].map_field("some_field1").id_arr, np.array([])
        )
        assert np.isnan(filtered_records["d"].map_field("some_field1").min())
        assert filtered_records["d"].count() == 0.0

    def test_stats(self):
        stats_index = pd.Index(
            ["Start Index", "End Index", "Total Duration", "Count"], dtype="object"
        )
        assert_series_equal(
            records.stats(),
            pd.Series(
                ["x", "z", pd.Timedelta("3 days 00:00:00"), 2.25],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            records.stats(column="a"),
            pd.Series(["x", "z", pd.Timedelta("3 days 00:00:00"), 3], index=stats_index, name="a"),
        )
        assert_series_equal(
            records.stats(column="g1", group_by=group_by),
            pd.Series(["x", "z", pd.Timedelta("3 days 00:00:00"), 6], index=stats_index, name="g1"),
        )
        assert_series_equal(records["c"].stats(), records.stats(column="c"))
        assert_series_equal(records["c"].stats(), records.stats(column="c", group_by=False))
        assert_series_equal(records_grouped["g2"].stats(), records_grouped.stats(column="g2"))
        assert_series_equal(
            records_grouped["g2"].stats(), records.stats(column="g2", group_by=group_by)
        )
        stats_df = records.stats(agg_func=None)
        assert stats_df.shape == (4, 4)
        assert_index_equal(stats_df.index, records.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_resample(self):
        dt_records = records.replace(
            wrapper=records.wrapper.replace(
                index=pd.date_range("2020-01-01", periods=len(records.wrapper.index))
            )
        )
        np.testing.assert_array_equal(
            dt_records.resample("1h").idx_arr,
            np.array([0, 24, 48, 0, 24, 48, 0, 24, 48]),
        )
        np.testing.assert_array_equal(
            dt_records.resample("10h").idx_arr,
            np.array([0, 2, 4, 0, 2, 4, 0, 2, 4]),
        )
        np.testing.assert_array_equal(
            dt_records.resample("3d").idx_arr,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )


# ############# ranges ############# #

ts = pd.DataFrame(
    {
        "a": [1, -1, 3, -1, 5, -1],
        "b": [-1, -1, -1, 4, 5, 6],
        "c": [1, 2, 3, -1, -1, -1],
        "d": [-1, -1, -1, -1, -1, -1],
    },
    index=[
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-04"),
        pd.Timestamp("2020-01-05"),
        pd.Timestamp("2020-01-07"),
        pd.Timestamp("2020-01-08"),
    ],
)

ranges = vbt.Ranges.from_array(ts, wrapper_kwargs=dict(freq="1 days"))
ranges_grouped = vbt.Ranges.from_array(ts, wrapper_kwargs=dict(freq="1 days", group_by=group_by))


class TestRanges:
    def test_row_stack(self):
        ts2 = ts * 2
        ts2.index = pd.date_range("2020-01-09", "2020-01-14")
        ranges1 = vbt.Ranges.from_array(ts, wrapper_kwargs=dict(freq="1 days"))
        ranges2 = vbt.Ranges.from_array(ts2, wrapper_kwargs=dict(freq="1 days"))
        new_ranges = vbt.Ranges.row_stack(ranges1, ranges2)
        assert_frame_equal(new_ranges.close, pd.concat((ts, ts2)))
        with pytest.raises(Exception):
            vbt.Ranges.row_stack(ranges1.replace(close=None), ranges2)
        with pytest.raises(Exception):
            vbt.Ranges.row_stack(ranges1, ranges2.replace(close=None))
        new_ranges = vbt.Ranges.row_stack(ranges1.replace(close=None), ranges2.replace(close=None))
        assert new_ranges.close is None

    def test_column_stack(self):
        ts2 = ts * 2
        ts2.columns = ["e", "f", "g", "h"]
        ranges1 = vbt.Ranges.from_array(ts, wrapper_kwargs=dict(freq="1 days"))
        ranges2 = vbt.Ranges.from_array(ts2, wrapper_kwargs=dict(freq="1 days"))
        new_ranges = vbt.Ranges.column_stack(ranges1, ranges2)
        assert_frame_equal(new_ranges.close, pd.concat((ts, ts2), axis=1))
        with pytest.raises(Exception):
            vbt.Ranges.column_stack(ranges1.replace(close=None), ranges2)
        with pytest.raises(Exception):
            vbt.Ranges.column_stack(ranges1, ranges2.replace(close=None))
        new_ranges = vbt.Ranges.column_stack(
            ranges1.replace(close=None), ranges2.replace(close=None)
        )
        assert new_ranges.close is None

    def test_indexing(self):
        ranges2 = ranges.loc["2020-01-02":"2020-01-05", ["a", "c"]]
        assert_index_equal(
            ranges2.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-02", "2020-01-04", "2020-01-05"], dtype="datetime64[ns]", freq=None
            ),
        )
        assert_index_equal(ranges2.wrapper.columns, ranges.wrapper.columns[[0, 2]])
        assert_frame_equal(ranges2.close, ranges.close.loc["2020-01-02":"2020-01-05", ["a", "c"]])
        assert_records_close(
            ranges2.values,
            np.array([(1, 0, 1, 2, 1)], dtype=range_dt),
        )

    def test_mapped_fields(self):
        for name in range_dt.names:
            np.testing.assert_array_equal(getattr(ranges, name).values, ranges.values[name])

    def test_from_array(self):
        assert_records_close(
            ranges.values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 2, 3, 1),
                    (2, 0, 4, 5, 1),
                    (0, 1, 3, 5, 0),
                    (0, 2, 0, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert ranges.wrapper.freq == day_dt
        assert_index_equal(ranges_grouped.wrapper.grouper.group_by, group_by)
        assert_records_close(
            vbt.Ranges.from_array(ts, jitted=dict(parallel=True)).values,
            vbt.Ranges.from_array(ts, jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            vbt.Ranges.from_array(ts, chunked=True).values,
            vbt.Ranges.from_array(ts, chunked=False).values,
        )

    def test_from_delta(self):
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=2).values,
            np.array(
                [
                    (0, 0, 1, 3, 1),
                    (1, 0, 3, 5, 1),
                    (2, 0, 5, 5, 0),
                    (0, 1, 5, 5, 0),
                    (0, 2, 3, 5, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=3).values,
            np.array(
                [
                    (0, 0, 1, 4, 1),
                    (1, 0, 3, 5, 0),
                    (2, 0, 5, 5, 0),
                    (0, 1, 5, 5, 0),
                    (0, 2, 3, 5, 0),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=4).values,
            np.array(
                [
                    (0, 0, 1, 5, 1),
                    (1, 0, 3, 5, 0),
                    (2, 0, 5, 5, 0),
                    (0, 1, 5, 5, 0),
                    (0, 2, 3, 5, 0),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta="2d").values,
            np.array(
                [
                    (0, 0, 1, 2, 1),
                    (1, 0, 3, 4, 1),
                    (2, 0, 5, 5, 0),
                    (0, 1, 5, 5, 0),
                    (0, 2, 3, 4, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta="3d").values,
            np.array(
                [
                    (0, 0, 1, 3, 1),
                    (1, 0, 3, 5, 1),
                    (2, 0, 5, 5, 0),
                    (0, 1, 5, 5, 0),
                    (0, 2, 3, 5, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta="4d").values,
            np.array(
                [
                    (0, 0, 1, 4, 1),
                    (1, 0, 3, 5, 0),
                    (2, 0, 5, 5, 0),
                    (0, 1, 5, 5, 0),
                    (0, 2, 3, 5, 0),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=-2).values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 1, 3, 1),
                    (2, 0, 3, 5, 1),
                    (0, 1, 3, 5, 1),
                    (0, 2, 1, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=-3).values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 0, 3, 1),
                    (2, 0, 2, 5, 1),
                    (0, 1, 2, 5, 1),
                    (0, 2, 0, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=-4).values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 0, 3, 1),
                    (2, 0, 1, 5, 1),
                    (0, 1, 1, 5, 1),
                    (0, 2, 0, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta="-2d").values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 1, 3, 1),
                    (2, 0, 3, 5, 1),
                    (0, 1, 3, 5, 1),
                    (0, 2, 1, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta="-3d").values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 1, 3, 1),
                    (2, 0, 3, 5, 1),
                    (0, 1, 3, 5, 1),
                    (0, 2, 1, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta="-4d").values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 0, 3, 1),
                    (2, 0, 2, 5, 1),
                    (0, 1, 2, 5, 1),
                    (0, 2, 0, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=2, idx_field_or_arr="start_idx").values,
            np.array(
                [
                    (0, 0, 0, 2, 1),
                    (1, 0, 2, 4, 1),
                    (2, 0, 4, 5, 0),
                    (0, 1, 3, 5, 1),
                    (0, 2, 0, 2, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=2, idx_field_or_arr="start_idx").values,
            vbt.Ranges.from_delta(ranges, delta=2, idx_field_or_arr=ranges.start_idx.values).values,
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges.start_idx, delta=2).values,
            vbt.Ranges.from_delta(ranges, delta=2).values,
        )
        assert_records_close(
            vbt.Ranges.from_delta(
                ranges.map_field("start_idx", idx_arr=ranges.start_idx.values), delta=2
            ).values,
            vbt.Ranges.from_delta(
                ranges.start_idx, delta=2, idx_field_or_arr=ranges.start_idx.values
            ).values,
        )
        assert vbt.Ranges.from_delta(ranges, delta=2).open is None
        assert vbt.Ranges.from_delta(ranges, delta=2).high is None
        assert vbt.Ranges.from_delta(ranges, delta=2).low is None
        assert_frame_equal(vbt.Ranges.from_delta(ranges, delta=2).close, ts)
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=2, jitted=dict(parallel=True)).values,
            vbt.Ranges.from_delta(ranges, delta=2, jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            vbt.Ranges.from_delta(ranges, delta=2, chunked=True).values,
            vbt.Ranges.from_delta(ranges, delta=2, chunked=False).values,
        )

    def test_records_readable(self):
        records_readable = ranges.records_readable

        np.testing.assert_array_equal(
            records_readable["Range Id"].values, np.array([0, 1, 2, 0, 0])
        )
        np.testing.assert_array_equal(
            records_readable["Column"].values, np.array(["a", "a", "a", "b", "c"])
        )
        np.testing.assert_array_equal(
            records_readable["Start Index"].values,
            np.array(
                [
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["End Index"].values,
            np.array(
                [
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Status"].values,
            np.array(["Closed", "Closed", "Closed", "Open", "Closed"]),
        )

    def test_first_pd_mask(self):
        assert_series_equal(
            ranges["a"].first_pd_mask,
            pd.Series([True, False, True, False, True, False], index=ts.index, name="a"),
        )
        assert_frame_equal(
            ranges.first_pd_mask,
            pd.DataFrame(
                [
                    [True, False, True, False],
                    [False, False, False, False],
                    [True, False, False, False],
                    [False, True, False, False],
                    [True, False, False, False],
                    [False, False, False, False],
                ],
                index=ts.index,
                columns=ts.columns,
            ),
        )
        assert_frame_equal(
            ranges_grouped.first_pd_mask,
            pd.DataFrame(
                [
                    [True, True],
                    [False, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, False],
                ],
                index=ts.index,
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )

    def test_last_pd_mask(self):
        assert_series_equal(
            ranges["a"].last_pd_mask,
            pd.Series([True, False, True, False, True, False], index=ts.index, name="a"),
        )
        assert_frame_equal(
            ranges.last_pd_mask,
            pd.DataFrame(
                [
                    [True, False, False, False],
                    [False, False, False, False],
                    [True, False, True, False],
                    [False, False, False, False],
                    [True, False, False, False],
                    [False, True, False, False],
                ],
                index=ts.index,
                columns=ts.columns,
            ),
        )
        assert_frame_equal(
            ranges_grouped.last_pd_mask,
            pd.DataFrame(
                [
                    [True, False],
                    [False, False],
                    [True, True],
                    [False, False],
                    [True, False],
                    [True, False],
                ],
                index=ts.index,
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )

    def test_ranges_pd_mask(self):
        assert_series_equal(ranges["a"].ranges_pd_mask, ts["a"] != -1)
        assert_frame_equal(ranges.ranges_pd_mask, ts != -1)
        assert_frame_equal(
            ranges_grouped.ranges_pd_mask,
            pd.DataFrame(
                [
                    [True, True],
                    [False, True],
                    [True, True],
                    [True, False],
                    [True, False],
                    [True, False],
                ],
                index=ts.index,
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )
        assert_frame_equal(
            ranges.get_ranges_pd_mask(jitted=dict(parallel=True)),
            ranges.get_ranges_pd_mask(jitted=dict(parallel=False)),
        )
        assert_frame_equal(
            ranges.get_ranges_pd_mask(chunked=True), ranges.get_ranges_pd_mask(chunked=False)
        )

    def test_duration(self):
        np.testing.assert_array_equal(ranges["a"].duration.values, np.array([1, 1, 1]))
        np.testing.assert_array_equal(ranges.duration.values, np.array([1, 1, 1, 3, 3]))
        np.testing.assert_array_equal(
            ranges.get_duration(jitted=dict(parallel=True)).values,
            ranges.get_duration(jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            ranges.get_duration(chunked=True).values,
            ranges.get_duration(chunked=False).values,
        )

    def test_real_duration(self):
        np.testing.assert_array_equal(
            ranges["a"].real_duration.values,
            np.array([86400000000000, 86400000000000, 86400000000000], dtype="timedelta64[ns]"),
        )
        np.testing.assert_array_equal(
            ranges.real_duration.values,
            np.array(
                [
                    86400000000000,
                    86400000000000,
                    86400000000000,
                    86400000000000 * 4,
                    86400000000000 * 4,
                ],
                dtype="timedelta64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            ranges.get_real_duration(jitted=dict(parallel=True)).values,
            ranges.get_real_duration(jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            ranges.get_real_duration(chunked=True).values,
            ranges.get_real_duration(chunked=False).values,
        )

    def test_avg_duration(self):
        assert ranges["a"].avg_duration == pd.Timedelta("1 days 00:00:00")
        assert_series_equal(
            ranges.avg_duration,
            pd.Series(
                np.array(
                    [86400000000000, 259200000000000, 259200000000000, "NaT"],
                    dtype="timedelta64[ns]",
                ),
                index=wrapper.columns,
            ).rename("avg_duration"),
        )
        assert_series_equal(
            ranges_grouped.avg_duration,
            pd.Series(
                np.array([129600000000000, 259200000000000], dtype="timedelta64[ns]"),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("avg_duration"),
        )
        assert_series_equal(
            ranges_grouped.get_avg_duration(real=True),
            pd.Series(
                np.array([151200000000000, 345600000000000], dtype="timedelta64[ns]"),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("avg_real_duration"),
        )
        assert_series_equal(
            ranges.get_avg_duration(jitted=dict(parallel=True)),
            ranges.get_avg_duration(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ranges.get_avg_duration(chunked=True), ranges.get_avg_duration(chunked=False)
        )

    def test_filter_min_duration(self):
        assert_records_close(
            ranges.filter_min_duration(1).values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 2, 3, 1),
                    (2, 0, 4, 5, 1),
                    (0, 1, 3, 5, 0),
                    (0, 2, 0, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_min_duration(2).values,
            np.array(
                [(0, 1, 3, 5, 0), (0, 2, 0, 3, 1)],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_min_duration(4).values,
            np.array(
                [],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_min_duration(1).values,
            ranges.filter_min_duration("1d").values,
        )
        assert_records_close(
            ranges.filter_min_duration(2).values,
            ranges.filter_min_duration("2d").values,
        )
        assert_records_close(
            ranges.filter_min_duration(4).values,
            ranges.filter_min_duration("4d").values,
        )
        assert_records_close(
            ranges.filter_min_duration("1d", real=True).values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 2, 3, 1),
                    (2, 0, 4, 5, 1),
                    (0, 1, 3, 5, 0),
                    (0, 2, 0, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_min_duration("2d", real=True).values,
            np.array(
                [(0, 1, 3, 5, 0), (0, 2, 0, 3, 1)],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_min_duration("4d", real=True).values,
            np.array(
                [(0, 1, 3, 5, 0), (0, 2, 0, 3, 1)],
                dtype=range_dt,
            ),
        )

    def test_filter_max_duration(self):
        assert_records_close(
            ranges.filter_max_duration(3).values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 2, 3, 1),
                    (2, 0, 4, 5, 1),
                    (0, 1, 3, 5, 0),
                    (0, 2, 0, 3, 1),
                ],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_max_duration(1).values,
            np.array(
                [(0, 0, 0, 1, 1), (1, 0, 2, 3, 1), (2, 0, 4, 5, 1)],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_max_duration(0).values,
            np.array(
                [],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_max_duration(3).values,
            ranges.filter_max_duration("3d").values,
        )
        assert_records_close(
            ranges.filter_max_duration(1).values,
            ranges.filter_max_duration("1d").values,
        )
        assert_records_close(
            ranges.filter_max_duration(0).values,
            ranges.filter_max_duration("0d").values,
        )
        assert_records_close(
            ranges.filter_max_duration("3d", real=True).values,
            np.array(
                [(0, 0, 0, 1, 1), (1, 0, 2, 3, 1), (2, 0, 4, 5, 1)],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_max_duration("1d", real=True).values,
            np.array(
                [(0, 0, 0, 1, 1), (1, 0, 2, 3, 1), (2, 0, 4, 5, 1)],
                dtype=range_dt,
            ),
        )
        assert_records_close(
            ranges.filter_max_duration("0d", real=True).values,
            np.array(
                [],
                dtype=range_dt,
            ),
        )

    def test_get_projections(self):
        assert_frame_equal(
            ranges.get_projections(),
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [-1.0, -0.3333333333333333, -0.2, 1.25, 2.0],
                    [np.nan, np.nan, np.nan, 1.5, 3.0],
                    [np.nan, np.nan, np.nan, np.nan, -1.0],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11"],
                    dtype="datetime64[ns]",
                    freq="D",
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        assert_frame_equal(
            ranges.get_projections(proj_start=1),
            pd.DataFrame(
                [[1.0, 1.0], [1.2, 1.5], [np.nan, -0.5]],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10"], dtype="datetime64[ns]", freq="D"
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        assert_frame_equal(
            ranges.get_projections(proj_start="1d"),
            pd.DataFrame(
                [[1.0, 1.0], [1.2, 1.5], [np.nan, -0.5]],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10"], dtype="datetime64[ns]", freq="D"
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        assert_frame_equal(
            ranges.get_projections(proj_period=10),
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [-1.0, -0.3333333333333333, -0.2, 1.25, 2.0],
                    [np.nan, np.nan, np.nan, 1.5, 3.0],
                    [np.nan, np.nan, np.nan, np.nan, -1.0],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11"],
                    dtype="datetime64[ns]",
                    freq="D",
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        assert_frame_equal(
            ranges.get_projections(proj_period=3, extend=True),
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [-1.0, -0.3333333333333333, -0.2, 1.25, 2.0],
                    [3.0, 1.6666666666666665, np.nan, 1.5, 3.0],
                    [-1.0, -0.3333333333333333, np.nan, np.nan, -1.0],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11"],
                    dtype="datetime64[ns]",
                    freq="D",
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        assert_frame_equal(
            ranges.get_projections(proj_period="3d", extend=True),
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [-1.0, -0.3333333333333333, -0.2, 1.25, 2.0],
                    [3.0, 1.6666666666666665, np.nan, 1.5, 3.0],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10"], dtype="datetime64[ns]", freq="D"
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        assert_frame_equal(
            ranges.get_projections(rebase=False),
            pd.DataFrame(
                [
                    [1.0, 3.0, 5.0, 4.0, 1.0],
                    [-1.0, -1.0, -1.0, 5.0, 2.0],
                    [np.nan, np.nan, np.nan, 6.0, 3.0],
                    [np.nan, np.nan, np.nan, np.nan, -1.0],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11"],
                    dtype="datetime64[ns]",
                    freq="D",
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        assert_frame_equal(
            ranges.get_projections(proj_period=3, extend=True, ffill=True),
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [-1.0, -0.3333333333333333, -0.2, 1.25, 2.0],
                    [3.0, 1.6666666666666665, -0.2, 1.5, 3.0],
                    [-1.0, -0.3333333333333333, -0.2, 1.5, -1.0],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11"],
                    dtype="datetime64[ns]",
                    freq="D",
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        assert_frame_equal(
            ranges.get_projections(remove_empty=False, proj_start=1),
            pd.DataFrame(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [np.nan, np.nan, np.nan, 1.2, 1.5],
                    [np.nan, np.nan, np.nan, np.nan, -0.5],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-08", "2020-01-09", "2020-01-10"], dtype="datetime64[ns]", freq="D"
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("b", 0),
                        ("c", 0),
                    ],
                    names=[None, "range_id"],
                ),
            ),
        )
        np.testing.assert_array_equal(
            ranges.get_projections(return_raw=True)[0],
            np.array([0, 1, 2, 3, 4]),
        )
        np.testing.assert_array_equal(
            ranges.get_projections(return_raw=True)[1],
            ranges.get_projections().values.T,
        )

    def test_max_duration(self):
        assert ranges["a"].max_duration == pd.Timedelta("1 days 00:00:00")
        assert_series_equal(
            ranges.max_duration,
            pd.Series(
                np.array(
                    [86400000000000, 259200000000000, 259200000000000, "NaT"],
                    dtype="timedelta64[ns]",
                ),
                index=wrapper.columns,
            ).rename("max_duration"),
        )
        assert_series_equal(
            ranges_grouped.max_duration,
            pd.Series(
                np.array([259200000000000, 259200000000000], dtype="timedelta64[ns]"),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("max_duration"),
        )
        assert_series_equal(
            ranges_grouped.get_max_duration(real=True),
            pd.Series(
                np.array([345600000000000, 345600000000000], dtype="timedelta64[ns]"),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("max_real_duration"),
        )
        assert_series_equal(
            ranges.get_max_duration(jitted=dict(parallel=True)),
            ranges.get_max_duration(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            ranges.get_max_duration(chunked=True), ranges.get_max_duration(chunked=False)
        )

    def test_coverage(self):
        assert ranges["a"].coverage == 0.5
        assert_series_equal(
            ranges.coverage,
            pd.Series(np.array([0.5, 0.5, 0.5, np.nan]), index=ts2.columns).rename("coverage"),
        )
        assert_series_equal(
            ranges.coverage,
            ranges.replace(records_arr=np.repeat(ranges.values, 2)).coverage,
        )
        assert_series_equal(
            ranges.replace(records_arr=np.repeat(ranges.values, 2)).get_coverage(overlapping=True),
            pd.Series(np.array([1.0, 1.0, 1.0, np.nan]), index=ts2.columns).rename("coverage"),
        )
        assert_series_equal(
            ranges.get_coverage(normalize=False),
            pd.Series(np.array([3.0, 3.0, 3.0, np.nan]), index=ts2.columns).rename("coverage"),
        )
        assert_series_equal(
            ranges.replace(records_arr=np.repeat(ranges.values, 2)).get_coverage(
                overlapping=True, normalize=False
            ),
            pd.Series(np.array([3.0, 3.0, 3.0, np.nan]), index=ts2.columns).rename("coverage"),
        )
        assert_series_equal(
            ranges_grouped.coverage,
            pd.Series(
                np.array([0.4166666666666667, 0.25]), index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("coverage"),
        )
        assert_series_equal(
            ranges_grouped.coverage,
            ranges_grouped.replace(records_arr=np.repeat(ranges_grouped.values, 2)).coverage,
        )
        assert_series_equal(
            ranges.get_coverage(jitted=dict(parallel=True)),
            ranges.get_coverage(jitted=dict(parallel=False)),
        )
        assert_series_equal(ranges.get_coverage(chunked=True), ranges.get_coverage(chunked=False))

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Records",
                "Coverage",
                "Overlap Coverage",
                "Duration: Min",
                "Duration: Median",
                "Duration: Max",
            ],
            dtype="object",
        )
        assert_series_equal(
            ranges.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    1.25,
                    0.5,
                    0.0,
                    pd.Timedelta("2 days 08:00:00"),
                    pd.Timedelta("2 days 08:00:00"),
                    pd.Timedelta("2 days 08:00:00"),
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            ranges.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    3,
                    0.5,
                    0.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            ranges.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    4,
                    0.4166666666666667,
                    0.2,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(ranges["c"].stats(), ranges.stats(column="c"))
        assert_series_equal(ranges["c"].stats(), ranges.stats(column="c", group_by=False))
        assert_series_equal(ranges_grouped["g2"].stats(), ranges_grouped.stats(column="g2"))
        assert_series_equal(
            ranges_grouped["g2"].stats(), ranges.stats(column="g2", group_by=group_by)
        )
        stats_df = ranges.stats(agg_func=None)
        assert stats_df.shape == (4, 9)
        assert_index_equal(stats_df.index, ranges.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_resample(self):
        np.testing.assert_array_equal(
            ranges.resample("1h").start_idx.values,
            np.array([0, 72, 144, 96, 0]),
        )
        np.testing.assert_array_equal(
            ranges.resample("1h").end_idx.values,
            np.array([24, 96, 168, 168, 96]),
        )
        np.testing.assert_array_equal(
            ranges.resample("10h").start_idx.values,
            np.array([0, 7, 14, 9, 0]),
        )
        np.testing.assert_array_equal(
            ranges.resample("10h").end_idx.values,
            np.array([2, 9, 16, 16, 9]),
        )
        np.testing.assert_array_equal(
            ranges.resample("3d").start_idx.values,
            np.array([0, 1, 2, 1, 0]),
        )
        np.testing.assert_array_equal(
            ranges.resample("3d").end_idx.values,
            np.array([0, 1, 2, 2, 1]),
        )
        assert_frame_equal(
            ranges.resample("1h").close,
            ranges.close.resample("1h").last().astype(float_),
        )
        assert_frame_equal(
            ranges.resample("10h").close,
            ranges.close.resample("10h").last().astype(float_),
        )
        assert_frame_equal(
            ranges.resample("3d").close,
            ranges.close.resample("3d").last().astype(float_),
        )
        assert_frame_equal(
            ranges.resample("1h", ffill_close=True).close,
            ranges.close.resample("1h").last().ffill().astype(float_),
        )
        assert_frame_equal(
            ranges.resample("10h", ffill_close=True).close,
            ranges.close.resample("10h").last().ffill().astype(float_),
        )
        assert_frame_equal(
            ranges.resample("3d", ffill_close=True).close,
            ranges.close.resample("3d").last().ffill().astype(float_),
        )
        assert_frame_equal(
            ranges.resample("1h", fbfill_close=True).close,
            ranges.close.resample("1h").last().ffill().bfill().astype(float_),
        )
        assert_frame_equal(
            ranges.resample("10h", fbfill_close=True).close,
            ranges.close.resample("10h").last().ffill().bfill().astype(float_),
        )
        assert_frame_equal(
            ranges.resample("3d", fbfill_close=True).close,
            ranges.close.resample("3d").last().ffill().bfill().astype(float_),
        )


ts2 = pd.DataFrame(
    {
        "a": [2, 1, 3, 1, 4, 1],
        "b": [1, 2, 1, 3, 1, 4],
        "c": [1, 2, 3, 2, 1, 2],
        "d": [1, 2, 3, 4, 5, 6],
    },
    index=pd.date_range("2020", periods=6),
)

pattern_ranges = vbt.PatternRanges.from_pattern_search(
    ts2,
    [1, 2, 1],
    interp_mode="linear",
    min_similarity=0,
    overlap_mode="allowall",
)
pattern_ranges_grouped = vbt.PatternRanges.from_pattern_search(
    ts2,
    [1, 2, 1],
    interp_mode="linear",
    min_similarity=0,
    overlap_mode="allowall",
    wrapper_kwargs=dict(group_by=group_by),
)


class TestPatternRanges:
    def test_row_stack(self):
        sr = pd.Series([1, 2, 3, 2, 3, 4, 3], index=pd.date_range("2020-01-01", periods=7))
        df = pd.DataFrame(
            {"a": [1, 2, 3, 2, 3, 4, 3], "b": [4, 3, 2, 3, 2, 1, 2]},
            index=pd.date_range("2020-01-08", periods=7),
        )
        pattern_ranges1 = vbt.PatternRanges.from_pattern_search(
            sr, [1, 2, 1], interp_mode="linear", min_similarity=0, overlap_mode="allowall"
        )
        pattern_ranges2 = vbt.PatternRanges.from_pattern_search(
            df, [1, 2, 1], interp_mode="linear", min_similarity=0, overlap_mode="allowall"
        )
        new_pattern_ranges = vbt.PatternRanges.row_stack(pattern_ranges1, pattern_ranges2)
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.5),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.0),
                    (3, 0, 3, 6, 1, 0.5),
                    (4, 0, 4, 6, 0, 1.0),
                    (5, 0, 7, 10, 1, 0.5),
                    (6, 0, 8, 11, 1, 1.0),
                    (7, 0, 9, 12, 1, 0.0),
                    (8, 0, 10, 13, 1, 0.5),
                    (9, 0, 11, 13, 0, 1.0),
                    (0, 1, 0, 3, 1, 0.5),
                    (1, 1, 1, 4, 1, 1.0),
                    (2, 1, 2, 5, 1, 0.0),
                    (3, 1, 3, 6, 1, 0.5),
                    (4, 1, 4, 6, 0, 1.0),
                    (5, 1, 7, 10, 1, 0.5),
                    (6, 1, 8, 11, 1, 0.0),
                    (7, 1, 9, 12, 1, 1.0),
                    (8, 1, 10, 13, 1, 0.5),
                    (9, 1, 11, 13, 0, 0.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_frame_equal(
            new_pattern_ranges.close, pd.concat((sr.vbt.tile(2, keys=["a", "b"]), df))
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                min_similarity=0,
                overlap_mode="allowall",
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                min_similarity=0,
                overlap_mode="allowall",
            ),
        ]
        pattern_ranges3 = vbt.PatternRanges.from_pattern_search(
            df, [1, 2, 1], interp_mode="linear", min_similarity=0.5, overlap_mode="allowall"
        )
        with pytest.raises(Exception):
            vbt.PatternRanges.row_stack(pattern_ranges1, pattern_ranges3)

    def test_column_stack(self):
        sr = pd.Series(
            [1, 2, 3, 2, 3, 4, 3], index=pd.date_range("2020-01-01", periods=7), name="a"
        )
        df = pd.DataFrame(
            {"b": [1, 2, 3, 2, 3, 4, 3], "c": [4, 3, 2, 3, 2, 1, 2]},
            index=pd.date_range("2020-01-03", periods=7),
        )
        pattern_ranges1 = vbt.PatternRanges.from_pattern_search(
            sr, [1, 2, 1], interp_mode="linear", min_similarity=0, overlap_mode="allowall"
        )
        pattern_ranges2 = vbt.PatternRanges.from_pattern_search(
            df, [1, 2, 1], interp_mode="linear", min_similarity=0.5, overlap_mode="allowall"
        )
        new_pattern_ranges = vbt.PatternRanges.column_stack(pattern_ranges1, pattern_ranges2)
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.5),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.0),
                    (3, 0, 3, 6, 1, 0.5),
                    (4, 0, 4, 6, 0, 1.0),
                    (0, 1, 2, 5, 1, 0.5),
                    (1, 1, 3, 6, 1, 1.0),
                    (2, 1, 5, 8, 1, 0.5),
                    (3, 1, 6, 8, 0, 1.0),
                    (0, 2, 2, 5, 1, 0.5),
                    (1, 2, 4, 7, 1, 1.0),
                    (2, 2, 5, 8, 1, 0.5),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_frame_equal(new_pattern_ranges.close, pd.concat((sr, df), axis=1))
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                min_similarity=0,
                overlap_mode="allowall",
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                min_similarity=0.5,
                overlap_mode="allowall",
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                min_similarity=0.5,
                overlap_mode="allowall",
            ),
        ]

    def test_indexing(self):
        new_pattern_ranges = pattern_ranges.loc["2020-01-02":"2020-01-05", ["a", "c"]]
        assert_index_equal(
            new_pattern_ranges.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        assert_index_equal(
            new_pattern_ranges.wrapper.columns, pattern_ranges.wrapper.columns[[0, 2]]
        )
        assert_frame_equal(
            new_pattern_ranges.close,
            pattern_ranges.close.loc["2020-01-02":"2020-01-05", ["a", "c"]],
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array([(1, 0, 0, 3, 1, 1.0), (1, 1, 0, 3, 1, 1.0)], dtype=pattern_range_dt),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                min_similarity=0,
                overlap_mode="allowall",
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                min_similarity=0,
                overlap_mode="allowall",
            ),
        ]

    def test_from_pattern_search(self):
        assert_records_close(
            vbt.PatternRanges.from_pattern_search(
                ts2,
                [1, 2, 1],
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ).values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 0, 3, 1, 1.0),
                    (1, 1, 1, 4, 1, 0.16666666666666663),
                    (2, 1, 2, 5, 1, 1.0),
                    (3, 1, 3, 5, 0, 0.11111111111111116),
                    (0, 2, 0, 3, 1, 0.5),
                    (1, 2, 1, 4, 1, 1.0),
                    (2, 2, 2, 5, 1, 0.5),
                    (3, 2, 3, 5, 0, 0.0),
                    (0, 3, 0, 3, 1, 0.5),
                    (1, 3, 1, 4, 1, 0.5),
                    (2, 3, 2, 5, 1, 0.5),
                    (3, 3, 3, 5, 0, 0.5),
                ],
                dtype=pattern_range_dt,
            ),
        )
        assert_records_close(
            vbt.PatternRanges.from_pattern_search(
                ts2,
                [1, 2, 1],
                window=2,
                max_window=4,
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ).values,
            np.array(
                [
                    (0, 0, 0, 2, 1, 0.5),
                    (1, 0, 0, 3, 1, 0.16666666666666663),
                    (2, 0, 0, 4, 1, 0.55),
                    (3, 0, 1, 3, 1, 0.5),
                    (4, 0, 1, 4, 1, 1.0),
                    (5, 0, 1, 5, 1, 0.4999999999999999),
                    (6, 0, 2, 4, 1, 0.5),
                    (7, 0, 2, 5, 1, 0.11111111111111116),
                    (8, 0, 2, 5, 0, 0.5),
                    (9, 0, 3, 5, 1, 0.5),
                    (10, 0, 3, 5, 0, 1.0),
                    (11, 0, 4, 5, 0, 0.5),
                    (0, 1, 0, 2, 1, 0.5),
                    (1, 1, 0, 3, 1, 1.0),
                    (2, 1, 0, 4, 1, 0.44999999999999996),
                    (3, 1, 1, 3, 1, 0.5),
                    (4, 1, 1, 4, 1, 0.16666666666666663),
                    (5, 1, 1, 5, 1, 0.55),
                    (6, 1, 2, 4, 1, 0.5),
                    (7, 1, 2, 5, 1, 1.0),
                    (8, 1, 2, 5, 0, 0.4999999999999999),
                    (9, 1, 3, 5, 1, 0.5),
                    (10, 1, 3, 5, 0, 0.11111111111111116),
                    (11, 1, 4, 5, 0, 0.5),
                    (0, 2, 0, 2, 1, 0.5),
                    (1, 2, 0, 3, 1, 0.5),
                    (2, 2, 0, 4, 1, 0.7000000000000001),
                    (3, 2, 1, 3, 1, 0.5),
                    (4, 2, 1, 4, 1, 1.0),
                    (5, 2, 1, 5, 1, 0.7),
                    (6, 2, 2, 4, 1, 0.5),
                    (7, 2, 2, 5, 1, 0.5),
                    (8, 2, 2, 5, 0, 0.30000000000000004),
                    (9, 2, 3, 5, 1, 0.5),
                    (10, 2, 3, 5, 0, 0.0),
                    (11, 2, 4, 5, 0, 0.5),
                    (0, 3, 0, 2, 1, 0.5),
                    (1, 3, 0, 3, 1, 0.5),
                    (2, 3, 0, 4, 1, 0.5999999999999999),
                    (3, 3, 1, 3, 1, 0.5),
                    (4, 3, 1, 4, 1, 0.5),
                    (5, 3, 1, 5, 1, 0.5999999999999999),
                    (6, 3, 2, 4, 1, 0.5),
                    (7, 3, 2, 5, 1, 0.5),
                    (8, 3, 2, 5, 0, 0.5999999999999999),
                    (9, 3, 3, 5, 1, 0.5),
                    (10, 3, 3, 5, 0, 0.5),
                    (11, 3, 4, 5, 0, 0.5),
                ],
                dtype=pattern_range_dt,
            ),
        )
        assert_records_close(
            vbt.PatternRanges.from_pattern_search(
                ts2,
                [1, 2, 1],
                window=2,
                max_window=4,
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
                window_select_prob=0.5,
                seed=42,
            ).values,
            np.array(
                [
                    (0, 0, 1, 3, 1, 0.5),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 1, 0.5),
                    (4, 0, 3, 5, 0, 1.0),
                    (5, 0, 4, 5, 0, 0.5),
                    (0, 1, 0, 2, 1, 0.5),
                    (1, 1, 0, 3, 1, 1.0),
                    (2, 1, 1, 3, 1, 0.5),
                    (3, 1, 1, 4, 1, 0.16666666666666663),
                    (4, 1, 1, 5, 1, 0.55),
                    (5, 1, 2, 4, 1, 0.5),
                    (6, 1, 3, 5, 0, 0.11111111111111116),
                    (0, 2, 0, 3, 1, 0.5),
                    (1, 2, 0, 4, 1, 0.7000000000000001),
                    (2, 2, 1, 3, 1, 0.5),
                    (3, 2, 1, 4, 1, 1.0),
                    (4, 2, 1, 5, 1, 0.7),
                    (5, 2, 2, 5, 1, 0.5),
                    (0, 3, 1, 3, 1, 0.5),
                    (1, 3, 1, 4, 1, 0.5),
                    (2, 3, 1, 5, 1, 0.5999999999999999),
                    (3, 3, 2, 4, 1, 0.5),
                    (4, 3, 2, 5, 1, 0.5),
                    (5, 3, 3, 5, 1, 0.5),
                ],
                dtype=pattern_range_dt,
            ),
        )
        assert_records_close(
            vbt.PatternRanges.from_pattern_search(
                ts2,
                [1, 2, 1],
                window=2,
                max_window=4,
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
                row_select_prob=0.5,
                seed=42,
            ).values,
            np.array(
                [
                    (0, 0, 0, 2, 1, 0.5),
                    (1, 0, 0, 3, 1, 0.16666666666666663),
                    (2, 0, 0, 4, 1, 0.55),
                    (3, 0, 1, 3, 1, 0.5),
                    (4, 0, 1, 4, 1, 1.0),
                    (5, 0, 1, 5, 1, 0.4999999999999999),
                    (6, 0, 4, 5, 0, 0.5),
                    (0, 1, 1, 3, 1, 0.5),
                    (1, 1, 1, 4, 1, 0.16666666666666663),
                    (2, 1, 1, 5, 1, 0.55),
                    (3, 1, 3, 5, 1, 0.5),
                    (4, 1, 3, 5, 0, 0.11111111111111116),
                    (5, 1, 4, 5, 0, 0.5),
                    (0, 2, 0, 2, 1, 0.5),
                    (1, 2, 0, 3, 1, 0.5),
                    (2, 2, 0, 4, 1, 0.7000000000000001),
                    (3, 2, 3, 5, 1, 0.5),
                    (4, 2, 3, 5, 0, 0.0),
                    (5, 2, 4, 5, 0, 0.5),
                    (0, 3, 2, 4, 1, 0.5),
                    (1, 3, 2, 5, 1, 0.5),
                    (2, 3, 2, 5, 0, 0.5999999999999999),
                    (3, 3, 3, 5, 1, 0.5),
                    (4, 3, 3, 5, 0, 0.5),
                ],
                dtype=pattern_range_dt,
            ),
        )
        assert_records_close(
            vbt.PatternRanges.from_pattern_search(
                ts2,
                [1, 2, 1],
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
                jitted=dict(parallel=True),
            ).values,
            vbt.PatternRanges.from_pattern_search(
                ts2,
                [1, 2, 1],
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
                jitted=dict(parallel=True),
            ).values,
        )

        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2["a"],
            [1, 2, 1],
            interp_mode="linear",
            roll_forward=True,
            min_similarity=0,
            overlap_mode="allowall",
            max_records=20,
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.Index(["a"], dtype="object"),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            )
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2["a"],
            [1, 2, 1],
            interp_mode="linear",
            roll_forward=True,
            min_similarity=vbt.Param(0),
            overlap_mode="allowall",
            max_records=20,
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.Index([0], dtype="int64", name="min_similarity"),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            )
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2["a"],
            [1, 2, 1],
            interp_mode="linear",
            roll_forward=True,
            min_similarity=vbt.Param([0, 0.5]),
            overlap_mode="allowall",
            max_records=20,
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 1, 4, 1, 1.0),
                    (1, 1, 3, 5, 0, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.Index([0.0, 0.5], dtype="float64", name="min_similarity"),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2[["a", "b"]],
            [1, 2, 1],
            interp_mode="linear",
            roll_forward=True,
            min_similarity=vbt.Param(0),
            overlap_mode="allowall",
            max_records=20,
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 0, 3, 1, 1.0),
                    (1, 1, 1, 4, 1, 0.16666666666666663),
                    (2, 1, 2, 5, 1, 1.0),
                    (3, 1, 3, 5, 0, 0.11111111111111116),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.MultiIndex.from_tuples([(0, "a"), (0, "b")], names=["min_similarity", None]),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2[["a", "b"]],
            [1, 2, 1],
            interp_mode="linear",
            roll_forward=True,
            min_similarity=vbt.Param([0, 0.5]),
            overlap_mode="allowall",
            max_records=20,
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 0, 3, 1, 1.0),
                    (1, 1, 1, 4, 1, 0.16666666666666663),
                    (2, 1, 2, 5, 1, 1.0),
                    (3, 1, 3, 5, 0, 0.11111111111111116),
                    (0, 2, 1, 4, 1, 1.0),
                    (1, 2, 3, 5, 0, 1.0),
                    (0, 3, 0, 3, 1, 1.0),
                    (1, 3, 2, 5, 1, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (0.5, "a"), (0.5, "b")], names=["min_similarity", None]
            ),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2["a"],
            search_configs=[
                vbt.PSC(
                    pattern=[1, 2, 1],
                    interp_mode="linear",
                    roll_forward=True,
                    min_similarity=0,
                    overlap_mode="allowall",
                    max_records=20,
                )
            ],
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.Index(["a"], dtype="object"),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            )
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2["a"],
            search_configs=[
                [
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0,
                        overlap_mode="allowall",
                        max_records=20,
                    )
                ]
            ],
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.Index(["a"], dtype="object"),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            )
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2["a"],
            search_configs=[
                [
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0,
                        overlap_mode="allowall",
                        max_records=20,
                    )
                ],
                [
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0.5,
                        overlap_mode="allowall",
                        max_records=20,
                    )
                ],
            ],
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 1, 4, 1, 1.0),
                    (1, 1, 3, 5, 0, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.Index([0, 1], dtype="int64", name="search_config"),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2[["a", "b"]],
            search_configs=[
                vbt.PSC(
                    pattern=[1, 2, 1],
                    interp_mode="linear",
                    roll_forward=True,
                    min_similarity=0,
                    overlap_mode="allowall",
                    max_records=20,
                )
            ],
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 0, 3, 1, 1.0),
                    (1, 1, 1, 4, 1, 0.16666666666666663),
                    (2, 1, 2, 5, 1, 1.0),
                    (3, 1, 3, 5, 0, 0.11111111111111116),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.Index(["a", "b"], dtype="object"),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2[["a", "b"]],
            search_configs=[
                vbt.PSC(
                    pattern=[1, 2, 1],
                    interp_mode="linear",
                    roll_forward=True,
                    min_similarity=0,
                    overlap_mode="allowall",
                    max_records=20,
                ),
                vbt.PSC(
                    pattern=[1, 2, 1],
                    interp_mode="linear",
                    roll_forward=True,
                    min_similarity=0.5,
                    overlap_mode="allowall",
                    max_records=20,
                ),
            ],
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 0, 3, 1, 1.0),
                    (1, 1, 1, 4, 1, 0.16666666666666663),
                    (2, 1, 2, 5, 1, 1.0),
                    (3, 1, 3, 5, 0, 0.11111111111111116),
                    (0, 2, 1, 4, 1, 1.0),
                    (1, 2, 3, 5, 0, 1.0),
                    (0, 3, 0, 3, 1, 1.0),
                    (1, 3, 2, 5, 1, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (0, "a"),
                    (0, "b"),
                    (1, "a"),
                    (1, "b"),
                ],
                names=["search_config", None],
            ),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2[["a", "b"]],
            search_configs=[
                [
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0,
                        overlap_mode="allowall",
                        max_records=20,
                    ),
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0.5,
                        overlap_mode="allowall",
                        max_records=20,
                    ),
                ]
            ],
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 0, 3, 1, 1.0),
                    (1, 1, 2, 5, 1, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.Index(["a", "b"], dtype="object"),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
        ]
        new_pattern_ranges = vbt.PatternRanges.from_pattern_search(
            ts2[["a", "b"]],
            search_configs=[
                [
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0,
                        overlap_mode="allowall",
                        max_records=20,
                    ),
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0.5,
                        overlap_mode="allowall",
                        max_records=20,
                    ),
                ],
                [
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0,
                        overlap_mode="allowall",
                        max_records=20,
                    ),
                    vbt.PSC(
                        pattern=[1, 2, 1],
                        interp_mode="linear",
                        roll_forward=True,
                        min_similarity=0.5,
                        overlap_mode="allowall",
                        max_records=20,
                    ),
                ],
            ],
        )
        assert_records_close(
            new_pattern_ranges.values,
            np.array(
                [
                    (0, 0, 0, 3, 1, 0.16666666666666663),
                    (1, 0, 1, 4, 1, 1.0),
                    (2, 0, 2, 5, 1, 0.11111111111111116),
                    (3, 0, 3, 5, 0, 1.0),
                    (0, 1, 0, 3, 1, 1.0),
                    (1, 1, 2, 5, 1, 1.0),
                    (0, 2, 0, 3, 1, 0.16666666666666663),
                    (1, 2, 1, 4, 1, 1.0),
                    (2, 2, 2, 5, 1, 0.11111111111111116),
                    (3, 2, 3, 5, 0, 1.0),
                    (0, 3, 0, 3, 1, 1.0),
                    (1, 3, 2, 5, 1, 1.0),
                ],
                dtype=pattern_range_dt,
            ),
        )
        pd.testing.assert_index_equal(
            new_pattern_ranges.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (0, "a"),
                    (1, "b"),
                    (2, "a"),
                    (3, "b"),
                ],
                names=["search_config", None],
            ),
        )
        assert new_pattern_ranges.search_configs == [
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0,
                overlap_mode="allowall",
                max_records=20,
            ),
            vbt.PatternRanges.resolve_search_config(
                pattern=np.array([1, 2, 1]),
                interp_mode="linear",
                roll_forward=True,
                min_similarity=0.5,
                overlap_mode="allowall",
                max_records=20,
            ),
        ]

    def test_records_readable(self):
        records_readable = pattern_ranges.records_readable

        np.testing.assert_array_equal(
            records_readable["Pattern Range Id"].values,
            np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        )
        np.testing.assert_array_equal(
            records_readable["Column"].values,
            np.array(
                ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "d", "d", "d", "d"]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Start Index"].values,
            np.array(
                [
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["End Index"].values,
            np.array(
                [
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Status"].values,
            np.array(
                [
                    "Closed",
                    "Closed",
                    "Closed",
                    "Open",
                    "Closed",
                    "Closed",
                    "Closed",
                    "Open",
                    "Closed",
                    "Closed",
                    "Closed",
                    "Open",
                    "Closed",
                    "Closed",
                    "Closed",
                    "Open",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Similarity"].values,
            np.array(
                [
                    0.16666666666666663,
                    1.0,
                    0.11111111111111116,
                    1.0,
                    1.0,
                    0.16666666666666663,
                    1.0,
                    0.11111111111111116,
                    0.5,
                    1.0,
                    0.5,
                    0.0,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                ]
            ),
        )

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Records",
                "Coverage",
                "Overlap Coverage",
                "Duration: Min",
                "Duration: Median",
                "Duration: Max",
                "Similarity: Min",
                "Similarity: Median",
                "Similarity: Max",
            ],
            dtype="object",
        )
        assert_series_equal(
            pattern_ranges.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-06 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    4.0,
                    1.0,
                    0.6666666666666666,
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    0.18055555555555558,
                    0.5416666666666666,
                    0.875,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            pattern_ranges.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-06 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    4,
                    1.0,
                    0.6666666666666666,
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    0.11111111111111116,
                    0.5833333333333333,
                    1.0,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            pattern_ranges.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-06 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    8,
                    0.5,
                    1.0,
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    pd.Timedelta("3 days 00:00:00"),
                    0.11111111111111116,
                    0.5833333333333333,
                    1.0,
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(pattern_ranges["c"].stats(), pattern_ranges.stats(column="c"))
        assert_series_equal(
            pattern_ranges["c"].stats(), pattern_ranges.stats(column="c", group_by=False)
        )
        assert_series_equal(
            pattern_ranges_grouped["g2"].stats(), pattern_ranges_grouped.stats(column="g2")
        )
        assert_series_equal(
            pattern_ranges_grouped["g2"].stats(),
            pattern_ranges.stats(column="g2", group_by=group_by),
        )
        stats_df = pattern_ranges.stats(agg_func=None)
        assert stats_df.shape == (4, 12)
        assert_index_equal(stats_df.index, pattern_ranges.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)


# ############# drawdowns ############# #

drawdowns = vbt.Drawdowns.from_price(ts2, wrapper_kwargs=dict(freq="1 days"))
drawdowns_grouped = vbt.Drawdowns.from_price(
    ts2, wrapper_kwargs=dict(freq="1 days", group_by=group_by)
)


class TestDrawdowns:
    def test_indexing(self):
        drawdowns2 = drawdowns.loc["2020-01-02":"2020-01-05", ["a", "c"]]
        assert_index_equal(drawdowns2.wrapper.index, drawdowns.wrapper.index[1:-1])
        assert_index_equal(drawdowns2.wrapper.columns, drawdowns.wrapper.columns[[0, 2]])
        assert_frame_equal(
            drawdowns2.close, drawdowns.close.loc["2020-01-02":"2020-01-05", ["a", "c"]]
        )
        assert_records_close(
            drawdowns2.values,
            np.array([(1, 0, 1, 2, 3, 3.0, 1.0, 4.0, 1)], dtype=drawdown_dt),
        )

    def test_mapped_fields(self):
        for name in drawdown_dt.names:
            np.testing.assert_array_equal(getattr(drawdowns, name).values, drawdowns.values[name])

    def test_ts(self):
        assert_frame_equal(drawdowns.close, ts2)
        assert_series_equal(drawdowns["a"].close, ts2["a"])
        assert_frame_equal(drawdowns_grouped["g1"].close, ts2[["a", "b"]])
        assert drawdowns.replace(close=None)["a"].close is None

    def test_from_price(self):
        assert_records_close(
            drawdowns.values,
            np.array(
                [
                    (0, 0, 0, 1, 2, 2.0, 1.0, 3.0, 1),
                    (1, 0, 2, 3, 4, 3.0, 1.0, 4.0, 1),
                    (2, 0, 4, 5, 5, 4.0, 1.0, 1.0, 0),
                    (0, 1, 1, 2, 3, 2.0, 1.0, 3.0, 1),
                    (1, 1, 3, 4, 5, 3.0, 1.0, 4.0, 1),
                    (0, 2, 2, 4, 5, 3.0, 1.0, 2.0, 0),
                ],
                dtype=drawdown_dt,
            ),
        )
        assert drawdowns.wrapper.freq == day_dt
        assert_index_equal(drawdowns_grouped.wrapper.grouper.group_by, group_by)
        assert_records_close(
            vbt.Drawdowns.from_price(ts2, jitted=dict(parallel=True)).values,
            vbt.Drawdowns.from_price(ts2, jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            vbt.Drawdowns.from_price(ts2, chunked=True).values,
            vbt.Drawdowns.from_price(ts2, chunked=False).values,
        )

    def test_records_readable(self):
        records_readable = drawdowns.records_readable

        np.testing.assert_array_equal(
            records_readable["Drawdown Id"].values, np.array([0, 1, 2, 0, 1, 0])
        )
        np.testing.assert_array_equal(
            records_readable["Column"].values, np.array(["a", "a", "a", "b", "b", "c"])
        )
        np.testing.assert_array_equal(
            records_readable["Start Index"].values,
            np.array(
                [
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Valley Index"].values,
            np.array(
                [
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["End Index"].values,
            np.array(
                [
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Start Value"].values, np.array([2.0, 3.0, 4.0, 2.0, 3.0, 3.0])
        )
        np.testing.assert_array_equal(
            records_readable["Valley Value"].values, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )
        np.testing.assert_array_equal(
            records_readable["End Value"].values, np.array([3.0, 4.0, 1.0, 3.0, 4.0, 2.0])
        )
        np.testing.assert_array_equal(
            records_readable["Status"].values,
            np.array(["Recovered", "Recovered", "Active", "Recovered", "Recovered", "Active"]),
        )

    def test_ranges(self):
        assert_records_close(
            drawdowns.ranges.values,
            np.array(
                [
                    (0, 0, 0, 2, 1),
                    (1, 0, 2, 4, 1),
                    (2, 0, 4, 5, 0),
                    (0, 1, 1, 3, 1),
                    (1, 1, 3, 5, 1),
                    (0, 2, 2, 5, 0),
                ],
                dtype=range_dt,
            ),
        )

    def test_decline_ranges(self):
        assert_records_close(
            drawdowns.decline_ranges.values,
            np.array(
                [
                    (0, 0, 0, 1, 1),
                    (1, 0, 2, 3, 1),
                    (2, 0, 4, 5, 0),
                    (0, 1, 1, 2, 1),
                    (1, 1, 3, 4, 1),
                    (0, 2, 2, 4, 0),
                ],
                dtype=range_dt,
            ),
        )

    def test_recovery_ranges(self):
        assert_records_close(
            drawdowns.recovery_ranges.values,
            np.array(
                [
                    (0, 0, 1, 2, 1),
                    (1, 0, 3, 4, 1),
                    (2, 0, 5, 5, 0),
                    (0, 1, 2, 3, 1),
                    (1, 1, 4, 5, 1),
                    (0, 2, 4, 5, 0),
                ],
                dtype=range_dt,
            ),
        )

    def test_drawdown(self):
        np.testing.assert_array_almost_equal(
            drawdowns["a"].drawdown.values, np.array([-0.5, -0.66666667, -0.75])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.drawdown.values,
            np.array([-0.5, -0.66666667, -0.75, -0.5, -0.66666667, -0.66666667]),
        )
        assert_frame_equal(
            drawdowns.drawdown.to_pd(),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [-0.5, np.nan, np.nan, np.nan],
                        [np.nan, -0.5, np.nan, np.nan],
                        [-0.6666666666666666, np.nan, np.nan, np.nan],
                        [-0.75, -0.6666666666666666, -0.6666666666666666, np.nan],
                    ]
                ),
                index=ts2.index,
                columns=ts2.columns,
            ),
        )
        np.testing.assert_array_equal(
            drawdowns.get_drawdown(jitted=dict(parallel=True)).values,
            drawdowns.get_drawdown(jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            drawdowns.get_drawdown(chunked=True).values,
            drawdowns.get_drawdown(chunked=False).values,
        )

    def test_avg_drawdown(self):
        assert drawdowns["a"].avg_drawdown == -0.6388888888888888
        assert_series_equal(
            drawdowns.avg_drawdown,
            pd.Series(
                np.array([-0.6388888888888888, -0.5833333333333333, -0.6666666666666666, np.nan]),
                index=wrapper.columns,
            ).rename("avg_drawdown"),
        )
        assert_series_equal(
            drawdowns_grouped.avg_drawdown,
            pd.Series(
                np.array([-0.6166666666666666, -0.6666666666666666]),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("avg_drawdown"),
        )
        assert_series_equal(
            drawdowns.get_avg_drawdown(jitted=dict(parallel=True)),
            drawdowns.get_avg_drawdown(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_avg_drawdown(chunked=True),
            drawdowns.get_avg_drawdown(chunked=False),
        )

    def test_max_drawdown(self):
        assert drawdowns["a"].max_drawdown == -0.75
        assert_series_equal(
            drawdowns.max_drawdown,
            pd.Series(
                np.array([-0.75, -0.6666666666666666, -0.6666666666666666, np.nan]),
                index=wrapper.columns,
            ).rename("max_drawdown"),
        )
        assert_series_equal(
            drawdowns_grouped.max_drawdown,
            pd.Series(
                np.array([-0.75, -0.6666666666666666]), index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("max_drawdown"),
        )
        assert_series_equal(
            drawdowns.get_max_drawdown(jitted=dict(parallel=True)),
            drawdowns.get_max_drawdown(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_max_drawdown(chunked=True),
            drawdowns.get_max_drawdown(chunked=False),
        )

    def test_recovery_return(self):
        np.testing.assert_array_almost_equal(
            drawdowns["a"].recovery_return.values, np.array([2.0, 3.0, 0.0])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.recovery_return.values, np.array([2.0, 3.0, 0.0, 2.0, 3.0, 1.0])
        )
        assert_frame_equal(
            drawdowns.recovery_return.to_pd(),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [2.0, np.nan, np.nan, np.nan],
                        [np.nan, 2.0, np.nan, np.nan],
                        [3.0, np.nan, np.nan, np.nan],
                        [0.0, 3.0, 1.0, np.nan],
                    ]
                ),
                index=ts2.index,
                columns=ts2.columns,
            ),
        )
        np.testing.assert_array_equal(
            drawdowns.get_recovery_return(jitted=dict(parallel=True)).values,
            drawdowns.get_recovery_return(jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            drawdowns.get_recovery_return(chunked=True).values,
            drawdowns.get_recovery_return(chunked=False).values,
        )

    def test_avg_recovery_return(self):
        assert drawdowns["a"].avg_recovery_return == 1.6666666666666667
        assert_series_equal(
            drawdowns.avg_recovery_return,
            pd.Series(
                np.array([1.6666666666666667, 2.5, 1.0, np.nan]), index=wrapper.columns
            ).rename("avg_recovery_return"),
        )
        assert_series_equal(
            drawdowns_grouped.avg_recovery_return,
            pd.Series(np.array([2.0, 1.0]), index=pd.Index(["g1", "g2"], dtype="object")).rename(
                "avg_recovery_return"
            ),
        )
        assert_series_equal(
            drawdowns.get_avg_recovery_return(jitted=dict(parallel=True)),
            drawdowns.get_avg_recovery_return(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_avg_recovery_return(chunked=True),
            drawdowns.get_avg_recovery_return(chunked=False),
        )

    def test_max_recovery_return(self):
        assert drawdowns["a"].max_recovery_return == 3.0
        assert_series_equal(
            drawdowns.max_recovery_return,
            pd.Series(np.array([3.0, 3.0, 1.0, np.nan]), index=wrapper.columns).rename(
                "max_recovery_return"
            ),
        )
        assert_series_equal(
            drawdowns_grouped.max_recovery_return,
            pd.Series(np.array([3.0, 1.0]), index=pd.Index(["g1", "g2"], dtype="object")).rename(
                "max_recovery_return"
            ),
        )
        assert_series_equal(
            drawdowns.get_max_recovery_return(jitted=dict(parallel=True)),
            drawdowns.get_max_recovery_return(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_max_recovery_return(chunked=True),
            drawdowns.get_max_recovery_return(chunked=False),
        )

    def test_duration(self):
        np.testing.assert_array_almost_equal(drawdowns["a"].duration.values, np.array([2, 2, 2]))
        np.testing.assert_array_almost_equal(
            drawdowns.duration.values, np.array([2, 2, 2, 2, 2, 4])
        )

    def test_avg_duration(self):
        assert drawdowns["a"].avg_duration == pd.Timedelta("2 days 00:00:00")
        assert_series_equal(
            drawdowns.avg_duration,
            pd.Series(
                np.array(
                    [172800000000000, 172800000000000, 345600000000000, "NaT"],
                    dtype="timedelta64[ns]",
                ),
                index=wrapper.columns,
            ).rename("avg_duration"),
        )
        assert_series_equal(
            drawdowns_grouped.avg_duration,
            pd.Series(
                np.array([172800000000000, 345600000000000], dtype="timedelta64[ns]"),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("avg_duration"),
        )

    def test_get_mask_duration(self):
        assert drawdowns["a"].max_duration == pd.Timedelta("2 days 00:00:00")
        assert_series_equal(
            drawdowns.max_duration,
            pd.Series(
                np.array(
                    [172800000000000, 172800000000000, 345600000000000, "NaT"],
                    dtype="timedelta64[ns]",
                ),
                index=wrapper.columns,
            ).rename("max_duration"),
        )
        assert_series_equal(
            drawdowns_grouped.max_duration,
            pd.Series(
                np.array([172800000000000, 345600000000000], dtype="timedelta64[ns]"),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("max_duration"),
        )
        assert_series_equal(
            drawdowns.get_max_duration(jitted=dict(parallel=True)),
            drawdowns.get_max_duration(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_max_duration(chunked=True),
            drawdowns.get_max_duration(chunked=False),
        )

    def test_coverage(self):
        assert drawdowns["a"].coverage == 1.0
        assert_series_equal(
            drawdowns.coverage,
            pd.Series(
                np.array([1.0, 0.6666666666666666, 0.6666666666666666, np.nan]), index=ts2.columns
            ).rename("coverage"),
        )
        assert_series_equal(
            drawdowns_grouped.coverage,
            pd.Series(
                np.array([0.5, 0.3333333333333333]), index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("coverage"),
        )

    def test_decline_duration(self):
        np.testing.assert_array_almost_equal(
            drawdowns["a"].decline_duration.values, np.array([1.0, 1.0, 1.0])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.decline_duration.values,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]),
        )

    def test_recovery_duration(self):
        np.testing.assert_array_almost_equal(
            drawdowns["a"].recovery_duration.values, np.array([1, 1, 0])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.recovery_duration.values, np.array([1, 1, 0, 1, 1, 1])
        )
        np.testing.assert_array_equal(
            drawdowns.get_recovery_duration(jitted=dict(parallel=True)).values,
            drawdowns.get_recovery_duration(jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            drawdowns.get_recovery_duration(chunked=True).values,
            drawdowns.get_recovery_duration(chunked=False).values,
        )

    def test_recovery_duration_ratio(self):
        np.testing.assert_array_almost_equal(
            drawdowns["a"].recovery_duration_ratio.values, np.array([1.0, 1.0, 0.0])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.recovery_duration_ratio.values,
            np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.5]),
        )
        np.testing.assert_array_equal(
            drawdowns.get_recovery_duration_ratio(jitted=dict(parallel=True)).values,
            drawdowns.get_recovery_duration_ratio(jitted=dict(parallel=False)).values,
        )
        np.testing.assert_array_equal(
            drawdowns.get_recovery_duration_ratio(chunked=True).values,
            drawdowns.get_recovery_duration_ratio(chunked=False).values,
        )

    def test_active_records(self):
        assert isinstance(drawdowns.status_active, vbt.Drawdowns)
        assert drawdowns.status_active.wrapper == drawdowns.wrapper
        assert_records_close(
            drawdowns["a"].status_active.values,
            np.array([(2, 0, 4, 5, 5, 4.0, 1.0, 1.0, 0)], dtype=drawdown_dt),
        )
        assert_records_close(
            drawdowns["a"].status_active.values, drawdowns.status_active["a"].values
        )
        assert_records_close(
            drawdowns.status_active.values,
            np.array(
                [(2, 0, 4, 5, 5, 4.0, 1.0, 1.0, 0), (0, 2, 2, 4, 5, 3.0, 1.0, 2.0, 0)],
                dtype=drawdown_dt,
            ),
        )

    def test_recovered_records(self):
        assert isinstance(drawdowns.status_recovered, vbt.Drawdowns)
        assert drawdowns.status_recovered.wrapper == drawdowns.wrapper
        assert_records_close(
            drawdowns["a"].status_recovered.values,
            np.array(
                [(0, 0, 0, 1, 2, 2.0, 1.0, 3.0, 1), (1, 0, 2, 3, 4, 3.0, 1.0, 4.0, 1)],
                dtype=drawdown_dt,
            ),
        )
        assert_records_close(
            drawdowns["a"].status_recovered.values, drawdowns.status_recovered["a"].values
        )
        assert_records_close(
            drawdowns.status_recovered.values,
            np.array(
                [
                    (0, 0, 0, 1, 2, 2.0, 1.0, 3.0, 1),
                    (1, 0, 2, 3, 4, 3.0, 1.0, 4.0, 1),
                    (0, 1, 1, 2, 3, 2.0, 1.0, 3.0, 1),
                    (1, 1, 3, 4, 5, 3.0, 1.0, 4.0, 1),
                ],
                dtype=drawdown_dt,
            ),
        )

    def test_active_drawdown(self):
        assert drawdowns["a"].active_drawdown == -0.75
        assert_series_equal(
            drawdowns.active_drawdown,
            pd.Series(
                np.array([-0.75, np.nan, -0.3333333333333333, np.nan]), index=wrapper.columns
            ).rename("active_drawdown"),
        )
        with pytest.raises(Exception):
            drawdowns_grouped.active_drawdown
        assert_series_equal(
            drawdowns.get_active_drawdown(jitted=dict(parallel=True)),
            drawdowns.get_active_drawdown(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_active_drawdown(chunked=True),
            drawdowns.get_active_drawdown(chunked=False),
        )

    def test_active_duration(self):
        assert drawdowns["a"].active_duration == pd.Timedelta("2 days 00:00:00")
        assert_series_equal(
            drawdowns.active_duration,
            pd.Series(
                np.array([172800000000000, "NaT", 345600000000000, "NaT"], dtype="timedelta64[ns]"),
                index=wrapper.columns,
            ).rename("active_duration"),
        )
        with pytest.raises(Exception):
            drawdowns_grouped.active_duration
        assert_series_equal(
            drawdowns.get_active_duration(jitted=dict(parallel=True)),
            drawdowns.get_active_duration(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_active_duration(chunked=True),
            drawdowns.get_active_duration(chunked=False),
        )

    def test_active_recovery(self):
        assert drawdowns["a"].active_recovery == 0.0
        assert_series_equal(
            drawdowns.active_recovery,
            pd.Series(np.array([0.0, np.nan, 0.5, np.nan]), index=wrapper.columns).rename(
                "active_recovery"
            ),
        )
        with pytest.raises(Exception):
            drawdowns_grouped.active_recovery
        assert_series_equal(
            drawdowns.get_active_recovery(jitted=dict(parallel=True)),
            drawdowns.get_active_recovery(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_active_recovery(chunked=True),
            drawdowns.get_active_recovery(chunked=False),
        )

    def test_active_recovery_return(self):
        assert drawdowns["a"].active_recovery_return == 0.0
        assert_series_equal(
            drawdowns.active_recovery_return,
            pd.Series(np.array([0.0, np.nan, 1.0, np.nan]), index=wrapper.columns).rename(
                "active_recovery_return"
            ),
        )
        with pytest.raises(Exception):
            drawdowns_grouped.active_recovery_return
        assert_series_equal(
            drawdowns.get_active_recovery_return(jitted=dict(parallel=True)),
            drawdowns.get_active_recovery_return(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_active_recovery_return(chunked=True),
            drawdowns.get_active_recovery_return(chunked=False),
        )

    def test_active_recovery_duration(self):
        assert drawdowns["a"].active_recovery_duration == pd.Timedelta("0 days 00:00:00")
        assert_series_equal(
            drawdowns.active_recovery_duration,
            pd.Series(
                np.array([0, "NaT", 86400000000000, "NaT"], dtype="timedelta64[ns]"),
                index=wrapper.columns,
            ).rename("active_recovery_duration"),
        )
        with pytest.raises(Exception):
            drawdowns_grouped.active_recovery_duration
        assert_series_equal(
            drawdowns.get_active_recovery_duration(jitted=dict(parallel=True)),
            drawdowns.get_active_recovery_duration(jitted=dict(parallel=False)),
        )
        assert_series_equal(
            drawdowns.get_active_recovery_duration(chunked=True),
            drawdowns.get_active_recovery_duration(chunked=False),
        )

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Coverage [%]",
                "Total Records",
                "Total Recovered Drawdowns",
                "Total Active Drawdowns",
                "Active Drawdown [%]",
                "Active Duration",
                "Active Recovery [%]",
                "Active Recovery Return [%]",
                "Active Recovery Duration",
                "Max Drawdown [%]",
                "Avg Drawdown [%]",
                "Max Drawdown Duration",
                "Avg Drawdown Duration",
                "Max Recovery Return [%]",
                "Avg Recovery Return [%]",
                "Max Recovery Duration",
                "Avg Recovery Duration",
                "Avg Recovery Duration Ratio",
            ],
            dtype="object",
        )
        assert_series_equal(
            drawdowns.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-06 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    77.77777777777777,
                    1.5,
                    1.0,
                    0.5,
                    54.166666666666664,
                    pd.Timedelta("3 days 00:00:00"),
                    25.0,
                    50.0,
                    pd.Timedelta("0 days 12:00:00"),
                    66.66666666666666,
                    58.33333333333333,
                    pd.Timedelta("2 days 00:00:00"),
                    pd.Timedelta("2 days 00:00:00"),
                    300.0,
                    250.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    1.0,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            drawdowns.stats(settings=dict(incl_active=True)),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-06 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    77.77777777777777,
                    1.5,
                    1.0,
                    0.5,
                    54.166666666666664,
                    pd.Timedelta("3 days 00:00:00"),
                    25.0,
                    50.0,
                    pd.Timedelta("0 days 12:00:00"),
                    69.44444444444444,
                    62.962962962962955,
                    pd.Timedelta("2 days 16:00:00"),
                    pd.Timedelta("2 days 16:00:00"),
                    300.0,
                    250.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    1.0,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            drawdowns.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-06 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    100.0,
                    3,
                    2,
                    1,
                    75.0,
                    pd.Timedelta("2 days 00:00:00"),
                    0.0,
                    0.0,
                    pd.Timedelta("0 days 00:00:00"),
                    66.66666666666666,
                    58.33333333333333,
                    pd.Timedelta("2 days 00:00:00"),
                    pd.Timedelta("2 days 00:00:00"),
                    300.0,
                    250.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    1.0,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            drawdowns.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-06 00:00:00"),
                    pd.Timedelta("6 days 00:00:00"),
                    50.0,
                    5,
                    4,
                    1,
                    66.66666666666666,
                    58.33333333333333,
                    pd.Timedelta("2 days 00:00:00"),
                    pd.Timedelta("2 days 00:00:00"),
                    300.0,
                    250.0,
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    1.0,
                ],
                index=pd.Index(
                    [
                        "Start Index",
                        "End Index",
                        "Total Duration",
                        "Coverage [%]",
                        "Total Records",
                        "Total Recovered Drawdowns",
                        "Total Active Drawdowns",
                        "Max Drawdown [%]",
                        "Avg Drawdown [%]",
                        "Max Drawdown Duration",
                        "Avg Drawdown Duration",
                        "Max Recovery Return [%]",
                        "Avg Recovery Return [%]",
                        "Max Recovery Duration",
                        "Avg Recovery Duration",
                        "Avg Recovery Duration Ratio",
                    ],
                    dtype="object",
                ),
                name="g1",
            ),
        )
        assert_series_equal(drawdowns["c"].stats(), drawdowns.stats(column="c"))
        assert_series_equal(drawdowns["c"].stats(), drawdowns.stats(column="c", group_by=False))
        assert_series_equal(drawdowns_grouped["g2"].stats(), drawdowns_grouped.stats(column="g2"))
        assert_series_equal(
            drawdowns_grouped["g2"].stats(), drawdowns.stats(column="g2", group_by=group_by)
        )
        stats_df = drawdowns.stats(agg_func=None)
        assert stats_df.shape == (4, 21)
        assert_index_equal(stats_df.index, drawdowns.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_resample(self):
        np.testing.assert_array_equal(
            drawdowns.resample("1h").start_idx.values,
            np.array([0, 48, 96, 24, 72, 48]),
        )
        np.testing.assert_array_equal(
            drawdowns.resample("1h").end_idx.values,
            np.array([48, 96, 120, 72, 120, 120]),
        )
        np.testing.assert_array_equal(
            drawdowns.resample("10h").start_idx.values,
            np.array([0, 4, 9, 2, 7, 4]),
        )
        np.testing.assert_array_equal(
            drawdowns.resample("10h").end_idx.values,
            np.array([4, 9, 12, 7, 12, 12]),
        )
        np.testing.assert_array_equal(
            drawdowns.resample("3d").start_idx.values,
            np.array([0, 0, 1, 0, 1, 0]),
        )
        np.testing.assert_array_equal(
            drawdowns.resample("3d").end_idx.values,
            np.array([0, 1, 1, 1, 1, 1]),
        )
        assert_frame_equal(
            drawdowns.resample("1h").close,
            drawdowns.close.resample("1h").last().astype(float_),
        )
        assert_frame_equal(
            drawdowns.resample("10h").close,
            drawdowns.close.resample("10h").last().astype(float_),
        )
        assert_frame_equal(
            drawdowns.resample("3d").close,
            drawdowns.close.resample("3d").last().astype(float_),
        )
        assert_frame_equal(
            drawdowns.resample("1h", ffill_close=True).close,
            drawdowns.close.resample("1h").last().ffill().astype(float_),
        )
        assert_frame_equal(
            drawdowns.resample("10h", ffill_close=True).close,
            drawdowns.close.resample("10h").last().ffill().astype(float_),
        )
        assert_frame_equal(
            drawdowns.resample("3d", ffill_close=True).close,
            drawdowns.close.resample("3d").last().ffill().astype(float_),
        )
        assert_frame_equal(
            drawdowns.resample("1h", fbfill_close=True).close,
            drawdowns.close.resample("1h").last().ffill().bfill().astype(float_),
        )
        assert_frame_equal(
            drawdowns.resample("10h", fbfill_close=True).close,
            drawdowns.close.resample("10h").last().ffill().bfill().astype(float_),
        )
        assert_frame_equal(
            drawdowns.resample("3d", fbfill_close=True).close,
            drawdowns.close.resample("3d").last().ffill().bfill().astype(float_),
        )


# ############# orders ############# #

open = pd.Series(
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
    index=pd.date_range("2020", periods=8),
).vbt.tile(4, keys=["a", "b", "c", "d"])
high = pd.Series(
    [1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25],
    index=pd.date_range("2020", periods=8),
).vbt.tile(4, keys=["a", "b", "c", "d"])
low = pd.Series(
    [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25],
    index=pd.date_range("2020", periods=8),
).vbt.tile(4, keys=["a", "b", "c", "d"])
close = pd.Series(
    [1, 2, 3, 4, 5, 6, 7, 8],
    index=pd.date_range("2020", periods=8),
).vbt.tile(4, keys=["a", "b", "c", "d"])

size = np.full(close.shape, np.nan, dtype=float_)
size[:, 0] = [1, 0.1, -1, -0.1, np.nan, 1, -1, 2]
size[:, 1] = [-1, -0.1, 1, 0.1, np.nan, -1, 1, -2]
size[:, 2] = [1, 0.1, -1, -0.1, np.nan, 1, -2, 2]
orders = vbt.Portfolio.from_orders(
    open=open,
    high=high,
    low=low,
    close=close,
    size=size,
    fees=0.01,
    freq="1 days",
).orders
orders_grouped = orders.regroup(group_by)


class TestOrders:
    def test_row_stack(self):
        close2 = close * 2
        close2.index = pd.date_range("2020-01-09", "2020-01-16")
        orders1 = vbt.Portfolio.from_orders(close, size, fees=0.01, freq="1 days").orders
        orders2 = vbt.Portfolio.from_orders(close2, size, fees=0.01, freq="1 days").orders
        new_orders = vbt.Orders.row_stack(orders1, orders2)
        assert_frame_equal(new_orders.close, pd.concat((close, close2)))
        with pytest.raises(Exception):
            vbt.Orders.row_stack(orders1.replace(close=None), orders2)
        with pytest.raises(Exception):
            vbt.Orders.row_stack(orders1, orders2.replace(close=None))
        new_orders = vbt.Orders.row_stack(orders1.replace(close=None), orders2.replace(close=None))
        assert new_orders.close is None

    def test_column_stack(self):
        close2 = close * 2
        close2.columns = ["e", "f", "g", "h"]
        orders1 = vbt.Portfolio.from_orders(close, size, fees=0.01, freq="1 days").orders
        orders2 = vbt.Portfolio.from_orders(close2, size, fees=0.01, freq="1 days").orders
        new_orders = vbt.Orders.column_stack(orders1, orders2)
        assert_frame_equal(new_orders.close, pd.concat((close, close2), axis=1))
        with pytest.raises(Exception):
            vbt.Orders.column_stack(orders1.replace(close=None), orders2)
        with pytest.raises(Exception):
            vbt.Orders.column_stack(orders1, orders2.replace(close=None))
        new_orders = vbt.Orders.column_stack(
            orders1.replace(close=None), orders2.replace(close=None)
        )
        assert new_orders.close is None

    def test_indexing(self):
        orders2 = orders.loc["2020-01-03":"2020-01-06", ["a", "c"]]
        assert_index_equal(orders2.wrapper.index, orders.wrapper.index[2:-2])
        assert_index_equal(orders2.wrapper.columns, orders.wrapper.columns[[0, 2]])
        assert_frame_equal(orders2.close, orders.close.loc["2020-01-03":"2020-01-06", ["a", "c"]])
        assert_records_close(
            orders2.values,
            np.array(
                [
                    (2, 0, 0, 1.0, 3.0, 0.03, 1),
                    (3, 0, 1, 0.1, 4.0, 0.004, 1),
                    (4, 0, 3, 1.0, 6.0, 0.06, 0),
                    (2, 1, 0, 1.0, 3.0, 0.03, 1),
                    (3, 1, 1, 0.1, 4.0, 0.004, 1),
                    (4, 1, 3, 1.0, 6.0, 0.06, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_mapped_fields(self):
        for name in order_dt.names:
            np.testing.assert_array_equal(getattr(orders, name).values, orders.values[name])

    def test_close(self):
        assert_frame_equal(orders.close, close)
        assert_series_equal(orders["a"].close, close["a"])
        assert_frame_equal(orders_grouped["g1"].close, close[["a", "b"]])
        assert orders.replace(close=None)["a"].close is None

    def test_records_readable(self):
        records_readable = orders.records_readable

        np.testing.assert_array_equal(
            records_readable["Order Id"].values,
            np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]),
        )
        np.testing.assert_array_equal(
            records_readable["Index"].values,
            np.array(
                [
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Column"].values,
            np.array(
                [
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Size"].values,
            np.array(
                [
                    1.0,
                    0.1,
                    1.0,
                    0.1,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    0.1,
                    1.0,
                    0.1,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    0.1,
                    1.0,
                    0.1,
                    1.0,
                    2.0,
                    2.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Price"].values,
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    7.0,
                    8.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Fees"].values,
            np.array(
                [
                    0.01,
                    0.002,
                    0.03,
                    0.004,
                    0.06,
                    0.07,
                    0.16,
                    0.01,
                    0.002,
                    0.03,
                    0.004,
                    0.06,
                    0.07,
                    0.16,
                    0.01,
                    0.002,
                    0.03,
                    0.004,
                    0.06,
                    0.14,
                    0.16,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Side"].values,
            np.array(
                [
                    "Buy",
                    "Buy",
                    "Sell",
                    "Sell",
                    "Buy",
                    "Sell",
                    "Buy",
                    "Sell",
                    "Sell",
                    "Buy",
                    "Buy",
                    "Sell",
                    "Buy",
                    "Sell",
                    "Buy",
                    "Buy",
                    "Sell",
                    "Sell",
                    "Buy",
                    "Sell",
                    "Buy",
                ]
            ),
        )

    def test_buy_records(self):
        assert isinstance(orders.side_buy, vbt.Orders)
        assert orders.side_buy.wrapper == orders.wrapper
        assert_records_close(
            orders["a"].side_buy.values,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.01, 0),
                    (1, 0, 1, 0.1, 2.0, 0.002, 0),
                    (4, 0, 5, 1.0, 6.0, 0.06, 0),
                    (6, 0, 7, 2.0, 8.0, 0.16, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(orders["a"].side_buy.values, orders.side_buy["a"].values)
        assert_records_close(
            orders.side_buy.values,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.01, 0),
                    (1, 0, 1, 0.1, 2.0, 0.002, 0),
                    (4, 0, 5, 1.0, 6.0, 0.06, 0),
                    (6, 0, 7, 2.0, 8.0, 0.16, 0),
                    (2, 1, 2, 1.0, 3.0, 0.03, 0),
                    (3, 1, 3, 0.1, 4.0, 0.004, 0),
                    (5, 1, 6, 1.0, 7.0, 0.07, 0),
                    (0, 2, 0, 1.0, 1.0, 0.01, 0),
                    (1, 2, 1, 0.1, 2.0, 0.002, 0),
                    (4, 2, 5, 1.0, 6.0, 0.06, 0),
                    (6, 2, 7, 2.0, 8.0, 0.16, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_sell_records(self):
        assert isinstance(orders.side_sell, vbt.Orders)
        assert orders.side_sell.wrapper == orders.wrapper
        assert_records_close(
            orders["a"].side_sell.values,
            np.array(
                [
                    (2, 0, 2, 1.0, 3.0, 0.03, 1),
                    (3, 0, 3, 0.1, 4.0, 0.004, 1),
                    (5, 0, 6, 1.0, 7.0, 0.07, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(orders["a"].side_sell.values, orders.side_sell["a"].values)
        assert_records_close(
            orders.side_sell.values,
            np.array(
                [
                    (2, 0, 2, 1.0, 3.0, 0.03, 1),
                    (3, 0, 3, 0.1, 4.0, 0.004, 1),
                    (5, 0, 6, 1.0, 7.0, 0.07, 1),
                    (0, 1, 0, 1.0, 1.0, 0.01, 1),
                    (1, 1, 1, 0.1, 2.0, 0.002, 1),
                    (4, 1, 5, 1.0, 6.0, 0.06, 1),
                    (6, 1, 7, 2.0, 8.0, 0.16, 1),
                    (2, 2, 2, 1.0, 3.0, 0.03, 1),
                    (3, 2, 3, 0.1, 4.0, 0.004, 1),
                    (5, 2, 6, 2.0, 7.0, 0.14, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_weighted_price(self):
        assert orders["a"].weighted_price == 5.419354838709678
        pd.testing.assert_series_equal(
            orders.weighted_price,
            pd.Series(
                [
                    5.419354838709678,
                    5.419354838709678,
                    5.638888888888889,
                    np.nan,
                ],
                index=close.columns,
            ).rename("weighted_price"),
        )

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Records",
                "Side Counts: Buy",
                "Side Counts: Sell",
                "Size: Min",
                "Size: Median",
                "Size: Max",
                "Fees: Min",
                "Fees: Median",
                "Fees: Max",
                "Weighted Buy Price",
                "Weighted Sell Price",
            ],
            dtype="object",
        )
        assert_series_equal(
            orders.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    5.25,
                    2.75,
                    2.5,
                    0.10000000000000002,
                    1.0,
                    2.0,
                    0.002,
                    0.03,
                    0.16,
                    5.423151374370888,
                    5.4079402545177535,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            orders.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    7,
                    4,
                    3,
                    0.1,
                    1.0,
                    2.0,
                    0.002,
                    0.03,
                    0.16,
                    5.658536585365854,
                    4.9523809523809526,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            orders.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    14,
                    7,
                    7,
                    0.1,
                    1.0,
                    2.0,
                    0.002,
                    0.03,
                    0.16,
                    5.419354838709677,
                    5.419354838709678,
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(orders["c"].stats(), orders.stats(column="c"))
        assert_series_equal(orders["c"].stats(), orders.stats(column="c", group_by=False))
        assert_series_equal(orders_grouped["g2"].stats(), orders_grouped.stats(column="g2"))
        assert_series_equal(
            orders_grouped["g2"].stats(), orders.stats(column="g2", group_by=group_by)
        )
        stats_df = orders.stats(agg_func=None)
        assert stats_df.shape == (4, 14)
        assert_index_equal(stats_df.index, orders.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_resample(self):
        np.testing.assert_array_equal(
            orders.resample("1h").idx_arr,
            np.array(
                [
                    0,
                    24,
                    48,
                    72,
                    120,
                    144,
                    168,
                    0,
                    24,
                    48,
                    72,
                    120,
                    144,
                    168,
                    0,
                    24,
                    48,
                    72,
                    120,
                    144,
                    168,
                ]
            ),
        )
        np.testing.assert_array_equal(
            orders.resample("10h").idx_arr,
            np.array([0, 2, 4, 7, 12, 14, 16, 0, 2, 4, 7, 12, 14, 16, 0, 2, 4, 7, 12, 14, 16]),
        )
        np.testing.assert_array_equal(
            orders.resample("3d").idx_arr,
            np.array([0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 1, 1, 2, 2]),
        )
        assert_frame_equal(
            orders.resample("1h").close,
            orders.close.resample("1h").last().astype(float_),
        )
        assert_frame_equal(
            orders.resample("10h").close,
            orders.close.resample("10h").last().astype(float_),
        )
        assert_frame_equal(
            orders.resample("3d").close,
            orders.close.resample("3d").last().astype(float_),
        )
        assert_frame_equal(
            orders.resample("1h", ffill_close=True).close,
            orders.close.resample("1h").last().ffill().astype(float_),
        )
        assert_frame_equal(
            orders.resample("10h", ffill_close=True).close,
            orders.close.resample("10h").last().ffill().astype(float_),
        )
        assert_frame_equal(
            orders.resample("3d", ffill_close=True).close,
            orders.close.resample("3d").last().ffill().astype(float_),
        )
        assert_frame_equal(
            orders.resample("1h", fbfill_close=True).close,
            orders.close.resample("1h").last().ffill().bfill().astype(float_),
        )
        assert_frame_equal(
            orders.resample("10h", fbfill_close=True).close,
            orders.close.resample("10h").last().ffill().bfill().astype(float_),
        )
        assert_frame_equal(
            orders.resample("3d", fbfill_close=True).close,
            orders.close.resample("3d").last().ffill().bfill().astype(float_),
        )


fs_orders = vbt.Portfolio.from_signals(
    open=open,
    high=high,
    low=low,
    close=close,
    entries=np.where(size > 0, True, False),
    exits=np.where(size < 0, True, False),
    order_type="limit",
    direction="both",
    accumulate=True,
    size=np.abs(size),
    fees=0.01,
    freq="1 days",
).orders
fs_orders_grouped = fs_orders.regroup(group_by)


class TestFSOrders:
    def test_records_readable(self):
        records_readable = fs_orders.records_readable

        np.testing.assert_array_equal(
            records_readable["Order Id"].values,
            np.array([0, 1, 0, 1, 0, 1]),
        )
        np.testing.assert_array_equal(
            records_readable["Signal Index"].values,
            np.array(
                [
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Creation Index"].values,
            np.array(
                [
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Fill Index"].values,
            np.array(
                [
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Column"].values,
            np.array(["a", "a", "b", "b", "c", "c"]),
        )
        np.testing.assert_array_equal(
            records_readable["Size"].values,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]),
        )
        np.testing.assert_array_equal(
            records_readable["Price"].values,
            np.array([3.0, 7.0, 1.0, 6.0, 3.0, 7.0]),
        )
        np.testing.assert_array_equal(
            records_readable["Fees"].values,
            np.array([0.03, 0.07, 0.01, 0.06, 0.03, 0.14]),
        )
        np.testing.assert_array_equal(
            records_readable["Side"].values,
            np.array(["Sell", "Sell", "Sell", "Sell", "Sell", "Sell"]),
        )
        np.testing.assert_array_equal(
            records_readable["Type"].values,
            np.array(["Limit", "Limit", "Limit", "Limit", "Limit", "Limit"]),
        )
        np.testing.assert_array_equal(
            records_readable["Stop Type"].values,
            np.array([None, None, None, None, None, None]),
        )

    def test_ranges(self):
        assert_records_close(
            fs_orders.ranges.values,
            np.array(
                [
                    (0, 0, 2, 3, 1),
                    (1, 0, 6, 7, 1),
                    (0, 1, 0, 1, 1),
                    (1, 1, 5, 6, 1),
                    (0, 2, 2, 3, 1),
                    (1, 2, 6, 7, 1),
                ],
                dtype=range_dt,
            ),
        )

    def test_creation_ranges(self):
        assert_records_close(
            fs_orders.creation_ranges.values,
            np.array(
                [
                    (0, 0, 2, 2, 1),
                    (1, 0, 6, 6, 1),
                    (0, 1, 0, 0, 1),
                    (1, 1, 5, 5, 1),
                    (0, 2, 2, 2, 1),
                    (1, 2, 6, 6, 1),
                ],
                dtype=range_dt,
            ),
        )

    def test_fill_ranges(self):
        assert_records_close(
            fs_orders.fill_ranges.values,
            np.array(
                [
                    (0, 0, 2, 3, 1),
                    (1, 0, 6, 7, 1),
                    (0, 1, 0, 1, 1),
                    (1, 1, 5, 6, 1),
                    (0, 2, 2, 3, 1),
                    (1, 2, 6, 7, 1),
                ],
                dtype=range_dt,
            ),
        )

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Records",
                "Side Counts: Buy",
                "Side Counts: Sell",
                "Type Counts: Market",
                "Type Counts: Limit",
                "Stop Type Counts: None",
                "Stop Type Counts: SL",
                "Stop Type Counts: TSL",
                "Stop Type Counts: TTP",
                "Stop Type Counts: TP",
                "Stop Type Counts: TD",
                "Stop Type Counts: DT",
                "Size: Min",
                "Size: Median",
                "Size: Max",
                "Fees: Min",
                "Fees: Median",
                "Fees: Max",
                "Weighted Buy Price",
                "Weighted Sell Price",
                "Avg Signal-Creation Duration",
                "Avg Creation-Fill Duration",
                "Avg Signal-Fill Duration",
            ],
            dtype="object",
        )
        assert_series_equal(
            fs_orders.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    1.5,
                    0.0,
                    1.5,
                    0.0,
                    1.5,
                    1.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.1666666666666667,
                    1.3333333333333333,
                    0.023333333333333334,
                    0.05666666666666667,
                    0.09000000000000001,
                    np.nan,
                    4.722222222222222,
                    pd.Timedelta("0 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            fs_orders.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    2,
                    0,
                    2,
                    0,
                    2,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1.0,
                    1.0,
                    1.0,
                    0.03,
                    0.05,
                    0.07,
                    np.nan,
                    5.0,
                    pd.Timedelta("0 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            fs_orders.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    4,
                    0,
                    4,
                    0,
                    4,
                    4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1.0,
                    1.0,
                    1.0,
                    0.01,
                    0.045,
                    0.07,
                    np.nan,
                    4.25,
                    pd.Timedelta("0 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                    pd.Timedelta("1 days 00:00:00"),
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(fs_orders["c"].stats(), fs_orders.stats(column="c"))
        assert_series_equal(fs_orders["c"].stats(), fs_orders.stats(column="c", group_by=False))
        assert_series_equal(fs_orders_grouped["g2"].stats(), fs_orders_grouped.stats(column="g2"))
        assert_series_equal(
            fs_orders_grouped["g2"].stats(), fs_orders.stats(column="g2", group_by=group_by)
        )
        stats_df = fs_orders.stats(agg_func=None)
        assert stats_df.shape == (4, 26)
        assert_index_equal(stats_df.index, fs_orders.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)


# ############# trades ############# #

exit_trades = vbt.ExitTrades.from_orders(orders)
exit_trades_grouped = vbt.ExitTrades.from_orders(orders_grouped)


class TestExitTrades:
    def test_row_stack(self):
        close2 = close * 2
        close2.index = pd.date_range("2020-01-09", "2020-01-16")
        trades1 = vbt.Portfolio.from_orders(close, size, fees=0.01, freq="1 days").exit_trades
        trades2 = vbt.Portfolio.from_orders(close2, size, fees=0.01, freq="1 days").exit_trades
        new_trades = vbt.Trades.row_stack(trades1, trades2)
        assert_frame_equal(new_trades.close, pd.concat((close, close2)))
        with pytest.raises(Exception):
            vbt.Orders.row_stack(trades1.replace(close=None), trades2)
        with pytest.raises(Exception):
            vbt.Orders.row_stack(trades1, trades2.replace(close=None))
        new_trades = vbt.Trades.row_stack(trades1.replace(close=None), trades2.replace(close=None))
        assert new_trades.close is None

    def test_column_stack(self):
        close2 = close * 2
        close2.columns = ["e", "f", "g", "h"]
        trades1 = vbt.Portfolio.from_orders(close, size, fees=0.01, freq="1 days").exit_trades
        trades2 = vbt.Portfolio.from_orders(close2, size, fees=0.01, freq="1 days").exit_trades
        new_trades = vbt.Trades.column_stack(trades1, trades2)
        assert_frame_equal(new_trades.close, pd.concat((close, close2), axis=1))
        with pytest.raises(Exception):
            vbt.Orders.column_stack(trades1.replace(close=None), trades2)
        with pytest.raises(Exception):
            vbt.Orders.column_stack(trades1, trades2.replace(close=None))
        new_trades = vbt.Trades.column_stack(
            trades1.replace(close=None), trades2.replace(close=None)
        )
        assert new_trades.close is None

    def test_indexing(self):
        exit_trades2 = exit_trades.loc["2020-01-05":, ["a", "c"]]
        assert_index_equal(exit_trades2.wrapper.index, exit_trades.wrapper.index[4:])
        assert_index_equal(exit_trades2.wrapper.columns, exit_trades.wrapper.columns[[0, 2]])
        assert_frame_equal(exit_trades2.close, exit_trades.close.loc["2020-01-05":, ["a", "c"]])
        assert_records_close(
            exit_trades2.values,
            np.array(
                [
                    (
                        2,
                        0,
                        1.0,
                        4,
                        1,
                        6.0,
                        0.06,
                        5,
                        2,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (3, 0, 2.0, 6, 3, 8.0, 0.16, -1, 3, 8.0, 0.0, -0.16, -0.01, 0, 0, 2),
                    (
                        2,
                        1,
                        1.0,
                        4,
                        1,
                        6.0,
                        0.06,
                        5,
                        2,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (
                        3,
                        1,
                        1.0,
                        5,
                        2,
                        7.0,
                        0.07,
                        6,
                        3,
                        8.0,
                        0.08,
                        -1.1500000000000001,
                        -0.1642857142857143,
                        1,
                        1,
                        2,
                    ),
                    (4, 1, 1.0, 6, 3, 8.0, 0.08, -1, 3, 8.0, 0.0, -0.08, -0.01, 0, 0, 3),
                ],
                dtype=trade_dt,
            ),
        )

    def test_mapped_fields(self):
        for name in trade_dt.names:
            if name == "return":
                np.testing.assert_array_equal(
                    exit_trades.returns.values, exit_trades.values[name]
                )
            else:
                np.testing.assert_array_equal(
                    getattr(exit_trades, name).values, exit_trades.values[name]
                )

    def test_close(self):
        assert_frame_equal(exit_trades.close, close)
        assert_series_equal(exit_trades["a"].close, close["a"])
        assert_frame_equal(exit_trades_grouped["g1"].close, close[["a", "b"]])
        assert exit_trades.replace(close=None)["a"].close is None

    def test_records_arr(self):
        assert_records_close(
            exit_trades.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (3, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2),
                    (
                        0,
                        1,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        -1.9500000000000002,
                        -1.7875000000000003,
                        1,
                        1,
                        0,
                    ),
                    (
                        1,
                        1,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        -0.29600000000000026,
                        -2.7133333333333334,
                        1,
                        1,
                        0,
                    ),
                    (
                        2,
                        1,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        -1.1300000000000001,
                        -0.18833333333333335,
                        1,
                        1,
                        1,
                    ),
                    (3, 1, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 1, 0, 2),
                    (
                        0,
                        2,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
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
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        2,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (
                        3,
                        2,
                        1.0,
                        5,
                        6,
                        7.0,
                        0.07,
                        6,
                        7,
                        8.0,
                        0.08,
                        -1.1500000000000001,
                        -0.1642857142857143,
                        1,
                        1,
                        2,
                    ),
                    (4, 2, 1.0, 6, 7, 8.0, 0.08, -1, 7, 8.0, 0.0, -0.08, -0.01, 0, 0, 3),
                ],
                dtype=trade_dt,
            ),
        )
        reversed_col_orders = orders.replace(
            records_arr=np.concatenate(
                (
                    orders.values[orders.values["col"] == 2],
                    orders.values[orders.values["col"] == 1],
                    orders.values[orders.values["col"] == 0],
                )
            )
        )
        assert_records_close(
            vbt.ExitTrades.from_orders(reversed_col_orders).values, exit_trades.values
        )
        assert_records_close(
            vbt.ExitTrades.from_orders(orders, jitted=dict(parallel=True)).values,
            vbt.ExitTrades.from_orders(orders, jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            vbt.ExitTrades.from_orders(orders, chunked=True).values,
            vbt.ExitTrades.from_orders(orders, chunked=False).values,
        )

    def test_records_readable(self):
        records_readable = exit_trades.records_readable

        np.testing.assert_array_equal(
            records_readable["Exit Trade Id"].values,
            np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4]),
        )
        np.testing.assert_array_equal(
            records_readable["Column"].values,
            np.array(["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "c"]),
        )
        np.testing.assert_array_equal(
            records_readable["Size"].values,
            np.array(
                [
                    1.0,
                    0.10000000000000009,
                    1.0,
                    2.0,
                    1.0,
                    0.10000000000000009,
                    1.0,
                    2.0,
                    1.0,
                    0.10000000000000009,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Entry Index"].values,
            np.array(
                [
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Avg Entry Price"].values,
            np.array(
                [
                    1.0909090909090908,
                    1.0909090909090908,
                    6.0,
                    8.0,
                    1.0909090909090908,
                    1.0909090909090908,
                    6.0,
                    8.0,
                    1.0909090909090908,
                    1.0909090909090908,
                    6.0,
                    7.0,
                    8.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Entry Fees"].values,
            np.array(
                [
                    0.010909090909090908,
                    0.0010909090909090918,
                    0.06,
                    0.16,
                    0.010909090909090908,
                    0.0010909090909090918,
                    0.06,
                    0.16,
                    0.010909090909090908,
                    0.0010909090909090918,
                    0.06,
                    0.07,
                    0.08,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Exit Index"].values,
            np.array(
                [
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Avg Exit Price"].values,
            np.array([3.0, 4.0, 7.0, 8.0, 3.0, 4.0, 7.0, 8.0, 3.0, 4.0, 7.0, 8.0, 8.0]),
        )
        np.testing.assert_array_equal(
            records_readable["Exit Fees"].values,
            np.array(
                [0.03, 0.004, 0.07, 0.0, 0.03, 0.004, 0.07, 0.0, 0.03, 0.004, 0.07, 0.08, 0.0]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["PnL"].values,
            np.array(
                [
                    1.8681818181818182,
                    0.2858181818181821,
                    0.8699999999999999,
                    -0.16,
                    -1.9500000000000002,
                    -0.29600000000000026,
                    -1.1300000000000001,
                    -0.16,
                    1.8681818181818182,
                    0.2858181818181821,
                    0.8699999999999999,
                    -1.1500000000000001,
                    -0.08,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Return"].values,
            np.array(
                [
                    1.7125000000000001,
                    2.62,
                    0.145,
                    -0.01,
                    -1.7875000000000003,
                    -2.7133333333333334,
                    -0.18833333333333335,
                    -0.01,
                    1.7125000000000001,
                    2.62,
                    0.145,
                    -0.1642857142857143,
                    -0.01,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Direction"].values,
            np.array(
                [
                    "Long",
                    "Long",
                    "Long",
                    "Long",
                    "Short",
                    "Short",
                    "Short",
                    "Short",
                    "Long",
                    "Long",
                    "Long",
                    "Short",
                    "Long",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Status"].values,
            np.array(
                [
                    "Closed",
                    "Closed",
                    "Closed",
                    "Open",
                    "Closed",
                    "Closed",
                    "Closed",
                    "Open",
                    "Closed",
                    "Closed",
                    "Closed",
                    "Closed",
                    "Open",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Position Id"].values,
            np.array([0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 3]),
        )

    def test_ranges(self):
        assert_records_close(
            exit_trades.ranges.values,
            np.array(
                [
                    (0, 0, 0, 2, 1),
                    (1, 0, 0, 3, 1),
                    (2, 0, 5, 6, 1),
                    (3, 0, 7, 7, 0),
                    (0, 1, 0, 2, 1),
                    (1, 1, 0, 3, 1),
                    (2, 1, 5, 6, 1),
                    (3, 1, 7, 7, 0),
                    (0, 2, 0, 2, 1),
                    (1, 2, 0, 3, 1),
                    (2, 2, 5, 6, 1),
                    (3, 2, 6, 7, 1),
                    (4, 2, 7, 7, 0),
                ],
                dtype=range_dt,
            ),
        )

    def test_duration(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].duration.values, np.array([2, 3, 1, 1])
        )
        np.testing.assert_array_almost_equal(
            exit_trades.duration.values,
            np.array([2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 1]),
        )

    def test_winning_records(self):
        assert isinstance(exit_trades.winning, vbt.ExitTrades)
        assert exit_trades.winning.wrapper == exit_trades.wrapper
        assert_records_close(
            exit_trades["a"].winning.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                ],
                dtype=trade_dt,
            ),
        )
        assert_records_close(exit_trades["a"].winning.values, exit_trades.winning["a"].values)
        assert_records_close(
            exit_trades.winning.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (
                        0,
                        2,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
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
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        2,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                ],
                dtype=trade_dt,
            ),
        )

    def test_losing_records(self):
        assert isinstance(exit_trades.losing, vbt.ExitTrades)
        assert exit_trades.losing.wrapper == exit_trades.wrapper
        assert_records_close(
            exit_trades["a"].losing.values,
            np.array(
                [(3, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2)],
                dtype=trade_dt,
            ),
        )
        assert_records_close(exit_trades["a"].losing.values, exit_trades.losing["a"].values)
        assert_records_close(
            exit_trades.losing.values,
            np.array(
                [
                    (3, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2),
                    (
                        0,
                        1,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        -1.9500000000000002,
                        -1.7875000000000003,
                        1,
                        1,
                        0,
                    ),
                    (
                        1,
                        1,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        -0.29600000000000026,
                        -2.7133333333333334,
                        1,
                        1,
                        0,
                    ),
                    (
                        2,
                        1,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        -1.1300000000000001,
                        -0.18833333333333335,
                        1,
                        1,
                        1,
                    ),
                    (3, 1, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 1, 0, 2),
                    (
                        3,
                        2,
                        1.0,
                        5,
                        6,
                        7.0,
                        0.07,
                        6,
                        7,
                        8.0,
                        0.08,
                        -1.1500000000000001,
                        -0.1642857142857143,
                        1,
                        1,
                        2,
                    ),
                    (4, 2, 1.0, 6, 7, 8.0, 0.08, -1, 7, 8.0, 0.0, -0.08, -0.01, 0, 0, 3),
                ],
                dtype=trade_dt,
            ),
        )

    def test_winning_streak(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].winning_streak.values, np.array([1, 2, 3, 0])
        )
        np.testing.assert_array_almost_equal(
            exit_trades.winning_streak.values,
            np.array([1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0]),
        )

    def test_losing_streak(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].losing_streak.values, np.array([0, 0, 0, 1])
        )
        np.testing.assert_array_almost_equal(
            exit_trades.losing_streak.values,
            np.array([0, 0, 0, 1, 1, 2, 3, 4, 0, 0, 0, 1, 2]),
        )

    def test_win_rate(self):
        assert exit_trades["a"].win_rate == 0.75
        assert_series_equal(
            exit_trades.win_rate,
            pd.Series(np.array([0.75, 0.0, 0.6, np.nan]), index=close.columns).rename("win_rate"),
        )
        assert_series_equal(
            exit_trades_grouped.win_rate,
            pd.Series(np.array([0.375, 0.6]), index=pd.Index(["g1", "g2"], dtype="object")).rename(
                "win_rate"
            ),
        )

    def test_profit_factor(self):
        assert exit_trades["a"].profit_factor == 18.9
        assert_series_equal(
            exit_trades.profit_factor,
            pd.Series(np.array([18.9, 0.0, 2.458536585365853, np.nan]), index=ts2.columns).rename(
                "profit_factor"
            ),
        )
        assert_series_equal(
            exit_trades_grouped.profit_factor,
            pd.Series(
                np.array([0.818181818181818, 2.458536585365853]),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("profit_factor"),
        )
        assert_series_equal(
            exit_trades.rel_profit_factor,
            pd.Series(
                np.array([447.75, 0.0, 25.6905737704918, np.nan]),
                index=ts2.columns,
            ).rename("rel_profit_factor"),
        )

    def test_expectancy(self):
        assert exit_trades["a"].expectancy == 0.716
        assert_series_equal(
            exit_trades.expectancy,
            pd.Series(np.array([0.716, -0.884, 0.3588, np.nan]), index=ts2.columns).rename(
                "expectancy"
            ),
        )
        assert_series_equal(
            exit_trades_grouped.expectancy,
            pd.Series(
                np.array([-0.084, 0.3588]), index=pd.Index(["g1", "g2"], dtype="object")
            ).rename("expectancy"),
        )
        assert_series_equal(
            exit_trades.rel_expectancy,
            pd.Series(
                np.array([1.116875, -1.1747916666666667, 0.860642857142857, np.nan]),
                index=ts2.columns,
            ).rename("rel_expectancy"),
        )

    def test_sqn(self):
        assert exit_trades["a"].sqn == 1.634155521947584
        assert_series_equal(
            exit_trades.sqn,
            pd.Series(
                np.array([1.634155521947584, -2.1300730719161516, 0.7166040292904912, np.nan]),
                index=ts2.columns,
            ).rename("sqn"),
        )
        assert_series_equal(
            exit_trades_grouped.sqn,
            pd.Series(
                np.array([-0.20404670761704502, 0.7166040292904912]),
                index=pd.Index(["g1", "g2"], dtype="object"),
            ).rename("sqn"),
        )
        assert_series_equal(
            exit_trades.rel_sqn,
            pd.Series(
                np.array([1.7607073523274486, -1.8069510003568587, 1.5530869626255832, np.nan]),
                index=ts2.columns,
            ).rename("rel_sqn"),
        )

    def test_best_price(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].best_price.values, np.array([3.0, 4.0, 7.0, 8.0])
        )
        np.testing.assert_array_almost_equal(
            exit_trades.best_price.values,
            np.array([3.0, 4.0, 7.0, 8.0, 1.0, 1.0, 6.0, 8.0, 3.0, 4.0, 7.0, 7.0, 8.0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_best_price(entry_price_open=True).values,
            np.array([3.0, 4.0, 7.0, 8.25, 0.25, 0.25, 5.25, 7.25, 3.0, 4.0, 7.0, 6.25, 8.25]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_best_price(exit_price_close=True).values,
            np.array([3.25, 4.25, 7.25, 8.0, 1.0, 1.0, 6.0, 8.0, 3.25, 4.25, 7.25, 7.0, 8.0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_best_price(entry_price_open=True, exit_price_close=True).values,
            np.array(
                [3.25, 4.25, 7.25, 8.25, 0.25, 0.25, 5.25, 7.25, 3.25, 4.25, 7.25, 6.25, 8.25]
            ),
        )

    def test_worst_price(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].worst_price.values, np.array([1.0, 1.0, 6.0, 8.0])
        )
        np.testing.assert_array_almost_equal(
            exit_trades.worst_price.values,
            np.array([1.0, 1.0, 6.0, 8.0, 3.0, 4.0, 7.0, 8.0, 1.0, 1.0, 6.0, 8.0, 8.0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_worst_price(entry_price_open=True).values,
            np.array([0.25, 0.25, 5.25, 7.25, 3.0, 4.0, 7.0, 8.25, 0.25, 0.25, 5.25, 8.0, 7.25]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_worst_price(exit_price_close=True).values,
            np.array([1.0, 1.0, 6.0, 8.0, 3.25, 4.25, 7.25, 8.0, 1.0, 1.0, 6.0, 8.25, 8.0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_worst_price(entry_price_open=True, exit_price_close=True).values,
            np.array(
                [0.25, 0.25, 5.25, 7.25, 3.25, 4.25, 7.25, 8.25, 0.25, 0.25, 5.25, 8.25, 7.25]
            ),
        )

    def test_best_price_idx(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].best_price_idx.values, np.array([2, 3, 1, 0])
        )
        np.testing.assert_array_almost_equal(
            exit_trades.best_price_idx.values,
            np.array([2, 3, 1, 0, 0, 0, 0, 0, 2, 3, 1, 0, 0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_best_price_idx(entry_price_open=True).values,
            np.array([2, 3, 1, 0, 0, 0, 0, 0, 2, 3, 1, 0, 0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_best_price_idx(exit_price_close=True).values,
            np.array([2, 3, 1, 0, 0, 0, 0, 0, 2, 3, 1, 0, 0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_best_price_idx(entry_price_open=True, exit_price_close=True).values,
            np.array([2, 3, 1, 0, 0, 0, 0, 0, 2, 3, 1, 0, 0]),
        )

    def test_worst_price_idx(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].worst_price_idx.values, np.array([0, 0, 0, 0])
        )
        np.testing.assert_array_almost_equal(
            exit_trades.worst_price_idx.values,
            np.array([0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 1, 0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_worst_price_idx(entry_price_open=True).values,
            np.array([0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 1, 0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_worst_price_idx(exit_price_close=True).values,
            np.array([0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 1, 0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_worst_price_idx(entry_price_open=True, exit_price_close=True).values,
            np.array([0, 0, 0, 0, 2, 3, 1, 0, 0, 0, 0, 1, 0]),
        )

    def test_mfe(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].mfe.values,
            np.array([1.9090909090909092, 0.2909090909090912, 1.0, 0.0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.mfe.values,
            np.array(
                [
                    1.9090909090909092,
                    0.2909090909090912,
                    1.0,
                    0.0,
                    0.09090909090909083,
                    0.00909090909090909,
                    0.0,
                    0.0,
                    1.9090909090909092,
                    0.2909090909090912,
                    1.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_mfe(entry_price_open=True).values,
            np.array(
                [
                    1.9090909090909092,
                    0.2909090909090912,
                    1.0,
                    0.5,
                    0.8409090909090908,
                    0.08409090909090916,
                    0.75,
                    1.5,
                    1.9090909090909092,
                    0.2909090909090912,
                    1.0,
                    0.75,
                    0.25,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_mfe(exit_price_close=True).values,
            np.array(
                [
                    2.159090909090909,
                    0.3159090909090912,
                    1.25,
                    0.0,
                    0.09090909090909083,
                    0.00909090909090909,
                    0.0,
                    0.0,
                    2.159090909090909,
                    0.3159090909090912,
                    1.25,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_mfe(entry_price_open=True, exit_price_close=True).values,
            np.array(
                [
                    2.159090909090909,
                    0.3159090909090912,
                    1.25,
                    0.5,
                    0.8409090909090908,
                    0.08409090909090916,
                    0.75,
                    1.5,
                    2.159090909090909,
                    0.3159090909090912,
                    1.25,
                    0.75,
                    0.25,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.mfe_returns.values,
            np.array(
                [
                    1.7500000000000002,
                    2.666666666666667,
                    0.16666666666666666,
                    0.0,
                    0.09090909090909083,
                    0.09090909090909083,
                    0.0,
                    0.0,
                    1.7500000000000002,
                    2.666666666666667,
                    0.16666666666666666,
                    0.0,
                    0.0,
                ]
            ),
        )

    def test_mae(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].mae.values,
            np.array([-0.09090909090909083, -0.00909090909090909, 0.0, 0.0]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.mae.values,
            np.array(
                [
                    -0.09090909090909083,
                    -0.00909090909090909,
                    0.0,
                    0.0,
                    -1.9090909090909092,
                    -0.2909090909090912,
                    -1.0,
                    0.0,
                    -0.09090909090909083,
                    -0.00909090909090909,
                    0.0,
                    -1.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_mae(entry_price_open=True).values,
            np.array(
                [
                    -0.8409090909090908,
                    -0.08409090909090916,
                    -0.75,
                    -1.5,
                    -1.9090909090909092,
                    -0.2909090909090912,
                    -1.0,
                    -0.5,
                    -0.8409090909090908,
                    -0.08409090909090916,
                    -0.75,
                    -1.0,
                    -0.75,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_mae(exit_price_close=True).values,
            np.array(
                [
                    -0.09090909090909083,
                    -0.00909090909090909,
                    0.0,
                    0.0,
                    -2.159090909090909,
                    -0.3159090909090912,
                    -1.25,
                    0.0,
                    -0.09090909090909083,
                    -0.00909090909090909,
                    0.0,
                    -1.25,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_mae(entry_price_open=True, exit_price_close=True).values,
            np.array(
                [
                    -0.8409090909090908,
                    -0.08409090909090916,
                    -0.75,
                    -1.5,
                    -2.159090909090909,
                    -0.3159090909090912,
                    -1.25,
                    -0.5,
                    -0.8409090909090908,
                    -0.08409090909090916,
                    -0.75,
                    -1.25,
                    -0.75,
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.mae_returns.values,
            np.array(
                [
                    -0.08333333333333326,
                    -0.08333333333333326,
                    0.0,
                    0.0,
                    -0.6363636363636364,
                    -0.7272727272727273,
                    -0.14285714285714285,
                    0.0,
                    -0.08333333333333326,
                    -0.08333333333333326,
                    0.0,
                    -0.125,
                    0.0,
                ]
            ),
        )

    def test_expanding_mfe(self):
        assert_frame_equal(
            exit_trades["a"].expanding_mfe,
            pd.DataFrame(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.1590909090909092, 0.11590909090909102, 1.0, np.nan],
                    [1.9090909090909092, 0.2159090909090911, np.nan, np.nan],
                    [np.nan, 0.2909090909090912, np.nan, np.nan],
                ],
                columns=pd.Index([0, 1, 2, 3], dtype="int64", name="id"),
            ),
        )
        assert_frame_equal(
            exit_trades.expanding_mfe,
            pd.DataFrame(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.09090909090909083,
                        0.00909090909090909,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        1.1590909090909092,
                        0.11590909090909102,
                        1.0,
                        np.nan,
                        0.09090909090909083,
                        0.00909090909090909,
                        0.0,
                        np.nan,
                        1.1590909090909092,
                        0.11590909090909102,
                        1.0,
                        0.0,
                        np.nan,
                    ],
                    [
                        1.9090909090909092,
                        0.2159090909090911,
                        np.nan,
                        np.nan,
                        0.09090909090909083,
                        0.00909090909090909,
                        np.nan,
                        np.nan,
                        1.9090909090909092,
                        0.2159090909090911,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        0.2909090909090912,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.00909090909090909,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.2909090909090912,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("a", 3),
                        ("b", 0),
                        ("b", 1),
                        ("b", 2),
                        ("b", 3),
                        ("c", 0),
                        ("c", 1),
                        ("c", 2),
                        ("c", 3),
                        ("c", 4),
                    ],
                    names=[None, "id"],
                ),
            ),
        )
        assert_frame_equal(
            exit_trades.expanding_mfe_returns,
            pd.DataFrame(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.09090909090909083,
                        0.09090909090909083,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        1.0625000000000002,
                        1.0625000000000002,
                        0.16666666666666666,
                        np.nan,
                        0.09090909090909083,
                        0.09090909090909083,
                        0.0,
                        np.nan,
                        1.0625000000000002,
                        1.0625000000000002,
                        0.16666666666666666,
                        0.0,
                        np.nan,
                    ],
                    [
                        1.7500000000000002,
                        1.979166666666667,
                        np.nan,
                        np.nan,
                        0.09090909090909083,
                        0.09090909090909083,
                        np.nan,
                        np.nan,
                        1.7500000000000002,
                        1.979166666666667,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        2.666666666666667,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.09090909090909083,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.666666666666667,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("a", 3),
                        ("b", 0),
                        ("b", 1),
                        ("b", 2),
                        ("b", 3),
                        ("c", 0),
                        ("c", 1),
                        ("c", 2),
                        ("c", 3),
                        ("c", 4),
                    ],
                    names=[None, "id"],
                ),
            ),
        )
        assert_frame_equal(
            exit_trades.get_expanding_mfe(chunked=True),
            exit_trades.get_expanding_mfe(chunked=False),
        )

    def test_expanding_mae(self):
        assert_frame_equal(
            exit_trades["a"].expanding_mae,
            pd.DataFrame(
                [
                    [-0.09090909090909083, -0.00909090909090909, 0.0, 0.0],
                    [-0.09090909090909083, -0.00909090909090909, 0.0, np.nan],
                    [-0.09090909090909083, -0.00909090909090909, np.nan, np.nan],
                    [np.nan, -0.00909090909090909, np.nan, np.nan],
                ],
                columns=pd.Index([0, 1, 2, 3], dtype="int64", name="id"),
            ),
        )
        assert_frame_equal(
            exit_trades.expanding_mae,
            pd.DataFrame(
                [
                    [
                        -0.09090909090909083,
                        -0.00909090909090909,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.09090909090909083,
                        -0.00909090909090909,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        -0.09090909090909083,
                        -0.00909090909090909,
                        0.0,
                        np.nan,
                        -1.1590909090909092,
                        -0.11590909090909102,
                        -1.0,
                        np.nan,
                        -0.09090909090909083,
                        -0.00909090909090909,
                        0.0,
                        -1.0,
                        np.nan,
                    ],
                    [
                        -0.09090909090909083,
                        -0.00909090909090909,
                        np.nan,
                        np.nan,
                        -1.9090909090909092,
                        -0.2159090909090911,
                        np.nan,
                        np.nan,
                        -0.09090909090909083,
                        -0.00909090909090909,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        -0.00909090909090909,
                        np.nan,
                        np.nan,
                        np.nan,
                        -0.2909090909090912,
                        np.nan,
                        np.nan,
                        np.nan,
                        -0.00909090909090909,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("a", 3),
                        ("b", 0),
                        ("b", 1),
                        ("b", 2),
                        ("b", 3),
                        ("c", 0),
                        ("c", 1),
                        ("c", 2),
                        ("c", 3),
                        ("c", 4),
                    ],
                    names=[None, "id"],
                ),
            ),
        )
        assert_frame_equal(
            exit_trades.expanding_mae_returns,
            pd.DataFrame(
                [
                    [
                        -0.08333333333333326,
                        -0.08333333333333326,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.08333333333333326,
                        -0.08333333333333326,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        -0.08333333333333326,
                        -0.08333333333333326,
                        0.0,
                        np.nan,
                        -0.5151515151515151,
                        -0.5151515151515151,
                        -0.14285714285714285,
                        np.nan,
                        -0.08333333333333326,
                        -0.08333333333333326,
                        0.0,
                        -0.125,
                        np.nan,
                    ],
                    [
                        -0.08333333333333326,
                        -0.08333333333333326,
                        np.nan,
                        np.nan,
                        -0.6363636363636364,
                        -0.6643356643356644,
                        np.nan,
                        np.nan,
                        -0.08333333333333326,
                        -0.08333333333333326,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        -0.08333333333333326,
                        np.nan,
                        np.nan,
                        np.nan,
                        -0.7272727272727273,
                        np.nan,
                        np.nan,
                        np.nan,
                        -0.08333333333333326,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ],
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a", 0),
                        ("a", 1),
                        ("a", 2),
                        ("a", 3),
                        ("b", 0),
                        ("b", 1),
                        ("b", 2),
                        ("b", 3),
                        ("c", 0),
                        ("c", 1),
                        ("c", 2),
                        ("c", 3),
                        ("c", 4),
                    ],
                    names=[None, "id"],
                ),
            ),
        )
        assert_frame_equal(
            exit_trades.get_expanding_mae(chunked=True),
            exit_trades.get_expanding_mae(chunked=False),
        )

    def test_edge_ratio(self):
        assert np.isnan(exit_trades["a"].edge_ratio)
        np.testing.assert_array_almost_equal(
            exit_trades.edge_ratio.values,
            np.array([np.nan, np.nan, np.nan, np.nan]),
        )
        assert exit_trades["a"].get_edge_ratio(volatility=1) == 32.00000000000003
        np.testing.assert_array_almost_equal(
            exit_trades.get_edge_ratio(volatility=1).values,
            np.array([32.00000000000003, 0.031249999999999976, 2.9090909090909096, np.nan]),
        )
        assert exit_trades["a"].get_edge_ratio(volatility=1, max_duration=2) == 26.25000000000002
        np.testing.assert_array_almost_equal(
            exit_trades.get_edge_ratio(volatility=1, max_duration=2).values,
            np.array([26.25000000000002, 0.038095238095238064, 2.3863636363636367, np.nan]),
        )

    def test_running_edge_ratio(self):
        np.testing.assert_array_almost_equal(
            exit_trades["a"].running_edge_ratio.values,
            np.array([np.nan, np.nan, np.nan]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.running_edge_ratio.values,
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades["a"].get_running_edge_ratio(volatility=1).values,
            np.array([17.75, 16.25, 26.5]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_running_edge_ratio(volatility=1).values,
            np.array(
                [
                    [17.750000000000014, 0.05633802816901404, 2.9583333333333344, np.nan],
                    [16.250000000000014, 0.06153846153846148, 16.250000000000014, np.nan],
                    [26.500000000000025, 0.03773584905660374, 26.500000000000025, np.nan],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades["a"].get_running_edge_ratio(volatility=1, max_duration=2).values,
            np.array([17.75, 16.25]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_running_edge_ratio(volatility=1, max_duration=2).values,
            np.array(
                [
                    [17.750000000000014, 0.05633802816901404, 2.9583333333333344, np.nan],
                    [16.250000000000014, 0.06153846153846148, 16.250000000000014, np.nan],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            exit_trades["a"].get_running_edge_ratio(volatility=1, incl_shorter=True).values,
            np.array([17.750000000000014, 26.25000000000002, 31.50000000000003]),
        )
        np.testing.assert_array_almost_equal(
            exit_trades.get_running_edge_ratio(volatility=1, incl_shorter=True).values,
            np.array(
                [
                    [17.750000000000014, 0.05633802816901404, 2.9583333333333344, np.nan],
                    [26.25000000000002, 0.038095238095238064, 2.3863636363636367, np.nan],
                    [31.50000000000003, 0.031746031746031717, 2.8636363636363646, np.nan],
                ]
            ),
        )

    def test_long_records(self):
        assert isinstance(exit_trades.direction_long, vbt.ExitTrades)
        assert exit_trades.direction_long.wrapper == exit_trades.wrapper
        assert_records_close(
            exit_trades["a"].direction_long.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (3, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2),
                ],
                dtype=trade_dt,
            ),
        )
        assert_records_close(
            exit_trades["a"].direction_long.values, exit_trades.direction_long["a"].values
        )
        assert_records_close(
            exit_trades.direction_long.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (3, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2),
                    (
                        0,
                        2,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
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
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        2,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (4, 2, 1.0, 6, 7, 8.0, 0.08, -1, 7, 8.0, 0.0, -0.08, -0.01, 0, 0, 3),
                ],
                dtype=trade_dt,
            ),
        )

    def test_short_records(self):
        assert isinstance(exit_trades.direction_short, vbt.ExitTrades)
        assert exit_trades.direction_short.wrapper == exit_trades.wrapper
        assert_records_close(exit_trades["a"].direction_short.values, np.array([], dtype=trade_dt))
        assert_records_close(
            exit_trades["a"].direction_short.values, exit_trades.direction_short["a"].values
        )
        assert_records_close(
            exit_trades.direction_short.values,
            np.array(
                [
                    (
                        0,
                        1,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        -1.9500000000000002,
                        -1.7875000000000003,
                        1,
                        1,
                        0,
                    ),
                    (
                        1,
                        1,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        -0.29600000000000026,
                        -2.7133333333333334,
                        1,
                        1,
                        0,
                    ),
                    (
                        2,
                        1,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        -1.1300000000000001,
                        -0.18833333333333335,
                        1,
                        1,
                        1,
                    ),
                    (3, 1, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 1, 0, 2),
                    (
                        3,
                        2,
                        1.0,
                        5,
                        6,
                        7.0,
                        0.07,
                        6,
                        7,
                        8.0,
                        0.08,
                        -1.1500000000000001,
                        -0.1642857142857143,
                        1,
                        1,
                        2,
                    ),
                ],
                dtype=trade_dt,
            ),
        )

    def test_open_records(self):
        assert isinstance(exit_trades.status_open, vbt.ExitTrades)
        assert exit_trades.status_open.wrapper == exit_trades.wrapper
        assert_records_close(
            exit_trades["a"].status_open.values,
            np.array(
                [(3, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2)],
                dtype=trade_dt,
            ),
        )
        assert_records_close(
            exit_trades["a"].status_open.values, exit_trades.status_open["a"].values
        )
        assert_records_close(
            exit_trades.status_open.values,
            np.array(
                [
                    (3, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2),
                    (3, 1, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 1, 0, 2),
                    (4, 2, 1.0, 6, 7, 8.0, 0.08, -1, 7, 8.0, 0.0, -0.08, -0.01, 0, 0, 3),
                ],
                dtype=trade_dt,
            ),
        )

    def test_closed_records(self):
        assert isinstance(exit_trades.status_closed, vbt.ExitTrades)
        assert exit_trades.status_closed.wrapper == exit_trades.wrapper
        assert_records_close(
            exit_trades["a"].status_closed.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                ],
                dtype=trade_dt,
            ),
        )
        assert_records_close(
            exit_trades["a"].status_closed.values, exit_trades.status_closed["a"].values
        )
        assert_records_close(
            exit_trades.status_closed.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (
                        0,
                        1,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        -1.9500000000000002,
                        -1.7875000000000003,
                        1,
                        1,
                        0,
                    ),
                    (
                        1,
                        1,
                        0.10000000000000009,
                        0,
                        0,
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        -0.29600000000000026,
                        -2.7133333333333334,
                        1,
                        1,
                        0,
                    ),
                    (
                        2,
                        1,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        -1.1300000000000001,
                        -0.18833333333333335,
                        1,
                        1,
                        1,
                    ),
                    (
                        0,
                        2,
                        1.0,
                        0,
                        0,
                        1.0909090909090908,
                        0.010909090909090908,
                        2,
                        2,
                        3.0,
                        0.03,
                        1.8681818181818182,
                        1.7125000000000001,
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
                        1.0909090909090908,
                        0.0010909090909090918,
                        3,
                        3,
                        4.0,
                        0.004,
                        0.2858181818181821,
                        2.62,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        2,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (
                        3,
                        2,
                        1.0,
                        5,
                        6,
                        7.0,
                        0.07,
                        6,
                        7,
                        8.0,
                        0.08,
                        -1.1500000000000001,
                        -0.1642857142857143,
                        1,
                        1,
                        2,
                    ),
                ],
                dtype=trade_dt,
            ),
        )

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "First Trade Start",
                "Last Trade End",
                "Coverage",
                "Overlap Coverage",
                "Total Records",
                "Total Long Trades",
                "Total Short Trades",
                "Total Closed Trades",
                "Total Open Trades",
                "Open Trade PnL",
                "Win Rate [%]",
                "Max Win Streak",
                "Max Loss Streak",
                "Best Trade [%]",
                "Worst Trade [%]",
                "Avg Winning Trade [%]",
                "Avg Losing Trade [%]",
                "Avg Winning Trade Duration",
                "Avg Losing Trade Duration",
                "Profit Factor",
                "Expectancy",
                "SQN",
                "Edge Ratio",
            ],
            dtype="object",
        )
        assert_series_equal(
            exit_trades.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("5 days 08:00:00"),
                    pd.Timedelta("2 days 00:00:00"),
                    3.25,
                    2.0,
                    1.25,
                    2.5,
                    0.75,
                    -0.1,
                    58.333333333333336,
                    2.0,
                    1.3333333333333333,
                    168.38888888888889,
                    -91.08730158730158,
                    149.25,
                    -86.3670634920635,
                    pd.Timedelta("2 days 00:00:00"),
                    pd.Timedelta("1 days 12:00:00"),
                    np.inf,
                    0.11705555555555548,
                    0.18931590012681135,
                    np.nan,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            exit_trades.stats(settings=dict(incl_open=True)),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("5 days 08:00:00"),
                    pd.Timedelta("2 days 00:00:00"),
                    3.25,
                    2.0,
                    1.25,
                    2.5,
                    0.75,
                    -0.1,
                    58.333333333333336,
                    2.0,
                    2.3333333333333335,
                    174.33333333333334,
                    -96.25396825396825,
                    149.25,
                    -42.39781746031746,
                    pd.Timedelta("2 days 00:00:00"),
                    pd.Timedelta("1 days 06:00:00"),
                    7.11951219512195,
                    0.06359999999999993,
                    0.07356215977397455,
                    np.nan,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            exit_trades.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    pd.Timedelta("2 days 00:00:00"),
                    4,
                    4,
                    0,
                    3,
                    1,
                    -0.16,
                    100.0,
                    3,
                    0,
                    262.0,
                    14.499999999999998,
                    149.25,
                    np.nan,
                    pd.Timedelta("2 days 00:00:00"),
                    pd.NaT,
                    np.inf,
                    1.008,
                    2.181955050824476,
                    np.nan,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            exit_trades.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    8,
                    4,
                    4,
                    6,
                    2,
                    -0.32,
                    50.0,
                    3,
                    3,
                    262.0,
                    -271.3333333333333,
                    149.25,
                    -156.30555555555557,
                    pd.Timedelta("2 days 00:00:00"),
                    pd.Timedelta("2 days 00:00:00"),
                    0.895734597156398,
                    -0.058666666666666756,
                    -0.10439051512510047,
                    np.nan,
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_index_equal(
            exit_trades.stats(tags="trades").index,
            pd.Index(
                [
                    "First Trade Start",
                    "Last Trade End",
                    "Total Long Trades",
                    "Total Short Trades",
                    "Total Closed Trades",
                    "Total Open Trades",
                    "Open Trade PnL",
                    "Win Rate [%]",
                    "Max Win Streak",
                    "Max Loss Streak",
                    "Best Trade [%]",
                    "Worst Trade [%]",
                    "Avg Winning Trade [%]",
                    "Avg Losing Trade [%]",
                    "Avg Winning Trade Duration",
                    "Avg Losing Trade Duration",
                    "Profit Factor",
                    "Expectancy",
                    "SQN",
                    "Edge Ratio",
                ],
                dtype="object",
            ),
        )
        assert_series_equal(exit_trades["c"].stats(), exit_trades.stats(column="c"))
        assert_series_equal(exit_trades["c"].stats(), exit_trades.stats(column="c", group_by=False))
        assert_series_equal(
            exit_trades_grouped["g2"].stats(), exit_trades_grouped.stats(column="g2")
        )
        assert_series_equal(
            exit_trades_grouped["g2"].stats(),
            exit_trades.stats(column="g2", group_by=group_by),
        )
        stats_df = exit_trades.stats(agg_func=None)
        assert stats_df.shape == (4, 26)
        assert_index_equal(stats_df.index, exit_trades.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_resample(self):
        np.testing.assert_array_equal(
            exit_trades.resample("1h").start_idx.values,
            np.array([0, 0, 120, 168, 0, 0, 120, 168, 0, 0, 120, 144, 168]),
        )
        np.testing.assert_array_equal(
            exit_trades.resample("1h").end_idx.values,
            np.array([48, 72, 144, 168, 48, 72, 144, 168, 48, 72, 144, 168, 168]),
        )
        np.testing.assert_array_equal(
            exit_trades.resample("10h").start_idx.values,
            np.array([0, 0, 12, 16, 0, 0, 12, 16, 0, 0, 12, 14, 16]),
        )
        np.testing.assert_array_equal(
            exit_trades.resample("10h").end_idx.values,
            np.array([4, 7, 14, 16, 4, 7, 14, 16, 4, 7, 14, 16, 16]),
        )
        np.testing.assert_array_equal(
            exit_trades.resample("3d").start_idx.values,
            np.array([0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 2]),
        )
        np.testing.assert_array_equal(
            exit_trades.resample("3d").end_idx.values,
            np.array([0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2]),
        )
        assert_frame_equal(
            exit_trades.resample("1h").close,
            exit_trades.close.resample("1h").last().astype(float_),
        )
        assert_frame_equal(
            exit_trades.resample("10h").close,
            exit_trades.close.resample("10h").last().astype(float_),
        )
        assert_frame_equal(
            exit_trades.resample("3d").close,
            exit_trades.close.resample("3d").last().astype(float_),
        )
        assert_frame_equal(
            exit_trades.resample("1h", ffill_close=True).close,
            exit_trades.close.resample("1h").last().ffill().astype(float_),
        )
        assert_frame_equal(
            exit_trades.resample("10h", ffill_close=True).close,
            exit_trades.close.resample("10h").last().ffill().astype(float_),
        )
        assert_frame_equal(
            exit_trades.resample("3d", ffill_close=True).close,
            exit_trades.close.resample("3d").last().ffill().astype(float_),
        )
        assert_frame_equal(
            exit_trades.resample("1h", fbfill_close=True).close,
            exit_trades.close.resample("1h").last().ffill().bfill().astype(float_),
        )
        assert_frame_equal(
            exit_trades.resample("10h", fbfill_close=True).close,
            exit_trades.close.resample("10h").last().ffill().bfill().astype(float_),
        )
        assert_frame_equal(
            exit_trades.resample("3d", fbfill_close=True).close,
            exit_trades.close.resample("3d").last().ffill().bfill().astype(float_),
        )


entry_trades = vbt.EntryTrades.from_orders(orders)
entry_trades_grouped = vbt.EntryTrades.from_orders(orders_grouped)


class TestEntryTrades:
    def test_records_arr(self):
        assert_records_close(
            entry_trades.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.0,
                        0,
                        0,
                        1.0,
                        0.01,
                        3,
                        3,
                        3.0909090909090904,
                        0.03090909090909091,
                        2.05,
                        2.05,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0.1,
                        1,
                        1,
                        2.0,
                        0.002,
                        3,
                        3,
                        3.0909090909090904,
                        0.003090909090909091,
                        0.10399999999999998,
                        0.5199999999999999,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (3, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2),
                    (
                        0,
                        1,
                        1.0,
                        0,
                        0,
                        1.0,
                        0.01,
                        3,
                        3,
                        3.0909090909090904,
                        0.03090909090909091,
                        -2.131818181818181,
                        -2.131818181818181,
                        1,
                        1,
                        0,
                    ),
                    (
                        1,
                        1,
                        0.1,
                        1,
                        1,
                        2.0,
                        0.002,
                        3,
                        3,
                        3.0909090909090904,
                        0.003090909090909091,
                        -0.11418181818181816,
                        -0.5709090909090908,
                        1,
                        1,
                        0,
                    ),
                    (
                        2,
                        1,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        -1.1300000000000001,
                        -0.18833333333333335,
                        1,
                        1,
                        1,
                    ),
                    (3, 1, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 1, 0, 2),
                    (
                        0,
                        2,
                        1.0,
                        0,
                        0,
                        1.0,
                        0.01,
                        3,
                        3,
                        3.0909090909090904,
                        0.03090909090909091,
                        2.05,
                        2.05,
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
                        2.0,
                        0.002,
                        3,
                        3,
                        3.0909090909090904,
                        0.003090909090909091,
                        0.10399999999999998,
                        0.5199999999999999,
                        0,
                        1,
                        0,
                    ),
                    (
                        2,
                        2,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (
                        3,
                        2,
                        1.0,
                        5,
                        6,
                        7.0,
                        0.07,
                        6,
                        7,
                        8.0,
                        0.08,
                        -1.1500000000000001,
                        -0.1642857142857143,
                        1,
                        1,
                        2,
                    ),
                    (4, 2, 1.0, 6, 7, 8.0, 0.08, -1, 7, 8.0, 0.0, -0.08, -0.01, 0, 0, 3),
                ],
                dtype=trade_dt,
            ),
        )
        reversed_col_orders = orders.replace(
            records_arr=np.concatenate(
                (
                    orders.values[orders.values["col"] == 2],
                    orders.values[orders.values["col"] == 1],
                    orders.values[orders.values["col"] == 0],
                )
            )
        )
        assert_records_close(
            vbt.EntryTrades.from_orders(reversed_col_orders).values, entry_trades.values
        )
        assert_records_close(
            vbt.EntryTrades.from_orders(orders, jitted=dict(parallel=True)).values,
            vbt.EntryTrades.from_orders(orders, jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            vbt.EntryTrades.from_orders(orders, chunked=True).values,
            vbt.EntryTrades.from_orders(orders, chunked=False).values,
        )

    def test_records_readable(self):
        records_readable = entry_trades.records_readable

        np.testing.assert_array_equal(
            records_readable["Entry Trade Id"].values,
            np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4]),
        )
        np.testing.assert_array_equal(
            records_readable["Position Id"].values,
            np.array([0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 3]),
        )


positions = vbt.Positions.from_trades(exit_trades)
positions_grouped = vbt.Positions.from_trades(exit_trades_grouped)


class TestPositions:
    def test_records_arr(self):
        assert_records_close(
            positions.values,
            np.array(
                [
                    (
                        0,
                        0,
                        1.1,
                        0,
                        0,
                        1.0909090909090908,
                        0.012,
                        3,
                        3,
                        3.090909090909091,
                        0.034,
                        2.1540000000000004,
                        1.7950000000000004,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        0,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (2, 0, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 0, 0, 2),
                    (
                        0,
                        1,
                        1.1,
                        0,
                        0,
                        1.0909090909090908,
                        0.012,
                        3,
                        3,
                        3.090909090909091,
                        0.034,
                        -2.246,
                        -1.8716666666666668,
                        1,
                        1,
                        0,
                    ),
                    (
                        1,
                        1,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        -1.1300000000000001,
                        -0.18833333333333335,
                        1,
                        1,
                        1,
                    ),
                    (2, 1, 2.0, 6, 7, 8.0, 0.16, -1, 7, 8.0, 0.0, -0.16, -0.01, 1, 0, 2),
                    (
                        0,
                        2,
                        1.1,
                        0,
                        0,
                        1.0909090909090908,
                        0.012,
                        3,
                        3,
                        3.090909090909091,
                        0.034,
                        2.1540000000000004,
                        1.7950000000000004,
                        0,
                        1,
                        0,
                    ),
                    (
                        1,
                        2,
                        1.0,
                        4,
                        5,
                        6.0,
                        0.06,
                        5,
                        6,
                        7.0,
                        0.07,
                        0.8699999999999999,
                        0.145,
                        0,
                        1,
                        1,
                    ),
                    (
                        2,
                        2,
                        1.0,
                        5,
                        6,
                        7.0,
                        0.07,
                        6,
                        7,
                        8.0,
                        0.08,
                        -1.1500000000000001,
                        -0.1642857142857143,
                        1,
                        1,
                        2,
                    ),
                    (3, 2, 1.0, 6, 7, 8.0, 0.08, -1, 7, 8.0, 0.0, -0.08, -0.01, 0, 0, 3),
                ],
                dtype=trade_dt,
            ),
        )
        reversed_col_trades = exit_trades.replace(
            records_arr=np.concatenate(
                (
                    exit_trades.values[exit_trades.values["col"] == 2],
                    exit_trades.values[exit_trades.values["col"] == 1],
                    exit_trades.values[exit_trades.values["col"] == 0],
                )
            )
        )
        assert_records_close(
            vbt.Positions.from_trades(reversed_col_trades).values, positions.values
        )
        assert_records_close(
            vbt.Positions.from_trades(entry_trades, jitted=dict(parallel=True)).values,
            vbt.Positions.from_trades(entry_trades, jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            vbt.Positions.from_trades(entry_trades, chunked=True).values,
            vbt.Positions.from_trades(entry_trades, chunked=False).values,
        )
        assert_records_close(
            vbt.Positions.from_trades(exit_trades, jitted=dict(parallel=True)).values,
            vbt.Positions.from_trades(exit_trades, jitted=dict(parallel=False)).values,
        )
        assert_records_close(
            vbt.Positions.from_trades(exit_trades, chunked=True).values,
            vbt.Positions.from_trades(exit_trades, chunked=False).values,
        )

    def test_records_readable(self):
        records_readable = positions.records_readable

        np.testing.assert_array_equal(
            records_readable["Position Id"].values, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
        )
        assert "Parent Id" not in records_readable.columns


# ############# logs ############# #

logs = vbt.Portfolio.from_orders(close, size, fees=0.01, log=True, freq="1 days").logs
logs_grouped = logs.regroup(group_by)


class TestLogs:
    def test_indexing(self):
        logs2 = logs.loc["2020-01-03":"2020-01-04", ["a", "c"]]
        assert_index_equal(logs2.wrapper.index, logs.wrapper.index[2:4])
        assert_index_equal(logs2.wrapper.columns, logs.wrapper.columns[[0, 2]])
        assert_records_close(
            logs2.values,
            np.array(
                [
                    (
                        2,
                        0,
                        0,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        98.788,
                        1.1,
                        0.0,
                        0.0,
                        98.788,
                        3.0,
                        102.088,
                        -1.0,
                        np.inf,
                        0,
                        2,
                        0.01,
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
                        1.0,
                        3.0,
                        0.03,
                        1,
                        0,
                        -1,
                        101.758,
                        0.10000000000000009,
                        0.0,
                        0.0,
                        101.758,
                        3.0,
                        102.088,
                        2,
                    ),
                    (
                        3,
                        0,
                        0,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        101.758,
                        0.10000000000000009,
                        0.0,
                        0.0,
                        101.758,
                        4.0,
                        102.158,
                        -0.1,
                        np.inf,
                        0,
                        2,
                        0.01,
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
                        0.1,
                        4.0,
                        0.004,
                        1,
                        0,
                        -1,
                        102.154,
                        0.0,
                        0.0,
                        0.0,
                        102.154,
                        4.0,
                        102.158,
                        3,
                    ),
                    (
                        2,
                        2,
                        1,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        98.788,
                        1.1,
                        0.0,
                        0.0,
                        98.788,
                        3.0,
                        102.088,
                        -1.0,
                        np.inf,
                        0,
                        2,
                        0.01,
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
                        1.0,
                        3.0,
                        0.03,
                        1,
                        0,
                        -1,
                        101.758,
                        0.10000000000000009,
                        0.0,
                        0.0,
                        101.758,
                        3.0,
                        102.088,
                        2,
                    ),
                    (
                        3,
                        2,
                        1,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        101.758,
                        0.10000000000000009,
                        0.0,
                        0.0,
                        101.758,
                        4.0,
                        102.158,
                        -0.1,
                        np.inf,
                        0,
                        2,
                        0.01,
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
                        0.1,
                        4.0,
                        0.004,
                        1,
                        0,
                        -1,
                        102.154,
                        0.0,
                        0.0,
                        0.0,
                        102.154,
                        4.0,
                        102.158,
                        3,
                    ),
                ],
                dtype=log_dt,
            ),
        )

    def test_mapped_fields(self):
        for name in log_dt.names:
            np.testing.assert_array_equal(getattr(logs, name).values, logs.values[name])

    def test_records_readable(self):
        records_readable = logs.records_readable

        np.testing.assert_array_equal(
            records_readable["Log Id"].values,
            np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Index"].values,
            np.array(
                [
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                    "2020-01-01T00:00:00.000000000",
                    "2020-01-02T00:00:00.000000000",
                    "2020-01-03T00:00:00.000000000",
                    "2020-01-04T00:00:00.000000000",
                    "2020-01-05T00:00:00.000000000",
                    "2020-01-06T00:00:00.000000000",
                    "2020-01-07T00:00:00.000000000",
                    "2020-01-08T00:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Column"].values,
            np.array(
                [
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "b",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "c",
                    "d",
                    "d",
                    "d",
                    "d",
                    "d",
                    "d",
                    "d",
                    "d",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Group"].values,
            np.array(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[PA] Open"].values,
            np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[PA] High"].values,
            np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[PA] Low"].values,
            np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[PA] Close"].values,
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST0] Cash"].values,
            np.array(
                [
                    100.0,
                    98.99,
                    98.788,
                    101.758,
                    102.154,
                    102.154,
                    96.094,
                    103.024,
                    100.0,
                    100.99,
                    101.18799999999999,
                    98.15799999999999,
                    97.75399999999999,
                    97.75399999999999,
                    103.69399999999999,
                    96.624,
                    100.0,
                    98.99,
                    98.788,
                    101.758,
                    102.154,
                    102.154,
                    96.094,
                    109.95400000000001,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST0] Position"].values,
            np.array(
                [
                    0.0,
                    1.0,
                    1.1,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.1,
                    -0.10000000000000009,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.1,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST0] Debt"].values,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.2,
                    0.10909090909090917,
                    0.0,
                    0.0,
                    6.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    7.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST0] Locked Cash"].values,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.2,
                    0.10909090909090917,
                    0.0,
                    0.0,
                    6.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    7.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST0] Free Cash"].values,
            np.array(
                [
                    100.0,
                    98.99,
                    98.788,
                    101.758,
                    102.154,
                    102.154,
                    96.094,
                    103.024,
                    100.0,
                    98.99,
                    98.788,
                    97.93981818181818,
                    97.754,
                    97.754,
                    91.694,
                    96.624,
                    100.0,
                    98.99,
                    98.788,
                    101.758,
                    102.154,
                    102.154,
                    96.094,
                    95.95400000000001,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST0] Valuation Price"].values,
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST0] Value"].values,
            np.array(
                [
                    100.0,
                    100.99,
                    102.088,
                    102.158,
                    102.154,
                    102.154,
                    103.094,
                    103.024,
                    100.0,
                    98.99,
                    97.88799999999999,
                    97.75799999999998,
                    97.75399999999999,
                    97.75399999999999,
                    96.69399999999999,
                    96.624,
                    100.0,
                    100.99,
                    102.088,
                    102.158,
                    102.154,
                    102.154,
                    103.094,
                    101.95400000000001,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Size"].values,
            np.array(
                [
                    1.0,
                    0.1,
                    -1.0,
                    -0.1,
                    np.nan,
                    1.0,
                    -1.0,
                    2.0,
                    -1.0,
                    -0.1,
                    1.0,
                    0.1,
                    np.nan,
                    -1.0,
                    1.0,
                    -2.0,
                    1.0,
                    0.1,
                    -1.0,
                    -0.1,
                    np.nan,
                    1.0,
                    -2.0,
                    2.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Price"].values,
            np.array(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Size Type"].values,
            np.array(
                [
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                    "Amount",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Direction"].values,
            np.array(
                [
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                    "Both",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Fees"].values,
            np.array(
                [
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Fixed Fees"].values,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Slippage"].values,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Min Size"].values,
            np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Max Size"].values,
            np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Size Granularity"].values,
            np.array(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Leverage"].values,
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Leverage Mode"].values,
            np.array(
                [
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                    "Lazy",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Rejection Prob"].values,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Price Area Violation Mode"].values,
            np.array(
                [
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                    "Ignore",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Allow Partial"].values,
            np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Raise Rejection"].values,
            np.array(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[REQ] Log"].values,
            np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[RES] Size"].values,
            np.array(
                [
                    1.0,
                    0.1,
                    1.0,
                    0.1,
                    np.nan,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    0.1,
                    1.0,
                    0.1,
                    np.nan,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    0.1,
                    1.0,
                    0.1,
                    np.nan,
                    1.0,
                    2.0,
                    2.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[RES] Price"].values,
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    np.nan,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    np.nan,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    np.nan,
                    6.0,
                    7.0,
                    8.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[RES] Fees"].values,
            np.array(
                [
                    0.01,
                    0.002,
                    0.03,
                    0.004,
                    np.nan,
                    0.06,
                    0.07,
                    0.16,
                    0.01,
                    0.002,
                    0.03,
                    0.004,
                    np.nan,
                    0.06,
                    0.07,
                    0.16,
                    0.01,
                    0.002,
                    0.03,
                    0.004,
                    np.nan,
                    0.06,
                    0.14,
                    0.16,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[RES] Side"].values,
            np.array(
                [
                    "Buy",
                    "Buy",
                    "Sell",
                    "Sell",
                    None,
                    "Buy",
                    "Sell",
                    "Buy",
                    "Sell",
                    "Sell",
                    "Buy",
                    "Buy",
                    None,
                    "Sell",
                    "Buy",
                    "Sell",
                    "Buy",
                    "Buy",
                    "Sell",
                    "Sell",
                    None,
                    "Buy",
                    "Sell",
                    "Buy",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[RES] Status"].values,
            np.array(
                [
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Ignored",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Ignored",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Ignored",
                    "Filled",
                    "Filled",
                    "Filled",
                    "Ignored",
                    "Ignored",
                    "Ignored",
                    "Ignored",
                    "Ignored",
                    "Ignored",
                    "Ignored",
                    "Ignored",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[RES] Status Info"].values,
            np.array(
                [
                    None,
                    None,
                    None,
                    None,
                    "SizeNaN",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "SizeNaN",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "SizeNaN",
                    None,
                    None,
                    None,
                    "SizeNaN",
                    "SizeNaN",
                    "SizeNaN",
                    "SizeNaN",
                    "SizeNaN",
                    "SizeNaN",
                    "SizeNaN",
                    "SizeNaN",
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST1] Cash"].values,
            np.array(
                [
                    98.99,
                    98.788,
                    101.758,
                    102.154,
                    102.154,
                    96.094,
                    103.024,
                    86.864,
                    100.99,
                    101.18799999999999,
                    98.15799999999999,
                    97.75399999999999,
                    97.75399999999999,
                    103.69399999999999,
                    96.624,
                    112.464,
                    98.99,
                    98.788,
                    101.758,
                    102.154,
                    102.154,
                    96.094,
                    109.95400000000001,
                    93.79400000000001,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST1] Position"].values,
            np.array(
                [
                    1.0,
                    1.1,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    2.0,
                    -1.0,
                    -1.1,
                    -0.10000000000000009,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -2.0,
                    1.0,
                    1.1,
                    0.10000000000000009,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST1] Debt"].values,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.2,
                    0.10909090909090917,
                    0.0,
                    0.0,
                    6.0,
                    0.0,
                    16.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    7.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST1] Locked Cash"].values,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.2,
                    0.10909090909090917,
                    0.0,
                    0.0,
                    6.0,
                    0.0,
                    16.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    7.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST1] Free Cash"].values,
            np.array(
                [
                    98.99,
                    98.788,
                    101.758,
                    102.154,
                    102.154,
                    96.094,
                    103.024,
                    86.864,
                    98.99,
                    98.788,
                    97.93981818181818,
                    97.754,
                    97.754,
                    91.694,
                    96.624,
                    80.464,
                    98.99,
                    98.788,
                    101.758,
                    102.154,
                    102.154,
                    96.094,
                    95.95400000000001,
                    93.79400000000001,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST1] Valuation Price"].values,
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["[ST1] Value"].values,
            np.array(
                [
                    100.0,
                    100.99,
                    102.088,
                    102.158,
                    102.154,
                    102.154,
                    103.094,
                    103.024,
                    100.0,
                    98.99,
                    97.88799999999999,
                    97.75799999999998,
                    97.75399999999999,
                    97.75399999999999,
                    96.69399999999999,
                    96.624,
                    100.0,
                    100.99,
                    102.088,
                    102.158,
                    102.154,
                    102.154,
                    103.094,
                    101.95400000000001,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            records_readable["Order Id"].values,
            np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    -1,
                    4,
                    5,
                    6,
                    0,
                    1,
                    2,
                    3,
                    -1,
                    4,
                    5,
                    6,
                    0,
                    1,
                    2,
                    3,
                    -1,
                    4,
                    5,
                    6,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ),
        )

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Records",
                "Status Counts: Filled",
                "Status Counts: Ignored",
                "Status Counts: Rejected",
                "Status Info Counts: None",
                "Status Info Counts: SizeNaN",
            ],
            dtype="object",
        )
        assert_series_equal(
            logs.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    8.0,
                    5.25,
                    2.75,
                    0.0,
                    5.25,
                    2.75,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            logs.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    8,
                    7,
                    1,
                    0,
                    7,
                    1,
                ],
                index=stats_index,
                name="a",
            ),
        )
        assert_series_equal(
            logs.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-08 00:00:00"),
                    pd.Timedelta("8 days 00:00:00"),
                    16,
                    14,
                    2,
                    0,
                    14,
                    2,
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(logs["c"].stats(), logs.stats(column="c"))
        assert_series_equal(logs["c"].stats(), logs.stats(column="c", group_by=False))
        assert_series_equal(logs_grouped["g2"].stats(), logs_grouped.stats(column="g2"))
        assert_series_equal(logs_grouped["g2"].stats(), logs.stats(column="g2", group_by=group_by))
        stats_df = logs.stats(agg_func=None)
        assert stats_df.shape == (4, 9)
        assert_index_equal(stats_df.index, logs.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_count(self):
        assert logs["a"].count() == 8
        assert_series_equal(
            logs.count(),
            pd.Series(
                np.array([8, 8, 8, 8]), index=pd.Index(["a", "b", "c", "d"], dtype="object")
            ).rename("count"),
        )
        assert_series_equal(
            logs_grouped.count(),
            pd.Series(np.array([16, 16]), index=pd.Index(["g1", "g2"], dtype="object")).rename(
                "count"
            ),
        )

    def test_resample(self):
        np.testing.assert_array_equal(
            logs.resample("1h").idx_arr,
            np.array(
                [
                    0,
                    24,
                    48,
                    72,
                    96,
                    120,
                    144,
                    168,
                    0,
                    24,
                    48,
                    72,
                    96,
                    120,
                    144,
                    168,
                    0,
                    24,
                    48,
                    72,
                    96,
                    120,
                    144,
                    168,
                    0,
                    24,
                    48,
                    72,
                    96,
                    120,
                    144,
                    168,
                ]
            ),
        )
        np.testing.assert_array_equal(
            logs.resample("10h").idx_arr,
            np.array(
                [
                    0,
                    2,
                    4,
                    7,
                    9,
                    12,
                    14,
                    16,
                    0,
                    2,
                    4,
                    7,
                    9,
                    12,
                    14,
                    16,
                    0,
                    2,
                    4,
                    7,
                    9,
                    12,
                    14,
                    16,
                    0,
                    2,
                    4,
                    7,
                    9,
                    12,
                    14,
                    16,
                ]
            ),
        )
        np.testing.assert_array_equal(
            logs.resample("3d").idx_arr,
            np.array(
                [
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                ]
            ),
        )
