import os

import pytest
import vectorbtpro as vbt
from vectorbtpro.base import indexes, indexing, reshaping

from tests.utils import *

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


# ############# reshaping ############# #


class TestReshaping:
    def test_soft_to_ndim(self):
        np.testing.assert_array_equal(reshaping.soft_to_ndim(a2, 1), a2)
        assert_series_equal(reshaping.soft_to_ndim(sr2, 1), sr2)
        assert_series_equal(reshaping.soft_to_ndim(df2, 1), df2.iloc[:, 0])
        assert_frame_equal(reshaping.soft_to_ndim(df4, 1), df4)  # cannot -> do nothing
        np.testing.assert_array_equal(reshaping.soft_to_ndim(a2, 2), a2[:, None])
        assert_frame_equal(reshaping.soft_to_ndim(sr2, 2), sr2.to_frame())
        assert_frame_equal(reshaping.soft_to_ndim(df2, 2), df2)

    def test_to_1d(self):
        np.testing.assert_array_equal(reshaping.to_1d(None), np.array([None]))
        np.testing.assert_array_equal(reshaping.to_1d(0), np.array([0]))
        np.testing.assert_array_equal(reshaping.to_1d(a2), a2)
        assert_series_equal(reshaping.to_1d(sr2), sr2)
        assert_series_equal(reshaping.to_1d(df2), df2.iloc[:, 0])
        np.testing.assert_array_equal(reshaping.to_1d(df2, raw=True), df2.iloc[:, 0].values)

    def test_to_2d(self):
        np.testing.assert_array_equal(reshaping.to_2d(None), np.array([[None]]))
        np.testing.assert_array_equal(reshaping.to_2d(0), np.array([[0]]))
        np.testing.assert_array_equal(reshaping.to_2d(a2), a2[:, None])
        assert_frame_equal(reshaping.to_2d(sr2), sr2.to_frame())
        assert_frame_equal(reshaping.to_2d(df2), df2)
        np.testing.assert_array_equal(reshaping.to_2d(df2, raw=True), df2.values)

    def test_repeat_axis0(self):
        target = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(reshaping.repeat(0, 3, axis=0), np.full(3, 0))
        np.testing.assert_array_equal(reshaping.repeat(a2, 3, axis=0), target)
        assert_series_equal(
            reshaping.repeat(sr2, 3, axis=0),
            pd.Series(target, index=indexes.repeat_index(sr2.index, 3), name=sr2.name),
        )
        assert_frame_equal(
            reshaping.repeat(df2, 3, axis=0),
            pd.DataFrame(target, index=indexes.repeat_index(df2.index, 3), columns=df2.columns),
        )

    def test_repeat_axis1(self):
        target = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(reshaping.repeat(0, 3, axis=1), np.full((1, 3), 0))
        np.testing.assert_array_equal(reshaping.repeat(a2, 3, axis=1), target)
        assert_frame_equal(
            reshaping.repeat(sr2, 3, axis=1),
            pd.DataFrame(target, index=sr2.index, columns=indexes.repeat_index([sr2.name], 3)),
        )
        assert_frame_equal(
            reshaping.repeat(df2, 3, axis=1),
            pd.DataFrame(target, index=df2.index, columns=indexes.repeat_index(df2.columns, 3)),
        )

    def test_tile_axis0(self):
        target = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(reshaping.tile(0, 3, axis=0), np.full(3, 0))
        np.testing.assert_array_equal(reshaping.tile(a2, 3, axis=0), target)
        assert_series_equal(
            reshaping.tile(sr2, 3, axis=0),
            pd.Series(target, index=indexes.tile_index(sr2.index, 3), name=sr2.name),
        )
        assert_frame_equal(
            reshaping.tile(df2, 3, axis=0),
            pd.DataFrame(target, index=indexes.tile_index(df2.index, 3), columns=df2.columns),
        )

    def test_tile_axis1(self):
        target = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(reshaping.tile(0, 3, axis=1), np.full((1, 3), 0))
        np.testing.assert_array_equal(reshaping.tile(a2, 3, axis=1), target)
        assert_frame_equal(
            reshaping.tile(sr2, 3, axis=1),
            pd.DataFrame(target, index=sr2.index, columns=indexes.tile_index([sr2.name], 3)),
        )
        assert_frame_equal(
            reshaping.tile(df2, 3, axis=1),
            pd.DataFrame(target, index=df2.index, columns=indexes.tile_index(df2.columns, 3)),
        )

    def test_broadcast_numpy(self):
        # 1d
        broadcasted_arrs = list(np.broadcast_arrays(0, a1, a2))
        broadcasted = reshaping.broadcast(0, a1, a2)
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(broadcasted[i], broadcasted_arrs[i])
        # 2d
        broadcasted_arrs = list(np.broadcast_arrays(0, a1, a2[:, None], a3, a4, a5))
        broadcasted = reshaping.broadcast(0, a1, a2, a3, a4, a5)
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(broadcasted[i], broadcasted_arrs[i])

    def test_broadcast_axis(self):
        x1 = np.array([1])
        x2 = np.array([[1, 2, 3]])
        x3 = np.array([[1], [2], [3]])
        dct = dict(x1=x1, x2=x2, x3=x3)
        out_dct = reshaping.broadcast(dct, axis=0)
        np.testing.assert_array_equal(out_dct["x1"], np.array([[1], [1], [1]]))
        np.testing.assert_array_equal(out_dct["x2"], np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
        np.testing.assert_array_equal(out_dct["x3"], np.array([[1], [2], [3]]))
        out_dct = reshaping.broadcast(dct, axis=1)
        np.testing.assert_array_equal(out_dct["x1"], np.array([[1, 1, 1]]))
        np.testing.assert_array_equal(out_dct["x2"], np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(out_dct["x3"], np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))

    def test_broadcast_stack(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from="stack",
            columns_from="stack",
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples(
                        [("x1", "x2"), ("x1", "y2"), ("x1", "z2")], names=["i1", "i2"]
                    ),
                    name=None,
                ),
            )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(
            np.broadcast_arrays(
                *[x if np.asarray(x).ndim != 1 else x[:, None] for x in to_broadcast_a],
                *[x.to_frame() for x in to_broadcast_sr],
                *to_broadcast_df,
            )
        )
        broadcasted = reshaping.broadcast(
            *to_broadcast_a,
            *to_broadcast_sr,
            *to_broadcast_df,
            index_from="stack",
            columns_from="stack",
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples(
                        [
                            ("x1", "x2", "x3", "x4", "x5", "x6"),
                            ("x1", "y2", "x3", "y4", "x5", "y6"),
                            ("x1", "z2", "x3", "z4", "x5", "z6"),
                        ],
                        names=["i1", "i2", "i3", "i4", "i5", "i6"],
                    ),
                    columns=pd.MultiIndex.from_tuples(
                        [
                            ("a3", "a4", "a5", "a6"),
                            ("a3", "a4", "b5", "b6"),
                            ("a3", "a4", "c5", "c6"),
                        ],
                        names=["c3", "c4", "c5", "c6"],
                    ),
                ),
            )

        broadcasted = reshaping.broadcast(
            pd.DataFrame([[1, 2, 3]], columns=pd.Index(["a", "b", "c"], name="i1")),
            pd.DataFrame([[4, 5, 6]], columns=pd.Index(["a", "b", "c"], name="i2")),
            index_from="stack",
            columns_from="stack",
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        assert_frame_equal(
            broadcasted[0],
            pd.DataFrame(
                [[1, 2, 3]],
                columns=pd.MultiIndex.from_tuples(
                    [("a", "a"), ("b", "b"), ("c", "c")], names=["i1", "i2"]
                ),
            ),
        )
        assert_frame_equal(
            broadcasted[1],
            pd.DataFrame(
                [[4, 5, 6]],
                columns=pd.MultiIndex.from_tuples(
                    [("a", "a"), ("b", "b"), ("c", "c")], names=["i1", "i2"]
                ),
            ),
        )

    def test_broadcast_keep(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from="keep",
            columns_from="keep",
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(4):
            assert_series_equal(
                broadcasted[i],
                pd.Series(broadcasted_arrs[i], index=pd.RangeIndex(start=0, stop=3, step=1)),
            )
        assert_series_equal(
            broadcasted[4],
            pd.Series(
                broadcasted_arrs[4], index=pd.Index(["x1", "x1", "x1"], name="i1"), name=sr1.name
            ),
        )
        assert_series_equal(
            broadcasted[5], pd.Series(broadcasted_arrs[5], index=sr2.index, name=sr2.name)
        )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(
            np.broadcast_arrays(
                *[x if np.asarray(x).ndim != 1 else x[:, None] for x in to_broadcast_a],
                *[x.to_frame() for x in to_broadcast_sr],
                *to_broadcast_df,
            )
        )
        broadcasted = reshaping.broadcast(
            *to_broadcast_a,
            *to_broadcast_sr,
            *to_broadcast_df,
            index_from="keep",
            columns_from="keep",
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(7):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=pd.RangeIndex(start=0, stop=3, step=1),
                    columns=pd.RangeIndex(start=0, stop=3, step=1),
                ),
            )
        assert_frame_equal(
            broadcasted[7],
            pd.DataFrame(
                broadcasted_arrs[7],
                index=pd.Index(["x1", "x1", "x1"], dtype="object", name="i1"),
                columns=pd.Index(["a1", "a1", "a1"], dtype="object"),
            ),
        )
        assert_frame_equal(
            broadcasted[8],
            pd.DataFrame(
                broadcasted_arrs[8],
                index=sr2.index,
                columns=pd.Index(["a2", "a2", "a2"], dtype="object"),
            ),
        )
        assert_frame_equal(
            broadcasted[9],
            pd.DataFrame(
                broadcasted_arrs[9],
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.RangeIndex(start=0, stop=3, step=1),
            ),
        )
        assert_frame_equal(
            broadcasted[10],
            pd.DataFrame(
                broadcasted_arrs[10],
                index=pd.Index(["x3", "x3", "x3"], dtype="object", name="i3"),
                columns=pd.Index(["a3", "a3", "a3"], dtype="object", name="c3"),
            ),
        )
        assert_frame_equal(
            broadcasted[11],
            pd.DataFrame(
                broadcasted_arrs[11],
                index=df2.index,
                columns=pd.Index(["a4", "a4", "a4"], dtype="object", name="c4"),
            ),
        )
        assert_frame_equal(
            broadcasted[12],
            pd.DataFrame(
                broadcasted_arrs[12],
                index=pd.Index(["x5", "x5", "x5"], dtype="object", name="i5"),
                columns=df3.columns,
            ),
        )
        assert_frame_equal(
            broadcasted[13],
            pd.DataFrame(broadcasted_arrs[13], index=df4.index, columns=df4.columns),
        )

    def test_broadcast_specify(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from=multi_i,
            columns_from=["name"],  # should translate to Series name
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            assert_series_equal(
                broadcasted[i], pd.Series(broadcasted_arrs[i], index=multi_i, name="name")
            )
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from=multi_i,
            columns_from=[0],  # should translate to None
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            assert_series_equal(
                broadcasted[i], pd.Series(broadcasted_arrs[i], index=multi_i, name=None)
            )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(
            np.broadcast_arrays(
                *[x if np.asarray(x).ndim != 1 else x[:, None] for x in to_broadcast_a],
                *[x.to_frame() for x in to_broadcast_sr],
                *to_broadcast_df,
            )
        )
        broadcasted = reshaping.broadcast(
            *to_broadcast_a,
            *to_broadcast_sr,
            *to_broadcast_df,
            index_from=multi_i,
            columns_from=multi_c,
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(broadcasted_arrs[i], index=multi_i, columns=multi_c),
            )

    def test_broadcast_idx(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from=-1,
            columns_from=-1,  # should translate to Series name
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            assert_series_equal(
                broadcasted[i],
                pd.Series(broadcasted_arrs[i], index=sr2.index, name=sr2.name),
            )
        with pytest.raises(Exception):
            reshaping.broadcast(
                *to_broadcast,
                index_from=0,
                columns_from=0,
                ignore_sr_names=True,
                align_index=False,
                clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
            )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(
            np.broadcast_arrays(
                *[x if np.asarray(x).ndim != 1 else x[:, None] for x in to_broadcast_a],
                *[x.to_frame() for x in to_broadcast_sr],
                *to_broadcast_df,
            )
        )
        broadcasted = reshaping.broadcast(
            *to_broadcast_a,
            *to_broadcast_sr,
            *to_broadcast_df,
            index_from=-1,
            columns_from=-1,
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(broadcasted_arrs[i], index=df4.index, columns=df4.columns),
            )

    def test_broadcast_strict(self):
        # 1d
        to_broadcast = sr1, sr2
        with pytest.raises(Exception):
            reshaping.broadcast(
                *to_broadcast,
                index_from="strict",  # changing index not allowed
                columns_from="stack",
                ignore_sr_names=True,
                align_index=False,
                clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
            )
        # 2d
        to_broadcast = df1, df2
        with pytest.raises(Exception):
            reshaping.broadcast(
                *to_broadcast,
                index_from="stack",
                columns_from="strict",  # changing columns not allowed
                ignore_sr_names=True,
                align_index=False,
                clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
            )

    def test_broadcast_dirty(self):
        # 1d
        to_broadcast = sr2, 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            index_from="stack",
            columns_from="stack",
            ignore_sr_names=False,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=False, drop_redundant=False),
        )
        for i in range(len(broadcasted)):
            assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples(
                        [("x2", "x1", "x2"), ("y2", "x1", "y2"), ("z2", "x1", "z2")],
                        names=["i2", "i1", "i2"],
                    ),
                    name=("a2", "a1", "a2"),
                ),
            )

    def test_broadcast_to_shape(self):
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = []
        for x in to_broadcast:
            if isinstance(x, pd.Series):
                x = x.to_frame()
            elif np.asarray(x).ndim == 1:
                x = x[:, None]
            broadcasted_arrs.append(np.broadcast_to(x, (3, 3)))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            to_shape=(3, 3),
            index_from="stack",
            columns_from="stack",
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples(
                        [("x1", "x2"), ("x1", "y2"), ("x1", "z2")], names=["i1", "i2"]
                    ),
                    columns=None,
                ),
            )

    @pytest.mark.parametrize(
        "test_to_pd",
        [False, [False, False, False, False, False, False]],
    )
    def test_broadcast_to_pd(self, test_to_pd):
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshaping.broadcast(
            *to_broadcast,
            to_pd=test_to_pd,  # to NumPy
            index_from="stack",
            columns_from="stack",
            ignore_sr_names=True,
            align_index=False,
            clean_index_kwargs=dict(drop_duplicates=True, drop_redundant=True),
        )
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(broadcasted[i], broadcasted_arrs[i])

    def test_broadcast_require_kwargs(self):
        a, b = reshaping.broadcast(np.empty((1,)), np.empty((1,)))  # readonly
        assert not a.flags.writeable
        assert not b.flags.writeable
        a, b = reshaping.broadcast(
            np.empty((1,)),
            np.empty((1,)),
            require_kwargs=[{"requirements": "W"}, {}],
        )  # writeable
        assert a.flags.writeable
        assert not b.flags.writeable
        a, b = reshaping.broadcast(
            np.empty((1,)),
            np.empty((1,)),
            require_kwargs=[{"requirements": ("W", "C")}, {}],
        )  # writeable, C order
        assert a.flags.writeable  # writeable since it was copied to make C order
        assert not b.flags.writeable
        assert not np.isfortran(a)
        assert not np.isfortran(b)

    def test_broadcast_mapping(self):
        result = reshaping.broadcast(dict(zero=0, a2=a2, sr2=sr2))
        assert type(result) == dict
        assert_series_equal(result["zero"], pd.Series([0, 0, 0], name=sr2.name, index=sr2.index))
        assert_series_equal(result["a2"], pd.Series([1, 2, 3], name=sr2.name, index=sr2.index))
        assert_series_equal(result["sr2"], pd.Series([1, 2, 3], name=sr2.name, index=sr2.index))

    def test_broadcast_individual(self):
        result = reshaping.broadcast(
            dict(zero=0, a2=a2, sr2=sr2),
            keep_flex={"_def": True, "sr2": False},
            require_kwargs={"_def": dict(dtype=float), "a2": dict(dtype=int)},
        )
        np.testing.assert_array_equal(result["zero"], np.array([[0.0]]))
        np.testing.assert_array_equal(result["a2"], np.array([[1], [2], [3]]))
        assert_series_equal(
            result["sr2"], pd.Series([1.0, 2.0, 3.0], name=sr2.name, index=sr2.index)
        )
        result = reshaping.broadcast(
            dict(
                zero=vbt.BCO(0),
                a2=vbt.BCO(a2, min_ndim=1, require_kwargs=dict(dtype=int)),
                sr2=vbt.BCO(sr2, keep_flex=False),
            ),
            keep_flex=True,
            require_kwargs=dict(dtype=float),
        )
        np.testing.assert_array_equal(result["zero"], np.array([[0.0]]))
        np.testing.assert_array_equal(result["a2"], np.array([1, 2, 3]))
        assert_series_equal(
            result["sr2"], pd.Series([1.0, 2.0, 3.0], name=sr2.name, index=sr2.index)
        )
        result = reshaping.broadcast(
            0,
            a2,
            sr2,
            keep_flex={"_def": True, 2: False},
            min_ndim={1: 1},
            require_kwargs={"_def": dict(dtype=float), 1: dict(dtype=int)},
        )
        np.testing.assert_array_equal(result[0], np.array([[0.0]]))
        np.testing.assert_array_equal(result[1], np.array([1, 2, 3]))
        assert_series_equal(result[2], pd.Series([1.0, 2.0, 3.0], name=sr2.name, index=sr2.index))
        result = reshaping.broadcast(
            0,
            a2,
            sr2,
            keep_flex=[True, True, False],
            min_ndim=[1, 2, 1],
            require_kwargs=[dict(dtype=float), dict(dtype=int), dict(dtype=float)],
        )
        np.testing.assert_array_equal(result[0], np.array([0.0]))
        np.testing.assert_array_equal(result[1], np.array([[1], [2], [3]]))
        assert_series_equal(result[2], pd.Series([1.0, 2.0, 3.0], name=sr2.name, index=sr2.index))

    def test_broadcast_refs(self):
        result = reshaping.broadcast(
            dict(
                a=vbt.Ref("b"),
                b=vbt.Ref("c"),
                c=vbt.BCO(vbt.Ref("d"), keep_flex=True),
                d=vbt.BCO(sr2, keep_flex=False),
            )
        )
        np.testing.assert_array_equal(result["a"], sr2.values[:, None])
        np.testing.assert_array_equal(result["b"], sr2.values[:, None])
        np.testing.assert_array_equal(result["c"], sr2.values[:, None])
        assert_series_equal(result["d"], sr2)

    def test_broadcast_defaults(self):
        result = reshaping.broadcast(
            dict(
                a=vbt.Ref("b"),
                b=vbt.Ref("c"),
                c=vbt.Default(vbt.BCO(vbt.Ref("d"), keep_flex=True)),
                d=vbt.BCO(vbt.Default(sr2), keep_flex=False),
            ),
            keep_wrap_default=False,
        )
        assert not isinstance(result["a"], vbt.Default)
        assert not isinstance(result["b"], vbt.Default)
        assert not isinstance(result["c"], vbt.Default)
        assert not isinstance(result["d"], vbt.Default)

        result = reshaping.broadcast(
            dict(
                a=vbt.Ref("b"),
                b=vbt.Ref("c"),
                c=vbt.Default(vbt.BCO(vbt.Ref("d"), keep_flex=True)),
                d=vbt.BCO(vbt.Default(sr2), keep_flex=False),
            ),
            keep_wrap_default=True,
        )
        assert not isinstance(result["a"], vbt.Default)
        assert not isinstance(result["b"], vbt.Default)
        assert isinstance(result["c"], vbt.Default)
        assert isinstance(result["d"], vbt.Default)
        np.testing.assert_array_equal(result["a"], sr2.values[:, None])
        np.testing.assert_array_equal(result["b"], sr2.values[:, None])
        np.testing.assert_array_equal(result["c"].value, sr2.values[:, None])
        assert_series_equal(result["d"].value, sr2)

    def test_broadcast_none(self):
        result = reshaping.broadcast(
            dict(a=None, b=vbt.Default(None), c=vbt.BCO(vbt.Ref("d")), d=vbt.BCO(None)),
            keep_wrap_default=True,
        )
        assert result["a"] is None
        assert isinstance(result["b"], vbt.Default)
        assert result["b"].value is None
        assert result["c"] is None
        assert result["d"] is None

    def test_broadcast_product(self):
        p = pd.Index([1, 2, 3], name="p")

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p, _p = reshaping.broadcast(
            0,
            a2,
            sr2,
            vbt.Param(p),
            align_index=False,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_p, reshaping.broadcast_to(p.values[None], _sr2_p))

        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            align_index=False,
        )
        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p, _p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            vbt.Param(p),
            align_index=False,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a3_p, _a3.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_df2_p, _df2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_p, reshaping.broadcast_to(np.repeat(p.values, 3)[None], _df2_p))

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            keep_flex=True,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p, _p = reshaping.broadcast(
            0,
            a2,
            sr2,
            vbt.Param(p),
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_p, p.values[None])

        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            keep_flex=True,
            align_index=False,
        )
        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p, _p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            vbt.Param(p),
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_a3_p, np.tile(_a3, (1, 3)))
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_sr2_p, _df2)
        np.testing.assert_array_equal(_p, np.repeat(p.values, 3)[None])

    def test_broadcast_tile(self):
        p = pd.Index([1, 2, 3], name="p")

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p = reshaping.broadcast(
            0,
            a2,
            sr2,
            tile=len(p),
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p)))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p)))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p)))

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p = reshaping.broadcast(
            0,
            a2,
            sr2,
            tile=p,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p), keys=p))

        _0_p, _a2_p, _sr2_p, _p = reshaping.broadcast(
            0,
            a2,
            sr2,
            vbt.Param(p),
            tile=p,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_p, reshaping.broadcast_to(np.tile(p.values, 3)[None], _sr2_p))

        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            align_index=False,
        )
        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            tile=p,
            align_index=False,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_a3_p, _a3.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p), keys=p))
        assert_frame_equal(_df2_p, _df2.vbt.tile(len(p), keys=p))

        _0_p, _a2_p, a3_p, _sr2_p, _df2_p, _p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            vbt.Param(p),
            tile=p,
            align_index=False,
        )
        assert_frame_equal(_0_p, _0.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_a2_p, _a2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_a3_p, _a3.vbt.tile(len(p), keys=p))
        assert_frame_equal(_sr2_p, _sr2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(_df2_p, _df2.vbt.tile(len(p) ** 2, keys=indexes.combine_indexes(p, p)))
        assert_frame_equal(
            _p, reshaping.broadcast_to(np.tile(np.repeat(p.values, 3), 3)[None], _df2_p)
        )

        _0, _a2, _sr2 = reshaping.broadcast(
            0,
            a2,
            sr2,
            keep_flex=True,
            align_index=False,
        )
        _0_p, _a2_p, _sr2_p = reshaping.broadcast(
            0,
            a2,
            sr2,
            tile=p,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_sr2_p, _sr2)

        _0_p, _a2_p, _sr2_p, _p = reshaping.broadcast(
            0,
            a2,
            sr2,
            vbt.Param(p),
            tile=p,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_p, np.tile(p.values, 3)[None])

        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            keep_flex=True,
            align_index=False,
        )
        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            tile=p,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_a3_p, np.tile(_a3, (1, 3)))
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_sr2_p, _df2)

        _0_p, _a2_p, _a3_p, _sr2_p, _df2_p, _p = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            vbt.Param(p),
            tile=p,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0_p, _0)
        np.testing.assert_array_equal(_a2_p, _a2)
        np.testing.assert_array_equal(_a3_p, np.tile(np.tile(_a3, (1, 3)), (1, 3)))
        np.testing.assert_array_equal(_sr2_p, _sr2)
        np.testing.assert_array_equal(_sr2_p, _df2)
        np.testing.assert_array_equal(_p, np.tile(np.repeat(p.values, 3), 3)[None])

    def test_broadcast_level(self):
        result, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param([1, 2]),
                b=vbt.Param([False, True]),
                c=vbt.Param(["x", "y"]),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], np.array([[1, 1, 1, 1, 2, 2, 2, 2]]))
        np.testing.assert_array_equal(
            result["b"], np.array([[False, False, True, True, False, False, True, True]])
        )
        np.testing.assert_array_equal(
            result["c"], np.array([["x", "y", "x", "y", "x", "y", "x", "y"]])
        )
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (1, False, "x"),
                    (1, False, "y"),
                    (1, True, "x"),
                    (1, True, "y"),
                    (2, False, "x"),
                    (2, False, "y"),
                    (2, True, "x"),
                    (2, True, "y"),
                ],
                names=["a", "b", "c"],
            ),
        )
        result, wrapper = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param([1, 2])),
                b=vbt.BCO(vbt.Param([False, True])),
                c=vbt.BCO(vbt.Param(["x", "y"])),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], np.array([[1, 1, 1, 1, 2, 2, 2, 2]]))
        np.testing.assert_array_equal(
            result["b"], np.array([[False, False, True, True, False, False, True, True]])
        )
        np.testing.assert_array_equal(
            result["c"], np.array([["x", "y", "x", "y", "x", "y", "x", "y"]])
        )
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (1, False, "x"),
                    (1, False, "y"),
                    (1, True, "x"),
                    (1, True, "y"),
                    (2, False, "x"),
                    (2, False, "y"),
                    (2, True, "x"),
                    (2, True, "y"),
                ],
                names=["a", "b", "c"],
            ),
        )

        result, wrapper = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param(1)),
                b=vbt.BCO(vbt.Param([False, True])),
                c=vbt.BCO(vbt.Param(["x", "y", "z"])),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], np.array([[1, 1, 1, 1, 1, 1]]))
        np.testing.assert_array_equal(
            result["b"], np.array([[False, False, False, True, True, True]])
        )
        np.testing.assert_array_equal(result["c"], np.array([["x", "y", "z", "x", "y", "z"]]))
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (1, False, "x"),
                    (1, False, "y"),
                    (1, False, "z"),
                    (1, True, "x"),
                    (1, True, "y"),
                    (1, True, "z"),
                ],
                names=["a", "b", "c"],
            ),
        )

        result2, wrapper2 = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param(1, level=0)),
                b=vbt.BCO(vbt.Param([False, True], level=1)),
                c=vbt.BCO(vbt.Param(["x", "y", "z"], level=2)),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], result2["a"])
        np.testing.assert_array_equal(result["b"], result2["b"])
        np.testing.assert_array_equal(result["c"], result2["c"])
        assert_index_equal(wrapper.columns, wrapper2.columns)
        result2, wrapper2 = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param(1, level=0)),
                b=vbt.BCO(vbt.Param([False, True], level=1)),
                c=vbt.BCO(vbt.Param(["x", "y", "z"], level=-1)),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], result2["a"])
        np.testing.assert_array_equal(result["b"], result2["b"])
        np.testing.assert_array_equal(result["c"], result2["c"])
        assert_index_equal(wrapper.columns, wrapper2.columns)

        result, wrapper = reshaping.broadcast(
            dict(
                a=vbt.BCO(vbt.Param(1, level=0)),
                b=vbt.BCO(vbt.Param([False, True], level=1)),
                c=vbt.BCO(vbt.Param(["x", "y", "z"], level=0)),
                sr=pd.Series([1, 2, 3]),
            ),
            keep_flex=True,
            return_wrapper=True,
        )
        np.testing.assert_array_equal(result["a"], np.array([[1, 1, 1, 1, 1, 1]]))
        np.testing.assert_array_equal(
            result["b"], np.array([[False, True, False, True, False, True]])
        )
        np.testing.assert_array_equal(result["c"], np.array([["x", "x", "y", "y", "z", "z"]]))
        assert_index_equal(
            wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (1, "x", False),
                    (1, "x", True),
                    (1, "y", False),
                    (1, "y", True),
                    (1, "z", False),
                    (1, "z", True),
                ],
                names=["a", "c", "b"],
            ),
        )

        with pytest.raises(Exception):
            reshaping.broadcast(
                dict(
                    a=vbt.BCO(vbt.Param(1)),
                    b=vbt.BCO(vbt.Param([False, True], level=0)),
                    c=vbt.BCO(vbt.Param(["x", "y", "z"], level=1)),
                )
            )
        with pytest.raises(Exception):
            reshaping.broadcast(
                dict(
                    a=vbt.BCO(vbt.Param(1, level=0)),
                    b=vbt.BCO(vbt.Param([False, True], level=1)),
                    c=vbt.BCO(vbt.Param(["x", "y", "z"])),
                )
            )
        with pytest.raises(Exception):
            reshaping.broadcast(
                dict(
                    a=vbt.BCO(vbt.Param(1, level=0)),
                    b=vbt.BCO(vbt.Param([False, True], level=1)),
                    c=vbt.BCO(vbt.Param(["x", "y", "z"], level=3)),
                )
            )

    def test_broadcast_product_keys(self):
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(
                    pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="a3"), name="a2"),
                    keys=pd.Index(["x", "y", "z"], name="a4"),
                ),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["x", "y", "z"], dtype="object", name="a4"))
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(
                    pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="a3"), name="a2"),
                    keys=pd.Index(["x", "y", "z"]),
                ),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["x", "y", "z"], dtype="object", name="a2"))
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(
                    pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="a3"), name="a2")
                ),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c"], dtype="object", name="a2"))
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]), name="a2")),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c"], dtype="object", name="a2"))
        _, wrapper = reshaping.broadcast(
            dict(
                a=vbt.Param(pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]))),
                sr=pd.Series([1, 2, 3]),
            ),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index(["a", "b", "c"], dtype="object", name="a"))
        _, wrapper = reshaping.broadcast(
            dict(a=vbt.Param(pd.Series([1, 2, 3], name="a2")), sr=pd.Series([1, 2, 3])),
            return_wrapper=True,
        )
        assert_index_equal(wrapper.columns, pd.Index([1, 2, 3], dtype="int64", name="a2"))

    def test_broadcast_meta(self):
        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            keep_flex=True,
            align_index=False,
        )
        np.testing.assert_array_equal(_0, np.array([[0]]))
        np.testing.assert_array_equal(_a2, a2[:, None])
        np.testing.assert_array_equal(_a3, a3)
        np.testing.assert_array_equal(_sr2, sr2.values[:, None])
        np.testing.assert_array_equal(_df2, df2.values)
        _0, _a2, _a3, _sr2, _df2 = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            keep_flex=[False, True, True, True, True],
            align_index=False,
        )
        test_shape = (3, 3)
        test_index = pd.MultiIndex.from_tuples(
            [("x2", "x4"), ("y2", "y4"), ("z2", "z4")], names=["i2", "i4"]
        )
        test_columns = pd.Index(["a4", "a4", "a4"], name="c4", dtype="object")
        assert_frame_equal(
            _0,
            pd.DataFrame(np.zeros(test_shape, dtype=int), index=test_index, columns=test_columns),
        )
        np.testing.assert_array_equal(_a2, a2[:, None])
        np.testing.assert_array_equal(_a3, a3)
        np.testing.assert_array_equal(_sr2, sr2.values[:, None])
        np.testing.assert_array_equal(_df2, df2.values)
        _, wrapper = reshaping.broadcast(
            0,
            a2,
            a3,
            sr2,
            df2,
            return_wrapper=True,
            align_index=False,
        )
        assert wrapper.shape == test_shape
        assert_index_equal(wrapper.index, test_index)
        assert_index_equal(wrapper.columns, test_columns)

    def test_broadcast_align(self):
        index1 = pd.date_range("2020-01-01", periods=3)
        index2 = pd.date_range("2020-01-02", periods=3)
        index3 = pd.date_range("2020-01-03", periods=3)
        columns2 = pd.MultiIndex.from_tuples([(0, "a"), (0, "b"), (1, "a"), (1, "b")])
        columns3 = pd.MultiIndex.from_tuples(
            [
                (2, 0, "a"),
                (2, 0, "b"),
                (2, 1, "a"),
                (2, 1, "b"),
                (3, 0, "a"),
                (3, 0, "b"),
                (3, 1, "a"),
                (3, 1, "b"),
            ]
        )
        sr1 = pd.Series(np.arange(len(index1)), index=index1)
        df2 = pd.DataFrame(
            np.reshape(np.arange(len(index2) * len(columns2)), (len(index2), len(columns2))),
            index=index2,
            columns=columns2,
        )
        df3 = pd.DataFrame(
            np.reshape(np.arange(len(index3) * len(columns3)), (len(index3), len(columns3))),
            index=index3,
            columns=columns3,
        )
        _df1, _df2, _df3 = reshaping.broadcast(sr1, df2, df3, align_index=True, align_columns=True)
        assert_frame_equal(
            _df1,
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.date_range("2020-01-01", periods=5),
                columns=columns3,
            ),
        )
        assert_frame_equal(
            _df2,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0, 4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0, 11.0],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.date_range("2020-01-01", periods=5),
                columns=columns3,
            ),
        )
        assert_frame_equal(
            _df3,
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                    ]
                ),
                index=pd.date_range("2020-01-01", periods=5),
                columns=columns3,
            ),
        )
        _df12, _df22, _df32 = reshaping.broadcast(
            sr1,
            df2,
            df3,
            align_index=True,
            align_columns=True,
            reindex_kwargs=dict(fill_value=0),
        )
        assert_frame_equal(_df12, _df1.fillna(0).astype(int))
        assert_frame_equal(_df22, _df2.fillna(0).astype(int))
        assert_frame_equal(_df32, _df3.fillna(0).astype(int))

    def test_broadcast_special(self):
        i = pd.date_range("2020-01-01", "2020-01-05")
        c = pd.Index(["a", "b", "c"], name="c")
        sr = pd.Series(np.nan, index=i, name=c[0])
        df = pd.DataFrame(np.nan, index=i, columns=c)
        _, obj = reshaping.broadcast(
            sr,
            indexing.index_dict(
                {
                    vbt.rowidx(0): 100,
                    "_def": 0,
                }
            ),
        )
        assert_series_equal(
            obj,
            pd.Series([100, 0, 0, 0, 0], index=i, name=c[0]),
        )
        _, obj = reshaping.broadcast(
            df,
            indexing.index_dict(
                {
                    vbt.idx(1, 1): 100,
                    "_def": 1,
                }
            ),
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
        _, obj = reshaping.broadcast(
            df,
            indexing.index_dict(
                {
                    vbt.idx(1, 1): 100,
                    "_def": 1,
                }
            ),
            keep_flex=True,
        )
        np.testing.assert_array_equal(
            obj,
            np.array(
                [
                    [1, 1, 1],
                    [1, 100, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ]
            ),
        )
        _, obj = reshaping.broadcast(df, vbt.RepEval("wrapper.fill(0)"))
        assert_frame_equal(
            obj,
            pd.DataFrame(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                index=i,
                columns=c,
            ),
        )

    def test_broadcast_to(self):
        np.testing.assert_array_equal(
            reshaping.broadcast_to(0, a5, align_index=False), np.broadcast_to(0, a5.shape)
        )
        assert_series_equal(
            reshaping.broadcast_to(0, sr2, align_index=False),
            pd.Series(np.broadcast_to(0, sr2.shape), index=sr2.index, name=sr2.name),
        )
        assert_frame_equal(
            reshaping.broadcast_to(0, df5, align_index=False),
            pd.DataFrame(np.broadcast_to(0, df5.shape), index=df5.index, columns=df5.columns),
        )
        assert_frame_equal(
            reshaping.broadcast_to(sr2, df5, align_index=False),
            pd.DataFrame(
                np.broadcast_to(sr2.to_frame(), df5.shape), index=df5.index, columns=df5.columns
            ),
        )
        assert_frame_equal(
            reshaping.broadcast_to(sr2, df5, index_from=0, columns_from=0, align_index=False),
            pd.DataFrame(
                np.broadcast_to(sr2.to_frame(), df5.shape),
                index=sr2.index,
                columns=pd.Index(["a2", "a2", "a2"], dtype="object"),
            ),
        )

    @pytest.mark.parametrize(
        "test_input",
        [0, a2, a5, sr2, df5, np.zeros((2, 2, 2))],
    )
    def test_broadcast_to_array_of(self, test_input):
        # broadcasting first element to be an array out of the second argument
        np.testing.assert_array_equal(
            reshaping.broadcast_to_array_of(0.1, test_input),
            np.full((1, *np.asarray(test_input).shape), 0.1),
        )
        np.testing.assert_array_equal(
            reshaping.broadcast_to_array_of([0.1], test_input),
            np.full((1, *np.asarray(test_input).shape), 0.1),
        )
        np.testing.assert_array_equal(
            reshaping.broadcast_to_array_of([0.1, 0.2], test_input),
            np.concatenate(
                (
                    np.full((1, *np.asarray(test_input).shape), 0.1),
                    np.full((1, *np.asarray(test_input).shape), 0.2),
                ),
            ),
        )
        np.testing.assert_array_equal(
            reshaping.broadcast_to_array_of(
                np.expand_dims(np.asarray(test_input), 0), test_input
            ),  # do nothing
            np.expand_dims(np.asarray(test_input), 0),
        )

    def test_broadcast_to_axis_of(self):
        np.testing.assert_array_equal(
            reshaping.broadcast_to_axis_of(10, np.empty((2,)), 0), np.full(2, 10)
        )
        assert reshaping.broadcast_to_axis_of(10, np.empty((2,)), 1) == 10
        np.testing.assert_array_equal(
            reshaping.broadcast_to_axis_of(10, np.empty((2, 3)), 0), np.full(2, 10)
        )
        np.testing.assert_array_equal(
            reshaping.broadcast_to_axis_of(10, np.empty((2, 3)), 1), np.full(3, 10)
        )
        assert reshaping.broadcast_to_axis_of(10, np.empty((2, 3)), 2) == 10

    def test_unstack_to_array(self):
        i = pd.MultiIndex.from_arrays([[1, 1, 2, 2], [3, 4, 3, 4], ["a", "b", "c", "d"]])
        sr = pd.Series([1, 2, 3, 4], index=i)
        np.testing.assert_array_equal(
            reshaping.unstack_to_array(sr),
            np.asarray(
                [
                    [[1.0, np.nan, np.nan, np.nan], [np.nan, 2.0, np.nan, np.nan]],
                    [[np.nan, np.nan, 3.0, np.nan], [np.nan, np.nan, np.nan, 4.0]],
                ]
            ),
        )
        np.testing.assert_array_equal(
            reshaping.unstack_to_array(sr, levels=(0,)), np.array([2.0, 4.0])
        )
        np.testing.assert_array_equal(
            reshaping.unstack_to_array(sr, levels=(2, 0)),
            np.asarray(
                [
                    [1.0, np.nan],
                    [2.0, np.nan],
                    [np.nan, 3.0],
                    [np.nan, 4.0],
                ]
            ),
        )

    def test_make_symmetric(self):
        assert_frame_equal(
            reshaping.make_symmetric(sr2),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, 1.0, 2.0, 3.0],
                        [1.0, np.nan, np.nan, np.nan],
                        [2.0, np.nan, np.nan, np.nan],
                        [3.0, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.Index(["a2", "x2", "y2", "z2"], dtype="object", name=("i2", None)),
                columns=pd.Index(["a2", "x2", "y2", "z2"], dtype="object", name=("i2", None)),
            ),
        )
        assert_frame_equal(
            reshaping.make_symmetric(df2),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, 1.0, 2.0, 3.0],
                        [1.0, np.nan, np.nan, np.nan],
                        [2.0, np.nan, np.nan, np.nan],
                        [3.0, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.Index(["a4", "x4", "y4", "z4"], dtype="object", name=("i4", "c4")),
                columns=pd.Index(["a4", "x4", "y4", "z4"], dtype="object", name=("i4", "c4")),
            ),
        )
        assert_frame_equal(
            reshaping.make_symmetric(df5),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, 1.0, 4.0, 7.0],
                        [np.nan, np.nan, np.nan, 2.0, 5.0, 8.0],
                        [np.nan, np.nan, np.nan, 3.0, 6.0, 9.0],
                        [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
                        [4.0, 5.0, 6.0, np.nan, np.nan, np.nan],
                        [7.0, 8.0, 9.0, np.nan, np.nan, np.nan],
                    ]
                ),
                index=pd.MultiIndex.from_tuples(
                    [
                        ("a7", "a8"),
                        ("b7", "b8"),
                        ("c7", "c8"),
                        ("x7", "x8"),
                        ("y7", "y8"),
                        ("z7", "z8"),
                    ],
                    names=[("i7", "c7"), ("i8", "c8")],
                ),
                columns=pd.MultiIndex.from_tuples(
                    [
                        ("a7", "a8"),
                        ("b7", "b8"),
                        ("c7", "c8"),
                        ("x7", "x8"),
                        ("y7", "y8"),
                        ("z7", "z8"),
                    ],
                    names=[("i7", "c7"), ("i8", "c8")],
                ),
            ),
        )
        assert_frame_equal(
            reshaping.make_symmetric(pd.Series([1, 2, 3], name="yo"), sort=False),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan, 1.0],
                        [np.nan, np.nan, np.nan, 2.0],
                        [np.nan, np.nan, np.nan, 3.0],
                        [1.0, 2.0, 3.0, np.nan],
                    ]
                ),
                index=pd.Index([0, 1, 2, "yo"], dtype="object"),
                columns=pd.Index([0, 1, 2, "yo"], dtype="object"),
            ),
        )

    def test_unstack_to_df(self):
        assert_frame_equal(
            reshaping.unstack_to_df(df5.iloc[0]),
            pd.DataFrame(
                np.array([[1.0, np.nan, np.nan], [np.nan, 2.0, np.nan], [np.nan, np.nan, 3.0]]),
                index=pd.Index(["a7", "b7", "c7"], dtype="object", name="c7"),
                columns=pd.Index(["a8", "b8", "c8"], dtype="object", name="c8"),
            ),
        )
        i = pd.MultiIndex.from_arrays([[1, 1, 2, 2], [3, 4, 3, 4], ["a", "b", "c", "d"]])
        sr = pd.Series([1, 2, 3, 4], index=i)
        assert_frame_equal(
            reshaping.unstack_to_df(sr, index_levels=0, column_levels=1),
            pd.DataFrame(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                index=pd.Index([1, 2], dtype="int64"),
                columns=pd.Index([3, 4], dtype="int64"),
            ),
        )
        assert_frame_equal(
            reshaping.unstack_to_df(sr, index_levels=(0, 1), column_levels=2),
            pd.DataFrame(
                np.array(
                    [
                        [1.0, np.nan, np.nan, np.nan],
                        [np.nan, 2.0, np.nan, np.nan],
                        [np.nan, np.nan, 3.0, np.nan],
                        [np.nan, np.nan, np.nan, 4.0],
                    ]
                ),
                index=pd.MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)]),
                columns=pd.Index(["a", "b", "c", "d"], dtype="object"),
            ),
        )
        assert_frame_equal(
            reshaping.unstack_to_df(sr, index_levels=0, column_levels=1, symmetric=True),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, 1.0, 2.0],
                        [np.nan, np.nan, 3.0, 4.0],
                        [1.0, 3.0, np.nan, np.nan],
                        [2.0, 4.0, np.nan, np.nan],
                    ]
                ),
                index=pd.Index([1, 2, 3, 4], dtype="int64"),
                columns=pd.Index([1, 2, 3, 4], dtype="int64"),
            ),
        )
