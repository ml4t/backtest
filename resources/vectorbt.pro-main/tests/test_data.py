import os
from datetime import datetime, timedelta, timezone

import pytest
import vectorbtpro as vbt
from vectorbtpro._dtypes import *
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import to_timezone

from tests.utils import *

seed = 42


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


# ############# base ############# #


class MyData(vbt.Data):
    @classmethod
    def fetch_key(
        cls,
        key,
        shape=(5, 3),
        start_date=datetime(2020, 1, 1),
        columns=None,
        index_mask=None,
        column_mask=None,
        return_numeric=False,
        return_arr=False,
        tz_localize=None,
        is_update=False,
        return_none=False,
        return_empty=False,
        raise_error=False,
    ):
        if raise_error:
            raise ValueError()
        if return_none:
            return None
        if return_empty:
            if len(shape) == 2:
                a = np.empty((0, shape[1]), dtype=object)
            else:
                a = np.empty((0,), dtype=object)
            if return_arr:
                return a
            if len(shape) == 2:
                return pd.DataFrame(a, columns=columns)
            return pd.Series(a, name=columns)
        np.random.seed(seed)
        a = np.empty(shape, dtype=float_ if return_numeric else object)
        if a.ndim == 1:
            if return_numeric:
                a[:] = np.arange(len(a))
            else:
                for i in range(a.shape[0]):
                    a[i] = str(key) + "_" + str(i)
                    if is_update:
                        a[i] += "_u"
        else:
            if return_numeric:
                a[:, :] = np.arange(a.shape[0] * a.shape[1]).reshape(a.shape)
            else:
                for col in range(a.shape[1]):
                    for i in range(a.shape[0]):
                        if columns is not None:
                            a[i, col] = str(key) + "_" + str(columns[col]) + "_" + str(i)
                        else:
                            a[i, col] = str(key) + "_" + str(col) + "_" + str(i)
                        if is_update:
                            a[i, col] += "_u"
        if return_arr:
            return a
        index = [start_date + timedelta(days=i) for i in range(a.shape[0])]
        if a.ndim == 1:
            sr = pd.Series(a, index=index, name=columns)
            if index_mask is not None:
                sr = sr.loc[index_mask]
            if tz_localize is not None:
                sr = sr.tz_localize(tz_localize)
            return sr
        df = pd.DataFrame(a, index=index, columns=columns)
        if index_mask is not None:
            df = df.loc[index_mask]
        if column_mask is not None:
            df = df.loc[:, column_mask]
        if tz_localize is not None:
            df = df.tz_localize(tz_localize)
        return df

    @classmethod
    def fetch_feature(cls, *args, **kwargs):
        return cls.fetch_key(*args, **kwargs)

    @classmethod
    def fetch_symbol(cls, *args, **kwargs):
        return cls.fetch_key(*args, **kwargs)

    def update_feature(self, feature, n=1, **kwargs):
        fetch_kwargs = self.select_feature_kwargs(feature, self.fetch_kwargs)
        start_date = self.select_last_index(feature)
        shape = fetch_kwargs.pop("shape", (5, 3))
        new_shape = (n, shape[1]) if len(shape) > 1 else (n,)
        kwargs = merge_dicts(fetch_kwargs, dict(start_date=start_date), kwargs)
        return self.fetch_feature(feature, shape=new_shape, is_update=True, **kwargs)

    def update_symbol(self, symbol, n=1, **kwargs):
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        start_date = self.select_last_index(symbol)
        shape = fetch_kwargs.pop("shape", (5, 3))
        new_shape = (n, shape[1]) if len(shape) > 1 else (n,)
        kwargs = merge_dicts(fetch_kwargs, dict(start_date=start_date), kwargs)
        return self.fetch_symbol(symbol, shape=new_shape, is_update=True, **kwargs)


class TestData:
    def test_symbol_row_stack(self):
        data1 = MyData.pull(
            ["S1", "S2"],
            shape=(4, 1),
            columns=["F1"],
            start_date=pd.Timestamp("2020-01-01"),
        ).to_symbol_oriented()
        data2 = MyData.pull(
            ["S1", "S2"],
            shape=(6, 3),
            columns=["F1", "F2", "F3"],
            start_date=pd.Timestamp("2020-01-05"),
        ).to_symbol_oriented()
        new_data = MyData.row_stack((data1, data2))
        assert_index_equal(new_data.wrapper.index, data1.wrapper.index.append(data2.wrapper.index))
        assert_index_equal(new_data.wrapper.columns, data2.wrapper.columns)
        assert_frame_equal(
            new_data.data["S1"],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F1_0", "S1_F1_0"],
                    ["S1_F1_1", "S1_F1_1", "S1_F1_1"],
                    ["S1_F1_2", "S1_F1_2", "S1_F1_2"],
                    ["S1_F1_3", "S1_F1_3", "S1_F1_3"],
                    ["S1_F1_0", "S1_F2_0", "S1_F3_0"],
                    ["S1_F1_1", "S1_F2_1", "S1_F3_1"],
                    ["S1_F1_2", "S1_F2_2", "S1_F3_2"],
                    ["S1_F1_3", "S1_F2_3", "S1_F3_3"],
                    ["S1_F1_4", "S1_F2_4", "S1_F3_4"],
                    ["S1_F1_5", "S1_F2_5", "S1_F3_5"],
                ],
                index=new_data.wrapper.index,
                columns=new_data.wrapper.columns,
            ),
        )
        assert_frame_equal(
            new_data.data["S2"],
            pd.DataFrame(
                [
                    ["S2_F1_0", "S2_F1_0", "S2_F1_0"],
                    ["S2_F1_1", "S2_F1_1", "S2_F1_1"],
                    ["S2_F1_2", "S2_F1_2", "S2_F1_2"],
                    ["S2_F1_3", "S2_F1_3", "S2_F1_3"],
                    ["S2_F1_0", "S2_F2_0", "S2_F3_0"],
                    ["S2_F1_1", "S2_F2_1", "S2_F3_1"],
                    ["S2_F1_2", "S2_F2_2", "S2_F3_2"],
                    ["S2_F1_3", "S2_F2_3", "S2_F3_3"],
                    ["S2_F1_4", "S2_F2_4", "S2_F3_4"],
                    ["S2_F1_5", "S2_F2_5", "S2_F3_5"],
                ],
                index=new_data.wrapper.index,
                columns=new_data.wrapper.columns,
            ),
        )
        assert new_data.fetch_kwargs == data2.fetch_kwargs
        assert new_data.returned_kwargs == data2.returned_kwargs
        assert new_data.last_index == data2.last_index
        with pytest.raises(Exception):
            MyData.row_stack((data1.select(["S1"]), data2))
        with pytest.raises(Exception):
            MyData.row_stack((data1, data2.select(["S1"])))

    def test_feature_row_stack(self):
        data1 = MyData.pull(
            ["S1"],
            shape=(4, 2),
            columns=["F1", "F2"],
            start_date=pd.Timestamp("2020-01-01"),
        ).to_feature_oriented()
        data2 = MyData.pull(
            ["S1", "S2", "S3"],
            shape=(6, 2),
            columns=["F1", "F2"],
            start_date=pd.Timestamp("2020-01-05"),
        ).to_feature_oriented()
        new_data = MyData.row_stack((data1, data2))
        assert_index_equal(new_data.wrapper.index, data1.wrapper.index.append(data2.wrapper.index))
        assert_index_equal(
            new_data.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [("S1", "S1"), ("S1", "S2"), ("S1", "S3")], names=["symbol", "symbol"]
            ),
        )
        assert_frame_equal(
            new_data.data["F1"],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F1_0", "S1_F1_0"],
                    ["S1_F1_1", "S1_F1_1", "S1_F1_1"],
                    ["S1_F1_2", "S1_F1_2", "S1_F1_2"],
                    ["S1_F1_3", "S1_F1_3", "S1_F1_3"],
                    ["S1_F1_0", "S2_F1_0", "S3_F1_0"],
                    ["S1_F1_1", "S2_F1_1", "S3_F1_1"],
                    ["S1_F1_2", "S2_F1_2", "S3_F1_2"],
                    ["S1_F1_3", "S2_F1_3", "S3_F1_3"],
                    ["S1_F1_4", "S2_F1_4", "S3_F1_4"],
                    ["S1_F1_5", "S2_F1_5", "S3_F1_5"],
                ],
                index=new_data.wrapper.index,
                columns=new_data.wrapper.columns,
            ),
        )
        assert_frame_equal(
            new_data.data["F2"],
            pd.DataFrame(
                [
                    ["S1_F2_0", "S1_F2_0", "S1_F2_0"],
                    ["S1_F2_1", "S1_F2_1", "S1_F2_1"],
                    ["S1_F2_2", "S1_F2_2", "S1_F2_2"],
                    ["S1_F2_3", "S1_F2_3", "S1_F2_3"],
                    ["S1_F2_0", "S2_F2_0", "S3_F2_0"],
                    ["S1_F2_1", "S2_F2_1", "S3_F2_1"],
                    ["S1_F2_2", "S2_F2_2", "S3_F2_2"],
                    ["S1_F2_3", "S2_F2_3", "S3_F2_3"],
                    ["S1_F2_4", "S2_F2_4", "S3_F2_4"],
                    ["S1_F2_5", "S2_F2_5", "S3_F2_5"],
                ],
                index=new_data.wrapper.index,
                columns=new_data.wrapper.columns,
            ),
        )
        assert new_data.fetch_kwargs == data2.fetch_kwargs
        assert new_data.returned_kwargs == data2.returned_kwargs
        assert new_data.last_index == data2.last_index
        with pytest.raises(Exception):
            MyData.row_stack((data1.select(["F1"]), data2))
        with pytest.raises(Exception):
            MyData.row_stack((data1, data2.select(["F2"])))

    def test_symbol_column_stack(self):
        data1 = MyData.pull(
            ["S1", "S2"],
            shape=(5, 1),
            columns=["F1"],
        ).to_symbol_oriented()
        data2 = MyData.pull(
            ["S1", "S2"],
            shape=(5, 3),
            columns=["F2", "F3", "F4"],
        ).to_symbol_oriented()
        new_data = MyData.column_stack((data1, data2), fetch_kwargs={"S1": {}, "S2": {}})
        assert_index_equal(new_data.wrapper.index, data1.wrapper.index)
        assert_index_equal(
            new_data.wrapper.columns, data1.wrapper.columns.append(data2.wrapper.columns)
        )
        assert_frame_equal(
            new_data.data["S1"],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0", "S1_F3_0", "S1_F4_0"],
                    ["S1_F1_1", "S1_F2_1", "S1_F3_1", "S1_F4_1"],
                    ["S1_F1_2", "S1_F2_2", "S1_F3_2", "S1_F4_2"],
                    ["S1_F1_3", "S1_F2_3", "S1_F3_3", "S1_F4_3"],
                    ["S1_F1_4", "S1_F2_4", "S1_F3_4", "S1_F4_4"],
                ],
                index=new_data.wrapper.index,
                columns=new_data.wrapper.columns,
            ),
        )
        assert_frame_equal(
            new_data.data["S2"],
            pd.DataFrame(
                [
                    ["S2_F1_0", "S2_F2_0", "S2_F3_0", "S2_F4_0"],
                    ["S2_F1_1", "S2_F2_1", "S2_F3_1", "S2_F4_1"],
                    ["S2_F1_2", "S2_F2_2", "S2_F3_2", "S2_F4_2"],
                    ["S2_F1_3", "S2_F2_3", "S2_F3_3", "S2_F4_3"],
                    ["S2_F1_4", "S2_F2_4", "S2_F3_4", "S2_F4_4"],
                ],
                index=new_data.wrapper.index,
                columns=new_data.wrapper.columns,
            ),
        )
        with pytest.raises(Exception):
            MyData.column_stack((data1, data2))
        with pytest.raises(Exception):
            MyData.column_stack((data1.select(["S1"]), data2))
        with pytest.raises(Exception):
            MyData.column_stack((data1, data2.select(["S1"])))

    def test_feature_column_stack(self):
        data1 = MyData.pull(
            ["S1"],
            shape=(5, 2),
            columns=["F1", "F2"],
        ).to_feature_oriented()
        data2 = MyData.pull(
            ["S2", "S3", "S4"],
            shape=(5, 2),
            columns=["F1", "F2"],
        ).to_feature_oriented()
        new_data = MyData.column_stack((data1, data2))
        assert_index_equal(new_data.wrapper.index, data1.wrapper.index)
        assert_index_equal(
            new_data.wrapper.columns, data1.wrapper.columns.append(data2.wrapper.columns)
        )
        assert_frame_equal(
            new_data.data["F1"],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S2_F1_0", "S3_F1_0", "S4_F1_0"],
                    ["S1_F1_1", "S2_F1_1", "S3_F1_1", "S4_F1_1"],
                    ["S1_F1_2", "S2_F1_2", "S3_F1_2", "S4_F1_2"],
                    ["S1_F1_3", "S2_F1_3", "S3_F1_3", "S4_F1_3"],
                    ["S1_F1_4", "S2_F1_4", "S3_F1_4", "S4_F1_4"],
                ],
                index=new_data.wrapper.index,
                columns=new_data.wrapper.columns,
            ),
        )
        assert_frame_equal(
            new_data.data["F2"],
            pd.DataFrame(
                [
                    ["S1_F2_0", "S2_F2_0", "S3_F2_0", "S4_F2_0"],
                    ["S1_F2_1", "S2_F2_1", "S3_F2_1", "S4_F2_1"],
                    ["S1_F2_2", "S2_F2_2", "S3_F2_2", "S4_F2_2"],
                    ["S1_F2_3", "S2_F2_3", "S3_F2_3", "S4_F2_3"],
                    ["S1_F2_4", "S2_F2_4", "S3_F2_4", "S4_F2_4"],
                ],
                index=new_data.wrapper.index,
                columns=new_data.wrapper.columns,
            ),
        )
        with pytest.raises(Exception):
            MyData.column_stack((data1.select(["F1"]), data2))
        with pytest.raises(Exception):
            MyData.column_stack((data1, data2.select(["F2"])))

    def test_config(self, tmp_path):
        original_data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        for data in [original_data.to_symbol_oriented(), original_data.to_feature_oriented()]:
            assert MyData.loads(data.dumps()) == data
            data.save(tmp_path / "data")
            new_data = MyData.load(tmp_path / "data")
            assert new_data == data
            data.save(tmp_path / "data", file_format="ini")
            new_data = MyData.load(tmp_path / "data", file_format="ini")
            assert new_data == data

    @pytest.mark.parametrize("test_keys_are_features", [False, True])
    def test_fetch(self, test_keys_are_features):
        assert_series_equal(
            MyData.pull(
                "S1", keys_are_features=test_keys_are_features, shape=(5,), return_arr=True
            ).data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"]),
        )
        assert_frame_equal(
            MyData.pull(
                "S1", keys_are_features=test_keys_are_features, shape=(5, 3), return_arr=True
            ).data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_4", "S1_1_4", "S1_2_4"],
                ]
            ),
        )
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,)).data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"], index=index),
        )
        assert_series_equal(
            MyData.pull(
                "S1", keys_are_features=test_keys_are_features, shape=(5,), columns="F1"
            ).data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"], index=index, name="F1"),
        )
        assert_frame_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5, 3)).data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_4", "S1_1_4", "S1_2_4"],
                ],
                index=index,
            ),
        )
        assert_frame_equal(
            MyData.pull(
                "S1",
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                columns=["F1", "F2", "F3"],
            ).data["S1"],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0", "S1_F3_0"],
                    ["S1_F1_1", "S1_F2_1", "S1_F3_1"],
                    ["S1_F1_2", "S1_F2_2", "S1_F3_2"],
                    ["S1_F1_3", "S1_F2_3", "S1_F3_3"],
                    ["S1_F1_4", "S1_F2_4", "S1_F3_4"],
                ],
                index=index,
                columns=pd.Index(["F1", "F2", "F3"], dtype="object"),
            ),
        )
        assert_series_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5,)).data[
                "S1"
            ],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"], index=index),
        )
        assert_series_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5,)).data[
                "S2"
            ],
            pd.Series(["S2_0", "S2_1", "S2_2", "S2_3", "S2_4"], index=index),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3)).data[
                "S1"
            ],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_4", "S1_1_4", "S1_2_4"],
                ],
                index=index,
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3)).data[
                "S2"
            ],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", "S2_2_0"],
                    ["S2_0_1", "S2_1_1", "S2_2_1"],
                    ["S2_0_2", "S2_1_2", "S2_2_2"],
                    ["S2_0_3", "S2_1_3", "S2_2_3"],
                    ["S2_0_4", "S2_1_4", "S2_2_4"],
                ],
                index=index,
            ),
        )
        index2 = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz="utc",
        ).tz_convert(to_timezone("Europe/Berlin"))
        assert_series_equal(
            MyData.pull(
                "S1",
                keys_are_features=test_keys_are_features,
                shape=(5,),
                tz_localize="utc",
                tz_convert="Europe/Berlin",
            ).data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"], index=index2),
        )
        index_mask = vbt.key_dict(
            {"S1": [False, True, True, True, True], "S2": [True, True, True, True, False]}
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            ).data["S1"],
            pd.Series([np.nan, "S1_1", "S1_2", "S1_3", "S1_4"], index=index),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            ).data["S2"],
            pd.Series(["S2_0", "S2_1", "S2_2", "S2_3", np.nan], index=index),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            ).data["S1"],
            pd.Series(["S1_1", "S1_2", "S1_3"], index=index[1:4]),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            ).data["S2"],
            pd.Series(["S2_1", "S2_2", "S2_3"], index=index[1:4]),
        )
        column_mask = vbt.key_dict({"S1": [False, True, True], "S2": [True, True, False]})
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            ).data["S1"],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, "S1_1_1", "S1_2_1"],
                    [np.nan, "S1_1_2", "S1_2_2"],
                    [np.nan, "S1_1_3", "S1_2_3"],
                    [np.nan, "S1_1_4", "S1_2_4"],
                ],
                index=index,
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            ).data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", np.nan],
                    ["S2_0_1", "S2_1_1", np.nan],
                    ["S2_0_2", "S2_1_2", np.nan],
                    ["S2_0_3", "S2_1_3", np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                index=index,
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).data["S1"],
            pd.DataFrame(
                [["S1_1_1"], ["S1_1_2"], ["S1_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).data["S2"],
            pd.DataFrame(
                [["S2_1_1"], ["S2_1_2"], ["S2_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        keys = {
            "S1": dict(index_mask=[False, True, True, True, True]),
            "S2": dict(index_mask=[True, True, True, True, False]),
        }
        assert_series_equal(
            MyData.pull(
                keys, keys_are_features=test_keys_are_features, shape=(5,), missing_index="nan"
            ).data["S1"],
            pd.Series([np.nan, "S1_1", "S1_2", "S1_3", "S1_4"], index=index),
        )
        assert_series_equal(
            MyData.pull(
                keys, keys_are_features=test_keys_are_features, shape=(5,), missing_index="nan"
            ).data["S2"],
            pd.Series(["S2_0", "S2_1", "S2_2", "S2_3", np.nan], index=index),
        )
        assert_series_equal(
            MyData.pull(
                keys, keys_are_features=test_keys_are_features, shape=(5,), missing_index="drop"
            ).data["S1"],
            pd.Series(["S1_1", "S1_2", "S1_3"], index=index[1:4]),
        )
        assert_series_equal(
            MyData.pull(
                keys, keys_are_features=test_keys_are_features, shape=(5,), missing_index="drop"
            ).data["S2"],
            pd.Series(["S2_1", "S2_2", "S2_3"], index=index[1:4]),
        )
        assert (
            len(
                MyData.pull(
                    ["S1", "S2"],
                    keys_are_features=test_keys_are_features,
                    shape=(5, 3),
                    return_none=vbt.key_dict({"S1": True, "S2": False}),
                ).keys
            )
            == 1
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                return_none=vbt.key_dict({"S1": True, "S2": False}),
            ).data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", "S2_2_0"],
                    ["S2_0_1", "S2_1_1", "S2_2_1"],
                    ["S2_0_2", "S2_1_2", "S2_2_2"],
                    ["S2_0_3", "S2_1_3", "S2_2_3"],
                    ["S2_0_4", "S2_1_4", "S2_2_4"],
                ],
                index=index,
            ),
        )
        assert (
            len(
                MyData.pull(
                    ["S1", "S2"],
                    keys_are_features=test_keys_are_features,
                    shape=(5, 3),
                    raise_error=vbt.key_dict({"S1": True, "S2": False}),
                    skip_on_error=True,
                ).keys
            )
            == 1
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                raise_error=vbt.key_dict({"S1": True, "S2": False}),
                skip_on_error=True,
            ).data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", "S2_2_0"],
                    ["S2_0_1", "S2_1_1", "S2_2_1"],
                    ["S2_0_2", "S2_1_2", "S2_2_2"],
                    ["S2_0_3", "S2_1_3", "S2_2_3"],
                    ["S2_0_4", "S2_1_4", "S2_2_4"],
                ],
                index=index,
            ),
        )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                raise_error=vbt.key_dict({"S1": True, "S2": False}),
                skip_on_error=False,
            )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="raise",
                missing_columns="nan",
            )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="raise",
            )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="test",
                missing_columns="nan",
            )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="test",
            )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                return_none=True,
            )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                return_empty=True,
            )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                raise_error=True,
                skip_on_error=False,
            )
        with pytest.raises(Exception):
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                raise_error=True,
                skip_on_error=True,
            )

    @pytest.mark.parametrize("test_keys_are_features", [False, True])
    def test_update(self, test_keys_are_features):
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,), return_arr=True)
            .update()
            .data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_0_u"]),
        )
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,), return_arr=True)
            .update(concat=False)
            .data["S1"],
            pd.Series(["S1_0_u"], index=pd.Index([4], dtype="int64")),
        )
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,), return_arr=True)
            .update(n=2)
            .data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_0_u", "S1_1_u"]),
        )
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,), return_arr=True)
            .update(n=2, concat=False)
            .data["S1"],
            pd.Series(["S1_0_u", "S1_1_u"], index=pd.Index([4, 5], dtype="int64")),
        )
        assert_frame_equal(
            MyData.pull(
                "S1", keys_are_features=test_keys_are_features, shape=(5, 3), return_arr=True
            )
            .update()
            .data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_0_u", "S1_1_0_u", "S1_2_0_u"],
                ]
            ),
        )
        assert_frame_equal(
            MyData.pull(
                "S1", keys_are_features=test_keys_are_features, shape=(5, 3), return_arr=True
            )
            .update(concat=False)
            .data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0_u", "S1_1_0_u", "S1_2_0_u"],
                ],
                index=pd.Index([4], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                "S1", keys_are_features=test_keys_are_features, shape=(5, 3), return_arr=True
            )
            .update(n=2)
            .data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_0_u", "S1_1_0_u", "S1_2_0_u"],
                    ["S1_0_1_u", "S1_1_1_u", "S1_2_1_u"],
                ]
            ),
        )
        assert_frame_equal(
            MyData.pull(
                "S1", keys_are_features=test_keys_are_features, shape=(5, 3), return_arr=True
            )
            .update(n=2, concat=False)
            .data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0_u", "S1_1_0_u", "S1_2_0_u"],
                    ["S1_0_1_u", "S1_1_1_u", "S1_2_1_u"],
                ],
                index=pd.Index([4, 5], dtype="int64"),
            ),
        )
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,))
            .update()
            .data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_0_u"], index=index),
        )
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,))
            .update(concat=False)
            .data["S1"],
            pd.Series(["S1_0_u"], index=index[[-1]]),
        )
        updated_index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
                "2020-01-06 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,))
            .update(n=2)
            .data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_0_u", "S1_1_u"], index=updated_index),
        )
        assert_series_equal(
            MyData.pull("S1", keys_are_features=test_keys_are_features, shape=(5,))
            .update(n=2, concat=False)
            .data["S1"],
            pd.Series(
                ["S1_0_u", "S1_1_u"],
                index=pd.DatetimeIndex(
                    ["2020-01-05 00:00:00+00:00", "2020-01-06 00:00:00+00:00"],
                    dtype="datetime64[ns, UTC]",
                    freq=None,
                ),
            ),
        )
        index2 = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz="utc",
        ).tz_convert(to_timezone("Europe/Berlin"))
        assert_series_equal(
            MyData.pull(
                "S1",
                keys_are_features=test_keys_are_features,
                shape=(5,),
                tz_localize="utc",
                tz_convert="Europe/Berlin",
            )
            .update(tz_localize=None)
            .data["S1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_0_u"], index=index2),
        )
        assert_series_equal(
            MyData.pull(
                "S1",
                keys_are_features=test_keys_are_features,
                shape=(5,),
                tz_localize="utc",
                tz_convert="Europe/Berlin",
            )
            .update(tz_localize=None, concat=False)
            .data["S1"],
            pd.Series(["S1_0_u"], index=index2[[-1]]),
        )
        index_mask = vbt.key_dict(
            {"S1": [False, True, True, True, True], "S2": [True, True, True, True, False]}
        )
        update_index_mask = vbt.key_dict({"S1": [True], "S2": [False]})
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            )
            .update(index_mask=update_index_mask)
            .data["S1"],
            pd.Series([np.nan, "S1_1", "S1_2", "S1_3", "S1_0_u"], index=index),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            )
            .update(index_mask=update_index_mask)
            .data["S2"],
            pd.Series(["S2_0", "S2_1", "S2_2", "S2_3", np.nan], index=index),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data["S1"],
            pd.Series(["S1_0_u"], index=index[[-1]], dtype=object),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data["S2"],
            pd.Series([np.nan], index=index[[-1]], dtype=object),
        )
        update_index_mask2 = vbt.key_dict({"S1": [True, False, False], "S2": [True, False, True]})
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data["S1"],
            pd.Series([np.nan, "S1_1", "S1_2", "S1_3", "S1_0_u", np.nan], index=updated_index),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data["S2"],
            pd.Series(
                [
                    "S2_0",
                    "S2_1",
                    "S2_2",
                    "S2_0_u",
                    np.nan,
                    "S2_2_u",
                ],
                index=updated_index,
            ),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data["S1"],
            pd.Series(["S1_3", "S1_0_u", np.nan], index=updated_index[-3:]),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="nan",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data["S2"],
            pd.Series(
                [
                    "S2_0_u",
                    np.nan,
                    "S2_2_u",
                ],
                index=updated_index[-3:],
            ),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            )
            .update(index_mask=update_index_mask)
            .data["S1"],
            pd.Series(["S1_1", "S1_2", "S1_3"], index=index[1:4]),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            )
            .update(index_mask=update_index_mask)
            .data["S2"],
            pd.Series(["S2_1", "S2_2", "S2_3"], index=index[1:4]),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data["S1"],
            pd.Series(
                [], index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None), dtype=object
            ),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data["S2"],
            pd.Series(
                [], index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None), dtype=object
            ),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data["S1"],
            pd.Series(["S1_1", "S1_2", "S1_3"], index=index[1:4]),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data["S2"],
            pd.Series(["S2_1", "S2_2", "S2_0_u"], index=index[1:4]),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data["S1"],
            pd.Series(
                ["S1_3"],
                index=pd.DatetimeIndex(
                    ["2020-01-04 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
            ),
        )
        assert_series_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5,),
                index_mask=index_mask,
                missing_index="drop",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data["S2"],
            pd.Series(
                ["S2_0_u"],
                index=pd.DatetimeIndex(
                    ["2020-01-04 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
            ),
        )
        column_mask = vbt.key_dict({"S1": [False, True, True], "S2": [True, True, False]})
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask)
            .data["S1"],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, "S1_1_1", "S1_2_1"],
                    [np.nan, "S1_1_2", "S1_2_2"],
                    [np.nan, "S1_1_3", "S1_2_3"],
                    [np.nan, "S1_1_0_u", "S1_2_0_u"],
                ],
                index=index,
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask)
            .data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", np.nan],
                    ["S2_0_1", "S2_1_1", np.nan],
                    ["S2_0_2", "S2_1_2", np.nan],
                    ["S2_0_3", "S2_1_3", np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                index=index,
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data["S1"],
            pd.DataFrame(
                [
                    [np.nan, "S1_1_0_u", "S1_2_0_u"],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-05 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
            ).astype({0: float, 1: object, 2: object}),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data["S2"],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                ],
                index=pd.DatetimeIndex(
                    ["2020-01-05 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
            ).astype({0: object, 1: object, 2: float}),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data["S1"],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, "S1_1_1", "S1_2_1"],
                    [np.nan, "S1_1_2", "S1_2_2"],
                    [np.nan, "S1_1_3", "S1_2_3"],
                    [np.nan, "S1_1_0_u", "S1_2_0_u"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", np.nan],
                    ["S2_0_1", "S2_1_1", np.nan],
                    ["S2_0_2", "S2_1_2", np.nan],
                    ["S2_0_0_u", "S2_1_0_u", np.nan],
                    [np.nan, np.nan, np.nan],
                    ["S2_0_2_u", "S2_1_2_u", np.nan],
                ],
                index=updated_index,
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data["S1"],
            pd.DataFrame(
                [
                    [np.nan, "S1_1_3", "S1_2_3"],
                    [np.nan, "S1_1_0_u", "S1_2_0_u"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index[3:],
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0_u", "S2_1_0_u", np.nan],
                    [np.nan, np.nan, np.nan],
                    ["S2_0_2_u", "S2_1_2_u", np.nan],
                ],
                index=updated_index[3:],
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask)
            .data["S1"],
            pd.DataFrame(
                [["S1_1_1"], ["S1_1_2"], ["S1_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask)
            .data["S2"],
            pd.DataFrame(
                [["S2_1_1"], ["S2_1_2"], ["S2_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data["S1"],
            pd.DataFrame(
                [],
                index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None),
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data["S2"],
            pd.DataFrame(
                [],
                index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None),
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data["S1"],
            pd.DataFrame(
                [["S1_1_1"], ["S1_1_2"], ["S1_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data["S2"],
            pd.DataFrame(
                [["S2_1_1"], ["S2_1_2"], ["S2_1_0_u"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data["S1"],
            pd.DataFrame(
                [["S1_1_3"]],
                index=pd.DatetimeIndex(
                    ["2020-01-04 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert_frame_equal(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data["S2"],
            pd.DataFrame(
                [["S2_1_0_u"]],
                index=pd.DatetimeIndex(
                    ["2020-01-04 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert vbt.key_dict(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).last_index
        ) == vbt.key_dict({"S1": index[4], "S2": index[3]})
        assert vbt.key_dict(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2)
            .last_index
        ) == vbt.key_dict(
            {
                "S1": updated_index[4],
                "S2": updated_index[5],
            }
        )
        assert vbt.key_dict(
            MyData.pull(
                ["S1", "S2"],
                keys_are_features=test_keys_are_features,
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .last_index
        ) == vbt.key_dict(
            {
                "S1": updated_index[4],
                "S2": updated_index[5],
            }
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, return_none=vbt.key_dict({"S1": True, "S2": False}))
            .data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_4", "S1_1_4", "S1_2_4"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, return_none=vbt.key_dict({"S1": True, "S2": False}))
            .data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", "S2_2_0"],
                    ["S2_0_1", "S2_1_1", "S2_2_1"],
                    ["S2_0_2", "S2_1_2", "S2_2_2"],
                    ["S2_0_3", "S2_1_3", "S2_2_3"],
                    ["S2_0_0_u", "S2_1_0_u", "S2_2_0_u"],
                    ["S2_0_1_u", "S2_1_1_u", "S2_2_1_u"],
                ],
                index=updated_index,
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, return_none=vbt.key_dict({"S1": True, "S2": False}), concat=False)
            .data["S1"],
            pd.DataFrame(
                [["S1_0_4", "S1_1_4", "S1_2_4"], [np.nan, np.nan, np.nan]],
                index=pd.DatetimeIndex(
                    ["2020-01-05 00:00:00+00:00", "2020-01-06 00:00:00+00:00"],
                    dtype="datetime64[ns, UTC]",
                    freq=None,
                ),
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, return_none=vbt.key_dict({"S1": True, "S2": False}), concat=False)
            .data["S2"],
            pd.DataFrame(
                [["S2_0_0_u", "S2_1_0_u", "S2_2_0_u"], ["S2_0_1_u", "S2_1_1_u", "S2_2_1_u"]],
                index=pd.DatetimeIndex(
                    ["2020-01-05 00:00:00+00:00", "2020-01-06 00:00:00+00:00"],
                    dtype="datetime64[ns, UTC]",
                    freq=None,
                ),
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, return_empty=vbt.key_dict({"S1": True, "S2": False}))
            .data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_4", "S1_1_4", "S1_2_4"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, return_empty=vbt.key_dict({"S1": True, "S2": False}))
            .data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", "S2_2_0"],
                    ["S2_0_1", "S2_1_1", "S2_2_1"],
                    ["S2_0_2", "S2_1_2", "S2_2_2"],
                    ["S2_0_3", "S2_1_3", "S2_2_3"],
                    ["S2_0_0_u", "S2_1_0_u", "S2_2_0_u"],
                    ["S2_0_1_u", "S2_1_1_u", "S2_2_1_u"],
                ],
                index=updated_index,
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, raise_error=vbt.key_dict({"S1": True, "S2": False}), skip_on_error=True)
            .data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_4", "S1_1_4", "S1_2_4"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, raise_error=vbt.key_dict({"S1": True, "S2": False}), skip_on_error=True)
            .data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", "S2_2_0"],
                    ["S2_0_1", "S2_1_1", "S2_2_1"],
                    ["S2_0_2", "S2_1_2", "S2_2_2"],
                    ["S2_0_3", "S2_1_3", "S2_2_3"],
                    ["S2_0_0_u", "S2_1_0_u", "S2_2_0_u"],
                    ["S2_0_1_u", "S2_1_1_u", "S2_2_1_u"],
                ],
                index=updated_index,
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, raise_error=True, skip_on_error=True)
            .data["S1"],
            pd.DataFrame(
                [
                    ["S1_0_0", "S1_1_0", "S1_2_0"],
                    ["S1_0_1", "S1_1_1", "S1_2_1"],
                    ["S1_0_2", "S1_1_2", "S1_2_2"],
                    ["S1_0_3", "S1_1_3", "S1_2_3"],
                    ["S1_0_4", "S1_1_4", "S1_2_4"],
                ],
                index=index,
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], keys_are_features=test_keys_are_features, shape=(5, 3))
            .update(n=2, raise_error=True, skip_on_error=True)
            .data["S2"],
            pd.DataFrame(
                [
                    ["S2_0_0", "S2_1_0", "S2_2_0"],
                    ["S2_0_1", "S2_1_1", "S2_2_1"],
                    ["S2_0_2", "S2_1_2", "S2_2_2"],
                    ["S2_0_3", "S2_1_3", "S2_2_3"],
                    ["S2_0_4", "S2_1_4", "S2_2_4"],
                ],
                index=index,
            ),
        )

    def test_feature_wrapper(self):
        data = MyData.pull("S1", shape=(5,), columns="F1").to_feature_oriented()
        assert_index_equal(
            data.feature_wrapper.columns,
            pd.Index(["F1"], dtype="object"),
        )
        assert data.to_symbol_oriented().feature_wrapper == data.feature_wrapper
        data = MyData.pull("S1", shape=(5, 1), columns=["F1"]).to_feature_oriented()
        assert_index_equal(
            data.feature_wrapper.columns,
            pd.Index(["F1"], dtype="object"),
        )
        assert data.to_symbol_oriented().feature_wrapper == data.feature_wrapper
        data = MyData.pull("S1", shape=(5, 2), columns=["F1", "F2"]).to_feature_oriented()
        assert_index_equal(
            data.feature_wrapper.columns,
            pd.Index(["F1", "F2"], dtype="object"),
        )
        assert data.to_symbol_oriented().feature_wrapper == data.feature_wrapper
        data = MyData.pull("S1", shape=(5, 2), columns=["F1", "F2"]).to_feature_oriented(
            level_name=True
        )
        assert_index_equal(
            data.feature_wrapper.columns,
            pd.Index(["F1", "F2"], dtype="object", name="feature"),
        )
        assert data.to_symbol_oriented().feature_wrapper == data.feature_wrapper
        data = MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"]).to_feature_oriented()
        assert_index_equal(
            data.feature_wrapper.columns,
            pd.Index(["F1", "F2", "F3"], dtype="object"),
        )
        assert data.to_symbol_oriented().get_feature_wrapper(
            features=["F1", "F3"]
        ) == data.get_feature_wrapper(features=["F1", "F3"])

    def test_symbol_wrapper(self):
        data = MyData.pull("S1", shape=(5,), columns="F1").to_symbol_oriented()
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.Index(["S1"], name="symbol"),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper
        data = MyData.pull(["S1"], shape=(5,), columns="F1").to_symbol_oriented()
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.Index(["S1"], name="symbol"),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper
        data = MyData.pull(["S1", "S2"], shape=(5,), columns="F1").to_symbol_oriented()
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.Index(["S1", "S2"], name="symbol"),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper
        data = MyData.pull(["S1", "S2"], shape=(5,), columns="F1").to_symbol_oriented(
            level_name=False
        )
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.Index(["S1", "S2"]),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper
        data = MyData.pull(["S1", "S2", "S3"], shape=(5,), columns="F1").to_symbol_oriented()
        assert_index_equal(
            data.get_symbol_wrapper(symbols=["S1", "S3"]).columns,
            pd.Index(["S1", "S3"], name="symbol"),
        )
        assert data.to_feature_oriented().get_symbol_wrapper(
            symbols=["S1", "S3"]
        ) == data.get_symbol_wrapper(symbols=["S1", "S3"])
        data = MyData.pull("S1", classes="C1", shape=(5,), columns="F1").to_symbol_oriented()
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.MultiIndex.from_tuples([("C1", "S1")], names=["symbol_class", "symbol"]),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper
        data = MyData.pull(
            "S1", classes=dict(c1="C1", c2="C2"), shape=(5,), columns="F1"
        ).to_symbol_oriented()
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.MultiIndex.from_tuples([("C1", "C2", "S1")], names=["c1", "c2", "symbol"]),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper
        data = MyData.pull(
            ["S1", "S2"], classes="C1", shape=(5,), columns="F1"
        ).to_symbol_oriented()
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.MultiIndex.from_tuples(
                [("C1", "S1"), ("C1", "S2")], names=["symbol_class", "symbol"]
            ),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper
        data = MyData.pull(
            ["S1", "S2"], classes=["C1", "C2"], shape=(5,), columns="F1"
        ).to_symbol_oriented()
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.MultiIndex.from_tuples(
                [("C1", "S1"), ("C2", "S2")], names=["symbol_class", "symbol"]
            ),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper
        data = MyData.pull(
            ["S1", "S2"],
            classes=[dict(c1="C1", c2="C2"), dict(c1="C3", c2="C4")],
            shape=(5,),
            columns="F1",
        ).to_symbol_oriented()
        assert_index_equal(
            data.symbol_wrapper.columns,
            pd.MultiIndex.from_tuples(
                [("C1", "C2", "S1"), ("C3", "C4", "S2")], names=["c1", "c2", "symbol"]
            ),
        )
        assert data.to_feature_oriented().symbol_wrapper == data.symbol_wrapper

    def test_concat(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        assert_series_equal(
            MyData.pull("S1", shape=(5,), columns="F1").concat()["F1"],
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"], index=index, name="S1"),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], shape=(5,), columns="F1").concat()["F1"],
            pd.DataFrame(
                [
                    ["S1_0", "S2_0"],
                    ["S1_1", "S2_1"],
                    ["S1_2", "S2_2"],
                    ["S1_3", "S2_3"],
                    ["S1_4", "S2_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_series_equal(
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"]).concat()["F1"],
            pd.Series(
                ["S1_F1_0", "S1_F1_1", "S1_F1_2", "S1_F1_3", "S1_F1_4"], index=index, name="S1"
            ),
        )
        assert_series_equal(
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"]).concat()["F2"],
            pd.Series(
                ["S1_F2_0", "S1_F2_1", "S1_F2_2", "S1_F2_3", "S1_F2_4"], index=index, name="S1"
            ),
        )
        assert_series_equal(
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"]).concat()["F3"],
            pd.Series(
                ["S1_F3_0", "S1_F3_1", "S1_F3_2", "S1_F3_3", "S1_F3_4"], index=index, name="S1"
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]).concat()["F1"],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S2_F1_0"],
                    ["S1_F1_1", "S2_F1_1"],
                    ["S1_F1_2", "S2_F1_2"],
                    ["S1_F1_3", "S2_F1_3"],
                    ["S1_F1_4", "S2_F1_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]).concat()["F2"],
            pd.DataFrame(
                [
                    ["S1_F2_0", "S2_F2_0"],
                    ["S1_F2_1", "S2_F2_1"],
                    ["S1_F2_2", "S2_F2_2"],
                    ["S1_F2_3", "S2_F2_3"],
                    ["S1_F2_4", "S2_F2_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_frame_equal(
            MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]).concat()["F3"],
            pd.DataFrame(
                [
                    ["S1_F3_0", "S2_F3_0"],
                    ["S1_F3_1", "S2_F3_1"],
                    ["S1_F3_2", "S2_F3_2"],
                    ["S1_F3_3", "S2_F3_3"],
                    ["S1_F3_4", "S2_F3_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )

    def test_get(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        original_data = MyData.pull("S1", shape=(5,), columns="F1")
        assert_series_equal(
            original_data.to_symbol_oriented().get(),
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"], index=index, name="F1"),
        )
        assert_series_equal(
            original_data.to_feature_oriented().get(),
            pd.Series(["S1_0", "S1_1", "S1_2", "S1_3", "S1_4"], index=index, name="S1"),
        )
        original_data = MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"])
        assert_frame_equal(
            original_data.to_symbol_oriented().get(),
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0", "S1_F3_0"],
                    ["S1_F1_1", "S1_F2_1", "S1_F3_1"],
                    ["S1_F1_2", "S1_F2_2", "S1_F3_2"],
                    ["S1_F1_3", "S1_F2_3", "S1_F3_3"],
                    ["S1_F1_4", "S1_F2_4", "S1_F3_4"],
                ],
                index=index,
                columns=pd.Index(["F1", "F2", "F3"], dtype="object"),
            ),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get(),
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0", "S1_F3_0"],
                    ["S1_F1_1", "S1_F2_1", "S1_F3_1"],
                    ["S1_F1_2", "S1_F2_2", "S1_F3_2"],
                    ["S1_F1_3", "S1_F2_3", "S1_F3_3"],
                    ["S1_F1_4", "S1_F2_4", "S1_F3_4"],
                ],
                index=index,
                columns=pd.Index(["F1", "F2", "F3"], dtype="object"),
            ),
        )
        original_data = MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"])
        assert_series_equal(
            original_data.to_symbol_oriented().get("F1"),
            pd.Series(
                ["S1_F1_0", "S1_F1_1", "S1_F1_2", "S1_F1_3", "S1_F1_4"], index=index, name="F1"
            ),
        )
        assert_series_equal(
            original_data.to_feature_oriented().get("F1"),
            pd.Series(
                ["S1_F1_0", "S1_F1_1", "S1_F1_2", "S1_F1_3", "S1_F1_4"], index=index, name="S1"
            ),
        )
        original_data = MyData.pull(["S1", "S2"], shape=(5,), columns="F1")
        assert_frame_equal(
            original_data.to_symbol_oriented().get(),
            pd.DataFrame(
                [
                    ["S1_0", "S2_0"],
                    ["S1_1", "S2_1"],
                    ["S1_2", "S2_2"],
                    ["S1_3", "S2_3"],
                    ["S1_4", "S2_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get(),
            pd.DataFrame(
                [
                    ["S1_0", "S2_0"],
                    ["S1_1", "S2_1"],
                    ["S1_2", "S2_2"],
                    ["S1_3", "S2_3"],
                    ["S1_4", "S2_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        original_data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        assert_frame_equal(
            original_data.to_symbol_oriented().get("F1"),
            pd.DataFrame(
                [
                    ["S1_F1_0", "S2_F1_0"],
                    ["S1_F1_1", "S2_F1_1"],
                    ["S1_F1_2", "S2_F1_2"],
                    ["S1_F1_3", "S2_F1_3"],
                    ["S1_F1_4", "S2_F1_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get("F1"),
            pd.DataFrame(
                [
                    ["S1_F1_0", "S2_F1_0"],
                    ["S1_F1_1", "S2_F1_1"],
                    ["S1_F1_2", "S2_F1_2"],
                    ["S1_F1_3", "S2_F1_3"],
                    ["S1_F1_4", "S2_F1_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        original_data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        assert_frame_equal(
            original_data.to_symbol_oriented().get(["F1", "F2"])[0],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S2_F1_0"],
                    ["S1_F1_1", "S2_F1_1"],
                    ["S1_F1_2", "S2_F1_2"],
                    ["S1_F1_3", "S2_F1_3"],
                    ["S1_F1_4", "S2_F1_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get(["F1", "F2"])[0],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S2_F1_0"],
                    ["S1_F1_1", "S2_F1_1"],
                    ["S1_F1_2", "S2_F1_2"],
                    ["S1_F1_3", "S2_F1_3"],
                    ["S1_F1_4", "S2_F1_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_frame_equal(
            original_data.to_symbol_oriented().get(["F1", "F2"], per="symbol")[0],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0"],
                    ["S1_F1_1", "S1_F2_1"],
                    ["S1_F1_2", "S1_F2_2"],
                    ["S1_F1_3", "S1_F2_3"],
                    ["S1_F1_4", "S1_F2_4"],
                ],
                index=index,
                columns=pd.Index(["F1", "F2"]),
            ),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get(["F1", "F2"], per="symbol")[0],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0"],
                    ["S1_F1_1", "S1_F2_1"],
                    ["S1_F1_2", "S1_F2_2"],
                    ["S1_F1_3", "S1_F2_3"],
                    ["S1_F1_4", "S1_F2_4"],
                ],
                index=index,
                columns=pd.Index(["F1", "F2"]),
            ),
        )
        original_data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        assert_frame_equal(
            original_data.to_symbol_oriented().get()[0],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S2_F1_0"],
                    ["S1_F1_1", "S2_F1_1"],
                    ["S1_F1_2", "S2_F1_2"],
                    ["S1_F1_3", "S2_F1_3"],
                    ["S1_F1_4", "S2_F1_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get()[0],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S2_F1_0"],
                    ["S1_F1_1", "S2_F1_1"],
                    ["S1_F1_2", "S2_F1_2"],
                    ["S1_F1_3", "S2_F1_3"],
                    ["S1_F1_4", "S2_F1_4"],
                ],
                index=index,
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )
        assert_frame_equal(
            original_data.to_symbol_oriented().get(per="symbol")[0],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0", "S1_F3_0"],
                    ["S1_F1_1", "S1_F2_1", "S1_F3_1"],
                    ["S1_F1_2", "S1_F2_2", "S1_F3_2"],
                    ["S1_F1_3", "S1_F2_3", "S1_F3_3"],
                    ["S1_F1_4", "S1_F2_4", "S1_F3_4"],
                ],
                index=index,
                columns=pd.Index(["F1", "F2", "F3"]),
            ),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get(per="symbol")[0],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0", "S1_F3_0"],
                    ["S1_F1_1", "S1_F2_1", "S1_F3_1"],
                    ["S1_F1_2", "S1_F2_2", "S1_F3_2"],
                    ["S1_F1_3", "S1_F2_3", "S1_F3_3"],
                    ["S1_F1_4", "S1_F2_4", "S1_F3_4"],
                ],
                index=index,
                columns=pd.Index(["F1", "F2", "F3"]),
            ),
        )
        original_data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        assert_frame_equal(
            original_data.to_symbol_oriented().get(symbols="S1"),
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"]).to_symbol_oriented().get(),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get(symbols="S1"),
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"]).to_feature_oriented().get(),
        )
        assert_frame_equal(
            original_data.to_symbol_oriented().get(symbols=["S1"])[0],
            MyData.pull(["S1"], shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_symbol_oriented()
            .get()[0],
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get(symbols=["S1"])[0],
            MyData.pull(["S1"], shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_feature_oriented()
            .get()[0],
        )
        assert_series_equal(
            original_data.to_symbol_oriented().get("F1", symbols="S1"),
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_symbol_oriented()
            .get("F1"),
        )
        assert_series_equal(
            original_data.to_feature_oriented().get("F1", symbols="S1"),
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_feature_oriented()
            .get("F1"),
        )
        assert_frame_equal(
            original_data.to_symbol_oriented().get(["F1"], symbols="S1"),
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_symbol_oriented()
            .get(["F1"]),
        )
        assert_frame_equal(
            original_data.to_feature_oriented().get(["F1"], symbols="S1"),
            MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_feature_oriented()
            .get(["F1"]),
        )

    def test_add(self):
        data1 = MyData.pull(["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"])
        data2 = MyData.pull(["S1", "S2"], shape=(5, 2), columns=["F4", "F5"])
        added_data = data1.add("F4", data2.get("F4"))
        added_data = added_data.add("F5", data2.get("F5"))
        assert added_data.features == ["F1", "F2", "F3", "F4", "F5"]
        assert added_data.symbols == ["S1", "S2", "S3"]
        data3 = MyData.pull(["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"])
        data4 = MyData.pull(["S4", "S5"], shape=(5, 2), columns=["F1", "F2"])
        added_data = data3.add("S4", data4.get(symbol="S4"))
        added_data = added_data.add("S5", data4.get(symbol="S5"))
        assert added_data.features == ["F1", "F2", "F3"]
        assert added_data.symbols == ["S1", "S2", "S3", "S4", "S5"]

    def test_remove(self):
        data = MyData.pull(["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"])
        removed_data = data.remove("S1")
        assert removed_data.features == ["F1", "F2", "F3"]
        assert removed_data.symbols == ["S2", "S3"]
        removed_data = data.remove(["S1", "S2"])
        assert removed_data.features == ["F1", "F2", "F3"]
        assert removed_data.symbols == ["S3"]
        with pytest.raises(Exception):
            data.remove(["S1", "S2", "S3"])
        removed_data = data.remove("F1")
        assert removed_data.features == ["F2", "F3"]
        assert removed_data.symbols == ["S1", "S2", "S3"]
        removed_data = data.remove(["F1", "F2"])
        assert removed_data.features == ["F3"]
        assert removed_data.symbols == ["S1", "S2", "S3"]
        with pytest.raises(Exception):
            data.remove(["F1", "F2", "F3"])

    def test_select(self):
        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_symbol_oriented()
        assert (
            data.select("S1")
            == MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"]).to_symbol_oriented()
        )
        assert (
            data.select(["S1"])
            == MyData.pull(["S1"], shape=(5, 3), columns=["F1", "F2", "F3"]).to_symbol_oriented()
        )
        assert (
            data.select(["S1"])
            != MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"]).to_symbol_oriented()
        )
        assert (
            data.select(["S1", "S3"])
            == MyData.pull(
                ["S1", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"]
            ).to_symbol_oriented()
        )
        assert (
            data.select(["S1", "S3"])
            != MyData.pull(
                ["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]
            ).to_symbol_oriented()
        )
        with pytest.raises(Exception):
            data.select("S4")

        assert data.select("S1").to_feature_oriented() == data.to_feature_oriented()["S1"]
        assert data.select(["S1"]).to_feature_oriented() == data.to_feature_oriented()[["S1"]]
        assert data.select(["S1"]).to_feature_oriented() != data.to_feature_oriented()["S1"]
        assert (
            data.select(["S1", "S3"]).to_feature_oriented()
            == data.to_feature_oriented()[["S1", "S3"]]
        )
        assert (
            data.select(["S1", "S3"]).to_feature_oriented()
            != data.to_feature_oriented()[["S1", "S2"]]
        )

        assert data.to_feature_oriented().select("F1") == data["F1"].to_feature_oriented()
        assert data.to_feature_oriented().select(["F1"]) == data[["F1"]].to_feature_oriented()
        assert data.to_feature_oriented().select(["F1"]) != data["F1"].to_feature_oriented()
        assert (
            data.to_feature_oriented().select(["F1", "F3"])
            == data[["F1", "F3"]].to_feature_oriented()
        )
        assert (
            data.to_feature_oriented().select(["F1", "F3"])
            != data[["F1", "F2"]].to_feature_oriented()
        )
        with pytest.raises(Exception):
            data.to_feature_oriented().select(["F4"])

    def test_rename(self):
        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_symbol_oriented()
        renamed_data = data.rename({"S1": "S3", "S3": "S1"})
        assert renamed_data.features == ["F1", "F2", "F3"]
        assert renamed_data.symbols == ["S3", "S2", "S1"]
        assert list(renamed_data.fetch_kwargs.keys()) == ["S3", "S2", "S1"]
        assert list(renamed_data.returned_kwargs.keys()) == ["S3", "S2", "S1"]
        assert list(renamed_data.last_index.keys()) == ["S3", "S2", "S1"]
        assert list(renamed_data.delisted.keys()) == ["S3", "S2", "S1"]

        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_feature_oriented()
        renamed_data = data.rename({"F1": "F3", "F3": "F1"})
        assert renamed_data.features == ["F3", "F2", "F1"]
        assert renamed_data.symbols == ["S1", "S2", "S3"]
        assert list(renamed_data.fetch_kwargs.keys()) == ["S1", "S2", "S3"]
        assert list(renamed_data.returned_kwargs.keys()) == ["S1", "S2", "S3"]
        assert list(renamed_data.last_index.keys()) == ["S1", "S2", "S3"]
        assert list(renamed_data.delisted.keys()) == ["S1", "S2", "S3"]

    def test_merge(self):
        def _fix(data):
            return data.replace(missing_index="nan", missing_columns="nan")

        data = MyData.pull(["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"])
        data01 = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        data2 = MyData.pull(["S3"], shape=(5, 3), columns=["F1", "F2", "F3"])
        assert MyData.merge(
            data01.to_symbol_oriented(),
            data2.to_symbol_oriented(),
        ) == _fix(data.to_symbol_oriented())
        assert MyData.merge(
            data01.to_feature_oriented(),
            data2.to_feature_oriented(),
        ) == _fix(data.to_feature_oriented())
        data12 = MyData.pull(["S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"])
        assert MyData.merge(
            data01.to_symbol_oriented(),
            data12.to_symbol_oriented(),
        ) == _fix(data.to_symbol_oriented())
        assert MyData.merge(
            data01.to_feature_oriented(),
            data12.to_feature_oriented(),
        ) == _fix(data.to_feature_oriented())
        data12 = MyData.pull(
            ["S2", "S3"], shape=(3, 2), start_date=datetime(2020, 1, 3), columns=["F3", "F4"]
        )
        merged_data1 = MyData.merge(data01.to_symbol_oriented(), data12.to_symbol_oriented())
        merged_data2 = MyData.merge(data01.to_feature_oriented(), data12.to_feature_oriented())
        assert_frame_equal(
            merged_data1.data["S1"],
            pd.DataFrame(
                [
                    ["S1_F1_0", "S1_F2_0", "S1_F3_0", np.nan],
                    ["S1_F1_1", "S1_F2_1", "S1_F3_1", np.nan],
                    ["S1_F1_2", "S1_F2_2", "S1_F3_2", np.nan],
                    ["S1_F1_3", "S1_F2_3", "S1_F3_3", np.nan],
                    ["S1_F1_4", "S1_F2_4", "S1_F3_4", np.nan],
                ],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01 00:00:00+00:00",
                        "2020-01-02 00:00:00+00:00",
                        "2020-01-03 00:00:00+00:00",
                        "2020-01-04 00:00:00+00:00",
                        "2020-01-05 00:00:00+00:00",
                    ],
                    freq="d",
                ),
                columns=pd.Index(["F1", "F2", "F3", "F4"], dtype="object"),
            ),
        )
        assert_frame_equal(
            merged_data1.data["S2"],
            pd.DataFrame(
                [
                    ["S2_F1_0", "S2_F2_0", "S2_F3_0", np.nan],
                    ["S2_F1_1", "S2_F2_1", "S2_F3_1", np.nan],
                    ["S2_F1_2", "S2_F2_2", "S2_F3_0", "S2_F4_0"],
                    ["S2_F1_3", "S2_F2_3", "S2_F3_1", "S2_F4_1"],
                    ["S2_F1_4", "S2_F2_4", "S2_F3_2", "S2_F4_2"],
                ],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01 00:00:00+00:00",
                        "2020-01-02 00:00:00+00:00",
                        "2020-01-03 00:00:00+00:00",
                        "2020-01-04 00:00:00+00:00",
                        "2020-01-05 00:00:00+00:00",
                    ],
                    freq="d",
                ),
                columns=pd.Index(["F1", "F2", "F3", "F4"], dtype="object"),
            ),
        )
        assert_frame_equal(
            merged_data1.data["S3"],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, "S3_F3_0", "S3_F4_0"],
                    [np.nan, np.nan, "S3_F3_1", "S3_F4_1"],
                    [np.nan, np.nan, "S3_F3_2", "S3_F4_2"],
                ],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01 00:00:00+00:00",
                        "2020-01-02 00:00:00+00:00",
                        "2020-01-03 00:00:00+00:00",
                        "2020-01-04 00:00:00+00:00",
                        "2020-01-05 00:00:00+00:00",
                    ],
                    freq="d",
                ),
                columns=pd.Index(["F1", "F2", "F3", "F4"], dtype="object"),
            ),
        )
        assert_frame_equal(
            merged_data1.data["S1"].astype(object),
            merged_data2.to_symbol_oriented().data["S1"].astype(object),
        )
        assert_frame_equal(
            merged_data1.data["S2"].astype(object),
            merged_data2.to_symbol_oriented().data["S2"].astype(object),
        )
        assert_frame_equal(
            merged_data1.data["S3"].astype(object),
            merged_data2.to_symbol_oriented().data["S3"].astype(object),
        )
        assert_frame_equal(
            merged_data1.to_feature_oriented().data["F1"].astype(object),
            merged_data2.data["F1"].astype(object),
        )
        assert_frame_equal(
            merged_data1.to_feature_oriented().data["F2"].astype(object),
            merged_data2.data["F2"].astype(object),
        )
        assert_frame_equal(
            merged_data1.to_feature_oriented().data["F3"].astype(object),
            merged_data2.data["F3"].astype(object),
        )
        assert_frame_equal(
            merged_data1.to_feature_oriented().data["F4"].astype(object),
            merged_data2.data["F4"].astype(object),
        )

    def test_indexing(self):
        assert (
            MyData.pull(["S1", "S2"], shape=(5,), columns="F1")
            .to_symbol_oriented()
            .iloc[:3]
            .wrapper
            == MyData.pull(["S1", "S2"], shape=(3,), columns="F1").to_symbol_oriented().wrapper
        )
        assert (
            MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_symbol_oriented()
            .iloc[:3]
            .wrapper
            == MyData.pull(["S1", "S2"], shape=(3, 3), columns=["F1", "F2", "F3"])
            .to_symbol_oriented()
            .wrapper
        )
        assert (
            MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_symbol_oriented()["F1"]
            .wrapper
            == MyData.pull(["S1", "S2"], shape=(5,), columns="F1").to_symbol_oriented().wrapper
        )
        assert (
            MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
            .to_symbol_oriented()[["F1"]]
            .wrapper
            == MyData.pull(["S1", "S2"], shape=(5, 1), columns=["F1"]).to_symbol_oriented().wrapper
        )
        assert (
            MyData.pull("S1", shape=(5, 2), columns=["F1", "F2"])
            .to_feature_oriented()
            .iloc[:3]
            .wrapper
            == MyData.pull("S1", shape=(3, 2), columns=["F1", "F2"]).to_feature_oriented().wrapper
        )
        assert (
            MyData.pull(["S1", "S2"], shape=(5, 2), columns=["F1", "F2"])
            .to_feature_oriented()
            .iloc[:3]
            .wrapper
            == MyData.pull(["S1", "S2"], shape=(3, 2), columns=["F1", "F2"])
            .to_feature_oriented()
            .wrapper
        )
        assert (
            MyData.pull(["S1", "S2"], shape=(5, 2), columns=["F1", "F2"])
            .to_feature_oriented()["S1"]
            .wrapper
            == MyData.pull("S1", shape=(5, 2), columns=["F1", "F2"]).to_feature_oriented().wrapper
        )
        assert (
            MyData.pull(["S1", "S2"], shape=(5, 2), columns=["F1", "F2"])
            .to_feature_oriented()[["S1"]]
            .wrapper
            == MyData.pull(["S1"], shape=(5, 2), columns=["F1", "F2"]).to_feature_oriented().wrapper
        )

    def test_transform(self):
        def transform(x):
            return x.iloc[::2]

        def assert_data_equal(new_data, data):
            for k in data.data:
                if isinstance(data.data[k], pd.Series):
                    assert_series_equal(new_data.data[k], data.data[k].iloc[::2])
                else:
                    assert_frame_equal(new_data.data[k], data.data[k].iloc[::2])

        original_data = MyData.pull("S1", shape=(5, 3), columns=["F1", "F2", "F3"])
        for data in [original_data.to_symbol_oriented(), original_data.to_feature_oriented()]:
            assert_data_equal(data.transform(transform), data)
            assert_data_equal(data.transform(transform, per_symbol=True), data)
            assert_data_equal(data.transform(transform, per_feature=True), data)
            assert_data_equal(data.transform(transform, per_symbol=True, per_feature=True), data)
            assert_data_equal(
                data.transform(transform, per_symbol=True, per_feature=True, pass_frame=True), data
            )
        original_data = MyData.pull("S1", shape=(5,))
        for data in [original_data.to_symbol_oriented(), original_data.to_feature_oriented()]:
            assert_data_equal(data.transform(transform), data)
            assert_data_equal(data.transform(transform, per_symbol=True), data)
            assert_data_equal(data.transform(transform, per_feature=True), data)
            assert_data_equal(data.transform(transform, per_symbol=True, per_feature=True), data)
            assert_data_equal(
                data.transform(transform, per_symbol=True, per_feature=True, pass_frame=True), data
            )
        original_data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        for data in [original_data.to_symbol_oriented(), original_data.to_feature_oriented()]:
            assert_data_equal(data.transform(transform), data)
            assert_data_equal(data.transform(transform, per_symbol=True), data)
            assert_data_equal(data.transform(transform, per_feature=True), data)
            assert_data_equal(data.transform(transform, per_symbol=True, per_feature=True), data)
            assert_data_equal(
                data.transform(transform, per_symbol=True, per_feature=True, pass_frame=True), data
            )
        original_data = MyData.pull(["S1", "S2"], shape=(5,))
        for data in [original_data.to_symbol_oriented(), original_data.to_feature_oriented()]:
            assert_data_equal(data.transform(transform), data)
            assert_data_equal(data.transform(transform, per_symbol=True), data)
            assert_data_equal(data.transform(transform, per_feature=True), data)
            assert_data_equal(data.transform(transform, per_symbol=True, per_feature=True), data)
            assert_data_equal(
                data.transform(transform, per_symbol=True, per_feature=True, pass_frame=True), data
            )

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        ohlcv_data = vbt.Data.from_data(
            vbt.symbol_dict(
                S1=pd.DataFrame(
                    {
                        "Open": [1, 2, 3, 4, 5],
                        "High": [2.5, 3.5, 4.5, 5.5, 6.5],
                        "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
                        "Close": [2, 3, 4, 5, 6],
                        "Volume": [1, 2, 3, 2, 1],
                        "Other": [3, 2, 1, 2, 3],
                    },
                    index=pd.date_range("2020-01-01", "2020-01-05"),
                )
            ),
            single_key=True,
        )
        ohlcv_data.feature_config["Other"] = dict(
            resample_func=lambda self, obj, resampler, **kwargs: obj.vbt.resample_apply(
                resampler, vbt.nb.mean_reduce_nb, **kwargs
            )
        )
        for data in [ohlcv_data.to_symbol_oriented(), ohlcv_data.to_feature_oriented()]:
            assert_frame_equal(
                data.to_symbol_oriented().resample(test_freq).get(),
                pd.concat(
                    (
                        data.get(["Open", "High", "Low", "Close", "Volume"])
                        .vbt.ohlcv.resample(test_freq)
                        .obj,
                        data.get(["Other"]).resample(test_freq).mean(),
                    ),
                    axis=1,
                ),
            )

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_realign(self, test_freq):
        ohlcv_data = vbt.Data.from_data(
            vbt.symbol_dict(
                S1=pd.DataFrame(
                    {
                        "Open": [1, 2, 3, 4, 5],
                        "High": [2.5, 3.5, 4.5, 5.5, 6.5],
                        "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
                        "Close": [2, 3, 4, 5, 6],
                        "Volume": [1, 2, 3, 2, 1],
                        "Other": [3, 2, 1, 2, 3],
                    },
                    index=pd.date_range("2020-01-01", "2020-01-05"),
                )
            ),
            single_key=True,
        )
        ohlcv_data.feature_config["Other"] = dict(
            realign_func=lambda self, obj, resampler, **kwargs: obj.vbt.realign_opening(
                resampler, **kwargs
            )
        )
        for data in [ohlcv_data.to_symbol_oriented(), ohlcv_data.to_feature_oriented()]:
            assert_frame_equal(
                data.realign(test_freq).get(),
                pd.concat(
                    (
                        data.get(["Open"]).vbt.realign_opening(test_freq),
                        data.get(["High"]).vbt.realign_closing(test_freq),
                        data.get(["Low"]).vbt.realign_closing(test_freq),
                        data.get(["Close"]).vbt.realign_closing(test_freq),
                        data.get(["Volume"]).vbt.realign_closing(test_freq),
                        data.get(["Other"]).vbt.realign_opening(test_freq),
                    ),
                    axis=1,
                ),
            )

    def test_run(self):
        original_data = MyData.pull(
            ["S1", "S2"],
            shape=(5, 6),
            columns=["open", "high", "low", "close", "volume", "some_column"],
            return_numeric=True,
        )
        for data in [original_data.to_symbol_oriented(), original_data.to_feature_oriented()]:
            assert_frame_equal(data.run("from_holding").open, data.open)
            assert_frame_equal(data.run("from_holding").high, data.high)
            assert_frame_equal(data.run("from_holding").low, data.low)
            assert_frame_equal(data.run("from_holding").close, data.close)
            assert_frame_equal(data.run("ma", 3).ma, vbt.MA.run(data.close, 3).ma)
            assert_frame_equal(data.run("ma", 3, unpack=True), vbt.MA.run(data.close, 3).ma)
            assert_frame_equal(data.run("ma", 3, unpack="dict")["ma"], vbt.MA.run(data.close, 3).ma)
            assert_frame_equal(
                data.run("bbands", 3, unpack=True)[0], vbt.BBANDS.run(data.close, 3).upper
            )
            assert_frame_equal(
                data.run("bbands", 3, unpack=True)[1], vbt.BBANDS.run(data.close, 3).middle
            )
            assert_frame_equal(
                data.run("bbands", 3, unpack=True)[2], vbt.BBANDS.run(data.close, 3).lower
            )
            assert_frame_equal(
                data.run("bbands", 3, unpack="dict")["upper"], vbt.BBANDS.run(data.close, 3).upper
            )
            assert_frame_equal(
                data.run("bbands", 3, unpack="dict")["middle"], vbt.BBANDS.run(data.close, 3).middle
            )
            assert_frame_equal(
                data.run("bbands", 3, unpack="dict")["lower"], vbt.BBANDS.run(data.close, 3).lower
            )
            assert_frame_equal(
                data.run("talib:sma", 3).real, vbt.talib("SMA").run(data.close, 3).real
            )
            assert_frame_equal(
                data.run("pandas_ta:sma", 3).sma, vbt.pandas_ta("SMA").run(data.close, 3).sma
            )
            assert_frame_equal(data.run("wqa101:1").out, vbt.wqa101(1).run(data.close).out)
            assert_frame_equal(
                data.run("talib_sma", 3).real, vbt.talib("SMA").run(data.close, 3).real
            )
            assert_frame_equal(
                data.run("pandas_ta_sma", 3).sma, vbt.pandas_ta("SMA").run(data.close, 3).sma
            )
            assert_frame_equal(data.run("wqa101_1").out, vbt.wqa101(1).run(data.close).out)
            assert_frame_equal(data.run("sma", 3).real, vbt.talib("SMA").run(data.close, 3).real)
            assert_frame_equal(
                data.run("talib_func_bbands", 3)[0], vbt.talib_func("BBANDS")(data.close, 3)[0]
            )
            assert_frame_equal(
                data.run("talib_func_bbands", 3)[1], vbt.talib_func("BBANDS")(data.close, 3)[1]
            )
            assert_frame_equal(
                data.run("talib_func_bbands", 3)[2], vbt.talib_func("BBANDS")(data.close, 3)[2]
            )
            assert_frame_equal(
                data.run("talib_func:bbands", 3)[0], vbt.talib_func("BBANDS")(data.close, 3)[0]
            )
            assert_frame_equal(
                data.run("talib_func:bbands", 3)[1], vbt.talib_func("BBANDS")(data.close, 3)[1]
            )
            assert_frame_equal(
                data.run("talib_func:bbands", 3)[2], vbt.talib_func("BBANDS")(data.close, 3)[2]
            )
            assert_frame_equal(
                data.run("bbands", 3, location="talib_func")[0],
                vbt.talib_func("BBANDS")(data.close, 3)[0],
            )
            assert_frame_equal(
                data.run("bbands", 3, location="talib_func")[1],
                vbt.talib_func("BBANDS")(data.close, 3)[1],
            )
            assert_frame_equal(
                data.run("bbands", 3, location="talib_func")[2],
                vbt.talib_func("BBANDS")(data.close, 3)[2],
            )
            assert_frame_equal(data.run(lambda open: open), data.open)
            assert_frame_equal(data.run(lambda x, open: open + x, 100), data.open + 100)
            assert_frame_equal(data.run(lambda open, x: open + x, 100), data.open + 100)
            assert_frame_equal(
                data.run(lambda open, x, y=2: open + x + y, 100, 200), data.open + 100 + 200
            )
            assert_frame_equal(
                data.run(lambda open, x, y=2: open + x + y, x=100, y=200), data.open + 100 + 200
            )
            assert_frame_equal(data.run(lambda x, data: data.open + x, 100), data.open + 100)
            assert_frame_equal(
                data.run(lambda x, y: x.open + y, 100, rename_args={"x": "data"}), data.open + 100
            )
            assert_frame_equal(
                data.run(["talib_sma", "talib_ema"], timeperiod=3, hide_params=True),
                pd.DataFrame(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [9.0, 9.0, 9.0, 9.0],
                        [15.0, 15.0, 15.0, 15.0],
                        [21.0, 21.0, 21.0, 21.0],
                    ],
                    index=data.index,
                    columns=pd.MultiIndex.from_tuples(
                        [
                            ("talib_sma", "real", "S1"),
                            ("talib_sma", "real", "S2"),
                            ("talib_ema", "real", "S1"),
                            ("talib_ema", "real", "S2"),
                        ],
                        names=["run_func", "output", "symbol"],
                    ),
                ),
            )
            assert_frame_equal(
                data.run(
                    ["talib_sma", "talib_ema"],
                    timeperiod=vbt.run_func_dict(talib_sma=3, talib_ema=4),
                    hide_params=True,
                ),
                pd.DataFrame(
                    [
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [9.0, 9.0, np.nan, np.nan],
                        [15.0, 15.0, 12.0, 12.0],
                        [21.0, 21.0, 18.0, 18.0],
                    ],
                    index=data.index,
                    columns=pd.MultiIndex.from_tuples(
                        [
                            ("talib_sma", "real", "S1"),
                            ("talib_sma", "real", "S2"),
                            ("talib_ema", "real", "S1"),
                            ("talib_ema", "real", "S2"),
                        ],
                        names=["run_func", "output", "symbol"],
                    ),
                ),
            )
            assert_frame_equal(
                data.run(
                    ["talib_sma", "talib_ema"],
                    timeperiod=vbt.run_func_dict(talib_sma=3, talib_ema=4),
                    hide_params=True,
                ),
                data.run(
                    ["talib_sma", "talib_ema"],
                    timeperiod=vbt.run_func_dict({0: 3, 1: 4}),
                    hide_params=True,
                ),
            )
            assert_frame_equal(
                data.run(
                    ["talib_sma", "talib_ema"],
                    timeperiod=vbt.run_func_dict(talib_sma=3, talib_ema=4),
                    hide_params=True,
                ),
                data.run(
                    ["talib_sma", "talib_ema"],
                    talib_sma=vbt.run_arg_dict({"timeperiod": 3}),
                    talib_ema=vbt.run_arg_dict({"timeperiod": 4}),
                    hide_params=True,
                ),
            )
            assert_frame_equal(
                data.run(
                    ["talib_sma", "talib_ema"],
                    timeperiod=vbt.run_func_dict(talib_sma=3, talib_ema=4),
                    hide_params=True,
                ),
                data.run(
                    ["sma", "ema"],
                    timeperiod=vbt.run_func_dict(talib_sma=3, talib_ema=4),
                    hide_params=True,
                    location="talib",
                    prepend_location=True,
                ),
            )
            assert_frame_equal(
                data.run(
                    ["talib_func_sma", "talib_func_ema"],
                    timeperiod=vbt.run_func_dict(talib_func_sma=3, talib_func_ema=4),
                ),
                data.run(
                    ["sma", "ema"],
                    timeperiod=vbt.run_func_dict(talib_func_sma=3, talib_func_ema=4),
                    location="talib_func",
                    prepend_location=True,
                ),
            )
            assert_frame_equal(
                data.run(
                    ["talib_func_sma", "talib_func_ema"],
                    timeperiod=vbt.run_func_dict(sma=3, ema=4),
                    prepend_location=False,
                ),
                data.run(
                    ["sma", "ema"],
                    timeperiod=vbt.run_func_dict(sma=3, ema=4),
                    location="talib_func",
                    prepend_location=False,
                ),
            )

    def test_symbol_to_csv(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_symbol_oriented()

        def _load_and_check_symbol(k, path, **kwargs):
            df = pd.read_csv(path, parse_dates=True, index_col=0, **kwargs).squeeze("columns")
            df.index.freq = df.index.inferred_freq
            assert_frame_equal(df, data.data[k])

        data.to_csv(tmp_path)
        _load_and_check_symbol("S1", tmp_path / "S1.csv")
        _load_and_check_symbol("S2", tmp_path / "S2.csv")

        data.to_csv(
            vbt.symbol_dict({"S1": tmp_path / "csv_data", "S2": tmp_path / "csv_data"}),
            ext=vbt.symbol_dict({"S1": "csv", "S2": "tsv"}),
            sep=vbt.symbol_dict({"S1": ",", "S2": "\t"}),
            mkdir_kwargs=dict(mkdir=True),
        )
        with pytest.raises(Exception):
            data.to_csv(
                vbt.feature_dict({"S1": tmp_path / "csv_data", "S2": tmp_path / "csv_data"}),
                ext=vbt.feature_dict({"S1": "csv", "S2": "tsv"}),
                sep=vbt.feature_dict({"S1": ",", "S2": "\t"}),
                mkdir_kwargs=dict(mkdir=True),
            )
        _load_and_check_symbol("S1", tmp_path / "csv_data/S1.csv", sep=",")
        _load_and_check_symbol("S2", tmp_path / "csv_data/S2.tsv", sep="\t")

        data.to_csv(
            path_or_buf=vbt.symbol_dict(
                {"S1": tmp_path / "my_S1.csv", "S2": tmp_path / "my_S2.csv"}
            )
        )
        _load_and_check_symbol("S1", tmp_path / "my_S1.csv")
        _load_and_check_symbol("S2", tmp_path / "my_S2.csv")

        data.to_csv(
            vbt.symbol_dict({"S1": tmp_path / "csv_data", "S2": tmp_path / "csv_data"}),
            ext=vbt.symbol_dict({"S1": "csv", "S2": "tsv"}),
            sep=vbt.symbol_dict({"S1": ",", "S2": "\t"}),
        )
        _load_and_check_symbol("S1", tmp_path / "csv_data/S1.csv", sep=",")
        _load_and_check_symbol("S2", tmp_path / "csv_data/S2.tsv", sep="\t")

    def test_feature_to_csv(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 2), columns=["F1", "F2"]
        ).to_feature_oriented()

        def _load_and_check_feature(k, path, **kwargs):
            df = pd.read_csv(path, parse_dates=True, index_col=0, **kwargs).squeeze("columns")
            df.index.freq = df.index.inferred_freq
            df.columns.name = "symbol"
            assert_frame_equal(df, data.data[k])

        data.to_csv(tmp_path)
        _load_and_check_feature("F1", tmp_path / "F1.csv")
        _load_and_check_feature("F2", tmp_path / "F2.csv")

        data.to_csv(
            vbt.feature_dict({"F1": tmp_path / "csv_data", "F2": tmp_path / "csv_data"}),
            ext=vbt.feature_dict({"F1": "csv", "F2": "tsv"}),
            sep=vbt.feature_dict({"F1": ",", "F2": "\t"}),
            mkdir_kwargs=dict(mkdir=True),
        )
        with pytest.raises(Exception):
            data.to_csv(
                vbt.symbol_dict({"F1": tmp_path / "csv_data", "F2": tmp_path / "csv_data"}),
                ext=vbt.symbol_dict({"F1": "csv", "F2": "tsv"}),
                sep=vbt.symbol_dict({"F1": ",", "F2": "\t"}),
                mkdir_kwargs=dict(mkdir=True),
            )
        _load_and_check_feature("F1", tmp_path / "csv_data/F1.csv", sep=",")
        _load_and_check_feature("F2", tmp_path / "csv_data/F2.tsv", sep="\t")

        data.to_csv(
            path_or_buf=vbt.feature_dict(
                {"F1": tmp_path / "my_F1.csv", "F2": tmp_path / "my_F2.csv"}
            )
        )
        _load_and_check_feature("F1", tmp_path / "my_F1.csv")
        _load_and_check_feature("F2", tmp_path / "my_F2.csv")

        data.to_csv(
            vbt.feature_dict({"F1": tmp_path / "csv_data", "F2": tmp_path / "csv_data"}),
            ext=vbt.feature_dict({"F1": "csv", "F2": "tsv"}),
            sep=vbt.feature_dict({"F1": ",", "F2": "\t"}),
        )
        _load_and_check_feature("F1", tmp_path / "csv_data/F1.csv", sep=",")
        _load_and_check_feature("F2", tmp_path / "csv_data/F2.tsv", sep="\t")

    def test_symbol_to_hdf(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_symbol_oriented()

        def _load_and_check_symbol(k, path, key=None, **kwargs):
            if key is None:
                key = k
            df = pd.read_hdf(path, key, **kwargs)
            df.index.freq = df.index.inferred_freq
            assert_frame_equal(df, data.data[k])

        data.to_hdf(tmp_path)
        _load_and_check_symbol("S1", tmp_path / "MyData.h5")
        _load_and_check_symbol("S2", tmp_path / "MyData.h5")

        data.to_hdf(
            vbt.symbol_dict({"S1": tmp_path / "hdf_data/S1.h5", "S2": tmp_path / "hdf_data/S2.h5"}),
            mkdir_kwargs=dict(mkdir=True),
        )
        with pytest.raises(Exception):
            data.to_hdf(
                vbt.feature_dict(
                    {"S1": tmp_path / "hdf_data/S1.h5", "S2": tmp_path / "hdf_data/S2.h5"}
                ),
                mkdir_kwargs=dict(mkdir=True),
            )
        _load_and_check_symbol("S1", tmp_path / "hdf_data/S1.h5")
        _load_and_check_symbol("S2", tmp_path / "hdf_data/S2.h5")

        data.to_hdf(
            vbt.symbol_dict(
                {"S1": tmp_path / "hdf_data/my_data.h5", "S2": tmp_path / "hdf_data/my_data.h5"}
            ),
            key=vbt.symbol_dict({"S1": "df1", "S2": "df2"}),
        )
        _load_and_check_symbol("S1", tmp_path / "hdf_data/my_data.h5", key="df1")
        _load_and_check_symbol("S2", tmp_path / "hdf_data/my_data.h5", key="df2")

        data.to_hdf(
            path_or_buf=vbt.symbol_dict(
                {"S1": tmp_path / "hdf_data/my_data.h5", "S2": tmp_path / "hdf_data/my_data.h5"},
            ),
            key=vbt.symbol_dict({"S1": "df1", "S2": "df2"}),
        )
        _load_and_check_symbol("S1", tmp_path / "hdf_data/my_data.h5", key="df1")
        _load_and_check_symbol("S2", tmp_path / "hdf_data/my_data.h5", key="df2")

    def test_feature_to_hdf(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 2), columns=["F1", "F2"]
        ).to_feature_oriented()

        def _load_and_check_feature(k, path, key=None, **kwargs):
            if key is None:
                key = k
            df = pd.read_hdf(path, key, **kwargs)
            df.index.freq = df.index.inferred_freq
            assert_frame_equal(df, data.data[k])

        data.to_hdf(tmp_path)
        _load_and_check_feature("F1", tmp_path / "MyData.h5")
        _load_and_check_feature("F2", tmp_path / "MyData.h5")

        data.to_hdf(
            vbt.feature_dict(
                {"F1": tmp_path / "hdf_data/F1.h5", "F2": tmp_path / "hdf_data/F2.h5"}
            ),
            mkdir_kwargs=dict(mkdir=True),
        )
        with pytest.raises(Exception):
            data.to_hdf(
                vbt.symbol_dict(
                    {"F1": tmp_path / "hdf_data/F1.h5", "F2": tmp_path / "hdf_data/F2.h5"}
                ),
                mkdir_kwargs=dict(mkdir=True),
            )
        _load_and_check_feature("F1", tmp_path / "hdf_data/F1.h5")
        _load_and_check_feature("F2", tmp_path / "hdf_data/F2.h5")

        data.to_hdf(
            vbt.feature_dict(
                {"F1": tmp_path / "hdf_data/my_data.h5", "F2": tmp_path / "hdf_data/my_data.h5"}
            ),
            key=vbt.feature_dict({"F1": "df1", "F2": "df2"}),
        )
        _load_and_check_feature("F1", tmp_path / "hdf_data/my_data.h5", key="df1")
        _load_and_check_feature("F2", tmp_path / "hdf_data/my_data.h5", key="df2")

        data.to_hdf(
            path_or_buf=vbt.feature_dict(
                {"F1": tmp_path / "hdf_data/my_data.h5", "F2": tmp_path / "hdf_data/my_data.h5"},
            ),
            key=vbt.feature_dict({"F1": "df1", "F2": "df2"}),
        )
        _load_and_check_feature("F1", tmp_path / "hdf_data/my_data.h5", key="df1")
        _load_and_check_feature("F2", tmp_path / "hdf_data/my_data.h5", key="df2")

    def test_symbol_to_feather(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_symbol_oriented()

        def _load_and_check_symbol(k, path, **kwargs):
            df = pd.read_feather(path, **kwargs)
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index("index")
                df.index.name = None
            df.index.freq = df.index.inferred_freq
            assert_frame_equal(df, data.data[k])

        data.to_feather(tmp_path)
        _load_and_check_symbol("S1", tmp_path / "S1.feather")
        _load_and_check_symbol("S2", tmp_path / "S2.feather")

        data.to_feather(tmp_path / "feather_data", mkdir_kwargs=dict(mkdir=True))
        _load_and_check_symbol("S1", tmp_path / "feather_data/S1.feather")
        _load_and_check_symbol("S2", tmp_path / "feather_data/S2.feather")

        data.to_feather(
            path_or_buf=vbt.symbol_dict(
                {"S1": tmp_path / "my_S1.feather", "S2": tmp_path / "my_S2.feather"}
            )
        )
        _load_and_check_symbol("S1", tmp_path / "my_S1.feather")
        _load_and_check_symbol("S2", tmp_path / "my_S2.feather")

    def test_feature_to_feather(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 2), columns=["F1", "F2"]
        ).to_feature_oriented()

        def _load_and_check_symbol(k, path, **kwargs):
            df = pd.read_feather(path, **kwargs)
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index("index")
                df.index.name = None
            df.index.freq = df.index.inferred_freq
            assert_frame_equal(df, data.data[k])

        data.to_feather(tmp_path)
        _load_and_check_symbol("F1", tmp_path / "F1.feather")
        _load_and_check_symbol("F2", tmp_path / "F2.feather")

        data.to_feather(tmp_path / "feather_data", mkdir_kwargs=dict(mkdir=True))
        _load_and_check_symbol("F1", tmp_path / "feather_data/F1.feather")
        _load_and_check_symbol("F2", tmp_path / "feather_data/F2.feather")

        data.to_feather(
            path_or_buf=vbt.feature_dict(
                {"F1": tmp_path / "my_F1.feather", "F2": tmp_path / "my_F2.feather"}
            )
        )
        _load_and_check_symbol("F1", tmp_path / "my_F1.feather")
        _load_and_check_symbol("F2", tmp_path / "my_F2.feather")

    def test_symbol_to_parquet(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_symbol_oriented()

        def _load_and_check_symbol(k, path, **kwargs):
            df = pd.read_parquet(path, **kwargs)
            df.index.freq = df.index.inferred_freq
            df = df[data.data[k].columns].astype(object)
            assert_frame_equal(df, data.data[k])

        data.to_parquet(tmp_path)
        _load_and_check_symbol("S1", tmp_path / "S1.parquet")
        _load_and_check_symbol("S2", tmp_path / "S2.parquet")

        data.to_parquet(tmp_path / "parquet_data", mkdir_kwargs=dict(mkdir=True))
        _load_and_check_symbol("S1", tmp_path / "parquet_data/S1.parquet")
        _load_and_check_symbol("S2", tmp_path / "parquet_data/S2.parquet")

        data.to_parquet(
            tmp_path / "parquet_data",
            partition_cols=["F1"],
            mkdir_kwargs=dict(mkdir=True),
        )
        _load_and_check_symbol("S1", tmp_path / "parquet_data/S1")
        _load_and_check_symbol("S2", tmp_path / "parquet_data/S2")

        data.to_parquet(
            tmp_path / "parquet_data2",
            partition_by=[0, 0, 0, 1, 1],
            mkdir_kwargs=dict(mkdir=True),
        )
        _load_and_check_symbol("S1", tmp_path / "parquet_data2/S1")
        _load_and_check_symbol("S2", tmp_path / "parquet_data2/S2")

        data.to_parquet(
            path_or_buf=vbt.symbol_dict(
                {"S1": tmp_path / "my_S1.parquet", "S2": tmp_path / "my_S2.parquet"}
            )
        )
        _load_and_check_symbol("S1", tmp_path / "my_S1.parquet")
        _load_and_check_symbol("S2", tmp_path / "my_S2.parquet")

    def test_feature_to_parquet(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 2), columns=["F1", "F2"]
        ).to_feature_oriented()

        def _load_and_check_feature(k, path, **kwargs):
            df = pd.read_parquet(path, **kwargs)
            df.index.freq = df.index.inferred_freq
            df = df[data.data[k].columns].astype(object)
            assert_frame_equal(df, data.data[k])

        data.to_parquet(tmp_path)
        _load_and_check_feature("F1", tmp_path / "F1.parquet")
        _load_and_check_feature("F2", tmp_path / "F2.parquet")

        data.to_parquet(tmp_path / "parquet_data", mkdir_kwargs=dict(mkdir=True))
        _load_and_check_feature("F1", tmp_path / "parquet_data/F1.parquet")
        _load_and_check_feature("F2", tmp_path / "parquet_data/F2.parquet")

        data.to_parquet(
            tmp_path / "parquet_data",
            partition_cols=["S1"],
            mkdir_kwargs=dict(mkdir=True),
        )
        _load_and_check_feature("F1", tmp_path / "parquet_data/F1")
        _load_and_check_feature("F2", tmp_path / "parquet_data/F2")

        data.to_parquet(
            tmp_path / "parquet_data2",
            partition_by=[0, 0, 0, 1, 1],
            mkdir_kwargs=dict(mkdir=True),
        )
        _load_and_check_feature("F1", tmp_path / "parquet_data2/F1")
        _load_and_check_feature("F2", tmp_path / "parquet_data2/F2")

        data.to_parquet(
            path_or_buf=vbt.feature_dict(
                {"F1": tmp_path / "my_F1.parquet", "F2": tmp_path / "my_F2.parquet"}
            )
        )
        _load_and_check_feature("F1", tmp_path / "my_F1.parquet")
        _load_and_check_feature("F2", tmp_path / "my_F2.parquet")

    def test_symbol_to_sql(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_symbol_oriented()

        def _load_and_check_symbol(k, name, engine_url, **kwargs):
            from sqlalchemy import create_engine

            df = pd.read_sql_table(
                name,
                create_engine(engine_url).connect(),
                index_col="index",
                **kwargs,
            ).squeeze("columns")
            df.index.freq = df.index.inferred_freq
            df.index.name = None
            df = df.tz_localize("utc")
            assert_frame_equal(df, data.data[k])

        engine_url = "sqlite:///" + str(tmp_path / "temp.db")
        data.to_sql(engine_url)
        _load_and_check_symbol("S1", "S1", engine_url)
        _load_and_check_symbol("S2", "S2", engine_url)

        engine_url1 = "sqlite:///" + str(tmp_path / "temp1.db")
        engine_url2 = "sqlite:///" + str(tmp_path / "temp2.db")
        data.to_sql(
            vbt.symbol_dict({"S1": engine_url1, "S2": engine_url2}),
            name=vbt.symbol_dict({"S1": "T1", "S2": "T2"}),
        )
        _load_and_check_symbol("S1", "T1", engine_url1)
        _load_and_check_symbol("S2", "T2", engine_url2)

    def test_feature_to_sql(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 2), columns=["F1", "F2"]
        ).to_feature_oriented()

        def _load_and_check_feature(k, name, engine_url, **kwargs):
            from sqlalchemy import create_engine

            df = pd.read_sql_table(
                name,
                create_engine(engine_url).connect(),
                index_col="index",
                **kwargs,
            ).squeeze("columns")
            df.index.freq = df.index.inferred_freq
            df.index.name = None
            df = df.tz_localize("utc")
            df.columns.name = "symbol"
            assert_frame_equal(df, data.data[k])

        engine_url = "sqlite:///" + str(tmp_path / "temp.db")
        data.to_sql(engine_url)
        _load_and_check_feature("F1", "F1", engine_url)
        _load_and_check_feature("F2", "F2", engine_url)

        engine_url1 = "sqlite:///" + str(tmp_path / "temp1.db")
        engine_url2 = "sqlite:///" + str(tmp_path / "temp2.db")
        data.to_sql(
            vbt.feature_dict({"F1": engine_url1, "F2": engine_url2}),
            name=vbt.feature_dict({"F1": "T1", "F2": "T2"}),
        )
        _load_and_check_feature("F1", "T1", engine_url1)
        _load_and_check_feature("F2", "T2", engine_url2)

    def test_symbol_to_duckdb(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"]
        ).to_symbol_oriented()

        def _load_and_check_symbol(k, table, database, schema=None):
            import duckdb

            connection = duckdb.connect(str(database))
            if schema is None:
                df = connection.sql("SELECT * FROM " + table).df()
            else:
                df = connection.sql("SELECT * FROM " + schema + "." + table).df()
            df = df.set_index("index", drop=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("utc")
            df.index = df.index.astype("datetime64[ns, UTC]")
            df.index.name = None
            df.index.freq = df.index.inferred_freq
            assert_frame_equal(df, data.data[k])

        connection = tmp_path / "database.duckdb"

        data.to_duckdb(connection=connection)
        _load_and_check_symbol("S1", "S1", connection)
        _load_and_check_symbol("S2", "S2", connection)

        data.to_duckdb(connection=connection, schema="hello")
        _load_and_check_symbol("S1", "S1", connection, schema="hello")
        _load_and_check_symbol("S2", "S2", connection, schema="hello")

        data.to_duckdb(
            connection=connection,
            table=vbt.symbol_dict({"S1": "T1", "S2": "T2"}),
            schema=vbt.symbol_dict({"S1": "schema1", "S2": "schema2"}),
        )
        _load_and_check_symbol("S1", "T1", connection, schema="schema1")
        _load_and_check_symbol("S2", "T2", connection, schema="schema2")

        data.to_duckdb(connection=connection, write_path=tmp_path, write_format="csv")
        _load_and_check_symbol("S1", "'{}'".format(str(tmp_path / "S1.csv")), connection)
        _load_and_check_symbol("S2", "'{}'".format(str(tmp_path / "S2.csv")), connection)

        data.to_duckdb(
            connection=connection,
            write_path=vbt.symbol_dict({"S1": tmp_path / "S1.csv", "S2": tmp_path / "S2.tsv"}),
            write_options=vbt.symbol_dict({"S1": "SEP ','", "S2": "SEP '\t'"}),
        )
        _load_and_check_symbol("S1", "'{}'".format(str(tmp_path / "S1.csv")), connection)
        _load_and_check_symbol("S2", "'{}'".format(str(tmp_path / "S2.tsv")), connection)

        data.to_duckdb(
            connection=connection,
            write_path=vbt.symbol_dict({"S1": tmp_path / "S1.csv", "S2": tmp_path / "S2.tsv"}),
            write_options=dict(sep=vbt.symbol_dict({"S1": ",", "S2": "\t"})),
        )
        _load_and_check_symbol("S1", "'{}'".format(str(tmp_path / "S1.csv")), connection)
        _load_and_check_symbol("S2", "'{}'".format(str(tmp_path / "S2.tsv")), connection)

    def test_feature_to_duckdb(self, tmp_path):
        data = MyData.pull(
            ["S1", "S2", "S3"], shape=(5, 2), columns=["F1", "F2"]
        ).to_feature_oriented()

        def _load_and_check_feature(k, table, database, schema=None):
            import duckdb

            connection = duckdb.connect(str(database))
            if schema is None:
                df = connection.sql("SELECT * FROM " + table).df()
            else:
                df = connection.sql("SELECT * FROM " + schema + "." + table).df()
            df = df.set_index("index", drop=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("utc")
            df.index = df.index.astype("datetime64[ns, UTC]")
            df.index.name = None
            df.index.freq = df.index.inferred_freq
            df.columns.name = "symbol"
            assert_frame_equal(df, data.data[k])

        connection = tmp_path / "database.duckdb"

        data.to_duckdb(connection=connection)
        _load_and_check_feature("F1", "F1", connection)
        _load_and_check_feature("F2", "F2", connection)

        data.to_duckdb(connection=connection, schema="hello")
        _load_and_check_feature("F1", "F1", connection, schema="hello")
        _load_and_check_feature("F2", "F2", connection, schema="hello")

        data.to_duckdb(
            connection=connection,
            table=vbt.feature_dict({"F1": "T1", "F2": "T2"}),
            schema=vbt.feature_dict({"F1": "schema1", "F2": "schema2"}),
        )
        _load_and_check_feature("F1", "T1", connection, schema="schema1")
        _load_and_check_feature("F2", "T2", connection, schema="schema2")

        data.to_duckdb(connection=connection, write_path=tmp_path, write_format="csv")
        _load_and_check_feature("F1", "'{}'".format(str(tmp_path / "F1.csv")), connection)
        _load_and_check_feature("F2", "'{}'".format(str(tmp_path / "F2.csv")), connection)

        data.to_duckdb(
            connection=connection,
            write_path=vbt.feature_dict({"F1": tmp_path / "F1.csv", "F2": tmp_path / "F2.tsv"}),
            write_options=vbt.feature_dict({"F1": "SEP ','", "F2": "SEP '\t'"}),
        )
        _load_and_check_feature("F1", "'{}'".format(str(tmp_path / "F1.csv")), connection)
        _load_and_check_feature("F2", "'{}'".format(str(tmp_path / "F2.tsv")), connection)

        data.to_duckdb(
            connection=connection,
            write_path=vbt.feature_dict({"F1": tmp_path / "F1.csv", "F2": tmp_path / "F2.tsv"}),
            write_options=dict(sep=vbt.feature_dict({"F1": ",", "F2": "\t"})),
        )
        _load_and_check_feature("F1", "'{}'".format(str(tmp_path / "F1.csv")), connection)
        _load_and_check_feature("F2", "'{}'".format(str(tmp_path / "F2.tsv")), connection)

    def test_sql(self):
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        target_df = data.concat()["F1"]
        target_df.index.freq = None
        target_df.columns.name = None
        assert_frame_equal(data.sql("SELECT * FROM F1"), target_df)

    def test_symbol_stats(self):
        index_mask = vbt.symbol_dict(
            {"S1": [False, True, True, True, True], "S2": [True, True, True, True, False]}
        )
        column_mask = vbt.symbol_dict({"S1": [False, True, True], "S2": [True, True, False]})
        data = MyData.pull(
            ["S1", "S2"],
            shape=(5, 3),
            index_mask=index_mask,
            column_mask=column_mask,
            missing_index="nan",
            missing_columns="nan",
            columns=["F1", "F2", "F3"],
        ).to_symbol_oriented()

        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Symbols",
                "Null Counts: S1",
                "Null Counts: S2",
            ],
            dtype="object",
        )
        assert_series_equal(
            data.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="utc"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="utc"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    7,
                    7,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            data.stats(column="F1"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="utc"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="utc"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    5,
                    1,
                ],
                index=stats_index,
                name="F1",
            ),
        )
        assert_series_equal(
            data.stats(group_by=True),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="utc"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="utc"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    7,
                    7,
                ],
                index=stats_index,
                name="group",
            ),
        )
        assert_series_equal(data["F1"].stats(), data.stats(column="F1"))
        assert_series_equal(
            data.replace(wrapper=data.wrapper.replace(group_by=True)).stats(),
            data.stats(group_by=True),
        )
        stats_df = data.stats(agg_func=None)
        assert stats_df.shape == (3, 6)
        assert_index_equal(stats_df.index, data.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    def test_feature_stats(self):
        index_mask = vbt.symbol_dict(
            {"S1": [False, True, True, True, True], "S2": [True, True, True, True, False]}
        )
        column_mask = vbt.symbol_dict({"S1": [False, True, True], "S2": [True, True, False]})
        data = MyData.pull(
            ["S1", "S2"],
            shape=(5, 3),
            index_mask=index_mask,
            column_mask=column_mask,
            missing_index="nan",
            missing_columns="nan",
            columns=["F1", "F2", "F3"],
        ).to_feature_oriented()

        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Features",
                "Null Counts: F1",
                "Null Counts: F2",
                "Null Counts: F3",
            ],
            dtype="object",
        )
        assert_series_equal(
            data.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="utc"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="utc"),
                    pd.Timedelta("5 days 00:00:00"),
                    3,
                    6,
                    2,
                    6,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            data.stats(column=0),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="utc"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="utc"),
                    pd.Timedelta("5 days 00:00:00"),
                    3,
                    5,
                    1,
                    1,
                ],
                index=stats_index,
                name=0,
            ),
        )
        assert_series_equal(
            data.stats(group_by=True),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="utc"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="utc"),
                    pd.Timedelta("5 days 00:00:00"),
                    3,
                    6,
                    2,
                    6,
                ],
                index=stats_index,
                name="group",
            ),
        )
        assert_series_equal(data["S1"].stats(), data.stats(column="S1"))
        assert_series_equal(
            data.replace(wrapper=data.wrapper.replace(group_by=True)).stats(),
            data.stats(group_by=True),
        )
        stats_df = data.stats(agg_func=None)
        assert stats_df.shape == (2, 7)
        assert_index_equal(stats_df.index, data.wrapper.columns)
        assert_index_equal(stats_df.columns, stats_index)

    @pytest.mark.parametrize("test_to_feature_oriented", [False, True])
    def test_items(self, test_to_feature_oriented):
        data = MyData.pull(["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"])
        if test_to_feature_oriented:
            data = data.to_feature_oriented(
                classes=vbt.feature_dict(
                    {"F1": dict(k="K1"), "F2": dict(k="K1"), "F3": dict(k="K2")}
                )
            )
        else:
            data = data.to_symbol_oriented(
                classes=vbt.symbol_dict(
                    {"S1": dict(k="K1"), "S2": dict(k="K1"), "S3": dict(k="K2")}
                )
            )
        items = list(data.items())
        assert len(items) == 3
        assert items[0][0] == "S1"
        assert items[0][1] == data.select_symbol_idxs(0)
        assert items[1][0] == "S2"
        assert items[1][1] == data.select_symbol_idxs(1)
        assert items[2][0] == "S3"
        assert items[2][1] == data.select_symbol_idxs(2)
        items = list(data.items(keep_2d=True))
        assert len(items) == 3
        assert items[0][0] == "S1"
        assert items[0][1] == data.select_symbol_idxs([0])
        assert items[1][0] == "S2"
        assert items[1][1] == data.select_symbol_idxs([1])
        assert items[2][0] == "S3"
        assert items[2][1] == data.select_symbol_idxs([2])
        items = list(data.items(group_by=["G1", "G1", "G2"]))
        assert len(items) == 2
        assert items[0][0] == "G1"
        assert items[0][1] == data.select_symbol_idxs([0, 1])
        assert items[1][0] == "G2"
        assert items[1][1] == data.select_symbol_idxs(2)
        items = list(data.items(group_by=["G1", "G1", "G2"], keep_2d=True))
        assert len(items) == 2
        assert items[0][0] == "G1"
        assert items[0][1] == data.select_symbol_idxs([0, 1])
        assert items[1][0] == "G2"
        assert items[1][1] == data.select_symbol_idxs([2])
        items = list(data.items(group_by=True))
        assert len(items) == 1
        assert items[0][0] == "group"
        assert items[0][1] == data
        items = list(data.items(over="features"))
        assert len(items) == 3
        assert items[0][0] == "F1"
        assert items[0][1] == data.select_feature_idxs(0)
        assert items[1][0] == "F2"
        assert items[1][1] == data.select_feature_idxs(1)
        assert items[2][0] == "F3"
        assert items[2][1] == data.select_feature_idxs(2)
        items = list(data.items(over="features", keep_2d=True))
        assert len(items) == 3
        assert items[0][0] == "F1"
        assert items[0][1] == data.select_feature_idxs([0])
        assert items[1][0] == "F2"
        assert items[1][1] == data.select_feature_idxs([1])
        assert items[2][0] == "F3"
        assert items[2][1] == data.select_feature_idxs([2])
        items = list(data.items(over="features", group_by=["G1", "G1", "G2"]))
        assert len(items) == 2
        assert items[0][0] == "G1"
        assert items[0][1] == data.select_feature_idxs([0, 1])
        assert items[1][0] == "G2"
        assert items[1][1] == data.select_feature_idxs(2)
        items = list(data.items(over="features", group_by=["G1", "G1", "G2"], keep_2d=True))
        assert len(items) == 2
        assert items[0][0] == "G1"
        assert items[0][1] == data.select_feature_idxs([0, 1])
        assert items[1][0] == "G2"
        assert items[1][1] == data.select_feature_idxs([2])
        items = list(data.items(over="features", group_by=True))
        assert len(items) == 1
        assert items[0][0] == "group"
        assert items[0][1] == data
        items = list(data.items(over="keys", group_by="k"))
        assert len(items) == 2
        assert items[0][0] == "K1"
        assert items[0][1] == data.select_keys([data.keys[0], data.keys[1]])
        assert items[1][0] == "K2"
        assert items[1][1] == data.select_keys(data.keys[2])
        with pytest.raises(Exception):
            list(data.items(over="keys", group_by=True, apply_group_by=True))
        items = list(data.items(over="columns", group_by=True, apply_group_by=True))
        assert len(items) == 1
        assert items[0][0] == "group"
        assert items[0][1] == data.regroup(True)

    def test_as_params(self):
        data = MyData.pull(["S1", "S2", "S3"], shape=(5, 3), columns=["F1", "F2", "F3"])
        param = data.as_param()
        assert_index_equal(param.keys, data.symbol_wrapper.columns)
        param = data.as_param(group_by=True)
        assert_index_equal(param.keys, data.symbol_wrapper.get_columns(True))
        param = data.as_param(over="columns")
        assert_index_equal(param.keys, data.wrapper.columns)
        param = data.as_param(over="columns", group_by=True)
        assert_index_equal(param.keys, data.wrapper.get_columns(True))


# ############# custom ############# #


class TestCustom:
    def test_csv_data(self, tmp_path):
        sr = pd.Series(np.arange(10), index=pd.date_range("2020", periods=10, tz="utc"))
        sr.to_csv(tmp_path / "temp.csv")
        csv_data = vbt.CSVData.pull(tmp_path / "temp.csv")
        assert_series_equal(csv_data.get(), sr)
        csv_data = vbt.CSVData.pull("TEMP", paths=tmp_path / "temp.csv")
        assert csv_data.symbols[0] == "TEMP"
        assert_series_equal(csv_data.get(), sr)
        csv_data = vbt.CSVData.pull("TEMP", paths=[tmp_path / "temp.csv"])
        assert csv_data.symbols[0] == "TEMP"
        assert_series_equal(csv_data.get(), sr)
        csv_data = vbt.CSVData.pull(["TEMP"], paths=tmp_path / "temp.csv")
        assert csv_data.symbols[0] == "TEMP"
        assert_series_equal(csv_data.get()["TEMP"], sr.rename("TEMP"))
        csv_data = vbt.CSVData.pull(tmp_path / "temp.csv", start="2020-01-03")
        assert_series_equal(csv_data.get(), sr.iloc[2:], check_freq=False)
        assert csv_data.returned_kwargs["temp"]["last_row"] == 9
        csv_data = vbt.CSVData.pull(tmp_path / "temp.csv", end="2020-01-05")
        assert_series_equal(csv_data.get(), sr.iloc[:4], check_freq=False)
        assert csv_data.returned_kwargs["temp"]["last_row"] == 3
        csv_data = vbt.CSVData.pull(tmp_path / "temp.csv", start="2020-01-03", end="2020-01-05")
        assert_series_equal(csv_data.get(), sr.iloc[2:4], check_freq=False)
        assert csv_data.returned_kwargs["temp"]["last_row"] == 3
        csv_data = vbt.CSVData.pull(tmp_path / "temp.csv", start_row=2, end_row=4)
        assert_series_equal(csv_data.get(), sr.iloc[2:4], check_freq=False)
        assert csv_data.returned_kwargs["temp"]["last_row"] == 3
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.Index(["0", "1"], dtype="object")
        df.to_csv(tmp_path / "temp.csv")
        csv_data = vbt.CSVData.pull(tmp_path / "temp.csv", iterator=True)
        assert_frame_equal(csv_data.get(), df)
        csv_data = vbt.CSVData.pull(tmp_path / "temp.csv", chunksize=1)
        assert_frame_equal(csv_data.get(), df)
        csv_data = vbt.CSVData.pull(
            tmp_path / "temp.csv", chunksize=1, chunk_func=lambda x: list(x)[-1]
        )
        assert_frame_equal(csv_data.get(), df.iloc[[-1]], check_freq=False)
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.MultiIndex.from_tuples([("1", "2"), ("3", "4")], names=["a", "b"])
        df.to_csv(tmp_path / "temp.csv")
        csv_data = vbt.CSVData.pull(tmp_path / "temp.csv", header=[0, 1], start_row=0, end_row=2)
        assert_frame_equal(csv_data.get(), df.iloc[:2], check_freq=False)
        assert csv_data.returned_kwargs["temp"]["last_row"] == 1
        csv_data = csv_data.update()
        assert_frame_equal(csv_data.get(), df.iloc[:2], check_freq=False)
        assert csv_data.returned_kwargs["temp"]["last_row"] == 1
        csv_data = csv_data.update(end_row=3)
        csv_data.get()
        assert_frame_equal(csv_data.get(), df.iloc[:3], check_freq=False)
        assert csv_data.returned_kwargs["temp"]["last_row"] == 2
        csv_data = csv_data.update(end_row=None)
        assert_frame_equal(csv_data.get(), df)
        assert csv_data.returned_kwargs["temp"]["last_row"] == 9

        data1 = MyData.pull(shape=(5,))
        data2 = MyData.pull(shape=(6,))
        data3 = MyData.pull(shape=(7,))
        result_data = vbt.Data.from_data(
            {"data1": data1.get(), "data2": data2.get(), "data3": data3.get()}
        )

        data1.get().to_csv(tmp_path / "data1.csv")
        data2.get().to_csv(tmp_path / "data2.csv")
        data3.get().to_csv(tmp_path / "data3.csv")
        csv_data = vbt.CSVData.pull(tmp_path / "data*.csv")
        assert_frame_equal(csv_data.get(), result_data.get())
        (tmp_path / "data").mkdir(exist_ok=True)
        data1.get().to_csv(tmp_path / "data/data1.csv")
        data2.get().to_csv(tmp_path / "data/data2.csv")
        data3.get().to_csv(tmp_path / "data/data3.csv")
        csv_data = vbt.CSVData.pull(tmp_path / "data")
        assert_frame_equal(csv_data.get(), result_data.get())
        csv_data = vbt.CSVData.pull(
            [tmp_path / "data/data1.csv", tmp_path / "data/data2.csv", tmp_path / "data/data3.csv"],
        )
        assert_frame_equal(csv_data.get(), result_data.get())
        csv_data = vbt.CSVData.pull(
            paths=[
                tmp_path / "data/data1.csv",
                tmp_path / "data/data2.csv",
                tmp_path / "data/data3.csv",
            ],
        )
        assert_frame_equal(csv_data.get(), result_data.get())
        csv_data = vbt.CSVData.pull(
            symbols=["DATA1", "DATA2", "DATA3"],
            paths=[
                tmp_path / "data/data1.csv",
                tmp_path / "data/data2.csv",
                tmp_path / "data/data3.csv",
            ],
        )
        assert_frame_equal(
            csv_data.get(),
            result_data.get().rename(
                columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}
            ),
        )
        csv_data = vbt.CSVData.pull(
            vbt.symbol_dict(
                {
                    "DATA1": tmp_path / "data/data1.csv",
                    "DATA2": tmp_path / "data/data2.csv",
                    "DATA3": tmp_path / "data/data3.csv",
                }
            )
        )
        assert_frame_equal(
            csv_data.get(),
            result_data.get().rename(
                columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}
            ),
        )
        with pytest.raises(Exception):
            vbt.CSVData.pull("DATA")
        with pytest.raises(Exception):
            vbt.CSVData.pull("DATA", paths=tmp_path / "data/data*.csv")
        with pytest.raises(Exception):
            vbt.CSVData.pull(["DATA1", "DATA2"], paths=tmp_path / "data/data1.csv")
        with pytest.raises(Exception):
            vbt.CSVData.pull(None)
        with pytest.raises(Exception):
            vbt.CSVData.pull("S1")

    def test_hdf_data(self, tmp_path):
        sr = pd.Series(np.arange(10), index=pd.date_range("2020", periods=10, tz="utc"))
        sr.to_hdf(tmp_path / "temp.h5", "s", format="table")
        hdf_data = vbt.HDFData.pull(tmp_path / "temp.h5" / "s")
        assert_series_equal(hdf_data.get(), sr)
        hdf_data = vbt.HDFData.pull("S1", paths=tmp_path / "temp.h5" / "s")
        assert hdf_data.symbols[0] == "S1"
        assert_series_equal(hdf_data.get(), sr)
        hdf_data = vbt.HDFData.pull("S1", paths=[tmp_path / "temp.h5" / "s"])
        assert hdf_data.symbols[0] == "S1"
        assert_series_equal(hdf_data.get(), sr)
        hdf_data = vbt.HDFData.pull(["S1"], paths=tmp_path / "temp.h5" / "s")
        assert hdf_data.symbols[0] == "S1"
        assert_series_equal(hdf_data.get()["S1"], sr.rename("S1"))
        hdf_data = vbt.HDFData.pull(tmp_path / "temp.h5" / "s", start="2020-01-03")
        assert_series_equal(hdf_data.get(), sr.iloc[2:], check_freq=False)
        assert hdf_data.returned_kwargs["s"]["last_row"] == 9
        hdf_data = vbt.HDFData.pull(tmp_path / "temp.h5" / "s", end="2020-01-05")
        assert_series_equal(hdf_data.get(), sr.iloc[:4], check_freq=False)
        assert hdf_data.returned_kwargs["s"]["last_row"] == 3
        hdf_data = vbt.HDFData.pull(
            tmp_path / "temp.h5" / "s", start="2020-01-03", end="2020-01-05"
        )
        assert_series_equal(hdf_data.get(), sr.iloc[2:4], check_freq=False)
        assert hdf_data.returned_kwargs["s"]["last_row"] == 3
        hdf_data = vbt.HDFData.pull(tmp_path / "temp.h5" / "s", start_row=2, end_row=4)
        assert_series_equal(hdf_data.get(), sr.iloc[2:4], check_freq=False)
        assert hdf_data.returned_kwargs["s"]["last_row"] == 3
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.Index(["0", "1"], dtype="object")
        df.to_hdf(tmp_path / "temp.h5", "df", format="table")
        hdf_data = vbt.HDFData.pull(tmp_path / "temp.h5" / "df", iterator=True)
        assert_frame_equal(hdf_data.get(), df)
        hdf_data = vbt.HDFData.pull(tmp_path / "temp.h5" / "df", chunksize=1)
        assert_frame_equal(hdf_data.get(), df)
        hdf_data = vbt.HDFData.pull(
            tmp_path / "temp.h5" / "df", chunksize=1, chunk_func=lambda x: list(x)[-1]
        )
        assert_frame_equal(hdf_data.get(), df.iloc[[-1]], check_freq=False)
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.MultiIndex.from_tuples([("1", "2"), ("3", "4")], names=["a", "b"])
        df.to_hdf(tmp_path / "temp.h5", "df")
        hdf_data = vbt.HDFData.pull(
            tmp_path / "temp.h5" / "df", header=[0, 1], start_row=0, end_row=2
        )
        assert_frame_equal(hdf_data.get(), df.iloc[:2], check_freq=False)
        assert hdf_data.returned_kwargs["df"]["last_row"] == 1
        hdf_data = hdf_data.update()
        assert_frame_equal(hdf_data.get(), df.iloc[:2], check_freq=False)
        assert hdf_data.returned_kwargs["df"]["last_row"] == 1
        hdf_data = hdf_data.update(end_row=3)
        hdf_data.get()
        assert_frame_equal(hdf_data.get(), df.iloc[:3], check_freq=False)
        assert hdf_data.returned_kwargs["df"]["last_row"] == 2
        hdf_data = hdf_data.update(end_row=None)
        assert_frame_equal(hdf_data.get(), df)
        assert hdf_data.returned_kwargs["df"]["last_row"] == 9

        data1 = MyData.pull(shape=(5,))
        data2 = MyData.pull(shape=(6,))
        data3 = MyData.pull(shape=(7,))
        result_data = vbt.Data.from_data(
            {"data1": data1.get(), "data2": data2.get(), "data3": data3.get()}
        )

        data1.get().to_hdf(tmp_path / "data1.h5", "data1")
        data2.get().to_hdf(tmp_path / "data2.h5", "data2")
        data3.get().to_hdf(tmp_path / "data3.h5", "data3")
        hdf_data = vbt.HDFData.pull(tmp_path / "data*.h5")
        assert_frame_equal(hdf_data.get(), result_data.get())
        (tmp_path / "data").mkdir(exist_ok=True)
        data1.get().to_hdf(tmp_path / "data/data1.h5", "data1")
        data2.get().to_hdf(tmp_path / "data/data2.h5", "data2")
        data3.get().to_hdf(tmp_path / "data/data3.h5", "data3")
        hdf_data = vbt.HDFData.pull(tmp_path / "data")
        assert_frame_equal(hdf_data.get(), result_data.get())
        hdf_data = vbt.HDFData.pull(
            [tmp_path / "data/data1.h5", tmp_path / "data/data2.h5", tmp_path / "data/data3.h5"],
        )
        assert_frame_equal(hdf_data.get(), result_data.get())
        hdf_data = vbt.HDFData.pull(
            paths=[
                tmp_path / "data/data1.h5",
                tmp_path / "data/data2.h5",
                tmp_path / "data/data3.h5",
            ],
        )
        assert_frame_equal(hdf_data.get(), result_data.get())
        hdf_data = vbt.HDFData.pull(
            symbols=["DATA1", "DATA2", "DATA3"],
            paths=[
                tmp_path / "data/data1.h5",
                tmp_path / "data/data2.h5",
                tmp_path / "data/data3.h5",
            ],
        )
        assert_frame_equal(
            hdf_data.get(),
            result_data.get().rename(
                columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}
            ),
        )
        hdf_data = vbt.HDFData.pull(
            vbt.symbol_dict(
                {
                    "DATA1": tmp_path / "data/data1.h5/data1",
                    "DATA2": tmp_path / "data/data2.h5/data2",
                    "DATA3": tmp_path / "data/data3.h5/data3",
                }
            )
        )
        assert_frame_equal(
            hdf_data.get(),
            result_data.get().rename(
                columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}
            ),
        )
        with pytest.raises(Exception):
            vbt.HDFData.pull("DATA")
        with pytest.raises(Exception):
            vbt.HDFData.pull("DATA", paths=tmp_path / "data/data*.h5")
        with pytest.raises(Exception):
            vbt.HDFData.pull(["DATA1", "DATA2"], paths=tmp_path / "data/data1.h5")
        with pytest.raises(Exception):
            vbt.HDFData.pull(None)
        with pytest.raises(Exception):
            vbt.HDFData.pull("S1")

        (tmp_path / "data2").mkdir(exist_ok=True)
        data1.get().to_hdf(tmp_path / "data2/data.h5", "data1")
        data2.get().to_hdf(tmp_path / "data2/data.h5", "data2")
        data3.get().to_hdf(tmp_path / "data2/data.h5", "data3")
        hdf_data = vbt.HDFData.pull(tmp_path / "data2/data.h5")
        assert_frame_equal(hdf_data.get(), result_data.get())

        (tmp_path / "data3").mkdir(exist_ok=True)
        data1.get().to_hdf(tmp_path / "data3/data.h5", "data1")
        data2.get().to_hdf(tmp_path / "data3/data.h5", "data2")
        data3.get().to_hdf(tmp_path / "data3/data.h5", "data3")
        hdf_data = vbt.HDFData.pull(tmp_path / "data3")
        assert_frame_equal(hdf_data.get(), result_data.get())

        (tmp_path / "data4").mkdir(exist_ok=True)
        data1.get().to_hdf(tmp_path / "data4/data.h5", "/folder/data1")
        data2.get().to_hdf(tmp_path / "data4/data.h5", "/folder/data2")
        data3.get().to_hdf(tmp_path / "data4/data.h5", "/folder/data3")
        hdf_data = vbt.HDFData.pull(tmp_path / "data4/data.h5/folder")
        assert_frame_equal(hdf_data.get(), result_data.get())
        hdf_data = vbt.HDFData.pull(
            symbols=["DATA1", "DATA2", "DATA3"],
            paths=[
                tmp_path / "data4/data.h5/folder/data1",
                tmp_path / "data4/data.h5/folder/data2",
                tmp_path / "data4/data.h5/folder/data3",
            ],
        )
        assert_frame_equal(
            hdf_data.get(),
            result_data.get().rename(
                columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}
            ),
        )
        (tmp_path / "data5").mkdir(exist_ok=True)
        data1.get().to_hdf(tmp_path / "data5/data.h5", "/data1/folder")
        data2.get().to_hdf(tmp_path / "data5/data.h5", "/data2/folder")
        data3.get().to_hdf(tmp_path / "data5/data.h5", "/data3/folder")
        with pytest.raises(Exception):
            vbt.HDFData.pull(tmp_path / "data5/data.h5/folder")

        (tmp_path / "data6").mkdir(exist_ok=True)
        data1.get().to_hdf(tmp_path / "data6/data.h5", "/data1/folder/data1")
        data2.get().to_hdf(tmp_path / "data6/data.h5", "/data2/folder/data2")
        data3.get().to_hdf(tmp_path / "data6/data.h5", "/data3/folder/data3")
        hdf_data = vbt.HDFData.pull(tmp_path / "data6/data.h5/data*/folder/*")
        assert_frame_equal(hdf_data.get(), result_data.get())

        (tmp_path / "data7").mkdir(exist_ok=True)
        (tmp_path / "data7/data").mkdir(exist_ok=True)
        data1.get().to_hdf(tmp_path / "data7/data/data.h5", "/data1/folder/data1")
        data2.get().to_hdf(tmp_path / "data7/data/data.h5", "/data2/folder/data2")
        data3.get().to_hdf(tmp_path / "data7/data/data.h5", "/data3/folder/data3")
        hdf_data = vbt.HDFData.pull(tmp_path / "data7/**/data.h5/data*/folder/*")
        assert_frame_equal(hdf_data.get(), result_data.get())

        with pytest.raises(Exception):
            vbt.HDFData.pull(tmp_path / "data7/data/data.h5/folder/data4")

    def test_feather_data(self, tmp_path):
        sr = pd.Series(
            np.arange(10), index=pd.date_range("2020", periods=10, tz="utc"), name="hello"
        )
        sr.to_frame().reset_index().to_feather(tmp_path / "temp.feather")
        feather_data = vbt.FeatherData.pull(tmp_path / "temp.feather")
        assert_series_equal(feather_data.get(), sr)
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.Index(["0", "1"], dtype="object")
        df.reset_index().to_feather(tmp_path / "temp.feather")
        feather_data = vbt.FeatherData.pull(tmp_path / "temp.feather")
        assert_frame_equal(feather_data.get(), df)

    def test_parquet_data(self, tmp_path):
        sr = pd.Series(
            np.arange(10), index=pd.date_range("2020", periods=10, tz="utc"), name="hello"
        )
        sr.to_frame().to_parquet(tmp_path / "temp.parquet")
        parquet_data = vbt.ParquetData.pull(tmp_path / "temp.parquet")
        assert_series_equal(parquet_data.get(), sr)
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.Index(["0", "1"], dtype="object")
        df.to_parquet(tmp_path / "temp.parquet")
        parquet_data = vbt.ParquetData.pull(tmp_path / "temp.parquet")
        assert_frame_equal(parquet_data.get(), df)
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.Index(["0", "1"], dtype="object")
        df["group"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        df.to_parquet(tmp_path / "temp", partition_cols=["group"])
        parquet_data = vbt.ParquetData.pull(tmp_path / "temp")
        assert_frame_equal(parquet_data.get(), df.drop("group", axis=1))
        parquet_data = vbt.ParquetData.pull(tmp_path / "temp", keep_partition_cols=True)
        assert_frame_equal(parquet_data.get().astype(int), df)
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.Index(["0", "1"], dtype="object")
        df["group_0"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        df["group_1"] = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
        df.to_parquet(tmp_path / "temp2", partition_cols=["group_0", "group_1"])
        parquet_data = vbt.ParquetData.pull(tmp_path / "temp2")
        assert_frame_equal(parquet_data.get(), df.drop(["group_0", "group_1"], axis=1))
        parquet_data = vbt.ParquetData.pull(tmp_path / "temp2", keep_partition_cols=True)
        assert_frame_equal(parquet_data.get().astype(int), df)
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)), index=pd.date_range("2020", periods=10, tz="utc")
        )
        df.columns = pd.Index(["0", "1"], dtype="object")
        df["A"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        df["B"] = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
        df.to_parquet(tmp_path / "temp3", partition_cols=["A", "B"])
        parquet_data = vbt.ParquetData.pull(tmp_path / "temp3", keep_partition_cols=False)
        assert_frame_equal(parquet_data.get(), df.drop(["A", "B"], axis=1))
        parquet_data = vbt.ParquetData.pull(tmp_path / "temp3")
        assert_frame_equal(parquet_data.get().astype(int), df)

    def test_sql_data(self, tmp_path):
        from sqlalchemy import create_engine

        engine_url = "sqlite:///" + str(tmp_path / "temp.db")
        sr = pd.Series(np.arange(10), name="hello")
        sr.to_sql("SR", engine_url, if_exists="replace")
        sql_data = vbt.SQLData.pull("SR", engine=engine_url)
        assert_series_equal(sql_data.get(), sr)
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, start=2)
        assert_series_equal(sql_data.get(), sr.iloc[2:])
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, end=4)
        assert_series_equal(sql_data.get(), sr.iloc[:4])
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, start=2, end=4)
        assert_series_equal(sql_data.get(), sr.iloc[2:4])
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, start=2, align_dates=False)
        assert_series_equal(sql_data.get(), sr.iloc[2:])
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, end=4, align_dates=False)
        assert_series_equal(sql_data.get(), sr.iloc[:4])
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, start=2, end=4, align_dates=False)
        assert_series_equal(sql_data.get(), sr.iloc[2:4])
        sr = pd.Series(
            np.arange(10), index=pd.date_range("2020", periods=10, tz="utc"), name="hello"
        )
        sr.to_sql("SR", engine_url, if_exists="replace")
        sql_data = vbt.SQLData.pull("SR", engine=engine_url)
        assert_series_equal(sql_data.get(), sr)
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, start="2020-01-03")
        assert_series_equal(sql_data.get(), sr.iloc[2:])
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, end="2020-01-05")
        assert_series_equal(sql_data.get(), sr.iloc[:4])
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, start="2020-01-03", end="2020-01-05")
        assert_series_equal(sql_data.get(), sr.iloc[2:4], check_freq=False)
        sr = pd.Series(
            np.arange(10),
            index=pd.date_range("2020", periods=10, tz="America/New_York"),
            name="hello",
        )
        sr.tz_convert("utc").to_sql("SR", engine_url, if_exists="replace")
        sql_data = vbt.SQLData.pull("SR", engine=engine_url, tz="America/New_York")
        assert_series_equal(sql_data.get(), sr)
        sql_data = vbt.SQLData.pull(
            "SR", engine=engine_url, start="2020-01-03", tz="America/New_York"
        )
        assert_series_equal(sql_data.get(), sr.iloc[2:])
        sql_data = vbt.SQLData.pull(
            "SR", engine=engine_url, end="2020-01-05", tz="America/New_York"
        )
        assert_series_equal(sql_data.get(), sr.iloc[:4])
        sql_data = vbt.SQLData.pull(
            "SR", engine=engine_url, start="2020-01-03", end="2020-01-05", tz="America/New_York"
        )
        assert_series_equal(sql_data.get(), sr.iloc[2:4], check_freq=False)
        sql_data = vbt.SQLData.pull(
            "SR", engine=engine_url, end="2020-01-05", tz="America/New_York"
        )
        sql_data = sql_data.update(end=None)
        assert_series_equal(sql_data.get(), sr)

        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)),
            index=pd.date_range("2020", periods=10, tz="utc"),
            columns=pd.Index(["A", "B"]),
        )
        df["row_number"] = np.arange(len(df.index))
        df.to_sql("DF", engine_url, if_exists="replace")
        sql_data = vbt.SQLData.pull("DF", engine=engine_url)
        assert_frame_equal(sql_data.get(), df)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, start_row=2)
        assert_frame_equal(sql_data.get(), df.iloc[2:])
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, end_row=4)
        assert_frame_equal(sql_data.get(), df.iloc[:4])
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, start_row=2, end_row=4)
        assert_frame_equal(sql_data.get(), df.iloc[2:4], check_freq=False)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, end_row=4)
        sql_data = sql_data.update(end_row=None)
        assert_frame_equal(sql_data.get(), df)

        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)),
            index=pd.date_range("2020", periods=10, tz="utc"),
            columns=pd.Index(["A", "B"]),
        )
        df.to_sql("DF", engine_url, if_exists="replace")
        sql_data = vbt.SQLData.pull("DF", engine=engine_url)
        assert_frame_equal(sql_data.get(), df)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, chunksize=1)
        assert_frame_equal(sql_data.get(), df)
        sql_data = vbt.SQLData.pull(
            "DF", engine=engine_url, chunksize=1, chunk_func=lambda x: list(x)[-1]
        )
        assert_frame_equal(sql_data.get(), df.iloc[[-1]], check_freq=False)

        df = pd.DataFrame(
            np.arange(50).reshape((10, 5)),
            columns=pd.Index(["A", "B", "C", "D", "E"]),
        )
        df.to_sql("DF", engine_url, if_exists="replace")
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, columns=[1], squeeze=False)
        assert_frame_equal(sql_data.get(), df[["A"]])
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, columns=["A"], squeeze=False)
        assert_frame_equal(sql_data.get(), df[["A"]])
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, columns=["a"], squeeze=False)
        assert_frame_equal(sql_data.get(), df[["A"]])
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, columns=["index", "a"], squeeze=False)
        assert_frame_equal(sql_data.get(), df[["A"]])

        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col=1)
        assert_index_equal(sql_data.get().index, pd.Index(df["A"]))
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col=[1])
        assert_index_equal(sql_data.get().index, pd.Index(df["A"]))
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col="A")
        assert_index_equal(sql_data.get().index, pd.Index(df["A"]))
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col=["A"])
        assert_index_equal(sql_data.get().index, pd.Index(df["A"]))
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col="a")
        assert_index_equal(sql_data.get().index, pd.Index(df["A"]))
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col=["a"])
        assert_index_equal(sql_data.get().index, pd.Index(df["A"]))
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col=["a", "b"])
        assert_index_equal(sql_data.get().index, pd.MultiIndex.from_frame(df[["A", "B"]]))
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col=["a", "b"], start=(20, 21))
        assert_index_equal(sql_data.get().index, pd.MultiIndex.from_frame(df[["A", "B"]])[4:])
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, index_col=["a", "b"], end=(25, 26))
        assert_index_equal(sql_data.get().index, pd.MultiIndex.from_frame(df[["A", "B"]])[:5])
        sql_data = vbt.SQLData.pull(
            "DF", engine=engine_url, index_col=["a", "b"], start=(20, 21), end=(25, 26)
        )
        assert_index_equal(sql_data.get().index, pd.MultiIndex.from_frame(df[["A", "B"]])[4:5])

        sql_data = vbt.SQLData.pull(
            "DF",
            query='SELECT "index", "A" FROM DF',
            engine=engine_url,
            squeeze=False,
            index_col="index",
        )
        assert_frame_equal(sql_data.get(), df[["A"]])
        sql_data = vbt.SQLData.pull(
            "DF",
            query='SELECT "index", "A" FROM DF WHERE "index" >= 5',
            engine=engine_url,
            squeeze=False,
            index_col="index",
        )
        assert_frame_equal(sql_data.get(), df[["A"]].iloc[5:])
        sql_data = vbt.SQLData.pull(
            "DF",
            query='SELECT "index", "A" FROM DF WHERE "index" < 5',
            engine=engine_url,
            squeeze=False,
            index_col="index",
        )
        sql_data = sql_data.update()
        assert_frame_equal(sql_data.get(), df[["A"]].iloc[:5])
        sql_data = sql_data.update(query='SELECT "index", "A" FROM DF WHERE "index" >= 5')
        assert_frame_equal(sql_data.get(), df[["A"]])

        sql_data = vbt.SQLData.pull(
            "DF",
            engine=engine_url,
            parse_dates=["index"],
            to_utc=False,
            tz_localize=False,
            tz_convert=False,
        )
        new_df = df.copy(deep=False)
        new_df.index = pd.to_datetime(new_df.index, unit="s")
        assert_frame_equal(sql_data.get(), new_df, check_freq=False)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, parse_dates=0)
        new_df = df.copy(deep=False)
        new_df.index = pd.to_datetime(new_df.index, unit="s", utc=True)
        assert_frame_equal(sql_data.get(), new_df, check_freq=False)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, parse_dates=[0])
        new_df = df.copy(deep=False)
        new_df.index = pd.to_datetime(new_df.index, unit="s", utc=True)
        assert_frame_equal(sql_data.get(), new_df, check_freq=False)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, parse_dates="index")
        new_df = df.copy(deep=False)
        new_df.index = pd.to_datetime(new_df.index, unit="s", utc=True)
        assert_frame_equal(sql_data.get(), new_df, check_freq=False)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, parse_dates=["index"])
        new_df = df.copy(deep=False)
        new_df.index = pd.to_datetime(new_df.index, unit="s", utc=True)
        assert_frame_equal(sql_data.get(), new_df, check_freq=False)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, parse_dates={0: {"unit": "ns"}})
        new_df = df.copy(deep=False)
        new_df.index = pd.to_datetime(new_df.index, utc=True)
        assert_frame_equal(sql_data.get(), new_df, check_freq=False)
        sql_data = vbt.SQLData.pull("DF", engine=engine_url, parse_dates={"index": {"unit": "ns"}})
        new_df = df.copy(deep=False)
        new_df.index = pd.to_datetime(new_df.index, utc=True)
        assert_frame_equal(sql_data.get(), new_df, check_freq=False)

        engine_url = "sqlite:///" + str(tmp_path / "temp1.db")
        sr1 = pd.Series(np.arange(0, 10), name="hello")
        sr2 = pd.Series(np.arange(10, 20), name="hello")
        sr1.to_sql("SR1", engine_url, if_exists="replace")
        sr2.to_sql("SR2", engine_url, if_exists="replace")
        sql_data = vbt.SQLData.pull(engine=engine_url)
        assert_series_equal(sql_data.select("SR1").get(), sr1)
        assert_series_equal(sql_data.select("SR2").get(), sr2)
        sql_data = vbt.SQLData.pull(engine=create_engine(engine_url))
        assert_series_equal(sql_data.select("SR1").get(), sr1)
        assert_series_equal(sql_data.select("SR2").get(), sr2)
        vbt.settings.data.custom["sql"]["engine"] = engine_url
        sql_data = vbt.SQLData.pull()
        assert_series_equal(sql_data.select("SR1").get(), sr1)
        assert_series_equal(sql_data.select("SR2").get(), sr2)
        vbt.settings.data.custom["sql"]["engines"]["sqlite"] = dict(engine=engine_url)
        sql_data = vbt.SQLData.pull(engine="sqlite")
        assert_series_equal(sql_data.select("SR1").get(), sr1)
        assert_series_equal(sql_data.select("SR2").get(), sr2)
        sql_data = vbt.SQLData.pull(engine_name="sqlite")
        assert_series_equal(sql_data.select("SR1").get(), sr1)
        assert_series_equal(sql_data.select("SR2").get(), sr2)
        vbt.settings.data.custom["sql"]["engine"] = "sqlite"
        sql_data = vbt.SQLData.pull()
        assert_series_equal(sql_data.select("SR1").get(), sr1)
        assert_series_equal(sql_data.select("SR2").get(), sr2)
        vbt.settings.data.custom["sql"]["engine"] = None
        vbt.settings.data.custom["sql"]["engine_name"] = "sqlite"
        sql_data = vbt.SQLData.pull()
        assert_series_equal(sql_data.select("SR1").get(), sr1)
        assert_series_equal(sql_data.select("SR2").get(), sr2)

    def test_duckdb_data(self, tmp_path):
        from duckdb import connect, default_connection

        connection = connect(str(tmp_path / "database.duckdb"))
        sr = pd.Series(np.arange(10), name="hello")
        connection.register("_SR", sr.to_frame().reset_index())
        connection.execute("CREATE TABLE SR AS SELECT * FROM _SR")
        duckdb_data = vbt.DuckDBData.pull("SR", connection=connection)
        assert_series_equal(duckdb_data.get(), sr)
        connection.execute("DROP TABLE SR")
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)),
            index=pd.date_range("2020", periods=10, tz="utc"),
            columns=pd.Index(["A", "B"]),
        )
        connection.register("_DF", df.tz_localize(None).reset_index())
        connection.execute("CREATE TABLE DF AS SELECT * FROM _DF")
        duckdb_data = vbt.DuckDBData.pull("DF", connection=connection)
        assert_frame_equal(duckdb_data.get(), df)
        connection.execute("DROP TABLE DF")
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)),
            index=pd.date_range("2020", periods=10, tz="America/New_York"),
            columns=pd.Index(["A", "B"]),
        )
        connection.register("_DF", df.tz_convert("utc").tz_localize(None).reset_index())
        connection.execute("CREATE TABLE DF AS SELECT * FROM _DF")
        duckdb_data = vbt.DuckDBData.pull("DF", connection=connection, tz="America/New_York")
        assert_frame_equal(duckdb_data.get(), df)
        duckdb_data = vbt.DuckDBData.pull(
            "SYMBOL",
            query="SELECT * FROM DF",
            connection=connection,
            tz="America/New_York",
        )
        assert_frame_equal(duckdb_data.get(), df)
        duckdb_data = vbt.DuckDBData.pull(
            "SYMBOL",
            query="SELECT * FROM DF WHERE index < TIMESTAMP '2020-01-06 05:00:00.000000'",
            connection=connection,
            tz="America/New_York",
        )
        assert_frame_equal(duckdb_data.get(), df.iloc[:5])
        duckdb_data = vbt.DuckDBData.pull(
            "SYMBOL",
            query="SELECT * FROM DF WHERE index < $end",
            connection=connection,
            tz="America/New_York",
            params=dict(
                end=pd.Timestamp("2020-01-06", tz="America/New_York")
                .tz_convert("utc")
                .tz_localize(None)
                .to_pydatetime()
            ),
        )
        assert_frame_equal(duckdb_data.get(), df.iloc[:5])
        connection.execute("DROP TABLE DF")
        df = pd.DataFrame(
            np.arange(20).reshape((10, 2)),
            index=pd.date_range("2020", periods=10, tz="utc"),
            columns=pd.Index(["A", "B"]),
        )
        default_connection = default_connection()
        default_connection.register("_DF", df.tz_localize(None).reset_index())
        csv_path = tmp_path / "df.csv"
        default_connection.execute(f"COPY (SELECT * FROM _DF) TO '{str(csv_path)}'")
        duckdb_data = vbt.DuckDBData.pull(
            "SYMBOL", read_path=csv_path, read_options=dict(auto_detect=True)
        )
        duckdb_df = duckdb_data.get()
        duckdb_df.index = duckdb_df.index.astype("datetime64[ns, UTC]")
        assert_frame_equal(duckdb_df, df)
        parquet_path = tmp_path / "df.parquet"
        default_connection.execute(f"COPY (SELECT * FROM _DF) TO '{str(parquet_path)}'")
        duckdb_data = vbt.DuckDBData.pull("SYMBOL", read_path=parquet_path)
        duckdb_df = duckdb_data.get()
        duckdb_df.index = duckdb_df.index.astype("datetime64[ns, UTC]")
        assert_frame_equal(duckdb_df, df)

    def test_random_data(self):
        assert_series_equal(
            vbt.RandomData.pull(start="2021-01-01 UTC", end="2021-01-06 UTC", seed=42).get(),
            pd.Series(
                [
                    100.49671415301123,
                    100.35776307348756,
                    101.00776880200878,
                    102.54614727815496,
                    102.3060320136544,
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
            ),
        )
        assert_series_equal(
            vbt.RandomData.pull(
                start="2021-01-01 UTC", end="2021-01-06 UTC", symmetric=True, seed=42
            ).get(),
            pd.Series(
                [
                    100.49671415301123,
                    100.35795492796039,
                    101.00796189910105,
                    102.54634331617359,
                    102.30678851828695,
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
            ),
        )
        assert_frame_equal(
            vbt.RandomData.pull(
                columns=pd.Index([0, 1], name="path"),
                start="2021-01-01 UTC",
                end="2021-01-06 UTC",
                seed=42,
            ).get(),
            pd.DataFrame(
                [
                    [100.49671415301123, 99.7658630430508],
                    [100.35776307348756, 101.34137833772823],
                    [101.00776880200878, 102.11910727009419],
                    [102.54614727815496, 101.63968421831567],
                    [102.3060320136544, 102.1911405333112],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
                columns=pd.Index([0, 1], name="path"),
            ),
        )
        assert_frame_equal(
            vbt.RandomData.pull(
                ["S1", "S2"], start="2021-01-01 UTC", end="2021-01-06 UTC", seed=42
            ).get(),
            pd.DataFrame(
                [
                    [100.49671415301123, 100.49671415301123],
                    [100.35776307348756, 100.35776307348756],
                    [101.00776880200878, 101.00776880200878],
                    [102.54614727815496, 102.54614727815496],
                    [102.3060320136544, 102.3060320136544],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )

    def test_random_ohlc_data(self):
        assert_frame_equal(
            vbt.RandomOHLCData.pull(
                start="2021-01-01 UTC",
                end="2021-01-06 UTC",
                seed=42,
                n_ticks=10,
            ).get(),
            pd.DataFrame(
                [
                    [100.04967141530112, 100.4487295660908, 100.03583811740049, 100.4487295660908],
                    [100.40217984758935, 100.40217984758935, 99.65708696443218, 99.65708696443218],
                    [99.8031492512559, 99.8031492512559, 99.43592824293678, 99.43592824293678],
                    [99.37609698741984, 99.56016916393129, 99.10790499793339, 99.12741550259568],
                    [99.20061778610567, 99.21761762546865, 98.8767232135222, 98.8767232135222],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="1D",
                ),
                columns=pd.Index(["Open", "High", "Low", "Close"], dtype="object"),
            ),
        )

    def test_gbm_data(self):
        assert_series_equal(
            vbt.GBMData.pull(start="2021-01-01 UTC", end="2021-01-06 UTC", seed=42).get(),
            pd.Series(
                [
                    100.49292505095792,
                    100.34905764408163,
                    100.99606643427086,
                    102.54091282498935,
                    102.29597577584751,
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
            ),
        )
        assert_frame_equal(
            vbt.GBMData.pull(
                columns=pd.Index([0, 1], name="path"),
                start="2021-01-01 UTC",
                end="2021-01-06 UTC",
                seed=42,
            ).get(),
            pd.DataFrame(
                [
                    [100.49292505095792, 99.76114874768454],
                    [100.34905764408163, 101.34402779029647],
                    [100.99606643427086, 102.119662952671],
                    [102.54091282498935, 101.6362789823718],
                    [102.29597577584751, 102.1841061387023],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
                columns=pd.Index([0, 1], name="path"),
            ),
        )
        assert_frame_equal(
            vbt.GBMData.pull(
                ["S1", "S2"], start="2021-01-01 UTC", end="2021-01-06 UTC", seed=42
            ).get(),
            pd.DataFrame(
                [
                    [100.49292505095792, 100.49292505095792],
                    [100.34905764408163, 100.34905764408163],
                    [100.99606643427086, 100.99606643427086],
                    [102.54091282498935, 102.54091282498935],
                    [102.29597577584751, 102.29597577584751],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
                columns=pd.Index(["S1", "S2"], name="symbol"),
            ),
        )

    def test_gbm_ohlc_data(self):
        assert_frame_equal(
            vbt.GBMOHLCData.pull(
                start="2021-01-01 UTC",
                end="2021-01-06 UTC",
                seed=42,
                n_ticks=10,
            ).get(),
            pd.DataFrame(
                [
                    [
                        100.04963372876203,
                        100.44856416230552,
                        100.03575137446511,
                        100.44856416230552,
                    ],
                    [100.40197510375286, 100.40197510375286, 99.6569924965322, 99.6569924965322],
                    [99.80311183354723, 99.80311183354723, 99.4356577399506, 99.4356577399506],
                    [99.37579495605948, 99.55998737434406, 99.10782027760916, 99.12728312249477],
                    [99.20046274334108, 99.2174144041126, 98.87648952724908, 98.87648952724908],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="1D",
                ),
                columns=pd.Index(["Open", "High", "Low", "Close"], dtype="object"),
            ),
        )


# ############# updater ############# #


class TestDataUpdater:
    def test_update(self):
        data = MyData.pull("S1", shape=(5,), return_arr=True)
        updater = vbt.DataUpdater(data)
        updater.update()
        assert updater.data == data.update()
        assert updater.config["data"] == data.update()

    def test_update_every(self):
        data = MyData.pull("S1", shape=(5,), return_arr=True)
        kwargs = dict(call_count=0)

        class DataUpdater(vbt.DataUpdater):
            def update(self, kwargs):
                super().update()
                kwargs["call_count"] += 1
                if kwargs["call_count"] == 5:
                    raise vbt.CancelledError

        updater = DataUpdater(data)
        updater.update_every(kwargs=kwargs)
        for i in range(5):
            data = data.update()
        assert updater.data == data
        assert updater.config["data"] == data


# ############# saver ############# #


class TestCSVDataSaver:
    def test_update(self, tmp_path):
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        saver = vbt.CSVDataSaver(
            data,
            save_kwargs=dict(
                path_or_buf=tmp_path / "saver",
                mkdir_kwargs=dict(mkdir=True),
            ),
        )
        saver.init_save_data()
        saver.update(n=2)
        updated_data = data.update(n=2, concat=False)
        assert saver.data == updated_data
        saved_result0 = pd.concat((data.data["S1"].iloc[:-1], updated_data.data["S1"]), axis=0)
        saved_result0.index.freq = "D"
        saved_result1 = pd.concat((data.data["S2"].iloc[:-1], updated_data.data["S2"]), axis=0)
        saved_result1.index.freq = "D"
        assert_frame_equal(vbt.CSVData.pull(tmp_path / "saver").data["S1"], saved_result0)
        assert_frame_equal(vbt.CSVData.pull(tmp_path / "saver").data["S2"], saved_result1)

        new_data = saver.data
        new_saver = vbt.CSVDataSaver(
            new_data,
            save_kwargs=dict(
                path_or_buf=tmp_path / "saver",
                mkdir_kwargs=dict(mkdir=True),
            ),
        )
        new_saver.update(n=2)
        new_updated_data = new_data.update(n=2, concat=False)
        assert new_saver.data == new_updated_data
        new_saved_result0 = pd.concat(
            (data.data["S1"].iloc[:-1], new_data.data["S1"].iloc[:-1], new_updated_data.data["S1"]),
            axis=0,
        )
        new_saved_result0.index.freq = "D"
        new_saved_result1 = pd.concat(
            (data.data["S2"].iloc[:-1], new_data.data["S2"].iloc[:-1], new_updated_data.data["S2"]),
            axis=0,
        )
        new_saved_result1.index.freq = "D"
        assert_frame_equal(vbt.CSVData.pull(tmp_path / "saver").data["S1"], new_saved_result0)
        assert_frame_equal(vbt.CSVData.pull(tmp_path / "saver").data["S2"], new_saved_result1)

    def test_update_every(self, tmp_path):
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        call_count = [0]

        class CSVDataSaver(vbt.CSVDataSaver):
            def update(self, call_count, **kwargs):
                super().update(**kwargs)
                call_count[0] += 1
                if call_count[0] == 5:
                    raise vbt.CancelledError

        saver = CSVDataSaver(
            data,
            save_kwargs=dict(
                path_or_buf=tmp_path / "saver",
                mkdir_kwargs=dict(mkdir=True),
            ),
        )
        saver.init_save_data()
        saver.update_every(call_count=call_count)
        for i in range(5):
            data = data.update()
        assert_frame_equal(vbt.CSVData.pull(tmp_path / "saver").data["S1"], data.data["S1"])
        assert_frame_equal(vbt.CSVData.pull(tmp_path / "saver").data["S2"], data.data["S2"])


class TestHDFDataSaver:
    def test_update(self, tmp_path):
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        saver = vbt.HDFDataSaver(
            data,
            save_kwargs=dict(
                path_or_buf=tmp_path / "saver.h5",
                mkdir_kwargs=dict(mkdir=True),
                min_itemsize=10,
            ),
        )
        saver.init_save_data()
        saver.update(n=2)
        updated_data = data.update(n=2, concat=False)
        assert saver.data == updated_data
        saved_result0 = pd.concat((data.data["S1"].iloc[:-1], updated_data.data["S1"]), axis=0)
        saved_result0.index.freq = "D"
        saved_result1 = pd.concat((data.data["S2"].iloc[:-1], updated_data.data["S2"]), axis=0)
        saved_result1.index.freq = "D"
        assert_frame_equal(vbt.HDFData.pull(tmp_path / "saver.h5").data["S1"], saved_result0)
        assert_frame_equal(vbt.HDFData.pull(tmp_path / "saver.h5").data["S2"], saved_result1)

        new_data = saver.data
        new_saver = vbt.HDFDataSaver(
            new_data,
            save_kwargs=dict(
                path_or_buf=tmp_path / "saver.h5",
                mkdir_kwargs=dict(mkdir=True),
                min_itemsize=10,
            ),
        )
        new_saver.update(n=2)
        new_updated_data = new_data.update(n=2, concat=False)
        assert new_saver.data == new_updated_data
        new_saved_result0 = pd.concat(
            (data.data["S1"].iloc[:-1], new_data.data["S1"].iloc[:-1], new_updated_data.data["S1"]),
            axis=0,
        )
        new_saved_result0.index.freq = "D"
        new_saved_result1 = pd.concat(
            (data.data["S2"].iloc[:-1], new_data.data["S2"].iloc[:-1], new_updated_data.data["S2"]),
            axis=0,
        )
        new_saved_result1.index.freq = "D"
        assert_frame_equal(vbt.HDFData.pull(tmp_path / "saver.h5").data["S1"], new_saved_result0)
        assert_frame_equal(vbt.HDFData.pull(tmp_path / "saver.h5").data["S2"], new_saved_result1)

    def test_update_every(self, tmp_path):
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        call_count = [0]

        class HDFDataSaver(vbt.HDFDataSaver):
            def update(self, call_count, **kwargs):
                super().update(**kwargs)
                call_count[0] += 1
                if call_count[0] == 5:
                    raise vbt.CancelledError

        saver = HDFDataSaver(
            data,
            save_kwargs=dict(
                path_or_buf=tmp_path / "saver.h5",
                mkdir_kwargs=dict(mkdir=True),
                min_itemsize=10,
            ),
        )
        saver.init_save_data()
        saver.update_every(call_count=call_count)
        for i in range(5):
            data = data.update()
        assert_frame_equal(vbt.HDFData.pull(tmp_path / "saver.h5").data["S1"], data.data["S1"])
        assert_frame_equal(vbt.HDFData.pull(tmp_path / "saver.h5").data["S2"], data.data["S2"])


class TestSQLDataSaver:
    def test_update(self, tmp_path):
        engine_url = "sqlite:///" + str(tmp_path / "temp.db")
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])

        saver = vbt.SQLDataSaver(
            data,
            save_kwargs=dict(
                engine=engine_url,
            ),
        )
        saver.init_save_data()
        saver.update(n=2)
        updated_data = data.update(n=2, concat=False)
        assert saver.data == updated_data
        saved_result0 = pd.concat((data.data["S1"].iloc[:-1], updated_data.data["S1"]), axis=0)
        saved_result0.index.freq = "D"
        saved_result1 = pd.concat((data.data["S2"].iloc[:-1], updated_data.data["S2"]), axis=0)
        saved_result1.index.freq = "D"
        assert_frame_equal(vbt.SQLData.pull(engine=engine_url).data["S1"], saved_result0)
        assert_frame_equal(vbt.SQLData.pull(engine=engine_url).data["S2"], saved_result1)

        new_data = saver.data
        new_saver = vbt.SQLDataSaver(
            new_data,
            save_kwargs=dict(
                engine=engine_url,
            ),
        )
        new_saver.update(n=2)
        new_updated_data = new_data.update(n=2, concat=False)
        assert new_saver.data == new_updated_data
        new_saved_result0 = pd.concat(
            (data.data["S1"].iloc[:-1], new_data.data["S1"].iloc[:-1], new_updated_data.data["S1"]),
            axis=0,
        )
        new_saved_result0.index.freq = "D"
        new_saved_result1 = pd.concat(
            (data.data["S2"].iloc[:-1], new_data.data["S2"].iloc[:-1], new_updated_data.data["S2"]),
            axis=0,
        )
        new_saved_result1.index.freq = "D"
        assert_frame_equal(vbt.SQLData.pull(engine=engine_url).data["S1"], new_saved_result0)
        assert_frame_equal(vbt.SQLData.pull(engine=engine_url).data["S2"], new_saved_result1)

    def test_update_every(self, tmp_path):
        engine_url = "sqlite:///" + str(tmp_path / "temp.db")
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        call_count = [0]

        class SQLDataSaver(vbt.SQLDataSaver):
            def update(self, call_count, **kwargs):
                super().update(**kwargs)
                call_count[0] += 1
                if call_count[0] == 5:
                    raise vbt.CancelledError

        saver = SQLDataSaver(
            data,
            save_kwargs=dict(
                engine=engine_url,
            ),
        )
        saver.init_save_data()
        saver.update_every(call_count=call_count)
        for i in range(5):
            data = data.update()
        assert_frame_equal(vbt.SQLData.pull(engine=engine_url).data["S1"], data.data["S1"])
        assert_frame_equal(vbt.SQLData.pull(engine=engine_url).data["S2"], data.data["S2"])


class TestDuckDBDataSaver:
    def test_update(self, tmp_path):
        connection_url = str(tmp_path / "database.duckdb")
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])

        saver = vbt.DuckDBDataSaver(
            data,
            save_kwargs=dict(
                connection=connection_url,
            ),
        )
        saver.init_save_data()
        saver.update(n=2)
        updated_data = data.update(n=2, concat=False)
        assert saver.data == updated_data
        saved_result0 = pd.concat((data.data["S1"].iloc[:-1], updated_data.data["S1"]), axis=0)
        saved_result0.index.freq = "D"
        saved_result1 = pd.concat((data.data["S2"].iloc[:-1], updated_data.data["S2"]), axis=0)
        saved_result1.index.freq = "D"
        assert_frame_equal(vbt.DuckDBData.pull(connection=connection_url).data["S1"], saved_result0)
        assert_frame_equal(vbt.DuckDBData.pull(connection=connection_url).data["S2"], saved_result1)

        new_data = saver.data
        new_saver = vbt.DuckDBDataSaver(
            new_data,
            save_kwargs=dict(
                connection=connection_url,
            ),
        )
        new_saver.update(n=2)
        new_updated_data = new_data.update(n=2, concat=False)
        assert new_saver.data == new_updated_data
        new_saved_result0 = pd.concat(
            (data.data["S1"].iloc[:-1], new_data.data["S1"].iloc[:-1], new_updated_data.data["S1"]),
            axis=0,
        )
        new_saved_result0.index.freq = "D"
        new_saved_result1 = pd.concat(
            (data.data["S2"].iloc[:-1], new_data.data["S2"].iloc[:-1], new_updated_data.data["S2"]),
            axis=0,
        )
        new_saved_result1.index.freq = "D"
        assert_frame_equal(
            vbt.DuckDBData.pull(connection=connection_url).data["S1"], new_saved_result0
        )
        assert_frame_equal(
            vbt.DuckDBData.pull(connection=connection_url).data["S2"], new_saved_result1
        )

    def test_update_every(self, tmp_path):
        connection_url = str(tmp_path / "database.duckdb")
        data = MyData.pull(["S1", "S2"], shape=(5, 3), columns=["F1", "F2", "F3"])
        call_count = [0]

        class DuckDBDataSaver(vbt.DuckDBDataSaver):
            def update(self, call_count, **kwargs):
                super().update(**kwargs)
                call_count[0] += 1
                if call_count[0] == 5:
                    raise vbt.CancelledError

        saver = DuckDBDataSaver(
            data,
            save_kwargs=dict(
                connection=connection_url,
            ),
        )
        saver.init_save_data()
        saver.update_every(call_count=call_count)
        for i in range(5):
            data = data.update()
        assert_frame_equal(
            vbt.DuckDBData.pull(connection=connection_url).data["S1"], data.data["S1"]
        )
        assert_frame_equal(
            vbt.DuckDBData.pull(connection=connection_url).data["S2"], data.data["S2"]
        )
