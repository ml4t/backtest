# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a base class for records that utilize OHLC data."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb
from vectorbtpro.records.base import Records
from vectorbtpro.records.decorators import attach_shortcut_properties
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import ReadonlyConfig

if tp.TYPE_CHECKING:
    from vectorbtpro.data.base import Data as DataT
else:
    DataT = "vectorbtpro.data.base.Data"

__all__ = [
    "PriceRecords",
]

__pdoc__ = {}

price_records_shortcut_config = ReadonlyConfig(
    dict(
        bar_open_time=dict(obj_type="mapped"),
        bar_close_time=dict(obj_type="mapped"),
        bar_open=dict(obj_type="mapped"),
        bar_high=dict(obj_type="mapped"),
        bar_low=dict(obj_type="mapped"),
        bar_close=dict(obj_type="mapped"),
    )
)
"""_"""

__pdoc__[
    "price_records_shortcut_config"
] = f"""Configuration for shortcut properties attached to `PriceRecords`.

```python
{price_records_shortcut_config.prettify_doc()}
```
"""

PriceRecordsT = tp.TypeVar("PriceRecordsT", bound="PriceRecords")


@attach_shortcut_properties(price_records_shortcut_config)
class PriceRecords(Records):
    """Class extending `vectorbtpro.records.base.Records` for records that can make use of OHLC data.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        records_arr (RecordArray): Array of records.
        open (Optional[ArrayLike]): Array of open prices.
        high (Optional[ArrayLike]): Array of high prices.
        low (Optional[ArrayLike]): Array of low prices.
        close (Optional[ArrayLike]): Array of close prices.
        **kwargs: Keyword arguments for `vectorbtpro.records.base.Records`.
    """

    def __init__(
        self,
        wrapper: ArrayWrapper,
        records_arr: tp.RecordArray,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        Records.__init__(
            self,
            wrapper,
            records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )

        if open is not None:
            open = to_2d_array(open)
        if high is not None:
            high = to_2d_array(high)
        if low is not None:
            low = to_2d_array(low)
        if close is not None:
            close = to_2d_array(close)

        self._open = open
        self._high = high
        self._low = low
        self._close = close

    @classmethod
    def from_records(
        cls: tp.Type[PriceRecordsT],
        wrapper: ArrayWrapper,
        records: tp.RecordArray,
        data: tp.Optional[DataT] = None,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        attach_data: bool = True,
        **kwargs,
    ) -> PriceRecordsT:
        """Build `PriceRecords` from records.

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            records (RecordArray): Array of records.
            data (Optional[Data]): Data object containing OHLC data.
            open (Optional[ArrayLike]): Array of open prices.
            high (Optional[ArrayLike]): Array of high prices.
            low (Optional[ArrayLike]): Array of low prices.
            close (Optional[ArrayLike]): Array of close prices.
            attach_data (bool): Flag indicating whether to attach the OHLC data.
            **kwargs: Keyword arguments for `PriceRecords`.

        Returns:
            PriceRecords: New instance of `PriceRecords`.
        """
        if open is None and data is not None:
            open = data.open
        if high is None and data is not None:
            high = data.high
        if low is None and data is not None:
            low = data.low
        if close is None and data is not None:
            close = data.close
        return cls(
            wrapper,
            records,
            open=open if attach_data else None,
            high=high if attach_data else None,
            low=low if attach_data else None,
            close=close if attach_data else None,
            **kwargs,
        )

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[PriceRecordsT],
        *objs: tp.MaybeSequence[PriceRecordsT],
        **kwargs,
    ) -> tp.Kwargs:
        kwargs = Records.resolve_row_stack_kwargs(*objs, **kwargs)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PriceRecords):
                raise TypeError("Each object to be merged must be an instance of PriceRecords")
        for price_name in ("open", "high", "low", "close"):
            if price_name not in kwargs:
                price_objs = []
                stack_price_objs = True
                for obj in objs:
                    if getattr(obj, price_name) is not None:
                        price_objs.append(getattr(obj, price_name))
                    else:
                        stack_price_objs = False
                        break
                if stack_price_objs:
                    kwargs[price_name] = kwargs["wrapper"].row_stack_arrs(
                        *price_objs,
                        group_by=False,
                        wrap=False,
                    )
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[PriceRecordsT],
        *objs: tp.MaybeSequence[PriceRecordsT],
        reindex_kwargs: tp.KwargsLike = None,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `PriceRecords` after stacking along columns.

        Args:
            *objs (MaybeSequence[PriceRecords]): `PriceRecords` instances to be stacked.
            reindex_kwargs (KwargsLike): Keyword arguments for `pd.DataFrame.reindex`.
            ffill_close (bool): If True, forward-fill missing values in the close prices.
            fbfill_close (bool): If True, forward and backward-fill missing values in the close prices.
            **kwargs: Keyword arguments for `PriceRecords`.

        Returns:
            Kwargs: Resolved keyword arguments.
        """
        kwargs = Records.resolve_column_stack_kwargs(*objs, reindex_kwargs=reindex_kwargs, **kwargs)
        kwargs.pop("reindex_kwargs", None)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PriceRecords):
                raise TypeError("Each object to be merged must be an instance of PriceRecords")
        for price_name in ("open", "high", "low", "close"):
            if price_name not in kwargs:
                price_objs = []
                stack_price_objs = True
                for obj in objs:
                    if getattr(obj, "_" + price_name) is not None:
                        price_objs.append(getattr(obj, price_name))
                    else:
                        stack_price_objs = False
                        break
                if stack_price_objs:
                    new_price = kwargs["wrapper"].column_stack_arrs(
                        *price_objs,
                        reindex_kwargs=reindex_kwargs,
                        group_by=False,
                        wrap=True,
                    )
                    if price_name == "close":
                        if fbfill_close:
                            new_price = new_price.vbt.fbfill()
                        elif ffill_close:
                            new_price = new_price.vbt.ffill()
                    kwargs[price_name] = new_price.values
        return kwargs

    def indexing_func_meta(self, *args, records_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on `PriceRecords` and return updated metadata.

        Args:
            *args: Positional arguments for `vectorbtpro.records.base.Records.indexing_func_meta`.
            records_meta (DictLike): Metadata from the indexing operation on the records.
            **kwargs: Keyword arguments for `vectorbtpro.records.base.Records.indexing_func_meta`.

        Returns:
            dict: Dictionary containing updated indexing metadata and OHLC arrays.
        """
        if records_meta is None:
            records_meta = Records.indexing_func_meta(self, *args, **kwargs)
        prices = {}
        for price_name in ("open", "high", "low", "close"):
            if getattr(self, "_" + price_name) is not None:
                new_price = ArrayWrapper.select_from_flex_array(
                    getattr(self, "_" + price_name),
                    row_idxs=records_meta["wrapper_meta"]["row_idxs"],
                    col_idxs=records_meta["wrapper_meta"]["col_idxs"],
                    rows_changed=records_meta["wrapper_meta"]["rows_changed"],
                    columns_changed=records_meta["wrapper_meta"]["columns_changed"],
                )
            else:
                new_price = None
            prices[price_name] = new_price
        return {**records_meta, **prices}

    def indexing_func(
        self: PriceRecordsT, *args, price_records_meta: tp.DictLike = None, **kwargs
    ) -> PriceRecordsT:
        """Perform indexing on `PriceRecords`.

        Args:
            *args: Positional arguments for `PriceRecords.indexing_func_meta`.
            price_records_meta (DictLike): Metadata from the indexing operation on the price records.
            **kwargs: Keyword arguments for `PriceRecords.indexing_func_meta`.

        Returns:
            PriceRecords: New `PriceRecords` instance after indexing.
        """
        if price_records_meta is None:
            price_records_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=price_records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=price_records_meta["new_records_arr"],
            open=price_records_meta["open"],
            high=price_records_meta["high"],
            low=price_records_meta["low"],
            close=price_records_meta["close"],
        )

    def resample(
        self: PriceRecordsT,
        *args,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        records_meta: tp.DictLike = None,
        **kwargs,
    ) -> PriceRecordsT:
        """Perform resampling on `PriceRecords`.

        Args:
            *args: Positional arguments for `PriceRecords.resample_meta`.
            ffill_close (bool): If True, forward-fill missing values in the close prices.
            fbfill_close (bool): If True, forward and backward-fill missing values in the close prices.
            records_meta (DictLike): Metadata from the resampling operation on the records.
            **kwargs: Keyword arguments for `PriceRecords.resample_meta`.

        Returns:
            PriceRecords: New `PriceRecords` instance after resampling.
        """
        if records_meta is None:
            records_meta = self.resample_meta(*args, **kwargs)
        if self._open is None:
            new_open = None
        else:
            new_open = self.open.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.first_reduce_nb,
            )
        if self._high is None:
            new_high = None
        else:
            new_high = self.high.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.max_reduce_nb,
            )
        if self._low is None:
            new_low = None
        else:
            new_low = self.low.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.min_reduce_nb,
            )
        if self._close is None:
            new_close = None
        else:
            new_close = self.close.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.last_reduce_nb,
            )
            if fbfill_close:
                new_close = new_close.vbt.fbfill()
            elif ffill_close:
                new_close = new_close.vbt.ffill()
        return self.replace(
            wrapper=records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=records_meta["new_records_arr"],
            open=new_open,
            high=new_high,
            low=new_low,
            close=new_close,
        )

    @property
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Open price.

        Returns:
            Optional[SeriesFrame]: Wrapped open price data, or None if not available.
        """
        if self._open is None:
            return None
        return self.wrapper.wrap(self._open, group_by=False)

    @property
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """High price.

        Returns:
            Optional[SeriesFrame]: Wrapped high price data, or None if not available.
        """
        if self._high is None:
            return None
        return self.wrapper.wrap(self._high, group_by=False)

    @property
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Low price.

        Returns:
            Optional[SeriesFrame]: Wrapped low price data, or None if not available.
        """
        if self._low is None:
            return None
        return self.wrapper.wrap(self._low, group_by=False)

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """Close price series.

        Returns:
            Optional[SeriesFrame]: Wrapped close price data, or None if not available.
        """
        if self._close is None:
            return None
        return self.wrapper.wrap(self._close, group_by=False)

    def get_bar_open_time(self, **kwargs) -> MappedArray:
        """Return a mapped array with the opening time of the bar.

        Args:
            **kwargs: Keyword arguments for `PriceRecords.map_array`.

        Returns:
            MappedArray: Mapped array of bar open times.
        """
        return self.map_array(self.wrapper.index[self.idx_arr], **kwargs)

    def get_bar_close_time(self, **kwargs) -> MappedArray:
        """Return a mapped array with the closing time of the bar.

        Args:
            **kwargs: Keyword arguments for `PriceRecords.map_array`.

        Returns:
            MappedArray: Mapped array of bar close times.

        !!! note
            Ensure that `wrapper.freq` is provided, as it is required to compute the closing time.
        """
        if self.wrapper.freq is None:
            raise ValueError("Must provide frequency")
        return self.map_array(
            Resampler.get_rbound_index(
                index=self.wrapper.index[self.idx_arr], freq=self.wrapper.freq
            ),
            **kwargs,
        )

    def get_bar_open(self, **kwargs) -> MappedArray:
        """Return a mapped array with the open price of the bar.

        Args:
            **kwargs: Keyword arguments for `PriceRecords.apply`.

        Returns:
            MappedArray: Mapped array of bar open prices.
        """
        return self.apply(nb.bar_price_nb, self._open, **kwargs)

    def get_bar_high(self, **kwargs) -> MappedArray:
        """Return a mapped array with the high price of the bar.

        Args:
            **kwargs: Keyword arguments for `PriceRecords.apply`.

        Returns:
            MappedArray: Mapped array of bar high prices.
        """
        return self.apply(nb.bar_price_nb, self._high, **kwargs)

    def get_bar_low(self, **kwargs) -> MappedArray:
        """Return a mapped array with the low price of the bar.

        Args:
            **kwargs: Keyword arguments for `PriceRecords.apply`.

        Returns:
            MappedArray: Mapped array of bar low prices.
        """
        return self.apply(nb.bar_price_nb, self._low, **kwargs)

    def get_bar_close(self, **kwargs) -> MappedArray:
        """Return a mapped array with the close price of the bar.

        Args:
            **kwargs: Keyword arguments for `PriceRecords.apply`.

        Returns:
            MappedArray: Mapped array of bar close prices.
        """
        return self.apply(nb.bar_price_nb, self._close, **kwargs)
