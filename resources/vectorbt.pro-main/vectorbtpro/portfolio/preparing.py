# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes for preparing portfolio simulations."""

from collections import namedtuple
from functools import cached_property as cachedproperty

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.decorators import attach_arg_properties, override_arg_config
from vectorbtpro.base.preparing import BasePreparer
from vectorbtpro.base.reshaping import broadcast, broadcast_array_to, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.data.base import Data, OHLCDataMixin
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.sim_range import SimRangeMixin
from vectorbtpro.portfolio import enums, nb
from vectorbtpro.portfolio.call_seq import build_call_seq, require_call_seq
from vectorbtpro.portfolio.orders import FSOrders
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg, register_jitted
from vectorbtpro.signals import nb as signals_nb
from vectorbtpro.utils import checks
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.config import Configured, ReadonlyConfig, merge_dicts
from vectorbtpro.utils.mapping import to_field_mapping
from vectorbtpro.utils.template import CustomTemplate, RepFunc, substitute_templates
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "PFPrepResult",
    "BasePFPreparer",
    "FOPreparer",
    "FSPreparer",
    "FOFPreparer",
    "FDOFPreparer",
]

__pdoc__ = {}


@register_jitted(cache=True)
def valid_price_from_ago_1d_nb(price: tp.Array1d) -> tp.Array1d:
    """Compute the `from_ago` values for a price array.

    Args:
        price (Array1d): 1D array of prices.

    Returns:
        Array1d: Array where each element represents the number of steps since the last non-NaN price.
    """
    from_ago = np.empty(price.shape, dtype=int_)
    for i in range(price.shape[0] - 1, -1, -1):
        if i > 0 and not np.isnan(price[i]):
            for j in range(i - 1, -1, -1):
                if not np.isnan(price[j]):
                    break
            from_ago[i] = i - j
        else:
            from_ago[i] = 1
    return from_ago


PFPrepResultT = tp.TypeVar("PFPrepResultT", bound="PFPrepResult")


class PFPrepResult(Configured):
    """Class representing the result of portfolio preparation.

    Args:
        target_func (Optional[Callable]): Target function.
        target_args (KwargsLike): Dictionary of arguments for the target function.
        pf_args (KwargsLike): Dictionary of portfolio configuration arguments.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(
        self,
        target_func: tp.Optional[tp.Callable] = None,
        target_args: tp.KwargsLike = None,
        pf_args: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            target_func=target_func,
            target_args=target_args,
            pf_args=pf_args,
            **kwargs,
        )

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        """Target function from the configuration.

        Returns:
            Optional[Callable]: Target function.
        """
        return self.config["target_func"]

    @cachedproperty
    def target_args(self) -> tp.Kwargs:
        """Target arguments from the configuration.

        Returns:
            Kwargs: Dictionary containing target arguments.
        """
        return self.config["target_args"]

    @cachedproperty
    def pf_args(self) -> tp.KwargsLike:
        """Portfolio arguments from the configuration.

        Returns:
            KwargsLike: Dictionary containing portfolio arguments.
        """
        return self.config["pf_args"]


base_arg_config = ReadonlyConfig(
    dict(
        data=dict(),
        open=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        high=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        low=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        close=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        bm_close=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        cash_earnings=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        init_cash=dict(map_enum_kwargs=dict(enum=enums.InitCashMode, look_for_type=str)),
        init_position=dict(),
        init_price=dict(),
        cash_deposits=dict(),
        group_by=dict(),
        cash_sharing=dict(),
        freq=dict(),
        sim_start=dict(),
        sim_end=dict(),
        call_seq=dict(map_enum_kwargs=dict(enum=enums.CallSeqType, look_for_type=str)),
        attach_call_seq=dict(),
        keep_inout_flex=dict(),
        in_outputs=dict(has_default=False),
    )
)
"""_"""

__pdoc__["base_arg_config"] = f"""Argument configuration for `BasePFPreparer`.

```python
{base_arg_config.prettify_doc()}
```
"""


@attach_arg_properties
@override_arg_config(base_arg_config)
class BasePFPreparer(BasePreparer):
    """Base class for preparing portfolio simulations.

    This class preprocesses and aligns simulation parameters for portfolio analysis.

    !!! info
        For default settings, see `vectorbtpro._settings.portfolio`.
    """

    _settings_path: tp.SettingsPath = "portfolio"

    @classmethod
    def find_target_func(cls, target_func_name: str) -> tp.Callable:
        return getattr(nb, target_func_name)

    # ############# Ready arguments ############# #

    @cachedproperty
    def init_cash_mode(self) -> tp.Optional[int]:
        """Initial cash mode value used in portfolio simulation.

        Returns:
            Optional[int]: Initial cash mode if valid, otherwise None.
        """
        init_cash = self["init_cash"]
        if checks.is_int(init_cash) and init_cash in enums.InitCashMode:
            return init_cash
        return None

    @cachedproperty
    def group_by(self) -> tp.GroupByLike:
        """Grouping specification used for portfolio simulation.

        Returns:
            GroupByLike: Grouping specification, which can be None, a string, or an array-like object.
        """
        group_by = self["group_by"]
        if group_by is None and self.cash_sharing:
            return True
        return group_by

    @cachedproperty
    def auto_call_seq(self) -> bool:
        """Flag indicating whether automatic call sequence is enabled.

        Returns:
            bool: True if automatic call sequence is enabled, otherwise False.
        """
        call_seq = self["call_seq"]
        return checks.is_int(call_seq) and call_seq == enums.CallSeqType.Auto

    @classmethod
    def parse_data(
        cls,
        data: tp.Union[None, OHLCDataMixin, str, tp.ArrayLike],
        all_ohlc: bool = False,
    ) -> tp.Optional[OHLCDataMixin]:
        """Parse a data input into an OHLC data instance.

        Args:
            data (Union[None, OHLCDataMixin, str, ArrayLike]): Input data to parse.

                Can be an OHLC data instance, a string identifier, or an array-like object.
            all_ohlc (bool): Flag indicating whether to require all OHLC attributes.

        Returns:
            Optional[OHLCDataMixin]: Instance with OHLC data if parsing succeeds, otherwise None.
        """
        if data is None:
            return None
        if isinstance(data, OHLCDataMixin):
            return data
        if isinstance(data, str):
            return Data.from_data_str(data)
        if isinstance(data, pd.DataFrame):
            ohlcv_acc = data.vbt.ohlcv
            if all_ohlc and ohlcv_acc.has_ohlc:
                return ohlcv_acc
            if not all_ohlc and ohlcv_acc.has_any_ohlc:
                return ohlcv_acc
        return None

    @cachedproperty
    def data(self) -> tp.Optional[OHLCDataMixin]:
        """OHLC data used for portfolio simulation.

        Returns:
            Optional[OHLCDataMixin]: Instance with OHLC data if available, otherwise None.
        """
        return self.parse_data(self["data"])

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre__open(self) -> tp.ArrayLike:
        """Argument `open` before broadcasting.

        Returns:
            ArrayLike: Open prices before broadcasting.
        """
        open = self["open"]
        if open is None:
            if self.data is not None:
                open = self.data.open
            if open is None:
                return np.nan
        return open

    @cachedproperty
    def pre__high(self) -> tp.ArrayLike:
        """Argument `high` before broadcasting.

        Returns:
            ArrayLike: High prices before broadcasting.
        """
        high = self["high"]
        if high is None:
            if self.data is not None:
                high = self.data.high
            if high is None:
                return np.nan
        return high

    @cachedproperty
    def pre__low(self) -> tp.ArrayLike:
        """Argument `low` before broadcasting.

        Returns:
            ArrayLike: Low prices before broadcasting.
        """
        low = self["low"]
        if low is None:
            if self.data is not None:
                low = self.data.low
            if low is None:
                return np.nan
        return low

    @cachedproperty
    def pre__close(self) -> tp.ArrayLike:
        """Argument `close` before broadcasting.

        Returns:
            ArrayLike: Close prices before broadcasting.
        """
        close = self["close"]
        if close is None:
            if self.data is not None:
                close = self.data.close
            if close is None:
                return np.nan
        return close

    @cachedproperty
    def pre__bm_close(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `bm_close` before broadcasting.

        Returns:
            Optional[ArrayLike]: Benchmark close prices before broadcasting if available, otherwise None.
        """
        bm_close = self["bm_close"]
        if bm_close is not None and not isinstance(bm_close, bool):
            return bm_close
        return np.nan

    @cachedproperty
    def pre__init_cash(self) -> tp.ArrayLike:
        """Argument `init_cash` before broadcasting.

        Returns:
            ArrayLike: Initial cash before broadcasting.
        """
        if self.init_cash_mode is not None:
            return np.inf
        return self["init_cash"]

    @cachedproperty
    def pre__init_position(self) -> tp.ArrayLike:
        """Argument `init_position` before broadcasting.

        Returns:
            ArrayLike: Initial position before broadcasting.
        """
        return self["init_position"]

    @cachedproperty
    def pre__init_price(self) -> tp.ArrayLike:
        """Argument `init_price` before broadcasting.

        Returns:
            ArrayLike: Initial price before broadcasting.
        """
        return self["init_price"]

    @cachedproperty
    def pre__cash_deposits(self) -> tp.ArrayLike:
        """Argument `cash_deposits` before broadcasting.

        Returns:
            ArrayLike: Cash deposits before broadcasting.
        """
        return self["cash_deposits"]

    @cachedproperty
    def pre__freq(self) -> tp.Optional[tp.FrequencyLike]:
        """Argument `freq` before casting to nanosecond format.

        Returns:
            Optional[FrequencyLike]: Frequency before broadcasting.
        """
        freq = self["freq"]
        if freq is None and self.data is not None:
            return self.data.symbol_wrapper.freq
        return freq

    @cachedproperty
    def pre__call_seq(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `call_seq` before broadcasting.

        Returns:
            Optional[ArrayLike]: Call sequence before broadcasting.
        """
        if self.auto_call_seq:
            return None
        return self["call_seq"]

    @cachedproperty
    def pre__in_outputs(self) -> tp.Union[None, tp.NamedTuple, CustomTemplate]:
        """Argument `in_outputs` before broadcasting.

        Returns:
            Union[None, NamedTuple, CustomTemplate]: In-place outputs before broadcasting.
        """
        in_outputs = self["in_outputs"]
        if (
            in_outputs is not None
            and not isinstance(in_outputs, CustomTemplate)
            and not checks.is_namedtuple(in_outputs)
        ):
            in_outputs = to_field_mapping(in_outputs)
            in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        return in_outputs

    # ############# After broadcasting ############# #

    @cachedproperty
    def cs_group_lens(self) -> tp.GroupLens:
        """Cash sharing aware group lengths.

        Returns:
            GroupLens: Group lengths adjusted for cash sharing.
        """
        cs_group_lens = self.wrapper.grouper.get_group_lens(
            group_by=None if self.cash_sharing else False
        )
        checks.assert_subdtype(cs_group_lens, np.integer, arg_name="cs_group_lens")
        return cs_group_lens

    @cachedproperty
    def group_lens(self) -> tp.GroupLens:
        """Group lengths for portfolio data columns.

        Returns:
            GroupLens: Group lengths corresponding to the wrapper's columns.
        """
        return self.wrapper.grouper.get_group_lens(group_by=self.group_by)

    @cachedproperty
    def sim_group_lens(self) -> tp.GroupLens:
        """Simulation group lengths identical to group lengths.

        Returns:
            GroupLens: Simulation group lengths.
        """
        return self.group_lens

    def align_pc_arr(
        self,
        arr: tp.ArrayLike,
        group_lens: tp.Optional[tp.GroupLens] = None,
        check_dtype: tp.Optional[tp.DTypeLike] = None,
        cast_to_dtype: tp.Optional[tp.DTypeLike] = None,
        reduce_func: tp.Union[None, str, tp.Callable] = None,
        arg_name: tp.Optional[str] = None,
    ) -> tp.Array1d:
        """Align an array to match the portfolio's column structure.

        Args:
            arr (ArrayLike): Input array to align.
            group_lens (Optional[GroupLens]): Array defining the number of columns in each group.
            check_dtype (Optional[DTypeLike]): Data type to validate the array elements.
            cast_to_dtype (Optional[DTypeLike]): Target data type for casting.
            reduce_func (Union[None, str, Callable]): Reduction function to apply on groups, if applicable.
            arg_name (Optional[str]): Name of the argument for error messaging.

        Returns:
            Array1d: Aligned array with one element per portfolio column.
        """
        arr = np.asarray(arr)
        if check_dtype is not None:
            checks.assert_subdtype(arr, check_dtype, arg_name=arg_name)
        if cast_to_dtype is not None:
            arr = np.require(arr, dtype=cast_to_dtype)
        if arr.size > 1 and group_lens is not None and reduce_func is not None:
            if len(self.group_lens) == len(arr) != len(group_lens) == len(self.wrapper.columns):
                new_arr = np.empty(len(self.wrapper.columns), dtype=int_)
                col_generator = self.wrapper.grouper.iter_group_idxs()
                for i, cols in enumerate(col_generator):
                    new_arr[cols] = arr[i]
                arr = new_arr
            if len(self.wrapper.columns) == len(arr) != len(group_lens):
                new_arr = np.empty(len(group_lens), dtype=int_)
                col_generator = self.wrapper.grouper.iter_group_lens(group_lens)
                for i, cols in enumerate(col_generator):
                    if isinstance(reduce_func, str):
                        new_arr[i] = getattr(arr[cols], reduce_func)()
                    else:
                        new_arr[i] = reduce_func(arr[cols])
                arr = new_arr
        if group_lens is not None:
            return broadcast_array_to(arr, len(group_lens))
        return broadcast_array_to(arr, len(self.wrapper.columns))

    @cachedproperty
    def init_cash(self) -> tp.Array1d:
        """Aligned initial cash values computed across portfolio groups.

        Returns:
            Array1d: Initial cash values after alignment.
        """
        return self.align_pc_arr(
            self.pre__init_cash,
            group_lens=self.cs_group_lens,
            check_dtype=np.number,
            cast_to_dtype=float_,
            reduce_func="sum",
            arg_name="init_cash",
        )

    @cachedproperty
    def init_position(self) -> tp.Array1d:
        """Aligned initial position values for portfolio simulation.

        Returns:
            Array1d: Aligned initial position values.

        !!! note
            A warning is issued if non-zero positions are detected with undefined initial prices.
        """
        init_position = self.align_pc_arr(
            self.pre__init_position,
            check_dtype=np.number,
            cast_to_dtype=float_,
            arg_name="init_position",
        )
        if (((init_position > 0) | (init_position < 0)) & np.isnan(self.init_price)).any():
            warn("Initial position has undefined price. Set init_price.")
        return init_position

    @cachedproperty
    def init_price(self) -> tp.Array1d:
        """Initial price array aligned with portfolio configuration.

        Returns:
            Array1d: Aligned initial price values.
        """
        return self.align_pc_arr(
            self.pre__init_price,
            check_dtype=np.number,
            cast_to_dtype=float_,
            arg_name="init_price",
        )

    @cachedproperty
    def cash_deposits(self) -> tp.ArrayLike:
        """Cash deposits array broadcasted to match portfolio groups.

        Returns:
            ArrayLike: Cash deposits broadcasted to the required shape.
        """
        cash_deposits = self["cash_deposits"]
        checks.assert_subdtype(cash_deposits, np.number, arg_name="cash_deposits")
        return broadcast(
            cash_deposits,
            to_shape=(self.target_shape[0], len(self.cs_group_lens)),
            to_pd=False,
            keep_flex=self.keep_inout_flex,
            reindex_kwargs=dict(fill_value=0.0),
            require_kwargs=self.broadcast_kwargs.get("require_kwargs", {}),
        )

    @cachedproperty
    def auto_sim_start(self) -> tp.Optional[tp.ArrayLike]:
        """Automatically computed simulation start values.

        Returns:
            Optional[ArrayLike]: Automatic simulation start values, or None.
        """
        return None

    @cachedproperty
    def auto_sim_end(self) -> tp.Optional[tp.ArrayLike]:
        """Automatically computed simulation end values.

        Returns:
            Optional[ArrayLike]: Automatic simulation end values, or None.
        """
        return None

    @cachedproperty
    def sim_start(self) -> tp.Optional[tp.ArrayLike]:
        """Simulation start indices for the simulation range.

        If `sim_start` is set to "auto" (case-insensitive), the value is derived from
        automatic simulation start settings.

        Returns:
            Optional[ArrayLike]: Array of simulation start indices, or None.
        """
        sim_start = self["sim_start"]
        if sim_start is None:
            return None
        if isinstance(sim_start, str) and sim_start.lower() == "auto":
            sim_start = self.auto_sim_start
            if sim_start is None:
                return None
        sim_start_arr = np.asarray(sim_start)
        if np.issubdtype(sim_start_arr.dtype, np.integer):
            if sim_start_arr.ndim == 0:
                return sim_start
            new_sim_start = sim_start_arr
        else:
            if sim_start_arr.ndim == 0:
                return SimRangeMixin.resolve_sim_start_value(sim_start, wrapper=self.wrapper)
            new_sim_start = np.empty(len(sim_start), dtype=int_)
            for i in range(len(sim_start)):
                new_sim_start[i] = SimRangeMixin.resolve_sim_start_value(
                    sim_start[i], wrapper=self.wrapper
                )
        return self.align_pc_arr(
            new_sim_start,
            group_lens=self.sim_group_lens,
            check_dtype=np.integer,
            cast_to_dtype=int_,
            reduce_func="min",
            arg_name="sim_start",
        )

    @cachedproperty
    def sim_end(self) -> tp.Optional[tp.ArrayLike]:
        """Simulation end indices for the simulation range.

        If `sim_end` is set to "auto" (case-insensitive), the value is derived from
        automatic simulation end settings.

        Returns:
            Optional[ArrayLike]: Array of simulation end indices, or None.
        """
        sim_end = self["sim_end"]
        if sim_end is None:
            return None
        if isinstance(sim_end, str) and sim_end.lower() == "auto":
            sim_end = self.auto_sim_end
            if sim_end is None:
                return None
        sim_end_arr = np.asarray(sim_end)
        if np.issubdtype(sim_end_arr.dtype, np.integer):
            if sim_end_arr.ndim == 0:
                return sim_end
            new_sim_end = sim_end_arr
        else:
            if sim_end_arr.ndim == 0:
                return SimRangeMixin.resolve_sim_end_value(sim_end, wrapper=self.wrapper)
            new_sim_end = np.empty(len(sim_end), dtype=int_)
            for i in range(len(sim_end)):
                new_sim_end[i] = SimRangeMixin.resolve_sim_end_value(
                    sim_end[i], wrapper=self.wrapper
                )
        return self.align_pc_arr(
            new_sim_end,
            group_lens=self.sim_group_lens,
            check_dtype=np.integer,
            cast_to_dtype=int_,
            reduce_func="max",
            arg_name="sim_end",
        )

    @cachedproperty
    def call_seq(self) -> tp.Optional[tp.ArrayLike]:
        """Call sequence array for portfolio operations.

        Determines the order of method calls for portfolio updates.

        Returns:
            Optional[ArrayLike]: Computed call sequence, or None.
        """
        call_seq = self.pre__call_seq
        if call_seq is None and self.attach_call_seq:
            call_seq = enums.CallSeqType.Default
        if call_seq is not None:
            if checks.is_any_array(call_seq):
                call_seq = require_call_seq(
                    broadcast(call_seq, to_shape=self.target_shape, to_pd=False)
                )
            else:
                call_seq = build_call_seq(
                    self.target_shape, self.group_lens, call_seq_type=call_seq
                )
        if call_seq is not None:
            checks.assert_subdtype(call_seq, np.integer, arg_name="call_seq")
        return call_seq

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                group_lens=self.group_lens,
                cs_group_lens=self.cs_group_lens,
                cash_sharing=self.cash_sharing,
                init_cash=self.init_cash,
                init_position=self.init_position,
                init_price=self.init_price,
                cash_deposits=self.cash_deposits,
                sim_start=self.sim_start,
                sim_end=self.sim_end,
                call_seq=self.call_seq,
                auto_call_seq=self.auto_call_seq,
                attach_call_seq=self.attach_call_seq,
                in_outputs=self.pre__in_outputs,
            ),
            BasePreparer.template_context.func(self),
        )

    @cachedproperty
    def in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        """Template-substituted in-place outputs.

        Substitutes templates in the input outputs using the template context.

        Returns:
            Optional[NamedTuple]: Outputs after template substitution.
        """
        return substitute_templates(
            self.pre__in_outputs, self.template_context, eval_id="in_outputs"
        )

    # ############# Result ############# #

    @cachedproperty
    def pf_args(self) -> tp.KwargsLike:
        """Portfolio arguments for initialization.

        Constructs a dictionary of parameters—including market data arrays, cash, and
        price information—to be passed to the portfolio.

        Returns:
            KwargsLike: Dictionary containing portfolio parameters.
        """
        kwargs = dict()
        for k, v in self.config.items():
            if k not in self.arg_config and k != "arg_config":
                kwargs[k] = v
        return dict(
            wrapper=self.wrapper,
            open=self.open if self.pre__open is not np.nan else None,
            high=self.high if self.pre__high is not np.nan else None,
            low=self.low if self.pre__low is not np.nan else None,
            close=self.close,
            cash_sharing=self.cash_sharing,
            init_cash=self.init_cash if self.init_cash_mode is None else self.init_cash_mode,
            init_position=self.init_position,
            init_price=self.init_price,
            bm_close=(
                self.bm_close
                if (self["bm_close"] is not None and not isinstance(self["bm_close"], bool))
                else self["bm_close"]
            ),
            **kwargs,
        )

    @cachedproperty
    def result(self) -> PFPrepResult:
        """Portfolio preparation result.

        Encapsulates the target function, target arguments, and portfolio configuration used for initialization.

        Returns:
            PFPrepResult: Instance representing the portfolio preparation result.
        """
        return PFPrepResult(
            target_func=self.target_func, target_args=self.target_args, pf_args=self.pf_args
        )


BasePFPreparer.override_arg_config_doc(__pdoc__)

order_arg_config = ReadonlyConfig(
    dict(
        size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
            fill_default=False,
        ),
        price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PriceType.Close)),
        ),
        size_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.SizeType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.SizeType.Amount)),
        ),
        direction=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.Direction),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.Direction.Both)),
        ),
        fees=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        fixed_fees=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        slippage=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        min_size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        max_size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        size_granularity=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        leverage=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=1.0)),
        ),
        leverage_mode=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.LeverageMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.LeverageMode.Lazy)),
        ),
        reject_prob=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        price_area_vio_mode=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PriceAreaVioMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PriceAreaVioMode.Ignore)),
        ),
        allow_partial=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=True)),
        ),
        raise_reject=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        log=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
    )
)
"""_"""

__pdoc__["order_arg_config"] = f"""Configuration for order-related arguments.

```python
{order_arg_config.prettify_doc()}
```
"""

fo_arg_config = ReadonlyConfig(
    dict(
        cash_dividends=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        val_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ValPriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        from_ago=dict(
            broadcast=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0)),
        ),
        ffill_val_price=dict(),
        update_value=dict(),
        save_state=dict(),
        save_value=dict(),
        save_returns=dict(),
        skip_empty=dict(),
        max_order_records=dict(),
        max_log_records=dict(),
    )
)
"""_"""

__pdoc__["fo_arg_config"] = f"""Configuration for `FOPreparer` arguments.

```python
{fo_arg_config.prettify_doc()}
```
"""


@attach_arg_properties
@override_arg_config(fo_arg_config)
@override_arg_config(order_arg_config)
class FOPreparer(BasePFPreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_orders`.

    This class processes and configures input arguments necessary for constructing a portfolio from orders.

    !!! info
        For default settings, see `from_orders` in `vectorbtpro._settings.portfolio`.
    """

    _settings_path: tp.SettingsPath = "portfolio.from_orders"

    # ############# Ready arguments ############# #

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        """Staticized argument.

        Returns:
            StaticizedOption: Staticized option.

        Raises:
            ValueError: This method doesn't support staticization.
        """
        raise ValueError("This method doesn't support staticization")

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre__from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago` before broadcasting.

        Returns:
            ArrayLike: From-ago value before broadcasting.
        """
        from_ago = self["from_ago"]
        if from_ago is not None:
            return from_ago
        return 0

    @cachedproperty
    def pre__max_order_records(self) -> tp.Optional[int]:
        """Argument `max_order_records` before broadcasting.

        Returns:
            Optional[int]: Maximum order records before broadcasting.
        """
        return self["max_order_records"]

    @cachedproperty
    def pre__max_log_records(self) -> tp.Optional[int]:
        """Argument `max_log_records` before broadcasting.

        Returns:
            Optional[int]: Maximum log records before broadcasting.
        """
        return self["max_log_records"]

    # ############# After broadcasting ############# #

    @cachedproperty
    def sim_group_lens(self) -> tp.GroupLens:
        return self.cs_group_lens

    @cachedproperty
    def auto_sim_start(self) -> tp.Optional[tp.ArrayLike]:
        size = to_2d_array(self.size)
        if size.shape[0] == 1:
            return None
        first_valid_idx = generic_nb.first_valid_index_nb(size, check_inf=False)
        first_valid_idx = np.where(first_valid_idx == -1, 0, first_valid_idx)
        return first_valid_idx

    @cachedproperty
    def auto_sim_end(self) -> tp.Optional[tp.ArrayLike]:
        size = to_2d_array(self.size)
        if size.shape[0] == 1:
            return None
        last_valid_idx = generic_nb.last_valid_index_nb(size, check_inf=False)
        last_valid_idx = np.where(last_valid_idx == -1, len(self.wrapper.index), last_valid_idx + 1)
        return last_valid_idx

    @cachedproperty
    def price_and_from_ago(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike]:
        """Processed `price` and `from_ago` arguments after broadcasting.

        Returns:
            Tuple[ArrayLike, ArrayLike]: Tuple where the first element is the processed `price` array and
                the second element is the corresponding `from_ago` array.
        """
        price = self.post__price
        from_ago = self.post__from_ago
        if self["from_ago"] is None:
            if price.size == 1 or price.shape[0] == 1:
                next_open_mask = price == enums.PriceType.NextOpen
                next_close_mask = price == enums.PriceType.NextClose
                next_valid_open_mask = price == enums.PriceType.NextValidOpen
                next_valid_close_mask = price == enums.PriceType.NextValidClose

                if next_valid_open_mask.any() or next_valid_close_mask.any():
                    new_price = np.empty(self.wrapper.shape_2d, float_)
                    new_from_ago = np.empty(self.wrapper.shape_2d, int_)
                    if next_valid_open_mask.any():
                        open = broadcast_array_to(self.open, self.wrapper.shape_2d)
                    if next_valid_close_mask.any():
                        close = broadcast_array_to(self.close, self.wrapper.shape_2d)

                    for i in range(price.size):
                        price_item = price.item(i)
                        if price_item == enums.PriceType.NextOpen:
                            new_price[:, i] = enums.PriceType.Open
                            new_from_ago[:, i] = 1
                        elif price_item == enums.PriceType.NextClose:
                            new_price[:, i] = enums.PriceType.Close
                            new_from_ago[:, i] = 1
                        elif price_item == enums.PriceType.NextValidOpen:
                            new_price[:, i] = enums.PriceType.Open
                            new_from_ago[:, i] = valid_price_from_ago_1d_nb(open[:, i])
                        elif price_item == enums.PriceType.NextValidClose:
                            new_price[:, i] = enums.PriceType.Close
                            new_from_ago[:, i] = valid_price_from_ago_1d_nb(close[:, i])
                    price = new_price
                    from_ago = new_from_ago

                elif next_open_mask.any() or next_close_mask.any():
                    price = price.astype(float_)
                    price[next_open_mask] = enums.PriceType.Open
                    price[next_close_mask] = enums.PriceType.Close
                    from_ago = np.full(price.shape, 0, dtype=int_)
                    from_ago[next_open_mask] = 1
                    from_ago[next_close_mask] = 1
        return price, from_ago

    @cachedproperty
    def price(self) -> tp.ArrayLike:
        """Processed `price` argument derived from `price_and_from_ago`.

        Returns:
            ArrayLike: Processed `price` array.
        """
        return self.price_and_from_ago[0]

    @cachedproperty
    def from_ago(self) -> tp.ArrayLike:
        """Processed `from_ago` argument derived from `price_and_from_ago`.

        Returns:
            ArrayLike: Processed `from_ago` array.
        """
        return self.price_and_from_ago[1]

    @cachedproperty
    def max_order_records(self) -> tp.Optional[int]:
        """Maximum number of order records.

        Returns:
            Optional[int]: Maximum order records calculated based on the data shape and
                target configuration.
        """
        max_order_records = self.pre__max_order_records
        if max_order_records is None:
            _size = self.post__size
            if _size.size == 1:
                max_order_records = self.target_shape[0] * int(not np.isnan(_size.item(0)))
            else:
                if _size.shape[0] == 1 and self.target_shape[0] > 1:
                    max_order_records = self.target_shape[0] * int(np.any(~np.isnan(_size)))
                else:
                    max_order_records = int(np.max(np.sum(~np.isnan(_size), axis=0)))
        return max_order_records

    @cachedproperty
    def max_log_records(self) -> tp.Optional[int]:
        """Maximum number of log records.

        Returns:
            Optional[int]: Maximum log records determined based on the provided log data and
                target configuration.
        """
        max_log_records = self.pre__max_log_records
        if max_log_records is None:
            _log = self.post__log
            if _log.size == 1:
                max_log_records = self.target_shape[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and self.target_shape[0] > 1:
                    max_log_records = self.target_shape[0] * int(np.any(_log))
                else:
                    max_log_records = int(np.max(np.sum(_log, axis=0)))
        return max_log_records

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                group_lens=self.group_lens if self.dynamic_mode else self.cs_group_lens,
                ffill_val_price=self.ffill_val_price,
                update_value=self.update_value,
                save_state=self.save_state,
                save_value=self.save_value,
                save_returns=self.save_returns,
                max_order_records=self.max_order_records,
                max_log_records=self.max_log_records,
            ),
            BasePFPreparer.template_context.func(self),
        )

    # ############# Result ############# #

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        func = jit_reg.resolve_option(nb.from_orders_nb, self.jitted)
        func = ch_reg.resolve_option(func, self.chunked)
        return func

    @cachedproperty
    def target_arg_map(self) -> tp.Kwargs:
        target_arg_map = dict(BasePFPreparer.target_arg_map.func(self))
        target_arg_map["group_lens"] = "cs_group_lens"
        return target_arg_map


FOPreparer.override_arg_config_doc(__pdoc__)

fs_arg_config = ReadonlyConfig(
    dict(
        size=dict(
            fill_default=True,
        ),
        cash_dividends=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        entries=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        exits=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        long_entries=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        long_exits=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        short_entries=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        short_exits=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        adjust_func_nb=dict(),
        adjust_args=dict(type="args", substitute_templates=True),
        signal_func_nb=dict(),
        signal_args=dict(type="args", substitute_templates=True),
        post_signal_func_nb=dict(),
        post_signal_args=dict(type="args", substitute_templates=True),
        post_segment_func_nb=dict(),
        post_segment_args=dict(type="args", substitute_templates=True),
        order_mode=dict(),
        val_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ValPriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        accumulate=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.AccumulationMode, ignore_type=(int, bool)),
            subdtype=(np.integer, np.bool_),
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.AccumulationMode.Disabled)),
        ),
        upon_long_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.ConflictMode.Ignore)),
        ),
        upon_short_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.ConflictMode.Ignore)),
        ),
        upon_dir_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.DirectionConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(
                reindex_kwargs=dict(fill_value=enums.DirectionConflictMode.Ignore)
            ),
        ),
        upon_opposite_entry=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OppositeEntryMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(
                reindex_kwargs=dict(fill_value=enums.OppositeEntryMode.ReverseReduce)
            ),
        ),
        order_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OrderType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.OrderType.Market)),
        ),
        limit_delta=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        limit_tif=dict(
            broadcast=True,
            is_td=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        limit_expiry=dict(
            broadcast=True,
            is_dt=True,
            last_before=False,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        limit_reverse=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        limit_order_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.LimitOrderPrice, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.LimitOrderPrice.Limit)),
        ),
        upon_adj_limit_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(
                reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepIgnore)
            ),
        ),
        upon_opp_limit_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(
                reindex_kwargs=dict(fill_value=enums.PendingConflictMode.CancelExecute)
            ),
        ),
        use_stops=dict(),
        stop_ladder=dict(map_enum_kwargs=dict(enum=enums.StopLadderMode, look_for_type=str)),
        sl_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tsl_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tsl_th=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tp_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        td_stop=dict(
            broadcast=True,
            is_td=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        dt_stop=dict(
            broadcast=True,
            is_dt=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        stop_entry_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopEntryPrice, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopEntryPrice.Close)),
        ),
        stop_exit_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopExitPrice, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopExitPrice.Stop)),
        ),
        stop_exit_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopExitType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopExitType.Close)),
        ),
        stop_order_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OrderType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.OrderType.Market)),
        ),
        stop_limit_delta=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        upon_stop_update=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopUpdateMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopUpdateMode.Override)),
        ),
        upon_adj_stop_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(
                reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepExecute)
            ),
        ),
        upon_opp_stop_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(
                reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepExecute)
            ),
        ),
        delta_format=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.DeltaFormat),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.DeltaFormat.Percent)),
        ),
        time_delta_format=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.TimeDeltaFormat),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.TimeDeltaFormat.Index)),
        ),
        from_ago=dict(
            broadcast=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0)),
        ),
        ffill_val_price=dict(),
        update_value=dict(),
        fill_pos_info=dict(),
        save_state=dict(),
        save_value=dict(),
        save_returns=dict(),
        skip_empty=dict(),
        max_order_records=dict(),
        max_log_records=dict(),
        records=dict(
            rename_fields=dict(
                entry="entries",
                exit="exits",
                long_entry="long_entries",
                long_exit="long_exits",
                short_entry="short_entries",
                short_exit="short_exits",
            )
        ),
    )
)
"""_"""

__pdoc__["fs_arg_config"] = f"""Argument configuration for `FSPreparer`.

```python
{fs_arg_config.prettify_doc()}
```
"""


@attach_arg_properties
@override_arg_config(fs_arg_config)
@override_arg_config(order_arg_config)
class FSPreparer(BasePFPreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_signals`.

    This class processes and configures input arguments necessary for constructing a portfolio from signals.

    !!! info
        For default settings, see `from_signals` in `vectorbtpro._settings.portfolio`.
    """

    _settings_path: tp.SettingsPath = "portfolio.from_signals"

    # ############# Mode resolution ############# #

    @cachedproperty
    def pre__staticized(self) -> tp.StaticizedOption:
        """Pre-resolution value for the `staticized` argument.

        If `staticized` is a boolean, it is converted to a dict if True or set to None if False.
        If it is a dict and does not include a `"func"` key, a default function
        `vectorbtpro.portfolio.nb.from_signals.from_signal_func_nb` is added.

        Returns:
            StaticizedOption: Pre-resolved value for the `staticized` argument.
        """
        staticized = self["staticized"]
        if isinstance(staticized, bool):
            if staticized:
                staticized = dict()
            else:
                staticized = None
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if "func" not in staticized:
                staticized["func"] = nb.from_signal_func_nb
        return staticized

    @cachedproperty
    def order_mode(self) -> bool:
        """Processed value of the `order_mode` argument.

        If not set, defaults to False.

        Returns:
            bool: True if order mode is enabled, False otherwise.
        """
        order_mode = self["order_mode"]
        if order_mode is None:
            order_mode = False
        return order_mode

    @cachedproperty
    def dynamic_mode(self) -> bool:
        """Indicates whether dynamic mode is enabled.

        It is enabled if any of the adjustment, signal, post-signal, or post-segment functions are provided,
        or if order mode or a non-None staticized configuration is set.

        Returns:
            bool: True if dynamic mode is enabled, False otherwise.
        """
        return (
            self["adjust_func_nb"] is not None
            or self["signal_func_nb"] is not None
            or self["post_signal_func_nb"] is not None
            or self["post_segment_func_nb"] is not None
            or self.order_mode
            or self.pre__staticized is not None
        )

    @cachedproperty
    def implicit_mode(self) -> bool:
        """Indicates whether implicit signals mode is enabled.

        This is determined by the presence of the `entries` or `exits` arguments.

        Returns:
            bool: True if implicit signals mode is enabled, False otherwise.
        """
        return self["entries"] is not None or self["exits"] is not None

    @cachedproperty
    def explicit_mode(self) -> bool:
        """Indicates whether explicit mode is enabled.

        This is determined by the presence of the `long_entries` or `long_exits` arguments.

        Returns:
            bool: True if explicit mode is enabled, False otherwise.
        """
        return self["long_entries"] is not None or self["long_exits"] is not None

    @cachedproperty
    def pre__ls_mode(self) -> bool:
        """Indicates whether direction-aware mode is enabled before resolution.

        This is based on explicit mode or if `short_entries` or `short_exits` are provided.

        Returns:
            bool: True if pre-resolution direction-aware mode is enabled, False otherwise.
        """
        return (
            self.explicit_mode
            or self["short_entries"] is not None
            or self["short_exits"] is not None
        )

    @cachedproperty
    def pre__signals_mode(self) -> bool:
        """Indicates whether signals mode is enabled before resolution.

        It is enabled if either implicit signals mode or direction-aware mode is active.

        Returns:
            bool: True if pre-resolution signals mode is enabled, False otherwise.
        """
        return self.implicit_mode or self.pre__ls_mode

    @cachedproperty
    def ls_mode(self) -> bool:
        """Indicates whether direction-aware mode is enabled after resolution.

        If no pre-resolution signals mode, order mode, or signal function is detected, returns True.

        Returns:
            bool: True if direction-aware mode is enabled, False otherwise.

        Raises:
            ValueError: If both direction and short signal arrays are used together.
        """
        if not self.pre__signals_mode and not self.order_mode and self["signal_func_nb"] is None:
            return True
        ls_mode = self.pre__ls_mode
        if self.config.get("direction", None) is not None and ls_mode:
            raise ValueError("Direction and short signal arrays cannot be used together")
        return ls_mode

    @cachedproperty
    def signals_mode(self) -> bool:
        """Indicates whether signals mode is enabled after resolution.

        If no pre-resolution signals mode, order mode, or signal function is provided, returns True.

        Returns:
            bool: True if signals mode is enabled, False otherwise.

        Raises:
            ValueError: If both signals mode and order mode are activated simultaneously.
        """
        if not self.pre__signals_mode and not self.order_mode and self["signal_func_nb"] is None:
            return True
        signals_mode = self.pre__signals_mode
        if signals_mode and self.order_mode:
            raise ValueError("Signal arrays and order mode cannot be used together")
        return signals_mode

    @cachedproperty
    def signal_func_mode(self) -> bool:
        """Indicates whether signal function mode is enabled.

        This is active when dynamic mode is enabled but neither signals mode nor order mode is active.

        Returns:
            bool: True if signal function mode is enabled, False otherwise.
        """
        return self.dynamic_mode and not self.signals_mode and not self.order_mode

    @cachedproperty
    def adjust_func_nb(self) -> tp.Optional[tp.Callable]:
        """Processed `adjust_func_nb` argument.

        In dynamic mode, if `adjust_func_nb` is not provided, it returns
        `vectorbtpro.portfolio.nb.from_signals.no_adjust_func_nb`; otherwise,
        the provided callable is returned. If dynamic mode is inactive, returns None.

        Returns:
            Optional[Callable]: Adjustment function callable or None.
        """
        if self.dynamic_mode:
            if self["adjust_func_nb"] is None:
                return nb.no_adjust_func_nb
            return self["adjust_func_nb"]
        return None

    @cachedproperty
    def signal_func_nb(self) -> tp.Optional[tp.Callable]:
        """Processed `signal_func_nb` argument.

        In dynamic mode, if not explicitly provided, returns a default signal function based on the active mode:

        * `vectorbtpro.portfolio.nb.from_signals.ls_signal_func_nb` if direction-aware mode is active.
        * `vectorbtpro.portfolio.nb.from_signals.dir_signal_func_nb` if signals mode is active.
        * `vectorbtpro.portfolio.nb.from_signals.order_signal_func_nb` if order mode is active.

        If a value is provided, it is returned. Outside dynamic mode, returns None.

        Returns:
            Optional[Callable]: Signal function callable or None.
        """
        if self.dynamic_mode:
            if self["signal_func_nb"] is None:
                if self.ls_mode:
                    return nb.ls_signal_func_nb
                if self.signals_mode:
                    return nb.dir_signal_func_nb
                if self.order_mode:
                    return nb.order_signal_func_nb
                return None
            return self["signal_func_nb"]
        return None

    @cachedproperty
    def post_signal_func_nb(self) -> tp.Optional[tp.Callable]:
        """Processed `post_signal_func_nb` argument.

        In dynamic mode, if not provided, a default `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`
        is returned; otherwise, the provided callable is used.

        If dynamic mode is inactive, returns None.

        Returns:
            Optional[Callable]: Post signal function callable or None.
        """
        if self.dynamic_mode:
            if self["post_signal_func_nb"] is None:
                return nb.no_post_func_nb
            return self["post_signal_func_nb"]
        return None

    @cachedproperty
    def post_segment_func_nb(self) -> tp.Optional[tp.Callable]:
        """Processed `post_segment_func_nb` argument.

        In dynamic mode, if not provided, returns `vectorbtpro.portfolio.nb.from_signals.save_post_segment_func_nb`
        if saving state, value, or returns is enabled; otherwise, returns
        `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.

        If a value is provided, it is returned. Outside dynamic mode, returns None.

        Returns:
            Optional[Callable]: Post segment function callable or None.
        """
        if self.dynamic_mode:
            if self["post_segment_func_nb"] is None:
                if self.save_state or self.save_value or self.save_returns:
                    return nb.save_post_segment_func_nb
                return nb.no_post_func_nb
            return self["post_segment_func_nb"]
        return None

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        """Processed `staticized` argument.

        If dynamic mode is active, adapts the staticized dictionary by setting appropriate signal and
        adjustment functions based on the active mode and provided arguments.

        Returns:
            StaticizedOption: Adapted staticized argument.
        """
        staticized = self.pre__staticized
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if self.dynamic_mode:
                if self["signal_func_nb"] is None:
                    if self.ls_mode:
                        self.adapt_staticized_to_udf(
                            staticized, "ls_signal_func_nb", "signal_func_nb"
                        )
                        staticized["suggest_fname"] = "from_ls_signal_func_nb"
                    elif self.signals_mode:
                        self.adapt_staticized_to_udf(
                            staticized, "dir_signal_func_nb", "signal_func_nb"
                        )
                        staticized["suggest_fname"] = "from_dir_signal_func_nb"
                    elif self.order_mode:
                        self.adapt_staticized_to_udf(
                            staticized, "order_signal_func_nb", "signal_func_nb"
                        )
                        staticized["suggest_fname"] = "from_order_signal_func_nb"
                else:
                    self.adapt_staticized_to_udf(
                        staticized, self["signal_func_nb"], "signal_func_nb"
                    )
                if self["adjust_func_nb"] is not None:
                    self.adapt_staticized_to_udf(
                        staticized, self["adjust_func_nb"], "adjust_func_nb"
                    )
                if self["post_signal_func_nb"] is not None:
                    self.adapt_staticized_to_udf(
                        staticized, self["post_signal_func_nb"], "post_signal_func_nb"
                    )
                if self["post_segment_func_nb"] is not None:
                    self.adapt_staticized_to_udf(
                        staticized, self["post_segment_func_nb"], "post_segment_func_nb"
                    )
                elif self.save_state or self.save_value or self.save_returns:
                    self.adapt_staticized_to_udf(
                        staticized, "save_post_segment_func_nb", "post_segment_func_nb"
                    )
        return staticized

    @cachedproperty
    def pre__chunked(self) -> tp.ChunkedOption:
        """Pre-template substituted value for the `chunked` argument.

        Returns:
            ChunkedOption: Pre-template substituted chunked option.
        """
        return self["chunked"]

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre__entries(self) -> tp.ArrayLike:
        """Pre-broadcast value for the `entries` argument.

        Defaults to False if `entries` is not provided.

        Returns:
            ArrayLike: Pre-broadcast entries value.
        """
        return self["entries"] if self["entries"] is not None else False

    @cachedproperty
    def pre__exits(self) -> tp.ArrayLike:
        """Pre-broadcast value for the `exits` argument.

        Defaults to False if `exits` is not provided.

        Returns:
            ArrayLike: Pre-broadcast exits value.
        """
        return self["exits"] if self["exits"] is not None else False

    @cachedproperty
    def pre__long_entries(self) -> tp.ArrayLike:
        """Pre-broadcast value for the `long_entries` argument.

        Defaults to False if `long_entries` is not provided.

        Returns:
            ArrayLike: Pre-broadcast long entries value.
        """
        return self["long_entries"] if self["long_entries"] is not None else False

    @cachedproperty
    def pre__long_exits(self) -> tp.ArrayLike:
        """Pre-broadcast value for the `long_exits` argument.

        Defaults to False if `long_exits` is not provided.

        Returns:
            ArrayLike: Pre-broadcast long exits value.
        """
        return self["long_exits"] if self["long_exits"] is not None else False

    @cachedproperty
    def pre__short_entries(self) -> tp.ArrayLike:
        """Pre-broadcast value for the `short_entries` argument.

        Defaults to False if `short_entries` is not provided.

        Returns:
            ArrayLike: Pre-broadcast short entries value.
        """
        return self["short_entries"] if self["short_entries"] is not None else False

    @cachedproperty
    def pre__short_exits(self) -> tp.ArrayLike:
        """Pre-broadcast value for the `short_exits` argument.

        Defaults to False if `short_exits` is not provided.

        Returns:
            ArrayLike: Pre-broadcast short exits value.
        """
        return self["short_exits"] if self["short_exits"] is not None else False

    @cachedproperty
    def pre__from_ago(self) -> tp.ArrayLike:
        """Pre-broadcast value for the `from_ago` argument.

        If not provided, defaults to 0.

        Returns:
            ArrayLike: Pre-broadcast from-ago value.
        """
        from_ago = self["from_ago"]
        if from_ago is not None:
            return from_ago
        return 0

    @cachedproperty
    def pre__max_log_records(self) -> tp.Optional[int]:
        """Pre-resolution value for the `max_log_records` argument.

        Returns:
            Optional[int]: Pre-broadcast `max_log_records` value.
        """
        return self["max_log_records"]

    @classmethod
    def init_in_outputs(
        cls,
        wrapper: ArrayWrapper,
        group_lens: tp.Optional[tp.GroupLens] = None,
        cash_sharing: bool = False,
        save_state: bool = True,
        save_value: bool = True,
        save_returns: bool = True,
    ) -> enums.FSInOutputs:
        """Initialize a `vectorbtpro.portfolio.enums.FSInOutputs` instance.

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            group_lens (Optional[GroupLens]): Array defining the number of columns in each group.

                If None and `cash_sharing` is True, they are computed from the wrapper.
            cash_sharing (bool): Flag indicating whether cash is shared among assets of the same group.
            save_state (bool): Flag to record the account state.

                See `vectorbtpro.portfolio.enums.AccountState`.
            save_value (bool): Flag to record the portfolio value.
            save_returns (bool): Flag to record the portfolio returns.

        Returns:
            FSInOutputs: Initialized `vectorbtpro.portfolio.enums.FSInOutputs` instance.
        """
        if cash_sharing:
            if group_lens is None:
                group_lens = wrapper.grouper.get_group_lens()
        return nb.init_FSInOutputs_nb(
            wrapper.shape_2d,
            group_lens,
            cash_sharing=cash_sharing,
            save_state=save_state,
            save_value=save_value,
            save_returns=save_returns,
        )

    @cachedproperty
    def pre__in_outputs(self) -> tp.Union[None, tp.NamedTuple, CustomTemplate]:
        if self.dynamic_mode:
            if self["post_segment_func_nb"] is None:
                if self.save_state or self.save_value or self.save_returns:
                    return RepFunc(self.init_in_outputs)
            return BasePFPreparer.pre__in_outputs.func(self)
        if self["in_outputs"] is not None:
            raise ValueError("Argument in_outputs cannot be used in fixed mode")
        return None

    # ############# Broadcasting ############# #

    @cachedproperty
    def def_broadcast_kwargs(self) -> tp.Kwargs:
        def_broadcast_kwargs = dict(BasePFPreparer.def_broadcast_kwargs.func(self))
        new_def_broadcast_kwargs = dict()
        if self.order_mode:
            new_def_broadcast_kwargs["keep_flex"] = dict(
                size=False,
                size_type=False,
                min_size=False,
                max_size=False,
            )
            new_def_broadcast_kwargs["min_ndim"] = dict(
                size=2,
                size_type=2,
                min_size=2,
                max_size=2,
            )
            new_def_broadcast_kwargs["require_kwargs"] = dict(
                size=dict(requirements="O"),
                size_type=dict(requirements="O"),
                min_size=dict(requirements="O"),
                max_size=dict(requirements="O"),
            )
        if self.stop_ladder:
            new_def_broadcast_kwargs["axis"] = dict(
                sl_stop=1,
                tsl_stop=1,
                tp_stop=1,
                td_stop=1,
                dt_stop=1,
            )
            new_def_broadcast_kwargs["merge_kwargs"] = dict(
                sl_stop=dict(reset_index="from_start", fill_value=np.nan),
                tsl_stop=dict(reset_index="from_start", fill_value=np.nan),
                tp_stop=dict(reset_index="from_start", fill_value=np.nan),
                td_stop=dict(reset_index="from_start", fill_value=-1),
                dt_stop=dict(reset_index="from_start", fill_value=-1),
            )
        return merge_dicts(def_broadcast_kwargs, new_def_broadcast_kwargs)

    # ############# After broadcasting ############# #

    @cachedproperty
    def sim_group_lens(self) -> tp.GroupLens:
        if not self.dynamic_mode:
            return self.cs_group_lens
        return self.group_lens

    @cachedproperty
    def signals(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike, tp.ArrayLike, tp.ArrayLike]:
        """Post-broadcast value for the signal arrays.

        Returns:
            Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: Tuple containing:

                * `long_entries`: Broadcasted long entry signals.
                * `long_exits`: Broadcasted long exit signals.
                * `short_entries`: Broadcasted short entry signals.
                * `short_exits`: Broadcasted short exit signals.
        """
        if not self.dynamic_mode and not self.ls_mode:
            entries = self.post__entries
            exits = self.post__exits
            direction = self.post__direction
            if direction.size == 1:
                _direction = direction.item(0)
                if _direction == enums.Direction.LongOnly:
                    long_entries = entries
                    long_exits = exits
                    short_entries = np.array([[False]])
                    short_exits = np.array([[False]])
                elif _direction == enums.Direction.ShortOnly:
                    long_entries = np.array([[False]])
                    long_exits = np.array([[False]])
                    short_entries = entries
                    short_exits = exits
                else:
                    long_entries = entries
                    long_exits = np.array([[False]])
                    short_entries = exits
                    short_exits = np.array([[False]])
            else:
                return nb.dir_to_ls_signals_nb(
                    target_shape=self.target_shape,
                    entries=entries,
                    exits=exits,
                    direction=direction,
                )
        else:
            if self.explicit_mode and self.implicit_mode:
                long_entries = self.post__entries | self.post__long_entries
                long_exits = self.post__exits | self.post__long_exits
                short_entries = self.post__entries | self.post__short_entries
                short_exits = self.post__exits | self.post__short_exits
            elif self.explicit_mode:
                long_entries = self.post__long_entries
                long_exits = self.post__long_exits
                short_entries = self.post__short_entries
                short_exits = self.post__short_exits
            else:
                long_entries = self.post__entries
                long_exits = self.post__exits
                short_entries = self.post__short_entries
                short_exits = self.post__short_exits
        return long_entries, long_exits, short_entries, short_exits

    @cachedproperty
    def long_entries(self) -> tp.ArrayLike:
        """Post-broadcast value for the `long_entries` argument.

        Returns:
            ArrayLike: Broadcasted long entries array.
        """
        return self.signals[0]

    @cachedproperty
    def long_exits(self) -> tp.ArrayLike:
        """Post-broadcast value for the `long_exits` argument.

        Returns:
            ArrayLike: Broadcasted long exits array.
        """
        return self.signals[1]

    @cachedproperty
    def short_entries(self) -> tp.ArrayLike:
        """Post-broadcast value for the `short_entries` argument.

        Returns:
            ArrayLike: Broadcasted short entries array.
        """
        return self.signals[2]

    @cachedproperty
    def short_exits(self) -> tp.ArrayLike:
        """Post-broadcast value for the `short_exits` argument.

        Returns:
            ArrayLike: Broadcasted short exits array.
        """
        return self.signals[3]

    @cachedproperty
    def combined_mask(self) -> tp.Array2d:
        """Combined signal mask computed using element-wise OR on all signal arrays.

        Returns:
            Array2d: 2D array representing the combined signal mask.
        """
        long_entries = to_2d_array(self.long_entries)
        long_exits = to_2d_array(self.long_exits)
        short_entries = to_2d_array(self.short_entries)
        short_exits = to_2d_array(self.short_exits)
        return long_entries | long_exits | short_entries | short_exits

    @cachedproperty
    def auto_sim_start(self) -> tp.Optional[tp.ArrayLike]:
        if self.combined_mask.shape[0] == 1:
            return None
        first_signal_idx = signals_nb.nth_index_nb(self.combined_mask, 0)
        return np.where(first_signal_idx == -1, 0, first_signal_idx)

    @cachedproperty
    def auto_sim_end(self) -> tp.Optional[tp.ArrayLike]:
        if self.combined_mask.shape[0] == 1:
            return None
        last_signal_idx = signals_nb.nth_index_nb(self.combined_mask, -1)
        return np.where(last_signal_idx == -1, len(self.wrapper.index), last_signal_idx + 1)

    @cachedproperty
    def price_and_from_ago(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike]:
        """Post-broadcast value for the `price` and `from_ago` arguments.

        Returns:
            Tuple[ArrayLike, ArrayLike]: Tuple containing:

                * `price`: Broadcasted price array.
                * `from_ago`: Broadcasted from-ago array.
        """
        price = self.post__price
        from_ago = self.post__from_ago
        if self["from_ago"] is None:
            if price.size == 1 or price.shape[0] == 1:
                next_open_mask = price == enums.PriceType.NextOpen
                next_close_mask = price == enums.PriceType.NextClose
                if next_open_mask.any() or next_close_mask.any():
                    price = price.astype(float_)
                    price[next_open_mask] = enums.PriceType.Open
                    price[next_close_mask] = enums.PriceType.Close
                    from_ago = np.full(price.shape, 0, dtype=int_)
                    from_ago[next_open_mask] = 1
                    from_ago[next_close_mask] = 1
        return price, from_ago

    @cachedproperty
    def price(self) -> tp.ArrayLike:
        """Post-broadcast value for the `price` argument.

        Returns:
            ArrayLike: Broadcasted price array.
        """
        return self.price_and_from_ago[0]

    @cachedproperty
    def from_ago(self) -> tp.ArrayLike:
        """Post-broadcast value for the `from_ago` argument.

        Returns:
            ArrayLike: Broadcasted from-ago array.
        """
        return self.price_and_from_ago[1]

    @cachedproperty
    def max_log_records(self) -> tp.Optional[int]:
        """Post-broadcast value for the `max_log_records` argument.

        Returns:
            Optional[int]: Calculated number of maximum log records.
        """
        max_log_records = self.pre__max_log_records
        if max_log_records is None:
            _log = self.post__log
            if _log.size == 1:
                max_log_records = self.target_shape[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and self.target_shape[0] > 1:
                    max_log_records = self.target_shape[0] * int(np.any(_log))
                else:
                    max_log_records = int(np.max(np.sum(_log, axis=0)))
        return max_log_records

    @cachedproperty
    def use_stops(self) -> bool:
        """Post-broadcast value for the `use_stops` argument.

        Indicates whether stop orders are enabled based on configuration and signal conditions.

        Returns:
            bool: True if stop orders should be used; otherwise, False.
        """
        if self["use_stops"] is None:
            if self.stop_ladder:
                use_stops = True
            else:
                if self.dynamic_mode:
                    use_stops = True
                else:
                    if (
                        not np.all(np.isnan(self.sl_stop))
                        or not np.all(np.isnan(self.tsl_stop))
                        or not np.all(np.isnan(self.tp_stop))
                        or np.any(self.td_stop != -1)
                        or np.any(self.dt_stop != -1)
                    ):
                        use_stops = True
                    else:
                        use_stops = False
        else:
            use_stops = self["use_stops"]
        return use_stops

    @cachedproperty
    def use_limit_orders(self) -> bool:
        """Indicates whether limit orders are active based on order types and stop order settings.

        Returns:
            bool: True if limit orders are used; otherwise, False.
        """
        if np.any(self.order_type == enums.OrderType.Limit):
            return True
        if self.use_stops and np.any(self.stop_order_type == enums.OrderType.Limit):
            return True
        return False

    @cachedproperty
    def basic_mode(self) -> bool:
        """Indicates whether basic mode is active by verifying that neither stop orders nor limit orders are applied.

        Returns:
            bool: True if basic mode is enabled; otherwise, False.
        """
        return not self.use_stops and not self.use_limit_orders

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                order_mode=self.order_mode,
                use_stops=self.use_stops,
                stop_ladder=self.stop_ladder,
                adjust_func_nb=self.adjust_func_nb,
                adjust_args=self.pre__adjust_args,
                signal_func_nb=self.signal_func_nb,
                signal_args=self.pre__signal_args,
                post_signal_func_nb=self.post_signal_func_nb,
                post_signal_args=self.pre__post_signal_args,
                post_segment_func_nb=self.post_segment_func_nb,
                post_segment_args=self.pre__post_segment_args,
                ffill_val_price=self.ffill_val_price,
                update_value=self.update_value,
                fill_pos_info=self.fill_pos_info,
                save_state=self.save_state,
                save_value=self.save_value,
                save_returns=self.save_returns,
                max_order_records=self.max_order_records,
                max_log_records=self.max_log_records,
            ),
            BasePFPreparer.template_context.func(self),
        )

    @cachedproperty
    def signal_args(self) -> tp.Args:
        """Tuple of arguments for the signal function configured based on the strategy mode.

        Returns:
            Args: Arguments to be passed to the signal function.
        """
        if self.dynamic_mode:
            if self["signal_func_nb"] is None:
                if self.ls_mode:
                    return (
                        self.long_entries,
                        self.long_exits,
                        self.short_entries,
                        self.short_exits,
                        self.from_ago,
                        *((self.adjust_func_nb,) if self.staticized is None else ()),
                        self.adjust_args,
                    )
                if self.signals_mode:
                    return (
                        self.entries,
                        self.exits,
                        self.direction,
                        self.from_ago,
                        *((self.adjust_func_nb,) if self.staticized is None else ()),
                        self.adjust_args,
                    )
                if self.order_mode:
                    return (
                        self.size,
                        self.price,
                        self.size_type,
                        self.direction,
                        self.min_size,
                        self.max_size,
                        self.val_price,
                        self.from_ago,
                        *((self.adjust_func_nb,) if self.staticized is None else ()),
                        self.adjust_args,
                    )
        return self.post__signal_args

    @cachedproperty
    def post_segment_args(self) -> tp.Args:
        """Tuple of arguments for the post-segment processing function based on configuration.

        Returns:
            Args: Arguments to be used for post-segment processing.
        """
        if self.dynamic_mode:
            if self["post_segment_func_nb"] is None:
                if self.save_state or self.save_value or self.save_returns:
                    return (
                        self.save_state,
                        self.save_value,
                        self.save_returns,
                    )
        return self.post__post_segment_args

    @cachedproperty
    def chunked(self) -> tp.ChunkedOption:
        """Chunked option for the `chunked` argument.

        In dynamic mode, it specializes the chunked option for the arguments of the signal function
        with the appropriate argument taking specifications based on the active mode.

        Returns:
            ChunkedOption: Chunked option for the signal function.
        """
        if self.dynamic_mode:
            if self["signal_func_nb"] is None:
                if self.ls_mode:
                    return ch.specialize_chunked_option(
                        self.pre__chunked,
                        arg_take_spec=dict(
                            signal_args=ch.ArgsTaker(
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                *((None,) if self.staticized is None else ()),
                                ch.ArgsTaker(),
                            )
                        ),
                    )
                if self.signals_mode:
                    return ch.specialize_chunked_option(
                        self.pre__chunked,
                        arg_take_spec=dict(
                            signal_args=ch.ArgsTaker(
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                *((None,) if self.staticized is None else ()),
                                ch.ArgsTaker(),
                            )
                        ),
                    )
                if self.order_mode:
                    return ch.specialize_chunked_option(
                        self.pre__chunked,
                        arg_take_spec=dict(
                            signal_args=ch.ArgsTaker(
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                *((None,) if self.staticized is None else ()),
                                ch.ArgsTaker(),
                            )
                        ),
                    )
        return self.pre__chunked

    # ############# Result ############# #

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        if self.dynamic_mode:
            func = self.resolve_dynamic_target_func("from_signal_func_nb", self.staticized)
        elif not self.basic_mode:
            func = nb.from_signals_nb
        else:
            func = nb.from_basic_signals_nb
        func = jit_reg.resolve_option(func, self.jitted)
        func = ch_reg.resolve_option(func, self.chunked)
        return func

    @cachedproperty
    def target_arg_map(self) -> tp.Kwargs:
        target_arg_map = dict(BasePFPreparer.target_arg_map.func(self))
        if self.dynamic_mode:
            if self.staticized is not None:
                target_arg_map["signal_func_nb"] = None
                target_arg_map["post_signal_func_nb"] = None
                target_arg_map["post_segment_func_nb"] = None
        else:
            target_arg_map["group_lens"] = "cs_group_lens"
        return target_arg_map

    @cachedproperty
    def pf_args(self) -> tp.KwargsLike:
        pf_args = dict(BasePFPreparer.pf_args.func(self))
        pf_args["orders_cls"] = FSOrders
        return pf_args


FSPreparer.override_arg_config_doc(__pdoc__)

fof_arg_config = ReadonlyConfig(
    dict(
        segment_mask=dict(),
        call_pre_segment=dict(),
        call_post_segment=dict(),
        pre_sim_func_nb=dict(),
        pre_sim_args=dict(type="args", substitute_templates=True),
        post_sim_func_nb=dict(),
        post_sim_args=dict(type="args", substitute_templates=True),
        pre_group_func_nb=dict(),
        pre_group_args=dict(type="args", substitute_templates=True),
        post_group_func_nb=dict(),
        post_group_args=dict(type="args", substitute_templates=True),
        pre_row_func_nb=dict(),
        pre_row_args=dict(type="args", substitute_templates=True),
        post_row_func_nb=dict(),
        post_row_args=dict(type="args", substitute_templates=True),
        pre_segment_func_nb=dict(),
        pre_segment_args=dict(type="args", substitute_templates=True),
        post_segment_func_nb=dict(),
        post_segment_args=dict(type="args", substitute_templates=True),
        order_func_nb=dict(),
        order_args=dict(type="args", substitute_templates=True),
        flex_order_func_nb=dict(),
        flex_order_args=dict(type="args", substitute_templates=True),
        post_order_func_nb=dict(),
        post_order_args=dict(type="args", substitute_templates=True),
        ffill_val_price=dict(),
        update_value=dict(),
        fill_pos_info=dict(),
        track_value=dict(),
        row_wise=dict(),
        max_order_records=dict(),
        max_log_records=dict(),
    )
)
"""_"""

__pdoc__["fof_arg_config"] = f"""Readonly argument configuration for `FOFPreparer`.

```python
{fof_arg_config.prettify_doc()}
```
"""


@attach_arg_properties
@override_arg_config(fof_arg_config)
class FOFPreparer(BasePFPreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_order_func`.

    Configures order processing functions including simulation preparation and execution,
    staticizing user-defined functions while ensuring compatibility with flexible and row-wise modes.

    !!! info
        For default settings, see `from_order_func` in `vectorbtpro._settings.portfolio`.
    """

    _settings_path: tp.SettingsPath = "portfolio.from_order_func"

    # ############# Mode resolution ############# #

    @cachedproperty
    def pre__staticized(self) -> tp.StaticizedOption:
        """Pre-resolution value for the `staticized` argument.

        Returns:
            StaticizedOption: Staticized option.
        """
        staticized = self["staticized"]
        if isinstance(staticized, bool):
            if staticized:
                staticized = dict()
            else:
                staticized = None
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if "func" not in staticized:
                if not self.flexible and not self.row_wise:
                    staticized["func"] = nb.from_order_func_nb
                elif not self.flexible and self.row_wise:
                    staticized["func"] = nb.from_order_func_rw_nb
                elif self.flexible and not self.row_wise:
                    staticized["func"] = nb.from_flex_order_func_nb
                else:
                    staticized["func"] = nb.from_flex_order_func_rw_nb
        return staticized

    @cachedproperty
    def flexible(self) -> bool:
        """Indicates whether flexible mode is enabled.

        Returns:
            bool: True if flexible mode is enabled; otherwise, False.
        """
        return self["flex_order_func_nb"] is not None

    @cachedproperty
    def pre_sim_func_nb(self) -> tp.Callable:
        """Function for simulation preprocessing.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb` if not set.

        Returns:
            Callable: Pre-simulation function callable.
        """
        pre_sim_func_nb = self["pre_sim_func_nb"]
        if pre_sim_func_nb is None:
            pre_sim_func_nb = nb.no_pre_func_nb
        return pre_sim_func_nb

    @cachedproperty
    def post_sim_func_nb(self) -> tp.Callable:
        """Function for simulation postprocessing.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb` if not provided.

        Returns:
            Callable: Post-simulation function callable.
        """
        post_sim_func_nb = self["post_sim_func_nb"]
        if post_sim_func_nb is None:
            post_sim_func_nb = nb.no_post_func_nb
        return post_sim_func_nb

    @cachedproperty
    def pre_group_func_nb(self) -> tp.Callable:
        """Function for simulation group preprocessing.

        Ensures it is not used in row-wise simulations.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb` if not provided.

        Returns:
            Callable: Pre-group function callable.
        """
        pre_group_func_nb = self["pre_group_func_nb"]
        if self.row_wise and pre_group_func_nb is not None:
            raise ValueError("Cannot use pre_group_func_nb in a row-wise simulation")
        if pre_group_func_nb is None:
            pre_group_func_nb = nb.no_pre_func_nb
        return pre_group_func_nb

    @cachedproperty
    def post_group_func_nb(self) -> tp.Callable:
        """Function for simulation group postprocessing.

        Ensures it is not used in row-wise simulations.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb` if not provided.

        Returns:
            Callable: Post-group function callable.
        """
        post_group_func_nb = self["post_group_func_nb"]
        if self.row_wise and post_group_func_nb is not None:
            raise ValueError("Cannot use post_group_func_nb in a row-wise simulation")
        if post_group_func_nb is None:
            post_group_func_nb = nb.no_post_func_nb
        return post_group_func_nb

    @cachedproperty
    def pre_row_func_nb(self) -> tp.Callable:
        """Function for simulation row preprocessing.

        Ensures it is only used in row-wise simulations.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb` if not provided.

        Returns:
            Callable: Pre-row function callable.
        """
        pre_row_func_nb = self["pre_row_func_nb"]
        if not self.row_wise and pre_row_func_nb is not None:
            raise ValueError("Cannot use pre_row_func_nb in a column-wise simulation")
        if pre_row_func_nb is None:
            pre_row_func_nb = nb.no_pre_func_nb
        return pre_row_func_nb

    @cachedproperty
    def post_row_func_nb(self) -> tp.Callable:
        """Function for simulation row postprocessing.

        Ensures it is only used in row-wise simulations.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb` if not provided.

        Returns:
            Callable: Post-row function callable.
        """
        post_row_func_nb = self["post_row_func_nb"]
        if not self.row_wise and post_row_func_nb is not None:
            raise ValueError("Cannot use post_row_func_nb in a column-wise simulation")
        if post_row_func_nb is None:
            post_row_func_nb = nb.no_post_func_nb
        return post_row_func_nb

    @cachedproperty
    def pre_segment_func_nb(self) -> tp.Callable:
        """Function for simulation segment preprocessing.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb` if not provided.

        Returns:
            Callable: Pre-segment function callable.
        """
        pre_segment_func_nb = self["pre_segment_func_nb"]
        if pre_segment_func_nb is None:
            pre_segment_func_nb = nb.no_pre_func_nb
        return pre_segment_func_nb

    @cachedproperty
    def post_segment_func_nb(self) -> tp.Callable:
        """Function for simulation segment postprocessing.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb` if not provided.

        Returns:
            Callable: Post-segment function callable.
        """
        post_segment_func_nb = self["post_segment_func_nb"]
        if post_segment_func_nb is None:
            post_segment_func_nb = nb.no_post_func_nb
        return post_segment_func_nb

    @cachedproperty
    def order_func_nb(self) -> tp.Callable:
        """Function for processing orders.

        Expects one of `order_func_nb` or `flex_order_func_nb` to be provided based on the flexible mode.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_order_func_nb` if not set.

        Returns:
            Callable: Order function callable.
        """
        order_func_nb = self["order_func_nb"]
        if self.flexible and order_func_nb is not None:
            raise ValueError("Must provide either order_func_nb or flex_order_func_nb")
        if not self.flexible and order_func_nb is None:
            raise ValueError("Must provide either order_func_nb or flex_order_func_nb")
        if order_func_nb is None:
            order_func_nb = nb.no_order_func_nb
        return order_func_nb

    @cachedproperty
    def flex_order_func_nb(self) -> tp.Callable:
        """Flexible function for processing orders.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_flex_order_func_nb` if not set.

        Returns:
            Callable: Flexible order function callable.
        """
        flex_order_func_nb = self["flex_order_func_nb"]
        if flex_order_func_nb is None:
            flex_order_func_nb = nb.no_flex_order_func_nb
        return flex_order_func_nb

    @cachedproperty
    def post_order_func_nb(self) -> tp.Callable:
        """Function for postprocessing orders.

        Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb` if not provided.

        Returns:
            Callable: Post-order function callable.
        """
        post_order_func_nb = self["post_order_func_nb"]
        if post_order_func_nb is None:
            post_order_func_nb = nb.no_post_func_nb
        return post_order_func_nb

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        """Resolved `staticized` argument after applying adaptations to user-defined functions.

        Returns:
            StaticizedOption: Resolved staticized option.
        """
        staticized = self.pre__staticized
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if self["pre_sim_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["pre_sim_func_nb"], "pre_sim_func_nb")
            if self["post_sim_func_nb"] is not None:
                self.adapt_staticized_to_udf(
                    staticized, self["post_sim_func_nb"], "post_sim_func_nb"
                )
            if self["pre_group_func_nb"] is not None:
                self.adapt_staticized_to_udf(
                    staticized, self["pre_group_func_nb"], "pre_group_func_nb"
                )
            if self["post_group_func_nb"] is not None:
                self.adapt_staticized_to_udf(
                    staticized, self["post_group_func_nb"], "post_group_func_nb"
                )
            if self["pre_row_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["pre_row_func_nb"], "pre_row_func_nb")
            if self["post_row_func_nb"] is not None:
                self.adapt_staticized_to_udf(
                    staticized, self["post_row_func_nb"], "post_row_func_nb"
                )
            if self["pre_segment_func_nb"] is not None:
                self.adapt_staticized_to_udf(
                    staticized, self["pre_segment_func_nb"], "pre_segment_func_nb"
                )
            if self["post_segment_func_nb"] is not None:
                self.adapt_staticized_to_udf(
                    staticized, self["post_segment_func_nb"], "post_segment_func_nb"
                )
            if self["order_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["order_func_nb"], "order_func_nb")
            if self["flex_order_func_nb"] is not None:
                self.adapt_staticized_to_udf(
                    staticized, self["flex_order_func_nb"], "flex_order_func_nb"
                )
            if self["post_order_func_nb"] is not None:
                self.adapt_staticized_to_udf(
                    staticized, self["post_order_func_nb"], "post_order_func_nb"
                )
        return staticized

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre__call_seq(self) -> tp.Optional[tp.ArrayLike]:
        if self.auto_call_seq:
            raise ValueError(
                "CallSeqType.Auto must be implemented manually. Use sort_call_seq_nb in pre_segment_func_nb."
            )
        return self["call_seq"]

    @cachedproperty
    def pre__segment_mask(self) -> tp.ArrayLike:
        """Pre-broadcast value for the `segment_mask` argument.

        Returns:
            ArrayLike: Segment mask before broadcasting.
        """
        return self["segment_mask"]

    # ############# After broadcasting ############# #

    @cachedproperty
    def sim_start(self) -> tp.Optional[tp.ArrayLike]:
        sim_start = self["sim_start"]
        if sim_start is None:
            return None
        return BasePFPreparer.sim_start.func(self)

    @cachedproperty
    def sim_end(self) -> tp.Optional[tp.ArrayLike]:
        sim_end = self["sim_end"]
        if sim_end is None:
            return None
        return BasePFPreparer.sim_end.func(self)

    @cachedproperty
    def segment_mask(self) -> tp.ArrayLike:
        """Segment mask array indicating segment boundaries in the target data.

        If `FOFPreparer.pre__segment_mask` is an integer, a boolean mask with evenly spaced
        True values is generated based on the target shape and group lengths.
        Otherwise, the provided mask is broadcast to match the required shape.

        Returns:
            ArrayLike: Post-broadcasted segment mask.
        """
        segment_mask = self.pre__segment_mask
        if checks.is_int(segment_mask):
            if self.keep_inout_flex:
                _segment_mask = np.full((self.target_shape[0], 1), False)
            else:
                _segment_mask = np.full((self.target_shape[0], len(self.group_lens)), False)
            _segment_mask[0::segment_mask] = True
            segment_mask = _segment_mask
        else:
            segment_mask = broadcast(
                segment_mask,
                to_shape=(self.target_shape[0], len(self.group_lens)),
                to_pd=False,
                keep_flex=self.keep_inout_flex,
                reindex_kwargs=dict(fill_value=False),
                require_kwargs=self.broadcast_kwargs.get("require_kwargs", {}),
            )
        checks.assert_subdtype(segment_mask, np.bool_, arg_name="segment_mask")
        return segment_mask

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                segment_mask=self.segment_mask,
                call_pre_segment=self.call_pre_segment,
                call_post_segment=self.call_post_segment,
                pre_sim_func_nb=self.pre_sim_func_nb,
                pre_sim_args=self.pre__pre_sim_args,
                post_sim_func_nb=self.post_sim_func_nb,
                post_sim_args=self.pre__post_sim_args,
                pre_group_func_nb=self.pre_group_func_nb,
                pre_group_args=self.pre__pre_group_args,
                post_group_func_nb=self.post_group_func_nb,
                post_group_args=self.pre__post_group_args,
                pre_row_func_nb=self.pre_row_func_nb,
                pre_row_args=self.pre__pre_row_args,
                post_row_func_nb=self.post_row_func_nb,
                post_row_args=self.pre__post_row_args,
                pre_segment_func_nb=self.pre_segment_func_nb,
                pre_segment_args=self.pre__pre_segment_args,
                post_segment_func_nb=self.post_segment_func_nb,
                post_segment_args=self.pre__post_segment_args,
                order_func_nb=self.order_func_nb,
                order_args=self.pre__order_args,
                flex_order_func_nb=self.flex_order_func_nb,
                flex_order_args=self.pre__flex_order_args,
                post_order_func_nb=self.post_order_func_nb,
                post_order_args=self.pre__post_order_args,
                ffill_val_price=self.ffill_val_price,
                update_value=self.update_value,
                fill_pos_info=self.fill_pos_info,
                track_value=self.track_value,
                max_order_records=self.max_order_records,
                max_log_records=self.max_log_records,
            ),
            BasePFPreparer.template_context.func(self),
        )

    # ############# Result ############# #

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        if not self.row_wise and not self.flexible:
            func = self.resolve_dynamic_target_func("from_order_func_nb", self.staticized)
        elif not self.row_wise and self.flexible:
            func = self.resolve_dynamic_target_func("from_flex_order_func_nb", self.staticized)
        elif self.row_wise and not self.flexible:
            func = self.resolve_dynamic_target_func("from_order_func_rw_nb", self.staticized)
        else:
            func = self.resolve_dynamic_target_func("from_flex_order_func_rw_nb", self.staticized)
        func = jit_reg.resolve_option(func, self.jitted)
        func = ch_reg.resolve_option(func, self.chunked)
        return func

    @cachedproperty
    def target_arg_map(self) -> tp.Kwargs:
        target_arg_map = dict(BasePFPreparer.target_arg_map.func(self))
        if self.staticized is not None:
            target_arg_map["pre_sim_func_nb"] = None
            target_arg_map["post_sim_func_nb"] = None
            target_arg_map["pre_group_func_nb"] = None
            target_arg_map["post_group_func_nb"] = None
            target_arg_map["pre_row_func_nb"] = None
            target_arg_map["post_row_func_nb"] = None
            target_arg_map["pre_segment_func_nb"] = None
            target_arg_map["post_segment_func_nb"] = None
            target_arg_map["order_func_nb"] = None
            target_arg_map["flex_order_func_nb"] = None
            target_arg_map["post_order_func_nb"] = None
        return target_arg_map


fdof_arg_config = ReadonlyConfig(
    dict(
        val_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ValPriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        flexible=dict(),
    )
)
"""_"""

__pdoc__["fdof_arg_config"] = f"""Argument config for `FDOFPreparer`.

```python
{fdof_arg_config.prettify_doc()}
```
"""


@attach_arg_properties
@override_arg_config(fdof_arg_config)
@override_arg_config(order_arg_config)
class FDOFPreparer(FOFPreparer):
    """Class for preparing the portfolio using `vectorbtpro.portfolio.base.Portfolio.from_def_order_func`.

    !!! info
        For default settings, see `from_def_order_func` in `vectorbtpro._settings.portfolio`.
    """

    _settings_path: tp.SettingsPath = "portfolio.from_def_order_func"

    # ############# Mode resolution ############# #

    @cachedproperty
    def flexible(self) -> bool:
        return self["flexible"]

    @cachedproperty
    def pre_segment_func_nb(self) -> tp.Callable:
        """Pre-segment processing function.

        If not specified in the configuration, the default function is selected based on the `flexible` flag:

        * Returns `vectorbtpro.portfolio.nb.from_order_func.def_flex_pre_segment_func_nb` if `flexible` is True.
        * Returns `vectorbtpro.portfolio.nb.from_order_func.def_pre_segment_func_nb` otherwise.

        Returns:
            Callable: Function used to process pre-segment data.
        """
        pre_segment_func_nb = self["pre_segment_func_nb"]
        if pre_segment_func_nb is None:
            if self.flexible:
                pre_segment_func_nb = nb.def_flex_pre_segment_func_nb
            else:
                pre_segment_func_nb = nb.def_pre_segment_func_nb
        return pre_segment_func_nb

    @cachedproperty
    def order_func_nb(self) -> tp.Callable:
        """Order processing function.

        If `order_func_nb` is not provided in the configuration, the default
        `vectorbtpro.portfolio.nb.from_order_func.def_order_func_nb` is used.

        Returns:
            Callable: Function used for order processing.

        Raises:
            ValueError: If `order_func_nb` is provided when `flexible` is True.
        """
        order_func_nb = self["order_func_nb"]
        if self.flexible and order_func_nb is not None:
            raise ValueError("Argument order_func_nb cannot be provided when flexible=True")
        if order_func_nb is None:
            order_func_nb = nb.def_order_func_nb
        return order_func_nb

    @cachedproperty
    def flex_order_func_nb(self) -> tp.Callable:
        """Flexible order processing function.

        If `flex_order_func_nb` is not provided, the default
        `vectorbtpro.portfolio.nb.from_order_func.def_flex_order_func_nb` is used.

        Returns:
            Callable: Function used for processing flexible orders.

        Raises:
            ValueError: If `flex_order_func_nb` is provided when `flexible` is False.
        """
        flex_order_func_nb = self["flex_order_func_nb"]
        if not self.flexible and flex_order_func_nb is not None:
            raise ValueError("Argument flex_order_func_nb cannot be provided when flexible=False")
        if flex_order_func_nb is None:
            flex_order_func_nb = nb.def_flex_order_func_nb
        return flex_order_func_nb

    @cachedproperty
    def pre__chunked(self) -> tp.ChunkedOption:
        """Argument `chunked` before template substitution.

        Returns:
            ChunkedOption: Pre-broadcasted chunked option.
        """
        return self["chunked"]

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        staticized = FOFPreparer.staticized.func(self)
        if isinstance(staticized, dict):
            if "pre_segment_func_nb" not in staticized:
                self.adapt_staticized_to_udf(
                    staticized, self.pre_segment_func_nb, "pre_segment_func_nb"
                )
            if "order_func_nb" not in staticized:
                self.adapt_staticized_to_udf(staticized, self.order_func_nb, "order_func_nb")
            if "flex_order_func_nb" not in staticized:
                self.adapt_staticized_to_udf(
                    staticized, self.flex_order_func_nb, "flex_order_func_nb"
                )
        return staticized

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre__call_seq(self) -> tp.Optional[tp.ArrayLike]:
        return BasePFPreparer.pre__call_seq.func(self)

    # ############# After broadcasting ############# #

    @cachedproperty
    def auto_sim_start(self) -> tp.Optional[tp.ArrayLike]:
        return FOPreparer.auto_sim_start.func(self)

    @cachedproperty
    def auto_sim_end(self) -> tp.Optional[tp.ArrayLike]:
        return FOPreparer.auto_sim_end.func(self)

    # ############# Template substitution ############# #

    @cachedproperty
    def pre_segment_args(self) -> tp.Args:
        """Tuple of arguments for the pre-segment function used in template substitution.

        Returns:
            Args: Pre-segment function arguments.
        """
        return (
            self.val_price,
            self.price,
            self.size,
            self.size_type,
            self.direction,
            self.auto_call_seq,
        )

    @cachedproperty
    def any_order_args(self) -> tp.Args:
        """Either `order_args` or `flex_order_args`.

        Returns:
            Args: Order function arguments.
        """
        return (
            self.size,
            self.price,
            self.size_type,
            self.direction,
            self.fees,
            self.fixed_fees,
            self.slippage,
            self.min_size,
            self.max_size,
            self.size_granularity,
            self.leverage,
            self.leverage_mode,
            self.reject_prob,
            self.price_area_vio_mode,
            self.allow_partial,
            self.raise_reject,
            self.log,
        )

    @cachedproperty
    def order_args(self) -> tp.Args:
        """Tuple of arguments for the order function.

        If `flexible` is True, returns `FDOFPreparer.post__order_args`;
        otherwise, returns `FDOFPreparer.any_order_args`.

        Returns:
            Args: Order function arguments.
        """
        if self.flexible:
            return self.post__order_args
        return self.any_order_args

    @cachedproperty
    def flex_order_args(self) -> tp.Args:
        """Tuple of arguments for the flexible order function.

        If `flexible` is False, returns `FDOFPreparer.post__flex_order_args`;
        otherwise, returns `FDOFPreparer.any_order_args`.

        Returns:
            Args: Flexible order function arguments.
        """
        if not self.flexible:
            return self.post__flex_order_args
        return self.any_order_args

    @cachedproperty
    def chunked(self) -> tp.ChunkedOption:
        """Specialized chunked configuration.

        An argument taker specification is created for `pre_segment_args` and either `order_args`
        or `flex_order_args` based on the `flexible` flag, and applied to the `chunked` option.

        Returns:
            ChunkedOption: Specialized chunked configuration.
        """
        arg_take_spec = dict()
        arg_take_spec["pre_segment_args"] = ch.ArgsTaker(*[base_ch.flex_array_gl_slicer] * 5, None)
        if self.flexible:
            arg_take_spec["flex_order_args"] = ch.ArgsTaker(*[base_ch.flex_array_gl_slicer] * 17)
        else:
            arg_take_spec["order_args"] = ch.ArgsTaker(*[base_ch.flex_array_gl_slicer] * 17)
        return ch.specialize_chunked_option(self.pre__chunked, arg_take_spec=arg_take_spec)
