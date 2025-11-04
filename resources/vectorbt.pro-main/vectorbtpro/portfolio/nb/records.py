# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for processing portfolio records."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import add_nb, is_close_nb, is_close_or_less_nb, is_less_nb
from vectorbtpro.utils.template import Rep

invalid_size_msg = "Encountered an order with size 0 or less"
invalid_price_msg = "Encountered an order with price less than 0"


@register_jitted(cache=True)
def records_within_sim_range_nb(
    target_shape: tp.Shape,
    records: tp.RecordArray,
    col_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.RecordArray:
    """Return records within the specified simulation range.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        records (RecordArray): Array of records.
        col_arr (Array1d): Array of column indices corresponding to each element in `records`.
        idx_arr (Array1d): Array of row indices corresponding to each element in `records`.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        RecordArray: Array of records filtered to lie within the simulation range.
    """
    out = np.empty(len(records), dtype=records.dtype)
    k = 0

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for r in range(len(records)):
        _sim_start = sim_start_[col_arr[r]]
        _sim_end = sim_end_[col_arr[r]]
        if _sim_start >= _sim_end:
            continue
        if _sim_start <= idx_arr[r] < _sim_end:
            out[k] = records[r]
            k += 1
    return out[:k]


@register_jitted(cache=True)
def apply_weights_to_orders_nb(
    order_records: tp.RecordArray,
    col_arr: tp.Array1d,
    weights: tp.Array1d,
) -> tp.RecordArray:
    """Apply weights to order records by scaling sizes and fees.

    Args:
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        col_arr (Array1d): Array of column indices.
        weights (Array1d): Array of weights used to scale order sizes and fees per column.

    Returns:
        RecordArray: New array of order records after applying weights and filtering orders with zero size.
    """
    order_records = order_records.copy()
    out = np.empty(len(order_records), dtype=order_records.dtype)
    k = 0

    for r in range(len(order_records)):
        order_record = order_records[r]
        col = col_arr[r]
        if not np.isnan(weights[col]):
            order_record["size"] = weights[col] * order_record["size"]
            order_record["fees"] = weights[col] * order_record["fees"]
            if order_record["size"] != 0:
                out[k] = order_record
                k += 1
        else:
            out[k] = order_record
            k += 1
    return out[:k]


@register_jitted(cache=True)
def weighted_price_reduce_meta_nb(
    idxs: tp.Array1d,
    col: int,
    size_arr: tp.Array1d,
    price_arr: tp.Array1d,
) -> float:
    """Calculate the size-weighted average price.

    Args:
        idxs (Array1d): Array of indices to consider for the computation.
        col (int): Column index (unused in computation).
        size_arr (Array1d): Array of sizes corresponding to prices.
        price_arr (Array1d): Array of prices corresponding to sizes.

    Returns:
        float: Size-weighted average price, or NaN if no valid data is available.
    """
    if len(idxs) == 0:
        return np.nan
    size_price_sum = 0.0
    size_sum = 0.0
    for i in range(len(idxs)):
        j = idxs[i]
        if not np.isnan(size_arr[j]) and not np.isnan(price_arr[j]):
            size_price_sum += size_arr[j] * price_arr[j]
            size_sum += size_arr[j]
    if size_sum == 0:
        return np.nan
    return size_price_sum / size_sum


@register_jitted(cache=True)
def fill_trade_record_nb(
    new_records: tp.RecordArray,
    r: int,
    col: int,
    size: float,
    entry_order_id: int,
    entry_idx: int,
    entry_price: float,
    entry_fees: float,
    exit_order_id: int,
    exit_idx: int,
    exit_price: float,
    exit_fees: float,
    direction: int,
    status: int,
    parent_id: int,
) -> None:
    """Fill a trade record in the provided records array.

    Args:
        new_records (RecordArray): Array of records to be filled.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        r (int): Index at which to fill the trade record.
        col (int): Column index associated with the trade.
        size (float): Trade size.
        entry_order_id (int): ID of the entry order.
        entry_idx (int): Row index of the entry order.
        entry_price (float): Size-weighted average entry price of the trade.
        entry_fees (float): Fees associated with the entry order.
        exit_order_id (int): ID of the exit order.
        exit_idx (int): Row index of the exit order.
        exit_price (float): Size-weighted average exit price of the trade.
        exit_fees (float): Fees associated with the exit order.
        direction (int): Direction of the trade (buy/sell).

            See `vectorbtpro.portfolio.enums.TradeDirection`.
        status (int): Trade status indicating whether the position is closed.

            See `vectorbtpro.portfolio.enums.TradeStatus`.
        parent_id (int): Identifier linking trades to a parent trade.

    Returns:
        None: Function modifies `new_records` in place.
    """
    # Calculate PnL and return
    pnl, ret = get_trade_stats_nb(size, entry_price, entry_fees, exit_price, exit_fees, direction)

    # Save trade
    new_records["id"][r] = r
    new_records["col"][r] = col
    new_records["size"][r] = size
    new_records["entry_order_id"][r] = entry_order_id
    new_records["entry_idx"][r] = entry_idx
    new_records["entry_price"][r] = entry_price
    new_records["entry_fees"][r] = entry_fees
    new_records["exit_order_id"][r] = exit_order_id
    new_records["exit_idx"][r] = exit_idx
    new_records["exit_price"][r] = exit_price
    new_records["exit_fees"][r] = exit_fees
    new_records["pnl"][r] = pnl
    new_records["return"][r] = ret
    new_records["direction"][r] = direction
    new_records["status"][r] = status
    new_records["parent_id"][r] = parent_id
    new_records["parent_id"][r] = parent_id


@register_jitted(cache=True)
def fill_entry_trades_in_position_nb(
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    col: int,
    sim_start: int,
    sim_end: int,
    first_c: int,
    last_c: int,
    init_price: float,
    first_entry_size: float,
    first_entry_fees: float,
    exit_idx: int,
    exit_size_sum: float,
    exit_gross_sum: float,
    exit_fees_sum: float,
    direction: int,
    status: int,
    parent_id: int,
    new_records: tp.RecordArray,
    r: int,
) -> int:
    """Fill entry trades within a single position and return the next trade ID.

    Iterates over order records corresponding to a position and computes trade details such as
    size-weighted exit price and fees.

    Args:
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        col (int): Column index for the orders.
        sim_start (int): Simulation start index.
        sim_end (int): Simulation end index.
        first_c (int): Index of the first entry order.
        last_c (int): Index of the last entry order.
        init_price (float): Initial price for the position.
        first_entry_size (float): Size of the first entry order.
        first_entry_fees (float): Fees of the first entry order.
        exit_idx (int): Row index of the exit order.
        exit_size_sum (float): Total size of exit orders.
        exit_gross_sum (float): Gross sum used to compute the weighted exit price.
        exit_fees_sum (float): Total exit fees for proportional allocation.
        direction (int): Trade direction.

            See `vectorbtpro.portfolio.enums.TradeDirection`.
        status (int): Trade status indicating whether the position is closed.

            See `vectorbtpro.portfolio.enums.TradeStatus`.
        parent_id (int): Identifier linking trades to a parent trade.
        new_records (RecordArray): Array of records to be filled.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        r (int): Current trade index used as the starting ID.

    Returns:
        int: Next trade ID after filling the entry trades.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens

    # Iterate over orders located within a single position
    for c in range(first_c, last_c + 1):
        if c == -1:
            entry_order_id = -1
            entry_idx = -1
            entry_size = first_entry_size
            entry_price = init_price
            entry_fees = first_entry_fees
        else:
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < sim_start or order_record["idx"] >= sim_end:
                continue
            entry_order_id = order_record["id"]
            entry_idx = order_record["idx"]
            entry_price = order_record["price"]
            order_side = order_record["side"]

            # Ignore exit orders
            if (direction == TradeDirection.Long and order_side == OrderSide.Sell) or (
                direction == TradeDirection.Short and order_side == OrderSide.Buy
            ):
                continue

            if c == first_c:
                entry_size = first_entry_size
                entry_fees = first_entry_fees
            else:
                entry_size = order_record["size"]
                entry_fees = order_record["fees"]

        # Take a size-weighted average of exit price
        exit_price = exit_gross_sum / exit_size_sum

        # Take a fraction of exit fees
        size_fraction = entry_size / exit_size_sum
        exit_fees = size_fraction * exit_fees_sum

        # Fill the record
        if status == TradeStatus.Closed:
            exit_order_record = order_records[col_idxs[col_start_idxs[col] + last_c]]
            if exit_order_record["idx"] < sim_start or exit_order_record["idx"] >= sim_end:
                exit_order_id = -1
            else:
                exit_order_id = exit_order_record["id"]
        else:
            exit_order_id = -1
        fill_trade_record_nb(
            new_records,
            r,
            col,
            entry_size,
            entry_order_id,
            entry_idx,
            entry_price,
            entry_fees,
            exit_order_id,
            exit_idx,
            exit_price,
            exit_fees,
            direction,
            status,
            parent_id,
        )
        r += 1

    return r


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        close=base_ch.FlexArraySlicer(axis=1),
        col_map=base_ch.GroupMapSlicer(),
        init_position=base_ch.FlexArraySlicer(),
        init_price=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_entry_trades_nb(
    order_records: tp.RecordArray,
    close: tp.FlexArray2dLike,
    col_map: tp.GroupMap,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.RecordArray:
    """Fill entry trade records by aggregating order records.

    Entry trade records are defined as buy orders in a long position and sell orders in a short position.

    Args:
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        close (FlexArray2dLike): Close price.

            Provided as a scalar, or per row, column, or element.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.

            Used to calculate the entry price when an initial position exists.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        RecordArray: Array of entry trade records aggregated from order records.

            Has the `vectorbtpro.portfolio.enums.trade_dt` dtype.

    !!! tip
        This function is parallelizable.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> close = order_price = np.array([
        ...     [1, 6],
        ...     [2, 5],
        ...     [3, 4],
        ...     [4, 3],
        ...     [5, 2],
        ...     [6, 1]
        ... ])
        >>> size = np.array([
        ...     [1, -1],
        ...     [0.1, -0.1],
        ...     [-1, 1],
        ...     [-0.1, 0.1],
        ...     [1, -1],
        ...     [-2, 2]
        ... ])
        >>> target_shape = close.shape
        >>> group_lens = np.full(target_shape[1], 1)
        >>> init_cash = np.full(target_shape[1], 100)

        >>> sim_out = vbt.pf_nb.from_orders_nb(
        ...     target_shape,
        ...     group_lens,
        ...     init_cash=init_cash,
        ...     size=size,
        ...     price=close,
        ...     fees=np.asarray([[0.01]]),
        ...     slippage=np.asarray([[0.01]])
        ... )

        >>> col_map = vbt.rec_nb.col_map_nb(sim_out.order_records['col'], target_shape[1])
        >>> entry_trade_records = vbt.pf_nb.get_entry_trades_nb(sim_out.order_records, close, col_map)
        >>> pd.DataFrame.from_records(entry_trade_records)
           id  col  size  entry_order_id  entry_idx  entry_price  entry_fees  \\
        0   0    0   1.0               0          0         1.01     0.01010
        1   1    0   0.1               1          1         2.02     0.00202
        2   2    0   1.0               4          4         5.05     0.05050
        3   3    0   1.0               5          5         5.94     0.05940
        4   0    1   1.0               0          0         5.94     0.05940
        5   1    1   0.1               1          1         4.95     0.00495
        6   2    1   1.0               4          4         1.98     0.01980
        7   3    1   1.0               5          5         1.01     0.01010

           exit_order_id  exit_idx  exit_price  exit_fees       pnl    return  \\
        0              3         3    3.060000   0.030600  2.009300  1.989406
        1              3         3    3.060000   0.003060  0.098920  0.489703
        2              5         5    5.940000   0.059400  0.780100  0.154475
        3             -1         5    6.000000   0.000000 -0.119400 -0.020101
        4              3         3    3.948182   0.039482  1.892936  0.318676
        5              3         3    3.948182   0.003948  0.091284  0.184411
        6              5         5    1.010000   0.010100  0.940100  0.474798
        7             -1         5    1.000000   0.000000 -0.020100 -0.019901

           direction  status  parent_id
        0          0       1          0
        1          0       1          0
        2          0       1          1
        3          1       0          2
        4          1       1          0
        5          1       1          0
        6          1       1          1
        7          0       0          2
        ```
    """
    close_ = to_2d_array_nb(np.asarray(close))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    max_records = np.max(col_lens) + 1
    new_records = np.empty((max_records, len(col_lens)), dtype=trade_dt)
    counts = np.full(len(col_lens), 0, dtype=int_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(close_.shape[0], col_lens.shape[0]),
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))

        if _init_position != 0:
            # Prepare initial position
            first_c = -1
            in_position = True
            parent_id = 0
            if _init_position >= 0:
                direction = TradeDirection.Long
            else:
                direction = TradeDirection.Short
            entry_size_sum = abs(_init_position)
            entry_gross_sum = abs(_init_position) * _init_price
            entry_fees_sum = 0.0
            exit_size_sum = 0.0
            exit_gross_sum = 0.0
            exit_fees_sum = 0.0
            first_entry_size = _init_position
            first_entry_fees = 0.0
        else:
            in_position = False
            parent_id = -1

        col_len = col_lens[col]
        if col_len == 0 and not in_position:
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            order_idx = order_record["idx"]
            order_size = order_record["size"]
            order_price = order_record["price"]
            order_fees = order_record["fees"]
            order_side = order_record["side"]

            if order_size <= 0.0:
                raise ValueError(invalid_size_msg)
            if order_price < 0.0:
                raise ValueError(invalid_price_msg)

            if not in_position:
                # New position opened
                first_c = c
                in_position = True
                parent_id += 1
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                entry_size_sum = 0.0
                entry_gross_sum = 0.0
                entry_fees_sum = 0.0
                exit_size_sum = 0.0
                exit_gross_sum = 0.0
                exit_fees_sum = 0.0
                first_entry_size = order_size
                first_entry_fees = order_fees

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) or (
                direction == TradeDirection.Short and order_side == OrderSide.Sell
            ):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                entry_fees_sum += order_fees

            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) or (
                direction == TradeDirection.Short and order_side == OrderSide.Buy
            ):
                if is_close_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position closed
                    last_c = c
                    in_position = False
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += order_size * order_price
                    exit_fees_sum += order_fees

                    # Fill trade records
                    counts[col] = fill_entry_trades_in_position_nb(
                        order_records,
                        col_map,
                        col,
                        _sim_start,
                        _sim_end,
                        first_c,
                        last_c,
                        _init_price,
                        first_entry_size,
                        first_entry_fees,
                        order_idx,
                        exit_size_sum,
                        exit_gross_sum,
                        exit_fees_sum,
                        direction,
                        TradeStatus.Closed,
                        parent_id,
                        new_records[:, col],
                        counts[col],
                    )
                elif is_less_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position decreased
                    exit_size_sum += order_size
                    exit_gross_sum += order_size * order_price
                    exit_fees_sum += order_fees
                else:
                    # Position closed
                    last_c = c
                    remaining_size = add_nb(entry_size_sum, -exit_size_sum)
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += remaining_size * order_price
                    exit_fees_sum += remaining_size / order_size * order_fees

                    # Fill trade records
                    counts[col] = fill_entry_trades_in_position_nb(
                        order_records,
                        col_map,
                        col,
                        _sim_start,
                        _sim_end,
                        first_c,
                        last_c,
                        _init_price,
                        first_entry_size,
                        first_entry_fees,
                        order_idx,
                        exit_size_sum,
                        exit_gross_sum,
                        exit_fees_sum,
                        direction,
                        TradeStatus.Closed,
                        parent_id,
                        new_records[:, col],
                        counts[col],
                    )

                    # New position opened
                    first_c = c
                    parent_id += 1
                    if order_side == OrderSide.Buy:
                        direction = TradeDirection.Long
                    else:
                        direction = TradeDirection.Short
                    entry_size_sum = add_nb(order_size, -remaining_size)
                    entry_gross_sum = entry_size_sum * order_price
                    entry_fees_sum = entry_size_sum / order_size * order_fees
                    first_entry_size = entry_size_sum
                    first_entry_fees = entry_fees_sum
                    exit_size_sum = 0.0
                    exit_gross_sum = 0.0
                    exit_fees_sum = 0.0

        if in_position and is_less_nb(exit_size_sum, entry_size_sum):
            # Position hasn't been closed
            last_c = col_len - 1
            remaining_size = add_nb(entry_size_sum, -exit_size_sum)
            exit_size_sum = entry_size_sum
            last_close = flex_select_nb(close_, _sim_end - 1, col)
            if np.isnan(last_close):
                for ri in range(_sim_end - 1, -1, -1):
                    _close = flex_select_nb(close_, ri, col)
                    if not np.isnan(_close):
                        last_close = _close
                        break
            exit_gross_sum += remaining_size * last_close
            exit_idx = _sim_end - 1

            # Fill trade records
            counts[col] = fill_entry_trades_in_position_nb(
                order_records,
                col_map,
                col,
                _sim_start,
                _sim_end,
                first_c,
                last_c,
                _init_price,
                first_entry_size,
                first_entry_fees,
                exit_idx,
                exit_size_sum,
                exit_gross_sum,
                exit_fees_sum,
                direction,
                TradeStatus.Open,
                parent_id,
                new_records[:, col],
                counts[col],
            )

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        close=base_ch.FlexArraySlicer(axis=1),
        col_map=base_ch.GroupMapSlicer(),
        init_position=base_ch.FlexArraySlicer(),
        init_price=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_exit_trades_nb(
    order_records: tp.RecordArray,
    close: tp.FlexArray2dLike,
    col_map: tp.GroupMap,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.RecordArray:
    """Fill exit trade records by aggregating order records.

    Exit trade records correspond to sell orders in a long position and buy orders in a short position.

    Args:
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        close (FlexArray2dLike): Close price.

            Provided as a scalar, or per row, column, or element.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        RecordArray: Array of aggregated exit trade records.

            Has the `vectorbtpro.portfolio.enums.trade_dt` dtype.

    !!! tip
        This function is parallelizable.

    Examples:
        Building upon the example in `get_exit_trades_nb`:

        ```pycon
        >>> exit_trade_records = vbt.pf_nb.get_exit_trades_nb(sim_out.order_records, close, col_map)
        >>> pd.DataFrame.from_records(exit_trade_records)
           id  col  size  entry_order_id  entry_idx  entry_price  entry_fees  \\
        0   0    0   1.0               0          0     1.101818    0.011018
        1   1    0   0.1               0          0     1.101818    0.001102
        2   2    0   1.0               4          4     5.050000    0.050500
        3   3    0   1.0               5          5     5.940000    0.059400
        4   0    1   1.0               0          0     5.850000    0.058500
        5   1    1   0.1               0          0     5.850000    0.005850
        6   2    1   1.0               4          4     1.980000    0.019800
        7   3    1   1.0               5          5     1.010000    0.010100

           exit_order_id  exit_idx  exit_price  exit_fees       pnl    return  \\
        0              2         2        2.97    0.02970  1.827464  1.658589
        1              3         3        3.96    0.00396  0.280756  2.548119
        2              5         5        5.94    0.05940  0.780100  0.154475
        3             -1         5        6.00    0.00000 -0.119400 -0.020101
        4              2         2        4.04    0.04040  1.711100  0.292496
        5              3         3        3.03    0.00303  0.273120  0.466872
        6              5         5        1.01    0.01010  0.940100  0.474798
        7             -1         5        1.00    0.00000 -0.020100 -0.019901

           direction  status  parent_id
        0          0       1          0
        1          0       1          0
        2          0       1          1
        3          1       0          2
        4          1       1          0
        5          1       1          0
        6          1       1          1
        7          0       0          2
        ```
    """
    close_ = to_2d_array_nb(np.asarray(close))
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    max_records = np.max(col_lens) + 1
    new_records = np.empty((max_records, len(col_lens)), dtype=trade_dt)
    counts = np.full(len(col_lens), 0, dtype=int_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(close_.shape[0], col_lens.shape[0]),
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))

        if _init_position != 0:
            # Prepare initial position
            in_position = True
            parent_id = 0
            entry_order_id = -1
            entry_idx = -1
            if _init_position >= 0:
                direction = TradeDirection.Long
            else:
                direction = TradeDirection.Short
            entry_size_sum = abs(_init_position)
            entry_gross_sum = abs(_init_position) * _init_price
            entry_fees_sum = 0.0
        else:
            in_position = False
            parent_id = -1

        col_len = col_lens[col]
        if col_len == 0 and not in_position:
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            order_idx = order_record["idx"]
            order_id = order_record["id"]
            order_size = order_record["size"]
            order_price = order_record["price"]
            order_fees = order_record["fees"]
            order_side = order_record["side"]

            if order_size <= 0.0:
                raise ValueError(invalid_size_msg)
            if order_price < 0.0:
                raise ValueError(invalid_price_msg)

            if not in_position:
                # Trade opened
                in_position = True
                entry_order_id = order_id
                entry_idx = order_idx
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                parent_id += 1
                entry_size_sum = 0.0
                entry_gross_sum = 0.0
                entry_fees_sum = 0.0

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) or (
                direction == TradeDirection.Short and order_side == OrderSide.Sell
            ):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                entry_fees_sum += order_fees

            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) or (
                direction == TradeDirection.Short and order_side == OrderSide.Buy
            ):
                if is_close_or_less_nb(order_size, entry_size_sum):
                    # Trade closed
                    if is_close_nb(order_size, entry_size_sum):
                        exit_size = entry_size_sum
                    else:
                        exit_size = order_size
                    exit_price = order_price
                    exit_fees = order_fees
                    exit_order_id = order_id
                    exit_idx = order_idx

                    # Take a size-weighted average of entry price
                    entry_price = entry_gross_sum / entry_size_sum

                    # Take a fraction of entry fees
                    size_fraction = exit_size / entry_size_sum
                    entry_fees = size_fraction * entry_fees_sum

                    fill_trade_record_nb(
                        new_records[:, col],
                        counts[col],
                        col,
                        exit_size,
                        entry_order_id,
                        entry_idx,
                        entry_price,
                        entry_fees,
                        exit_order_id,
                        exit_idx,
                        exit_price,
                        exit_fees,
                        direction,
                        TradeStatus.Closed,
                        parent_id,
                    )
                    counts[col] += 1

                    if is_close_nb(order_size, entry_size_sum):
                        # Position closed
                        entry_order_id = -1
                        entry_idx = -1
                        direction = -1
                        in_position = False
                    else:
                        # Position decreased, previous orders have now less impact
                        size_fraction = (entry_size_sum - order_size) / entry_size_sum
                        entry_size_sum *= size_fraction
                        entry_gross_sum *= size_fraction
                        entry_fees_sum *= size_fraction
                else:
                    # Trade reversed
                    # Close current trade
                    cl_exit_size = entry_size_sum
                    cl_exit_price = order_price
                    cl_exit_fees = cl_exit_size / order_size * order_fees
                    cl_exit_order_id = order_id
                    cl_exit_idx = order_idx

                    # Take a size-weighted average of entry price
                    entry_price = entry_gross_sum / entry_size_sum

                    # Take a fraction of entry fees
                    size_fraction = cl_exit_size / entry_size_sum
                    entry_fees = size_fraction * entry_fees_sum

                    fill_trade_record_nb(
                        new_records[:, col],
                        counts[col],
                        col,
                        cl_exit_size,
                        entry_order_id,
                        entry_idx,
                        entry_price,
                        entry_fees,
                        cl_exit_order_id,
                        cl_exit_idx,
                        cl_exit_price,
                        cl_exit_fees,
                        direction,
                        TradeStatus.Closed,
                        parent_id,
                    )
                    counts[col] += 1

                    # Open a new trade
                    entry_size_sum = order_size - cl_exit_size
                    entry_gross_sum = entry_size_sum * order_price
                    entry_fees_sum = order_fees - cl_exit_fees
                    entry_order_id = order_id
                    entry_idx = order_idx
                    if direction == TradeDirection.Long:
                        direction = TradeDirection.Short
                    else:
                        direction = TradeDirection.Long
                    parent_id += 1

        if in_position and is_less_nb(-entry_size_sum, 0):
            # Trade hasn't been closed
            exit_size = entry_size_sum
            last_close = flex_select_nb(close_, _sim_end - 1, col)
            if np.isnan(last_close):
                for ri in range(_sim_end - 1, -1, -1):
                    _close = flex_select_nb(close_, ri, col)
                    if not np.isnan(_close):
                        last_close = _close
                        break
            exit_price = last_close
            exit_fees = 0.0
            exit_order_id = -1
            exit_idx = _sim_end - 1

            # Take a size-weighted average of entry price
            entry_price = entry_gross_sum / entry_size_sum

            # Take a fraction of entry fees
            size_fraction = exit_size / entry_size_sum
            entry_fees = size_fraction * entry_fees_sum

            fill_trade_record_nb(
                new_records[:, col],
                counts[col],
                col,
                exit_size,
                entry_order_id,
                entry_idx,
                entry_price,
                entry_fees,
                exit_order_id,
                exit_idx,
                exit_price,
                exit_fees,
                direction,
                TradeStatus.Open,
                parent_id,
            )
            counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_jitted(cache=True)
def fill_position_record_nb(
    new_records: tp.RecordArray, r: int, trade_records: tp.RecordArray
) -> None:
    """Fill a position record by aggregating trade records.

    Aggregates multiple trade records into a consolidated position record by computing the total
    size, weighted average entry and exit prices, fees, profit, and return. The aggregated record
    is stored in `new_records` at the specified index.

    Args:
        new_records (RecordArray): Array of records to be filled.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        r (int): Index where the new position record will be inserted.
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.

    Returns:
        None: Function modifies `new_records` in place.
    """
    # Aggregate trades
    col = trade_records["col"][0]
    size = np.sum(trade_records["size"])
    entry_order_id = trade_records["entry_order_id"][0]
    entry_idx = trade_records["entry_idx"][0]
    entry_price = np.sum(trade_records["size"] * trade_records["entry_price"]) / size
    entry_fees = np.sum(trade_records["entry_fees"])
    exit_order_id = trade_records["exit_order_id"][-1]
    exit_idx = trade_records["exit_idx"][-1]
    exit_price = np.sum(trade_records["size"] * trade_records["exit_price"]) / size
    exit_fees = np.sum(trade_records["exit_fees"])
    direction = trade_records["direction"][-1]
    status = trade_records["status"][-1]
    pnl, ret = get_trade_stats_nb(size, entry_price, entry_fees, exit_price, exit_fees, direction)

    # Save position
    new_records["id"][r] = r
    new_records["col"][r] = col
    new_records["size"][r] = size
    new_records["entry_order_id"][r] = entry_order_id
    new_records["entry_idx"][r] = entry_idx
    new_records["entry_price"][r] = entry_price
    new_records["entry_fees"][r] = entry_fees
    new_records["exit_order_id"][r] = exit_order_id
    new_records["exit_idx"][r] = exit_idx
    new_records["exit_price"][r] = exit_price
    new_records["exit_fees"][r] = exit_fees
    new_records["pnl"][r] = pnl
    new_records["return"][r] = ret
    new_records["direction"][r] = direction
    new_records["status"][r] = status
    new_records["parent_id"][r] = r


@register_jitted(cache=True)
def copy_trade_record_nb(new_records: tp.RecordArray, r: int, trade_record: tp.Record) -> None:
    """Copy a trade record.

    Copies a single trade record into the provided `new_records` array at the specified index.

    Args:
        new_records (RecordArray): Array of records to be filled.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        r (int): Index in `new_records` where the trade record will be inserted.
        trade_record (Record): Trade record to copy.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.

    Returns:
        None: Function modifies `new_records` in place.
    """
    new_records["id"][r] = r
    new_records["col"][r] = trade_record["col"]
    new_records["size"][r] = trade_record["size"]
    new_records["entry_order_id"][r] = trade_record["entry_order_id"]
    new_records["entry_idx"][r] = trade_record["entry_idx"]
    new_records["entry_price"][r] = trade_record["entry_price"]
    new_records["entry_fees"][r] = trade_record["entry_fees"]
    new_records["exit_order_id"][r] = trade_record["exit_order_id"]
    new_records["exit_idx"][r] = trade_record["exit_idx"]
    new_records["exit_price"][r] = trade_record["exit_price"]
    new_records["exit_fees"][r] = trade_record["exit_fees"]
    new_records["pnl"][r] = trade_record["pnl"]
    new_records["return"][r] = trade_record["return"]
    new_records["direction"][r] = trade_record["direction"]
    new_records["status"][r] = trade_record["status"]
    new_records["parent_id"][r] = r


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_positions_nb(trade_records: tp.RecordArray, col_map: tp.GroupMap) -> tp.RecordArray:
    """Fill position records by aggregating trade records.

    Aggregates trade records into position records by grouping them using the provided column map.
    Trades—whether entry trades, exit trades, or positions—are processed uniformly to calculate
    aggregated metrics such as size, weighted entry and exit prices, fees, profit, and return.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        col_map (GroupMap): Tuple of indices and lengths for each column.

    Returns:
        RecordArray: Array of aggregated position records after repartitioning based on column counts.

            Has the `vectorbtpro.portfolio.enums.trade_dt` dtype.

    !!! tip
        This function is parallelizable.

    Examples:
        Building upon the example in `get_exit_trades_nb`:

        ```pycon
        >>> col_map = vbt.rec_nb.col_map_nb(exit_trade_records['col'], target_shape[1])
        >>> position_records = vbt.pf_nb.get_positions_nb(exit_trade_records, col_map)
        >>> pd.DataFrame.from_records(position_records)
           id  col  size  entry_order_id  entry_idx  entry_price  entry_fees  \\
        0   0    0   1.1               0          0     1.101818     0.01212
        1   1    0   1.0               4          4     5.050000     0.05050
        2   2    0   1.0               5          5     5.940000     0.05940
        3   0    1   1.1               0          0     5.850000     0.06435
        4   1    1   1.0               4          4     1.980000     0.01980
        5   2    1   1.0               5          5     1.010000     0.01010

           exit_order_id  exit_idx  exit_price  exit_fees      pnl    return  \\
        0              3         3    3.060000    0.03366  2.10822  1.739455
        1              5         5    5.940000    0.05940  0.78010  0.154475
        2             -1         5    6.000000    0.00000 -0.11940 -0.020101
        3              3         3    3.948182    0.04343  1.98422  0.308348
        4              5         5    1.010000    0.01010  0.94010  0.474798
        5             -1         5    1.000000    0.00000 -0.02010 -0.019901

           direction  status  parent_id
        0          0       1          0
        1          0       1          1
        2          1       0          2
        3          1       1          0
        4          1       1          1
        5          0       0          2
        ```
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    new_records = np.empty((np.max(col_lens), len(col_lens)), dtype=trade_dt)
    counts = np.full(len(col_lens), 0, dtype=int_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        last_position_id = -1
        from_trade_r = -1

        for c in range(col_len):
            trade_r = col_idxs[col_start_idxs[col] + c]
            record = trade_records[trade_r]

            if record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = record["id"]

            parent_id = record["parent_id"]

            if parent_id != last_position_id:
                if last_position_id != -1:
                    if trade_r - from_trade_r > 1:
                        fill_position_record_nb(
                            new_records[:, col], counts[col], trade_records[from_trade_r:trade_r]
                        )
                    else:
                        # Speed up
                        copy_trade_record_nb(
                            new_records[:, col], counts[col], trade_records[from_trade_r]
                        )
                    counts[col] += 1
                from_trade_r = trade_r
                last_position_id = parent_id

        if trade_r - from_trade_r > 0:
            fill_position_record_nb(
                new_records[:, col], counts[col], trade_records[from_trade_r : trade_r + 1]
            )
        else:
            # Speed up
            copy_trade_record_nb(new_records[:, col], counts[col], trade_records[from_trade_r])
        counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        close=ch.ArraySlicer(axis=1),
        col_map=base_ch.GroupMapSlicer(),
        init_position=base_ch.FlexArraySlicer(),
        init_price=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_long_view_orders_nb(
    order_records: tp.RecordArray,
    close: tp.Array2d,
    col_map: tp.GroupMap,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.RecordArray:
    """Return a view of order records corresponding to long positions only.

    Args:
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        close (Array2d): Array of close prices.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        RecordArray: Filtered order records containing only long position entries.

    !!! tip
        This function is parallelizable.
    """
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    order_records = order_records.copy()
    out = np.empty(order_records.shape, dtype=order_records.dtype)
    r = 0

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(close.shape[0], col_lens.shape[0]),
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))

        if _init_position != 0:
            # Prepare initial position
            in_position = True
            if _init_position >= 0:
                direction = TradeDirection.Long
            else:
                direction = TradeDirection.Short
            entry_size_sum = abs(_init_position)
            entry_gross_sum = abs(_init_position) * _init_price
            exit_size_sum = 0.0
            exit_gross_sum = 0.0
        else:
            in_position = False

        col_len = col_lens[col]
        if col_len == 0 and not in_position:
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            order_size = order_record["size"]
            order_price = order_record["price"]
            order_side = order_record["side"]
            order_fees = order_record["fees"]

            if order_size <= 0.0:
                raise ValueError(invalid_size_msg)
            if order_price < 0.0:
                raise ValueError(invalid_price_msg)

            if not in_position:
                # New position opened
                in_position = True
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                entry_size_sum = 0.0
                entry_gross_sum = 0.0
                exit_size_sum = 0.0
                exit_gross_sum = 0.0

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) or (
                direction == TradeDirection.Short and order_side == OrderSide.Sell
            ):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                if direction == TradeDirection.Long:
                    out[r] = order_record
                    r += 1
            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) or (
                direction == TradeDirection.Short and order_side == OrderSide.Buy
            ):
                if is_close_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position closed
                    in_position = False
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += order_size * order_price
                    if direction == TradeDirection.Long:
                        out[r] = order_record
                        r += 1
                elif is_less_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position decreased
                    exit_size_sum += order_size
                    exit_gross_sum += order_size * order_price
                    if direction == TradeDirection.Long:
                        out[r] = order_record
                        r += 1
                else:
                    # Position closed
                    remaining_size = add_nb(entry_size_sum, -exit_size_sum)

                    # New position opened
                    entry_size_sum = add_nb(order_size, -remaining_size)
                    entry_gross_sum = entry_size_sum * order_price
                    exit_size_sum = 0.0
                    exit_gross_sum = 0.0
                    if direction == TradeDirection.Long:
                        out[r] = order_record
                        out["size"][r] = remaining_size
                        out["fees"][r] = remaining_size / order_size * order_fees
                        r += 1
                    else:
                        out[r] = order_record
                        out["size"][r] = entry_size_sum
                        out["fees"][r] = entry_size_sum / order_size * order_fees
                        r += 1
                    if order_side == OrderSide.Buy:
                        direction = TradeDirection.Long
                    else:
                        direction = TradeDirection.Short

    return out[:r]


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        close=ch.ArraySlicer(axis=1),
        col_map=base_ch.GroupMapSlicer(),
        init_position=base_ch.FlexArraySlicer(),
        init_price=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_short_view_orders_nb(
    order_records: tp.RecordArray,
    close: tp.Array2d,
    col_map: tp.GroupMap,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.RecordArray:
    """Get view of orders in short positions only.

    Filters order records to return only those corresponding to short positions.
    The function simulates the evolution of trading positions for each column using the provided
    initial position and price, and applies simulation range filters defined by `sim_start` and `sim_end`.

    Args:
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        close (Array2d): 2D array of close prices.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        RecordArray: Filtered order records corresponding to short positions.

    !!! tip
        This function is parallelizable.
    """
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    order_records = order_records.copy()
    out = np.empty(order_records.shape, dtype=order_records.dtype)
    r = 0

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(close.shape[0], col_lens.shape[0]),
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))

        if _init_position != 0:
            # Prepare initial position
            in_position = True
            if _init_position >= 0:
                direction = TradeDirection.Long
            else:
                direction = TradeDirection.Short
            entry_size_sum = abs(_init_position)
            entry_gross_sum = abs(_init_position) * _init_price
            exit_size_sum = 0.0
            exit_gross_sum = 0.0
        else:
            in_position = False

        col_len = col_lens[col]
        if col_len == 0 and not in_position:
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            order_size = order_record["size"]
            order_price = order_record["price"]
            order_side = order_record["side"]
            order_fees = order_record["fees"]

            if order_size <= 0.0:
                raise ValueError(invalid_size_msg)
            if order_price < 0.0:
                raise ValueError(invalid_price_msg)

            if not in_position:
                # New position opened
                in_position = True
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                entry_size_sum = 0.0
                entry_gross_sum = 0.0
                exit_size_sum = 0.0
                exit_gross_sum = 0.0

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) or (
                direction == TradeDirection.Short and order_side == OrderSide.Sell
            ):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                if direction == TradeDirection.Short:
                    out[r] = order_record
                    r += 1
            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) or (
                direction == TradeDirection.Short and order_side == OrderSide.Buy
            ):
                if is_close_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position closed
                    in_position = False
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += order_size * order_price
                    if direction == TradeDirection.Short:
                        out[r] = order_record
                        r += 1
                elif is_less_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position decreased
                    exit_size_sum += order_size
                    exit_gross_sum += order_size * order_price
                    if direction == TradeDirection.Short:
                        out[r] = order_record
                        r += 1
                else:
                    # Position closed
                    remaining_size = add_nb(entry_size_sum, -exit_size_sum)

                    # New position opened
                    entry_size_sum = add_nb(order_size, -remaining_size)
                    entry_gross_sum = entry_size_sum * order_price
                    exit_size_sum = 0.0
                    exit_gross_sum = 0.0
                    if direction == TradeDirection.Short:
                        out[r] = order_record
                        out["size"][r] = remaining_size
                        out["fees"][r] = remaining_size / order_size * order_fees
                        r += 1
                    else:
                        out[r] = order_record
                        out["size"][r] = entry_size_sum
                        out["fees"][r] = entry_size_sum / order_size * order_fees
                        r += 1
                    if order_side == OrderSide.Buy:
                        direction = TradeDirection.Long
                    else:
                        direction = TradeDirection.Short

    return out[:r]


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        close=ch.ArraySlicer(axis=1),
        col_map=base_ch.GroupMapSlicer(),
        feature=None,
        init_position=base_ch.FlexArraySlicer(),
        init_price=base_ch.FlexArraySlicer(),
        fill_closed_position=None,
        fill_exit_price=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_position_feature_nb(
    order_records: tp.RecordArray,
    close: tp.Array2d,
    col_map: tp.GroupMap,
    feature: int = PositionFeature.EntryPrice,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    fill_closed_position: bool = False,
    fill_exit_price: bool = True,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute the position feature at each bar.

    Computes the specified position feature (either entry price or exit price) using order records
    and close prices across multiple columns. The calculation adapts based on the selected
    `feature` and the flags controlling the treatment of open and closed positions.

    Args:
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        close (Array2d): Array of market close prices.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        feature (int): Position feature to compute.

            See `vectorbtpro.portfolio.enums.PositionFeature`.
        init_position (FlexArray1dLike): Initial position.

            Provided as a scalar or per column.
        init_price (FlexArray1dLike): Initial position price.

            Provided as a scalar or per column.
        fill_closed_position (bool): If True, forward-fill missing values using
            prices from a previously closed position.
        fill_exit_price (bool): If True, fill exit prices for open positions using
            the current close price.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array of computed feature values matching the shape of `close`.

    !!! tip
        This function is parallelizable.
    """
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    out = np.full(close.shape, np.nan, dtype=float_)

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=(close.shape[0], col_lens.shape[0]),
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))

        if _init_position != 0:
            # Prepare initial position
            in_position = True
            was_in_position = True
            if _init_position >= 0:
                direction = TradeDirection.Long
            else:
                direction = TradeDirection.Short
            entry_size_sum = abs(_init_position)
            entry_gross_sum = abs(_init_position) * _init_price
            exit_size_sum = 0.0
            exit_gross_sum = 0.0
            last_order_idx = 0
        else:
            in_position = False
            was_in_position = False

        col_len = col_lens[col]
        if col_len == 0 and not in_position:
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            order_idx = order_record["idx"]
            order_size = order_record["size"]
            order_price = order_record["price"]
            order_side = order_record["side"]

            if order_size <= 0.0:
                raise ValueError(invalid_size_msg)
            if order_price < 0.0:
                raise ValueError(invalid_price_msg)

            if in_position:
                if feature == PositionFeature.EntryPrice:
                    if entry_size_sum != 0:
                        entry_price = entry_gross_sum / entry_size_sum
                        out[last_order_idx:order_idx, col] = entry_price
                elif feature == PositionFeature.ExitPrice:
                    if fill_exit_price:
                        remaining_size = add_nb(entry_size_sum, -exit_size_sum)
                        for i in range(last_order_idx, order_idx):
                            open_exit_size_sum = entry_size_sum
                            open_exit_gross_sum = exit_gross_sum + remaining_size * close[i, col]
                            exit_price = open_exit_gross_sum / open_exit_size_sum
                            out[i, col] = exit_price
                    else:
                        if exit_size_sum != 0:
                            exit_price = exit_gross_sum / exit_size_sum
                            out[last_order_idx:order_idx, col] = exit_price
            else:
                if was_in_position and fill_closed_position:
                    if feature == PositionFeature.EntryPrice:
                        if entry_size_sum != 0:
                            entry_price = entry_gross_sum / entry_size_sum
                            out[last_order_idx:order_idx, col] = entry_price
                    elif feature == PositionFeature.ExitPrice:
                        if exit_size_sum != 0:
                            exit_price = exit_gross_sum / exit_size_sum
                            out[last_order_idx:order_idx, col] = exit_price

                # New position opened
                in_position = True
                was_in_position = True
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                entry_size_sum = 0.0
                entry_gross_sum = 0.0
                exit_size_sum = 0.0
                exit_gross_sum = 0.0

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) or (
                direction == TradeDirection.Short and order_side == OrderSide.Sell
            ):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) or (
                direction == TradeDirection.Short and order_side == OrderSide.Buy
            ):
                if is_close_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position closed
                    in_position = False
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += order_size * order_price
                elif is_less_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position decreased
                    exit_size_sum += order_size
                    exit_gross_sum += order_size * order_price
                else:
                    # Position closed
                    remaining_size = add_nb(entry_size_sum, -exit_size_sum)

                    # New position opened
                    if order_side == OrderSide.Buy:
                        direction = TradeDirection.Long
                    else:
                        direction = TradeDirection.Short
                    entry_size_sum = add_nb(order_size, -remaining_size)
                    entry_gross_sum = entry_size_sum * order_price
                    exit_size_sum = 0.0
                    exit_gross_sum = 0.0

            last_order_idx = order_idx

        if in_position:
            if feature == PositionFeature.EntryPrice:
                if entry_size_sum != 0:
                    entry_price = entry_gross_sum / entry_size_sum
                    out[last_order_idx:_sim_end, col] = entry_price
            elif feature == PositionFeature.ExitPrice:
                if fill_exit_price:
                    remaining_size = add_nb(entry_size_sum, -exit_size_sum)
                    for i in range(last_order_idx, _sim_end):
                        open_exit_size_sum = entry_size_sum
                        open_exit_gross_sum = exit_gross_sum + remaining_size * close[i, col]
                        exit_price = open_exit_gross_sum / open_exit_size_sum
                        out[i, col] = exit_price
                else:
                    if exit_size_sum != 0:
                        exit_price = exit_gross_sum / exit_size_sum
                        out[last_order_idx:_sim_end, col] = exit_price
        elif was_in_position and fill_closed_position:
            if feature == PositionFeature.EntryPrice:
                if entry_size_sum != 0:
                    entry_price = entry_gross_sum / entry_size_sum
                    out[last_order_idx:_sim_end, col] = entry_price
            elif feature == PositionFeature.ExitPrice:
                if exit_size_sum != 0:
                    exit_price = exit_gross_sum / exit_size_sum
                    out[last_order_idx:_sim_end, col] = exit_price

    return out


@register_jitted(cache=True)
def price_status_nb(
    order_records: tp.RecordArray,
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
) -> tp.Array1d:
    """Return the status of each order's price relative to provided high and low arrays.

    Args:
        order_records (RecordArray): Array of order records.

            Must adhere to the `vectorbtpro.portfolio.enums.order_dt` dtype.
        high (Optional[FlexArray2d]): Array of high prices corresponding to orders.
        low (Optional[FlexArray2d]): Array of low prices corresponding to orders.

    Returns:
        Array1d: Array of `vectorbtpro.portfolio.enums.OrderPriceStatus` codes indicating
            the price condition for each order.
    """
    out = np.full(len(order_records), 0, dtype=int_)
    for i in range(len(order_records)):
        order = order_records[i]
        if high is not None:
            _high = float(flex_select_nb(high, order["idx"], order["col"]))
        else:
            _high = np.nan
        if low is not None:
            _low = flex_select_nb(low, order["idx"], order["col"])
        else:
            _low = np.nan

        if not np.isnan(_high) and order["price"] > _high:
            out[i] = OrderPriceStatus.AboveHigh
        elif not np.isnan(_low) and order["price"] < _low:
            out[i] = OrderPriceStatus.BelowLow
        elif np.isnan(_high) or np.isnan(_low):
            out[i] = OrderPriceStatus.Unknown
        else:
            out[i] = OrderPriceStatus.OK
    return out


@register_jitted(cache=True)
def trade_winning_streak_nb(trade_records: tp.RecordArray) -> tp.Array1d:
    """Return the current winning streak for each trade.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.

    Returns:
        Array1d: Array of integers representing the winning streak count for each trade.
    """
    out = np.full(len(trade_records), 0, dtype=int_)
    curr_rank = 0
    for i in range(len(trade_records)):
        if trade_records[i]["pnl"] > 0:
            curr_rank += 1
        else:
            curr_rank = 0
        out[i] = curr_rank
    return out


@register_jitted(cache=True)
def trade_losing_streak_nb(trade_records: tp.RecordArray) -> tp.Array1d:
    """Return the current losing streak for each trade.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.

    Returns:
        Array1d: Array of integers representing the losing streak count for each trade.
    """
    out = np.full(len(trade_records), 0, dtype=int_)
    curr_rank = 0
    for i in range(len(trade_records)):
        if trade_records[i]["pnl"] < 0:
            curr_rank += 1
        else:
            curr_rank = 0
        out[i] = curr_rank
    return out


@register_jitted(cache=True)
def win_rate_reduce_nb(pnl_arr: tp.Array1d) -> float:
    """Calculate the win rate from a profit and loss array.

    Args:
        pnl_arr (Array1d): Array of profit and loss values.

    Returns:
        float: Win rate computed as the ratio of winning entries to total valid entries,
            or NaN if no entries exist.
    """
    if pnl_arr.shape[0] == 0:
        return np.nan
    win_count = 0
    count = 0
    for i in range(len(pnl_arr)):
        if not np.isnan(pnl_arr[i]):
            count += 1
            if pnl_arr[i] > 0:
                win_count += 1
    if count == 0:
        return np.nan
    return win_count / count


@register_jitted(cache=True)
def profit_factor_reduce_nb(pnl_arr: tp.Array1d) -> float:
    """Calculate the profit factor from a profit and loss array.

    Args:
        pnl_arr (Array1d): Array of profit and loss values.

    Returns:
        float: Profit factor computed as the ratio of total profits to total losses,
            returning infinity if total losses are zero.
    """
    if pnl_arr.shape[0] == 0:
        return np.nan
    win_sum = 0
    loss_sum = 0
    count = 0
    for i in range(len(pnl_arr)):
        if not np.isnan(pnl_arr[i]):
            count += 1
            if pnl_arr[i] > 0:
                win_sum += pnl_arr[i]
            elif pnl_arr[i] < 0:
                loss_sum += abs(pnl_arr[i])
    if loss_sum == 0:
        return np.inf
    return win_sum / loss_sum


@register_jitted(cache=True)
def expectancy_reduce_nb(pnl_arr: tp.Array1d) -> float:
    """Calculate the expectancy from a profit and loss array.

    Args:
        pnl_arr (Array1d): Array of profit and loss values.

    Returns:
        float: Expectancy computed as the difference between weighted average win and loss,
            or NaN if no valid entries exist.
    """
    if pnl_arr.shape[0] == 0:
        return np.nan
    win_count = 0
    win_sum = 0
    loss_count = 0
    loss_sum = 0
    count = 0
    for i in range(len(pnl_arr)):
        if not np.isnan(pnl_arr[i]):
            count += 1
            if pnl_arr[i] > 0:
                win_count += 1
                win_sum += pnl_arr[i]
            elif pnl_arr[i] < 0:
                loss_count += 1
                loss_sum += abs(pnl_arr[i])
    if count == 0:
        return np.nan
    win_rate = win_count / count
    if win_count == 0:
        win_mean = 0.0
    else:
        win_mean = win_sum / win_count
    loss_rate = loss_count / count
    if loss_count == 0:
        loss_mean = 0.0
    else:
        loss_mean = loss_sum / loss_count
    return win_rate * win_mean - loss_rate * loss_mean


@register_jitted(cache=True)
def sqn_reduce_nb(pnl_arr: tp.Array1d, ddof: int = 0) -> float:
    """Calculate the System Quality Number (SQN) for a profit and loss array.

    Args:
        pnl_arr (Array1d): Array of profit and loss values.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: SQN computed as `sqrt(count) * mean / std`, or NaN if the standard deviation is zero.
    """
    count = generic_nb.nancnt_1d_nb(pnl_arr)
    mean = np.nanmean(pnl_arr)
    std = generic_nb.nanstd_1d_nb(pnl_arr, ddof=ddof)
    if std == 0:
        return np.nan
    return np.sqrt(count) * mean / std


@register_jitted(cache=True)
def trade_best_worst_price_nb(
    trade: tp.Record,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
    idx_relative: bool = True,
    cont_idx: int = -1,
    one_iteration: bool = False,
    vmin: float = np.nan,
    vmax: float = np.nan,
    imin: int = -1,
    imax: int = -1,
) -> tp.Tuple[float, float, int, int]:
    """Compute the best and worst prices during a trade along with their indices.

    Args:
        trade (Record): Trade record containing entry and exit indices, status, direction, and prices.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        open (Optional[FlexArray2d]): Array of open prices.
        high (Optional[FlexArray2d]): Array of high prices.
        low (Optional[FlexArray2d]): Array of low prices.
        close (FlexArray2d): Array of close prices.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
        idx_relative (bool): Return indices relative to the trade's entry index.
        cont_idx (int): Starting row index for computations.

            If set to -1 or equal to the trade's entry index, iteration begins at the entry index.
        one_iteration (bool): Stop the evaluation after one iteration.
        vmin (float): Initial best (minimum) price for continuation.
        vmax (float): Initial worst (maximum) price for continuation.
        imin (int): Initial index corresponding to the best price.
        imax (int): Initial index corresponding to the worst price.

    Returns:
        Tuple[float, float, int, int]:
            * For long trades, returns (best price, worst price, index of best price, index of worst price).
            * For short trades, returns (worst price, best price, index of worst price, index of best price).
    """
    from_i = trade["entry_idx"]
    to_i = trade["exit_idx"]
    trade_open = trade["status"] == TradeStatus.Open
    trade_long = trade["direction"] == TradeDirection.Long

    if cont_idx == -1 or cont_idx == from_i:
        cont_idx = from_i
        vmin = np.nan
        vmax = np.nan
        imin = -1
        imax = -1
    else:
        if trade_long:
            vmin, vmax, imin, imax = vmax, vmin, imax, imin
        if idx_relative:
            imin = from_i + imin
            imax = from_i + imax
    for i in range(cont_idx, to_i + 1):
        if i == from_i:
            if np.isnan(vmin) or trade["entry_price"] < vmin:
                vmin = trade["entry_price"]
                imin = i
            if np.isnan(vmax) or trade["entry_price"] > vmax:
                vmax = trade["entry_price"]
                imax = i
        if i > from_i or entry_price_open:
            if open is not None:
                _open = flex_select_nb(open, i, trade["col"])
                if np.isnan(vmin) or _open < vmin:
                    vmin = _open
                    imin = i
                if np.isnan(vmax) or _open > vmax:
                    vmax = _open
                    imax = i
        if (i > from_i or entry_price_open) and (i < to_i or exit_price_close or trade_open):
            if low is not None:
                _low = flex_select_nb(low, i, trade["col"])
                if np.isnan(vmin) or _low < vmin:
                    vmin = _low
                    imin = i
            if high is not None:
                _high = flex_select_nb(high, i, trade["col"])
                if np.isnan(vmax) or _high > vmax:
                    vmax = _high
                    imax = i
        if i < to_i or exit_price_close or trade_open:
            _close = flex_select_nb(close, i, trade["col"])
            if np.isnan(vmin) or _close < vmin:
                vmin = _close
                imin = i
            if np.isnan(vmax) or _close > vmax:
                vmax = _close
                imax = i
        if max_duration is not None:
            if from_i + max_duration == i:
                break
        if i == to_i:
            if np.isnan(vmin) or trade["exit_price"] < vmin:
                vmin = trade["exit_price"]
                imin = i
            if np.isnan(vmax) or trade["exit_price"] > vmax:
                vmax = trade["exit_price"]
                imax = i
        if one_iteration:
            break
    if idx_relative:
        imin = imin - from_i
        imax = imax - from_i
    if trade_long:
        return vmax, vmin, imax, imin
    return vmin, vmax, imin, imax


@register_chunkable(
    size=ch.ArraySizer(arg_query="trade_records", axis=0),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0),
        open=None,
        high=None,
        low=None,
        close=None,
        entry_price_open=None,
        exit_price_close=None,
        max_duration=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def best_price_nb(
    trade_records: tp.RecordArray,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
) -> tp.Array1d:
    """Compute the best price for each trade using `trade_best_worst_price_nb`.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        open (Optional[FlexArray2d]): Array of open prices.
        high (Optional[FlexArray2d]): Array of high prices.
        low (Optional[FlexArray2d]): Array of low prices.
        close (FlexArray2d): Array of close prices.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.

    Returns:
        Array1d: Array of best prices computed for each trade.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(len(trade_records), dtype=float_)
    for r in prange(len(trade_records)):
        trade = trade_records[r]
        out[r] = trade_best_worst_price_nb(
            trade=trade,
            open=open,
            high=high,
            low=low,
            close=close,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
        )[0]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="trade_records", axis=0),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0),
        open=None,
        high=None,
        low=None,
        close=None,
        entry_price_open=None,
        exit_price_close=None,
        max_duration=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def worst_price_nb(
    trade_records: tp.RecordArray,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
) -> tp.Array1d:
    """Compute the worst price for each trade using `trade_best_worst_price_nb`.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        open (Optional[FlexArray2d]): Array of open prices.
        high (Optional[FlexArray2d]): Array of high prices.
        low (Optional[FlexArray2d]): Array of low prices.
        close (FlexArray2d): Array of close prices.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.

    Returns:
        Array1d: Array of worst prices computed for each trade.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(len(trade_records), dtype=float_)
    for r in prange(len(trade_records)):
        trade = trade_records[r]
        out[r] = trade_best_worst_price_nb(
            trade=trade,
            open=open,
            high=high,
            low=low,
            close=close,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
        )[1]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="trade_records", axis=0),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0),
        open=None,
        high=None,
        low=None,
        close=None,
        entry_price_open=None,
        exit_price_close=None,
        max_duration=None,
        relative=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def best_price_idx_nb(
    trade_records: tp.RecordArray,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
    relative: bool = True,
) -> tp.Array1d:
    """Compute the index of the best price for each trade using `trade_best_worst_price_nb`.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        open (Optional[FlexArray2d]): Array of open prices.
        high (Optional[FlexArray2d]): Array of high prices.
        low (Optional[FlexArray2d]): Array of low prices.
        close (FlexArray2d): Array of close prices.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
        relative (bool): Return the index as relative to the trade's entry index.

    Returns:
        Array1d: Array of indices corresponding to the best prices for each trade.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(len(trade_records), dtype=float_)
    for r in prange(len(trade_records)):
        trade = trade_records[r]
        out[r] = trade_best_worst_price_nb(
            trade=trade,
            open=open,
            high=high,
            low=low,
            close=close,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            idx_relative=relative,
        )[2]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="trade_records", axis=0),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0),
        open=None,
        high=None,
        low=None,
        close=None,
        entry_price_open=None,
        exit_price_close=None,
        max_duration=None,
        relative=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def worst_price_idx_nb(
    trade_records: tp.RecordArray,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
    relative: bool = True,
) -> tp.Array1d:
    """Return worst price index for each trade by applying `trade_best_worst_price_nb` on each trade.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        open (Optional[FlexArray2d]): Array of open prices.
        high (Optional[FlexArray2d]): Array of high prices.
        low (Optional[FlexArray2d]): Array of low prices.
        close (FlexArray2d): Array of close prices.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.
        relative (bool): Flag indicating if the worst price index is relative to the trade's entry index.

    Returns:
        Array1d: Array of worst price indexes for each trade.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(len(trade_records), dtype=float_)
    for r in prange(len(trade_records)):
        trade = trade_records[r]
        out[r] = trade_best_worst_price_nb(
            trade=trade,
            open=open,
            high=high,
            low=low,
            close=close,
            entry_price_open=entry_price_open,
            exit_price_close=exit_price_close,
            max_duration=max_duration,
            idx_relative=relative,
        )[3]
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def expanding_best_price_nb(
    trade_records: tp.RecordArray,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
) -> tp.Array2d:
    """Return an expanding view of the best price for each trade.

    Calculates a 2D array where each column corresponds to a trade and each row represents
    the best price reached up to that time index, starting from the trade's entry.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        open (Optional[FlexArray2d]): Array of open prices.
        high (Optional[FlexArray2d]): Array of high prices.
        low (Optional[FlexArray2d]): Array of low prices.
        close (FlexArray2d): Array of close prices.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.

    Returns:
        Array2d: 2D array where each column represents a trade and each row contains
            the best price up to that time index.

    !!! tip
        This function is parallelizable.
    """
    if max_duration is None:
        _max_duration = 0
        for r in range(len(trade_records)):
            trade = trade_records[r]
            trade_duration = trade["exit_idx"] - trade["entry_idx"]
            if trade_duration > _max_duration:
                _max_duration = trade_duration
    else:
        _max_duration = max_duration
    out = np.full((_max_duration + 1, len(trade_records)), np.nan, dtype=float_)

    for r in prange(len(trade_records)):
        trade = trade_records[r]
        from_i = trade["entry_idx"]
        to_i = trade["exit_idx"]
        vmin = np.nan
        vmax = np.nan
        imin = -1
        imax = -1
        for i in range(from_i, to_i + 1):
            vmin, vmax, imin, imax = trade_best_worst_price_nb(
                trade=trade,
                open=open,
                high=high,
                low=low,
                close=close,
                entry_price_open=entry_price_open,
                exit_price_close=exit_price_close,
                max_duration=max_duration,
                cont_idx=i,
                one_iteration=True,
                vmin=vmin,
                vmax=vmax,
                imin=imin,
                imax=imax,
            )
            out[i - from_i, r] = vmin
            if max_duration is not None:
                if from_i + max_duration == i:
                    break
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def expanding_worst_price_nb(
    trade_records: tp.RecordArray,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
) -> tp.Array2d:
    """Return an expanding view of the worst price for each trade.

    Calculates a 2D array where each column corresponds to a trade and each row represents
    the worst price reached up to that time index, starting from the trade's entry.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        open (Optional[FlexArray2d]): Array of open prices.
        high (Optional[FlexArray2d]): Array of high prices.
        low (Optional[FlexArray2d]): Array of low prices.
        close (FlexArray2d): Array of close prices.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.

    Returns:
        Array2d: 2D array where each column represents a trade and each row contains
            the worst price up to that time index.

    !!! tip
        This function is parallelizable.
    """
    if max_duration is None:
        _max_duration = 0
        for r in range(len(trade_records)):
            trade = trade_records[r]
            trade_duration = trade["exit_idx"] - trade["entry_idx"]
            if trade_duration > _max_duration:
                _max_duration = trade_duration
    else:
        _max_duration = max_duration
    out = np.full((_max_duration + 1, len(trade_records)), np.nan, dtype=float_)

    for r in prange(len(trade_records)):
        trade = trade_records[r]
        from_i = trade["entry_idx"]
        to_i = trade["exit_idx"]
        vmin = np.nan
        vmax = np.nan
        imin = -1
        imax = -1
        for i in range(from_i, to_i + 1):
            vmin, vmax, imin, imax = trade_best_worst_price_nb(
                trade=trade,
                open=open,
                high=high,
                low=low,
                close=close,
                entry_price_open=entry_price_open,
                exit_price_close=exit_price_close,
                max_duration=max_duration,
                cont_idx=i,
                one_iteration=True,
                vmin=vmin,
                vmax=vmax,
                imin=imin,
                imax=imax,
            )
            out[i - from_i, r] = vmax
            if max_duration is not None:
                if from_i + max_duration == i:
                    break
    return out


@register_jitted(cache=True)
def trade_mfe_nb(
    size: float,
    direction: int,
    entry_price: float,
    best_price: float,
    use_returns: bool = False,
) -> float:
    """Compute the Maximum Favorable Excursion (MFE) for a trade.

    Calculates the favorable price movement for a trade based on its direction and
    whether to compute using returns.

    Args:
        size (float): Trade size used in the profit calculation.
        direction (int): Trade direction.

            See `vectorbtpro.portfolio.enums.TradeDirection`.
        entry_price (float): Price at trade entry.
        best_price (float): Best price reached during the trade.
        use_returns (bool): Flag indicating whether to compute using return-based values.

    Returns:
        float: Maximum Favorable Excursion for the trade.
    """
    if direction == TradeDirection.Long:
        if use_returns:
            return (best_price - entry_price) / entry_price
        return (best_price - entry_price) * size
    if use_returns:
        return (entry_price - best_price) / best_price
    return (entry_price - best_price) * size


@register_chunkable(
    size=ch.ArraySizer(arg_query="size", axis=0),
    arg_take_spec=dict(
        size=ch.ArraySlicer(axis=0),
        direction=ch.ArraySlicer(axis=0),
        entry_price=ch.ArraySlicer(axis=0),
        best_price=ch.ArraySlicer(axis=0),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def mfe_nb(
    size: tp.Array1d,
    direction: tp.Array1d,
    entry_price: tp.Array1d,
    best_price: tp.Array1d,
    use_returns: bool = False,
) -> tp.Array1d:
    """Apply `trade_mfe_nb` to compute the Maximum Favorable Excursion (MFE) for each trade.

    Iterates over the provided arrays and computes the MFE for each trade using the `trade_mfe_nb` function.

    Args:
        size (Array1d): Array of trade sizes.
        direction (Array1d): Array of trade direction indicators.
        entry_price (Array1d): Array of entry prices.
        best_price (Array1d): Array of best prices reached during trades.
        use_returns (bool): Flag indicating whether to compute using return-based values.

    Returns:
        Array1d: Array containing the MFE values for each trade.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(size.shape[0], dtype=float_)
    for r in prange(size.shape[0]):
        out[r] = trade_mfe_nb(
            size=size[r],
            direction=direction[r],
            entry_price=entry_price[r],
            best_price=best_price[r],
            use_returns=use_returns,
        )
    return out


@register_jitted(cache=True)
def trade_mae_nb(
    size: float,
    direction: int,
    entry_price: float,
    worst_price: float,
    use_returns: bool = False,
) -> float:
    """Compute the Maximum Adverse Excursion (MAE) for a trade.

    Calculates the adverse price movement for a trade based on its direction and
    whether to compute using returns.

    Args:
        size (float): Trade size used in the loss calculation.
        direction (int): Trade direction.

            See `vectorbtpro.portfolio.enums.TradeDirection`.
        entry_price (float): Price at trade entry.
        worst_price (float): Worst price reached during the trade.
        use_returns (bool): Flag indicating whether to compute using return-based values.

    Returns:
        float: Maximum Adverse Excursion for the trade.
    """
    if direction == TradeDirection.Long:
        if use_returns:
            return (worst_price - entry_price) / entry_price
        return (worst_price - entry_price) * size
    if use_returns:
        return (entry_price - worst_price) / worst_price
    return (entry_price - worst_price) * size


@register_chunkable(
    size=ch.ArraySizer(arg_query="size", axis=0),
    arg_take_spec=dict(
        size=ch.ArraySlicer(axis=0),
        direction=ch.ArraySlicer(axis=0),
        entry_price=ch.ArraySlicer(axis=0),
        worst_price=ch.ArraySlicer(axis=0),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def mae_nb(
    size: tp.Array1d,
    direction: tp.Array1d,
    entry_price: tp.Array1d,
    worst_price: tp.Array1d,
    use_returns: bool = False,
) -> tp.Array1d:
    """Apply `trade_mae_nb` to compute the Maximum Adverse Excursion (MAE) for each trade.

    Iterates over the provided arrays and computes the MAE for each trade using the `trade_mae_nb` function.

    Args:
        size (Array1d): Array of trade sizes.
        direction (Array1d): Array of trade direction indicators.
        entry_price (Array1d): Array of entry prices.
        worst_price (Array1d): Array of worst prices reached during trades.
        use_returns (bool): Flag indicating whether to compute using return-based values.

    Returns:
        Array1d: Array containing the MAE values for each trade.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(size.shape[0], dtype=float_)
    for r in prange(size.shape[0]):
        out[r] = trade_mae_nb(
            size=size[r],
            direction=direction[r],
            entry_price=entry_price[r],
            worst_price=worst_price[r],
            use_returns=use_returns,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="trade_records", axis=0),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0),
        expanding_best_price=ch.ArraySlicer(axis=1),
        use_returns=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def expanding_mfe_nb(
    trade_records: tp.RecordArray,
    expanding_best_price: tp.Array2d,
    use_returns: bool = False,
) -> tp.Array2d:
    """Get expanding maximum favorable excursion (MFE) for each trade using expanding best prices.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        expanding_best_price (Array2d): 2D array of expanding best prices for each trade.
        use_returns (bool): Flag indicating whether to compute using return-based values.

    Returns:
        Array2d: 2D array of expanding MFE values for each trade.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(expanding_best_price, dtype=float_)
    for r in prange(expanding_best_price.shape[1]):
        for i in range(expanding_best_price.shape[0]):
            out[i, r] = trade_mfe_nb(
                size=trade_records["size"][r],
                direction=trade_records["direction"][r],
                entry_price=trade_records["entry_price"][r],
                best_price=expanding_best_price[i, r],
                use_returns=use_returns,
            )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="trade_records", axis=0),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0),
        expanding_worst_price=ch.ArraySlicer(axis=1),
        use_returns=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def expanding_mae_nb(
    trade_records: tp.RecordArray,
    expanding_worst_price: tp.Array2d,
    use_returns: bool = False,
) -> tp.Array2d:
    """Get expanding maximum adverse excursion (MAE) for each trade using expanding worst prices.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        expanding_worst_price (Array2d): 2D array of expanding worst prices for each trade.
        use_returns (bool): Flag indicating whether to compute using return-based values.

    Returns:
        Array2d: 2D array of expanding MAE values for each trade.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(expanding_worst_price, dtype=float_)
    for r in prange(expanding_worst_price.shape[1]):
        for i in range(expanding_worst_price.shape[0]):
            out[i, r] = trade_mae_nb(
                size=trade_records["size"][r],
                direction=trade_records["direction"][r],
                entry_price=trade_records["entry_price"][r],
                worst_price=expanding_worst_price[i, r],
                use_returns=use_returns,
            )
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        open=None,
        high=None,
        low=None,
        close=None,
        volatility=None,
        entry_price_open=None,
        exit_price_close=None,
        max_duration=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def edge_ratio_nb(
    trade_records: tp.RecordArray,
    col_map: tp.GroupMap,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    volatility: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
) -> tp.Array1d:
    """Get the edge ratio for each column based on normalized trade outcomes.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        open (Optional[FlexArray2d]): 2D array representing open prices.
        high (Optional[FlexArray2d]): 2D array representing high prices.
        low (Optional[FlexArray2d]): 2D array representing low prices.
        close (FlexArray2d): 2D array representing close prices.
        volatility (FlexArray2d): 2D array containing volatility data.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.

    Returns:
        Array1d: 1D array containing the edge ratio for each column, computed as the ratio of
            the mean normalized MFE to the mean normalized MAE.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(len(col_lens), np.nan, dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]

        norm_mfe_sum = 0.0
        norm_mfe_cnt = 0
        norm_mae_sum = 0.0
        norm_mae_cnt = 0
        for r in ridxs:
            trade = trade_records[r]
            best_price, worst_price, _, _ = trade_best_worst_price_nb(
                trade=trade,
                open=open,
                high=high,
                low=low,
                close=close,
                entry_price_open=entry_price_open,
                exit_price_close=exit_price_close,
                max_duration=max_duration,
            )
            mfe = abs(
                trade_mfe_nb(
                    size=trade["size"],
                    direction=trade["direction"],
                    entry_price=trade["entry_price"],
                    best_price=best_price,
                    use_returns=False,
                )
            )
            mae = abs(
                trade_mae_nb(
                    size=trade["size"],
                    direction=trade["direction"],
                    entry_price=trade["entry_price"],
                    worst_price=worst_price,
                    use_returns=False,
                )
            )
            _volatility = flex_select_nb(volatility, trade["entry_idx"], trade["col"])
            if _volatility == 0:
                norm_mfe = np.nan
                norm_mae = np.nan
            else:
                norm_mfe = mfe / _volatility
                norm_mae = mae / _volatility
            if not np.isnan(norm_mfe):
                norm_mfe_sum += norm_mfe
                norm_mfe_cnt += 1
            if not np.isnan(norm_mae):
                norm_mae_sum += norm_mae
                norm_mae_cnt += 1
        if norm_mfe_cnt == 0:
            mean_mfe = np.nan
        else:
            mean_mfe = norm_mfe_sum / norm_mfe_cnt
        if norm_mae_cnt == 0:
            mean_mae = np.nan
        else:
            mean_mae = norm_mae_sum / norm_mae_cnt
        if mean_mae == 0:
            out[col] = np.nan
        else:
            out[col] = mean_mfe / mean_mae
    return out


@register_jitted(cache=True, tags={"can_parallel"})
def running_edge_ratio_nb(
    trade_records: tp.RecordArray,
    col_map: tp.GroupMap,
    open: tp.Optional[tp.FlexArray2d],
    high: tp.Optional[tp.FlexArray2d],
    low: tp.Optional[tp.FlexArray2d],
    close: tp.FlexArray2d,
    volatility: tp.FlexArray2d,
    entry_price_open: bool = False,
    exit_price_close: bool = False,
    max_duration: tp.Optional[int] = None,
    incl_shorter: bool = False,
) -> tp.Array2d:
    """Compute the running edge ratio for each column.

    The function calculates a 2D array where each element represents the ratio of the
    mean normalized maximum favorable excursion (MFE) to the mean normalized minimum adverse
    excursion (MAE) for trades associated with each column. Normalization is performed using
    the provided volatility array.

    Args:
        trade_records (RecordArray): Array of trade records.

            Must adhere to the `vectorbtpro.portfolio.enums.trade_dt` dtype.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        open (Optional[FlexArray2d]): Array of open prices.
        high (Optional[FlexArray2d]): Array of high prices.
        low (Optional[FlexArray2d]): Array of low prices.
        close (FlexArray2d): Array of close prices.
        volatility (FlexArray2d): Array of volatility values used for normalization.
        entry_price_open (bool): Include the open price of the entry bar when evaluating prices.
        exit_price_close (bool): Include the close price of the exit bar when evaluating prices.
        max_duration (Optional[int]): Maximum number of bars to evaluate price movements.

            If None, it is determined from the maximum trade duration in `trade_records`.
        incl_shorter (bool): Whether to include trades shorter than the current duration step.

    Returns:
        Array2d: 2D array with shape (max_duration, number of columns) containing the running edge ratio.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens

    if max_duration is None:
        _max_duration = 0
        for r in range(len(trade_records)):
            trade = trade_records[r]
            trade_duration = trade["exit_idx"] - trade["entry_idx"]
            if trade_duration > _max_duration:
                _max_duration = trade_duration
    else:
        _max_duration = max_duration
    out = np.full((_max_duration, len(col_lens)), np.nan, dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]

        for k in range(_max_duration):
            norm_mfe_sum = 0.0
            norm_mfe_cnt = 0
            norm_mae_sum = 0.0
            norm_mae_cnt = 0
            for r in ridxs:
                trade = trade_records[r]
                if not incl_shorter:
                    trade_duration = trade["exit_idx"] - trade["entry_idx"]
                    if trade_duration < k + 1:
                        continue
                best_price, worst_price, _, _ = trade_best_worst_price_nb(
                    trade=trade,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    entry_price_open=entry_price_open,
                    exit_price_close=exit_price_close,
                    max_duration=k + 1,
                )
                mfe = abs(
                    trade_mfe_nb(
                        size=trade["size"],
                        direction=trade["direction"],
                        entry_price=trade["entry_price"],
                        best_price=best_price,
                        use_returns=False,
                    )
                )
                mae = abs(
                    trade_mae_nb(
                        size=trade["size"],
                        direction=trade["direction"],
                        entry_price=trade["entry_price"],
                        worst_price=worst_price,
                        use_returns=False,
                    )
                )
                _volatility = flex_select_nb(volatility, trade["entry_idx"], trade["col"])
                if _volatility == 0:
                    norm_mfe = np.nan
                    norm_mae = np.nan
                else:
                    norm_mfe = mfe / _volatility
                    norm_mae = mae / _volatility
                if not np.isnan(norm_mfe):
                    norm_mfe_sum += norm_mfe
                    norm_mfe_cnt += 1
                if not np.isnan(norm_mae):
                    norm_mae_sum += norm_mae
                    norm_mae_cnt += 1
            if norm_mfe_cnt == 0:
                mean_mfe = np.nan
            else:
                mean_mfe = norm_mfe_sum / norm_mfe_cnt
            if norm_mae_cnt == 0:
                mean_mae = np.nan
            else:
                mean_mae = norm_mae_sum / norm_mae_cnt
            if mean_mae == 0:
                out[k, col] = np.nan
            else:
                out[k, col] = mean_mfe / mean_mae
    return out
