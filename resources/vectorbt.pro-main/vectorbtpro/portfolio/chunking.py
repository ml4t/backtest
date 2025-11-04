# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing extensions for chunking of portfolio."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.merging import column_stack_arrays, concat_arrays
from vectorbtpro.portfolio.enums import SimulationOutput
from vectorbtpro.records.chunking import merge_records
from vectorbtpro.utils.chunking import ArraySlicer, ChunkMeta
from vectorbtpro.utils.config import ReadonlyConfig
from vectorbtpro.utils.template import Rep

__all__ = []


def get_init_cash_slicer(ann_args: tp.AnnArgs) -> ArraySlicer:
    """Return the slicer for `init_cash` based on the cash sharing configuration.

    Args:
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.

    Returns:
        ArraySlicer: Slicer configured for slicing the initial cash values.
    """
    cash_sharing = ann_args["cash_sharing"]["value"]
    if cash_sharing:
        return base_ch.FlexArraySlicer()
    return base_ch.flex_1d_array_gl_slicer


def get_cash_deposits_slicer(ann_args: tp.AnnArgs) -> ArraySlicer:
    """Return the slicer for `cash_deposits` based on the cash sharing configuration.

    Args:
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.

    Returns:
        ArraySlicer: Slicer configured for slicing the cash deposit values.
    """
    cash_sharing = ann_args["cash_sharing"]["value"]
    if cash_sharing:
        return base_ch.FlexArraySlicer(axis=1)
    return base_ch.flex_array_gl_slicer


def in_outputs_merge_func(
    results: tp.List[SimulationOutput],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.AnnArgs,
    mapper: base_ch.GroupLensMapper,
) -> tp.NamedTuple:
    """Merge chunks of in-place output objects.

    Concatenates 1-dimensional arrays, stacks columns of 2-dimensional arrays, and
    merges record arrays using `vectorbtpro.records.chunking.merge_records`.
    Other object types will raise an error.

    Args:
        results (List[SimulationOutput]): List of simulation output chunks.
        chunk_meta (Iterable[ChunkMeta]): Iterable containing metadata for each chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.
        mapper (GroupLensMapper): Mapper for grouping and lens mapping.

    Returns:
        NamedTuple: Instance of the same type as `results[0].in_outputs` with merged data.
    """
    in_outputs = dict()
    for k, v in results[0].in_outputs._asdict().items():
        if v is None:
            in_outputs[k] = None
            continue
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Cannot merge in-place output object '{k}' of type {type(v)}")
        if v.ndim == 2:
            in_outputs[k] = column_stack_arrays([getattr(r.in_outputs, k) for r in results])
        elif v.ndim == 1:
            if v.dtype.fields is None:
                in_outputs[k] = np.concatenate([getattr(r.in_outputs, k) for r in results])
            else:
                records = [getattr(r.in_outputs, k) for r in results]
                in_outputs[k] = merge_records(records, chunk_meta, ann_args=ann_args, mapper=mapper)
        else:
            raise ValueError(
                f"Cannot merge in-place output object '{k}' with number of dimensions {v.ndim}"
            )
    return type(results[0].in_outputs)(**in_outputs)


def merge_sim_outs(
    results: tp.List[SimulationOutput],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.AnnArgs,
    mapper: base_ch.GroupLensMapper,
    in_outputs_merge_func: tp.Callable = in_outputs_merge_func,
    **kwargs,
) -> SimulationOutput:
    """Merge chunks of `vectorbtpro.portfolio.enums.SimulationOutput` instances.

    Merges various components including order and log records, cash deposits, cash earnings, call sequence,
    in-place outputs, and simulation timing arrays. If `vectorbtpro.portfolio.enums.SimulationOutput.in_outputs`
    is provided, a merge function such as `in_outputs_merge_func` must be used.

    Args:
        results (List[SimulationOutput]): List of simulation output chunks.
        chunk_meta (Iterable[ChunkMeta]): Iterable containing metadata for each chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.
        mapper (GroupLensMapper): Mapper for grouping and lens mapping.
        in_outputs_merge_func (Callable): Function to merge in-place output objects.
        **kwargs: Keyword arguments for `in_outputs_merge_func`.

    Returns:
        SimulationOutput: Merged simulation output instance.
    """
    order_records = [r.order_records for r in results]
    order_records = merge_records(order_records, chunk_meta, ann_args=ann_args, mapper=mapper)

    log_records = [r.log_records for r in results]
    log_records = merge_records(log_records, chunk_meta, ann_args=ann_args, mapper=mapper)

    target_shape = ann_args["target_shape"]["value"]
    if results[0].cash_deposits.shape == target_shape:
        cash_deposits = column_stack_arrays([r.cash_deposits for r in results])
    else:
        cash_deposits = results[0].cash_deposits
    if results[0].cash_earnings.shape == target_shape:
        cash_earnings = column_stack_arrays([r.cash_earnings for r in results])
    else:
        cash_earnings = results[0].cash_earnings
    if results[0].call_seq is not None:
        call_seq = column_stack_arrays([r.call_seq for r in results])
    else:
        call_seq = None
    if results[0].in_outputs is not None:
        in_outputs = in_outputs_merge_func(results, chunk_meta, ann_args, mapper, **kwargs)
    else:
        in_outputs = None
    if results[0].sim_start is not None:
        sim_start = concat_arrays([r.sim_start for r in results])
    else:
        sim_start = None
    if results[0].sim_end is not None:
        sim_end = concat_arrays([r.sim_end for r in results])
    else:
        sim_end = None
    return SimulationOutput(
        order_records=order_records,
        log_records=log_records,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        in_outputs=in_outputs,
        sim_start=sim_start,
        sim_end=sim_end,
    )


merge_sim_outs_config = ReadonlyConfig(
    dict(
        merge_func=merge_sim_outs,
        merge_kwargs=dict(
            chunk_meta=Rep("chunk_meta"),
            ann_args=Rep("ann_args"),
            mapper=base_ch.group_lens_mapper,
        ),
    )
)
"""Configuration for merging simulation outputs using `merge_sim_outs`."""
