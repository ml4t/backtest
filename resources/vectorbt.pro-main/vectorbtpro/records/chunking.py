# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing extensions for processing chunked record arrays and mapped arrays.

This module defines utility functions and default mappers for adjusting fields in record arrays
based on chunk metadata and for merging record array chunks.
"""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.chunking import GroupIdxsMapper, GroupLensMapper
from vectorbtpro.utils.chunking import ChunkMapper, ChunkMeta
from vectorbtpro.utils.parsing import Regex

__all__ = []

col_lens_mapper = GroupLensMapper(arg_query=Regex(r"(col_lens|col_map)"))
"""Default `GroupLensMapper` instance used to compute per-column lengths."""

col_idxs_mapper = GroupIdxsMapper(arg_query="col_map")
"""Default `GroupIdxsMapper` instance used to compute per-column indices."""


def fix_field_in_records(
    record_arrays: tp.List[tp.RecordArray],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.Optional[tp.AnnArgs] = None,
    mapper: tp.Optional[ChunkMapper] = None,
    field: str = "col",
) -> None:
    """Adjust a specified field in each record array chunk based on chunk metadata.

    Iterates through each chunk metadata object and increments the value of the specified field
    in the corresponding record array by the chunk's starting index. If a mapper is provided,
    the mapped starting index is used instead.

    Args:
        record_arrays (List[RecordArray]): List of record arrays to be adjusted.
        chunk_meta (Iterable[ChunkMeta]): Iterable containing metadata for each chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        ann_args (Optional[AnnArgs]): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.
        mapper (Optional[ChunkMapper]): Mapper used to transform chunk metadata before adjusting the field.
        field (str): Field identifier.

    Returns:
        None: Function modifies the record arrays in place.
    """
    for _chunk_meta in chunk_meta:
        if mapper is None:
            record_arrays[_chunk_meta.idx][field] += _chunk_meta.start
        else:
            _chunk_meta_mapped = mapper.map(_chunk_meta, ann_args=ann_args)
            record_arrays[_chunk_meta.idx][field] += _chunk_meta_mapped.start


def merge_records(
    results: tp.List[tp.RecordArray],
    chunk_meta: tp.Iterable[ChunkMeta],
    ann_args: tp.Optional[tp.AnnArgs] = None,
    mapper: tp.Optional[ChunkMapper] = None,
) -> tp.RecordArray:
    """Merge chunked record arrays into a single record array.

    Adjusts fields in each record array chunk based on the provided chunk metadata and concatenates
    them using NumPy. The `col` field is adjusted using a mapper if provided, and the `group` field
    is adjusted without mapping.

    Args:
        results (List[RecordArray]): List of record array chunks to merge.
        chunk_meta (Iterable[ChunkMeta]): Iterable containing metadata for each chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        ann_args (Optional[AnnArgs]): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.
        mapper (Optional[ChunkMapper]): Mapper to adjust the `col` field in the record arrays.

    Returns:
        RecordArray: Merged record array obtained from concatenating the adjusted record arrays.
    """
    if "col" in results[0].dtype.fields:
        fix_field_in_records(results, chunk_meta, ann_args=ann_args, mapper=mapper, field="col")
    if "group" in results[0].dtype.fields:
        fix_field_in_records(results, chunk_meta, field="group")
    return np.concatenate(results)
