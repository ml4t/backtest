# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for mapping between enum fields and values.

In vectorbtpro, enums are implemented as named tuple instances to facilitate their use with Numba.
Enum values start at 0, and a value of -1 indicates an undefined enum.
"""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.mapping import apply_mapping, to_value_mapping

__all__ = [
    "map_enum_fields",
]


def map_enum_fields(
    field: tp.Any,
    enum: tp.Union[tp.NamedTuple, tp.EnumMeta],
    enum_unkval: tp.Any = -1,
    ignore_type=int,
    **kwargs,
) -> tp.Any:
    """Map enum fields to corresponding values.

    Args:
        field (Any): Field to be mapped.
        enum (Union[NamedTuple, EnumMeta]): Enum type used for mapping.
        enum_unkval (Any): Value for unknown enumeration items.
        ignore_type (type): Type to ignore during mapping.
        **kwargs: Keyword arguments for `vectorbtpro.utils.mapping.apply_mapping`.

    Returns:
        Any: Mapped enum value.
    """
    mapping = to_value_mapping(enum, reverse=True, enum_unkval=enum_unkval)

    return apply_mapping(field, mapping, ignore_type=ignore_type, **kwargs)


def map_enum_values(
    value: tp.Any,
    enum: tp.Union[tp.NamedTuple, tp.EnumMeta],
    enum_unkval: tp.Any = -1,
    ignore_type=str,
    **kwargs,
) -> tp.Any:
    """Map enum values to corresponding fields.

    Args:
        value (Any): Value to be mapped.
        enum (Union[NamedTuple, EnumMeta]): Enum type used for mapping.
        enum_unkval (Any): Value for unknown enumeration items.
        ignore_type (type): Type to ignore during mapping.
        **kwargs: Keyword arguments for `vectorbtpro.utils.mapping.apply_mapping`.

    Returns:
        Any: Mapped enum field.
    """
    mapping = to_value_mapping(enum, reverse=False, enum_unkval=enum_unkval)

    return apply_mapping(value, mapping, ignore_type=ignore_type, **kwargs)
