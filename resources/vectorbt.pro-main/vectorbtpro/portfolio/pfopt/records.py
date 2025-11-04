# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes for working with allocation records."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.portfolio.enums import alloc_point_dt, alloc_range_dt
from vectorbtpro.records.base import Records
from vectorbtpro.records.decorators import override_field_config
from vectorbtpro.utils.config import Config, ReadonlyConfig

__all__ = [
    "AllocRanges",
    "AllocPoints",
]

__pdoc__ = {}

# ############# AllocRanges ############# #

alloc_ranges_field_config = ReadonlyConfig(
    dict(
        dtype=alloc_range_dt,
        settings={
            "idx": dict(name="alloc_idx"),  # remap field of Records
            "col": dict(
                title="Group", mapping="groups", group_indexing=True
            ),  # remap field of Records
            "alloc_idx": dict(title="Allocation Index", mapping="index"),
        },
    )
)
"""_"""

__pdoc__["alloc_ranges_field_config"] = f"""Field configuration for `AllocRanges`.

```python
{alloc_ranges_field_config.prettify_doc()}
```
"""

AllocRangesT = tp.TypeVar("AllocRangesT", bound="AllocRanges")


@override_field_config(alloc_ranges_field_config)
class AllocRanges(Ranges):
    """Class extending `vectorbtpro.generic.ranges.Ranges` for working with allocation range records.

    Requires `records_arr` to have all fields defined in `vectorbtpro.portfolio.enums.alloc_range_dt`.
    """

    @property
    def field_config(self) -> Config:
        return self._field_config


AllocRanges.override_field_config_doc(__pdoc__)

# ############# AllocPoints ############# #

alloc_points_field_config = ReadonlyConfig(
    dict(
        dtype=alloc_point_dt,
        settings={
            "idx": dict(name="alloc_idx"),  # remap field of Records
            "col": dict(
                title="Group", mapping="groups", group_indexing=True
            ),  # remap field of Records
            "alloc_idx": dict(title="Allocation Index", mapping="index"),
        },
    )
)
"""_"""

__pdoc__["alloc_points_field_config"] = f"""Field configuration for `AllocPoints`.

```python
{alloc_points_field_config.prettify_doc()}
```
"""

AllocPointsT = tp.TypeVar("AllocPointsT", bound="AllocPoints")


@override_field_config(alloc_points_field_config)
class AllocPoints(Records):
    """Class extending `vectorbtpro.records.base.Records` for working with allocation point records.

    Requires `records_arr` to have all fields defined in `vectorbtpro.portfolio.enums.alloc_point_dt`.
    """

    @property
    def field_config(self) -> Config:
        return self._field_config


AllocPoints.override_field_config_doc(__pdoc__)
