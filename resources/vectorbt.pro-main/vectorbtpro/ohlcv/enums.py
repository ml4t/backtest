# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing named tuples and enumerated types for OHLC(V) data."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify_doc

__pdoc__all__ = __all__ = [
    "PriceFeature",
]

__pdoc__ = {}


# ############# Enums ############# #


class PriceFeatureT(tp.NamedTuple):
    Open: int = 0
    High: int = 1
    Low: int = 2
    Close: int = 3


PriceFeature = PriceFeatureT()
"""_"""

__pdoc__["PriceFeature"] = f"""Price feature enumeration.

Fields:
    Open: Index for the open price.
    High: Index for the high price.
    Low: Index for the low price.
    Close: Index for the close price.

```python
{prettify_doc(PriceFeature)}
```
"""
