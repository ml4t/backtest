# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing utility functions and classes for constructing and managing knowledge assets.

Run for the examples:

```pycon
>>> dataset = [
...     {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
...     {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
...     {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
...     {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
...     {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123}
... ]
>>> asset = vbt.KnowledgeAsset(dataset)
```

!!! info
    For default settings, see `vectorbtpro._settings.knowledge`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.utils.knowledge.asset_pipelines import *
    from vectorbtpro.utils.knowledge.base_asset_funcs import *
    from vectorbtpro.utils.knowledge.base_assets import *
    from vectorbtpro.utils.knowledge.chatting import *
    from vectorbtpro.utils.knowledge.custom_asset_funcs import *
    from vectorbtpro.utils.knowledge.custom_assets import *
    from vectorbtpro.utils.knowledge.formatting import *
