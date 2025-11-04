# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing named tuples and enumerated types for signals."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify_doc

__pdoc__all__ = __all__ = [
    "StopType",
    "SignalRelation",
    "FactoryMode",
    "GenEnContext",
    "GenExContext",
    "GenEnExContext",
    "RankContext",
]

__pdoc__ = {}


# ############# Enums ############# #


class StopTypeT(tp.NamedTuple):
    SL: int = 0
    TSL: int = 1
    TTP: int = 2
    TP: int = 3
    TD: int = 4
    DT: int = 5


StopType = StopTypeT()
"""_"""

__pdoc__["StopType"] = f"""Stop type enumeration.

```python
{prettify_doc(StopType)}
```
"""


class SignalRelationT(tp.NamedTuple):
    OneOne: int = 0
    OneMany: int = 1
    ManyOne: int = 2
    ManyMany: int = 3
    Chain: int = 4
    AnyChain: int = 5


SignalRelation = SignalRelationT()
"""_"""

__pdoc__["SignalRelation"] = f"""Signal relation enumeration.

```python
{prettify_doc(SignalRelation)}
```

Fields:
    OneOne: One source signal maps to exactly one succeeding target signal.
    OneMany: One source signal can map to one or more succeeding target signals.
    ManyOne: One or more source signals can map to exactly one succeeding target signal.
    ManyMany: One or more source signals can map to one or more succeeding target signals.
    Chain: First source signal maps to the first target signal after it and vice versa.
    AnyChain: First signal maps to the first opposite signal after it and vice versa.
"""


class FactoryModeT(tp.NamedTuple):
    Entries: int = 0
    Exits: int = 1
    Both: int = 2
    Chain: int = 3


FactoryMode = FactoryModeT()
"""_"""

__pdoc__["FactoryMode"] = f"""Factory mode enumeration.

```python
{prettify_doc(FactoryMode)}
```

Fields:
    Entries: Generate entries only using `generate_func_nb`.

        Takes no input signal arrays.
        Produces one output signal array - `entries`.

        Such generators often have no suffix.
    Exits: Generate exits only using `generate_ex_func_nb`.

        Takes one input signal array - `entries`.
        Produces one output signal array - `exits`.

        Such generators often have suffix 'X'.
    Both: Generate both entries and exits using `generate_enex_func_nb`.

        Takes no input signal arrays.
        Produces two output signal arrays - `entries` and `exits`.

        Such generators often have suffix 'NX'.
    Chain: Generate a chain of entries and exits using `generate_enex_func_nb`.

        Takes one input signal array - `entries`.
        Produces two output signal arrays - `new_entries` and `exits`.

        Such generators often have suffix 'CX'.
"""


# ############# Named tuples ############# #


class GenEnContext(tp.NamedTuple):
    target_shape: tp.Shape
    only_once: bool
    wait: int
    entries_out: tp.Array2d
    out: tp.Array1d
    from_i: int
    to_i: int
    col: int


__pdoc__["GenEnContext"] = "Named tuple representing the context for an entry signal generator."
__pdoc__["GenEnContext.target_shape"] = "Target shape."
__pdoc__["GenEnContext.only_once"] = "Whether to run the placement function only once."
__pdoc__["GenEnContext.wait"] = "Number of ticks to wait before placing the next entry."
__pdoc__["GenEnContext.entries_out"] = "Output array with entries."
__pdoc__["GenEnContext.out"] = "Current segment of the output array with entries."
__pdoc__["GenEnContext.from_i"] = "Start index of the segment (inclusive)."
__pdoc__["GenEnContext.to_i"] = "End index of the segment (exclusive)."
__pdoc__["GenEnContext.col"] = "Column of the segment."


class GenExContext(tp.NamedTuple):
    entries: tp.Array2d
    until_next: bool
    skip_until_exit: bool
    exits_out: tp.Array2d
    out: tp.Array1d
    wait: int
    from_i: int
    to_i: int
    col: int


__pdoc__["GenExContext"] = "Named tuple representing the context for an exit signal generator."
__pdoc__["GenExContext.entries"] = "Input array with entries."
__pdoc__["GenExContext.until_next"] = "Whether to place signals up to the next entry signal."
__pdoc__["GenExContext.skip_until_exit"] = (
    "Whether to skip processing entry signals until the next exit."
)
__pdoc__["GenExContext.exits_out"] = "Output array with exits."
__pdoc__["GenExContext.out"] = "Current segment of the output array with exits."
__pdoc__["GenExContext.wait"] = "Number of ticks to wait before placing exits."
__pdoc__["GenExContext.from_i"] = "Start index of the segment (inclusive)."
__pdoc__["GenExContext.to_i"] = "End index of the segment (exclusive)."
__pdoc__["GenExContext.col"] = "Column of the segment."


class GenEnExContext(tp.NamedTuple):
    target_shape: tp.Shape
    entry_wait: int
    exit_wait: int
    entries_out: tp.Array2d
    exits_out: tp.Array2d
    entries_turn: bool
    wait: int
    out: tp.Array1d
    from_i: int
    to_i: int
    col: int


__pdoc__["GenExContext"] = (
    "Named tuple representing the context for an entry/exit signal generator."
)
__pdoc__["GenExContext.target_shape"] = "Target shape."
__pdoc__["GenExContext.entry_wait"] = "Number of ticks to wait before placing entries."
__pdoc__["GenExContext.exit_wait"] = "Number of ticks to wait before placing exits."
__pdoc__["GenExContext.entries_out"] = "Output array with entries."
__pdoc__["GenExContext.exits_out"] = "Output array with exits."
__pdoc__["GenExContext.entries_turn"] = "Whether the current turn is generating an entry."
__pdoc__["GenExContext.out"] = "Current segment of the output array with entries/exits."
__pdoc__["GenExContext.wait"] = "Number of ticks to wait before placing entries/exits."
__pdoc__["GenExContext.from_i"] = "Start index of the segment (inclusive)."
__pdoc__["GenExContext.to_i"] = "End index of the segment (exclusive)."
__pdoc__["GenExContext.col"] = "Column of the segment."


class RankContext(tp.NamedTuple):
    mask: tp.Array2d
    reset_by: tp.Optional[tp.Array2d]
    after_false: bool
    after_reset: bool
    reset_wait: int
    col: int
    i: int
    last_false_i: int
    last_reset_i: int
    all_sig_cnt: int
    all_part_cnt: int
    all_sig_in_part_cnt: int
    nonres_sig_cnt: int
    nonres_part_cnt: int
    nonres_sig_in_part_cnt: int
    sig_cnt: int
    part_cnt: int
    sig_in_part_cnt: int


__pdoc__["RankContext"] = "Named tuple representing the context for ranking signals."
__pdoc__["RankContext.mask"] = "Source mask."
__pdoc__["RankContext.reset_by"] = "Mask used for resetting."
__pdoc__["RankContext.after_false"] = (
    """Indicates whether to disregard the first partition of True values when no preceding False value exists."""
)
__pdoc__["RankContext.after_reset"] = (
    """Indicates whether to disregard the first partition of True values that occur before the first reset signal."""
)
__pdoc__["RankContext.reset_wait"] = (
    "Number of ticks to wait before resetting the current partition."
)
__pdoc__["RankContext.col"] = "Current column."
__pdoc__["RankContext.i"] = "Current row."
__pdoc__["RankContext.last_false_i"] = "Row of the last False value in the main mask."
__pdoc__["RankContext.last_reset_i"] = """Row index of the last True value in the resetting mask.

Does not account for `reset_wait`.
"""
__pdoc__["RankContext.all_sig_cnt"] = (
    """Total number of signals encountered, including the current one."""
)
__pdoc__["RankContext.all_part_cnt"] = (
    """Total number of partitions encountered, including the current partition."""
)
__pdoc__["RankContext.all_sig_in_part_cnt"] = (
    """Total number of signals encountered in the current partition, including the current signal."""
)
__pdoc__["RankContext.nonres_sig_cnt"] = (
    """Total number of non-resetting signals encountered, including the current one."""
)
__pdoc__["RankContext.nonres_part_cnt"] = (
    """Total number of non-resetting partitions encountered, including the current partition."""
)
__pdoc__["RankContext.nonres_sig_in_part_cnt"] = (
    """Total number of signals encountered in the current non-resetting partition, including the current signal."""
)
__pdoc__["RankContext.sig_cnt"] = (
    """Total number of valid and resetting signals encountered, including the current one."""
)
__pdoc__["RankContext.part_cnt"] = (
    """Total number of valid and resetting partitions encountered, including the current partition."""
)
__pdoc__["RankContext.sig_in_part_cnt"] = (
    """Total number of signals encountered in the current valid and resetting partition, including the current signal."""
)
