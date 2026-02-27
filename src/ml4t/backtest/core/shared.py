"""Shared core helpers for broker decomposition."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import ExitReason


@dataclass
class SubmitOrderOptions:
    """Internal options for submit_order behavior."""

    eligible_in_next_bar_mode: bool = False


def reason_to_exit_reason(reason: str) -> ExitReason:
    """Map human-readable rule reason to typed ExitReason."""
    reason_lower = reason.lower()
    if "stop_loss" in reason_lower:
        return ExitReason.STOP_LOSS
    elif "take_profit" in reason_lower:
        return ExitReason.TAKE_PROFIT
    elif "trailing" in reason_lower:
        return ExitReason.TRAILING_STOP
    elif "time" in reason_lower:
        return ExitReason.TIME_STOP
    elif "end_of_data" in reason_lower:
        return ExitReason.END_OF_DATA
    else:
        return ExitReason.SIGNAL
