"""Reporter implementations for capturing backtest events and results."""

import logging
from datetime import datetime
from typing import Any

from qengine.core.event import Event

logger = logging.getLogger(__name__)


class Reporter:
    """Abstract base class for reporters."""

    def on_start(self) -> None:
        """Called at start of backtest."""

    def on_event(self, event: Event) -> None:
        """Called for each event processed."""

    def on_end(self) -> None:
        """Called at end of backtest."""

    def reset(self) -> None:
        """Reset reporter state."""

    def get_report(self) -> Any:
        """Get the generated report."""


class InMemoryReporter(Reporter):
    """Reporter that stores all events and results in memory.

    This reporter captures:
    - All events processed during backtest
    - Timestamps and event counts
    - Summary statistics

    Useful for debugging and analysis of backtest execution.
    """

    def __init__(self, capture_all_events: bool = False):
        """Initialize in-memory reporter.

        Args:
            capture_all_events: If True, store all events (can use significant memory)
        """
        self.capture_all_events = capture_all_events
        self.events = []
        self.event_counts = {}
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.first_event_time: datetime | None = None
        self.last_event_time: datetime | None = None

    def on_start(self) -> None:
        """Mark start of backtest."""
        # Note: start_time will be set from first event
        self.start_time = None
        logger.debug("InMemoryReporter started")

    def on_event(self, event: Event) -> None:
        """Capture event details.

        Args:
            event: Event to record
        """
        # Track event counts by type
        event_type = (
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        )
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        # Track first and last event timestamps
        if self.first_event_time is None:
            self.first_event_time = event.timestamp
        self.last_event_time = event.timestamp

        # Optionally store full event
        if self.capture_all_events:
            self.events.append(
                {
                    "timestamp": event.timestamp,
                    "type": event_type,
                    "event": event,
                },
            )

    def on_end(self) -> None:
        """Mark end of backtest."""
        # Note: end_time will be set from last event
        if not self.end_time:
            self.end_time = self.last_event_time
        logger.debug(f"InMemoryReporter finished. Total events: {sum(self.event_counts.values())}")

    def get_report(self) -> dict[str, Any]:
        """Get summary report of captured events.

        Returns:
            Dictionary with event statistics and timing information
        """
        total_events = sum(self.event_counts.values())
        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.end_time and self.start_time
            else 0
        )

        report = {
            "summary": {
                "total_events": total_events,
                "event_types": len(self.event_counts),
                "duration_seconds": duration,
                "events_per_second": total_events / duration if duration > 0 else 0,
            },
            "event_counts": self.event_counts,
            "timing": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "first_event": self.first_event_time.isoformat() if self.first_event_time else None,
                "last_event": self.last_event_time.isoformat() if self.last_event_time else None,
            },
        }

        if self.capture_all_events:
            report["events"] = self.events

        # Add event type breakdown
        if self.event_counts:
            report["breakdown"] = {
                event_type: {
                    "count": count,
                    "percentage": (count / total_events * 100) if total_events > 0 else 0,
                }
                for event_type, count in self.event_counts.items()
            }

        return report

    def reset(self) -> None:
        """Reset reporter to initial state."""
        self.events.clear()
        self.event_counts.clear()
        self.start_time = None
        self.end_time = None
        self.first_event_time = None
        self.last_event_time = None


class ConsoleReporter(Reporter):
    """Reporter that logs events to console.

    Useful for real-time monitoring of backtest progress.
    """

    def __init__(self, log_level: str = "INFO", log_every_n: int = 1000):
        """Initialize console reporter.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_every_n: Log summary every N events
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_every_n = log_every_n
        self.event_count = 0
        self.event_counts = {}

    def on_start(self) -> None:
        """Log backtest start."""
        logger.log(self.log_level, "=" * 60)
        logger.log(self.log_level, "BACKTEST STARTED")
        logger.log(self.log_level, "=" * 60)

    def on_event(self, event: Event) -> None:
        """Log event if appropriate.

        Args:
            event: Event to potentially log
        """
        self.event_count += 1
        event_type = (
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        )
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        # Log periodic summary
        if self.event_count % self.log_every_n == 0:
            logger.log(
                self.log_level,
                f"Processed {self.event_count:,} events. "
                f"Latest: {event.timestamp} | "
                f"Types: {dict(sorted(self.event_counts.items()))}",
            )

    def on_end(self) -> None:
        """Log backtest completion."""
        logger.log(self.log_level, "=" * 60)
        logger.log(self.log_level, "BACKTEST COMPLETED")
        logger.log(self.log_level, f"Total Events: {self.event_count:,}")
        logger.log(self.log_level, f"Event Breakdown: {dict(sorted(self.event_counts.items()))}")
        logger.log(self.log_level, "=" * 60)

    def reset(self) -> None:
        """Reset reporter state."""
        self.event_count = 0
        self.event_counts.clear()

    def get_report(self) -> dict[str, Any]:
        """Get simple report for console reporter.

        Returns:
            Event count summary
        """
        return {
            "total_events": self.event_count,
            "event_counts": self.event_counts,
        }


__all__ = [
    "ConsoleReporter",
    "InMemoryReporter",
    "Reporter",
]
