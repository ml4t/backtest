"""Zipline callback and container conventions over explicit native simulation inputs."""

from .results import to_performance_frame
from .runner import run_algorithm

__all__ = ["run_algorithm", "to_performance_frame"]
