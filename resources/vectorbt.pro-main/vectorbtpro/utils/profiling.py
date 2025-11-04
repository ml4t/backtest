# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for profiling time and memory usage."""

import tracemalloc
from datetime import timedelta
from functools import partial, wraps
from timeit import Timer as Timer_timeit
from timeit import default_timer

import humanize

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

__all__ = [
    "Timer",
    "with_timer",
    "timeit",
    "with_timeit",
    "MemTracer",
    "with_memtracer",
]


class Timer(Base):
    """Context manager for measuring execution time using `timeit`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> with vbt.Timer() as timer:
        >>>     sleep(1)

        >>> print(timer.elapsed())
        1.01 seconds

        >>> timer.elapsed(readable=False)
        datetime.timedelta(seconds=1, microseconds=5110)
        ```
    """

    def __init__(self) -> None:
        self._start_time = default_timer()
        self._end_time = None

    @property
    def start_time(self) -> float:
        """Start time of the timer.

        Returns:
            float: Start time of the timer.
        """
        return self._start_time

    @property
    def end_time(self) -> float:
        """End time of the timer. If the timer is still running, returns the current time.

        Returns:
            float: End time of the timer or the current time if the timer is still running.
        """
        if self._end_time is None:
            return default_timer()
        return self._end_time

    def elapsed(self, readable: bool = True, **kwargs) -> tp.Union[str, timedelta]:
        """Return the elapsed time since the timer started.

        Args:
            readable (bool): Whether to use a human-readable format.
            **kwargs: Keyword arguments for `humanize.precisedelta`.

        Returns:
            Union[str, timedelta]: Elapsed time as a human-readable string or a timedelta.
        """
        elapsed = self.end_time - self.start_time
        elapsed_delta = timedelta(seconds=elapsed)
        if readable:
            if "minimum_unit" not in kwargs:
                kwargs["minimum_unit"] = "seconds" if elapsed >= 1 else "milliseconds"
            return humanize.precisedelta(elapsed_delta, **kwargs)
        return elapsed_delta

    def __enter__(self) -> tp.Self:
        self._start_time = default_timer()
        return self

    def __exit__(self, *args) -> None:
        self._end_time = default_timer()


def with_timer(
    *args,
    timer_kwargs: tp.KwargsLike = None,
    elapsed_kwargs: tp.KwargsLike = None,
    print_func: tp.Optional[tp.Callable] = None,
    print_format: tp.Optional[str] = None,
    print_kwargs: tp.KwargsLike = None,
) -> tp.Callable:
    """Decorator to measure the execution time of a function using `Timer`.

    This decorator wraps the function execution with a `Timer` to calculate its execution time,
    prints the formatted elapsed time, and returns the function's output.

    Args:
        func (Callable): Function to be decorated.
        timer_kwargs (KwargsLike): Keyword arguments for the `Timer` constructor.
        elapsed_kwargs (KwargsLike): Keyword arguments for the `Timer.elapsed` method.
        print_func (Optional[Callable]): Function used to print the time report.
        print_format (Optional[str]): Format string for displaying elapsed time.
        print_kwargs (KwargsLike): Keyword arguments for `print_func`.

    Returns:
        Callable: Decorated function.
    """

    if timer_kwargs is None:
        timer_kwargs = {}
    if elapsed_kwargs is None:
        elapsed_kwargs = {}
    if print_func is None:
        print_func = print
    if print_format is None:
        print_format = "{func_name} in {elapsed}"
    if print_kwargs is None:
        print_kwargs = {}

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with Timer(**timer_kwargs) as timer:
                out = func(*args, **kwargs)
            elapsed = timer.elapsed(**elapsed_kwargs)
            print_func(
                print_format.format(
                    func_name=func.__qualname__,
                    elapsed=elapsed,
                ),
                **print_kwargs,
            )
            return out

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


def timeit(func: tp.Callable, readable: bool = True, **kwargs) -> tp.Union[str, timedelta]:
    """Run timeit on a function to measure its average execution time.

    This function runs the given function using the `timeit` module to compute the average elapsed time
    over a suitable number of iterations.

    Args:
        func (Callable): Function to be timed.
        readable (bool): Whether to use a human-readable format.
        **kwargs: Keyword arguments for `humanize.precisedelta`.

    Returns:
        Union[str, timedelta]: Average execution time as a human-readable string or a timedelta.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> def my_func():
        ...     sleep(1)

        >>> elapsed = vbt.timeit(my_func)
        >>> print(elapsed)
        1.01 seconds

        >>> vbt.timeit(my_func, readable=False)
        datetime.timedelta(seconds=1, microseconds=1870)
        ```
    """
    timer = Timer_timeit(stmt=func)
    number, time_taken = timer.autorange()
    elapsed = time_taken / number
    elapsed_delta = timedelta(seconds=elapsed)
    if readable:
        if "minimum_unit" not in kwargs:
            kwargs["minimum_unit"] = "seconds" if elapsed >= 1 else "milliseconds"
        return humanize.precisedelta(elapsed_delta, **kwargs)
    return elapsed_delta


def with_timeit(
    *args,
    timeit_kwargs: tp.KwargsLike = None,
    print_func: tp.Optional[tp.Callable] = None,
    print_format: tp.Optional[str] = None,
    print_kwargs: tp.KwargsLike = None,
) -> tp.Callable:
    """Decorator to measure the average execution time of a function using `timeit`.

    This decorator computes the average execution time of the function using `timeit`, prints the formatted result,
    and returns the function's output.

    Args:
        func (Callable): Function to be decorated.
        timeit_kwargs (KwargsLike): Keyword arguments for the `timeit` function.
        print_func (Optional[Callable]): Function used to print the time report.
        print_format (Optional[str]): Format string for displaying elapsed time.
        print_kwargs (KwargsLike): Keyword arguments for `print_func`.

    Returns:
        Callable: Decorated function.
    """

    if timeit_kwargs is None:
        timeit_kwargs = {}
    if print_func is None:
        print_func = print
    if print_format is None:
        print_format = "{func_name} in {elapsed} on average"
    if print_kwargs is None:
        print_kwargs = {}

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            elapsed = timeit(partial(func, *args, **kwargs), **timeit_kwargs)
            print_func(
                print_format.format(
                    func_name=func.__qualname__,
                    elapsed=elapsed,
                ),
                **print_kwargs,
            )
            return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


class MemTracer(Base):
    """Context manager for tracing peak and final memory usage using `tracemalloc`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> with vbt.MemTracer() as tracer:
        >>>     np.random.uniform(size=1000000)

        >>> print(tracer.peak_usage())
        8.0 MB

        >>> tracer.peak_usage(readable=False)
        8005360
        ```
    """

    def __init__(self) -> None:
        self._final_usage = None
        self._peak_usage = None

    def final_usage(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Return the final memory usage recorded by `tracemalloc`.

        Args:
            readable (bool): Whether to use a human-readable format.
            **kwargs: Keyword arguments for `humanize.naturalsize`.

        Returns:
            Union[str, int]: Final memory usage as a human-readable string or as an integer in bytes.
        """
        if self._final_usage is None:
            final_usage = tracemalloc.get_traced_memory()[0]
        else:
            final_usage = self._final_usage
        if readable:
            return humanize.naturalsize(final_usage, **kwargs)
        return final_usage

    def peak_usage(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Return the peak memory usage recorded by `tracemalloc`.

        Args:
            readable (bool): Whether to use a human-readable format.
            **kwargs: Keyword arguments for `humanize.naturalsize`.

        Returns:
            Union[str, int]: Peak memory usage as a human-readable string or as an integer in bytes.
        """
        if self._peak_usage is None:
            peak_usage = tracemalloc.get_traced_memory()[1]
        else:
            peak_usage = self._peak_usage
        if readable:
            return humanize.naturalsize(peak_usage, **kwargs)
        return peak_usage

    def __enter__(self) -> tp.Self:
        tracemalloc.start()
        tracemalloc.clear_traces()
        return self

    def __exit__(self, *args) -> None:
        self._final_usage, self._peak_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()


def with_memtracer(
    *args,
    memtracer_kwargs: tp.KwargsLike = None,
    usage_kwargs: tp.KwargsLike = None,
    print_func: tp.Optional[tp.Callable] = None,
    print_format: tp.Optional[str] = None,
    print_kwargs: tp.KwargsLike = None,
) -> tp.Callable:
    """Run a function with `MemTracer`.

    This decorator wraps a function to monitor its memory usage using `MemTracer`.
    It executes the function within a memory tracking context and then prints the
    peak and final memory usage.

    Args:
        func (Callable): Function to be decorated.
        memtracer_kwargs (KwargsLike): Keyword arguments for `MemTracer`.
        usage_kwargs (KwargsLike): Keyword arguments for `MemTracer.peak_usage` and `MemTracer.final_usage`.
        print_func (Optional[Callable]): Function used to print the memory report.
        print_format (Optional[str]): Format string for displaying memory usage.
        print_kwargs (KwargsLike): Keyword arguments for `print_func`.

    Returns:
        Callable: Decorated function.
    """

    if memtracer_kwargs is None:
        memtracer_kwargs = {}
    if usage_kwargs is None:
        usage_kwargs = {}
    if print_func is None:
        print_func = print
    if print_format is None:
        print_format = (
            "{func_name} with peak usage of {peak_usage} and final usage of {final_usage}"
        )
    if print_kwargs is None:
        print_kwargs = {}

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with MemTracer(**memtracer_kwargs) as memtracer:
                out = func(*args, **kwargs)
            print_func(
                print_format.format(
                    func_name=func.__qualname__,
                    peak_usage=memtracer.peak_usage(**usage_kwargs),
                    final_usage=memtracer.final_usage(**usage_kwargs),
                ),
                **print_kwargs,
            )
            return out

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
