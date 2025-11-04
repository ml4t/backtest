# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing engines for executing functions.

!!! info
    For default settings, see `vectorbtpro._settings.execution`.
"""

import concurrent.futures
import enum
import inspect
import time
from contextlib import nullcontext
from functools import partial, wraps
from pathlib import Path

import pandas as pd
from numba.core.registry import CPUDispatcher

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import MISSING, DefineMixin, define
from vectorbtpro.utils.config import Configured, FrozenConfig, merge_dicts
from vectorbtpro.utils.merging import MergeFunc
from vectorbtpro.utils.parsing import (
    annotate_args,
    flat_ann_args_to_args,
    get_func_arg_names,
    match_and_set_flat_ann_arg,
    match_ann_arg,
)
from vectorbtpro.utils.path_ import file_exists, remove_dir
from vectorbtpro.utils.pbar import ProgressBar, ProgressHidden
from vectorbtpro.utils.pickling import load, save
from vectorbtpro.utils.template import CustomTemplate, substitute_templates
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from ray import ObjectRef as ObjectRefT
    from ray.remote_function import RemoteFunction as RemoteFunctionT
else:
    RemoteFunctionT = "ray.remote_function.RemoteFunction"
    ObjectRefT = "ray.ObjectRef"

__all__ = [
    "Task",
    "NoResult",
    "NoResultsException",
    "ExecutionEngine",
    "SerialEngine",
    "ThreadPoolEngine",
    "ProcessPoolEngine",
    "PathosEngine",
    "DaskEngine",
    "RayEngine",
    "Executor",
    "execute",
    "iterated",
]

TaskT = tp.TypeVar("TaskT", bound="Task")


@define
class Task(DefineMixin):
    """Class representing an executable task.

    Args:
        func (Callable): Function to execute.
        *args: Positional arguments for `func`.
        **kwargs: Keyword arguments for `func`.
    """

    func: tp.Callable = define.field()
    """Callable representing the function to execute."""

    args: tp.Args = define.field(factory=tuple)
    """Positional arguments for the function."""

    kwargs: tp.Kwargs = define.field(factory=dict)
    """Keyword arguments for the function."""

    def __init__(self, func: tp.Callable, *args, **kwargs) -> None:
        DefineMixin.__init__(self, func=func, args=args, kwargs=kwargs)

    @classmethod
    def from_tuple(cls: tp.Type[TaskT], tuple_: tp.Tuple[tp.Any, ...]) -> TaskT:
        """Construct a Task instance from a tuple representation.

        Args:
            tuple_ (Tuple[Any, ...]): Tuple representing a task.

                The tuple can be:

                * Two elements: if the second element is a tuple, it is treated as positional arguments.
                    If it is a dict, it is treated as keyword arguments.
                * Three elements: the second element is a tuple of positional arguments and the third
                    element is a dict of keyword arguments.

        Returns:
            Task: Task instance constructed from the tuple.
        """
        if len(tuple_) == 2:
            if isinstance(tuple_[1], tuple):
                return cls(tuple_[0], *tuple_[1])
            if isinstance(tuple_[1], dict):
                return cls(tuple_[0], **tuple_[1])
        if len(tuple_) == 3:
            if isinstance(tuple_[1], tuple) and isinstance(tuple_[2], dict):
                return cls(tuple_[0], *tuple_[1], **tuple_[2])
        return cls(*tuple_)

    def __iter__(self) -> tp.Iterator:
        return iter((self.func, self.args, self.kwargs))

    def __getitem__(self, item: int) -> tp.Any:
        return tuple(self)[item]

    def execute(self) -> tp.Any:
        """Run the task by calling its function with the provided arguments.

        Returns:
            Any: Result of executing the function.
        """
        return self.func(*self.args, **self.kwargs)

    def __call__(self) -> tp.Any:
        return self.execute()


class _NoResult(enum.Enum):
    """Enum class representing a no-result sentinel value."""

    NoResult = enum.auto()

    def __repr__(self):
        return "NoResult"

    def __bool__(self):
        return False


NoResult = _NoResult.NoResult
"""Sentinel value representing the absence of a result."""


class NoResultsException(Exception):
    """Exception raised when no valid results remain after filtering out `NoResult`."""

    pass


def filter_out_no_results(
    objs: tp.Iterable[tp.Any],
    keys: tp.Optional[tp.Index] = MISSING,
    raise_error: bool = True,
) -> tp.Union[tp.List, tp.Tuple[tp.List, tp.Optional[tp.Index]]]:
    """Filter out `NoResult` sentinel values from a collection of objects and adjust associated keys.

    Args:
        objs (Iterable[Any]): Sequence of objects to filter.
        keys (Optional[Index]): Sequence of keys corresponding to the objects.
        raise_error (bool): If True, raise a `NoResultsException` when no valid objects remain after filtering.

    Returns:
        Union[List[Any], Tuple[List[Any], Optional[Index]]]:
            If `keys` is provided, returns a tuple of the filtered objects and the corresponding filtered keys.
            Otherwise, returns the list of filtered objects.
    """
    skip_indices = set()
    for i, obj in enumerate(objs):
        if obj is NoResult:
            skip_indices.add(i)
    if len(skip_indices) > 0:
        new_objs = []
        keep_indices = []
        for i, obj in enumerate(objs):
            if i not in skip_indices:
                new_objs.append(obj)
                keep_indices.append(i)
        objs = new_objs
        if keys is not None and keys is not MISSING:
            if isinstance(keys, pd.Index):
                keys = keys[keep_indices]
            else:
                keys = [keys[i] for i in keep_indices]
        if len(objs) == 0 and raise_error:
            raise NoResultsException
    if keys is not MISSING:
        return objs, keys
    return objs


class ExecutionEngine(Configured):
    """Abstract class for execution engines that run tasks.

    This class defines the interface for executing a collection of tasks.
    """

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        """Execute a collection of tasks.

        Args:
            tasks (TasksLike): Tasks (i.e., functions with their arguments) to execute.
            size (Optional[int]): Hint for the number of tasks, useful if `tasks` is a generator.
            keys (Optional[IndexLike]): Keys associated with each task.

        Returns:
            ExecResults: Results of executing the tasks.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


class SerialEngine(ExecutionEngine):
    """Class for executing functions sequentially.

    Args:
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

            See `vectorbtpro.utils.pbar.ProgressBar`.
        clear_cache (Union[None, bool, int]): Indicates whether to clear vectorbtpro's cache after each iteration.

            If provided as an integer, clears the cache every specified number of tasks.
        collect_garbage (Union[None, bool, int]): Specifies whether to perform garbage collection
            after each iteration.

            If provided as an integer, collects garbage every specified number of tasks.
        delay (Optional[float]): Delay in seconds after each function call.
        **kwargs: Keyword arguments for `ExecutionEngine`.

    !!! info
        For default settings, see `engines.serial` in `vectorbtpro._settings.execution`.
    """

    _settings_path: tp.SettingsPath = "execution.engines.serial"

    def __init__(
        self,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Union[None, bool, int] = None,
        collect_garbage: tp.Union[None, bool, int] = None,
        delay: tp.Optional[float] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            clear_cache=clear_cache,
            collect_garbage=collect_garbage,
            delay=delay,
            **kwargs,
        )

        self._show_progress = self.resolve_setting(show_progress, "show_progress")
        self._pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        self._clear_cache = self.resolve_setting(clear_cache, "clear_cache")
        self._collect_garbage = self.resolve_setting(collect_garbage, "collect_garbage")
        self._delay = self.resolve_setting(delay, "delay")

    @property
    def show_progress(self) -> bool:
        """Indicates whether the progress bar is displayed.

        Returns:
            bool: True if the progress bar is shown, False otherwise.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for configuring the progress bar.

        See `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Keyword arguments for the progress bar.
        """
        return self._pbar_kwargs

    @property
    def clear_cache(self) -> tp.Union[bool, int]:
        """Indicates whether to clear vectorbtpro's cache after each iteration.

        If provided as an integer, clears the cache every specified number of tasks.

        Returns:
            Union[bool, int]: True to clear cache after each iteration, False to skip,
                or an integer specifying the interval for clearing the cache.
        """
        return self._clear_cache

    @property
    def collect_garbage(self) -> tp.Union[bool, int]:
        """Indicates whether to perform garbage collection after each iteration.

        If provided as an integer, collects garbage every specified number of tasks.

        Returns:
            Union[bool, int]: True to collect garbage after each iteration, False to skip,
                or an integer specifying the interval for collecting garbage.
        """
        return self._collect_garbage

    @property
    def delay(self) -> tp.Optional[float]:
        """Specifies the number of seconds to pause after each function call.

        Returns:
            Optional[float]: Delay in seconds. If None, no delay is applied.
        """
        return self._delay

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.base.indexes import to_any_index
        from vectorbtpro.registries.ca_registry import clear_cache, collect_garbage

        results = []
        if size is None and hasattr(tasks, "__len__"):
            size = len(tasks)
        elif keys is not None and hasattr(keys, "__len__"):
            size = len(keys)
        if keys is not None:
            keys = to_any_index(keys)
        pbar_kwargs = dict(self.pbar_kwargs)
        if "bar_id" not in pbar_kwargs:
            if keys is not None:
                if isinstance(keys, pd.MultiIndex):
                    pbar_kwargs["bar_id"] = tuple(keys.names)
                else:
                    pbar_kwargs["bar_id"] = keys.name
        pbar = ProgressBar(total=size, show_progress=self.show_progress, **pbar_kwargs)

        with pbar:
            if keys is not None:
                if isinstance(keys, pd.MultiIndex):
                    pbar.set_description(dict(zip(keys.names, keys[0])))
                else:
                    pbar.set_description(dict(zip(keys.names, [keys[0]])))

            for i, (func, args, kwargs) in enumerate(tasks):
                results.append(func(*args, **kwargs))
                if isinstance(self.clear_cache, bool):
                    if self.clear_cache:
                        clear_cache()
                elif i > 0 and (i + 1) % self.clear_cache == 0:
                    clear_cache()
                if isinstance(self.collect_garbage, bool):
                    if self.collect_garbage:
                        collect_garbage()
                elif i > 0 and (i + 1) % self.collect_garbage == 0:
                    collect_garbage()
                if self.delay is not None:
                    time.sleep(self.delay)

                if keys is not None and i + 1 < len(keys):
                    if isinstance(keys, pd.MultiIndex):
                        pbar.set_description(dict(zip(keys.names, keys[i + 1])))
                    else:
                        pbar.set_description(dict(zip(keys.names, [keys[i + 1]])))
                pbar.update()

        return results


class ThreadPoolEngine(ExecutionEngine):
    """Class for executing functions using `ThreadPoolExecutor` from `concurrent.futures`.

    Args:
        init_kwargs (KwargsLike): Keyword arguments for `ThreadPoolExecutor`.
        timeout (Optional[int]): Maximum number of seconds to wait.
        hide_inner_progress (Optional[bool]): Flag indicating whether to hide progress bars
            within individual threads.
        **kwargs: Keyword arguments for `ExecutionEngine`.

    !!! info
        For default settings, see `engines.threadpool` in `vectorbtpro._settings.execution`.
    """

    _settings_path: tp.SettingsPath = "execution.engines.threadpool"

    def __init__(
        self,
        init_kwargs: tp.KwargsLike = None,
        timeout: tp.Optional[int] = None,
        hide_inner_progress: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            timeout=timeout,
            hide_inner_progress=hide_inner_progress,
            **kwargs,
        )

        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._timeout = self.resolve_setting(timeout, "timeout")
        self._hide_inner_progress = self.resolve_setting(hide_inner_progress, "hide_inner_progress")

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Configuration keyword arguments for `concurrent.futures.ThreadPoolExecutor`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Timeout for waiting on task results.

        Returns:
            Optional[int]: Timeout in seconds, or None if no timeout is set.
        """
        return self._timeout

    @property
    def hide_inner_progress(self) -> bool:
        """Indicates whether progress bars within each thread are hidden.

        Returns:
            bool: True if inner progress bars are hidden, False otherwise.
        """
        return self._hide_inner_progress

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        if self.hide_inner_progress:
            inner_progress_context = ProgressHidden()
        else:
            inner_progress_context = nullcontext()
        with inner_progress_context:
            with concurrent.futures.ThreadPoolExecutor(**self.init_kwargs) as executor:
                futures = {}
                for i, (func, args, kwargs) in enumerate(tasks):
                    future = executor.submit(func, *args, **kwargs)
                    futures[future] = i
                results = [None] * len(futures)
                for fut in concurrent.futures.as_completed(futures, timeout=self.timeout):
                    results[futures[fut]] = fut.result()
                return results


class ProcessPoolEngine(ExecutionEngine):
    """Class for executing functions using `ProcessPoolExecutor` from `concurrent.futures`.

    Args:
        init_kwargs (KwargsLike): Keyword arguments for `ProcessPoolExecutor`.
        timeout (Optional[int]): Maximum number of seconds to wait.
        hide_inner_progress (Optional[bool]): Flag indicating whether to hide progress bars
            within individual threads.
        **kwargs: Keyword arguments for `ExecutionEngine`.

    !!! info
        For default settings, see `engines.processpool` in `vectorbtpro._settings.execution`.
    """

    _settings_path: tp.SettingsPath = "execution.engines.processpool"

    def __init__(
        self,
        init_kwargs: tp.KwargsLike = None,
        timeout: tp.Optional[int] = None,
        hide_inner_progress: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            timeout=timeout,
            hide_inner_progress=hide_inner_progress,
            **kwargs,
        )

        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._timeout = self.resolve_setting(timeout, "timeout")
        self._hide_inner_progress = self.resolve_setting(hide_inner_progress, "hide_inner_progress")

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Configuration keyword arguments for `concurrent.futures.ProcessPoolExecutor`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Timeout for waiting on task results.

        Returns:
            Optional[int]: Timeout in seconds, or None if no timeout is set.
        """
        return self._timeout

    @property
    def hide_inner_progress(self) -> bool:
        """Indicates whether progress bars within each thread are hidden.

        Returns:
            bool: True if inner progress bars are hidden, False otherwise.
        """
        return self._hide_inner_progress

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        if self.hide_inner_progress:
            inner_progress_context = ProgressHidden()
        else:
            inner_progress_context = nullcontext()
        with inner_progress_context:
            with concurrent.futures.ProcessPoolExecutor(**self.init_kwargs) as executor:
                futures = {}
                for i, (func, args, kwargs) in enumerate(tasks):
                    future = executor.submit(func, *args, **kwargs)
                    futures[future] = i
                results = [None] * len(futures)
                for fut in concurrent.futures.as_completed(futures, timeout=self.timeout):
                    results[futures[fut]] = fut.result()
                return results


def pass_kwargs_as_args(func: tp.Callable, args: tp.Args, kwargs: tp.Kwargs) -> tp.Any:
    """Return the result of calling `func` with the supplied arguments and keyword arguments.

    Used for compatibility with `pathos.pools.ParallelPool`.

    Args:
        func (Callable): Function to execute.
        args (Args): Positional arguments for the function.
        kwargs (Kwargs): Keyword arguments for the function.

    Returns:
        Any: Result of executing the function.
    """
    return func(*args, **kwargs)


class PathosEngine(ExecutionEngine):
    """Class for executing functions using `pathos`.

    Args:
        pool_type (Optional[str]): Pool type used for parallel execution.
        init_kwargs (KwargsLike): Keyword arguments used for initializing the pool.
        timeout (Optional[int]): Maximum number of seconds to wait.
        check_delay (Optional[float]): Delay in seconds between successive task status checks.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

            See `vectorbtpro.utils.pbar.ProgressBar`.
        hide_inner_progress (Optional[bool]): Flag indicating whether to hide progress bars
            within individual threads.
        join_pool (Optional[bool]): Flag indicating whether the pool should be joined after execution.
        **kwargs: Keyword arguments for `ExecutionEngine`.

    !!! info
        For default settings, see `engines.pathos` in `vectorbtpro._settings.execution`.
    """

    _settings_path: tp.SettingsPath = "execution.engines.pathos"

    def __init__(
        self,
        pool_type: tp.Optional[str] = None,
        init_kwargs: tp.KwargsLike = None,
        timeout: tp.Optional[int] = None,
        check_delay: tp.Optional[float] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        hide_inner_progress: tp.Optional[bool] = None,
        join_pool: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            pool_type=pool_type,
            init_kwargs=init_kwargs,
            timeout=timeout,
            check_delay=check_delay,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            hide_inner_progress=hide_inner_progress,
            join_pool=join_pool,
            **kwargs,
        )

        self._pool_type = self.resolve_setting(pool_type, "pool_type")
        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._timeout = self.resolve_setting(timeout, "timeout")
        self._check_delay = self.resolve_setting(check_delay, "check_delay")
        self._show_progress = self.resolve_setting(show_progress, "show_progress")
        self._pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        self._hide_inner_progress = self.resolve_setting(hide_inner_progress, "hide_inner_progress")
        self._join_pool = self.resolve_setting(join_pool, "join_pool")

    @property
    def pool_type(self) -> str:
        """Pool type used for parallel execution.

        Returns:
            str: Pool type, one of 'thread', 'process', or 'parallel'.
        """
        return self._pool_type

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for initializing the pool.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Maximum number of seconds to wait.

        Returns:
            Optional[int]: Timeout in seconds, or None if no timeout is set.
        """
        return self._timeout

    @property
    def check_delay(self) -> tp.Optional[float]:
        """Delay in seconds between successive task status checks.

        Returns:
            Optional[float]: Delay in seconds between task status checks.
        """
        return self._check_delay

    @property
    def show_progress(self) -> bool:
        """Flag indicating whether to display the progress bar.

        Returns:
            bool: True if the progress bar is shown, False otherwise.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for configuring the progress bar.

        See `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._pbar_kwargs

    @property
    def hide_inner_progress(self) -> bool:
        """Flag indicating whether to hide progress bars within individual threads.

        Returns:
            bool: True if inner progress bars are hidden, False otherwise.
        """
        return self._hide_inner_progress

    @property
    def join_pool(self) -> bool:
        """Flag indicating whether the pool should be joined after execution.

        Returns:
            bool: True if the pool should be joined, False otherwise.
        """
        return self._join_pool

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pathos")

        if self.pool_type.lower() in ("thread", "threadpool"):
            from pathos.pools import ThreadPool as Pool
        elif self.pool_type.lower() in ("process", "processpool"):
            from pathos.pools import ProcessPool as Pool
        elif self.pool_type.lower() in ("parallel", "parallelpool"):
            from pathos.pools import ParallelPool as Pool

            tasks = [(pass_kwargs_as_args, x, {}) for x in tasks]
        else:
            raise ValueError(f"Invalid pool_type: '{self.pool_type}'")
        if size is None and hasattr(tasks, "__len__"):
            size = len(tasks)
        elif keys is not None and hasattr(keys, "__len__"):
            size = len(keys)
        pbar_kwargs = dict(self.pbar_kwargs)
        if "bar_id" not in pbar_kwargs:
            if keys is not None:
                if isinstance(keys, pd.MultiIndex):
                    pbar_kwargs["bar_id"] = tuple(keys.names)
                else:
                    pbar_kwargs["bar_id"] = keys.name
        pbar = ProgressBar(total=size, show_progress=self.show_progress, **pbar_kwargs)

        if self.hide_inner_progress:
            inner_progress_context = ProgressHidden()
        else:
            inner_progress_context = nullcontext()
        with inner_progress_context:
            with Pool(**self.init_kwargs) as pool:
                async_results = []
                for func, args, kwargs in tasks:
                    async_result = pool.apipe(func, *args, **kwargs)
                    async_results.append(async_result)
                if self.timeout is not None or self.show_progress:
                    pending = set(async_results)
                    total_futures = len(pending)
                    if self.timeout is not None:
                        end_time = self.timeout + time.monotonic()
                    with pbar:
                        while pending:
                            pending = {
                                async_result for async_result in pending if not async_result.ready()
                            }
                            pbar.update_to(total_futures - len(pending))
                            if len(pending) == 0:
                                break
                            if self.timeout is not None:
                                if time.monotonic() > end_time:
                                    raise TimeoutError(
                                        "%d (of %d) futures unfinished"
                                        % (len(pending), total_futures)
                                    )
                            if self.check_delay is not None:
                                time.sleep(self.check_delay)
                if self.join_pool:
                    pool.close()
                    pool.join()
                    pool.clear()
                return [async_result.get() for async_result in async_results]


class MpireEngine(ExecutionEngine):
    """Class for executing functions using `WorkerPool` from `mpire`.

    Args:
        init_kwargs (KwargsLike): Keyword arguments used for initializing `mpire.WorkerPool`.
        apply_kwargs (KwargsLike): Keyword arguments for `mpire.WorkerPool.async_apply`.
        hide_inner_progress (Optional[bool]): Flag indicating whether to hide progress bars
            within individual threads.
        **kwargs: Keyword arguments for `ExecutionEngine`.

    !!! info
        For default settings, see `engines.mpire` in `vectorbtpro._settings.execution`.
    """

    _settings_path: tp.SettingsPath = "execution.engines.mpire"

    def __init__(
        self,
        init_kwargs: tp.KwargsLike = None,
        apply_kwargs: tp.KwargsLike = None,
        hide_inner_progress: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            apply_kwargs=apply_kwargs,
            hide_inner_progress=hide_inner_progress,
            **kwargs,
        )

        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._apply_kwargs = self.resolve_setting(apply_kwargs, "apply_kwargs", merge=True)
        self._hide_inner_progress = self.resolve_setting(hide_inner_progress, "hide_inner_progress")

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for initializing `mpire.WorkerPool`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._init_kwargs

    @property
    def apply_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `mpire.WorkerPool.async_apply`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._apply_kwargs

    @property
    def hide_inner_progress(self) -> bool:
        """Flag indicating whether inner progress bars are hidden.

        Returns:
            bool: True if inner progress bars are hidden, False otherwise.
        """
        return self._hide_inner_progress

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("mpire")
        from mpire import WorkerPool

        if self.hide_inner_progress:
            inner_progress_context = ProgressHidden()
        else:
            inner_progress_context = nullcontext()
        with inner_progress_context:
            with WorkerPool(**self.init_kwargs) as pool:
                async_results = []
                for i, (func, args, kwargs) in enumerate(tasks):
                    async_result = pool.apply_async(
                        func, args=args, kwargs=kwargs, **self.apply_kwargs
                    )
                    async_results.append(async_result)
                pool.stop_and_join()
                return [async_result.get() for async_result in async_results]


class DaskEngine(ExecutionEngine):
    """Class for executing functions in parallel using Dask.

    Args:
        compute_kwargs (KwargsLike): Keyword arguments for `dask.compute`.
        hide_inner_progress (Optional[bool]): Flag indicating whether to hide progress bars
            within individual threads.
        **kwargs: Keyword arguments for `ExecutionEngine`.

    !!! info
        For default settings, see `engines.dask` in `vectorbtpro._settings.execution`.

    !!! note
        Use multi-threading primarily for numeric code that releases the GIL
        (e.g., NumPy, Pandas, Scikit-Learn, Numba).
    """

    _settings_path: tp.SettingsPath = "execution.engines.dask"

    def __init__(
        self,
        compute_kwargs: tp.KwargsLike = None,
        hide_inner_progress: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            compute_kwargs=compute_kwargs,
            hide_inner_progress=hide_inner_progress,
            **kwargs,
        )

        self._compute_kwargs = self.resolve_setting(compute_kwargs, "compute_kwargs", merge=True)
        self._hide_inner_progress = self.resolve_setting(hide_inner_progress, "hide_inner_progress")

    @property
    def compute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `dask.compute`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._compute_kwargs

    @property
    def hide_inner_progress(self) -> bool:
        """Flag indicating whether progress bars should be hidden within each thread.

        Returns:
            bool: True if inner progress bars are hidden, False otherwise.
        """
        return self._hide_inner_progress

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("dask")
        import dask

        if self.hide_inner_progress:
            inner_progress_context = ProgressHidden()
        else:
            inner_progress_context = nullcontext()
        with inner_progress_context:
            results_delayed = []
            for func, args, kwargs in tasks:
                results_delayed.append(dask.delayed(func)(*args, **kwargs))
            return list(dask.compute(*results_delayed, **self.compute_kwargs))


class RayEngine(ExecutionEngine):
    """Class for executing functions in parallel using Ray.

    Args:
        restart (Optional[bool]): Flag to determine if the Ray runtime should be terminated and reinitialized.
        reuse_refs (Optional[bool]): Flag indicating if function and object references should be reused
            such that each unique object is stored only once.
        del_refs (Optional[bool]): Flag indicating if result object references should be explicitly deleted.
        shutdown (Optional[bool]): Flag indicating if the Ray runtime should be shut down upon job completion.
        init_kwargs (KwargsLike): Keyword arguments for `ray.init`.
        remote_kwargs (KwargsLike): Keyword arguments for `ray.remote`.
        hide_inner_progress (Optional[bool]): Flag indicating whether to hide progress bars
            within individual threads.
        **kwargs: Keyword arguments for `ExecutionEngine`.

    !!! info
        For default settings, see `engines.ray` in `vectorbtpro._settings.execution`.

    !!! note
        Ray spawns multiple processes rather than threads, so each argument and keyword argument must
        first be stored in an object store to be shared. Ensure that the computation with `func` takes
        sufficient time compared to the overhead of copying; otherwise, little to no speedup will be achieved.
    """

    _settings_path: tp.SettingsPath = "execution.engines.ray"

    def __init__(
        self,
        restart: tp.Optional[bool] = None,
        reuse_refs: tp.Optional[bool] = None,
        del_refs: tp.Optional[bool] = None,
        shutdown: tp.Optional[bool] = None,
        init_kwargs: tp.KwargsLike = None,
        remote_kwargs: tp.KwargsLike = None,
        hide_inner_progress: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            restart=restart,
            reuse_refs=reuse_refs,
            del_refs=del_refs,
            shutdown=shutdown,
            init_kwargs=init_kwargs,
            remote_kwargs=remote_kwargs,
            hide_inner_progress=hide_inner_progress,
            **kwargs,
        )

        self._restart = self.resolve_setting(restart, "restart")
        self._reuse_refs = self.resolve_setting(reuse_refs, "reuse_refs")
        self._del_refs = self.resolve_setting(del_refs, "del_refs")
        self._shutdown = self.resolve_setting(shutdown, "shutdown")
        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._remote_kwargs = self.resolve_setting(remote_kwargs, "remote_kwargs", merge=True)
        self._hide_inner_progress = self.resolve_setting(hide_inner_progress, "hide_inner_progress")

    @property
    def restart(self) -> bool:
        """Flag indicating if the Ray runtime should be terminated and reinitialized.

        Returns:
            bool: True if the Ray runtime is restarted, False otherwise.
        """
        return self._restart

    @property
    def reuse_refs(self) -> bool:
        """Flag indicating if function and object references are reused so that each
        unique object is stored only once.

        Returns:
            bool: True if references are reused, False otherwise.
        """
        return self._reuse_refs

    @property
    def del_refs(self) -> bool:
        """Flag indicating if result object references should be explicitly deleted.

        Returns:
            bool: True if result references are deleted, False otherwise.
        """
        return self._del_refs

    @property
    def shutdown(self) -> bool:
        """Flag indicating if the Ray runtime should be shut down upon job completion.

        Returns:
            bool: True if the Ray runtime is shut down, False otherwise.
        """
        return self._shutdown

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `ray.init`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._init_kwargs

    @property
    def remote_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `ray.remote`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._remote_kwargs

    @property
    def hide_inner_progress(self) -> bool:
        """Flag indicating if progress bars should be hidden within each thread.

        Returns:
            bool: True if inner progress bars are hidden, False otherwise.
        """
        return self._hide_inner_progress

    @classmethod
    def get_ray_refs(
        cls,
        tasks: tp.TasksLike,
        reuse_refs: bool = True,
        remote_kwargs: tp.KwargsLike = None,
    ) -> tp.List[tp.Tuple[RemoteFunctionT, tp.Tuple[ObjectRefT, ...], tp.Dict[str, ObjectRefT]]]:
        """Obtain Ray remote function references by storing arguments in the object store and
        applying `ray.remote` on functions.

        Args:
            tasks (TasksLike): Tasks (i.e., functions with their arguments) to execute.
            reuse_refs (bool): Flag indicating whether to reuse references for unique objects.
            remote_kwargs (KwargsLike): Keyword arguments for `ray.remote`.

        Returns:
            List[Tuple[RemoteFunction, Tuple[ObjectRef, ...], Dict[str, ObjectRef]]]:
                List of tuples, each containing a remote function reference, a tuple of object references
                for positional arguments, and a dictionary mapping keyword argument names to object references.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ray")
        import ray
        from ray import ObjectRef
        from ray.remote_function import RemoteFunction

        if remote_kwargs is None:
            remote_kwargs = {}

        func_id_remotes = {}
        obj_id_refs = {}
        task_refs = []
        for func, args, kwargs in tasks:
            # Get remote function
            if isinstance(func, RemoteFunction):
                func_remote = func
            else:
                if not reuse_refs or id(func) not in func_id_remotes:
                    if isinstance(func, CPUDispatcher):
                        # Numba-wrapped function is not recognized by ray as a function
                        _func = lambda *_args, **_kwargs: func(*_args, **_kwargs)
                    else:
                        _func = func
                    if len(remote_kwargs) > 0:
                        func_remote = ray.remote(**remote_kwargs)(_func)
                    else:
                        func_remote = ray.remote(_func)
                    if reuse_refs:
                        func_id_remotes[id(func)] = func_remote
                else:
                    func_remote = func_id_remotes[id(func)]

            # Get id of each (unique) arg
            arg_refs = ()
            for arg in args:
                if isinstance(arg, ObjectRef):
                    arg_ref = arg
                else:
                    if not reuse_refs or id(arg) not in obj_id_refs:
                        arg_ref = ray.put(arg)
                        obj_id_refs[id(arg)] = arg_ref
                    else:
                        arg_ref = obj_id_refs[id(arg)]
                arg_refs += (arg_ref,)

            # Get id of each (unique) kwarg
            kwarg_refs = {}
            for kwarg_name, kwarg in kwargs.items():
                if isinstance(kwarg, ObjectRef):
                    kwarg_ref = kwarg
                else:
                    if not reuse_refs or id(kwarg) not in obj_id_refs:
                        kwarg_ref = ray.put(kwarg)
                        obj_id_refs[id(kwarg)] = kwarg_ref
                    else:
                        kwarg_ref = obj_id_refs[id(kwarg)]
                kwarg_refs[kwarg_name] = kwarg_ref

            task_refs.append((func_remote, arg_refs, kwarg_refs))
        return task_refs

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ray")
        import ray

        if self.hide_inner_progress:
            inner_progress_context = ProgressHidden()
        else:
            inner_progress_context = nullcontext()
        with inner_progress_context:
            if self.restart:
                if ray.is_initialized():
                    ray.shutdown()
            if not ray.is_initialized():
                ray.init(**self.init_kwargs)
            task_refs = self.get_ray_refs(
                tasks, reuse_refs=self.reuse_refs, remote_kwargs=self.remote_kwargs
            )
            result_refs = []
            for func_remote, arg_refs, kwarg_refs in task_refs:
                result_refs.append(func_remote.remote(*arg_refs, **kwarg_refs))
            try:
                results = ray.get(result_refs)
            finally:
                if self.del_refs:
                    # clear object store
                    del result_refs
                if self.shutdown:
                    ray.shutdown()
            return results


class _Dummy(enum.Enum):
    """Enum class representing a dummy sentinel value."""

    DUMMY = enum.auto()

    def __repr__(self):
        return "DUMMY"

    def __bool__(self):
        return False


DUMMY = _Dummy.DUMMY
"""Sentinel representing a missing value."""


class Executor(Configured):
    """Class for executing functions using configurable execution engines and flexible task distribution.

    The supported values for `engine` include:

    * Name of the engine: Specifies the engine by name (see supported engines).
    * Subclass of `ExecutionEngine`: Instantiated with `engine_config`.
    * Instance of `ExecutionEngine`: Uses its `execute` method with `size`.
    * Callable: Invoked with `tasks`, `size` (if provided), and `engine_config`.

    Execution supports chunking. If `chunk_meta` is provided, tasks are executed per chunk.
    Otherwise, if either `n_chunks` or `chunk_len` is set, they are passed to
    `vectorbtpro.utils.chunking.iter_chunk_meta` to generate `chunk_meta`. Global defaults for
    `n_chunks` and `chunk_len` can be configured in engine-specific settings; they may also be set to
    "auto" to match the number of cores.

    For task distribution:

    * When `distribute` is "tasks", tasks within each chunk are distributed. If `tasks` is an iterable
        and `chunk_meta` indices are sorted, iteration occurs directly over `tasks`; otherwise, iteration
        occurs over `chunk_meta`. If `in_chunk_order` is True, results follow the order in `chunk_meta`;
        otherwise, they follow the order in `tasks`.
    * When `distribute` is "chunks", chunks are distributed, and tasks within each chunk are executed
        serially via `Executor.execute_serially`. Each chunk is compressed so that each unique function and
        its arguments are serialized only once.

    If `tasks` is a custom template, it is substituted once `chunk_meta` is established using
    `template_context`. All resolved functions and arguments are then passed to the executor.

    Optional callbacks:

    * `pre_chunk_func`: Called before processing a chunk. If it returns a non-None value, that value is
        appended to the results and the chunk is not executed. This behavior facilitates caching.
    * `post_chunk_func`: Called after processing a chunk. Should return either None to keep previous
        results or new execution results. Templates in `pre_chunk_kwargs` and `post_chunk_kwargs` are
        substituted and passed as keyword arguments.

    !!! note
        Both callbacks are effective only when `distribute` is "tasks" and chunking is enabled.

    Additional context includes:

    * `chunk_idx`: Current chunk index.
    * `call_indices`: List of call indices in the chunk.
    * `chunk_cache`: Call results from caching (for `pre_chunk_func`).
    * `call_results`: Execution results (for `post_chunk_func`).
    * `chunk_executed`: Indicates if the chunk was executed (for `post_chunk_func`).

    Optional overall execution callbacks:

    * `pre_execute_func`: Called before processing all tasks. Must return None.
        Receives, among others, the number of chunks (`n_chunks`) in its context.
    * `post_execute_func`: Called after processing all tasks. Should return either None to retain default
        results or new results. Its context includes the number of chunks (`n_chunks`) and the flattened
        list of results (`results`). If `post_execute_on_sorted` is True, this callback is executed after
        sorting the call indices.

    Args:
        engine (Optional[ExecutionEngineLike]): Execution engine.
        engine_config (KwargsLike): Configuration for the execution engine.
        min_size (Optional[int]): Minimum number of elements to split.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        n_chunks (Union[None, int, str]): Specification for the number of chunks.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        chunk_len (Union[None, int, str]): Specification for the length of each chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        chunk_meta (Optional[Iterable[ChunkMeta]]): Iterable containing metadata for each chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        distribute (Optional[str]): Distribution mode.

            * "tasks": Distributes tasks within each chunk.
            * "chunks": Distributes chunks.
        warmup (Optional[bool]): Flag indicating whether to execute the first task
            as a warmup before distribution.

            This is useful for engines that require a warmup run to optimize performance.
        in_chunk_order (Optional[bool]): Flag that determines whether results are returned
            in the order specified by `chunk_meta`.

            Otherwise, results follow the order of `tasks`.
        cache_chunks (Optional[bool]): Flag indicating whether chunks should be cached.
        chunk_cache_dir (Optional[PathLike]): Directory for storing chunk cache files.
        chunk_cache_save_kwargs (KwargsLike): Keyword arguments for saving chunk cache.

            See `vectorbtpro.utils.pickling.save`.
        chunk_cache_load_kwargs (KwargsLike): Keyword arguments for loading chunk cache.

            See `vectorbtpro.utils.pickling.load`.
        pre_clear_chunk_cache (Optional[bool]): Flag indicating if the chunk cache directory should
            be removed before execution.
        post_clear_chunk_cache (Optional[bool]): Flag indicating if the chunk cache directory should
            be removed after execution.
        release_chunk_cache (Optional[bool]): Flag that replaces the chunk cache with dummy objects
            after execution and loads the full cache after all chunks complete.
        chunk_clear_cache (Union[None, bool, int]): Specifies whether the global cache should
            be cleared after each chunk or every n chunks.
        chunk_collect_garbage (Union[None, bool, int]): Specifies whether garbage collection should
            be performed after each chunk or every n chunks.
        chunk_delay (Optional[float]): Delay in seconds after processing each chunk.
        pre_execute_func (Optional[Callable]): Function to be called before executing all tasks.
        pre_execute_kwargs (KwargsLike): Keyword arguments for `pre_execute_func`.
        pre_chunk_func (Optional[Callable]): Function to be called before executing each chunk.

            If the callable returns a value other than None, its return value is appended to
            the results and the chunk is skipped.
        pre_chunk_kwargs (KwargsLike): Keyword arguments for `pre_chunk_func`.
        post_chunk_func (Optional[Callable]): Function to be called after executing each chunk.

            If it returns None, the existing chunk results are retained; otherwise, its return
            value replaces them.
        post_chunk_kwargs (KwargsLike): Keyword arguments for `post_chunk_func`.
        post_execute_func (Optional[Callable]): Function to be called after executing all tasks.

            If it returns None, the default results are preserved; otherwise, its return value replaces them.
        post_execute_kwargs (KwargsLike): Keyword arguments for `post_execute_func`.
        post_execute_on_sorted (Optional[bool]): Flag indicating whether `post_execute_func` should
            be invoked after sorting call indices.
        filter_results (Optional[bool]): Flag indicating whether to filter
            `NoResult` results after execution.
        raise_no_results (Optional[bool]): Flag indicating whether to raise a
                `NoResultsException` exception if no results remain.

            This flag applies only when `filter_results` is True and is forwarded to the merging
            function if pre-configured.
        merge_func (MergeFuncLike): Function to merge the results.

            See `vectorbtpro.utils.merging.MergeFunc`.
        merge_kwargs (KwargsLike): Keyword arguments for `merge_func`.
        template_context (KwargsLike): Additional context for template substitution.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.

            If `engine` accepts `show_progress` and the key is absent in `engine_config`,
            it is forwarded to the engine.
        pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

            See `vectorbtpro.utils.pbar.ProgressBar`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.execution`.
    """

    _settings_path: tp.SettingsPath = "execution"

    def __init__(
        self,
        engine: tp.Optional[tp.ExecutionEngineLike] = None,
        engine_config: tp.KwargsLike = None,
        min_size: tp.Optional[int] = None,
        n_chunks: tp.Union[None, int, str] = None,
        chunk_len: tp.Union[None, int, str] = None,
        chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
        distribute: tp.Optional[str] = None,
        warmup: tp.Optional[bool] = None,
        in_chunk_order: tp.Optional[bool] = None,
        cache_chunks: tp.Optional[bool] = None,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        chunk_cache_save_kwargs: tp.KwargsLike = None,
        chunk_cache_load_kwargs: tp.KwargsLike = None,
        pre_clear_chunk_cache: tp.Optional[bool] = None,
        post_clear_chunk_cache: tp.Optional[bool] = None,
        release_chunk_cache: tp.Optional[bool] = None,
        chunk_clear_cache: tp.Union[None, bool, int] = None,
        chunk_collect_garbage: tp.Union[None, bool, int] = None,
        chunk_delay: tp.Optional[float] = None,
        pre_execute_func: tp.Optional[tp.Callable] = None,
        pre_execute_kwargs: tp.KwargsLike = None,
        pre_chunk_func: tp.Optional[tp.Callable] = None,
        pre_chunk_kwargs: tp.KwargsLike = None,
        post_chunk_func: tp.Optional[tp.Callable] = None,
        post_chunk_kwargs: tp.KwargsLike = None,
        post_execute_func: tp.Optional[tp.Callable] = None,
        post_execute_kwargs: tp.KwargsLike = None,
        post_execute_on_sorted: tp.Optional[bool] = None,
        filter_results: tp.Optional[bool] = None,
        raise_no_results: tp.Optional[bool] = None,
        merge_func: tp.MergeFuncLike = None,
        merge_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            engine=engine,
            engine_config=engine_config,
            min_size=min_size,
            n_chunks=n_chunks,
            chunk_len=chunk_len,
            chunk_meta=chunk_meta,
            distribute=distribute,
            warmup=warmup,
            in_chunk_order=in_chunk_order,
            cache_chunks=cache_chunks,
            chunk_cache_dir=chunk_cache_dir,
            chunk_cache_save_kwargs=chunk_cache_save_kwargs,
            chunk_cache_load_kwargs=chunk_cache_load_kwargs,
            pre_clear_chunk_cache=pre_clear_chunk_cache,
            post_clear_chunk_cache=post_clear_chunk_cache,
            release_chunk_cache=release_chunk_cache,
            chunk_clear_cache=chunk_clear_cache,
            chunk_collect_garbage=chunk_collect_garbage,
            chunk_delay=chunk_delay,
            pre_execute_func=pre_execute_func,
            pre_execute_kwargs=pre_execute_kwargs,
            pre_chunk_func=pre_chunk_func,
            pre_chunk_kwargs=pre_chunk_kwargs,
            post_chunk_func=post_chunk_func,
            post_chunk_kwargs=post_chunk_kwargs,
            post_execute_func=post_execute_func,
            post_execute_kwargs=post_execute_kwargs,
            post_execute_on_sorted=post_execute_on_sorted,
            filter_results=filter_results,
            raise_no_results=raise_no_results,
            merge_func=merge_func,
            merge_kwargs=merge_kwargs,
            template_context=template_context,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            **kwargs,
        )

        if engine_config is None:
            engine_config = {}
        engine, engine_name = self.resolve_engine(
            engine,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            **engine_config,
        )
        min_size = self.resolve_engine_setting(
            min_size,
            "min_size",
            engine_name=engine_name,
        )
        n_chunks = self.resolve_engine_setting(
            n_chunks,
            "n_chunks",
            engine_name=engine_name,
        )
        chunk_len = self.resolve_engine_setting(
            chunk_len,
            "chunk_len",
            engine_name=engine_name,
        )
        chunk_meta = self.resolve_engine_setting(
            chunk_meta,
            "chunk_meta",
            engine_name=engine_name,
        )
        distribute = self.resolve_engine_setting(
            distribute,
            "distribute",
            engine_name=engine_name,
        )
        warmup = self.resolve_engine_setting(
            warmup,
            "warmup",
            engine_name=engine_name,
        )
        in_chunk_order = self.resolve_engine_setting(
            in_chunk_order,
            "in_chunk_order",
            engine_name=engine_name,
        )
        cache_chunks = self.resolve_engine_setting(
            cache_chunks,
            "cache_chunks",
            engine_name=engine_name,
        )
        chunk_cache_dir = self.resolve_engine_setting(
            chunk_cache_dir,
            "chunk_cache_dir",
            engine_name=engine_name,
        )
        chunk_cache_save_kwargs = self.resolve_engine_setting(
            chunk_cache_save_kwargs,
            "chunk_cache_save_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        chunk_cache_load_kwargs = self.resolve_engine_setting(
            chunk_cache_load_kwargs,
            "chunk_cache_load_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        pre_clear_chunk_cache = self.resolve_engine_setting(
            pre_clear_chunk_cache,
            "pre_clear_chunk_cache",
            engine_name=engine_name,
        )
        post_clear_chunk_cache = self.resolve_engine_setting(
            post_clear_chunk_cache,
            "post_clear_chunk_cache",
            engine_name=engine_name,
        )
        release_chunk_cache = self.resolve_engine_setting(
            release_chunk_cache,
            "release_chunk_cache",
            engine_name=engine_name,
        )
        chunk_clear_cache = self.resolve_engine_setting(
            chunk_clear_cache,
            "chunk_clear_cache",
            engine_name=engine_name,
        )
        chunk_collect_garbage = self.resolve_engine_setting(
            chunk_collect_garbage,
            "chunk_collect_garbage",
            engine_name=engine_name,
        )
        chunk_delay = self.resolve_engine_setting(
            chunk_delay,
            "chunk_delay",
            engine_name=engine_name,
        )
        pre_execute_func = self.resolve_engine_setting(
            pre_execute_func,
            "pre_execute_func",
            engine_name=engine_name,
        )
        pre_execute_kwargs = self.resolve_engine_setting(
            pre_execute_kwargs,
            "pre_execute_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        pre_chunk_func = self.resolve_engine_setting(
            pre_chunk_func,
            "pre_chunk_func",
            engine_name=engine_name,
        )
        pre_chunk_kwargs = self.resolve_engine_setting(
            pre_chunk_kwargs,
            "pre_chunk_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        post_chunk_func = self.resolve_engine_setting(
            post_chunk_func,
            "post_chunk_func",
            engine_name=engine_name,
        )
        post_chunk_kwargs = self.resolve_engine_setting(
            post_chunk_kwargs,
            "post_chunk_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        post_execute_func = self.resolve_engine_setting(
            post_execute_func,
            "post_execute_func",
            engine_name=engine_name,
        )
        post_execute_kwargs = self.resolve_engine_setting(
            post_execute_kwargs,
            "post_execute_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        post_execute_on_sorted = self.resolve_engine_setting(
            post_execute_on_sorted,
            "post_execute_on_sorted",
            engine_name=engine_name,
        )
        if release_chunk_cache and post_execute_on_sorted:
            raise ValueError("Cannot use release_chunk_cache and post_execute_on_sorted together")
        filter_results = self.resolve_engine_setting(filter_results, "filter_results")
        raise_no_results = self.resolve_engine_setting(raise_no_results, "raise_no_results")
        merge_func = self.resolve_engine_setting(merge_func, "merge_func")
        merge_kwargs = self.resolve_engine_setting(merge_kwargs, "merge_kwargs", merge=True)
        template_context = self.resolve_engine_setting(
            template_context,
            "template_context",
            merge=True,
            engine_name=engine_name,
        )
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        self._engine = engine
        self._min_size = min_size
        self._n_chunks = n_chunks
        self._chunk_len = chunk_len
        self._chunk_meta = chunk_meta
        self._distribute = distribute
        self._warmup = warmup
        self._in_chunk_order = in_chunk_order
        self._cache_chunks = cache_chunks
        self._chunk_cache_dir = chunk_cache_dir
        self._chunk_cache_save_kwargs = chunk_cache_save_kwargs
        self._chunk_cache_load_kwargs = chunk_cache_load_kwargs
        self._pre_clear_chunk_cache = pre_clear_chunk_cache
        self._post_clear_chunk_cache = post_clear_chunk_cache
        self._release_chunk_cache = release_chunk_cache
        self._chunk_clear_cache = chunk_clear_cache
        self._chunk_collect_garbage = chunk_collect_garbage
        self._chunk_delay = chunk_delay
        self._pre_execute_func = pre_execute_func
        self._pre_execute_kwargs = pre_execute_kwargs
        self._pre_chunk_func = pre_chunk_func
        self._pre_chunk_kwargs = pre_chunk_kwargs
        self._post_chunk_func = post_chunk_func
        self._post_chunk_kwargs = post_chunk_kwargs
        self._post_execute_func = post_execute_func
        self._post_execute_kwargs = post_execute_kwargs
        self._post_execute_on_sorted = post_execute_on_sorted
        self._filter_results = filter_results
        self._raise_no_results = raise_no_results
        self._merge_func = merge_func
        self._merge_kwargs = merge_kwargs
        self._template_context = template_context
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs

    @property
    def engine(self) -> tp.Union[ExecutionEngine, tp.Callable]:
        """Engine resolved using `Executor.resolve_engine`.

        Returns:
            Union[ExecutionEngine, Callable]: Configured execution engine.
        """
        return self._engine

    @property
    def min_size(self) -> tp.Optional[int]:
        """Minimum number of elements to split, as defined in `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            Optional[int]: Minimum chunk size, if provided; otherwise, None.
        """
        return self._min_size

    @property
    def n_chunks(self) -> tp.Union[None, int, str]:
        """Specification for the number of chunks, as defined in `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            Union[None, int, str]: Number of chunks or mode, if provided; otherwise, None.
        """
        return self._n_chunks

    @property
    def chunk_len(self) -> tp.Union[None, int, str]:
        """Specification for the length of each chunk, as defined in `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            Union[None, int, str]: Chunk length or mode, if provided; otherwise, None.
        """
        return self._chunk_len

    @property
    def chunk_meta(self) -> tp.Optional[tp.ChunkMetaLike]:
        """Metadata for chunks, as defined in `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            Optional[ChunkMetaLike]: Metadata for chunks, if provided; otherwise, None.
        """
        return self._chunk_meta

    @property
    def distribute(self) -> str:
        """Distribution mode.

        * "tasks": Distributes tasks within each chunk.
        * "chunks": Distributes chunks.

        Returns:
            str: Distribution mode.
        """
        return self._distribute

    @property
    def warmup(self) -> bool:
        """Flag indicating whether to execute the first task as a warmup before distribution.

        This is useful for engines that require a warmup run to optimize performance.

        Returns:
            bool: True if warmup is enabled, False otherwise.
        """
        return self._warmup

    @property
    def in_chunk_order(self) -> bool:
        """Flag that determines whether results are returned in the order specified by `chunk_meta`.

        Otherwise, results follow the order of `tasks`.

        Returns:
            bool: True if results are in chunk order, False otherwise.
        """
        return self._in_chunk_order

    @property
    def cache_chunks(self) -> bool:
        """Flag indicating whether chunks should be cached.

        Returns:
            bool: True if chunks are cached, False otherwise.
        """
        return self._cache_chunks

    @property
    def chunk_cache_dir(self) -> tp.PathLike:
        """Directory for storing chunk cache files.

        Returns:
            PathLike: Directory for chunk cache files.
        """
        return self._chunk_cache_dir

    @property
    def chunk_cache_save_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `vectorbtpro.utils.pickling.save` during chunk caching.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._chunk_cache_save_kwargs

    @property
    def chunk_cache_load_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `vectorbtpro.utils.pickling.load` during chunk caching.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._chunk_cache_load_kwargs

    @property
    def pre_clear_chunk_cache(self) -> bool:
        """Flag indicating if the chunk cache directory should be removed before execution.

        Returns:
            bool: True if the chunk cache directory is removed before execution, False otherwise.
        """
        return self._pre_clear_chunk_cache

    @property
    def post_clear_chunk_cache(self) -> bool:
        """Flag indicating if the chunk cache directory should be removed after execution.

        Returns:
            bool: True if the chunk cache directory should be removed after execution, False otherwise.
        """
        return self._post_clear_chunk_cache

    @property
    def release_chunk_cache(self) -> bool:
        """Flag that replaces the chunk cache with dummy objects after execution and loads
        the full cache after all chunks complete.

        Returns:
            bool: True if the chunk cache is replaced with dummy objects, False otherwise.
        """
        return self._release_chunk_cache

    @property
    def chunk_clear_cache(self) -> tp.Union[bool, int]:
        """Specifies whether the global cache should be cleared after each chunk
        or every n chunks.

        Returns:
            Union[bool, int]: Number of chunks after which the cache is cleared (True for every chunk).
        """
        return self._chunk_clear_cache

    @property
    def chunk_collect_garbage(self) -> tp.Union[bool, int]:
        """Specifies whether garbage collection should be performed after each
        chunk or every n chunks.

        Returns:
            Union[bool, int]: Number of chunks after which garbage collection is performed (True for every chunk).
        """
        return self._chunk_collect_garbage

    @property
    def chunk_delay(self) -> tp.Optional[float]:
        """Delay in seconds after processing each chunk.

        Returns:
            Optional[float]: Delay in seconds after processing each chunk; None if not set.
        """
        return self._chunk_delay

    @property
    def pre_execute_func(self) -> tp.Optional[tp.Callable]:
        """Callable to be executed before processing all tasks.

        Returns:
            Optional[Callable]: Callable to be executed before processing all tasks; None if not set.
        """
        return self._pre_execute_func

    @property
    def pre_execute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `Executor.pre_execute_func`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._pre_execute_kwargs

    @property
    def pre_chunk_func(self) -> tp.Optional[tp.Callable]:
        """Callable executed before processing each chunk.

        If the callable returns a value other than None, its return value is appended to
        the results and the chunk is skipped.

        Returns:
            Optional[Callable]: Callable executed before processing each chunk; None if not set.
        """
        return self._pre_chunk_func

    @property
    def pre_chunk_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `Executor.pre_chunk_func`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._pre_chunk_kwargs

    @property
    def post_chunk_func(self) -> tp.Optional[tp.Callable]:
        """Callable executed after processing each chunk.

        If it returns None, the existing chunk results are retained; otherwise, its return
        value replaces them.

        Returns:
            Optional[Callable]: Callable executed after processing each chunk; None if not set.
        """
        return self._post_chunk_func

    @property
    def post_chunk_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `Executor.post_chunk_func`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._post_chunk_kwargs

    @property
    def post_execute_func(self) -> tp.Optional[tp.Callable]:
        """Callable executed after processing all tasks.

        If it returns None, the default results are preserved; otherwise, its return value replaces them.

        Returns:
            Optional[Callable]: Callable executed after processing all tasks; None if not set.
        """
        return self._post_execute_func

    @property
    def post_execute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `Executor.post_execute_func`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._post_execute_kwargs

    @property
    def post_execute_on_sorted(self) -> bool:
        """Flag indicating whether `Executor.post_execute_func` should be invoked
        after sorting call indices.

        Returns:
            bool: True if `post_execute_func` is invoked after sorting call indices, False otherwise.
        """
        return self._post_execute_on_sorted

    @property
    def filter_results(self) -> bool:
        """Flag determining if results equal to `NoResult` should be filtered out.

        Returns:
            bool: True if results are filtered, False otherwise.
        """
        return self._filter_results

    @property
    def raise_no_results(self) -> bool:
        """Flag indicating if a `NoResultsException` should be raised when no results are obtained.

        This flag applies only when `Executor.filter_results` is True and is forwarded to the merging
        function if pre-configured.

        Returns:
            bool: True if a `NoResultsException` is raised, False otherwise.
        """
        return self._raise_no_results

    @property
    def merge_func(self) -> tp.Optional[tp.MergeFuncLike]:
        """Function to merge the results.

        See `vectorbtpro.utils.merging.MergeFunc`.

        Returns:
            Optional[MergeFuncLike]: Callable for merging results; None if not set.
        """
        return self._merge_func

    @property
    def merge_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `Executor.merge_func`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._merge_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def show_progress(self) -> bool:
        """Flag that determines whether to display a progress bar when iterating over chunks.

        If `Executor.engine` accepts `show_progress` and the key is absent in `engine_config`,
        it is forwarded to the engine.

        Returns:
            bool: True if a progress bar is displayed, False otherwise.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for configuring the progress bar.

        See `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Configuration keyword arguments.
        """
        return self._pbar_kwargs

    @classmethod
    def get_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> dict:
        """Return engine-specific settings using the engine name as a subpath.

        Positional arguments are passed to `Executor.get_settings`.

        Args:
            *args: Positional arguments for `Executor.get_settings`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `Executor.get_settings`.

        Returns:
            dict: Settings for the specified engine.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.get_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> bool:
        """Return True if engine-specific settings exist using the given engine name as a subpath.

        Positional arguments are passed to `Executor.has_settings`.

        Args:
            *args: Positional arguments for `Executor.has_settings`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `Executor.has_settings`.

        Returns:
            bool: True if engine-specific settings exist, False otherwise.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.has_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def get_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """Return a specific engine setting using the engine name as a subpath.

        Positional arguments are passed to `Executor.get_setting`.

        Args:
            *args: Positional arguments for `Executor.get_setting`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `Executor.get_setting`.

        Returns:
            Any: Value of the specified engine setting.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.get_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> bool:
        """Return True if a specific engine setting exists using the given engine name as a subpath.

        Positional arguments are passed to `Executor.has_setting`.

        Args:
            *args: Positional arguments for `Executor.has_setting`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `Executor.has_setting`.

        Returns:
            bool: True if the engine setting exists, False otherwise.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.has_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_engine_setting(
        cls, *args, engine_name: tp.Optional[str] = None, **kwargs
    ) -> tp.Any:
        """Return a resolved engine setting using the engine name as a subpath.

        Positional arguments are passed to `Executor.resolve_setting`.

        Args:
            *args: Positional arguments for `Executor.resolve_setting`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `Executor.resolve_setting`.

        Returns:
            Any: Resolved engine setting.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.resolve_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def set_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> None:
        """Set engine-specific settings using the engine name as a subpath.

        Positional arguments are passed to `Executor.set_settings`.

        Args:
            *args: Positional arguments for `Executor.set_settings`.
            engine_name (Optional[str]): Name of the engine for retrieving custom settings.
            **kwargs: Keyword arguments for `Executor.set_settings`.

        Returns:
            None: Function modifies settings in place.
        """
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        cls.set_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_engine(
        cls,
        engine: tp.ExecutionEngineLike,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **engine_config,
    ) -> tp.Tuple[tp.Union[ExecutionEngine, tp.Callable], tp.Optional[str]]:
        """Resolve the engine based on the provided configuration and return the engine along with its name.

        This method determines the execution engine from various types including:

        * String representing the engine name.
        * Subclass or an instance of `ExecutionEngine`.
        * Callable that processes tasks.

        It applies additional configuration such as `show_progress` and `pbar_kwargs` if supported
        by the engine. If the engine is a subclass of `ExecutionEngine`, it is instantiated with
        the given `engine_config`. If the engine is an instance, it is replaced with updated configuration.
        For callables, the engine name is inferred from available settings.

        Args:
            engine (ExecutionEngineLike): Engine specification which can be a string,
                subclass, instance, or callable.
            show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
            pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

                See `vectorbtpro.utils.pbar.ProgressBar`.
            **engine_config: Additional engine configuration parameters.

        Returns:
            Tuple[Union[ExecutionEngine, Callable], Optional[str]]: Tuple containing
                the resolved engine and its name.

        !!! info
            For default settings, see `vectorbtpro._settings.execution`.
        """
        from vectorbtpro._settings import settings

        execution_cfg = settings["execution"]
        engines_cfg = execution_cfg["engines"]

        engine_name = None
        if engine is None:
            engine = execution_cfg["engine"]
        if isinstance(engine, str):
            if engine in engines_cfg:
                engine_name = engine
                engine = engines_cfg[engine_name]["cls"]
            elif engine.lower() in engines_cfg:
                engine_name = engine.lower()
                engine = engines_cfg[engine_name]["cls"]
        if isinstance(engine, str):
            globals_dict = globals()
            if engine in globals_dict:
                engine = globals_dict[engine]
            else:
                raise ValueError(f"Invalid engine name: '{engine}'")
        if isinstance(engine, type) and issubclass(engine, ExecutionEngine):
            if engine_name is None:
                for k, v in engines_cfg.items():
                    if v["cls"] is engine:
                        engine_name = k
            if show_progress is not None or pbar_kwargs is not None:
                func_arg_names = get_func_arg_names(engine.__init__)
                if show_progress is not None:
                    if (
                        "show_progress" in func_arg_names
                        or (engine_name is not None and "show_progress" in engines_cfg[engine_name])
                    ) and "show_progress" not in engine_config:
                        engine_config["show_progress"] = show_progress
                if pbar_kwargs is not None:
                    if (
                        "pbar_kwargs" in func_arg_names
                        or (engine_name is not None and "pbar_kwargs" in engines_cfg[engine_name])
                    ) and "pbar_kwargs" not in engine_config:
                        engine_config["pbar_kwargs"] = pbar_kwargs
            engine = engine(**engine_config)
        if not isinstance(engine, type) and isinstance(engine, ExecutionEngine):
            if engine_name is None:
                for k, v in engines_cfg.items():
                    if v["cls"] is type(engine):
                        engine_name = k
            if len(engine_config) > 0:
                engine = engine.replace(**engine_config)
        if callable(engine):
            if engine_name is None:
                for k, v in engines_cfg.items():
                    if v["cls"] is engine:
                        engine_name = k
            if engine_name is None:
                if engine.__name__ in engines_cfg:
                    engine_name = engine.__name__
            if show_progress is not None or pbar_kwargs is not None:
                func_arg_names = get_func_arg_names(engine)
                if show_progress is not None:
                    if (
                        "show_progress" in func_arg_names
                        or (engine_name is not None and "show_progress" in engines_cfg[engine_name])
                    ) and "show_progress" not in engine_config:
                        engine_config["show_progress"] = show_progress
                if pbar_kwargs is not None:
                    if (
                        "pbar_kwargs" in func_arg_names
                        or (engine_name is not None and "pbar_kwargs" in engines_cfg[engine_name])
                    ) and "pbar_kwargs" not in engine_config:
                        engine_config["pbar_kwargs"] = pbar_kwargs
            engine = partial(engine, **engine_config)
        if not isinstance(engine, ExecutionEngine) and not callable(engine):
            raise TypeError(f"Invalid engine: {engine}")
        return engine, engine_name

    @staticmethod
    def execute_serially(tasks: tp.TasksLike, id_objs: tp.Dict[int, tp.Any]) -> tp.ExecResults:
        """Execute tasks sequentially.

        Iterates over each task, resolves functions and their arguments using `id_objs`,
        and returns a list of results.

        Args:
            tasks (TasksLike): Tasks (i.e., functions with their arguments) to execute.
            id_objs (Dict[int, Any]): Dictionary mapping IDs to objects.

        Returns:
            ExecResults: List of results from executing the tasks.
        """
        results = []
        for func, args, kwargs in tasks:
            new_func = id_objs[func]
            new_args = tuple(id_objs[arg] for arg in args)
            new_kwargs = {k: id_objs[v] for k, v in kwargs.items()}
            results.append(new_func(*new_args, **new_kwargs))
        return results

    @classmethod
    def build_serial_chunk(cls, tasks: tp.TasksLike) -> Task:
        """Construct a serial execution chunk from the provided tasks.

        Args:
            tasks (TasksLike): Tasks (i.e., functions with their arguments) to execute.

        Returns:
            Task: Task object representing the serial execution of the provided tasks.
        """
        ref_ids = dict()
        id_objs = dict()

        def _prepare(x):
            if id(x) in ref_ids:
                return ref_ids[id(x)]
            new_id = len(id_objs)
            ref_ids[id(x)] = new_id
            id_objs[new_id] = x
            return new_id

        new_tasks = []
        for func, args, kwargs in tasks:
            new_func = _prepare(func)
            new_args = tuple(_prepare(arg) for arg in args)
            new_kwargs = {k: _prepare(v) for k, v in kwargs.items()}
            new_tasks.append(Task(new_func, *new_args, **new_kwargs))
        return Task(cls.execute_serially, new_tasks, id_objs)

    @classmethod
    def call_pre_execute_func(
        cls,
        cache_chunks: bool = False,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        pre_clear_chunk_cache: bool = False,
        pre_execute_func: tp.Optional[tp.Callable] = None,
        pre_execute_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> None:
        """Call pre-execution function from `Executor`.

        Args:
            cache_chunks (bool): Flag indicating whether chunk caching is enabled.
            chunk_cache_dir (Optional[PathLike]): Directory for cached chunks; required if cache
                clearing is requested.
            pre_clear_chunk_cache (bool): If True, clear the chunk cache directory before execution.
            pre_execute_func (Optional[Callable]): Function to be called before executing all tasks.
            pre_execute_kwargs (KwargsLike): Keyword arguments for `pre_execute_func`.
            template_context (KwargsLike): Additional context for template substitution.

        Returns:
            None
        """
        if cache_chunks and pre_clear_chunk_cache:
            if chunk_cache_dir is None:
                raise ValueError("Must provide chunk_cache_dir")
            remove_dir(chunk_cache_dir, missing_ok=True, with_contents=True)

        if pre_execute_kwargs is None:
            pre_execute_kwargs = {}
        if pre_execute_func is not None:
            pre_execute_func = substitute_templates(
                pre_execute_func,
                template_context,
                eval_id="pre_execute_func",
            )
            pre_execute_kwargs = substitute_templates(
                pre_execute_kwargs,
                template_context,
                eval_id="pre_execute_kwargs",
            )
            pre_execute_func(**pre_execute_kwargs)

    @classmethod
    def call_pre_chunk_func(
        cls,
        chunk_idx: int,
        call_indices: tp.List[int],
        cache_chunks: bool = False,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        chunk_cache_load_kwargs: tp.KwargsLike = None,
        release_chunk_cache: bool = False,
        pre_chunk_func: tp.Optional[tp.Callable] = None,
        pre_chunk_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Optional[tp.ExecResults]:
        """Call pre-chunk function from `Executor`, handling retrieval of cached chunk data
        and template substitutions.

        Args:
            chunk_idx (int): Index of the current chunk.
            call_indices (List[int]): Indices corresponding to the calls within the chunk.
            cache_chunks (bool): Flag indicating whether chunk caching is enabled.
            chunk_cache_dir (Optional[PathLike]): Directory for cached chunks; required if caching is enabled.
            chunk_cache_load_kwargs (KwargsLike): Keyword arguments for loading chunk cache.

                See `vectorbtpro.utils.pickling.load`.
            release_chunk_cache (bool): If True, release the chunk cache by substituting dummy data after loading.
            pre_chunk_func (Optional[Callable]): Function to be called before executing each chunk.

                If the callable returns a value other than None, its return value is appended to
                the results and the chunk is skipped.
            pre_chunk_kwargs (KwargsLike): Keyword arguments for `pre_chunk_func`.
            template_context (KwargsLike): Additional context for template substitution.

        Returns:
            Optional[ExecResults]: Result from `pre_chunk_func` if provided; otherwise, the loaded chunk cache.
        """
        chunk_cache = None
        if cache_chunks:
            if chunk_cache_dir is None:
                raise ValueError("Must provide chunk_cache_dir")
            if not isinstance(chunk_cache_dir, Path):
                chunk_cache_dir = Path(chunk_cache_dir)
            chunk_path = chunk_cache_dir / ("chunk_%d.pickle" % chunk_idx)
            if file_exists(chunk_path):
                if release_chunk_cache:
                    chunk_cache = [DUMMY] * len(call_indices)
                else:
                    if chunk_cache_load_kwargs is None:
                        chunk_cache_load_kwargs = {}
                    chunk_cache = load(chunk_path, **chunk_cache_load_kwargs)

        if pre_chunk_func is not None:
            template_context = merge_dicts(
                dict(
                    chunk_idx=chunk_idx,
                    call_indices=call_indices,
                    chunk_cache=chunk_cache,
                ),
                template_context,
            )
            pre_chunk_func = substitute_templates(
                pre_chunk_func,
                template_context,
                eval_id="pre_chunk_func",
            )
            pre_chunk_kwargs = substitute_templates(
                pre_chunk_kwargs,
                template_context,
                eval_id="pre_chunk_kwargs",
            )
            call_results = pre_chunk_func(**pre_chunk_kwargs)
            if call_results is not None:
                return call_results
        return chunk_cache

    @classmethod
    def call_execute(
        cls,
        engine: tp.Union[ExecutionEngine, tp.Callable],
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        """Call execution on the provided tasks using `ExecutionEngine.execute`.

        Args:
            engine (Union[ExecutionEngine, Callable]): Execution engine or callable to run the tasks.
            tasks (TasksLike): Tasks (i.e., functions with their arguments) to execute.
            size (Optional[int]): Size parameter for execution.
            keys (Optional[IndexLike]): Keys used in mapping task results.

        Returns:
            ExecResults: Results obtained from executing the tasks.
        """
        if isinstance(engine, ExecutionEngine):
            return engine.execute(tasks, size=size, keys=keys)
        func_arg_names = get_func_arg_names(engine)
        execute_kwargs = {}
        if "size" in func_arg_names:
            execute_kwargs["size"] = size
        if "keys" in func_arg_names:
            execute_kwargs["keys"] = keys
        return engine(tasks, **execute_kwargs)

    @classmethod
    def call_post_chunk_func(
        cls,
        chunk_idx: int,
        call_indices: tp.List[int],
        call_results: tp.ExecResults,
        cache_chunks: bool = False,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        chunk_cache_save_kwargs: tp.KwargsLike = None,
        release_chunk_cache: bool = False,
        chunk_clear_cache: tp.Union[bool, int] = False,
        chunk_collect_garbage: tp.Union[bool, int] = False,
        chunk_delay: tp.Optional[float] = None,
        post_chunk_func: tp.Optional[tp.Callable] = None,
        post_chunk_kwargs: tp.KwargsLike = None,
        chunk_executed: bool = True,
        template_context: tp.KwargsLike = None,
    ) -> tp.ExecResults:
        """Call post-chunk function from `Executor`, managing cache saving, resource cleanup,
        and optional post-processing.

        Args:
            chunk_idx (int): Index of the current chunk.
            call_indices (List[int]): Indices corresponding to the calls within the chunk.
            call_results (ExecResults): Results obtained from executing the chunk.
            cache_chunks (bool): Flag indicating whether chunk caching is enabled.
            chunk_cache_dir (Optional[PathLike]): Directory for saving or loading cached chunks;
                required if caching is enabled.
            chunk_cache_save_kwargs (KwargsLike): Keyword arguments for saving chunk cache.

                See `vectorbtpro.utils.pickling.save`.
            release_chunk_cache (bool): If True, release the chunk cache by substituting dummy data after saving.
            chunk_clear_cache (Union[bool, int]): Determines whether to clear the cache immediately or
                after a specified number of chunks.
            chunk_collect_garbage (Union[bool, int]): Determines whether to collect garbage immediately
                or after a specified number of chunks.
            chunk_delay (Optional[float]): Delay in seconds after processing the chunk.
            post_chunk_func (Optional[Callable]): Function to be called after executing each chunk.

                If it returns None, the existing chunk results are retained; otherwise, its return
                value replaces them.
            post_chunk_kwargs (KwargsLike): Keyword arguments for `post_chunk_func`.
            chunk_executed (bool): Indicates if the chunk was executed.
            template_context (KwargsLike): Additional context for template substitution.

        Returns:
            ExecResults: Updated call results after post-chunk processing.
        """
        from vectorbtpro.registries.ca_registry import clear_cache, collect_garbage

        if chunk_executed:
            if cache_chunks:
                if chunk_cache_dir is None:
                    raise ValueError("Must provide chunk_cache_dir")
                if not isinstance(chunk_cache_dir, Path):
                    chunk_cache_dir = Path(chunk_cache_dir)
                chunk_path = chunk_cache_dir / ("chunk_%d.pickle" % chunk_idx)
                if chunk_cache_save_kwargs is None:
                    chunk_cache_save_kwargs = {}
                save(call_results, chunk_path, **chunk_cache_save_kwargs)
                if release_chunk_cache:
                    call_results = [DUMMY] * len(call_indices)

        if post_chunk_func is not None:
            template_context = merge_dicts(
                dict(
                    chunk_idx=chunk_idx,
                    call_indices=call_indices,
                    call_results=call_results,
                    chunk_executed=chunk_executed,
                ),
                template_context,
            )
            post_chunk_func = substitute_templates(
                post_chunk_func,
                template_context,
                eval_id="post_chunk_func",
            )
            post_chunk_kwargs = substitute_templates(
                post_chunk_kwargs,
                template_context,
                eval_id="post_chunk_kwargs",
            )
            new_call_results = post_chunk_func(**post_chunk_kwargs)
            if new_call_results is not None:
                call_results = new_call_results

        if isinstance(chunk_clear_cache, bool):
            if chunk_clear_cache:
                clear_cache()
        elif chunk_idx > 0 and (chunk_idx + 1) % chunk_clear_cache == 0:
            clear_cache()
        if isinstance(chunk_collect_garbage, bool):
            if chunk_collect_garbage:
                collect_garbage()
        elif chunk_idx > 0 and (chunk_idx + 1) % chunk_collect_garbage == 0:
            collect_garbage()
        if chunk_delay is not None:
            time.sleep(chunk_delay)

        return call_results

    @classmethod
    def call_post_execute_func(
        cls,
        results: tp.ExecResults,
        cache_chunks: bool = False,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        chunk_cache_load_kwargs: tp.KwargsLike = None,
        post_clear_chunk_cache: bool = True,
        release_chunk_cache: bool = False,
        post_execute_func: tp.Optional[tp.Callable] = None,
        post_execute_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Optional[tp.ExecResults]:
        """Call post-execution function from `Executor`, aggregating cached results, clearing cache,
        and optionally post-processing overall execution results.

        Args:
            results (ExecResults): Initial execution results.
            cache_chunks (bool): Flag indicating whether chunk caching is enabled.
            chunk_cache_dir (Optional[PathLike]): Directory for cached chunks; required if caching is enabled.
            chunk_cache_load_kwargs (KwargsLike): Keyword arguments for loading chunk cache.

                See `vectorbtpro.utils.pickling.load`.
            post_clear_chunk_cache (bool): If True, clear the chunk cache directory after loading cached results.
            release_chunk_cache (bool): If True, release cached chunk data by replacing with dummy values.
            post_execute_func (Optional[Callable]): Function to be called after executing all tasks.

                If it returns None, the default results are preserved; otherwise, its return value replaces them.
            post_execute_kwargs (KwargsLike): Keyword arguments for `post_execute_func`.
            template_context (KwargsLike): Additional context for template substitution.

        Returns:
            Optional[ExecResults]: Post-processed execution results if modified; otherwise,
                the aggregated results.
        """
        if cache_chunks and release_chunk_cache:
            if chunk_cache_dir is None:
                raise ValueError("Must provide chunk_cache_dir")
            if not isinstance(chunk_cache_dir, Path):
                chunk_cache_dir = Path(chunk_cache_dir)
            if chunk_cache_load_kwargs is None:
                chunk_cache_load_kwargs = {}
            chunk_paths = Path(chunk_cache_dir).rglob("chunk_*.pickle")
            chunk_paths = sorted(chunk_paths, key=lambda x: int(x.name.split("_")[1].split(".")[0]))
            new_results = []
            for chunk_path in chunk_paths:
                chunk_cache = load(chunk_path, **chunk_cache_load_kwargs)
                new_results.extend(chunk_cache)
            results = new_results
        if cache_chunks and post_clear_chunk_cache:
            if chunk_cache_dir is None:
                raise ValueError("Must provide chunk_cache_dir")
            remove_dir(chunk_cache_dir, missing_ok=True, with_contents=True)

        if post_execute_func is not None:
            template_context = merge_dicts(
                dict(results=results),
                template_context,
            )
            post_execute_func = substitute_templates(
                post_execute_func,
                template_context,
                eval_id="post_execute_func",
            )
            post_execute_kwargs = substitute_templates(
                post_execute_kwargs,
                template_context,
                eval_id="post_execute_kwargs",
            )
            new_results = post_execute_func(**post_execute_kwargs)
            if new_results is not None:
                return new_results
        return results

    @classmethod
    def merge_results(
        cls,
        results: tp.ExecResults,
        keys: tp.Optional[tp.IndexLike] = None,
        filter_results: bool = False,
        raise_no_results: bool = True,
        merge_func: tp.MergeFuncLike = None,
        merge_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Union[tp.ExecResults, tp.MergeResult]:
        """Merge execution results.

        Args:
            results (ExecResults): Execution results from tasks.
            keys (Optional[IndexLike]): Index or keys associated with the results.
            filter_results (bool): Whether to filter out results that are `vectorbtpro.utils.execution.NoResult`.
            raise_no_results (bool): Flag indicating whether to raise a
                `NoResultsException` exception if no results remain.
            merge_func (MergeFuncLike): Function to merge the results.

                See `vectorbtpro.utils.merging.MergeFunc`.
            merge_kwargs (KwargsLike): Keyword arguments for `merge_func`.
            template_context (KwargsLike): Additional context for template substitution.

        Returns:
            Union[ExecResults, MergeResult]: Merged results if a merge function is provided;
                otherwise, the original results.
        """
        if filter_results:
            try:
                results, keys = filter_out_no_results(results, keys=keys)
            except NoResultsException as e:
                if raise_no_results:
                    raise e
                return NoResult
            no_results_filtered = True
        else:
            no_results_filtered = False
        if merge_func is not None:
            template_context = merge_dicts(
                dict(results=results),
                template_context,
            )
            from vectorbtpro.base.merging import is_merge_func_from_config

            if is_merge_func_from_config(merge_func):
                merge_kwargs = merge_dicts(
                    dict(
                        keys=keys,
                        filter_results=not no_results_filtered,
                        raise_no_results=raise_no_results,
                    ),
                    merge_kwargs,
                )
            if isinstance(merge_func, MergeFunc):
                merge_func = merge_func.replace(merge_kwargs=merge_kwargs, context=template_context)
            else:
                merge_func = MergeFunc(
                    merge_func, merge_kwargs=merge_kwargs, context=template_context
                )
            return merge_func(results)
        return results

    def run(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Union[tp.ExecResults, tp.MergeResult]:
        """Execute tasks with optional chunking, caching, and merging.

        Executes a collection of tasks, handling warmup calls, custom chunking, and caching as configured.
        Execution may involve filtering of results and subsequent merging using a specified merge function.

        Args:
            tasks (TasksLike): Tasks (i.e., functions with their arguments) to execute.
            size (Optional[int]): Total number of tasks if not inferable from the tasks collection.
            keys (Optional[IndexLike]): Index or keys associated with the tasks.

        Returns:
            Union[ExecResults, MergeResult]: Execution results, possibly merged using the provided
                merging function.
        """
        from vectorbtpro.base.indexes import to_any_index

        engine = self.engine
        min_size = self.min_size
        n_chunks = self.n_chunks
        chunk_len = self.chunk_len
        chunk_meta = self.chunk_meta
        distribute = self.distribute
        warmup = self.warmup
        in_chunk_order = self.in_chunk_order
        cache_chunks = self.cache_chunks
        chunk_cache_dir = self.chunk_cache_dir
        chunk_cache_load_kwargs = self.chunk_cache_load_kwargs
        chunk_cache_save_kwargs = self.chunk_cache_save_kwargs
        pre_clear_chunk_cache = self.pre_clear_chunk_cache
        post_clear_chunk_cache = self.post_clear_chunk_cache
        release_chunk_cache = self.release_chunk_cache
        chunk_clear_cache = self.chunk_clear_cache
        chunk_collect_garbage = self.chunk_collect_garbage
        chunk_delay = self.chunk_delay
        pre_execute_func = self.pre_execute_func
        pre_execute_kwargs = self.pre_execute_kwargs
        pre_chunk_func = self.pre_chunk_func
        pre_chunk_kwargs = self.pre_chunk_kwargs
        post_chunk_func = self.post_chunk_func
        post_chunk_kwargs = self.post_chunk_kwargs
        post_execute_func = self.post_execute_func
        post_execute_kwargs = self.post_execute_kwargs
        post_execute_on_sorted = self.post_execute_on_sorted
        filter_results = self.filter_results
        raise_no_results = self.raise_no_results
        merge_func = self.merge_func
        merge_kwargs = self.merge_kwargs
        template_context = self.template_context
        show_progress = self.show_progress
        pbar_kwargs = self.pbar_kwargs

        if keys is not None:
            keys = to_any_index(keys)
        pbar_kwargs = dict(pbar_kwargs)
        if "bar_id" not in pbar_kwargs:
            if keys is not None:
                if isinstance(keys, pd.MultiIndex):
                    pbar_kwargs["bar_id"] = ("chunk_tasks", tuple(keys.names))
                else:
                    pbar_kwargs["bar_id"] = ("chunk_tasks", keys.name)
        if cache_chunks and chunk_cache_dir is None:
            if "bar_id" in pbar_kwargs:
                import hashlib

                chunk_cache_dir_hash = hashlib.md5(
                    str(pbar_kwargs["bar_id"]).encode("utf-8")
                ).hexdigest()
                chunk_cache_dir = "chunk_cache_%s" % chunk_cache_dir_hash

        if warmup:
            if not hasattr(tasks, "__getitem__"):
                tasks = list(tasks)
            tasks[0][0](*tasks[0][1], **tasks[0][2])

        if n_chunks is None and chunk_len is None and chunk_meta is None:
            if isinstance(tasks, CustomTemplate):
                n_chunks = 1
            else:
                if cache_chunks:
                    warn("Cannot cache chunks without chunking")
                    cache_chunks = False
                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                if "n_chunks" not in template_context:
                    template_context["n_chunks"] = 1
                results = self.call_execute(engine, tasks, size=size, keys=keys)
                results = self.call_post_execute_func(
                    results,
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                    post_clear_chunk_cache=post_clear_chunk_cache,
                    release_chunk_cache=release_chunk_cache,
                    post_execute_func=post_execute_func,
                    post_execute_kwargs=post_execute_kwargs,
                    template_context=template_context,
                )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )

        if chunk_meta is None:
            from vectorbtpro.utils.chunking import iter_chunk_meta

            if not isinstance(tasks, CustomTemplate) and hasattr(tasks, "__len__"):
                _size = len(tasks)
            elif size is not None:
                _size = size
            elif keys is not None and hasattr(keys, "__len__"):
                _size = len(keys)
            else:
                if isinstance(tasks, CustomTemplate):
                    raise ValueError("When tasks is a template, must provide size")
                tasks = list(tasks)
                _size = len(tasks)
            chunk_meta = iter_chunk_meta(
                size=_size,
                min_size=min_size,
                n_chunks=n_chunks,
                chunk_len=chunk_len,
            )
            if "chunk_meta" not in template_context:
                template_context["chunk_meta"] = chunk_meta

        if isinstance(tasks, CustomTemplate):
            if cache_chunks:
                warn("Cannot cache chunks with custom chunking")
                cache_chunks = False
            tasks = substitute_templates(tasks, template_context, eval_id="tasks")
            if hasattr(tasks, "__len__"):
                size = len(tasks)
            elif keys is not None and hasattr(keys, "__len__"):
                size = len(keys)
            else:
                size = None
            self.call_pre_execute_func(
                cache_chunks=cache_chunks,
                chunk_cache_dir=chunk_cache_dir,
                pre_clear_chunk_cache=pre_clear_chunk_cache,
                pre_execute_func=pre_execute_func,
                pre_execute_kwargs=pre_execute_kwargs,
                template_context=template_context,
            )
            if "n_chunks" not in template_context:
                template_context["n_chunks"] = 1
            results = self.call_execute(
                engine,
                tasks,
                size=size,
                keys=keys,
            )
            results = self.call_post_execute_func(
                results,
                cache_chunks=cache_chunks,
                chunk_cache_dir=chunk_cache_dir,
                chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                post_clear_chunk_cache=post_clear_chunk_cache,
                release_chunk_cache=release_chunk_cache,
                post_execute_func=post_execute_func,
                post_execute_kwargs=post_execute_kwargs,
                template_context=template_context,
            )
            return self.merge_results(
                results,
                keys=keys,
                filter_results=filter_results,
                raise_no_results=raise_no_results,
                merge_func=merge_func,
                merge_kwargs=merge_kwargs,
                template_context=template_context,
            )

        last_idx = -1
        indices_sorted = True
        all_call_indices = []
        for _chunk_meta in chunk_meta:
            if _chunk_meta.indices is not None:
                call_indices = list(_chunk_meta.indices)
            else:
                if _chunk_meta.start is None or _chunk_meta.end is None:
                    raise ValueError("Each chunk must have a start and an end index")
                call_indices = list(range(_chunk_meta.start, _chunk_meta.end))
            if indices_sorted:
                for idx in call_indices:
                    if idx != last_idx + 1:
                        indices_sorted = False
                        break
                    last_idx = idx
            all_call_indices.append(call_indices)
        if "n_chunks" not in template_context:
            template_context["n_chunks"] = len(all_call_indices)

        if distribute.lower() == "tasks":
            if indices_sorted and not hasattr(tasks, "__len__"):
                results = []
                chunk_idx = 0
                _tasks = []

                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                with ProgressBar(
                    total=len(all_call_indices), show_progress=show_progress, **pbar_kwargs
                ) as pbar:
                    pbar.set_description(
                        dict(
                            chunk_tasks=f"{all_call_indices[chunk_idx][0]}..{all_call_indices[chunk_idx][-1]}"
                        )
                    )
                    for i, func_args in enumerate(tasks):
                        if i > all_call_indices[chunk_idx][-1]:
                            call_indices = all_call_indices[chunk_idx]
                            call_results = self.call_pre_chunk_func(
                                chunk_idx,
                                call_indices,
                                cache_chunks=cache_chunks,
                                chunk_cache_dir=chunk_cache_dir,
                                chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                                release_chunk_cache=release_chunk_cache,
                                pre_chunk_func=pre_chunk_func,
                                pre_chunk_kwargs=pre_chunk_kwargs,
                                template_context=template_context,
                            )
                            if call_results is None:
                                call_results = self.call_execute(
                                    engine,
                                    _tasks,
                                    size=len(call_indices),
                                    keys=keys[call_indices] if keys is not None else None,
                                )
                                chunk_executed = True
                            else:
                                chunk_executed = False
                            call_results = self.call_post_chunk_func(
                                chunk_idx,
                                call_indices,
                                call_results,
                                cache_chunks=cache_chunks,
                                chunk_cache_dir=chunk_cache_dir,
                                chunk_cache_save_kwargs=chunk_cache_save_kwargs,
                                release_chunk_cache=release_chunk_cache,
                                chunk_clear_cache=chunk_clear_cache,
                                chunk_collect_garbage=chunk_collect_garbage,
                                chunk_delay=chunk_delay,
                                post_chunk_func=post_chunk_func,
                                post_chunk_kwargs=post_chunk_kwargs,
                                chunk_executed=chunk_executed,
                                template_context=template_context,
                            )
                            results.extend(call_results)
                            chunk_idx += 1
                            _tasks = []
                            if chunk_idx < len(all_call_indices):
                                pbar.set_description(
                                    dict(
                                        chunk_tasks=f"{all_call_indices[chunk_idx][0]}..{all_call_indices[chunk_idx][-1]}"
                                    )
                                )
                            pbar.update()
                        _tasks.append(func_args)
                    if len(_tasks) > 0:
                        call_indices = all_call_indices[chunk_idx]
                        call_results = self.call_pre_chunk_func(
                            chunk_idx,
                            call_indices,
                            cache_chunks=cache_chunks,
                            chunk_cache_dir=chunk_cache_dir,
                            chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                            release_chunk_cache=release_chunk_cache,
                            pre_chunk_func=pre_chunk_func,
                            pre_chunk_kwargs=pre_chunk_kwargs,
                            template_context=template_context,
                        )
                        if call_results is None:
                            call_results = self.call_execute(
                                engine,
                                _tasks,
                                size=len(call_indices),
                                keys=keys[call_indices] if keys is not None else None,
                            )
                            chunk_executed = True
                        else:
                            chunk_executed = False
                        call_results = self.call_post_chunk_func(
                            chunk_idx,
                            call_indices,
                            call_results,
                            cache_chunks=cache_chunks,
                            chunk_cache_dir=chunk_cache_dir,
                            chunk_cache_save_kwargs=chunk_cache_save_kwargs,
                            release_chunk_cache=release_chunk_cache,
                            chunk_clear_cache=chunk_clear_cache,
                            chunk_collect_garbage=chunk_collect_garbage,
                            chunk_delay=chunk_delay,
                            post_chunk_func=post_chunk_func,
                            post_chunk_kwargs=post_chunk_kwargs,
                            chunk_executed=chunk_executed,
                            template_context=template_context,
                        )
                        results.extend(call_results)
                        pbar.update()
                results = self.call_post_execute_func(
                    results,
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                    post_clear_chunk_cache=post_clear_chunk_cache,
                    release_chunk_cache=release_chunk_cache,
                    post_execute_func=post_execute_func,
                    post_execute_kwargs=post_execute_kwargs,
                    template_context=template_context,
                )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )
            else:
                tasks = list(tasks)
                results = []
                output_indices = []

                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                with ProgressBar(
                    total=len(all_call_indices), show_progress=show_progress, **pbar_kwargs
                ) as pbar:
                    pbar.set_description(
                        dict(
                            chunk_tasks=f"{all_call_indices[0][0]}..{all_call_indices[0][-1]}"
                        )
                    )
                    for chunk_idx, call_indices in enumerate(all_call_indices):
                        call_results = self.call_pre_chunk_func(
                            chunk_idx,
                            call_indices,
                            cache_chunks=cache_chunks,
                            chunk_cache_dir=chunk_cache_dir,
                            chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                            release_chunk_cache=release_chunk_cache,
                            pre_chunk_func=pre_chunk_func,
                            pre_chunk_kwargs=pre_chunk_kwargs,
                            template_context=template_context,
                        )
                        if call_results is None:
                            _tasks = []
                            for idx in call_indices:
                                _tasks.append(tasks[idx])
                            call_results = self.call_execute(
                                engine,
                                _tasks,
                                size=len(call_indices),
                                keys=keys[call_indices] if keys is not None else None,
                            )
                            chunk_executed = True
                        else:
                            chunk_executed = False
                        call_results = self.call_post_chunk_func(
                            chunk_idx,
                            call_indices,
                            call_results,
                            cache_chunks=cache_chunks,
                            chunk_cache_dir=chunk_cache_dir,
                            chunk_cache_save_kwargs=chunk_cache_save_kwargs,
                            release_chunk_cache=release_chunk_cache,
                            chunk_clear_cache=chunk_clear_cache,
                            chunk_collect_garbage=chunk_collect_garbage,
                            chunk_delay=chunk_delay,
                            post_chunk_func=post_chunk_func,
                            post_chunk_kwargs=post_chunk_kwargs,
                            chunk_executed=chunk_executed,
                            template_context=template_context,
                        )
                        results.extend(call_results)
                        output_indices.extend(call_indices)
                        if chunk_idx + 1 < len(all_call_indices):
                            pbar.set_description(
                                dict(
                                    chunk_tasks=f"{all_call_indices[chunk_idx + 1][0]}..{all_call_indices[chunk_idx + 1][-1]}"
                                )
                            )
                        pbar.update()
                if not post_execute_on_sorted:
                    results = self.call_post_execute_func(
                        results,
                        cache_chunks=cache_chunks,
                        chunk_cache_dir=chunk_cache_dir,
                        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                        post_clear_chunk_cache=post_clear_chunk_cache,
                        release_chunk_cache=release_chunk_cache,
                        post_execute_func=post_execute_func,
                        post_execute_kwargs=post_execute_kwargs,
                        template_context=template_context,
                    )
                if not in_chunk_order and not indices_sorted:
                    results = [x for _, x in sorted(zip(output_indices, results))]
                if post_execute_on_sorted:
                    results = self.call_post_execute_func(
                        results,
                        cache_chunks=cache_chunks,
                        chunk_cache_dir=chunk_cache_dir,
                        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                        post_clear_chunk_cache=post_clear_chunk_cache,
                        release_chunk_cache=release_chunk_cache,
                        post_execute_func=post_execute_func,
                        post_execute_kwargs=post_execute_kwargs,
                        template_context=template_context,
                    )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )

        elif distribute.lower() == "chunks":
            if cache_chunks:
                warn("Cannot cache chunks with chunk distribution")
                cache_chunks = False
            if indices_sorted and not hasattr(tasks, "__len__"):
                chunk_idx = 0
                _tasks = []
                task_chunks = []

                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                for i, func_args in enumerate(tasks):
                    if i > all_call_indices[chunk_idx][-1]:
                        task_chunks.append(self.build_serial_chunk(_tasks))
                        chunk_idx += 1
                        _tasks = []
                    _tasks.append(func_args)
                if len(_tasks) > 0:
                    task_chunks.append(self.build_serial_chunk(_tasks))
                results = self.call_execute(
                    engine,
                    task_chunks,
                    size=len(task_chunks),
                )
                results = [x for o in results for x in o]
                results = self.call_post_execute_func(
                    results,
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                    post_clear_chunk_cache=post_clear_chunk_cache,
                    release_chunk_cache=release_chunk_cache,
                    post_execute_func=post_execute_func,
                    post_execute_kwargs=post_execute_kwargs,
                    template_context=template_context,
                )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )
            else:
                tasks = list(tasks)
                task_chunks = []
                output_indices = []

                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                for call_indices in all_call_indices:
                    _tasks = []
                    for idx in call_indices:
                        _tasks.append(tasks[idx])
                    task_chunks.append(self.build_serial_chunk(_tasks))
                    output_indices.extend(call_indices)
                results = self.call_execute(
                    engine,
                    task_chunks,
                    size=len(task_chunks),
                )
                results = [x for o in results for x in o]
                if not post_execute_on_sorted:
                    results = self.call_post_execute_func(
                        results,
                        cache_chunks=cache_chunks,
                        chunk_cache_dir=chunk_cache_dir,
                        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                        post_clear_chunk_cache=post_clear_chunk_cache,
                        release_chunk_cache=release_chunk_cache,
                        post_execute_func=post_execute_func,
                        post_execute_kwargs=post_execute_kwargs,
                        template_context=template_context,
                    )
                if not in_chunk_order and not indices_sorted:
                    results = [x for _, x in sorted(zip(output_indices, results))]
                if post_execute_on_sorted:
                    results = self.call_post_execute_func(
                        results,
                        cache_chunks=cache_chunks,
                        chunk_cache_dir=chunk_cache_dir,
                        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                        post_clear_chunk_cache=post_clear_chunk_cache,
                        release_chunk_cache=release_chunk_cache,
                        post_execute_func=post_execute_func,
                        post_execute_kwargs=post_execute_kwargs,
                        template_context=template_context,
                    )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )
        else:
            raise ValueError(f"Invalid distribute: '{self.distribute}'")


def execute(
    tasks: tp.TasksLike,
    size: tp.Optional[int] = None,
    keys: tp.Optional[tp.IndexLike] = None,
    executor: tp.Optional[tp.MaybeType[Executor]] = None,
    replace_executor: tp.Optional[bool] = None,
    merge_to_engine_config: tp.Optional[bool] = None,
    **kwargs,
) -> tp.MergeableResults:
    """Execute functions and their arguments using `Executor`.

    Executes the provided tasks by initializing or reusing an executor.
    Keyword arguments that are not expected by `Executor` or `engine_config`
    may be merged into `engine_config` if `merge_to_engine_config` is True.

    If an executor instance is supplied and `replace_executor` is True, creates a new
    `Executor` instance by updating non-None parameters.

    Args:
        tasks (TasksLike): Tasks (i.e., functions with their arguments) to execute.
        size (Optional[int]): Number of tasks to run concurrently.
        keys (Optional[IndexLike]): Identifiers corresponding to each task.
        executor (Optional[MaybeType[Executor]]): `Executor` class or instance used for executing iterations.

            If None, a default executor is selected from the configuration.
        replace_executor (Optional[bool]): Flag to create a new `Executor` instance by replacing non-None
            arguments when additional options are provided.
        merge_to_engine_config (Optional[bool]): Flag indicating whether keyword arguments not
            matching `Executor` keys should be merged into its `engine_config`.
        **kwargs: Keyword arguments for `Executor` or merged into `engine_config`.

    Returns:
        MergeableResults: Merged results from executing the tasks.

    !!! info
        For default settings, see `vectorbtpro._settings.execution`.
    """
    from vectorbtpro._settings import settings

    execution_cfg = settings["execution"]

    if executor is None:
        executor = execution_cfg["executor"]
    if executor is None:
        executor = Executor

    if merge_to_engine_config is None:
        merge_to_engine_config = execution_cfg["merge_to_engine_config"]
    if merge_to_engine_config and len(kwargs) > 0:
        arg_names_set = set(executor._expected_keys)
        engine_config = kwargs.pop("engine_config", None)
        if engine_config is None:
            _engine_config = {}
        else:
            _engine_config = dict(engine_config)
        engine_config_changed = False
        for k in list(kwargs.keys()):
            if k not in arg_names_set and k not in _engine_config:
                _engine_config[k] = kwargs.pop(k)
                engine_config_changed = True
        if engine_config_changed:
            kwargs["engine_config"] = _engine_config
        else:
            kwargs["engine_config"] = engine_config
    if isinstance(executor, type):
        checks.assert_subclass_of(executor, Executor, arg_name="executor")
        executor = executor(**kwargs)
    else:
        checks.assert_instance_of(executor, Executor, arg_name="executor")
        if replace_executor is None:
            replace_executor = execution_cfg["replace_executor"]
        if replace_executor and len(kwargs) > 0:
            executor = executor.replace(**kwargs)
    return executor.run(tasks, size=size, keys=keys)


def parse_iterable_and_keys(
    iterable_like: tp.Union[int, tp.Iterable],
    keys: tp.Optional[tp.IndexLike] = None,
) -> tp.Tuple[tp.Iterable, tp.Optional[tp.Index]]:
    """Parse the iterable and keys from the provided objects.

    Converts an iterable-like object into a standard iterable and derives associated keys if needed.
    If an integer is provided, it is interpreted as a range. For dictionaries, Pandas Index, or
    Pandas Series, keys are derived from their inherent structure. Otherwise, if the iterable is a
    sequence and keys are not provided, keys are extracted using `vectorbtpro.base.indexes.index_from_values`
    to avoid unnecessary materialization.

    Args:
        iterable_like (Union[int, Iterable]): Integer representing a total count or an iterable of values.

            If an integer is provided, it is interpreted as a range.
        keys (Optional[IndexLike]): Keys-like object used to index the iterable.

            If not provided and the iterable is a dictionary, Pandas Index, or Pandas Series,
            the keys are derived from the object.

            Otherwise, if the iterable is a sequence, keys are extracted using
            `vectorbtpro.base.indexes.index_from_values`.

    Returns:
        Tuple[Iterable, Optional[Index]]: Tuple containing the iterable values and the
            derived or provided keys.
    """
    if keys is not None:
        from vectorbtpro.base.indexes import to_any_index

        keys = to_any_index(keys)
    if checks.is_int(iterable_like):
        iterable = range(iterable_like)
        if keys is None:
            keys = pd.Index(iterable)
    elif isinstance(iterable_like, dict):
        iterable = iterable_like.values()
        if keys is None:
            keys = pd.Index(list(iterable_like.keys()))
    elif isinstance(iterable_like, pd.Index):
        iterable = iterable_like
        if keys is None:
            keys = iterable_like
    elif isinstance(iterable_like, pd.Series):
        iterable = iterable_like.values
        if keys is None:
            keys = iterable_like.index
    elif checks.is_sequence(iterable_like):
        from vectorbtpro.base.indexes import index_from_values

        iterable = list(iterable_like)
        if keys is None:
            keys = index_from_values(iterable_like)
    else:
        iterable = iterable_like
    return iterable, keys


def iterated(
    *args,
    over_arg: tp.Optional[tp.AnnArgQuery] = None,
    executor: tp.Optional[tp.MaybeType[Executor]] = None,
    replace_executor: tp.Optional[bool] = None,
    merge_to_engine_config: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Callable:
    """Return a decorator that executes a function iteratively using an `Executor`.

    This decorator creates a new function with the same signature as the original.

    The decorated function is executed iteratively for each element in an iterable determined
    by a specified argument. If `over_arg` is None, the first positional argument is used as the iterable.

    The executor options can be adjusted via the `options` attribute of the wrapper or by providing
    keyword arguments prefixed with an underscore. Explicit iteration keys and size can be provided
    as `_keys` and `_size` when the range-like object is an iterator.

    Keyword arguments not matching `Executor` keys are merged into its `engine_config` if
    `merge_to_engine_config` is True; otherwise, they are passed directly to `Executor`.

    If an executor instance is provided and `replace_executor` is True, a new `Executor` instance
    is created with non-None parameters replaced.

    If the decorated function returns `NoResult`, the current iteration is skipped and its index
    is removed from the final result.

    Args:
        func (Callable): Function to be decorated.
        over_arg (Optional[AnnArgQuery]): Query specifying which argument to iterate over.

            If None, the first positional argument is used.
        executor (Optional[MaybeType[Executor]]): `Executor` class or instance used for executing iterations.

            If None, a default executor is selected from the configuration.
        replace_executor (Optional[bool]): Flag to create a new `Executor` instance by replacing non-None
            arguments when additional options are provided.
        merge_to_engine_config (Optional[bool]): Flag indicating whether keyword arguments not
            matching `Executor` keys should be merged into its `engine_config`.
        **kwargs: Keyword arguments for `Executor` or the decorated function.

    Returns:
        Callable: Wrapper function that executes the original function iteratively.

    !!! info
        For default settings, see `vectorbtpro._settings.execution`.
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        from vectorbtpro._settings import settings

        execution_cfg = settings["execution"]

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            executor = kwargs.get("_executor")
            if executor is None:
                executor = wrapper.options["executor"]
            if executor is None:
                executor = execution_cfg["executor"]
            if executor is None:
                executor = Executor

            arg_names_set = set(executor._expected_keys)
            kwargs_options = {}
            for k in list(kwargs.keys()):
                if k.startswith("_"):
                    if k[1:] in wrapper.options or k[1:] in arg_names_set:
                        kwargs_options[k[1:]] = kwargs.pop(k)
            executor_kwargs = merge_dicts(wrapper.options, kwargs_options)
            _ = executor_kwargs.pop("executor")
            over_arg = executor_kwargs.pop("over_arg")
            replace_executor = executor_kwargs.pop("replace_executor")
            merge_to_engine_config = executor_kwargs.pop("merge_to_engine_config")
            size = executor_kwargs.pop("size", None)
            keys = executor_kwargs.pop("keys", None)

            if over_arg is None:
                iterable_like = args[0]
            else:
                ann_args = annotate_args(func, args, kwargs)
                iterable_like = match_ann_arg(ann_args, over_arg)
            iterable, keys = parse_iterable_and_keys(iterable_like, keys=keys)
            if keys is not None and size is None:
                size = len(keys)
            if keys is not None and keys.name is None:
                if over_arg is None:
                    keys = keys.rename(get_func_arg_names(func)[0])
                else:
                    ann_args = annotate_args(func, args, kwargs)
                    over_arg_name = match_ann_arg(ann_args, over_arg, return_name=True)
                    keys = keys.rename(over_arg_name)

            if over_arg is None:

                def _get_task_generator():
                    for x in iterable:
                        yield Task(func, x, *args[1:], **kwargs)

            else:

                def _get_task_generator():
                    for x in iterable:
                        flat_ann_args = annotate_args(func, args, kwargs, flatten=True)
                        match_and_set_flat_ann_arg(flat_ann_args, over_arg, x)
                        new_args, new_kwargs = flat_ann_args_to_args(flat_ann_args)
                        yield Task(func, *new_args, **new_kwargs)

            tasks = _get_task_generator()

            if merge_to_engine_config is None:
                merge_to_engine_config = execution_cfg["merge_to_engine_config"]
            if merge_to_engine_config and len(executor_kwargs) > 0:
                arg_names_set = set(executor._expected_keys)
                engine_config = executor_kwargs.pop("engine_config", None)
                if engine_config is None:
                    _engine_config = {}
                else:
                    _engine_config = dict(engine_config)
                engine_config_changed = False
                for k in list(executor_kwargs.keys()):
                    if k not in arg_names_set and k not in _engine_config:
                        _engine_config[k] = executor_kwargs.pop(k)
                        engine_config_changed = True
                if engine_config_changed:
                    executor_kwargs["engine_config"] = _engine_config
                else:
                    executor_kwargs["engine_config"] = engine_config
            if isinstance(executor, type):
                checks.assert_subclass_of(executor, Executor, arg_name="executor")
                executor = executor(**executor_kwargs)
            else:
                checks.assert_instance_of(executor, Executor, arg_name="executor")
                if replace_executor is None:
                    replace_executor = execution_cfg["replace_executor"]
                if replace_executor and len(executor_kwargs) > 0:
                    executor = executor.replace(**executor_kwargs)
            return executor.run(tasks, size=size, keys=keys)

        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.is_executor = True
        wrapper.options = FrozenConfig(
            over_arg=over_arg,
            executor=executor,
            replace_executor=replace_executor,
            merge_to_engine_config=merge_to_engine_config,
            **kwargs,
        )
        signature = inspect.signature(wrapper)
        lists_var_kwargs = False
        for k, v in signature.parameters.items():
            if v.kind == v.VAR_KEYWORD:
                lists_var_kwargs = True
                break
        if not lists_var_kwargs:
            var_kwargs_param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
            new_parameters = tuple(signature.parameters.values()) + (var_kwargs_param,)
            wrapper.__signature__ = signature.replace(parameters=new_parameters)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
