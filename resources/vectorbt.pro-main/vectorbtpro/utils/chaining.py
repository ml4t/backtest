# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for chaining operations."""

import inspect

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.decorators import hybrid_method

__all__ = [
    "Chainable",
]


class Chainable(Base):
    """Class for chainable objects providing a fluent interface.

    Instances can chain functions or tasks sequentially using the `pipe` and `chain` methods.
    """

    @hybrid_method
    def pipe(cls_or_self, func: tp.PipeFunc, *args, **kwargs) -> tp.Any:
        """Apply a chainable function to a chainable object.

        This method can be invoked as an instance or a class method. When called on an instance,
        the instance is automatically passed to the provided function unless a tuple argument specifies
        a custom insertion point or the function is already bound. When called on the class, only
        `*args` and `**kwargs` are passed.

        Args:
            func (PipeFunc): Function to apply.

                It can be:

                * Callable function.
                * String representing an attribute path to resolve via `vectorbtpro.utils.attr_.deep_getattr`.
                * Tuple where the first element is a callable or attribute path and the second element indicates
                    the positional or keyword argument where to pass the chainable instance.
            *args: Positional arguments for `func`.
            **kwargs: Keyword arguments for `func`.

        Returns:
            Any: Result of invoking the chainable function.
        """
        if isinstance(func, tuple):
            func, arg_name = func
            if not isinstance(cls_or_self, type):
                if isinstance(arg_name, int):
                    args = list(args)
                    args.insert(arg_name, cls_or_self)
                    args = tuple(args)
                else:
                    kwargs[arg_name] = cls_or_self
            prepend_to_args = False
        else:
            prepend_to_args = not isinstance(cls_or_self, type)
        if isinstance(func, str):
            from vectorbtpro.utils.attr_ import deep_getattr

            func = deep_getattr(cls_or_self, func, call_last_attr=False)
            if not callable(func) and len(args) == 0 and len(kwargs) == 0:
                return func
            if prepend_to_args:
                prepend_to_args = not inspect.ismethod(func)
        if prepend_to_args:
            args = (cls_or_self, *args)
        return func(*args, **kwargs)

    @hybrid_method
    def chain(cls_or_self, tasks: tp.PipeTasks) -> tp.Any:
        """Chain multiple tasks sequentially using the `pipe` method.

        This method iterates over a sequence of tasks, converting each task to a
        `vectorbtpro.utils.execution.Task` if necessary, and applies them one after the other
        via the `pipe` method.

        Args:
            tasks (PipeTasks): Collection of tasks to chain.

                Each task can be:

                * Instance of `vectorbtpro.utils.execution.Task`.
                * Tuple convertible to a `vectorbtpro.utils.execution.Task` via
                    `vectorbtpro.utils.execution.Task.from_tuple`.
                * Callable function to be wrapped as a `vectorbtpro.utils.execution.Task`.

        Returns:
            Any: Final result after applying all tasks sequentially.
        """
        from vectorbtpro.utils.execution import Task

        result = cls_or_self
        for task in tasks:
            if not isinstance(task, Task):
                if isinstance(task, tuple):
                    task = Task.from_tuple(task)
                else:
                    task = Task(task)
            func, args, kwargs = task
            result = result.pipe(func, *args, **kwargs)
        return result
