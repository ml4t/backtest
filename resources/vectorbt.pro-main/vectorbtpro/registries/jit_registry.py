# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a global registry for jittable functions.

Jitting refers to just-in-time compilation of functions to accelerate their execution.
A jitter is a decorator that wraps a regular Python function and returns a decorated version,
which may share the original signature or have a similar one. Jitters accept various options
to modify execution behavior, allowing a single function to be decorated by multiple jitter
instances (for example, one using `numba.jit` and another applying a `parallel=True` flag).

In addition to jitters, vectorbtpro introduces the concept of tasks. A single task can be implemented
using various jitter types (such as NumPy, Numba, and JAX). For example, a task that converts
prices to returns can be implemented using both NumPy and Numba. These implementations are
registered by `JITRegistry` as `JitableSetup` instances, stored in `JITRegistry.jitable_setups`,
and uniquely identified by the task ID and jitter type. Note that `JitableSetup` instances store
only the information needed to decorate a function.

The decorated function together with the applied jitter is registered as a `JittedSetup` instance
and stored in `JITRegistry.jitted_setups`. This mechanism acts as a cache to quickly retrieve an
already decorated function and avoid unnecessary recompilation.

Let's implement a task that computes the sum over an array using both NumPy and Numba:

```pycon
>>> from vectorbtpro import *

>>> @vbt.register_jitted(task_id_or_func='sum')
... def sum_np(a):
...     return a.sum()

>>> @vbt.register_jitted(task_id_or_func='sum')
... def sum_nb(a):
...     out = 0.
...     for i in range(a.shape[0]):
...         out += a[i]
...     return out
```

We can see that two new jitable setups were registered:

```pycon
>>> vbt.jit_reg.jitable_setups['sum']
{'np': JitableSetup(task_id='sum', jitter_id='np', py_func=<function sum_np at 0x7fea215b1e18>, jitter_kwargs={}, tags=None),
 'nb': JitableSetup(task_id='sum', jitter_id='nb', py_func=<function sum_nb at 0x7fea273d41e0>, jitter_kwargs={}, tags=None)}
```

Moreover, two jitted setups were registered for our decorated functions:

```pycon
>>> from vectorbtpro.registries.jit_registry import JitableSetup

>>> hash_np = JitableSetup.get_hash('sum', 'np')
>>> vbt.jit_reg.jitted_setups[hash_np]
{3527539: JittedSetup(jitter=<vectorbtpro.utils.jitting.NumPyJitter object at 0x7fea21506080>, jitted_func=<function sum_np at 0x7fea215b1e18>)}

>>> hash_nb = JitableSetup.get_hash('sum', 'nb')
>>> vbt.jit_reg.jitted_setups[hash_nb]
{6326224984503844995: JittedSetup(jitter=<vectorbtpro.utils.jitting.NumbaJitter object at 0x7fea214d0ba8>, jitted_func=CPUDispatcher(<function sum_nb at 0x7fea273d41e0>))}
```

These setups hold the decorated functions along with any options passed during registration.
When `JITRegistry.resolve` is called without additional keyword arguments, it returns the registered function:

```pycon
>>> jitted_func = vbt.jit_reg.resolve('sum', jitter='nb')
>>> jitted_func
CPUDispatcher(<function sum_nb at 0x7fea273d41e0>)

>>> jitted_func.targetoptions
{'nopython': True, 'nogil': True, 'parallel': False, 'boundscheck': False}
```

When additional options are provided, the function is redecorated and a new `JittedOption` instance is registered:

```pycon
>>> jitted_func = vbt.jit_reg.resolve('sum', jitter='nb', nopython=False)
>>> jitted_func
CPUDispatcher(<function sum_nb at 0x7fea273d41e0>)

>>> jitted_func.targetoptions
{'nopython': False, 'nogil': True, 'parallel': False, 'boundscheck': False}

>>> vbt.jit_reg.jitted_setups[hash_nb]
{6326224984503844995: JittedSetup(jitter=<vectorbtpro.utils.jitting.NumbaJitter object at 0x7fea214d0ba8>, jitted_func=CPUDispatcher(<function sum_nb at 0x7fea273d41e0>)),
 -2979374923679407948: JittedSetup(jitter=<vectorbtpro.utils.jitting.NumbaJitter object at 0x7fea00bf94e0>, jitted_func=CPUDispatcher(<function sum_nb at 0x7fea273d41e0>))}
```

## Templates

Templates can be used to dynamically select the jitter or its keyword arguments based on the current context.
For example, you can choose the NumPy jitter over others if multiple options are available for a given task:

```pycon
>>> vbt.jit_reg.resolve('sum', jitter=vbt.RepEval("'nb' if 'nb' in task_setups else None"))
CPUDispatcher(<function sum_nb at 0x7fea273d41e0>)
```

## Disabling

To disable jitting, pass `disable=True` to `JITRegistry.resolve`:

```pycon
>>> py_func = vbt.jit_reg.resolve('sum', jitter='nb', disable=True)
>>> py_func
<function __main__.sum_nb(a)>
```

You can also disable jitting globally:

```pycon
>>> vbt.settings.jitting['disable'] = True

>>> vbt.jit_reg.resolve('sum', jitter='nb')
<function __main__.sum_nb(a)>
```

!!! hint
    If no additional options are used and only one jitter is registered per task,
    you can disable resolution to improve performance.

!!! warning
    Disabling jitting globally only affects functions resolved via `JITRegistry.resolve`.
    Any decorated function that is directly invoked will execute normally.

## Jitted option

Since most vectorbtpro functions that call other jitted functions accept a `jitted` argument,
you can provide `jitted` as a dictionary with options, as a string denoting the jitter,
or as False to disable jitting (see `vectorbtpro.utils.jitting.resolve_jitted_option`):

```pycon
>>> def sum_arr(arr, jitted=None):
...     func = vbt.jit_reg.resolve_option('sum', jitted)
...     return func(arr)

>>> arr = np.random.uniform(size=1000000)

>>> %timeit sum_arr(arr, jitted='np')
319 µs ± 3.35 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

>>> %timeit sum_arr(arr, jitted='nb')
1.09 ms ± 4.13 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

>>> %timeit sum_arr(arr, jitted=dict(jitter='nb', disable=True))
133 ms ± 2.32 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

!!! hint
    As a rule of thumb, when a function accepts a `jitted` parameter, the jitted functions it
    calls are typically resolved using `JITRegistry.resolve_option`.

## Changing options upon registration

Options are typically specified during registration using `register_jitted`:

```pycon
>>> from numba import prange

>>> @vbt.register_jitted(parallel=True, tags={'can_parallel'})
... def sum_parallel_nb(a):
...     out = np.empty(a.shape[1])
...     for col in prange(a.shape[1]):
...         total = 0.
...         for i in range(a.shape[0]):
...             total += a[i, col]
...         out[col] = total
...     return out

>>> sum_parallel_nb.targetoptions
{'nopython': True, 'nogil': True, 'parallel': True, 'boundscheck': False}
```

If you want to change the registration options for vectorbtpro's built-in jitable functions,
such as `vectorbtpro.generic.nb.base.diff_nb`, you can update the settings.
For example, to disable caching for all Numba functions:

```pycon
>>> vbt.settings.jitting.jitters['nb']['override_options'] = dict(cache=False)
```

Since the functions have already been registered, this change will have no effect:

```pycon
>>> vbt.jit_reg.jitable_setups['vectorbtpro.generic.nb.base.diff_nb']['nb'].jitter_kwargs
{'cache': True}
```

To apply the changes, save the settings to a file and load them before any functions are imported:

```pycon
>>> vbt.settings.save('my_settings')
```

Then restart the runtime and set the settings path before importing vectorbtpro:

```pycon
>>> import os
>>> os.environ['VBT_SETTINGS_PATH'] = "my_settings"

>>> from vectorbtpro import *
>>> vbt.jit_reg.jitable_setups['vectorbtpro.generic.nb.base.diff_nb']['nb'].jitter_kwargs
{'cache': False}
```

You can also modify registration options for specific tasks or even replace Python functions.
For example, to change the default `ddof` from 0 to 1 in `vectorbtpro.generic.nb.base.nanstd_1d_nb`
and disable caching with Numba:

```pycon
>>> vbt.nb.nanstd_1d_nb(np.array([1, 2, 3]))
0.816496580927726

>>> def new_nanstd_1d_nb(arr, ddof=1):
...     return np.sqrt(vbt.nb.nanvar_1d_nb(arr, ddof=ddof))

>>> vbt.settings.jitting.jitters['nb']['tasks']['vectorbtpro.generic.nb.base.nanstd_1d_nb'] = dict(
...     replace_py_func=new_nanstd_1d_nb,
...     override_options=dict(
...         cache=False
...     )
... )

>>> vbt.settings.save('my_settings')
```

After restarting the runtime:

```pycon
>>> import os
>>> os.environ['VBT_SETTINGS_PATH'] = "my_settings"

>>> vbt.nb.nanstd_1d_nb(np.array([1, 2, 3]))
1.0
```

!!! note
    All the above examples require saving settings to a file, restarting the runtime,
    setting the environment variable `VBT_SETTINGS_PATH` to the file path, and then importing vectorbtpro.

## Changing options upon resolution

Alternatively, you can modify options during resolution using `JITRegistry.resolve_option`:

```pycon
>>> # For a specific Numba function
>>> vbt.settings.jitting.jitters['nb']['tasks']['vectorbtpro.generic.nb.base.diff_nb'] = dict(
...     resolve_kwargs=dict(
...         nogil=False
...     )
... )

>>> # The 'nogil' option is disabled
>>> vbt.jit_reg.resolve('vectorbtpro.generic.nb.base.diff_nb', jitter='nb').targetoptions
{'nopython': True, 'nogil': False, 'parallel': False, 'boundscheck': False}

>>> # Unchanged for another function
>>> vbt.jit_reg.resolve('sum', jitter='nb').targetoptions
{'nopython': True, 'nogil': True, 'parallel': False, 'boundscheck': False}

>>> # For each Numba function
>>> vbt.settings.jitting.jitters['nb']['resolve_kwargs'] = dict(nogil=False)

>>> # The 'nogil' option is disabled for diff_nb
>>> vbt.jit_reg.resolve('vectorbtpro.generic.nb.base.diff_nb', jitter='nb').targetoptions
{'nopython': True, 'nogil': False, 'parallel': False, 'boundscheck': False}

>>> # And for 'sum'
>>> vbt.jit_reg.resolve('sum', jitter='nb').targetoptions
{'nopython': True, 'nogil': False, 'parallel': False, 'boundscheck': False}
```

## Building custom jitters

Below is an example of creating a custom jitter based on `vectorbtpro.utils.jitting.NumbaJitter`
that converts any argument containing a Pandas object into a 2-dimensional NumPy array before decoration:

```pycon
>>> from functools import wraps
>>> from vectorbtpro.utils.jitting import NumbaJitter

>>> class SafeNumbaJitter(NumbaJitter):
...     def decorate(self, py_func, tags=None):
...         if self.wrapping_disabled:
...             return py_func
...
...         @wraps(py_func)
...         def wrapper(*args, **kwargs):
...             new_args = ()
...             for arg in args:
...                 if isinstance(arg, pd.Series):
...                     arg = np.expand_dims(arg.values, 1)
...                 elif isinstance(arg, pd.DataFrame):
...                     arg = arg.values
...                 new_args += (arg,)
...             new_kwargs = dict()
...             for k, v in kwargs.items():
...                 if isinstance(v, pd.Series):
...                     v = np.expand_dims(v.values, 1)
...                 elif isinstance(v, pd.DataFrame):
...                     v = v.values
...                 new_kwargs[k] = v
...             return NumbaJitter.decorate(self, py_func, tags=tags)(*new_args, **new_kwargs)
...         return wrapper
```

After defining the custom jitter class, register it globally:

```pycon
>>> vbt.settings.jitting.jitters['safe_nb'] = dict(cls=SafeNumbaJitter)
```

Finally, execute any Numba function using your new jitter:

```pycon
>>> func = vbt.jit_reg.resolve(
...     task_id_or_func=vbt.generic.nb.diff_nb,
...     jitter='safe_nb',
...     allow_new=True
... )
>>> func(pd.DataFrame([[1, 2], [3, 4]]))
array([[nan, nan],
       [ 2.,  2.]])
```

Using the vanilla Numba jitter with the same function results in an error:

```pycon
>>> func = vbt.jit_reg.resolve(task_id_or_func=vbt.generic.nb.diff_nb)
>>> func(pd.DataFrame([[1, 2], [3, 4]]))
Failed in nopython mode pipeline (step: nopython frontend)
non-precise type pyobject
```

!!! note
    Ensure you pass a function as `task_id_or_func` if the jitted function has not been registered yet.

    This custom jitter cannot be used to decorate Numba functions that are intended to be called
    from within other Numba functions, as the conversion is performed in Python.

!!! info
    For default settings, see `vectorbtpro._settings.jitting`.
"""

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import atomic_dict, merge_dicts
from vectorbtpro.utils.jitting import (
    Jitter,
    get_func_suffix,
    get_id_of_jitter_type,
    resolve_jitted_kwargs,
    resolve_jitter,
    resolve_jitter_type,
)
from vectorbtpro.utils.template import CustomTemplate, RepEval, substitute_templates

__all__ = [
    "JITRegistry",
    "jit_reg",
    "register_jitted",
]


def get_func_full_name(func: tp.Callable) -> str:
    """Return the full name of the given function for use as a task identifier.

    Concatenates the function's module and name.

    Args:
        func (Callable): Function for which to retrieve the full name.

    Returns:
        str: Full name of the function.
    """
    return func.__module__ + "." + func.__name__


@define
class JitableSetup(DefineMixin):
    """Class representing a jitable setup used for just-in-time compilation.

    !!! note
        The instance is hashed solely based on `task_id` and `jitter_id`.
    """

    task_id: tp.Hashable = define.field()
    """Task identifier."""

    jitter_id: tp.Hashable = define.field()
    """Jitter identifier."""

    py_func: tp.Callable = define.field()
    """Python function to be JIT-compiled."""

    jitter_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments for configuring the jitter.

    See `vectorbtpro.utils.jitting.resolve_jitter`."""

    tags: tp.SetLike = define.field(default=None)
    """Set of tags used for categorization."""

    @staticmethod
    def get_hash(task_id: tp.Hashable, jitter_id: tp.Hashable) -> int:
        return hash((task_id, jitter_id))

    @property
    def hash_key(self) -> tuple:
        return (self.task_id, self.jitter_id)


@define
class JittedSetup(DefineMixin):
    """Class representing a JIT-decorated setup.

    !!! note
        The hash is computed solely based on the sorted configuration of `jitter`.
        Two jitters with identical configurations yield the same hash,
        preventing redundant decoration.
    """

    jitter: Jitter = define.field()
    """Jitter instance used to decorate the function."""

    jitted_func: tp.Callable = define.field()
    """Function decorated using JIT."""

    @staticmethod
    def get_hash(jitter: Jitter) -> int:
        return hash(tuple(sorted(jitter.config.items())))

    @property
    def hash_key(self) -> tuple:
        return tuple(sorted(self.jitter.config.items()))


class JITRegistry(Base):
    """Class for registering jitted functions."""

    def __init__(self) -> None:
        self._jitable_setups = {}
        self._jitted_setups = {}

    @property
    def jitable_setups(self) -> tp.Dict[tp.Hashable, tp.Dict[tp.Hashable, JitableSetup]]:
        """Dictionary of registered `JitableSetup` instances by task ID and jitter ID.

        Returns:
            Dict[Hashable, Dict[Hashable, JitableSetup]]: Mapping of task ID
                to a dictionary of jitter IDs with their `JitableSetup` instances.
        """
        return self._jitable_setups

    @property
    def jitted_setups(self) -> tp.Dict[int, tp.Dict[int, JittedSetup]]:
        """Nested dictionary of registered `JittedSetup` instances keyed by the hash of
        their associated `JitableSetup` instance.

        Returns:
            Dict[int, Dict[int, JittedSetup]]: Mapping from the hash of a jitable setup
                to a dictionary of jitted setups.
        """
        return self._jitted_setups

    def register_jitable_setup(
        self,
        task_id: tp.Hashable,
        jitter_id: tp.Hashable,
        py_func: tp.Callable,
        jitter_kwargs: tp.KwargsLike = None,
        tags: tp.Optional[set] = None,
    ) -> JitableSetup:
        """Register a jitable setup.

        Args:
            task_id (Hashable): Unique identifier for the task.
            jitter_id (Hashable): Unique identifier for the jitter type.
            py_func (Callable): Python function to decorate.
            jitter_kwargs (KwargsLike): Keyword arguments for configuring the jitter.

                See `vectorbtpro.utils.jitting.resolve_jitter`.
            tags (set): Tags associated with the function.

        Returns:
            JitableSetup: Registered `JitableSetup` instance.
        """
        jitable_setup = JitableSetup(
            task_id=task_id,
            jitter_id=jitter_id,
            py_func=py_func,
            jitter_kwargs=jitter_kwargs,
            tags=tags,
        )
        if task_id not in self.jitable_setups:
            self.jitable_setups[task_id] = dict()
        if jitter_id not in self.jitable_setups[task_id]:
            self.jitable_setups[task_id][jitter_id] = jitable_setup
        return jitable_setup

    def register_jitted_setup(
        self,
        jitable_setup: JitableSetup,
        jitter: Jitter,
        jitted_func: tp.Callable,
    ) -> JittedSetup:
        """Register a jitted setup.

        Args:
            jitable_setup (JitableSetup): Associated jitable setup instance.
            jitter (Jitter): Jitter instance used for decoration.
            jitted_func (Callable): Jitted (decorated) version of the function.

        Returns:
            JittedSetup: Registered `JittedSetup` instance.
        """
        jitable_setup_hash = hash(jitable_setup)
        jitted_setup = JittedSetup(jitter=jitter, jitted_func=jitted_func)
        jitted_setup_hash = hash(jitted_setup)
        if jitable_setup_hash not in self.jitted_setups:
            self.jitted_setups[jitable_setup_hash] = dict()
        if jitted_setup_hash not in self.jitted_setups[jitable_setup_hash]:
            self.jitted_setups[jitable_setup_hash][jitted_setup_hash] = jitted_setup
        return jitted_setup

    def decorate_and_register(
        self,
        task_id: tp.Hashable,
        py_func: tp.Callable,
        jitter: tp.Optional[tp.JitterLike] = None,
        jitter_kwargs: tp.KwargsLike = None,
        tags: tp.Optional[set] = None,
    ) -> tp.Callable:
        """Decorate a jitable function and register both jitable and jitted setups.

        Args:
            task_id (Hashable): Unique identifier for the task.
            py_func (Callable): Python function to decorate.
            jitter (JitterLike): Jitter specification used for resolving the jitter.
            jitter_kwargs (KwargsLike): Keyword arguments for configuring the jitter.

                See `vectorbtpro.utils.jitting.resolve_jitter`.
            tags (set): Tags associated with the function.

        Returns:
            Callable: Decorated jitted function.
        """
        if jitter_kwargs is None:
            jitter_kwargs = {}
        jitter = resolve_jitter(jitter=jitter, py_func=py_func, **jitter_kwargs)
        jitter_id = get_id_of_jitter_type(type(jitter))
        if jitter_id is None:
            raise ValueError("Jitter id cannot be None: is jitter registered globally?")
        jitable_setup = self.register_jitable_setup(
            task_id, jitter_id, py_func, jitter_kwargs=jitter_kwargs, tags=tags
        )
        jitted_func = jitter.decorate(py_func, tags=tags)
        self.register_jitted_setup(jitable_setup, jitter, jitted_func)
        return jitted_func

    def match_jitable_setups(
        self,
        expression: tp.Optional[str] = None,
        context: tp.KwargsLike = None,
    ) -> tp.Set[JitableSetup]:
        """Match jitable setups against an expression evaluated within each setup's context.

        Args:
            expression (str): Expression to evaluate against each setup's context.
            context (KwargsLike): Additional context for template substitution.

        Returns:
            Set[JitableSetup]: Set of jitable setups that satisfy the expression.
        """
        matched_setups = set()
        for setups_by_jitter_id in self.jitable_setups.values():
            for setup in setups_by_jitter_id.values():
                if expression is None:
                    result = True
                else:
                    result = RepEval(expression).substitute(
                        context=merge_dicts(setup.asdict(), context)
                    )
                    checks.assert_instance_of(result, bool)

                if result:
                    matched_setups.add(setup)
        return matched_setups

    def match_jitted_setups(
        self,
        jitable_setup: JitableSetup,
        expression: tp.Optional[str] = None,
        context: tp.KwargsLike = None,
    ) -> tp.Set[JittedSetup]:
        """Match jitted setups for a given jitable setup using an expression evaluated within each setup's context.

        Args:
            jitable_setup (JitableSetup): Jitable setup for which to match jitted setups.
            expression (str): Expression to evaluate against each setup's context.
            context (KwargsLike): Additional context for template substitution.

        Returns:
            Set[JittedSetup]: Set of jitted setups that satisfy the expression.
        """
        matched_setups = set()
        for setup in self.jitted_setups[hash(jitable_setup)].values():
            if expression is None:
                result = True
            else:
                result = RepEval(expression).substitute(
                    context=merge_dicts(setup.asdict(), context)
                )
                checks.assert_instance_of(result, bool)

            if result:
                matched_setups.add(setup)
        return matched_setups

    def resolve(
        self,
        task_id_or_func: tp.Union[tp.Hashable, tp.Callable],
        jitter: tp.Optional[tp.Union[tp.JitterLike, CustomTemplate]] = None,
        disable: tp.Optional[tp.Union[bool, CustomTemplate]] = None,
        disable_resolution: tp.Optional[bool] = None,
        allow_new: tp.Optional[bool] = None,
        register_new: tp.Optional[bool] = None,
        return_missing_task: bool = False,
        template_context: tp.KwargsLike = None,
        tags: tp.Optional[set] = None,
        **jitter_kwargs,
    ) -> tp.Union[tp.Hashable, tp.Callable]:
        """Resolve and return the jitted function for the specified task identifier.

        This method uses the provided task identifier or function (`task_id_or_func`) to locate
        or create a jitted function based on configured jitter settings. Templates within `jitter`,
        `disable`, and `jitter_kwargs` are substituted using a merged `template_context`.
        If `disable_resolution` is enabled, the original `task_id_or_func` is returned unchanged.

        Jitter keyword arguments are merged in the following order:

        * `jitable_setup.jitter_kwargs`
        * `jitter.your_jitter.resolve_kwargs` in `vectorbtpro._settings.jitting`
        * `jitter.your_jitter.tasks.your_task.resolve_kwargs` in `vectorbtpro._settings.jitting`
        * `jitter_kwargs`

        If no jitted setup of type `JittedSetup` is found and `allow_new` is True, the function
        is decorated and returned instead of raising an error. If `return_missing_task` is True,
        `task_id_or_func` is returned when the task id is not found in `JITRegistry.jitable_setups`.

        Args:
            task_id_or_func (Union[Hashable, Callable]): Task identifier or a function.

                For details on valid formats, see `register_jitted`.
            jitter (Optional[Union[JitterLike, CustomTemplate]]): Jitter identifier or template.
            disable (Optional[Union[bool, CustomTemplate]]): Flag or template to disable decoration.
            disable_resolution (Optional[bool]): Flag to disable the resolution process.
            allow_new (Optional[bool]): Flag to allow creating a new jitted setup if none is found.
            register_new (Optional[bool]): Flag to register a new jitted setup.
            return_missing_task (bool): If True, returns `task_id_or_func` when the task is not registered.
            template_context (KwargsLike): Additional context for template substitution.
            tags (Optional[set]): Tags associated with the function.
            **jitter_kwargs: Keyword arguments for configuring the jitter.

        Returns:
            Union[Hashable, Callable]: Either the resolved jitted function or the original task
                identifier/function based on the resolution process.

        !!! note
            The `disable` parameter is used only by `JITRegistry` and not by `vectorbtpro.utils.jitting`.

        !!! note
            If multiple jitted setups are registered for a single task id, the `jitter` parameter
            must be explicitly provided.

        !!! info
            For default settings, see `vectorbtpro._settings.jitting`.
        """
        from vectorbtpro._settings import settings

        jitting_cfg = settings["jitting"]

        if disable_resolution is None:
            disable_resolution = jitting_cfg["disable_resolution"]
        if disable_resolution:
            return task_id_or_func

        if allow_new is None:
            allow_new = jitting_cfg["allow_new"]
        if register_new is None:
            register_new = jitting_cfg["register_new"]

        if hasattr(task_id_or_func, "py_func"):
            py_func = task_id_or_func.py_func
            task_id = get_func_full_name(py_func)
        elif callable(task_id_or_func):
            py_func = task_id_or_func
            task_id = get_func_full_name(py_func)
        else:
            py_func = None
            task_id = task_id_or_func

        if task_id not in self.jitable_setups:
            if not allow_new:
                if return_missing_task:
                    return task_id_or_func
                raise KeyError(f"Task id '{task_id}' not registered")
        task_setups = self.jitable_setups.get(task_id, dict())

        template_context = merge_dicts(
            jitting_cfg["template_context"],
            template_context,
            dict(task_id=task_id, py_func=py_func, task_setups=atomic_dict(task_setups)),
        )
        jitter = substitute_templates(jitter, template_context, eval_id="jitter")

        if jitter is None and py_func is not None:
            jitter = get_func_suffix(py_func)

        if jitter is None:
            if len(task_setups) > 1:
                raise ValueError(
                    f"There are multiple registered setups for task id '{task_id}'. Please specify the jitter."
                )
            elif len(task_setups) == 0:
                raise ValueError(f"There are no registered setups for task id '{task_id}'")
            jitable_setup = list(task_setups.values())[0]
            jitter = jitable_setup.jitter_id
            jitter_id = jitable_setup.jitter_id
        else:
            jitter_type = resolve_jitter_type(jitter=jitter)
            jitter_id = get_id_of_jitter_type(jitter_type)
            if jitter_id not in task_setups:
                if not allow_new:
                    raise KeyError(
                        f"Jitable setup with task id '{task_id}' and jitter id '{jitter_id}' not registered"
                    )
                jitable_setup = None
            else:
                jitable_setup = task_setups[jitter_id]
        if jitter_id is None:
            raise ValueError("Jitter id cannot be None: is jitter registered globally?")
        if jitable_setup is None and py_func is None:
            raise ValueError(
                f"Unable to find Python function for task id '{task_id}' and jitter id '{jitter_id}'"
            )

        template_context = merge_dicts(
            template_context,
            dict(jitter_id=jitter_id, jitter=jitter, jitable_setup=jitable_setup),
        )
        disable = substitute_templates(disable, template_context, eval_id="disable")
        if disable is None:
            disable = jitting_cfg["disable"]
        if disable:
            if jitable_setup is None:
                return py_func
            return jitable_setup.py_func

        if not isinstance(jitter, Jitter):
            jitter_cfg = jitting_cfg["jitters"].get(jitter_id, {})
            setup_cfg = jitter_cfg.get("tasks", {}).get(task_id, {})

            jitter_kwargs = merge_dicts(
                jitable_setup.jitter_kwargs if jitable_setup is not None else None,
                jitter_cfg.get("resolve_kwargs", None),
                setup_cfg.get("resolve_kwargs", None),
                jitter_kwargs,
            )
            jitter_kwargs = substitute_templates(
                jitter_kwargs, template_context, eval_id="jitter_kwargs"
            )
            jitter = resolve_jitter(jitter=jitter, **jitter_kwargs)

        if jitable_setup is not None:
            jitable_hash = hash(jitable_setup)
            jitted_hash = JittedSetup.get_hash(jitter)
            if (
                jitable_hash in self.jitted_setups
                and jitted_hash in self.jitted_setups[jitable_hash]
            ):
                return self.jitted_setups[jitable_hash][jitted_hash].jitted_func
        else:
            if register_new:
                return self.decorate_and_register(
                    task_id=task_id,
                    py_func=py_func,
                    jitter=jitter,
                    jitter_kwargs=jitter_kwargs,
                    tags=tags,
                )
            return jitter.decorate(py_func, tags=tags)

        jitted_func = jitter.decorate(jitable_setup.py_func, tags=jitable_setup.tags)
        self.register_jitted_setup(jitable_setup, jitter, jitted_func)

        return jitted_func

    def resolve_option(
        self,
        task_id: tp.Union[tp.Hashable, tp.Callable],
        option: tp.JittedOption,
        **kwargs,
    ) -> tp.Union[tp.Hashable, tp.Callable]:
        """Resolve the specified jitted option and return the corresponding function.

        This method first processes the provided `option` along with additional keyword arguments using
        `vectorbtpro.utils.jitting.resolve_jitted_kwargs`. If the result is None, resolution is configured with
        `disable=True`. The resolved keyword arguments are then passed to `JITRegistry.resolve` to obtain the
        corresponding jitted function.

        Args:
            task_id (Union[Hashable, Callable]): Task identifier or a function.

                Specifies the task for which the option is applied.
            option (JittedOption): Option to control JIT compilation.
            **kwargs: Keyword arguments for `vectorbtpro.utils.jitting.resolve_jitted_kwargs`.

        Returns:
            Union[Hashable, Callable]: Resolved jitted function or the original task identifier/function.
        """
        kwargs = resolve_jitted_kwargs(option=option, **kwargs)
        if kwargs is None:
            kwargs = dict(disable=True)
        return self.resolve(task_id, **kwargs)


jit_reg = JITRegistry()
"""Default registry instance of `JITRegistry`."""


def register_jitted(
    py_func: tp.Optional[tp.Callable] = None,
    task_id_or_func: tp.Optional[tp.Union[tp.Hashable, tp.Callable]] = None,
    registry: JITRegistry = jit_reg,
    tags: tp.Optional[set] = None,
    **options,
) -> tp.Callable:
    """Decorate and register a jitable function using `JITRegistry.decorate_and_register`.

    If `task_id_or_func` is a callable, its module and function name are used as the task identifier.
    The function name may include a suffix (e.g., `_nb`) to indicate the jitter variant.

    Options are merged in the following order:

    * `jitters.{jitter_id}.options` from `vectorbtpro._settings.jitting`
    * `jitters.{jitter_id}.tasks.{task_id}.options` from `vectorbtpro._settings.jitting`
    * `options`
    * `jitters.{jitter_id}.override_options` from `vectorbtpro._settings.jitting`
    * `jitters.{jitter_id}.tasks.{task_id}.override_options` from `vectorbtpro._settings.jitting`

    `py_func` may be overridden using `jitters.your_jitter.tasks.your_task.replace_py_func`
    in `vectorbtpro._settings.jitting`.

    Args:
        py_func (Optional[Callable]): Function to be decorated.

            If None, returns the decorator.
        task_id_or_func (Optional[Union[Hashable, Callable]]): Task identifier or a callable
            from which the task identifier is derived.
        registry (JITRegistry): Registry used to register the decorated function.
        tags (Optional[set]): Tags associated with the function.
        **options: Keyword arguments for configuration.

    Returns:
        Callable: Decorated function.

    !!! info
        For default settings, see `vectorbtpro._settings.jitting`.
    """

    def decorator(_py_func: tp.Callable) -> tp.Callable:
        nonlocal options

        from vectorbtpro._settings import settings

        jitting_cfg = settings["jitting"]

        if task_id_or_func is None:
            task_id = get_func_full_name(_py_func)
        elif hasattr(task_id_or_func, "py_func"):
            task_id = get_func_full_name(task_id_or_func.py_func)
        elif callable(task_id_or_func):
            task_id = get_func_full_name(task_id_or_func)
        else:
            task_id = task_id_or_func

        jitter = options.pop("jitter", None)
        jitter_type = resolve_jitter_type(jitter=jitter, py_func=_py_func)
        jitter_id = get_id_of_jitter_type(jitter_type)

        jitter_cfg = jitting_cfg["jitters"].get(jitter_id, {})
        setup_cfg = jitter_cfg.get("tasks", {}).get(task_id, {})
        options = merge_dicts(
            jitter_cfg.get("options", None),
            setup_cfg.get("options", None),
            options,
            jitter_cfg.get("override_options", None),
            setup_cfg.get("override_options", None),
        )
        if setup_cfg.get("replace_py_func", None) is not None:
            _py_func = setup_cfg["replace_py_func"]
            if task_id_or_func is None:
                task_id = get_func_full_name(_py_func)

        return registry.decorate_and_register(
            task_id=task_id,
            py_func=_py_func,
            jitter=jitter,
            jitter_kwargs=options,
            tags=tags,
        )

    if py_func is None:
        return decorator
    return decorator(py_func)
