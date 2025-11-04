# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for jitting.

!!! info
    For default settings, see `vectorbtpro._settings.jitting` and `vectorbtpro._settings.numba`.
"""

from numba import jit as nb_jit

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import Configured, merge_dicts

__all__ = [
    "jitted",
]


class Jitter(Configured):
    """Abstract class for jitting function decoration.

    Represents a configuration for jitting.

    When overriding `decorate`, ensure to check whether wrapping is globally disabled using `wrapping_disabled`.
    """

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    @property
    def wrapping_disabled(self) -> bool:
        """Global flag indicating whether jitting wrapping is disabled.

        Returns:
            bool: True if jitting wrapping is disabled, False otherwise.

        !!! info
            For default settings, see `vectorbtpro._settings.jitting`.
        """
        from vectorbtpro._settings import settings

        jitting_cfg = settings["jitting"]

        return jitting_cfg["disable_wrapping"]

    def decorate(self, py_func: tp.Callable, tags: tp.Optional[set] = None) -> tp.Callable:
        """Apply jitting decoration to a Python function.

        Args:
            py_func (Callable): Python function to decorate.
            tags (Optional[set]): Tags associated with the function.

        Returns:
            Callable: Decorated function.

        !!! abstract
            This method should be overridden in a subclass.
        """
        if self.wrapping_disabled:
            return py_func
        raise NotImplementedError


class NumPyJitter(Jitter):
    """Class for decorating functions that use NumPy.

    The decoration is a no-op and returns the original function unmodified.
    """

    def decorate(self, py_func: tp.Callable, tags: tp.Optional[set] = None) -> tp.Callable:
        return py_func


class NumbaJitter(Jitter):
    """Class for decorating functions using Numba.

    Args:
        fix_cannot_parallel (bool): Flag indicating whether to disable parallel execution if
            the 'can_parallel' tag is missing.
        nopython (bool): Flag indicating whether functions should be compiled in nopython mode.
        nogil (bool): Flag indicating whether to release the Global Interpreter Lock (GIL) during execution.
        parallel (bool): Flag indicating whether automatic parallelization is enabled.
        cache (bool): Flag indicating whether the compiled function should be cached on disk.
        boundscheck (bool): Flag indicating whether array bounds checking is enabled.
        **options: Keyword arguments provided to the Numba decorator.

    !!! note
        If `fix_cannot_parallel` is True, then `parallel=True` is ignored when the 'can_parallel' tag is absent.
    """

    def __init__(
        self,
        fix_cannot_parallel: bool = True,
        nopython: bool = True,
        nogil: bool = True,
        parallel: bool = False,
        cache: bool = False,
        boundscheck: bool = False,
        **options,
    ) -> None:
        Jitter.__init__(
            self,
            fix_cannot_parallel=fix_cannot_parallel,
            nopython=nopython,
            nogil=nogil,
            parallel=parallel,
            cache=cache,
            boundscheck=boundscheck,
            **options,
        )

        self._fix_cannot_parallel = fix_cannot_parallel
        self._nopython = nopython
        self._nogil = nogil
        self._parallel = parallel
        self._cache = cache
        self._boundscheck = boundscheck
        self._options = options

    @property
    def fix_cannot_parallel(self) -> bool:
        """Flag indicating whether to disable parallel execution if the 'can_parallel' tag is missing.

        Returns:
            bool: True if parallel execution should be disabled when 'can_parallel' is absent, False otherwise.
        """
        return self._fix_cannot_parallel

    @property
    def nopython(self) -> bool:
        """Flag indicating whether functions should be compiled in nopython mode.

        Returns:
            bool: True if nopython mode is enabled, False otherwise.
        """
        return self._nopython

    @property
    def nogil(self) -> bool:
        """Flag indicating whether to release the Global Interpreter Lock (GIL) during execution.

        Returns:
            bool: True if the GIL should be released during execution, False otherwise.
        """
        return self._nogil

    @property
    def parallel(self) -> bool:
        """Flag indicating whether automatic parallelization is enabled.

        Returns:
            bool: True if parallel execution is enabled, False otherwise.
        """
        return self._parallel

    @property
    def boundscheck(self) -> bool:
        """Flag indicating whether array bounds checking is enabled.

        Returns:
            bool: True if array bounds checking is turned on, False otherwise.
        """
        return self._boundscheck

    @property
    def cache(self) -> bool:
        """Flag indicating whether the compiled function should be cached on disk.

        Returns:
            bool: True if caching of the compiled function is enabled, False otherwise.
        """
        return self._cache

    @property
    def options(self) -> tp.Kwargs:
        """Dictionary of additional keyword arguments provided to the Numba decorator.

        Returns:
            Kwargs: Dictionary containing extra options for the Numba decorator.
        """
        return self._options

    def decorate(self, py_func: tp.Callable, tags: tp.Optional[set] = None) -> tp.Callable:
        if self.wrapping_disabled:
            return py_func

        if tags is None:
            tags = set()
        options = dict(self.options)
        parallel = self.parallel
        if self.fix_cannot_parallel and parallel and "can_parallel" not in tags:
            parallel = False
        cache = self.cache
        if parallel and cache:
            cache = False
        return nb_jit(
            nopython=self.nopython,
            nogil=self.nogil,
            parallel=parallel,
            cache=cache,
            boundscheck=self.boundscheck,
            **options,
        )(py_func)


def get_func_suffix(py_func: tp.Callable) -> tp.Optional[str]:
    """Retrieve the suffix from a function's name if it corresponds to a registered jitter configuration.

    Args:
        py_func (Callable): Python function to decorate.

    Returns:
        Optional[str]: Suffix in lowercase if recognized; otherwise, None.

    !!! info
        For default settings, see `vectorbtpro._settings.jitting`.
    """
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    splitted_name = py_func.__name__.split("_")
    if len(splitted_name) == 1:
        return None
    suffix = splitted_name[-1].lower()
    if suffix not in jitting_cfg["jitters"]:
        return None
    return suffix


def resolve_jitter_type(
    jitter: tp.Optional[tp.JitterLike] = None,
    py_func: tp.Optional[tp.Callable] = None,
) -> tp.Type[Jitter]:
    """Resolve the jitter type based on the provided parameter.

    Args:
        jitter (Optional[JitterLike]): Identifier, subclass, or instance of `Jitter`.

            * If `jitter` is None and a Python function is provided, infer the jitter type from
                the function's name via `get_func_suffix`.
            * If `jitter` is a string, retrieve the corresponding jitter class from
                `vectorbtpro._settings.jitting`.
            * If `jitter` is a subclass of `Jitter` or an instance thereof, return the appropriate class.
            * Otherwise, an error is raised.
        py_func (Optional[Callable]): Function used to infer the jitter type if `jitter` is None

    Returns:
        Type[Jitter]: Resolved jitter class.

    !!! info
        For default settings, see `vectorbtpro._settings.jitting`.
    """
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    if jitter is None:
        if py_func is None:
            raise ValueError("Could not parse jitter without a function")
        jitter = get_func_suffix(py_func)
        if jitter is None:
            raise ValueError(f"Could not parse jitter from suffix of function {py_func}")

    if isinstance(jitter, str):
        if jitter in jitting_cfg["jitters"]:
            jitter = jitting_cfg["jitters"][jitter]["cls"]
        else:
            found = False
            for k, v in jitting_cfg["jitters"].items():
                if jitter in v.get("aliases", set()):
                    jitter = v["cls"]
                    found = True
                    break
            if not found:
                raise ValueError(f"Jitter with name '{jitter}' not registered")
    if isinstance(jitter, str):
        globals_dict = globals()
        if jitter in globals_dict:
            jitter = globals_dict[jitter]
        else:
            raise ValueError(f"Invalid jitter name: '{jitter}'")
    if isinstance(jitter, type) and issubclass(jitter, Jitter):
        return jitter
    if isinstance(jitter, Jitter):
        return type(jitter)
    raise TypeError(f"Jitter type {jitter} is not supported")


def get_id_of_jitter_type(jitter_type: tp.Type[Jitter]) -> tp.Optional[tp.Hashable]:
    """Retrieve the identifier of a jitter type from the configuration in `vectorbtpro._settings.jitting`.

    Args:
        jitter_type (Type[Jitter]): Jitter class to look up.

    Returns:
        Optional[Hashable]: Identifier if found; otherwise, None.

    !!! info
        For default settings, see `vectorbtpro._settings.jitting`.
    """
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    for jitter_id, jitter_cfg in jitting_cfg["jitters"].items():
        jitter_cls = jitter_cfg["cls"]
        if isinstance(jitter_cls, str):
            globals_dict = globals()
            if jitter_cls in globals_dict:
                jitter_cls = globals_dict[jitter_cls]
            else:
                raise ValueError(f"Invalid jitter name: '{jitter_cls}'")
        if jitter_type is jitter_cls:
            return jitter_id
    return None


def resolve_jitted_option(option: tp.JittedOption = None) -> tp.KwargsLike:
    """Resolve and return keyword arguments for jitting based on the provided option.

    Args:
        option (JittedOption): Option to control JIT compilation.

            * True: Apply jitting with default settings.
            * False: Disable jitting (returns None).
            * str: Use the option as the jitter name.
            * dict: Interpret the option as a dictionary of keyword arguments for jitting.

    Returns:
        KwargsLike: Dictionary of keyword arguments for jitting, or None if jitting is disabled.

    !!! info
        For default settings, see `vectorbtpro._settings.jitting`.
    """
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    if option is None:
        option = jitting_cfg["option"]

    if isinstance(option, bool):
        if not option:
            return None
        return dict()
    if isinstance(option, dict):
        return option
    elif isinstance(option, str):
        return dict(jitter=option)
    raise TypeError(f"Type {type(option)} is invalid for a jitting option")


def specialize_jitted_option(option: tp.JittedOption = None, **kwargs) -> tp.KwargsLike:
    """Resolve a jitted option by merging its resolved keyword arguments with additional keyword arguments.

    Args:
        option (JittedOption): Option to control JIT compilation.

            See `resolve_jitted_option`.
        **kwargs: Keyword arguments to merge.

    Returns:
        KwargsLike: Merged dictionary of keyword arguments, or None if the option cannot be resolved.
    """
    jitted_kwargs = resolve_jitted_option(option)
    if jitted_kwargs is None:
        return None
    return merge_dicts(kwargs, jitted_kwargs)


def resolve_jitted_kwargs(option: tp.JittedOption = None, **kwargs) -> tp.KwargsLike:
    """Resolve keyword arguments for a jitted function by resolving an option and merging it with
    additional keyword arguments.

    Args:
        option (JittedOption): Option to control JIT compilation.

            See `resolve_jitted_option`.
        **kwargs: Keyword arguments to merge.

    Returns:
        KwargsLike: Merged dictionary of keyword arguments with keys from `option` taking precedence.

    !!! note
        Keys in `option` have more priority than in `kwargs`.

    !!! info
        For default settings, see `vectorbtpro._settings.jitting`.
    """
    from vectorbtpro._settings import settings

    jitting_cfg = settings["jitting"]

    jitted_kwargs = resolve_jitted_option(option=option)
    if jitted_kwargs is None:
        return None
    if isinstance(jitting_cfg["option"], dict):
        jitted_kwargs = merge_dicts(jitting_cfg["option"], kwargs, jitted_kwargs)
    else:
        jitted_kwargs = merge_dicts(kwargs, jitted_kwargs)
    return jitted_kwargs


def resolve_jitter(
    jitter: tp.Optional[tp.JitterLike] = None,
    py_func: tp.Optional[tp.Callable] = None,
    **jitter_kwargs,
) -> Jitter:
    """Resolve a jitter instance.

    Args:
        jitter (Optional[JitterLike]): Identifier, subclass, or instance of `Jitter`.

            See `resolve_jitter_type`.
        py_func (Optional[Callable]): Function used to infer the jitter type if `jitter` is None
        **jitter_kwargs: Keyword arguments for configuring the jitter.

    Returns:
        Jitter: Jitter instance based on the resolved configuration.

    !!! note
        If `jitter` is already an instance of `Jitter` and there are other keyword arguments, they are discarded.
    """
    if not isinstance(jitter, Jitter):
        jitter_type = resolve_jitter_type(jitter=jitter, py_func=py_func)
        jitter = jitter_type(**jitter_kwargs)
    return jitter


def jitted(*args, tags: tp.Optional[set] = None, **jitted_kwargs) -> tp.Callable:
    """Decorate a jitable function by applying jitting to the wrapped function.

    The jitting configuration is resolved using `resolve_jitter`. The wrapping mechanism can be disabled by
    setting the global `disable_wrapping` option.

    Args:
        *args: Positional arguments for the decorator.
        tags (Optional[set]): Tags associated with the function.
        **jitted_kwargs: Keyword arguments to resolve `jitter`.

    Returns:
        Callable: Decorated function with jitting applied.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> @vbt.jitted
        ... def my_func_nb():
        ...     total = 0
        ...     for i in range(1000000):
        ...         total += 1
        ...     return total

        >>> %timeit my_func_nb()
        68.1 ns ± 0.32 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
        ```

        Jitter is automatically detected using the suffix of the wrapped function.
    """

    def decorator(py_func: tp.Callable) -> tp.Callable:
        jitter = resolve_jitter(py_func=py_func, **jitted_kwargs)
        return jitter.decorate(py_func, tags=tags)

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
