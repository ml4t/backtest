# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing class and function decorators."""

import threading
from functools import wraps

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

if tp.TYPE_CHECKING:
    from vectorbtpro.registries.ca_registry import CARunSetup as CARunSetupT
else:
    CARunSetupT = "vectorbtpro.registries.ca_registry.CARunSetup"

__all__ = [
    "memoized_method",
    "class_property",
    "hybrid_property",
    "hybrid_method",
    "cacheable_property",
    "cached_property",
    "cacheable",
    "cached",
    "cacheable_method",
    "cached_method",
]

__pdoc__ = {}


# ############# Generic ############# #


def memoized_method(func: tp.Callable) -> tp.Callable:
    """Memoize a method's results in a thread-safe manner using a cache keyed by method arguments
    (excluding the instance).

    Args:
        func (Callable): Function to be decorated.

    Returns:
        Callable: Memoized method.
    """
    lock = threading.Lock()
    memo = {}

    @wraps(func)
    def wrapper(*args) -> tp.Any:
        with lock:
            if args[1:] in memo:
                return memo[args[1:]]
            else:
                rv = func(*args)
                memo[args[1:]] = rv
                return rv

    return wrapper


class class_property(Base):
    """Class for defining properties accessible directly on the class.

    Args:
        func (Callable): Function to be decorated as a class property.
    """

    def __init__(self, func: tp.Callable) -> None:
        self._func = func
        self.__doc__ = func.__doc__

    @property
    def func(self) -> tp.Callable:
        """Wrapped function.

        Returns the function itself, not the result of calling it.

        Returns:
            Callable: Decorated function.
        """
        return self._func

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        return self.func(owner)

    def __set__(self, instance: object, value: tp.Any) -> None:
        raise AttributeError("can't set attribute")

    def __reduce__(self) -> tp.Union[str, tp.Tuple]:
        state = {"__doc__": self.__doc__}
        return (type(self), (self.func,), state)

    def __setstate__(self, state: dict) -> None:
        self.__doc__ = state["__doc__"]


class hybrid_property(Base):
    """Class for defining a hybrid property that binds the first parameter to the class when accessed
    via the class, and to an instance when accessed via an instance.

    Args:
        func (Callable): Function to be decorated as a hybrid property.
    """

    def __init__(self, func: tp.Callable) -> None:
        self._func = func
        self.__doc__ = func.__doc__

    @property
    def func(self) -> tp.Callable:
        """Wrapped function.

        Returns the function itself, not the result of calling it.

        Returns:
            Callable: Decorated function.
        """
        return self._func

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        if instance is None:
            return self.func(owner)
        return self.func(instance)

    def __set__(self, instance: object, value: tp.Any) -> None:
        raise AttributeError("can't set attribute")

    def __reduce__(self) -> tp.Union[str, tp.Tuple]:
        state = {"__doc__": self.__doc__}
        return (type(self), (self.func,), state)

    def __setstate__(self, state: dict) -> None:
        self.__doc__ = state["__doc__"]


class hybrid_method(classmethod, Base):
    """Class for defining a hybrid method decorator that binds the first parameter to the class when
    called as a class method, and to the instance when called as an instance method."""

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, owner)


# ############# Custom properties ############# #

custom_propertyT = tp.TypeVar("custom_propertyT", bound="custom_property")


class custom_property(property, Base):
    """Class for defining a custom extensible property that stores the decorated function and
    its configuration options as attributes.

    Args:
        func (Callable): Function to be decorated as a property.
        **options: Configuration options for the property.

    !!! note
        `custom_property` instances are defined at the class level, so modifying the property affects all instances.
    """

    def __new__(
        cls: tp.Type[custom_propertyT], *args, **options
    ) -> tp.Union[tp.Callable, custom_propertyT]:
        if len(args) == 0:
            return lambda func: cls(func, **options)
        elif len(args) == 1:
            return super().__new__(cls)
        raise ValueError("Either function or keyword arguments must be passed")

    def __init__(self, func: tp.Callable, **options) -> None:
        property.__init__(self)

        self._func = func
        self._name = func.__name__
        self._options = options
        self.__doc__ = func.__doc__

    @property
    def func(self) -> tp.Callable:
        """Wrapped function.

        Returns the function itself, not the result of calling it.

        Returns:
            Callable: Decorated function.
        """
        return self._func

    @property
    def name(self) -> str:
        """Name of the decorated function.

        Returns:
            str: Name of the decorated function.
        """
        return self._name

    @property
    def options(self) -> tp.Kwargs:
        """Configuration options provided to the property.

        Returns:
            Kwargs: Configuration options.
        """
        return self._options

    def __set_name__(self, owner: tp.Type, name: str) -> None:
        self._name = name

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        if instance is None:
            return self
        return self.func(instance)

    def __set__(self, instance: object, value: tp.Any) -> None:
        raise AttributeError("Can't set attribute")

    def __call__(self, *args, **kwargs) -> tp.Any:
        pass

    def __reduce__(self) -> tp.Union[str, tp.Tuple]:
        state = {"_name": self._name, "__doc__": self.__doc__}
        return (type(self), (self.func,), state)

    def __setstate__(self, state: dict) -> None:
        self._name = state["_name"]
        self.__doc__ = state["__doc__"]


class cacheable_property(custom_property):
    """Class for defining a cacheable property extending `custom_property` to support caching of computed values.

    Args:
        func (Callable): Function to be decorated as a cacheable property.
        use_cache (bool): Whether to use caching.
        whitelist (bool): Whether to whitelist the property.
        **options: Configuration options for the property.

    !!! note
        This property assumes the instance remains unchanged; changes to dependent attributes are not detected.

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """

    def __init__(
        self, func: tp.Callable, use_cache: bool = False, whitelist: bool = False, **options
    ) -> None:
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        super().__init__(func, **options)

        self._init_use_cache = use_cache
        self._init_whitelist = whitelist
        if not caching_cfg["register_lazily"]:
            self.get_ca_setup()

    @property
    def init_use_cache(self) -> bool:
        """Initial `use_cache` value.

        Returns:
            bool: True if caching is enabled, False otherwise.
        """
        return self._init_use_cache

    @property
    def init_whitelist(self) -> bool:
        """Initial `whitelist` flag.

        Returns:
            bool: True if whitelisting is enabled, False otherwise.
        """
        return self._init_whitelist

    def get_ca_setup(self, instance: tp.Optional[object] = None) -> tp.Optional[CARunSetupT]:
        """Retrieve the caching setup as `vectorbtpro.registries.ca_registry.CARunSetup`
        when an instance is provided, or as `vectorbtpro.registries.ca_registry.CAUnboundSetup` if not.

        See `vectorbtpro.registries.ca_registry` for details on the caching procedure.

        Args:
            instance (Optional[object]): Instance to retrieve the setup for.

        Returns:
            Optional[CARunSetup]: Caching setup instance, or None if not applicable.
        """
        from vectorbtpro.registries.ca_registry import CARunSetup, CAUnboundSetup

        unbound_setup = CAUnboundSetup.get(
            self, use_cache=self.init_use_cache, whitelist=self.init_whitelist
        )
        if instance is None:
            return unbound_setup
        return CARunSetup.get(self, instance=instance)

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        if instance is None:
            return self
        run_setup = self.get_ca_setup(instance)
        if run_setup is None:
            return self.func(instance)
        return run_setup.run()


class cached_property(cacheable_property):
    """Class for defining a cached property, equivalent to using `cacheable_property` with `use_cache=True`.

    Args:
        func (Callable): Function to be decorated as a cacheable property.
        **options: Keyword arguments for `cacheable_property`.
    """

    def __init__(self, func: tp.Callable, **options) -> None:
        cacheable_property.__init__(self, func, use_cache=True, **options)


# ############# Custom functions ############# #


class custom_functionT(tp.Protocol):
    decorator_name: str
    func: tp.Callable
    name: str
    options: tp.Kwargs
    is_method: bool
    is_custom: bool

    def __call__(*args, **kwargs) -> tp.Any:
        pass


__pdoc__["custom_functionT"] = False


def custom_function(
    *args,
    _decorator_name: tp.Optional[str] = None,
    **options,
) -> tp.Union[tp.Callable, custom_functionT]:
    """Decorator that augments functions by attaching metadata and configuration options.

    Args:
        func (Callable): Function to be decorated.
        **options: Configuration options for the function.

    Returns:
        Union[Callable, custom_functionT]: Decorated function with attached metadata.
    """

    def decorator(func: tp.Callable) -> custom_functionT:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            return func(*args, **kwargs)

        wrapper.decorator_name = "custom_function" if _decorator_name is None else _decorator_name
        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.options = options
        wrapper.is_method = False
        wrapper.is_custom = True

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


class cacheable_functionT(custom_functionT):
    is_cacheable: bool
    get_ca_setup: tp.Callable[[], tp.Optional[CARunSetupT]]


__pdoc__["cacheable_functionT"] = False


def cacheable(
    *args,
    use_cache: bool = False,
    whitelist: bool = False,
    max_size: tp.Optional[int] = None,
    ignore_args: tp.Optional[tp.Iterable[tp.AnnArgQuery]] = None,
    _decorator_name: tp.Optional[str] = None,
    **options,
) -> tp.Union[tp.Callable, cacheable_functionT]:
    """Decorate a function to enable caching functionality.

    See notes on `cacheable_property`.

    Args:
        func (Callable): Function to be decorated.
        use_cache (bool): Whether to use caching.
        whitelist (bool): Whether to whitelist the function.
        max_size (Optional[int]): Maximum size of the cache.
        ignore_args (Optional[Iterable[AnnArgQuery]]): Arguments to ignore for caching.
        **options: Configuration options for the function.

    Returns:
        Union[Callable, cacheable_function]: Decorated function with caching enabled.

    !!! note
        To decorate an instance method, use `cacheable_method`.

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """

    def decorator(func: tp.Callable) -> cacheable_functionT:
        from vectorbtpro._settings import settings
        from vectorbtpro.registries.ca_registry import CARunSetup

        caching_cfg = settings["caching"]

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            run_setup = wrapper.get_ca_setup()
            if run_setup is None:
                return func(*args, **kwargs)
            return run_setup.run(*args, **kwargs)

        def get_ca_setup() -> tp.Optional[CARunSetup]:
            """Return the caching run setup instance of type `vectorbtpro.registries.ca_registry.CARunSetup`.

            See `vectorbtpro.registries.ca_registry` for details on the caching procedure.

            Returns:
                Optional[CARunSetup]: Caching run setup instance, or None if not applicable.
            """
            return CARunSetup.get(
                wrapper,
                use_cache=use_cache,
                whitelist=whitelist,
                max_size=max_size,
                ignore_args=ignore_args,
            )

        wrapper.decorator_name = "cacheable" if _decorator_name is None else _decorator_name
        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.options = options
        wrapper.is_method = False
        wrapper.is_custom = True
        wrapper.is_cacheable = True
        wrapper.get_ca_setup = get_ca_setup
        if not caching_cfg["register_lazily"]:
            wrapper.get_ca_setup()

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


def cached(*args, **options) -> tp.Union[tp.Callable, cacheable_functionT]:
    """Decorate a function with caching enabled, equivalent to using `cacheable` with `use_cache=True`.

    Args:
        func (Callable): Function to be decorated.
        **options: Configuration options for the function.

    Returns:
        Union[Callable, cacheable_function]: Decorated function with caching enabled.

    !!! note
        To decorate an instance method, use `cached_method`.
    """
    return cacheable(*args, use_cache=True, _decorator_name="cached", **options)


# ############# Custom methods ############# #


class custom_methodT(custom_functionT):
    def __call__(instance: object, *args, **kwargs) -> tp.Any:
        pass


__pdoc__["custom_methodT"] = False


def custom_method(
    *args,
    _decorator_name: tp.Optional[str] = None,
    **options,
) -> tp.Union[tp.Callable, custom_methodT]:
    """Decorate an instance method with custom behavior.

    Args:
        func (Callable): Function to be decorated.
        **options: Configuration options for the method.

    Returns:
        Union[Callable, custom_method]: Decorated method with attached metadata.
    """

    def decorator(func: tp.Callable) -> custom_methodT:
        @wraps(func)
        def wrapper(instance: object, *args, **kwargs) -> tp.Any:
            return func(instance, *args, **kwargs)

        wrapper.decorator_name = "custom_method" if _decorator_name is None else _decorator_name
        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.options = options
        wrapper.is_method = True
        wrapper.is_custom = True

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


class cacheable_methodT(custom_methodT):
    get_ca_setup: tp.Callable[[tp.Optional[object]], tp.Optional[CARunSetupT]]


__pdoc__["cacheable_methodT"] = False


def cacheable_method(
    *args,
    use_cache: bool = False,
    whitelist: bool = False,
    max_size: tp.Optional[int] = None,
    ignore_args: tp.Optional[tp.Iterable[tp.AnnArgQuery]] = None,
    _decorator_name: tp.Optional[str] = None,
    **options,
) -> tp.Union[tp.Callable, cacheable_methodT]:
    """Decorate an instance method to enable caching functionality.

    See notes on `cacheable_property`.

    Args:
        func (Callable): Function to be decorated.
        use_cache (bool): Whether to use caching.
        whitelist (bool): Whether to whitelist the method.
        max_size (Optional[int]): Maximum size of the cache.
        ignore_args (Optional[Iterable[AnnArgQuery]]): Arguments to ignore for caching.
        **options: Configuration options for the method.

    Returns:
        Union[Callable, cacheable_method]: Decorated method with caching enabled.

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """

    def decorator(func: tp.Callable) -> cacheable_methodT:
        from vectorbtpro._settings import settings
        from vectorbtpro.registries.ca_registry import CARunSetup, CAUnboundSetup

        caching_cfg = settings["caching"]

        @wraps(func)
        def wrapper(instance: object, *args, **kwargs) -> tp.Any:
            run_setup = wrapper.get_ca_setup(instance)
            if run_setup is None:
                return func(instance, *args, **kwargs)
            return run_setup.run(*args, **kwargs)

        def get_ca_setup(instance: tp.Optional[object] = None) -> tp.Optional[CARunSetup]:
            """Return the caching run setup instance of type `vectorbtpro.registries.ca_registry.CARunSetup`
            if an instance is provided, or return the unbound caching setup of type
            `vectorbtpro.registries.ca_registry.CAUnboundSetup` if not.

            See `vectorbtpro.registries.ca_registry` for details on the caching procedure.

            Args:
                instance (Optional[object]): Instance to retrieve the setup for.

            Returns:
                Optional[CARunSetup]: Caching run setup instance, or None if not applicable.
            """
            unbound_setup = CAUnboundSetup.get(wrapper, use_cache=use_cache, whitelist=whitelist)
            if instance is None:
                return unbound_setup
            return CARunSetup.get(
                wrapper, instance=instance, max_size=max_size, ignore_args=ignore_args
            )

        wrapper.decorator_name = "cacheable_method" if _decorator_name is None else _decorator_name
        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.options = options
        wrapper.is_method = True
        wrapper.is_custom = True
        wrapper.is_cacheable = True
        wrapper.get_ca_setup = get_ca_setup
        if not caching_cfg["register_lazily"]:
            wrapper.get_ca_setup()

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


def cached_method(*args, **options) -> tp.Union[tp.Callable, cacheable_methodT]:
    """Decorate an instance method with caching enabled, equivalent to using
    `cacheable_method` with `use_cache=True`.

    Args:
        func (Callable): Method to be decorated.
        **options: Configuration options for the method.

    Returns:
        Union[Callable, cacheable_method]: Decorated method with caching enabled.
    """
    return cacheable_method(*args, use_cache=True, _decorator_name="cached_method", **options)


cacheableT = tp.Union[cacheable_property, cacheable_functionT, cacheable_methodT]
