# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a global registry for cacheable functions.

This module provides a global registry that manages caching setups for functions decorated
with cacheable decorators. In vectorbtpro, caching is implemented by combining decorators with a registry.
Cacheable decorators such as `vectorbtpro.utils.decorators.cacheable` wrap a function to integrate caching
without storing results directly. Instead, they register a caching "setup" (see `CARunSetup`) with the registry,
which handles caching behavior and metrics.

## Runnable setups

A runnable setup encapsulates the execution of a cacheable function by:

* Determining whether the result should be cached.
* Executing the function with the provided arguments.
* Storing the result in the cache.
* Updating metrics such as hits, misses, and elapsed time.
* Returning the cached or computed result to the calling wrapper.

Each setup is stateful and uniquely identified by its associated function (via hashing), ensuring that only one
setup is registered globally per cacheable function.

```pycon
>>> from vectorbtpro import *

>>> my_func = lambda: np.random.uniform(size=1000000)

>>> # Decorator returns a wrapper
>>> my_ca_func = vbt.cached(my_func)

>>> # Wrapper registers a new setup
>>> my_ca_func.get_ca_setup()
CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<function <lambda> at 0x7fe14e94cae8>, instance=None, max_size=None, ignore_args=None, cache={})

>>> # Subsequent calls return the existing setup
>>> my_ca_func.get_ca_setup()
CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<function <lambda> at 0x7fe14e94cae8>, instance=None, max_size=None, ignore_args=None, cache={})

>>> # Only one CARunSetup object per wrapper and instance binding
>>> hash(my_ca_func.get_ca_setup()) == hash((my_ca_func, None))
True
```

When `my_ca_func` is called, it retrieves the setup from the registry and invokes `CARunSetup.run`.
The caching process is entirely managed by the setup, remaining transparent to `my_ca_func`.
To inspect caching metrics or access the cache, query the setup as shown below:

```pycon
>>> my_setup = my_ca_func.get_ca_setup()

>>> # Cache is initially empty
>>> my_setup.get_stats()
{
    'hash': 4792160544297109364,
    'string': '<bound func __main__.<lambda>>',
    'use_cache': True,
    'whitelist': False,
    'caching_enabled': True,
    'hits': 0,
    'misses': 0,
    'total_size': '0 Bytes',
    'total_elapsed': None,
    'total_saved': None,
    'first_run_time': None,
    'last_run_time': None,
    'first_hit_time': None,
    'last_hit_time': None,
    'creation_time': 'now',
    'last_update_time': None
}

>>> # After execution, the result is cached
>>> my_ca_func()
>>> my_setup.get_stats()
{
    'hash': 4792160544297109364,
    'string': '<bound func __main__.<lambda>>',
    'use_cache': True,
    'whitelist': False,
    'caching_enabled': True,
    'hits': 0,
    'misses': 1,
    'total_size': '8.0 MB',
    'total_elapsed': '11.33 milliseconds',
    'total_saved': '0 milliseconds',
    'first_run_time': 'now',
    'last_run_time': 'now',
    'first_hit_time': None,
    'last_hit_time': None,
    'creation_time': 'now',
    'last_update_time': None
}

>>> # Retrieving the cached result increases the hit count
>>> my_ca_func()
>>> my_setup.get_stats()
{
    'hash': 4792160544297109364,
    'string': '<bound func __main__.<lambda>>',
    'use_cache': True,
    'whitelist': False,
    'caching_enabled': True,
    'hits': 1,
    'misses': 1,
    'total_size': '8.0 MB',
    'total_elapsed': '11.33 milliseconds',
    'total_saved': '11.33 milliseconds',
    'first_run_time': 'now',
    'last_run_time': 'now',
    'first_hit_time': 'now',
    'last_hit_time': 'now',
    'creation_time': 'now',
    'last_update_time': None
}
```

## Enabling/disabling caching

To enable or disable caching, we can invoke `CARunSetup.enable_caching` and `CARunSetup.disable_caching`
respectively. This will set `CARunSetup.use_cache` flag to True or False. Even though we expressed
our disire to change caching rules, the final decision also depends on the global settings and
whether the setup is whitelisted in case caching is disabled globally. This decision is available
via `CARunSetup.caching_enabled`:

```pycon
>>> my_setup.disable_caching()
>>> my_setup.caching_enabled
False

>>> my_setup.enable_caching()
>>> my_setup.caching_enabled
True

>>> vbt.settings.caching['disable'] = True
>>> my_setup.caching_enabled
False

>>> my_setup.enable_caching()
UserWarning: This operation has no effect: caching is disabled globally and this setup is not whitelisted

>>> my_setup.enable_caching(force=True)
>>> my_setup.caching_enabled
True

>>> vbt.settings.caching['disable_whitelist'] = True
>>> my_setup.caching_enabled
False

>>> my_setup.enable_caching(force=True)
UserWarning: This operation has no effect: caching and whitelisting are disabled globally
```

To disable registration of new setups completely, use `disable_machinery`:

```pycon
>>> vbt.settings.caching['disable_machinery'] = True
```

## Setup hierarchy

Consider how to change caching rules for an entire instance or class at once. Even if every cacheable
function declared in a class is updated, ensuring that each future subclass or instance inherits
these changes can be challenging. To address this, vectorbtpro provides a set of stateful setups that
delegate operations to their child setups—all the way down to `CARunSetup`. The setup hierarchy
mirrors the inheritance hierarchy in object‐oriented programming.

![](/assets/images/api/setup_hierarchy.svg){: loading=lazy style="width:700px;" }

For example, calling `B.get_ca_setup().disable_caching()` disables caching for every current and
future subclass and instance of `B`, while leaving `A` or any other superclass of `B` unaffected.
Subsequently, each instance of `B` disables caching for every cacheable property and method.
In essence, the operation propagates from top to bottom.

Unbound setups appear outside their classes in the diagram because it is not straightforward to
deduce the class when using a cacheable decorator; therefore, these functions are treated as independent.
When you invoke `B.f.get_ca_setup().disable_caching()`, caching is disabled for the function `B.f`
for every current and future subclass and instance of `B`, with all other functions remaining unchanged.

What happens if caching is enabled for class `B` but disabled for the unbound function `B.f`?
In that case, the future method `b2.f` will inherit the state from the setup that was updated most recently.

Below is an illustration of how operations propagate from parent setups to child setups:

![](/assets/images/api/setup_propagation.svg){: loading=lazy style="width:800px;" }

The diagram above represents the following setup hierarchy:

```pycon
>>> # Populate setups at init
>>> vbt.settings.caching.reset()
>>> vbt.settings.caching['register_lazily'] = False

>>> class A(vbt.Cacheable):
...     @vbt.cached_property
...     def f1(self): pass

>>> class B(A):
...     def f2(self): pass

>>> class C(A):
...     @vbt.cached_method
...     def f2(self): pass

>>> b1 = B()
>>> c1 = C()
>>> c2 = C()

>>> print(vbt.prettify(A.get_ca_setup().get_setup_hierarchy()))
[
    {
        "parent": "<class __main__.B>",
        "children": [
            {
                "parent": "<instance of __main__.B>",
                "children": [
                    "<instance property __main__.B.f1>"
                ]
            }
        ]
    },
    {
        "parent": "<class __main__.C>",
        "children": [
            {
                "parent": "<instance of __main__.C>",
                "children": [
                    "<instance method __main__.C.f2>",
                    "<instance property __main__.C.f1>"
                ]
            },
            {
                "parent": "<instance of __main__.C>",
                "children": [
                    "<instance method __main__.C.f2>",
                    "<instance property __main__.C.f1>"
                ]
            }
        ]
    }
]

>>> print(vbt.prettify(A.f1.get_ca_setup().get_setup_hierarchy()))
[
    "<instance property __main__.C.f1>",
    "<instance property __main__.C.f1>",
    "<instance property __main__.B.f1>"
]

>>> print(vbt.prettify(C.f2.get_ca_setup().get_setup_hierarchy()))
[
    "<instance method __main__.C.f2>",
    "<instance method __main__.C.f2>"
]
```

To disable caching for the entire `A` class, use:

```pycon
>>> A.get_ca_setup().disable_caching()
>>> A.get_ca_setup().use_cache
False
>>> B.get_ca_setup().use_cache
False
>>> C.get_ca_setup().use_cache
False
```

This operation disables caching for `A`, its subclasses (`B` and `C`), their instances,
and any instance function, but it does not affect unbound functions such as `C.f1` and `C.f2`:

```pycon
>>> C.f1.get_ca_setup().use_cache
True
>>> C.f2.get_ca_setup().use_cache
True
```

This behavior occurs because unbound functions are not considered children of the classes
they are defined in. Consequently, any future instance method of `C` will remain uncached
if the class setup was updated more recently than the unbound function.

```pycon
>>> c3 = C()
>>> C.f2.get_ca_setup(c3).use_cache
False
```

If you want to disable caching for an entire class while leaving one function unaffected,
perform the following operations in order:

1. Disable caching on the class.
2. Enable caching on the unbound function.

```pycon
>>> A.get_ca_setup().disable_caching()
>>> C.f2.get_ca_setup().enable_caching()

>>> c4 = C()
>>> C.f2.get_ca_setup(c4).use_cache
True
```

## Getting statistics

The central registry of setups provides an easy way to locate any setup registered in any
part of vectorbtpro that meets specific conditions using `CacheableRegistry.match_setups`.

!!! note
    By default, setups are registered lazily — a setup is only registered when executed or explicitly called.
    To change this behavior, set `register_lazily` in the global settings to False.

For example, the following command displays the registered setups:

```pycon
>>> vbt.ca_reg.match_setups(kind=None)
{
    CAClassSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=None, whitelist=None, cls=<class '__main__.B'>),
    CAClassSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=None, whitelist=None, cls=<class '__main__.C'>),
    CAInstanceSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=None, whitelist=None, instance=<weakref at 0x7fe14e9d83b8; to 'B' at 0x7fe14e944978>),
    CAInstanceSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=None, whitelist=None, instance=<weakref at 0x7fe14e9d84f8; to 'C' at 0x7fe14e9448d0>),
    CAInstanceSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=None, whitelist=None, instance=<weakref at 0x7fe14e9d8688; to 'C' at 0x7fe1495111d0>),
    CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<function <lambda> at 0x7fe14e94cae8>, instance=None, max_size=None, ignore_args=None, cache={}),
    CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<function C.f2 at 0x7fe13959ee18>, instance=<weakref at 0x7fe14e9d85e8; to 'C' at 0x7fe14e9448d0>, max_size=None, ignore_args=None, cache={}),
    CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<function C.f2 at 0x7fe13959ee18>, instance=<weakref at 0x7fe14e9d8728; to 'C' at 0x7fe1495111d0>, max_size=None, ignore_args=None, cache={}),
    CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<vectorbtpro.utils.decorators.cached_property object at 0x7fe118045408>, instance=<weakref at 0x7fe14e9d8458; to 'B' at 0x7fe14e944978>, max_size=None, ignore_args=None, cache={}),
    CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<vectorbtpro.utils.decorators.cached_property object at 0x7fe118045408>, instance=<weakref at 0x7fe14e9d8598; to 'C' at 0x7fe14e9448d0>, max_size=None, ignore_args=None, cache={}),
    CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<vectorbtpro.utils.decorators.cached_property object at 0x7fe118045408>, instance=<weakref at 0x7fe14e9d86d8; to 'C' at 0x7fe1495111d0>, max_size=None, ignore_args=None, cache={}),
    CAUnboundSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<function C.f2 at 0x7fe13959ee18>),
    CAUnboundSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<vectorbtpro.utils.decorators.cached_property object at 0x7fe118045408>)
}
```

To obtain the runnable setup of any property or method named `f2`, run:

```pycon
>>> vbt.ca_reg.match_setups('f2', kind='runnable')
{
    CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<function C.f2 at 0x7fe13959ee18>, instance=<weakref at 0x7fe14e9d85e8; to 'C' at 0x7fe14e9448d0>, max_size=None, ignore_args=None, cache={}),
    CARunSetup(registry=<vectorbtpro.registries.ca_registry.CacheableRegistry object at 0x7fe14c27df60>, use_cache=True, whitelist=False, cacheable=<function C.f2 at 0x7fe13959ee18>, instance=<weakref at 0x7fe14e9d8728; to 'C' at 0x7fe1495111d0>, max_size=None, ignore_args=None, cache={})
}
```

Alternatively, retrieve statistics using `CAQueryDelegator.get_stats`,
which returns a DataFrame containing setup statistics as rows:

```pycon
>>> vbt.CAQueryDelegator('f2', kind='runnable').get_stats()
                                               string  use_cache  whitelist  \\
hash
 3506416602224216137  <instance method __main__.C.f2>       True      False
-4747092115268118855  <instance method __main__.C.f2>       True      False
-4748466030718995055  <instance method __main__.C.f2>       True      False

                      caching_enabled  hits  misses total_size total_elapsed  \\
hash
 3506416602224216137             True     0       0    0 Bytes          None
-4747092115268118855             True     0       0    0 Bytes          None
-4748466030718995055             True     0       0    0 Bytes          None

                     total_saved first_run_time last_run_time first_hit_time  \\
hash
 3506416602224216137        None           None          None           None
-4747092115268118855        None           None          None           None
-4748466030718995055        None           None          None           None

                     last_hit_time  creation_time last_update_time
hash
 3506416602224216137          None  9 minutes ago    9 minutes ago
-4747092115268118855          None  9 minutes ago    9 minutes ago
-4748466030718995055          None  9 minutes ago    9 minutes ago
```

## Clearing up

Instance and runnable setups hold weak references to their respective instances. Consequently,
deleting an instance does not persist its setup in memory—the setup is automatically deregistered.

To clear all caches, execute:

```pycon
>>> vbt.CAQueryDelegator().clear_cache()
```

## Resetting

Reset global caching flags with:

```pycon
>>> vbt.settings.caching.reset()
```

To deregister all setups, execute:

```pycon
>>> vbt.CAQueryDelegator(kind=None).deregister()
```

!!!
    For default settings, see `vectorbtpro._settings.caching`.
"""

import inspect
import sys
from collections.abc import ValuesView
from datetime import datetime, timedelta, timezone
from functools import wraps
from weakref import ReferenceType, ref

import attr
import humanize
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.caching import Cacheable
from vectorbtpro.utils.decorators import cacheable_property, cacheableT
from vectorbtpro.utils.formatting import ptable
from vectorbtpro.utils.hashing import UnhashableArgsError, hash_args
from vectorbtpro.utils.parsing import Regex, get_func_arg_names
from vectorbtpro.utils.profiling import Timer
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "CacheableRegistry",
    "ca_reg",
    "CAQuery",
    "CARule",
    "CAQueryDelegator",
    "get_cache_stats",
    "print_cache_stats",
    "clear_cache",
    "collect_garbage",
    "flush",
    "disable_caching",
    "enable_caching",
    "CachingDisabled",
    "with_caching_disabled",
    "CachingEnabled",
    "with_caching_enabled",
]

__pdoc__ = {}


class _GARBAGE:
    """Sentinel class representing a marker for garbage."""


def is_cacheable_function(cacheable: tp.Any) -> bool:
    """Determine whether the provided object is a cacheable function.

    Args:
        cacheable (Any): Object to evaluate for cacheable function behavior.

    Returns:
        bool: True if the object is a cacheable function, False otherwise.
    """
    return (
        callable(cacheable)
        and hasattr(cacheable, "is_method")
        and not cacheable.is_method
        and hasattr(cacheable, "is_cacheable")
        and cacheable.is_cacheable
    )


def is_cacheable_property(cacheable: tp.Any) -> bool:
    """Determine whether the provided object is a cacheable property.

    Args:
        cacheable (Any): Object to evaluate for cacheable property behavior.

    Returns:
        bool: True if the object is a cacheable property, False otherwise.
    """
    return isinstance(cacheable, cacheable_property)


def is_cacheable_method(cacheable: tp.Any) -> bool:
    """Determine whether the provided object is a cacheable method.

    Args:
        cacheable (Any): Object to evaluate for cacheable method behavior.

    Returns:
        bool: True if the object is a cacheable method, False otherwise.
    """
    return (
        callable(cacheable)
        and hasattr(cacheable, "is_method")
        and cacheable.is_method
        and hasattr(cacheable, "is_cacheable")
        and cacheable.is_cacheable
    )


def is_bindable_cacheable(cacheable: tp.Any) -> bool:
    """Determine whether the provided object is a bindable cacheable suitable for instance binding.

    Args:
        cacheable (Any): Object to evaluate for bindable cacheable behavior.

    Returns:
        bool: True if the object is bindable, False otherwise.
    """
    return is_cacheable_property(cacheable) or is_cacheable_method(cacheable)


def is_cacheable(cacheable: tp.Any) -> bool:
    """Determine whether the provided object is cacheable.

    Args:
        cacheable (Any): Object to evaluate for cacheable behavior.

    Returns:
        bool: True if the object is cacheable, False otherwise.
    """
    return is_cacheable_function(cacheable) or is_bindable_cacheable(cacheable)


def get_obj_id(instance: object) -> tp.Tuple[type, int]:
    """Retrieve the type and unique identifier of the given instance.

    Args:
        instance (object): Instance whose identity is being retrieved.

    Returns:
        Tuple[type, int]: Tuple containing the instance's type and its id.
    """
    return type(instance), id(instance)


def instance_converter(
    instance: tp.Optional[tp.Union[Cacheable, ReferenceType]],
) -> tp.Optional[tp.Union[Cacheable, ReferenceType]]:
    """Convert the provided instance to a weak reference if applicable.

    Args:
        instance (Optional[Union[Cacheable, ReferenceType]]): Instance to convert.

    Returns:
        Optional[Union[Cacheable, ReferenceType]]: Weak reference to the instance
            if it is not already a weak reference; otherwise, the unchanged instance.
    """
    if (
        instance is not None
        and instance is not _GARBAGE
        and not isinstance(instance, ReferenceType)
    ):
        return ref(instance)
    return instance


CAQueryT = tp.TypeVar("CAQueryT", bound="CAQuery")


@define
class CAQuery(DefineMixin):
    """Class that represents a query for matching and ranking setups."""

    cacheable: tp.Optional[tp.Union[tp.Callable, cacheableT, str, Regex]] = define.field(
        default=None
    )
    """Cacheable object or its identifier name (case-sensitive)."""

    instance: tp.Optional[tp.Union[Cacheable, ReferenceType]] = define.field(
        default=None, converter=instance_converter
    )
    """Weak reference to the instance to which `CAQuery.cacheable` is bound."""

    cls: tp.Optional[tp.TypeLike] = define.field(default=None)
    """Class of the instance to which `CAQuery.cacheable` is bound, or its name (case-sensitive)."""

    base_cls: tp.Optional[tp.TypeLike] = define.field(default=None)
    """Base class of the instance to which `CAQuery.cacheable` is bound, or its name (case-sensitive)."""

    options: tp.Optional[dict] = define.field(default=None)
    """Dictionary of options used for matching."""

    @classmethod
    def parse(cls: tp.Type[CAQueryT], query_like: tp.Any, use_base_cls: bool = True) -> CAQueryT:
        """Parse a query-like object.

        Args:
            query_like (Any): Object representing the query.

                Can be a callable, string, dictionary, type, or other object.
            use_base_cls (bool): Flag indicating whether the base class is used in query parsing.

        Returns:
            CAQuery: New `CAQuery` instance parsed from the input.

        !!! note
            Not all attribute combinations can be safely parsed by this function.
            For example, you cannot combine cacheable together with options.

        Examples:
            ```pycon
            >>> vbt.CAQuery.parse(lambda x: x)
            CAQuery(cacheable=<function <lambda> at 0x7fd4766c7730>, instance=None, cls=None, base_cls=None, options=None)

            >>> vbt.CAQuery.parse("a")
            CAQuery(cacheable='a', instance=None, cls=None, base_cls=None, options=None)

            >>> vbt.CAQuery.parse("A.a")
            CAQuery(cacheable='a', instance=None, cls=None, base_cls='A', options=None)

            >>> vbt.CAQuery.parse("A")
            CAQuery(cacheable=None, instance=None, cls=None, base_cls='A', options=None)

            >>> vbt.CAQuery.parse("A", use_base_cls=False)
            CAQuery(cacheable=None, instance=None, cls='A', base_cls=None, options=None)

            >>> vbt.CAQuery.parse(vbt.Regex("[A-B]"))
            CAQuery(cacheable=None, instance=None, cls=None, base_cls=Regex(pattern='[A-B]', flags=0), options=None)

            >>> vbt.CAQuery.parse(dict(my_option=100))
            CAQuery(cacheable=None, instance=None, cls=None, base_cls=None, options={'my_option': 100})
            ```
        """
        if query_like is None:
            return CAQuery()
        if isinstance(query_like, CAQuery):
            return query_like
        if isinstance(query_like, CABaseSetup):
            return query_like.query
        if isinstance(query_like, cacheable_property):
            return cls(cacheable=query_like)
        if isinstance(query_like, str) and query_like[0].islower():
            return cls(cacheable=query_like)
        if isinstance(query_like, str) and query_like[0].isupper() and "." in query_like:
            if use_base_cls:
                return cls(cacheable=query_like.split(".")[1], base_cls=query_like.split(".")[0])
            return cls(cacheable=query_like.split(".")[1], cls=query_like.split(".")[0])
        if isinstance(query_like, str) and query_like[0].isupper():
            if use_base_cls:
                return cls(base_cls=query_like)
            return cls(cls=query_like)
        if isinstance(query_like, Regex):
            if use_base_cls:
                return cls(base_cls=query_like)
            return cls(cls=query_like)
        if isinstance(query_like, type):
            if use_base_cls:
                return cls(base_cls=query_like)
            return cls(cls=query_like)
        if isinstance(query_like, tuple):
            if use_base_cls:
                return cls(base_cls=query_like)
            return cls(cls=query_like)
        if isinstance(query_like, dict):
            return cls(options=query_like)
        if callable(query_like):
            return cls(cacheable=query_like)
        return cls(instance=query_like)

    @property
    def instance_obj(self) -> tp.Optional[tp.Union[Cacheable, object]]:
        """Instance if available; if the weak reference is dead, returns a garbage sentinel.

        Returns:
            Optional[Union[Cacheable, object]]: Instance or a garbage sentinel.
        """
        if self.instance is _GARBAGE:
            return _GARBAGE
        if self.instance is not None and self.instance() is None:
            return _GARBAGE
        return self.instance() if self.instance is not None else None

    def matches_setup(self, setup: "CABaseSetup") -> bool:
        """Determine if the given setup matches the query criteria.

        Args:
            setup (CABaseSetup): Setup instance to validate against the query.

        Returns:
            bool: True if the setup meets the query criteria, otherwise False.

        Examples:
            Let's evaluate various queries:

            ```pycon
            >>> class A(vbt.Cacheable):
            ...     @vbt.cached_method(my_option=True)
            ...     def f(self):
            ...         return None

            >>> class B(A):
            ...     pass

            >>> @vbt.cached(my_option=False)
            ... def f():
            ...     return None

            >>> a = A()
            >>> b = B()

            >>> def match_query(query):
            ...     matched = []
            ...     if query.matches_setup(A.f.get_ca_setup()):  # unbound method
            ...         matched.append('A.f')
            ...     if query.matches_setup(A.get_ca_setup()):  # class
            ...         matched.append('A')
            ...     if query.matches_setup(a.get_ca_setup()):  # instance
            ...         matched.append('a')
            ...     if query.matches_setup(A.f.get_ca_setup(a)):  # instance method
            ...         matched.append('a.f')
            ...     if query.matches_setup(B.f.get_ca_setup()):  # unbound method
            ...         matched.append('B.f')
            ...     if query.matches_setup(B.get_ca_setup()):  # class
            ...         matched.append('B')
            ...     if query.matches_setup(b.get_ca_setup()):  # instance
            ...         matched.append('b')
            ...     if query.matches_setup(B.f.get_ca_setup(b)):  # instance method
            ...         matched.append('b.f')
            ...     if query.matches_setup(f.get_ca_setup()):  # function
            ...         matched.append('f')
            ...     return matched

            >>> match_query(vbt.CAQuery())
            ['A.f', 'A', 'a', 'a.f', 'B.f', 'B', 'b', 'b.f', 'f']
            >>> match_query(vbt.CAQuery(cacheable=A.f))
            ['A.f', 'a.f', 'B.f', 'b.f']
            >>> match_query(vbt.CAQuery(cacheable=B.f))
            ['A.f', 'a.f', 'B.f', 'b.f']
            >>> match_query(vbt.CAQuery(cls=A))
            ['A', 'a', 'a.f']
            >>> match_query(vbt.CAQuery(cls=B))
            ['B', 'b', 'b.f']
            >>> match_query(vbt.CAQuery(cls=vbt.Regex('[A-B]')))
            ['A', 'a', 'a.f', 'B', 'b', 'b.f']
            >>> match_query(vbt.CAQuery(base_cls=A))
            ['A', 'a', 'a.f', 'B', 'b', 'b.f']
            >>> match_query(vbt.CAQuery(base_cls=B))
            ['B', 'b', 'b.f']
            >>> match_query(vbt.CAQuery(instance=a))
            ['a', 'a.f']
            >>> match_query(vbt.CAQuery(instance=b))
            ['b', 'b.f']
            >>> match_query(vbt.CAQuery(instance=a, cacheable='f'))
            ['a.f']
            >>> match_query(vbt.CAQuery(instance=b, cacheable='f'))
            ['b.f']
            >>> match_query(vbt.CAQuery(options=dict(my_option=True)))
            ['A.f', 'a.f', 'B.f', 'b.f']
            >>> match_query(vbt.CAQuery(options=dict(my_option=False)))
            ['f']
            ```
        """
        if self.cacheable is not None:
            if not isinstance(setup, (CARunSetup, CAUnboundSetup)):
                return False
            if is_cacheable(self.cacheable):
                if (
                    setup.cacheable is not self.cacheable
                    and setup.cacheable.func is not self.cacheable.func
                ):
                    return False
            elif callable(self.cacheable):
                if setup.cacheable.func is not self.cacheable:
                    return False
            elif isinstance(self.cacheable, str):
                if setup.cacheable.name != self.cacheable:
                    return False
            elif isinstance(self.cacheable, Regex):
                if not self.cacheable.matches(setup.cacheable.name):
                    return False
            else:
                return False

        if self.instance_obj is not None:
            if not isinstance(setup, (CARunSetup, CAInstanceSetup)):
                return False
            if setup.instance_obj is not self.instance_obj:
                return False

        if self.cls is not None:
            if not isinstance(setup, (CARunSetup, CAInstanceSetup, CAClassSetup)):
                return False
            if isinstance(setup, (CARunSetup, CAInstanceSetup)) and setup.instance_obj is _GARBAGE:
                return False
            if isinstance(setup, (CARunSetup, CAInstanceSetup)) and not checks.is_class(
                type(setup.instance_obj),
                self.cls,
            ):
                return False
            if isinstance(setup, CAClassSetup) and not checks.is_class(setup.cls, self.cls):
                return False

        if self.base_cls is not None:
            if not isinstance(setup, (CARunSetup, CAInstanceSetup, CAClassSetup)):
                return False
            if isinstance(setup, (CARunSetup, CAInstanceSetup)) and setup.instance_obj is _GARBAGE:
                return False
            if isinstance(setup, (CARunSetup, CAInstanceSetup)) and not checks.is_subclass_of(
                type(setup.instance_obj),
                self.base_cls,
            ):
                return False
            if isinstance(setup, CAClassSetup) and not checks.is_subclass_of(
                setup.cls, self.base_cls
            ):
                return False

        if self.options is not None and len(self.options) > 0:
            if not isinstance(setup, (CARunSetup, CAUnboundSetup)):
                return False
            for k, v in self.options.items():
                if k not in setup.cacheable.options or setup.cacheable.options[k] != v:
                    return False

        return True

    @property
    def hash_key(self) -> tuple:
        return (
            self.cacheable,
            get_obj_id(self.instance_obj) if self.instance_obj is not None else None,
            self.cls,
            self.base_cls,
            tuple(self.options.items()) if self.options is not None else None,
        )


@define
class CARule(DefineMixin):
    """Class representing a rule enforced on setups that match a specified query."""

    query: CAQuery = define.field()
    """`CAQuery` used for matching."""

    enforce_func: tp.Optional[tp.Callable] = define.field()
    """Callable function to execute on the setup when it matches the rule."""

    kind: tp.Optional[tp.MaybeIterable[str]] = define.field(default=None)
    """Specifies the expected kind(s) of setup. For example, 'class', 'instance', 'unbound', or 'runnable'."""

    exclude: tp.Optional[tp.MaybeIterable["CABaseSetup"]] = define.field(default=None)
    """Setup or an iterable of setups to be excluded from matching."""

    filter_func: tp.Optional[tp.Callable] = define.field(default=None)
    """Callable function used to apply additional filtering on a setup."""

    def matches_setup(self, setup: "CABaseSetup") -> bool:
        """Return whether the specified setup satisfies the rule criteria.

        Args:
            setup (CABaseSetup): Setup instance to evaluate.

        Returns:
            bool: True if the setup meets the rule conditions, otherwise False.
        """
        if not self.query.matches_setup(setup):
            return False
        if self.kind is not None:
            kind = self.kind
            if isinstance(kind, str):
                kind = {kind}
            else:
                kind = set(kind)
            if isinstance(setup, CAClassSetup):
                setup_kind = "class"
            elif isinstance(setup, CAInstanceSetup):
                setup_kind = "instance"
            elif isinstance(setup, CAUnboundSetup):
                setup_kind = "unbound"
            else:
                setup_kind = "runnable"
            if setup_kind not in kind:
                return False
        if self.exclude is not None:
            exclude = self.exclude
            if exclude is None:
                exclude = set()
            if isinstance(exclude, CABaseSetup):
                exclude = {exclude}
            else:
                exclude = set(exclude)
            if setup in exclude:
                return False
        if self.filter_func is not None:
            if not self.filter_func(setup):
                return False
        return True

    def enforce(self, setup: "CABaseSetup") -> None:
        """Execute the enforce function on the setup if it satisfies the rule criteria.

        Args:
            setup (CABaseSetup): Setup instance to process.

        Returns:
            None
        """
        if self.matches_setup(setup):
            self.enforce_func(setup)

    @property
    def hash_key(self) -> tuple:
        return (
            self.query,
            self.enforce_func,
            self.kind,
            self.exclude if isinstance(self.exclude, CABaseSetup) else tuple(self.exclude),
            self.filter_func,
        )


class CacheableRegistry(Base):
    """Class that manages registration of cacheable setups.

    This class stores and retrieves setups for cacheable objects including class setups, instance setups,
    unbound setups, and run setups, along with registered rules.
    """

    def __init__(self) -> None:
        self._class_setups = dict()
        self._instance_setups = dict()
        self._unbound_setups = dict()
        self._run_setups = dict()
        self._rules = []

    @property
    def class_setups(self) -> tp.Dict[int, "CAClassSetup"]:
        """Dictionary of registered `CAClassSetup` instances keyed by hash.

        Returns:
            Dict[int, CAClassSetup]: Dictionary of class setups.
        """
        return self._class_setups

    @property
    def instance_setups(self) -> tp.Dict[int, "CAInstanceSetup"]:
        """Dictionary of registered `CAInstanceSetup` instances keyed by hash.

        Returns:
            Dict[int, CAInstanceSetup]: Dictionary of instance setups.
        """
        return self._instance_setups

    @property
    def unbound_setups(self) -> tp.Dict[int, "CAUnboundSetup"]:
        """Dictionary of registered `CAUnboundSetup` instances keyed by hash.

        Returns:
            Dict[int, CAUnboundSetup]: Dictionary of unbound setups.
        """
        return self._unbound_setups

    @property
    def run_setups(self) -> tp.Dict[int, "CARunSetup"]:
        """Dictionary of registered `CARunSetup` instances keyed by hash.

        Returns:
            Dict[int, CARunSetup]: Dictionary of run setups.
        """
        return self._run_setups

    @property
    def setups(self) -> tp.Dict[int, "CABaseSetup"]:
        """Dictionary with all registered setups combined from class, instance, unbound, and run setups.

        Returns:
            Dict[int, CABaseSetup]: Dictionary of all setups.
        """
        return {
            **self.class_setups,
            **self.instance_setups,
            **self.unbound_setups,
            **self.run_setups,
        }

    @property
    def rules(self) -> tp.List[CARule]:
        """List of registered `CARule` instances.

        Returns:
            List[CARule]: List of caching rules.
        """
        return self._rules

    def get_setup_by_hash(self, hash_: int) -> tp.Optional["CABaseSetup"]:
        """Return the setup corresponding to the specified hash.

        Args:
            hash_ (int): Hash of the setup.

        Returns:
            Optional[CABaseSetup]: Setup instance if found; otherwise, None.
        """
        if hash_ in self.class_setups:
            return self.class_setups[hash_]
        if hash_ in self.instance_setups:
            return self.instance_setups[hash_]
        if hash_ in self.unbound_setups:
            return self.unbound_setups[hash_]
        if hash_ in self.run_setups:
            return self.run_setups[hash_]
        return None

    def setup_registered(self, setup: "CABaseSetup") -> bool:
        """Return whether the specified setup is registered.

        Args:
            setup (CABaseSetup): Setup instance to check.

        Returns:
            bool: True if the setup is registered; otherwise, False.
        """
        return self.get_setup_by_hash(hash(setup)) is not None

    def register_setup(self, setup: "CABaseSetup") -> None:
        """Register a new setup.

        Args:
            setup (CABaseSetup): Setup instance to register.

        Returns:
            None
        """
        if isinstance(setup, CARunSetup):
            setups = self.run_setups
        elif isinstance(setup, CAUnboundSetup):
            setups = self.unbound_setups
        elif isinstance(setup, CAInstanceSetup):
            setups = self.instance_setups
        elif isinstance(setup, CAClassSetup):
            setups = self.class_setups
        else:
            raise TypeError(str(type(setup)))
        setups[hash(setup)] = setup

    def deregister_setup(self, setup: "CABaseSetup") -> None:
        """Deregister the specified setup.

        Removes the setup from its corresponding collection.
        To also deregister its children, call `CASetupDelegatorMixin.deregister`.

        Args:
            setup (CABaseSetup): Setup instance to deregister.

        Returns:
            None
        """
        if isinstance(setup, CARunSetup):
            setups = self.run_setups
        elif isinstance(setup, CAUnboundSetup):
            setups = self.unbound_setups
        elif isinstance(setup, CAInstanceSetup):
            setups = self.instance_setups
        elif isinstance(setup, CAClassSetup):
            setups = self.class_setups
        else:
            raise TypeError(str(type(setup)))
        if hash(setup) in setups:
            del setups[hash(setup)]

    def register_rule(self, rule: CARule) -> None:
        """Register a new caching rule.

        Args:
            rule (CARule): Rule instance to register.

        Returns:
            None
        """
        self.rules.append(rule)

    def deregister_rule(self, rule: CARule) -> None:
        """Deregister the specified caching rule.

        Args:
            rule (CARule): Rule instance to deregister.

        Returns:
            None
        """
        self.rules.remove(rule)

    def get_run_setup(
        self,
        cacheable: cacheableT,
        instance: tp.Optional[Cacheable] = None,
    ) -> tp.Optional["CARunSetup"]:
        """Return the run setup for the specified cacheable and instance.

        Args:
            cacheable (cacheable): Cacheable object.
            instance (Optional[Cacheable]): Instance associated with the cacheable.

        Returns:
            Optional[CARunSetup]: Run setup if available and valid; otherwise, None.
        """
        run_setup = self.run_setups.get(CARunSetup.get_hash(cacheable, instance=instance), None)
        if run_setup is not None and run_setup.instance_obj is _GARBAGE:
            self.deregister_setup(run_setup)
            return None
        return run_setup

    def get_unbound_setup(self, cacheable: cacheableT) -> tp.Optional["CAUnboundSetup"]:
        """Return the unbound setup for the specified cacheable.

        Args:
            cacheable (cacheable): Cacheable object.

        Returns:
            Optional[CAUnboundSetup]: Unbound setup if available; otherwise, None.
        """
        return self.unbound_setups.get(CAUnboundSetup.get_hash(cacheable), None)

    def get_instance_setup(self, instance: Cacheable) -> tp.Optional["CAInstanceSetup"]:
        """Return the instance setup for the specified cacheable instance.

        Args:
            instance (Cacheable): Cacheable instance.

        Returns:
            Optional[CAInstanceSetup]: Instance setup if available and valid; otherwise, None.
        """
        instance_setup = self.instance_setups.get(CAInstanceSetup.get_hash(instance), None)
        if instance_setup is not None and instance_setup.instance_obj is _GARBAGE:
            self.deregister_setup(instance_setup)
            return None
        return instance_setup

    def get_class_setup(self, cls: tp.Type[Cacheable]) -> tp.Optional["CAClassSetup"]:
        """Return the class setup for the specified cacheable class.

        Args:
            cls (Type[Cacheable]): Cacheable class.

        Returns:
            Optional[CAClassSetup]: Class setup if available; otherwise, None.
        """
        return self.class_setups.get(CAClassSetup.get_hash(cls), None)

    def match_setups(
        self,
        query_like: tp.MaybeIterable[tp.Any] = None,
        collapse: bool = False,
        kind: tp.Optional[tp.MaybeIterable[str]] = None,
        exclude: tp.Optional[tp.MaybeIterable["CABaseSetup"]] = None,
        exclude_children: bool = True,
        filter_func: tp.Optional[tp.Callable] = None,
    ) -> tp.Set["CABaseSetup"]:
        """Match all setups in the registry that satisfy the provided queries.

        This function parses one or more query-like objects using `CAQuery.parse` and returns
        a set of setups that match any of the queries. If `collapse` is True, child setups
        of a matched parent setup are excluded from the final results.

        Args:
            query_like (MaybeIterable[Any]): One or multiple query-like objects to match setups.

                They are parsed with `CAQuery.parse`.
            collapse (bool): If True, remove child setups belonging to any matched parent setup.
            kind (Optional[MaybeIterable[str]]): Specifies the expected kind(s) of setup to match.

                Supported values:

                * "class": Matches class setups (instances of `CAClassSetup`).
                * "instance": Matches instance setups (instances of `CAInstanceSetup`).
                * "unbound": Matches unbound setups (instances of `CAUnboundSetup`).
                * "runnable": Matches runnable setups (instances of `CARunSetup`).
            exclude (Optional[MaybeIterable[CABaseSetup]]): Setup or setups to exclude from matching.

                If you wish to retain child setups of an excluded setup, set `exclude_children` to False.
            exclude_children (bool): When True, excludes child setups of an excluded setup.

                This is applied only when `collapse` is True.
            filter_func (Optional[Callable]): Function that takes a setup and returns a boolean
                indicating whether the setup should be included.

        Returns:
            Set[CABaseSetup]: Set of setups matching the given criteria.

        !!! note
            `exclude_children` is applied only when `collapse` is True.
        """
        if not checks.is_iterable(query_like) or isinstance(query_like, (str, tuple)):
            query_like = [query_like]
        query_like = list(map(CAQuery.parse, query_like))
        if kind is None:
            kind = {"class", "instance", "unbound", "runnable"}
        if exclude is None:
            exclude = set()
        if isinstance(exclude, CABaseSetup):
            exclude = {exclude}
        else:
            exclude = set(exclude)

        matches = set()
        if not collapse:
            if isinstance(kind, str):
                if kind.lower() == "class":
                    setups = set(self.class_setups.values())
                elif kind.lower() == "instance":
                    setups = set(self.instance_setups.values())
                elif kind.lower() == "unbound":
                    setups = set(self.unbound_setups.values())
                elif kind.lower() == "runnable":
                    setups = set(self.run_setups.values())
                else:
                    raise ValueError(f"kind '{kind}' is not supported")
                for setup in setups:
                    if setup not in exclude:
                        for q in query_like:
                            if q.matches_setup(setup):
                                if filter_func is None or filter_func(setup):
                                    matches.add(setup)
                                break
            elif checks.is_iterable(kind):
                matches = set.union(
                    *[
                        self.match_setups(
                            query_like,
                            kind=k,
                            collapse=collapse,
                            exclude=exclude,
                            exclude_children=exclude_children,
                            filter_func=filter_func,
                        )
                        for k in kind
                    ]
                )
            else:
                raise TypeError(
                    f"kind must be either a string or a sequence of strings, not {type(kind)}"
                )
        else:
            if isinstance(kind, str):
                kind = {kind}
            else:
                kind = set(kind)
            collapse_setups = set()
            if "class" in kind:
                class_matches = set()
                for class_setup in self.class_setups.values():
                    for q in query_like:
                        if q.matches_setup(class_setup):
                            if filter_func is None or filter_func(class_setup):
                                if class_setup not in exclude:
                                    class_matches.add(class_setup)
                                if class_setup not in exclude or exclude_children:
                                    collapse_setups |= class_setup.child_setups
                            break
                for class_setup in class_matches:
                    if class_setup not in collapse_setups:
                        matches.add(class_setup)
            if "instance" in kind:
                for instance_setup in self.instance_setups.values():
                    if instance_setup not in collapse_setups:
                        for q in query_like:
                            if q.matches_setup(instance_setup):
                                if filter_func is None or filter_func(instance_setup):
                                    if instance_setup not in exclude:
                                        matches.add(instance_setup)
                                    if instance_setup not in exclude or exclude_children:
                                        collapse_setups |= instance_setup.child_setups
                                break
            if "unbound" in kind:
                for unbound_setup in self.unbound_setups.values():
                    if unbound_setup not in collapse_setups:
                        for q in query_like:
                            if q.matches_setup(unbound_setup):
                                if filter_func is None or filter_func(unbound_setup):
                                    if unbound_setup not in exclude:
                                        matches.add(unbound_setup)
                                    if unbound_setup not in exclude or exclude_children:
                                        collapse_setups |= unbound_setup.child_setups
                                break
            if "runnable" in kind:
                for run_setup in self.run_setups.values():
                    if run_setup not in collapse_setups:
                        for q in query_like:
                            if q.matches_setup(run_setup):
                                if filter_func is None or filter_func(run_setup):
                                    if run_setup not in exclude:
                                        matches.add(run_setup)
                                break
        return matches


ca_reg = CacheableRegistry()
"""Default cacheable registry instance of type `CacheableRegistry`."""


class CAMetrics(Base):
    """Abstract base class that provides properties for accessing caching-related metrics,
    including hit and miss counts, total cached size, elapsed time, time saved, and execution timestamps."""

    @property
    def hits(self) -> int:
        """Number of cache hits.

        A cache hit occurs when a requested object is found in the cache.

        Returns:
            int: Total count of times a cached object was successfully retrieved.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def misses(self) -> int:
        """Number of cache misses.

        A cache miss occurs when a requested object is not found in the cache.

        Returns:
            int: Total count of times an object retrieval failed due to absence in the cache.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def total_size(self) -> int:
        """Total size of all cached objects.

        Returns:
            int: Aggregate memory size in bytes of all objects stored in the cache.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def total_elapsed(self) -> tp.Optional[timedelta]:
        """Cumulative elapsed time during function execution.

        Returns:
            Optional[timedelta]: Sum of execution durations for all cached function runs,
                or None if the metric is not available.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def total_saved(self) -> tp.Optional[timedelta]:
        """Cumulative time saved by caching.

        Returns:
            Optional[timedelta]: Total time saved by fetching results from the cache
                instead of executing the function, or None if not determined.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def first_run_time(self) -> tp.Optional[datetime]:
        """Timestamp of the first function execution.

        Returns:
            Optional[datetime]: Datetime when the function was first executed,
                or None if it has never been run.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def last_run_time(self) -> tp.Optional[datetime]:
        """Timestamp of the most recent function execution.

        Returns:
            Optional[datetime]: Datetime of the most recent function run,
                or None if the function has not been executed.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def first_hit_time(self) -> tp.Optional[datetime]:
        """Timestamp of the first cache hit.

        Returns:
            Optional[datetime]: Datetime when the cache was hit for the first time,
                or None if there have been no cache hits.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def last_hit_time(self) -> tp.Optional[datetime]:
        """Timestamp of the most recent cache hit.

        Returns:
            Optional[datetime]: Datetime when the cache was hit most recently,
                or None if no cache hit has occurred.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def metrics(self) -> dict:
        """Dictionary containing all caching metrics.

        Returns:
            dict: Dictionary with keys:

                * `hits`
                * `misses`
                * `total_size`
                * `total_elapsed`
                * `total_saved`
                * `first_run_time`
                * `last_run_time`
                * `first_hit_time`
                * `last_hit_time`
        """
        return dict(
            hits=self.hits,
            misses=self.misses,
            total_size=self.total_size,
            total_elapsed=self.total_elapsed,
            total_saved=self.total_saved,
            first_run_time=self.first_run_time,
            last_run_time=self.last_run_time,
            first_hit_time=self.first_hit_time,
            last_hit_time=self.last_hit_time,
        )


@define
class CABaseSetup(CAMetrics, DefineMixin):
    """Base class that provides properties and methods for cache management."""

    registry: CacheableRegistry = define.field(default=ca_reg)
    """Cache registry of type `CacheableRegistry`."""

    use_cache: tp.Optional[bool] = define.field(default=None)
    """Indicates whether caching is enabled."""

    whitelist: tp.Optional[bool] = define.field(default=None)
    """Indicates if caching is maintained even when it is disabled globally."""

    active: bool = define.field(default=True)
    """Indicates whether the setup is active and can be registered or returned."""

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "_creation_time", datetime.now(timezone.utc))
        object.__setattr__(self, "_use_cache_lut", None)
        object.__setattr__(self, "_whitelist_lut", None)

    @property
    def query(self) -> CAQuery:
        """Query used to match this setup.

        Returns:
            CAQuery: Query instance that identifies or filters setups.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def caching_enabled(self) -> tp.Optional[bool]:
        """Whether caching is enabled for this setup.

        Caching is considered disabled under any of the following conditions:

        * Setup's `use_cache` flag is False.
        * Global caching is disabled (via settings["caching"]["disable"]) and the setup is not whitelisted.
        * Both global caching and whitelisting are disabled.

        Returns:
            Optional[bool]: Indicates the caching status:

                * True if caching is enabled for this setup.
                * False if caching is disabled.
                * None if the caching flag is undefined.

        !!! info
            For default settings, see `vectorbtpro._settings.caching`.
        """
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if self.use_cache is None:
            return None
        if self.use_cache:
            if not caching_cfg["disable"]:
                return True
            if not caching_cfg["disable_whitelist"]:
                if self.whitelist is None:
                    return None
                if self.whitelist:
                    return True
        return False

    def register(self) -> None:
        """Register the setup using `CacheableRegistry.register_setup`.

        Returns:
            None
        """
        self.registry.register_setup(self)

    def deregister(self) -> None:
        """Deregister the setup using `CacheableRegistry.deregister_setup`.

        Returns:
            None
        """
        self.registry.deregister_setup(self)

    @property
    def registered(self) -> bool:
        """Whether the setup is registered.

        Returns:
            bool: True if the setup is registered; otherwise, False.
        """
        return self.registry.setup_registered(self)

    def enforce_rules(self) -> None:
        """Enforce all registry rules on this setup.

        Returns:
            None
        """
        for rule in self.registry.rules:
            rule.enforce(self)

    def activate(self) -> None:
        """Activate the setup.

        Returns:
            None
        """
        object.__setattr__(self, "active", True)

    def deactivate(self) -> None:
        """Deactivate the setup.

        Returns:
            None
        """
        object.__setattr__(self, "active", False)

    def enable_whitelist(self) -> None:
        """Enable whitelisting for this setup.

        Returns:
            None
        """
        object.__setattr__(self, "whitelist", True)
        object.__setattr__(self, "_whitelist_lut", datetime.now(timezone.utc))

    def disable_whitelist(self) -> None:
        """Disable whitelisting for this setup.

        Returns:
            None
        """
        object.__setattr__(self, "whitelist", False)
        object.__setattr__(self, "_whitelist_lut", datetime.now(timezone.utc))

    def enable_caching(
        self, force: bool = False, silence_warnings: tp.Optional[bool] = None
    ) -> None:
        """Enable caching for this setup.

        Args:
            force (bool): Enable whitelisting when set to True.
            silence_warnings (bool): Flag to suppress warning messages.

        Returns:
            None

        !!! info
            For default settings, see `vectorbtpro._settings.caching`.
        """
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if silence_warnings is None:
            silence_warnings = caching_cfg["silence_warnings"]

        object.__setattr__(self, "use_cache", True)
        if force:
            object.__setattr__(self, "whitelist", True)
        else:
            if (
                caching_cfg["disable"]
                and not caching_cfg["disable_whitelist"]
                and not silence_warnings
            ):
                warn(
                    "This operation has no effect: caching is disabled globally and this setup is not whitelisted"
                )
        if caching_cfg["disable"] and caching_cfg["disable_whitelist"] and not silence_warnings:
            warn("This operation has no effect: caching and whitelisting are disabled globally")
        object.__setattr__(self, "_use_cache_lut", datetime.now(timezone.utc))

    def disable_caching(self, clear_cache: bool = True) -> None:
        """Disable caching for this setup.

        Args:
            clear_cache (bool): If True, also clear the cache.

        Returns:
            None
        """
        object.__setattr__(self, "use_cache", False)
        if clear_cache:
            self.clear_cache()
        object.__setattr__(self, "_use_cache_lut", datetime.now(timezone.utc))

    @property
    def creation_time(self) -> tp.datetime:
        """UTC time when this setup was created.

        Returns:
            datetime: UTC datetime representing when the setup was instantiated.
        """
        return object.__getattribute__(self, "_creation_time")

    @property
    def use_cache_lut(self) -> tp.Optional[datetime]:
        """Last update time for the `use_cache` flag.

        Returns:
            Optional[datetime]: Timestamp of the most recent update to the `use_cache` flag,
                or None if not set.
        """
        return object.__getattribute__(self, "_use_cache_lut")

    @property
    def whitelist_lut(self) -> tp.Optional[datetime]:
        """Last update time for the `whitelist` flag.

        Returns:
            Optional[datetime]: Timestamp of the most recent update to the `whitelist` flag,
                or None if not set.
        """
        return object.__getattribute__(self, "_whitelist_lut")

    @property
    def last_update_time(self) -> tp.Optional[datetime]:
        """Most recent update time between `use_cache` and `whitelist` flags.

        Returns:
            Optional[datetime]: Later timestamp between `use_cache_lut` and `whitelist_lut`,
                or None if both are None.
        """
        if self.use_cache_lut is None:
            return self.whitelist_lut
        elif self.whitelist_lut is None:
            return self.use_cache_lut
        elif self.use_cache_lut is None and self.whitelist_lut is None:
            return None
        return max(self.use_cache_lut, self.whitelist_lut)

    def clear_cache(self) -> None:
        """Clear the cache associated with this setup.

        Returns:
            None

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def same_type_setups(self) -> ValuesView:
        """Setups of the same type.

        Returns:
            ValuesView: Collection view of setups that are of the same type as this setup.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def short_str(self) -> str:
        """Concise string representation of the setup.

        Returns:
            str: Brief string that summarizes the setup.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def readable_name(self) -> str:
        """Human-readable name for the object associated with this setup.

        Returns:
            str: User-friendly name representing the setup's associated object.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def position_among_similar(self) -> tp.Optional[int]:
        """Determine the position of this setup among similar setups ordered by creation time.

        Returns:
            Optional[int]: Ordinal position (starting at 0) of this setup among similar ones,
                or None if the position cannot be determined.
        """
        i = 0
        for setup in self.same_type_setups:
            if self is setup:
                return i
            if setup.readable_name == self.readable_name:
                i += 1
        return None

    @property
    def readable_str(self) -> str:
        """Detailed string representation of the setup.

        Returns:
            str: Descriptive string combining the human-readable name and its position among similar setups.
        """
        return f"{self.readable_name}:{self.position_among_similar}"

    def get_stats(self, readable: bool = True, short_str: bool = False) -> dict:
        """Get stats of the setup as a dict with metrics.

        Args:
            readable (bool): Whether to use a human-readable format.
            short_str (bool): When True and `readable` is False, use the short string representation.

        Returns:
            dict: Dictionary containing the following metrics:

                * `hash`: Hash value of the setup instance.
                * `string`: String representation of the setup.
                * `use_cache`: Indicator if caching is enabled.
                * `whitelist`: Whitelist settings.
                * `caching_enabled`: Flag indicating whether caching is active.
                * `hits`: Number of cache hits.
                * `misses`: Number of cache misses.
                * `total_size`: Total size of cached data (humanized if `readable` is True).
                * `total_elapsed`: Total elapsed time for operations (formatted when applicable).
                * `total_saved`: Total time saved by caching (formatted when applicable).
                * `first_run_time`: Timestamp of the first run (humanized if `readable` is True).
                * `last_run_time`: Timestamp of the last run (humanized if `readable` is True).
                * `first_hit_time`: Timestamp of the first cache hit (humanized if `readable` is True).
                * `last_hit_time`: Timestamp of the last cache hit (humanized if `readable` is True).
                * `creation_time`: Timestamp when the setup was created (humanized if `readable` is True).
                * `last_update_time`: Timestamp when the setup was last updated (humanized if `readable` is True).
        """
        if short_str:
            string = self.short_str
        else:
            string = str(self)
        total_size = self.total_size
        total_elapsed = self.total_elapsed
        total_saved = self.total_saved
        first_run_time = self.first_run_time
        last_run_time = self.last_run_time
        first_hit_time = self.first_hit_time
        last_hit_time = self.last_hit_time
        creation_time = self.creation_time
        last_update_time = self.last_update_time

        if readable:
            string = self.readable_str
            total_size = humanize.naturalsize(total_size)
            if total_elapsed is not None:
                minimum_unit = "seconds" if total_elapsed.total_seconds() >= 1 else "milliseconds"
                total_elapsed = humanize.precisedelta(total_elapsed, minimum_unit)
            if total_saved is not None:
                minimum_unit = "seconds" if total_saved.total_seconds() >= 1 else "milliseconds"
                total_saved = humanize.precisedelta(total_saved, minimum_unit)
            if first_run_time is not None:
                first_run_time = humanize.naturaltime(
                    dt.to_naive_datetime(first_run_time),
                    when=dt.to_naive_datetime(datetime.now(timezone.utc)),
                )
            if last_run_time is not None:
                last_run_time = humanize.naturaltime(
                    dt.to_naive_datetime(last_run_time),
                    when=dt.to_naive_datetime(datetime.now(timezone.utc)),
                )
            if first_hit_time is not None:
                first_hit_time = humanize.naturaltime(
                    dt.to_naive_datetime(first_hit_time),
                    when=dt.to_naive_datetime(datetime.now(timezone.utc)),
                )
            if last_hit_time is not None:
                last_hit_time = humanize.naturaltime(
                    dt.to_naive_datetime(last_hit_time),
                    when=dt.to_naive_datetime(datetime.now(timezone.utc)),
                )
            if creation_time is not None:
                creation_time = humanize.naturaltime(
                    dt.to_naive_datetime(creation_time),
                    when=dt.to_naive_datetime(datetime.now(timezone.utc)),
                )
            if last_update_time is not None:
                last_update_time = humanize.naturaltime(
                    dt.to_naive_datetime(last_update_time),
                    when=dt.to_naive_datetime(datetime.now(timezone.utc)),
                )

        return dict(
            hash=hash(self),
            string=string,
            use_cache=self.use_cache,
            whitelist=self.whitelist,
            caching_enabled=self.caching_enabled,
            hits=self.hits,
            misses=self.misses,
            total_size=total_size,
            total_elapsed=total_elapsed,
            total_saved=total_saved,
            first_run_time=first_run_time,
            last_run_time=last_run_time,
            first_hit_time=first_hit_time,
            last_hit_time=last_hit_time,
            creation_time=creation_time,
            last_update_time=last_update_time,
        )


class CASetupDelegatorMixin(CAMetrics):
    """Mixin class that delegates cache management operations to child setup instances.

    This mixin provides methods to propagate cache management actions (e.g. deregistration,
    enabling/disabling caching or whitelist) to all child setups.
    """

    @property
    def child_setups(self) -> tp.Set[CABaseSetup]:
        """Set of child setup instances.

        Returns:
            Set[CABaseSetup]: Set of child setups associated with this setup.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_setup_hierarchy(self, readable: bool = True, short_str: bool = False) -> tp.List[dict]:
        """Return the hierarchical structure of setups by recursively traversing child setups.

        Args:
            readable (bool): Whether to use a human-readable format.
            short_str (bool): When True and `readable` is False, use the short string representation.

        Returns:
            List[dict]: List representing the setup hierarchy.
        """
        results = []
        for setup in self.child_setups:
            if readable:
                setup_obj = setup.readable_str
            elif short_str:
                setup_obj = setup.short_str
            else:
                setup_obj = setup
            if isinstance(setup, CASetupDelegatorMixin):
                results.append(
                    dict(parent=setup_obj, children=setup.get_setup_hierarchy(readable=readable))
                )
            else:
                results.append(setup_obj)
        return results

    def delegate(
        self,
        func: tp.Callable,
        exclude: tp.Optional[tp.MaybeIterable["CABaseSetup"]] = None,
        **kwargs,
    ) -> None:
        """Delegate a function call to each child setup.

        The callable `func` must accept a setup instance as its first parameter and return nothing.
        If a setup is an instance of `CASetupDelegatorMixin`, it should also accept an `exclude` keyword argument.

        Args:
            func (Callable): Function to apply to each child setup.
            exclude (Optional[MaybeIterable[CABaseSetup]]): Iterable of child setups to exclude from delegation.
            **kwargs: Keyword arguments for `func`.

        Returns:
            None
        """
        if exclude is None:
            exclude = set()
        if isinstance(exclude, CABaseSetup):
            exclude = {exclude}
        else:
            exclude = set(exclude)
        for setup in self.child_setups:
            if setup not in exclude:
                if isinstance(setup, CASetupDelegatorMixin):
                    func(setup, exclude=exclude, **kwargs)
                else:
                    func(setup, **kwargs)

    def deregister(self, **kwargs) -> None:
        """Delegate the deregistration action to each child setup.

        Calls the `deregister` method on all child setups.

        Args:
            **kwargs: Keyword arguments for `CABaseSetup.deregister`.

        Returns:
            None
        """
        self.delegate(lambda setup, **_kwargs: setup.deregister(**_kwargs), **kwargs)

    def enable_whitelist(self, **kwargs) -> None:
        """Delegate the enabling of whitelist to each child setup.

        Calls the `enable_whitelist` method on all child setups.

        Args:
            **kwargs: Keyword arguments for `CABaseSetup.enable_whitelist`.

        Returns:
            None
        """
        self.delegate(lambda setup, **_kwargs: setup.enable_whitelist(**_kwargs), **kwargs)

    def disable_whitelist(self, **kwargs) -> None:
        """Delegate the disabling of whitelist to each child setup.

        Calls the `disable_whitelist` method on all child setups.

        Args:
            **kwargs: Keyword arguments for `CABaseSetup.disable_whitelist`.

        Returns:
            None
        """
        self.delegate(lambda setup, **_kwargs: setup.disable_whitelist(**_kwargs), **kwargs)

    def enable_caching(self, **kwargs) -> None:
        """Delegate the enabling of caching to each child setup.

        Calls the `enable_caching` method on all child setups.

        Args:
            **kwargs: Keyword arguments for `CABaseSetup.enable_caching`.

        Returns:
            None
        """
        self.delegate(lambda setup, **_kwargs: setup.enable_caching(**_kwargs), **kwargs)

    def disable_caching(self, **kwargs) -> None:
        """Delegate the disabling of caching to each child setup.

        Calls the `disable_caching` method on all child setups.

        Args:
            **kwargs: Keyword arguments for `CABaseSetup.disable_caching`.

        Returns:
            None
        """
        self.delegate(lambda setup, **_kwargs: setup.disable_caching(**_kwargs), **kwargs)

    def clear_cache(self, **kwargs) -> None:
        """Delegate the clearing of cache to each child setup.

        Calls the `clear_cache` method on all child setups.

        Args:
            **kwargs: Keyword arguments for `CABaseSetup.clear_cache`.

        Returns:
            None
        """
        self.delegate(lambda setup, **_kwargs: setup.clear_cache(**_kwargs), **kwargs)

    @property
    def hits(self) -> int:
        return sum([setup.hits for setup in self.child_setups])

    @property
    def misses(self) -> int:
        return sum([setup.misses for setup in self.child_setups])

    @property
    def total_size(self) -> int:
        return sum([setup.total_size for setup in self.child_setups])

    @property
    def total_elapsed(self) -> tp.Optional[timedelta]:
        total_elapsed = None
        for setup in self.child_setups:
            elapsed = setup.total_elapsed
            if elapsed is not None:
                if total_elapsed is None:
                    total_elapsed = elapsed
                else:
                    total_elapsed += elapsed
        return total_elapsed

    @property
    def total_saved(self) -> tp.Optional[timedelta]:
        total_saved = None
        for setup in self.child_setups:
            saved = setup.total_saved
            if saved is not None:
                if total_saved is None:
                    total_saved = saved
                else:
                    total_saved += saved
        return total_saved

    @property
    def first_run_time(self) -> tp.Optional[datetime]:
        first_run_times = []
        for setup in self.child_setups:
            first_run_time = setup.first_run_time
            if first_run_time is not None:
                first_run_times.append(first_run_time)
        if len(first_run_times) == 0:
            return None
        return sorted(first_run_times)[0]

    @property
    def last_run_time(self) -> tp.Optional[datetime]:
        last_run_times = []
        for setup in self.child_setups:
            last_run_time = setup.last_run_time
            if last_run_time is not None:
                last_run_times.append(last_run_time)
        if len(last_run_times) == 0:
            return None
        return sorted(last_run_times)[-1]

    @property
    def first_hit_time(self) -> tp.Optional[datetime]:
        first_hit_times = []
        for setup in self.child_setups:
            first_hit_time = setup.first_hit_time
            if first_hit_time is not None:
                first_hit_times.append(first_hit_time)
        if len(first_hit_times) == 0:
            return None
        return sorted(first_hit_times)[0]

    @property
    def last_hit_time(self) -> tp.Optional[datetime]:
        last_hit_times = []
        for setup in self.child_setups:
            last_hit_time = setup.last_hit_time
            if last_hit_time is not None:
                last_hit_times.append(last_hit_time)
        if len(last_hit_times) == 0:
            return None
        return sorted(last_hit_times)[-1]

    def get_stats(
        self,
        readable: bool = True,
        short_str: bool = False,
        index_by_hash: bool = False,
        filter_func: tp.Optional[tp.Callable] = None,
        include: tp.Optional[tp.MaybeSequence[str]] = None,
        exclude: tp.Optional[tp.MaybeSequence[str]] = None,
    ) -> tp.Optional[tp.Frame]:
        """Return a DataFrame constructed from the stats dictionaries of child setups.

        Args:
            readable (bool): Whether to use a human-readable format.
            short_str (bool): When True and `readable` is False, use the short string representation.
            index_by_hash (bool): If True, set the DataFrame index to the setup hash.
            filter_func (Callable): Function to filter child setups.
            include (Sequence[str]): List of column names to include in the DataFrame.
            exclude (Sequence[str]): List of column names to exclude from the DataFrame.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the stats of child setups,
                or None if no setups are available.
        """
        if len(self.child_setups) == 0:
            return None
        df = pd.DataFrame(
            [
                setup.get_stats(readable=readable, short_str=short_str)
                for setup in self.child_setups
                if filter_func is None or filter_func(setup)
            ]
        )
        if index_by_hash:
            df.set_index("hash", inplace=True)
            df.index.name = "hash"
        else:
            df.set_index("string", inplace=True)
            df.index.name = "object"
        if include is not None:
            if isinstance(include, str):
                include = [include]
            columns = include
        else:
            columns = df.columns
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            columns = [c for c in columns if c not in exclude]
        if len(columns) == 0:
            return None
        return df[columns].sort_index()


class CABaseDelegatorSetup(CABaseSetup, CASetupDelegatorMixin):
    """Class representing a stateful setup that delegates cache management operations to child setups.

    This class first delegates cache management actions to its child setups and then applies
    the corresponding state changes to itself.
    """

    @property
    def child_setups(self) -> tp.Set[CABaseSetup]:
        return self.registry.match_setups(self.query, collapse=True)

    def deregister(self, **kwargs) -> None:
        CASetupDelegatorMixin.deregister(self, **kwargs)
        CABaseSetup.deregister(self)

    def enable_whitelist(self, **kwargs) -> None:
        CASetupDelegatorMixin.enable_whitelist(self, **kwargs)
        CABaseSetup.enable_whitelist(self)

    def disable_whitelist(self, **kwargs) -> None:
        CASetupDelegatorMixin.disable_whitelist(self, **kwargs)
        CABaseSetup.disable_whitelist(self)

    def enable_caching(
        self, force: bool = False, silence_warnings: tp.Optional[bool] = None, **kwargs
    ) -> None:
        CASetupDelegatorMixin.enable_caching(
            self, force=force, silence_warnings=silence_warnings, **kwargs
        )
        CABaseSetup.enable_caching(self, force=force, silence_warnings=silence_warnings)

    def disable_caching(self, clear_cache: bool = True, **kwargs) -> None:
        CASetupDelegatorMixin.disable_caching(self, clear_cache=clear_cache, **kwargs)
        CABaseSetup.disable_caching(self, clear_cache=False)

    def clear_cache(self, **kwargs) -> None:
        CASetupDelegatorMixin.clear_cache(self, **kwargs)


def assert_value_not_none(instance: object, attribute: attr.Attribute, value: tp.Any) -> None:
    """Assert that the provided value is not None.

    Args:
        instance (object): Instance on which the attribute is being set.
        attribute (Attribute): Attribute for which the value is being validated.
        value (Any): Value to check.

    Returns:
        None
    """
    if value is None:
        raise ValueError(f"Please provide {attribute.name}")


CAClassSetupT = tp.TypeVar("CAClassSetupT", bound="CAClassSetup")


@define
class CAClassSetup(CABaseDelegatorSetup, DefineMixin):
    """Class that represents a setup for a cacheable class.

    The provided class must be a subclass of `vectorbtpro.utils.caching.Cacheable`.

    Delegates cache management to its child setups:

    * `CAClassSetup` for subclass setups.
    * `CAInstanceSetup` for instance setups.

    If `use_cache` or `whitelist` are None, inherits non-empty values from its superclass setups
    following the method resolution order (MRO).

    !!! note
        Unbound setups are not considered children of class setups. See notes on `CAUnboundSetup`.
    """

    cls: tp.Type[Cacheable] = define.field(default=None, validator=assert_value_not_none)
    """Cacheable class."""

    @staticmethod
    def get_hash(cls: tp.Type[Cacheable]) -> int:
        return hash((cls,))

    @staticmethod
    def get_cacheable_superclasses(cls: tp.Type[Cacheable]) -> tp.List[tp.Type[Cacheable]]:
        """Return an ordered list of cacheable superclasses.

        Args:
            cls (Type[Cacheable]): Cacheable class whose superclasses are retrieved.

        Returns:
            List[Type[Cacheable]]: Ordered list of cacheable superclasses, excluding the class itself.
        """
        superclasses = []
        for super_cls in inspect.getmro(cls):
            if issubclass(super_cls, Cacheable):
                if super_cls is not cls:
                    superclasses.append(super_cls)
        return superclasses

    @staticmethod
    def get_superclass_setups(
        registry: CacheableRegistry, cls: tp.Type[Cacheable]
    ) -> tp.List["CAClassSetup"]:
        """Return setups for cacheable superclasses.

        Args:
            registry (CacheableRegistry): Registry to retrieve class setups.
            cls (Type[Cacheable]): Cacheable class to analyze.

        Returns:
            List[CAClassSetup]: List of setups corresponding to each cacheable superclass.
        """
        setups = []
        for super_cls in CAClassSetup.get_cacheable_superclasses(cls):
            if registry.get_class_setup(super_cls) is not None:
                setups.append(super_cls.get_ca_setup())
        return setups

    @staticmethod
    def get_cacheable_subclasses(cls: tp.Type[Cacheable]) -> tp.List[tp.Type[Cacheable]]:
        """Return an ordered list of cacheable subclasses.

        Args:
            cls (Type[Cacheable]): Cacheable class to examine.

        Returns:
            List[Type[Cacheable]]: Ordered list of all cacheable subclasses, excluding the class itself.
        """
        subclasses = []
        for sub_cls in cls.__subclasses__():
            if issubclass(sub_cls, Cacheable):
                if sub_cls is not cls:
                    subclasses.append(sub_cls)
            subclasses.extend(CAClassSetup.get_cacheable_subclasses(sub_cls))
        return subclasses

    @staticmethod
    def get_subclass_setups(
        registry: CacheableRegistry, cls: tp.Type[Cacheable]
    ) -> tp.List["CAClassSetup"]:
        """Return setups for cacheable subclasses.

        Args:
            registry (CacheableRegistry): Registry used to retrieve class setups.
            cls (Type[Cacheable]): Cacheable class to examine.

        Returns:
            List[CAClassSetup]: List of setups corresponding to each cacheable subclass.
        """
        setups = []
        for super_cls in CAClassSetup.get_cacheable_subclasses(cls):
            if registry.get_class_setup(super_cls) is not None:
                setups.append(super_cls.get_ca_setup())
        return setups

    @staticmethod
    def get_unbound_cacheables(cls: tp.Type[Cacheable]) -> tp.Set[cacheableT]:
        """Return a set of unbound cacheable members.

        Args:
            cls (Type[Cacheable]): Cacheable class.

        Returns:
            Set[cacheableT]: Set containing unbound cacheable members.
        """
        members = inspect.getmembers(cls, is_bindable_cacheable)
        return {attr for attr_name, attr in members}

    @staticmethod
    def get_unbound_setups(
        registry: CacheableRegistry, cls: tp.Type[Cacheable]
    ) -> tp.Set["CAUnboundSetup"]:
        """Return setups for unbound cacheables.

        Args:
            registry (CacheableRegistry): Registry used to retrieve unbound setups.
            cls (Type[Cacheable]): Cacheable class to inspect for unbound cacheable members.

        Returns:
            Set[CAUnboundSetup]: Set of setups corresponding to each unbound cacheable member.
        """
        setups = set()
        for cacheable in CAClassSetup.get_unbound_cacheables(cls):
            if registry.get_unbound_setup(cacheable) is not None:
                setups.add(cacheable.get_ca_setup())
        return setups

    @classmethod
    def get(
        cls: tp.Type[CAClassSetupT],
        cls_: tp.Type[Cacheable],
        registry: CacheableRegistry = ca_reg,
        **kwargs,
    ) -> tp.Optional[CAClassSetupT]:
        """Retrieve or create a cacheable class setup.

        Args:
            cls_ (Type[Cacheable]): Cacheable class to retrieve or register.
            registry (CacheableRegistry): Registry to use for retrieving the setup.
            **kwargs: Keyword arguments for `CAClassSetup`.

        Returns:
            Optional[CAClassSetup]: Retrieved or newly registered setup,
                or None if caching machinery is disabled or the setup is inactive.

        !!! info
            For default settings, see `vectorbtpro._settings.caching`.
        """
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if caching_cfg["disable_machinery"]:
            return None

        setup = registry.get_class_setup(cls_)
        if setup is not None:
            if not setup.active:
                return None
            return setup
        instance = cls(cls=cls_, registry=registry, **kwargs)
        instance.enforce_rules()
        if instance.active:
            instance.register()
        return instance

    def __attrs_post_init__(self) -> None:
        CABaseSetup.__attrs_post_init__(self)

        checks.assert_subclass_of(self.cls, Cacheable)

        use_cache = self.use_cache
        whitelist = self.whitelist
        if use_cache is None or whitelist is None:
            superclass_setups = self.superclass_setups[::-1]
            for setup in superclass_setups:
                if use_cache is None:
                    if setup.use_cache is not None:
                        object.__setattr__(self, "use_cache", setup.use_cache)
                if whitelist is None:
                    if setup.whitelist is not None:
                        object.__setattr__(self, "whitelist", setup.whitelist)

    @property
    def query(self) -> CAQuery:
        return CAQuery(base_cls=self.cls)

    @property
    def superclass_setups(self) -> tp.List["CAClassSetup"]:
        """List of setups for each cacheable superclass.

        Returns:
            List[CAClassSetup]: List of class setups corresponding to each cacheable
                superclass of `CAClassSetup.cls`.
        """
        return self.get_superclass_setups(self.registry, self.cls)

    @property
    def subclass_setups(self) -> tp.List["CAClassSetup"]:
        """List of setups for each cacheable subclass.

        Returns:
            List[CAClassSetup]: List of class setups corresponding to each cacheable
                subclass of `CAClassSetup.cls`.
        """
        return self.get_subclass_setups(self.registry, self.cls)

    @property
    def unbound_setups(self) -> tp.Set["CAUnboundSetup"]:
        """Set of setups for each unbound cacheable member.

        Returns:
            Set[CAUnboundSetup]: Set of unbound setup instances for cacheable members
                defined in `CAClassSetup.cls`.
        """
        return self.get_unbound_setups(self.registry, self.cls)

    @property
    def instance_setups(self) -> tp.Set["CAInstanceSetup"]:
        """Set of setups for instances associated with this class setup.

        Returns:
            Set[CAInstanceSetup]: Set of instance setup instances whose `class_setup` attribute
                is equal to `CAClassSetup`.
        """
        matches = set()
        for instance_setup in self.registry.instance_setups.values():
            if instance_setup.class_setup is self:
                matches.add(instance_setup)
        return matches

    @property
    def any_use_cache_lut(self) -> tp.Optional[datetime]:
        """Most recent update time for `use_cache` in this class or its superclasses.

        Returns:
            Optional[datetime]: Latest timestamp among `CAClassSetup.use_cache_lut` and
                those of its superclass setups, or None if no timestamp is available.
        """
        max_use_cache_lut = self.use_cache_lut
        for setup in self.superclass_setups:
            if setup.use_cache_lut is not None:
                if max_use_cache_lut is None or setup.use_cache_lut > max_use_cache_lut:
                    max_use_cache_lut = setup.use_cache_lut
        return max_use_cache_lut

    @property
    def any_whitelist_lut(self) -> tp.Optional[datetime]:
        """Most recent update time for `whitelist` in this class or its superclasses.

        Returns:
            Optional[datetime]: Latest timestamp among `CAClassSetup.whitelist_lut` and
                those of its superclass setups, or None if no timestamp is available.
        """
        max_whitelist_lut = self.whitelist_lut
        for setup in self.superclass_setups:
            if setup.whitelist_lut is not None:
                if max_whitelist_lut is None or setup.whitelist_lut > max_whitelist_lut:
                    max_whitelist_lut = setup.whitelist_lut
        return max_whitelist_lut

    @property
    def child_setups(self) -> tp.Set[tp.Union["CAClassSetup", "CAInstanceSetup"]]:
        return set(self.subclass_setups) | self.instance_setups

    @property
    def same_type_setups(self) -> ValuesView:
        return self.registry.class_setups.values()

    @property
    def short_str(self) -> str:
        return f"<class {self.cls.__module__}.{self.cls.__name__}>"

    @property
    def readable_name(self) -> str:
        return self.cls.__name__

    @property
    def hash_key(self) -> tuple:
        return (self.cls,)


CAInstanceSetupT = tp.TypeVar("CAInstanceSetupT", bound="CAInstanceSetup")


@define
class CAInstanceSetup(CABaseDelegatorSetup, DefineMixin):
    """Class that represents a setup for an instance with bound cacheable objects.

    The provided instance must be of `Cacheable` from `vectorbtpro.utils.caching`.

    Delegates cache management to its child setups of type `CARunSetup`.

    If `use_cache` or `whitelist` are None, they inherit values from the parent class setup.
    """

    instance: tp.Union[Cacheable, ReferenceType] = define.field(
        default=None, validator=assert_value_not_none
    )
    """Cacheable instance."""

    @staticmethod
    def get_hash(instance: Cacheable) -> int:
        return hash((get_obj_id(instance),))

    @classmethod
    def get(
        cls: tp.Type[CAInstanceSetupT],
        instance: Cacheable,
        registry: CacheableRegistry = ca_reg,
        **kwargs,
    ) -> tp.Optional[CAInstanceSetupT]:
        """Get or register a new instance setup from a `CacheableRegistry`.

        Args:
            instance (Cacheable): Cacheable instance for which to retrieve or create a setup.
            registry (CacheableRegistry): Registry to query for an existing setup.
            **kwargs: Keyword arguments for `CAInstanceSetup`.

        Returns:
            Optional[CAInstanceSetup]: Active setup for the instance if found, otherwise None.

        !!! info
            For default settings, see `vectorbtpro._settings.caching`.
        """
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if caching_cfg["disable_machinery"]:
            return None

        setup = registry.get_instance_setup(instance)
        if setup is not None:
            if not setup.active:
                return None
            return setup
        instance = cls(instance=instance, registry=registry, **kwargs)
        instance.enforce_rules()
        if instance.active:
            instance.register()
        return instance

    def __attrs_post_init__(self) -> None:
        CABaseSetup.__attrs_post_init__(self)

        if not isinstance(self.instance, ReferenceType):
            checks.assert_instance_of(self.instance, Cacheable)
            instance_ref = ref(self.instance, lambda ref: self.registry.deregister_setup(self))
            object.__setattr__(self, "instance", instance_ref)

        if self.use_cache is None or self.whitelist is None:
            class_setup = self.class_setup
            if self.use_cache is None:
                if class_setup.use_cache is not None:
                    object.__setattr__(self, "use_cache", class_setup.use_cache)
            if self.whitelist is None:
                if class_setup.whitelist is not None:
                    object.__setattr__(self, "whitelist", class_setup.whitelist)

    @property
    def query(self) -> CAQuery:
        return CAQuery(instance=self.instance_obj)

    @property
    def instance_obj(self) -> tp.Union[Cacheable, object]:
        """Underlying cacheable instance.

        This property returns the actual cacheable instance if it is still alive. If the instance has been
        garbage collected, it returns a designated garbage placeholder.

        Returns:
            Union[Cacheable, object]: Live instance of the cacheable object or a garbage marker.
        """
        if self.instance() is None:
            return _GARBAGE
        return self.instance()

    @property
    def contains_garbage(self) -> bool:
        """Indicates whether the underlying instance has been garbage collected.

        Returns:
            bool: True if the instance has been destroyed (garbage collected), otherwise False.
        """
        return self.instance_obj is _GARBAGE

    @property
    def class_setup(self) -> tp.Optional[CAClassSetup]:
        """Retrieves the cacheable class setup for the type of the underlying instance.

        If the instance is no longer available, None is returned.

        Returns:
            Optional[CAClassSetup]: Class setup for the instance's type, or None if the instance is garbage.
        """
        if self.contains_garbage:
            return None
        return CAClassSetup.get(type(self.instance_obj), self.registry)

    @property
    def unbound_setups(self) -> tp.Set["CAUnboundSetup"]:
        """Gets the set of unbound cacheable setups declared in the instance's class.

        Returns:
            Set[CAUnboundSetup]: Set of unbound setups associated with the class of the instance.

                Returns an empty set if the instance contains garbage.
        """
        if self.contains_garbage:
            return set()
        return self.class_setup.unbound_setups

    @property
    def run_setups(self) -> tp.Set["CARunSetup"]:
        """Retrieves all runnable cacheable setups bound to the instance.

        Returns:
            Set[CARunSetup]: Set of run setups associated with the instance.

                If the instance has been garbage collected, an empty set is returned.
        """
        if self.contains_garbage:
            return set()
        matches = set()
        for run_setup in self.registry.run_setups.values():
            if run_setup.instance_setup is self:
                matches.add(run_setup)
        return matches

    @property
    def child_setups(self) -> tp.Set["CARunSetup"]:
        return self.run_setups

    @property
    def same_type_setups(self) -> ValuesView:
        return self.registry.instance_setups.values()

    @property
    def short_str(self) -> str:
        if self.contains_garbage:
            return "<destroyed object>"
        return (
            f"<instance of {type(self.instance_obj).__module__}.{type(self.instance_obj).__name__}>"
        )

    @property
    def readable_name(self) -> str:
        if self.contains_garbage:
            return "_GARBAGE"
        return type(self.instance_obj).__name__.lower()

    @property
    def hash_key(self) -> tuple:
        return (get_obj_id(self.instance_obj),)


CAUnboundSetupT = tp.TypeVar("CAUnboundSetupT", bound="CAUnboundSetup")


@define
class CAUnboundSetup(CABaseDelegatorSetup, DefineMixin):
    """Class representing the setup for an unbound cacheable property or method.

    An unbound callable is declared in a class but is not yet bound to an instance.

    !!! note
        Unbound callables are regular functions without parent setups. Although declared within a class,
        there is no straightforward way to retrieve the class reference from the decorator.
        Therefore, searching for child setups of a specific class will not return unbound setups.

    Delegates cache management to its child setups of type `CARunSetup`. A single unbound cacheable
    can be bound to multiple instances, establishing a one-to-many relationship with `CARunSetup` instances.

    !!! hint
        Access unbound callables using class attributes instead of instance attributes.
    """

    cacheable: cacheableT = define.field(default=None, validator=assert_value_not_none)
    """Cacheable object associated with the unbound setup."""

    @staticmethod
    def get_hash(cacheable: cacheableT) -> int:
        return hash((cacheable,))

    @classmethod
    def get(
        cls: tp.Type[CAUnboundSetupT],
        cacheable: cacheableT,
        registry: CacheableRegistry = ca_reg,
        **kwargs,
    ) -> tp.Optional[CAUnboundSetupT]:
        """Get or register a new unbound setup from a `CacheableRegistry`.

        Args:
            cacheable (cacheable): Cacheable property or method for which to retrieve or create a setup.
            registry (CacheableRegistry): Registry to query for an existing setup.
            **kwargs: Keyword arguments for `CAUnboundSetup`.

        Returns:
            Optional[CAUnboundSetup]: Active unbound setup if found, otherwise None.

        !!! info
            For default settings, see `vectorbtpro._settings.caching`.
        """
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if caching_cfg["disable_machinery"]:
            return None

        setup = registry.get_unbound_setup(cacheable)
        if setup is not None:
            if not setup.active:
                return None
            return setup
        instance = cls(cacheable=cacheable, registry=registry, **kwargs)
        instance.enforce_rules()
        if instance.active:
            instance.register()
        return instance

    def __attrs_post_init__(self) -> None:
        CABaseSetup.__attrs_post_init__(self)

        if not is_bindable_cacheable(self.cacheable):
            raise TypeError("cacheable must be either cacheable_property or cacheable_method")

    @property
    def query(self) -> CAQuery:
        return CAQuery(cacheable=self.cacheable)

    @property
    def run_setups(self) -> tp.Set["CARunSetup"]:
        """Set of run setups for cacheables that have been bound to this unbound setup.

        Returns:
            Set[CARunSetup]: Set of run setups associated with the unbound cacheable.
        """
        matches = set()
        for run_setup in self.registry.run_setups.values():
            if run_setup.unbound_setup is self:
                matches.add(run_setup)
        return matches

    @property
    def child_setups(self) -> tp.Set["CARunSetup"]:
        return self.run_setups

    @property
    def same_type_setups(self) -> ValuesView:
        return self.registry.unbound_setups.values()

    @property
    def short_str(self) -> str:
        if is_cacheable_property(self.cacheable):
            return f"<unbound property {self.cacheable.func.__module__}.{self.cacheable.func.__name__}>"
        return f"<unbound method {self.cacheable.func.__module__}.{self.cacheable.func.__name__}>"

    @property
    def readable_name(self) -> str:
        if is_cacheable_property(self.cacheable):
            return f"{self.cacheable.func.__name__}"
        return f"{self.cacheable.func.__name__}()"

    @property
    def hash_key(self) -> tuple:
        return (self.cacheable,)


CARunSetupT = tp.TypeVar("CARunSetupT", bound="CARunSetup")


@define
class CARunResult(DefineMixin):
    """Class that represents a cached result from a run.

    !!! note
        This instance is uniquely identified by the hash of its arguments (`args_hash`).
    """

    args_hash: int = define.field()
    """Integer hash of the run arguments."""

    result: tp.Any = define.field()
    """Output produced by the run."""

    timer: Timer = define.field()
    """Timer instance that measures the execution duration of the run."""

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "_run_time", datetime.now(timezone.utc))
        object.__setattr__(self, "_hits", 0)
        object.__setattr__(self, "_first_hit_time", None)
        object.__setattr__(self, "_last_hit_time", None)

    @staticmethod
    def get_hash(args_hash: int) -> int:
        return hash((args_hash,))

    @property
    def result_size(self) -> int:
        """Memory size of the run result in bytes.

        Returns:
            int: Size in bytes of the cached run result.
        """
        return sys.getsizeof(self.result)

    @property
    def run_time(self) -> datetime:
        """Timestamp marking when the run was executed.

        Returns:
            datetime: UTC datetime when the function run was executed.
        """
        return object.__getattribute__(self, "_run_time")

    @property
    def hits(self) -> int:
        """Total number of cache hits recorded for this run result.

        Returns:
            int: Cumulative count of cache hits for this run result.
        """
        return object.__getattribute__(self, "_hits")

    @property
    def first_hit_time(self) -> tp.Optional[datetime]:
        """Timestamp of the first cache hit, if any.

        Returns:
            Optional[datetime]: UTC datetime of the first cache hit, or None if no hit has occurred.
        """
        return object.__getattribute__(self, "_first_hit_time")

    @property
    def last_hit_time(self) -> tp.Optional[datetime]:
        """Timestamp of the most recent cache hit, if any.

        Returns:
            Optional[datetime]: UTC datetime of the most recent cache hit, or None if no hit has occurred.
        """
        return object.__getattribute__(self, "_last_hit_time")

    def hit(self) -> tp.Any:
        """Record a cache hit and return the cached result.

        Updates the first and last hit timestamps and increments the hit counter.

        Returns:
            Any: Cached result from the run.
        """
        hit_time = datetime.now(timezone.utc)
        if self.first_hit_time is None:
            object.__setattr__(self, "_first_hit_time", hit_time)
        object.__setattr__(self, "_last_hit_time", hit_time)
        object.__setattr__(self, "_hits", self.hits + 1)
        return self.result

    @property
    def hash_key(self) -> tuple:
        return (self.args_hash,)


@define
class CARunSetup(CABaseSetup, DefineMixin):
    """Class that represents a runnable cacheable setup.

    Manages execution of functions and caching of results through the `CARunSetup.run` method.

    Accepts `cacheable` argument representing a cacheable property, method, or function from
    `vectorbtpro.utils.decorators`.

    The setup is uniquely identified by hashing the provided callable and, if applicable, the id of its
    bound instance.

    !!! note
        Cacheable properties and methods require an instance.

        Only one instance per unique combination of `cacheable` and `instance` is allowed concurrently.

    If `use_cache` or `whitelist` are None, the setup inherits a non-empty value from its parent instance setup
    or unbound setup, preferring the one updated most recently.

    !!! note
        Use the `CARunSetup.get` class method to create a setup. This method first checks if a setup with the
        same hash is already registered and active, returning it if found; otherwise, it creates and registers
        a new one. Direct instantiation of `CARunSetup` may raise an error if a duplicate setup exists.
    """

    cacheable: cacheableT = define.field(default=None, validator=assert_value_not_none)
    """Cacheable object (callable, property, or method) that defines caching behavior."""

    instance: tp.Union[Cacheable, ReferenceType] = define.field(default=None)
    """Instance to which the cacheable is bound."""

    max_size: tp.Optional[int] = define.field(default=None)
    """Maximum number of entries in `CARunSetup.cache`."""

    ignore_args: tp.Optional[tp.Iterable[tp.AnnArgQuery]] = define.field(default=None)
    """Arguments to ignore when computing the hash."""

    cache: tp.Dict[int, CARunResult] = define.field(factory=dict)
    """Dictionary of cached `CARunResult` instances keyed by their hash."""

    @staticmethod
    def get_hash(cacheable: cacheableT, instance: tp.Optional[Cacheable] = None) -> int:
        return hash((cacheable, get_obj_id(instance) if instance is not None else None))

    @classmethod
    def get(
        cls: tp.Type[CARunSetupT],
        cacheable: cacheableT,
        instance: tp.Optional[Cacheable] = None,
        registry: CacheableRegistry = ca_reg,
        **kwargs,
    ) -> tp.Optional[CARunSetupT]:
        """Retrieve an existing run setup from the registry or register a new one.

        Additional arguments are passed to the `CARunSetup` constructor.

        Args:
            cacheable (cacheable): Cacheable object (callable, property, or method)
                that defines caching behavior.
            instance (Optional[Cacheable]): Instance associated with the cacheable.
            registry (CacheableRegistry): Registry used to store and retrieve run setups.
            **kwargs: Keyword arguments for `CARunSetup`.

        Returns:
            Optional[CARunSetupT]: Existing or newly registered run setup,
                or None if caching machinery is disabled.

        !!! info
            For default settings, see `vectorbtpro._settings.caching`.
        """
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if caching_cfg["disable_machinery"]:
            return None

        setup = registry.get_run_setup(cacheable, instance=instance)
        if setup is not None:
            if not setup.active:
                return None
            return setup
        instance = cls(cacheable=cacheable, instance=instance, registry=registry, **kwargs)
        instance.enforce_rules()
        if instance.active:
            instance.register()
        return instance

    def __attrs_post_init__(self) -> None:
        CABaseSetup.__attrs_post_init__(self)

        if not is_cacheable(self.cacheable):
            raise TypeError(
                "cacheable must be either cacheable_property, cacheable_method, or cacheable"
            )
        if self.instance is None:
            if is_cacheable_property(self.cacheable):
                raise ValueError("CARunSetup requires an instance for cacheable_property")
            elif is_cacheable_method(self.cacheable):
                raise ValueError("CARunSetup requires an instance for cacheable_method")
        else:
            checks.assert_instance_of(self.instance, Cacheable)
            if is_cacheable_function(self.cacheable):
                raise ValueError("Cacheable functions can't have an instance")

        if self.instance is not None and not isinstance(self.instance, ReferenceType):
            checks.assert_instance_of(self.instance, Cacheable)
            instance_ref = ref(self.instance, lambda ref: self.registry.deregister_setup(self))
            object.__setattr__(self, "instance", instance_ref)

        if self.use_cache is None or self.whitelist is None:
            instance_setup = self.instance_setup
            unbound_setup = self.unbound_setup
            if self.use_cache is None:
                if (
                    instance_setup is not None
                    and unbound_setup is not None
                    and instance_setup.use_cache is not None
                    and unbound_setup.use_cache is not None
                ):
                    if unbound_setup.use_cache_lut is not None and (
                        instance_setup.class_setup.any_use_cache_lut is None
                        or unbound_setup.use_cache_lut
                        > instance_setup.class_setup.any_use_cache_lut
                    ):
                        # Unbound setup was updated more recently than any superclass setup
                        object.__setattr__(self, "use_cache", unbound_setup.use_cache)
                    else:
                        object.__setattr__(self, "use_cache", instance_setup.use_cache)
                elif instance_setup is not None and instance_setup.use_cache is not None:
                    object.__setattr__(self, "use_cache", instance_setup.use_cache)
                elif unbound_setup is not None and unbound_setup.use_cache is not None:
                    object.__setattr__(self, "use_cache", unbound_setup.use_cache)
            if self.whitelist is None:
                if (
                    instance_setup is not None
                    and unbound_setup is not None
                    and instance_setup.whitelist is not None
                    and unbound_setup.whitelist is not None
                ):
                    if unbound_setup.whitelist_lut is not None and (
                        instance_setup.class_setup.any_whitelist_lut is None
                        or unbound_setup.whitelist_lut
                        > instance_setup.class_setup.any_whitelist_lut
                    ):
                        # Unbound setup was updated more recently than any superclass setup
                        object.__setattr__(self, "whitelist", unbound_setup.whitelist)
                    else:
                        object.__setattr__(self, "whitelist", instance_setup.whitelist)
                elif instance_setup is not None and instance_setup.whitelist is not None:
                    object.__setattr__(self, "whitelist", instance_setup.whitelist)
                elif unbound_setup is not None and unbound_setup.whitelist is not None:
                    object.__setattr__(self, "whitelist", unbound_setup.whitelist)

    @property
    def query(self) -> CAQuery:
        return CAQuery(cacheable=self.cacheable, instance=self.instance_obj)

    @property
    def instance_obj(self) -> tp.Union[Cacheable, object]:
        """Bound instance object.

        If the referenced instance has been garbage collected, returns a garbage marker.

        Returns:
            Union[Cacheable, object]: Actual bound instance if it exists, or the garbage marker `_GARBAGE`.
        """
        if self.instance is not None and self.instance() is None:
            return _GARBAGE
        return self.instance() if self.instance is not None else None

    @property
    def contains_garbage(self) -> bool:
        """Indicates whether the bound instance has been destroyed (garbage collected).

        Returns:
            bool: True if the bound instance is no longer available (garbage collected), otherwise False.
        """
        return self.instance_obj is _GARBAGE

    @property
    def instance_setup(self) -> tp.Optional[CAInstanceSetup]:
        """Associated `CAInstanceSetup` for the bound instance.

        Returns:
            Optional[CAInstanceSetup]: `CAInstanceSetup` corresponding to the current instance
                if it exists and has not been garbage collected; otherwise, returns None.
        """
        if self.instance_obj is None or self.contains_garbage:
            return None
        return CAInstanceSetup.get(self.instance_obj, self.registry)

    @property
    def unbound_setup(self) -> tp.Optional[CAUnboundSetup]:
        """Associated `CAUnboundSetup` for the unbound cacheable.

        Returns:
            Optional[CAUnboundSetup]: `CAUnboundSetup` corresponding to the cacheable attribute,
                or None if it is not registered.
        """
        return self.registry.get_unbound_setup(self.cacheable)

    @property
    def hits(self) -> int:
        return sum([run_result.hits for run_result in self.cache.values()])

    @property
    def misses(self) -> int:
        return len(self.cache)

    @property
    def total_size(self) -> int:
        return sum([run_result.result_size for run_result in self.cache.values()])

    @property
    def total_elapsed(self) -> tp.Optional[timedelta]:
        total_elapsed = None
        for run_result in self.cache.values():
            elapsed = run_result.timer.elapsed(readable=False)
            if total_elapsed is None:
                total_elapsed = elapsed
            else:
                total_elapsed += elapsed
        return total_elapsed

    @property
    def total_saved(self) -> tp.Optional[timedelta]:
        total_saved = None
        for run_result in self.cache.values():
            saved = run_result.timer.elapsed(readable=False) * run_result.hits
            if total_saved is None:
                total_saved = saved
            else:
                total_saved += saved
        return total_saved

    @property
    def first_run_time(self) -> tp.Optional[datetime]:
        if len(self.cache) == 0:
            return None
        return list(self.cache.values())[0].run_time

    @property
    def last_run_time(self) -> tp.Optional[datetime]:
        if len(self.cache) == 0:
            return None
        return list(self.cache.values())[-1].run_time

    @property
    def first_hit_time(self) -> tp.Optional[datetime]:
        first_hit_times = []
        for run_result in self.cache.values():
            if run_result.first_hit_time is not None:
                first_hit_times.append(run_result.first_hit_time)
        if len(first_hit_times) == 0:
            return None
        return sorted(first_hit_times)[0]

    @property
    def last_hit_time(self) -> tp.Optional[datetime]:
        last_hit_times = []
        for run_result in self.cache.values():
            if run_result.last_hit_time is not None:
                last_hit_times.append(run_result.last_hit_time)
        if len(last_hit_times) == 0:
            return None
        return sorted(last_hit_times)[-1]

    def run_func(self, *args, **kwargs) -> tp.Any:
        """Run the setup's function without caching.

        Args:
            *args: Positional arguments for `CARunSetup.cacheable.func`.
            **kwargs: Keyword arguments for `CARunSetup.cacheable.func`.

        Returns:
            Any: Result returned by `CARunSetup.cacheable.func`.
        """
        if self.instance_obj is not None:
            return self.cacheable.func(self.instance_obj, *args, **kwargs)
        return self.cacheable.func(*args, **kwargs)

    def get_args_hash(self, *args, **kwargs) -> tp.Optional[int]:
        """Get the hash of the provided arguments using `vectorbtpro.utils.hashing.hash_args`.

        This method extends `CARunSetup.ignore_args` with additional ignore arguments defined
        in `vectorbtpro._settings.caching`. If no arguments are provided, it returns the hash of None.

        Args:
            *args: Positional arguments to be included in the hash computation.
            **kwargs: Keyword arguments to be included in the hash computation.

        Returns:
            Optional[int]: Computed hash of the provided arguments.

        !!! info
            For default settings, see `vectorbtpro._settings.caching`.
        """
        if len(args) == 0 and len(kwargs) == 0:
            return hash(None)

        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        ignore_args = list(caching_cfg["ignore_args"])
        if self.ignore_args is not None:
            ignore_args.extend(list(self.ignore_args))

        return hash_args(
            self.cacheable.func,
            args if self.instance_obj is None else (get_obj_id(self.instance_obj), *args),
            kwargs,
            ignore_args=ignore_args,
        )

    def run_func_and_cache(self, *args, **kwargs) -> tp.Any:
        """Run the setup's function and cache the result.

        This method calculates a hash for the given arguments using `CARunSetup.get_args_hash`.
        It then executes the function via `CARunSetup.run_func`, wraps the result with `CARunResult`,
        and stores the outcome in the cache using the computed hash. If a cached result exists,
        it is returned via `CARunResult.hit()`.

        Args:
            *args: Positional arguments for `CARunSetup.run_func`.
            **kwargs: Keyword arguments for `CARunSetup.run_func`.

        Returns:
            Any: Result from executing the function.
        """
        args_hash = self.get_args_hash(*args, **kwargs)
        run_result_hash = CARunResult.get_hash(args_hash)
        if run_result_hash in self.cache:
            return self.cache[run_result_hash].hit()
        if self.max_size is not None and self.max_size <= len(self.cache):
            del self.cache[list(self.cache.keys())[0]]
        with Timer() as timer:
            result = self.run_func(*args, **kwargs)
        run_result = CARunResult(args_hash, result, timer=timer)
        self.cache[run_result_hash] = run_result
        return result

    def run(self, *args, **kwargs) -> tp.Any:
        """Run the setup's function with caching control.

        If caching is enabled and the provided arguments are hashable, this method executes
        `CARunSetup.run_func_and_cache`. Otherwise, it falls back to executing `CARunSetup.run_func` directly.

        Args:
            *args: Positional arguments for `CARunSetup.run_func`.
            **kwargs: Keyword arguments for `CARunSetup.run_func`.

        Returns:
            Any: Result of the function execution.
        """
        if self.caching_enabled:
            try:
                return self.run_func_and_cache(*args, **kwargs)
            except UnhashableArgsError:
                pass
        return self.run_func(*args, **kwargs)

    def clear_cache(self) -> None:
        self.cache.clear()

    @property
    def same_type_setups(self) -> ValuesView:
        return self.registry.run_setups.values()

    @property
    def short_str(self) -> str:
        if self.contains_garbage:
            return "<destroyed object>"
        if is_cacheable_property(self.cacheable):
            return (
                f"<instance property {type(self.instance_obj).__module__}."
                f"{type(self.instance_obj).__name__}.{self.cacheable.func.__name__}>"
            )
        if is_cacheable_method(self.cacheable):
            return (
                f"<instance method {type(self.instance_obj).__module__}."
                f"{type(self.instance_obj).__name__}.{self.cacheable.func.__name__}>"
            )
        return f"<func {self.cacheable.__module__}.{self.cacheable.__name__}>"

    @property
    def readable_name(self) -> str:
        if self.contains_garbage:
            return "_GARBAGE"
        if is_cacheable_property(self.cacheable):
            return f"{type(self.instance_obj).__name__.lower()}.{self.cacheable.func.__name__}"
        if is_cacheable_method(self.cacheable):
            return f"{type(self.instance_obj).__name__.lower()}.{self.cacheable.func.__name__}()"
        return f"{self.cacheable.__name__}()"

    @property
    def readable_str(self) -> str:
        if self.contains_garbage:
            return f"_GARBAGE:{self.position_among_similar}"
        if is_cacheable_property(self.cacheable):
            return (
                f"{type(self.instance_obj).__name__.lower()}:"
                f"{self.instance_setup.position_among_similar}."
                f"{self.cacheable.func.__name__}"
            )
        if is_cacheable_method(self.cacheable):
            return (
                f"{type(self.instance_obj).__name__.lower()}:"
                f"{self.instance_setup.position_among_similar}."
                f"{self.cacheable.func.__name__}()"
            )
        return f"{self.cacheable.__name__}():{self.position_among_similar}"

    @property
    def hash_key(self) -> tuple:
        return self.cacheable, get_obj_id(
            self.instance_obj
        ) if self.instance_obj is not None else None


class CAQueryDelegator(CASetupDelegatorMixin):
    """Class that delegates setups matching a query using a registry.

    All positional and keyword arguments are passed to `CacheableRegistry.match_setups`.

    Args:
        *args: Positional arguments for querying setups.
        registry (CacheableRegistry): Registry used to match setups.
        collapse (bool): If True, remove child setups belonging to any matched parent setup.
        **kwargs: Keyword arguments for querying setups.
    """

    def __init__(
        self, *args, registry: CacheableRegistry = ca_reg, collapse: bool = True, **kwargs
    ) -> None:
        self._args = args
        kwargs["collapse"] = collapse
        self._kwargs = kwargs
        self._registry = registry

    @property
    def args(self) -> tp.Args:
        """Positional arguments used for querying setups.

        Returns:
            Args: Tuple or list of positional arguments stored in this instance.
        """
        return self._args

    @property
    def kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for querying setups.

        Returns:
            Kwargs: Dictionary of keyword arguments stored in this instance.
        """
        return self._kwargs

    @property
    def registry(self) -> CacheableRegistry:
        """Cacheable registry used for matching setups.

        Returns:
            CacheableRegistry: Registry instance used to match and retrieve setups.
        """
        return self._registry

    @property
    def child_setups(self) -> tp.Set[CABaseSetup]:
        return self.registry.match_setups(*self.args, **self.kwargs)


def get_cache_stats(*args, **kwargs) -> tp.Optional[tp.Frame]:
    """Retrieve caching statistics globally or for a specific object.

    Args:
        *args: Positional arguments for `CAQueryDelegator`.
        **kwargs: Keyword arguments distributed between `CAQueryDelegator` and `CAQueryDelegator.get_stats`.

    Returns:
        Optional[Frame]: DataFrame containing caching statistics or None if not available.
    """
    delegator_kwargs = {}
    stats_kwargs = {}
    if len(kwargs) > 0:
        overview_arg_names = get_func_arg_names(CAQueryDelegator.get_stats)
        for k in list(kwargs.keys()):
            if k in overview_arg_names:
                stats_kwargs[k] = kwargs.pop(k)
            else:
                delegator_kwargs[k] = kwargs.pop(k)
    else:
        delegator_kwargs = kwargs
    return CAQueryDelegator(*args, **delegator_kwargs).get_stats(**stats_kwargs)


def print_cache_stats(*args, **kwargs) -> None:
    """Print caching statistics globally or for a specific object.

    Args:
        *args: Positional arguments for `get_cache_stats`.
        **kwargs: Keyword arguments for `get_cache_stats`.

    Returns:
        None
    """
    ptable(get_cache_stats(*args, **kwargs))


def clear_cache(*args, **kwargs) -> None:
    """Clear the global cache or the cache of a specific object.

    Args:
        *args: Positional arguments for `CAQueryDelegator`.
        **kwargs: Keyword arguments for `CAQueryDelegator`.

    Returns:
        None
    """
    return CAQueryDelegator(*args, **kwargs).clear_cache()


def collect_garbage() -> None:
    """Collect garbage using the Python garbage collector.

    Returns:
        None
    """
    import gc

    gc.collect()


def flush() -> None:
    """Clear the cache and trigger garbage collection.

    Returns:
        None
    """
    clear_cache()
    collect_garbage()


def disable_caching(clear_cache: bool = True) -> None:
    """Disable caching globally.

    Args:
        clear_cache (bool): Whether to clear the cache when disabling caching.

    Returns:
        None

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """
    from vectorbtpro._settings import settings

    caching_cfg = settings["caching"]

    caching_cfg["disable"] = True
    caching_cfg["disable_whitelist"] = True
    caching_cfg["disable_machinery"] = True

    if clear_cache:
        CAQueryDelegator().clear_cache()


def enable_caching() -> None:
    """Enable caching globally.

    Returns:
        None

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """
    from vectorbtpro._settings import settings

    caching_cfg = settings["caching"]

    caching_cfg["disable"] = False
    caching_cfg["disable_whitelist"] = False
    caching_cfg["disable_machinery"] = False


class CachingDisabled(Base):
    """Context manager to temporarily disable caching based on a query.

    Args:
        query_like (Optional[Any]): Value used for parsing the query via `CAQuery.parse`.
        use_base_cls (bool): Flag indicating whether the base class is used in query parsing.
        kind (Optional[MaybeIterable[str]]): Specifies the expected kind(s) of setup to match.

            See `CARegistry.match_setups`.
        exclude (Optional[MaybeIterable[CABaseSetup]]): Setup or setups to exclude from matching.

            See `CARegistry.match_setups`.
        filter_func (Optional[Callable]): Function that takes a setup and returns a boolean
            indicating whether the setup should be included.

            See `CARegistry.match_setups`.
        registry (CacheableRegistry): Registry instance for caching setups.
        disable_whitelist (bool): Whether to disable the whitelist.
        disable_machinery (bool): Flag to disable caching machinery.
        clear_cache (bool): Whether to clear the cache when disabling caching.
        silence_warnings (bool): Flag to suppress warning messages.

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """

    def __init__(
        self,
        query_like: tp.Optional[tp.Any] = None,
        use_base_cls: bool = True,
        kind: tp.Optional[tp.MaybeIterable[str]] = None,
        exclude: tp.Optional[tp.MaybeIterable["CABaseSetup"]] = None,
        filter_func: tp.Optional[tp.Callable] = None,
        registry: CacheableRegistry = ca_reg,
        disable_whitelist: bool = True,
        disable_machinery: bool = True,
        clear_cache: bool = True,
        silence_warnings: bool = False,
    ) -> None:
        self._query_like = query_like
        self._use_base_cls = use_base_cls
        self._kind = kind
        self._exclude = exclude
        self._filter_func = filter_func
        self._registry = registry
        self._disable_whitelist = disable_whitelist
        self._disable_machinery = disable_machinery
        self._clear_cache = clear_cache
        self._silence_warnings = silence_warnings

        self._rule = None
        self._init_settings = None
        self._init_setup_settings = None

    @property
    def query_like(self) -> tp.Optional[tp.Any]:
        """Query-like object used for parsing as utilized by `CAQuery.parse`.

        Returns:
            Optional[Any]: Original query-like object provided for parsing, or None if not set.
        """
        return self._query_like

    @property
    def use_base_cls(self) -> bool:
        """Whether to use the base class during query parsing as specified in `CAQuery.parse`.

        Returns:
            bool: True if the base class should be used; otherwise, False.
        """
        return self._use_base_cls

    @property
    def kind(self) -> tp.Optional[tp.MaybeIterable[str]]:
        """Kind or kinds used for filtering setups, as described in `CARule.kind`.

        Returns:
            Optional[Iterable[str]]: String or iterable of strings indicating the kinds of setups,
                or None if unspecified.
        """
        return self._kind

    @property
    def exclude(self) -> tp.Optional[tp.MaybeIterable["CABaseSetup"]]:
        """Setups to be excluded from caching, as defined in `CARule.exclude`.

        Returns:
            Optional[Iterable[CABaseSetup]]: Single setup or an iterable of setups to exclude,
                or None if not specified.
        """
        return self._exclude

    @property
    def filter_func(self) -> tp.Optional[tp.Callable]:
        """Filter function used for setups selection, as described in `CARule.filter_func`.

        Returns:
            Optional[Callable]: Function that filters setups, or None if no filter is applied.
        """
        return self._filter_func

    @property
    def registry(self) -> CacheableRegistry:
        """`CacheableRegistry` instance used for caching setups.

        Returns:
            CacheableRegistry: Registry that manages all caching setups.
        """
        return self._registry

    @property
    def disable_whitelist(self) -> bool:
        """Whether the whitelist is disabled.

        Returns:
            bool: True if the whitelist is disabled; otherwise, False.
        """
        return self._disable_whitelist

    @property
    def disable_machinery(self) -> bool:
        """Whether the caching machinery is disabled.

        Returns:
            bool: True if caching machinery is disabled; otherwise, False.
        """
        return self._disable_machinery

    @property
    def clear_cache(self) -> bool:
        """Whether cache clearing is performed when disabling caching.

        Returns:
            bool: True if the cache should be cleared; otherwise, False.
        """
        return self._clear_cache

    @property
    def silence_warnings(self) -> bool:
        """Whether warnings are suppressed during caching operations.

        Returns:
            bool: True if warnings are silenced; otherwise, False.
        """
        return self._silence_warnings

    @property
    def rule(self) -> tp.Optional[CARule]:
        """`CARule` instance applied if one is set.

        Returns:
            Optional[CARule]: Current caching rule applied, or None if no rule is active.
        """
        return self._rule

    @property
    def init_settings(self) -> tp.Kwargs:
        """Initial global caching settings captured upon entering the context.

        Returns:
            Kwargs: Dictionary containing the original global caching settings.
        """
        return self._init_settings

    @property
    def init_setup_settings(self) -> tp.Dict[int, dict]:
        """Initial caching settings for each setup, mapped by setup hash.

        Returns:
            Dict[int, dict]: Dictionary where each key is a setup hash and each value is another
                dictionary containing the initial settings (such as `active`, `whitelist`,
                and `use_cache` flags) for that setup.
        """
        return self._init_setup_settings

    def __enter__(self) -> tp.Self:
        if self.query_like is None:
            from vectorbtpro._settings import settings

            caching_cfg = settings["caching"]

            self._init_settings = dict(
                disable=caching_cfg["disable"],
                disable_whitelist=caching_cfg["disable_whitelist"],
                disable_machinery=caching_cfg["disable_machinery"],
            )

            caching_cfg["disable"] = True
            caching_cfg["disable_whitelist"] = self.disable_whitelist
            caching_cfg["disable_machinery"] = self.disable_machinery

            if self.clear_cache:
                clear_cache()
        else:

            def _enforce_func(setup):
                if self.disable_machinery:
                    setup.deactivate()
                if self.disable_whitelist:
                    setup.disable_whitelist()
                setup.disable_caching(clear_cache=self.clear_cache)

            query = CAQuery.parse(self.query_like, use_base_cls=self.use_base_cls)
            rule = CARule(
                query,
                _enforce_func,
                kind=self.kind,
                exclude=self.exclude,
                filter_func=self.filter_func,
            )
            self._rule = rule
            self.registry.register_rule(rule)

            init_setup_settings = dict()
            for setup_hash, setup in self.registry.setups.items():
                init_setup_settings[setup_hash] = dict(
                    active=setup.active,
                    whitelist=setup.whitelist,
                    use_cache=setup.use_cache,
                )
                rule.enforce(setup)
            self._init_setup_settings = init_setup_settings

        return self

    def __exit__(self, *args) -> None:
        if self.query_like is None:
            from vectorbtpro._settings import settings

            caching_cfg = settings["caching"]

            caching_cfg["disable"] = self.init_settings["disable"]
            caching_cfg["disable_whitelist"] = self.init_settings["disable_whitelist"]
            caching_cfg["disable_machinery"] = self.init_settings["disable_machinery"]
        else:
            self.registry.deregister_rule(self.rule)

            for setup_hash, setup_settings in self.init_setup_settings.items():
                if setup_hash in self.registry.setups:
                    setup = self.registry.setups[setup_hash]
                    if self.disable_machinery and setup_settings["active"]:
                        setup.activate()
                    if self.disable_whitelist and setup_settings["whitelist"]:
                        setup.enable_whitelist()
                    if setup_settings["use_cache"]:
                        setup.enable_caching(silence_warnings=self.silence_warnings)


def with_caching_disabled(*args, **caching_disabled_kwargs) -> tp.Callable:
    """Decorator to execute a function within a caching-disabled context using `CachingDisabled`.

    Args:
        func (Callable): Function to be decorated.
        **caching_disabled_kwargs: Keyword arguments used to initialize `CachingDisabled`.

    Returns:
        Callable: Decorated function if a function is provided, or a decorator function.
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with CachingDisabled(**caching_disabled_kwargs):
                return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


class CachingEnabled(Base):
    """Context manager to temporarily enable caching based on a query.

    Args:
        query_like (Optional[Any]): Query specification to restrict caching behavior.
        use_base_cls (bool): Flag indicating whether the base class is used in query parsing.
        kind (Optional[MaybeIterable[str]]): Specifies the expected kind(s) of setup to match.

            See `CARegistry.match_setups`.
        exclude (Optional[MaybeIterable[CABaseSetup]]): Setup or setups to exclude from matching.

            See `CARegistry.match_setups`.
        filter_func (Optional[Callable]): Function that takes a setup and returns a boolean
            indicating whether the setup should be included.

            See `CARegistry.match_setups`.
        registry (CacheableRegistry): Registry used for managing caching setups.
        enable_whitelist (bool): Flag to enable whitelist for caching setups.
        enable_machinery (bool): Flag to enable caching machinery.
        clear_cache (bool): Flag to clear caches upon exiting the context.
        silence_warnings (bool): Flag to suppress warning messages.

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """

    def __init__(
        self,
        query_like: tp.Optional[tp.Any] = None,
        use_base_cls: bool = True,
        kind: tp.Optional[tp.MaybeIterable[str]] = None,
        exclude: tp.Optional[tp.MaybeIterable["CABaseSetup"]] = None,
        filter_func: tp.Optional[tp.Callable] = None,
        registry: CacheableRegistry = ca_reg,
        enable_whitelist: bool = True,
        enable_machinery: bool = True,
        clear_cache: bool = True,
        silence_warnings: bool = False,
    ) -> None:
        self._query_like = query_like
        self._use_base_cls = use_base_cls
        self._kind = kind
        self._exclude = exclude
        self._filter_func = filter_func
        self._registry = registry
        self._enable_whitelist = enable_whitelist
        self._enable_machinery = enable_machinery
        self._clear_cache = clear_cache
        self._silence_warnings = silence_warnings

        self._rule = None
        self._init_settings = None
        self._init_setup_settings = None

    @property
    def query_like(self) -> tp.Optional[tp.Any]:
        """Query specification used for determining caching rules.

        See `CAQuery.parse`.

        Returns:
            Optional[Any]: Query specification used for caching rules.
        """
        return self._query_like

    @property
    def use_base_cls(self) -> bool:
        """Flag indicating whether the base class is used in query parsing.

        See `CAQuery.parse`.

        Returns:
            bool: True if the base class is used during query parsing; otherwise, False.
        """
        return self._use_base_cls

    @property
    def kind(self) -> tp.Optional[tp.MaybeIterable[str]]:
        """Type indicator for the caching rule.

        See `CARule.kind`.

        Returns:
            Optional[Iterable[str]]: Type or types that indicate which caching rule to apply.
        """
        return self._kind

    @property
    def exclude(self) -> tp.Optional[tp.MaybeIterable["CABaseSetup"]]:
        """Setups to be excluded from the caching rule.

        See `CARule.exclude`.

        Returns:
            Optional[Iterable[CABaseSetup]]: Caching setups that should be excluded from the rule.
        """
        return self._exclude

    @property
    def filter_func(self) -> tp.Optional[tp.Callable]:
        """Filter function applied to caching setups.

        See `CARule.filter_func`.

        Returns:
            Optional[Callable]: Filter function used to select applicable caching setups.
        """
        return self._filter_func

    @property
    def registry(self) -> CacheableRegistry:
        """Caching registry of type `CacheableRegistry` used to manage caching setups.

        Returns:
            CacheableRegistry: Registry instance responsible for caching setup management.
        """
        return self._registry

    @property
    def enable_whitelist(self) -> bool:
        """Flag indicating if whitelist is enabled for caching setups.

        Returns:
            bool: True if whitelist is enabled; otherwise, False.
        """
        return self._enable_whitelist

    @property
    def enable_machinery(self) -> bool:
        """Flag indicating if caching machinery is active.

        Returns:
            bool: True if the caching machinery is active; otherwise, False.
        """
        return self._enable_machinery

    @property
    def clear_cache(self) -> bool:
        """Flag indicating whether to clear the global cache upon exiting or
        the local cache when disabling caching.

        Returns:
            bool: True if cache clearing is enabled; otherwise, False.
        """
        return self._clear_cache

    @property
    def silence_warnings(self) -> bool:
        """Flag determining whether caching-related warnings are silenced.

        Returns:
            bool: True if caching warnings are suppressed; otherwise, False.
        """
        return self._silence_warnings

    @property
    def rule(self) -> tp.Optional[CARule]:
        """Caching rule applied by this context manager.

        Returns:
            Optional[CARule]: Caching rule currently applied, or None if no rule is active.
        """
        return self._rule

    @property
    def init_settings(self) -> tp.Kwargs:
        """Initial global caching settings recorded upon entering the context.

        Returns:
            Kwargs: Dictionary of the global caching settings at the time of context entry.
        """
        return self._init_settings

    @property
    def init_setup_settings(self) -> tp.Dict[int, dict]:
        """Dictionary mapping setup hashes to their initial settings
        (active state, whitelist, and caching usage).

        Returns:
            Dict[int, dict]: Dictionary where keys are setup hashes and values are
                dictionaries of initial settings.
        """
        return self._init_setup_settings

    def __enter__(self) -> tp.Self:
        if self.query_like is None:
            from vectorbtpro._settings import settings

            caching_cfg = settings["caching"]

            self._init_settings = dict(
                disable=caching_cfg["disable"],
                disable_whitelist=caching_cfg["disable_whitelist"],
                disable_machinery=caching_cfg["disable_machinery"],
            )

            caching_cfg["disable"] = False
            caching_cfg["disable_whitelist"] = not self.enable_whitelist
            caching_cfg["disable_machinery"] = not self.enable_machinery
        else:

            def _enforce_func(setup):
                if self.enable_machinery:
                    setup.activate()
                if self.enable_whitelist:
                    setup.enable_whitelist()
                setup.enable_caching(silence_warnings=self.silence_warnings)

            query = CAQuery.parse(self.query_like, use_base_cls=self.use_base_cls)
            rule = CARule(
                query,
                _enforce_func,
                kind=self.kind,
                exclude=self.exclude,
                filter_func=self.filter_func,
            )
            self._rule = rule
            self.registry.register_rule(rule)

            init_setup_settings = dict()
            for setup_hash, setup in self.registry.setups.items():
                init_setup_settings[setup_hash] = dict(
                    active=setup.active,
                    whitelist=setup.whitelist,
                    use_cache=setup.use_cache,
                )
                rule.enforce(setup)
            self._init_setup_settings = init_setup_settings

        return self

    def __exit__(self, *args) -> None:
        if self.query_like is None:
            from vectorbtpro._settings import settings

            caching_cfg = settings["caching"]

            caching_cfg["disable"] = self.init_settings["disable"]
            caching_cfg["disable_whitelist"] = self.init_settings["disable_whitelist"]
            caching_cfg["disable_machinery"] = self.init_settings["disable_machinery"]

            if self.clear_cache:
                clear_cache()
        else:
            self.registry.deregister_rule(self.rule)

            for setup_hash, setup_settings in self.init_setup_settings.items():
                if setup_hash in self.registry.setups:
                    setup = self.registry.setups[setup_hash]
                    if self.enable_machinery and not setup_settings["active"]:
                        setup.deactivate()
                    if self.enable_whitelist and not setup_settings["whitelist"]:
                        setup.disable_whitelist()
                    if not setup_settings["use_cache"]:
                        setup.disable_caching(clear_cache=self.clear_cache)


def with_caching_enabled(*args, **caching_enabled_kwargs) -> tp.Callable:
    """Decorator to execute a function within a caching-enabled context.

    This decorator wraps a function so that it runs with caching enabled via `CachingEnabled`.
    Keyword arguments provided are passed to the `CachingEnabled` context manager.

    Args:
        func (Callable): Function to be decorated.
        **caching_enabled_kwargs: Keyword arguments for configuring `CachingEnabled`.

    Returns:
        Callable: Decorator that wraps the function to run with caching enabled.
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with CachingEnabled(**caching_enabled_kwargs):
                return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
