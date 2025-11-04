# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing class decorators for attaching magic methods."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import Config, ReadonlyConfig

__all__ = []

__pdoc__ = {}

binary_magic_config = ReadonlyConfig(
    {
        "__eq__": dict(func=np.equal),
        "__ne__": dict(func=np.not_equal),
        "__lt__": dict(func=np.less),
        "__gt__": dict(func=np.greater),
        "__le__": dict(func=np.less_equal),
        "__ge__": dict(func=np.greater_equal),
        # arithmetic ops
        "__add__": dict(func=np.add),
        "__sub__": dict(func=np.subtract),
        "__mul__": dict(func=np.multiply),
        "__pow__": dict(func=np.power),
        "__mod__": dict(func=np.mod),
        "__floordiv__": dict(func=np.floor_divide),
        "__truediv__": dict(func=np.true_divide),
        "__radd__": dict(func=lambda x, y: np.add(y, x)),
        "__rsub__": dict(func=lambda x, y: np.subtract(y, x)),
        "__rmul__": dict(func=lambda x, y: np.multiply(y, x)),
        "__rpow__": dict(func=lambda x, y: np.power(y, x)),
        "__rmod__": dict(func=lambda x, y: np.mod(y, x)),
        "__rfloordiv__": dict(func=lambda x, y: np.floor_divide(y, x)),
        "__rtruediv__": dict(func=lambda x, y: np.true_divide(y, x)),
        # mask ops
        "__and__": dict(func=np.bitwise_and),
        "__or__": dict(func=np.bitwise_or),
        "__xor__": dict(func=np.bitwise_xor),
        "__rand__": dict(func=lambda x, y: np.bitwise_and(y, x)),
        "__ror__": dict(func=lambda x, y: np.bitwise_or(y, x)),
        "__rxor__": dict(func=lambda x, y: np.bitwise_xor(y, x)),
    },
    options_=dict(as_attrs=False),
)
"""_"""

__pdoc__[
    "binary_magic_config"
] = f"""Configuration of binary magic methods for attaching to a class.

```python
{binary_magic_config.prettify_doc()}
```
"""


def attach_binary_magic_methods(
    translate_func: tp.BinaryTranslateFunc,
    config: tp.Optional[Config] = None,
) -> tp.ClassWrapper:
    """Attach binary magic methods to a class.

    Args:
        translate_func (BinaryTranslateFunc): Function that takes the instance,
            another operand, and a binary operator function, performs the operation, and returns the result.
        config (Optional[Config]): Configuration mapping of magic method names to settings.

            If not provided, defaults to `binary_magic_config`.

    Returns:
        ClassWrapper: Decorated class with attached binary magic methods.
    """
    if config is None:
        config = binary_magic_config

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        for target_name, settings in config.items():
            func = settings["func"]

            def new_method(
                self,
                other: tp.Any,
                _translate_func: tp.BinaryTranslateFunc = translate_func,
                _func: tp.Callable = func,
            ) -> tp.SeriesFrame:
                return _translate_func(self, other, _func)

            new_method.__name__ = target_name
            new_method.__module__ = cls.__module__
            new_method.__qualname__ = f"{cls.__name__}.{new_method.__name__}"
            setattr(cls, target_name, new_method)
        return cls

    return wrapper


unary_magic_config = ReadonlyConfig(
    {
        "__neg__": dict(func=np.negative),
        "__pos__": dict(func=np.positive),
        "__abs__": dict(func=np.absolute),
        "__invert__": dict(func=np.invert),
    }
)
"""_"""

__pdoc__["unary_magic_config"] = f"""Configuration of unary magic methods for attaching to a class.

```python
{unary_magic_config.prettify_doc()}
```
"""


def attach_unary_magic_methods(
    translate_func: tp.UnaryTranslateFunc,
    config: tp.Optional[Config] = None,
) -> tp.ClassWrapper:
    """Attach unary magic methods to a class.

    Args:
        translate_func (UnaryTranslateFunc): Function that takes the instance and a unary
            operator function, performs the operation, and returns the result.
        config (Optional[Config]): Configuration mapping of magic method names to settings.

            If not provided, defaults to `unary_magic_config`.

    Returns:
        ClassWrapper: Decorated class with attached unary magic methods.
    """
    if config is None:
        config = unary_magic_config

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        for target_name, settings in config.items():
            func = settings["func"]

            def new_method(
                self,
                _translate_func: tp.UnaryTranslateFunc = translate_func,
                _func: tp.Callable = func,
            ) -> tp.SeriesFrame:
                return _translate_func(self, _func)

            new_method.__name__ = target_name
            new_method.__module__ = cls.__module__
            new_method.__qualname__ = f"{cls.__name__}.{new_method.__name__}"
            setattr(cls, target_name, new_method)
        return cls

    return wrapper
