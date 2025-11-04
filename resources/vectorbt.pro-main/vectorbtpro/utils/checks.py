# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for validation during runtime."""

import datetime
import traceback
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from inspect import getmro, signature
from keyword import iskeyword
from types import BuiltinFunctionType, FunctionType, MethodType

import attr
import numba
import numpy as np
import pandas as pd
from numba.core.registry import CPUDispatcher

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "is_numba_enabled",
    "is_deep_equal",
]


class Comparable(Base):
    """Class for objects that support deep equality comparison."""

    def equals(self, other: tp.Any, *args, **kwargs) -> bool:
        """Return whether the current object is deeply equal to another object.

        Args:
            other (Any): Object to compare against.
            *args: Positional arguments for `is_deep_equal`.
            **kwargs: Keyword arguments for `is_deep_equal`.

        Returns:
            bool: True if the objects are equal, False otherwise.

        !!! note
            This method should accept all keyword arguments supported by `is_deep_equal`.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def __eq__(self, other: tp.Any) -> bool:
        return self.equals(other)


# ############# Checks ############# #


def is_classic_func(obj: tp.Any) -> bool:
    """Return whether the object is a classic Python function.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a classic function, False otherwise.
    """
    return isinstance(obj, FunctionType)


def is_builtin_func(obj: tp.Any) -> bool:
    """Return whether the object is a built-in function.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a built-in function, False otherwise.
    """
    return isinstance(obj, BuiltinFunctionType)


def is_method(obj: tp.Any) -> bool:
    """Return whether the object is a method.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a method, False otherwise.
    """
    return isinstance(obj, MethodType)


def is_numba_enabled() -> bool:
    """Return whether Numba is globally enabled.

    Returns:
        bool: True if Numba is enabled, False otherwise.
    """
    return numba.config.DISABLE_JIT != 1


def is_numba_func(obj: tp.Any) -> bool:
    """Return whether the object is identified as a Numba-compiled function based on configuration.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a Numba-compiled function, False otherwise.

    !!! info
        For default settings, see `vectorbtpro._settings.numba`.
    """
    from vectorbtpro._settings import settings

    numba_cfg = settings["numba"]

    if not numba_cfg["check_func_type"]:
        return True
    if not is_numba_enabled():
        if numba_cfg["check_func_suffix"]:
            if hasattr(obj, "__name__") and obj.__name__.endswith("_nb"):
                return True
            return False
        return False
    return isinstance(obj, CPUDispatcher)


def is_function(obj: tp.Any) -> bool:
    """Return whether the object is a lambda function, built-in function, method, or Numba-compiled function.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a function, False otherwise.
    """
    return is_classic_func(obj) or is_builtin_func(obj) or is_method(obj) or is_numba_func(obj)


def is_bool(obj: tp.Any) -> bool:
    """Return whether the object is a boolean value.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a boolean, False otherwise.
    """
    return isinstance(obj, (bool, np.bool_))


def is_int(obj: tp.Any) -> bool:
    """Return whether the object is an integer (excluding booleans and timedelta values).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is an integer, False otherwise.
    """
    return (
        isinstance(obj, (int, np.integer))
        and not isinstance(obj, np.timedelta64)
        and not is_bool(obj)
    )


def is_float(obj: tp.Any) -> bool:
    """Return whether the object is a float.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a float, False otherwise.
    """
    return isinstance(obj, (float, np.floating))


def is_number(obj: tp.Any) -> bool:
    """Return whether the object is a number (integer or float).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a number, False otherwise.
    """
    return is_int(obj) or is_float(obj)


def is_np_scalar(obj: tp.Any) -> bool:
    """Return whether the object is a NumPy scalar.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a NumPy scalar, False otherwise.
    """
    return isinstance(obj, np.generic)


def is_td(obj: tp.Any) -> bool:
    """Return whether the object is a timedelta object (from Pandas, datetime, or NumPy).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a timedelta, False otherwise.
    """
    return isinstance(obj, (pd.Timedelta, datetime.timedelta, np.timedelta64))


def is_td_like(obj: tp.Any) -> bool:
    """Return whether the object is timedelta-like (i.e., a timedelta object, a number, or a string).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is timedelta-like, False otherwise.
    """
    return is_td(obj) or is_number(obj) or isinstance(obj, str)


def is_frequency(obj: tp.Any) -> bool:
    """Return whether the object is a frequency object (a timedelta or a Pandas DateOffset).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a frequency object, False otherwise.
    """
    return is_td(obj) or isinstance(obj, pd.DateOffset)


def is_frequency_like(obj: tp.Any) -> bool:
    """Return whether the object is frequency-like (i.e., a frequency object, a number, or a string).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is frequency-like, False otherwise.
    """
    return is_frequency(obj) or is_number(obj) or isinstance(obj, str)


def is_dt(obj: tp.Any) -> bool:
    """Return whether the object is a datetime object (from Pandas, datetime, or NumPy).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a datetime, False otherwise.
    """
    return isinstance(obj, (pd.Timestamp, datetime.datetime, np.datetime64))


def is_dt_like(obj: tp.Any) -> bool:
    """Return whether the object is datetime-like (i.e., a datetime object, a number, or a string).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is datetime-like, False otherwise.
    """
    return is_dt(obj) or is_number(obj) or isinstance(obj, str)


def is_time(obj: tp.Any) -> bool:
    """Return whether the object is a time object.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a time, False otherwise.
    """
    return isinstance(obj, datetime.time)


def is_time_like(obj: tp.Any) -> bool:
    """Return whether the object is time-like (i.e., a time object or a string).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is time-like, False otherwise.
    """
    return is_time(obj) or isinstance(obj, str)


def is_np_array(obj: tp.Any) -> bool:
    """Return whether the object is a NumPy array.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a NumPy array, False otherwise.
    """
    return isinstance(obj, np.ndarray)


def is_record_array(obj: tp.Any) -> bool:
    """Return whether the object is a structured NumPy array.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a structured NumPy array, False otherwise.
    """
    return is_np_array(obj) and obj.dtype.fields is not None


def is_series(obj: tp.Any) -> bool:
    """Return whether the object is a Pandas Series.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a Pandas Series, False otherwise.
    """
    return isinstance(obj, pd.Series)


def is_index(obj: tp.Any) -> bool:
    """Return whether the object is a Pandas Index.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a Pandas Index, False otherwise.
    """
    return isinstance(obj, pd.Index)


def is_multi_index(obj: tp.Any) -> bool:
    """Return whether the object is a Pandas MultiIndex.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a Pandas MultiIndex, False otherwise.
    """
    return isinstance(obj, pd.MultiIndex)


def is_frame(obj: tp.Any) -> bool:
    """Return whether the object is a Pandas DataFrame.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a Pandas DataFrame, False otherwise.
    """
    return isinstance(obj, pd.DataFrame)


def is_pandas(obj: tp.Any) -> bool:
    """Return whether the object is a Pandas object (Series, Index, or DataFrame).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a Pandas object, False otherwise.
    """
    return is_series(obj) or is_index(obj) or is_frame(obj)


def is_any_array(obj: tp.Any) -> bool:
    """Return whether the object is any array-like object (a NumPy array or a Pandas object).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is any array-like object, False otherwise.
    """
    return is_pandas(obj) or isinstance(obj, np.ndarray)


def to_any_array(obj: tp.ArrayLike) -> tp.AnyArray:
    """Convert any array-like object to an array.

    Pandas objects are kept as-is.

    Args:
        obj (ArrayLike): Object to convert.

    Returns:
        AnyArray: Converted array.
    """
    if is_any_array(obj):
        return obj
    return np.asarray(obj)


def is_collection(obj: tp.Any) -> bool:
    """Return whether the object is considered a collection.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a collection, False otherwise.
    """
    if isinstance(obj, Collection):
        return True
    try:
        len(obj)
        return True
    except TypeError:
        return False


def is_complex_collection(obj: tp.Any) -> bool:
    """Return whether the object is a complex collection
    (i.e., a collection that is not a string, bytes, or bytearray).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a complex collection, False otherwise.
    """
    if isinstance(obj, (str, bytes, bytearray)):
        return False
    return is_collection(obj)


def is_iterable(obj: tp.Any) -> bool:
    """Return whether the object is iterable.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is iterable, False otherwise.
    """
    if isinstance(obj, Iterable):
        return True
    try:
        _ = iter(obj)
        return True
    except TypeError:
        return False


def is_complex_iterable(obj: tp.Any) -> bool:
    """Return whether the object is a complex iterable
    (i.e., iterable but not a string, bytes, or bytearray).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a complex iterable, False otherwise.
    """
    if isinstance(obj, (str, bytes, bytearray)):
        return False
    return is_iterable(obj)


def is_sequence(obj: tp.Any) -> bool:
    """Return whether the object is a sequence.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a sequence, False otherwise.
    """
    if isinstance(obj, Sequence):
        return True
    try:
        len(obj)
        obj[0:0]
        return True
    except (TypeError, IndexError, KeyError):
        return False


def is_complex_sequence(obj: tp.Any) -> bool:
    """Return whether the object is a complex sequence
    (i.e., a sequence that is not a string, bytes, or bytearray).

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a complex sequence, False otherwise.
    """
    if isinstance(obj, (str, bytes, bytearray)):
        return False
    return is_sequence(obj)


def is_hashable(obj: tp.Any) -> bool:
    """Return whether the object can be hashed.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is hashable, False otherwise.

    !!! note
        An object with a `__hash__` method might still be unhashable if invoking `hash` raises a TypeError.
    """
    if not isinstance(obj, Hashable):
        return False
    # Having __hash__() method does not mean that it's hashable
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def is_index_equal(obj1: tp.Any, obj2: tp.Any, check_names: bool = True) -> bool:
    """Return whether two indexes are equal.

    Args:
        obj1 (Any): First index to compare.
        obj2 (Any): Second index to compare.
        check_names (bool): Whether to check index names in addition to values.

    Returns:
        bool: True if the indexes are equal, False otherwise.
    """
    if not check_names:
        return pd.Index.equals(obj1, obj2)
    if isinstance(obj1, pd.MultiIndex) and isinstance(obj2, pd.MultiIndex):
        if obj1.names != obj2.names:
            return False
    elif isinstance(obj1, pd.MultiIndex) or isinstance(obj2, pd.MultiIndex):
        return False
    else:
        if obj1.name != obj2.name:
            return False
    return pd.Index.equals(obj1, obj2)


def is_default_index(obj: tp.Any, check_names: bool = True) -> bool:
    """Return whether the provided index is a basic range index.

    Args:
        obj (Any): Index to check.
        check_names (bool): Whether to check index names in addition to values.

    Returns:
        bool: True if the index is a default range index, False otherwise.
    """
    return is_index_equal(
        obj, pd.RangeIndex(start=0, stop=len(obj), step=1), check_names=check_names
    )


def is_namedtuple(obj: tp.Any) -> bool:
    """Return whether the given object is an instance of a namedtuple.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a namedtuple, False otherwise.
    """
    if not isinstance(obj, type):
        obj = type(obj)
    bases = obj.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(obj, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(field) == str for field in fields)


def is_record(obj: tp.Any) -> bool:
    """Return whether the given object is a NumPy record.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a NumPy record, False otherwise.
    """
    return (
        isinstance(obj, (np.void, np.record))
        and hasattr(obj.dtype, "names")
        and obj.dtype.names is not None
    )


def func_accepts_arg(
    func: tp.Callable, arg_name: str, arg_kind: tp.Optional[tp.MaybeTuple[int]] = None
) -> bool:
    """Return whether the function accepts an argument with the specified name.

    Args:
        func (Callable): Function to inspect.
        arg_name (str): Name of the argument to verify.
        arg_kind (Optional[MaybeTuple[int]]): Kind or kinds of argument to check.

    Returns:
        bool: True if the function accepts the argument, False otherwise.
    """
    sig = signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    if arg_kind is None:
        if arg_name.startswith("**"):
            return arg_name[2:] in [
                p.name for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD
            ]
        if arg_name.startswith("*"):
            return arg_name[1:] in [
                p.name for p in sig.parameters.values() if p.kind == p.VAR_POSITIONAL
            ]
        return arg_name in [
            p.name
            for p in sig.parameters.values()
            if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD
        ]
    return arg_name in [p.name for p in sig.parameters.values() if p.kind in arg_kind]


def is_equal(
    obj1: tp.Any,
    obj2: tp.Any,
    equality_func: tp.Callable[[tp.Any, tp.Any], bool] = lambda x, y: x == y,
) -> bool:
    """Return whether two objects are equal using the provided equality function.

    Args:
        obj1 (Any): First object for comparison.
        obj2 (Any): Second object for comparison.
        equality_func (Callable[[Any, Any], bool]): Function to evaluate equality.

    Returns:
        bool: True if the objects are equal, False otherwise.
    """
    try:
        return equality_func(obj1, obj2)
    except Exception:
        pass
    return False


def is_deep_equal(
    obj1: tp.Any,
    obj2: tp.Any,
    check_exact: bool = False,
    debug: bool = False,
    only_types: bool = False,
    _key: tp.Optional[str] = None,
    **kwargs,
) -> bool:
    """Return whether two objects are deeply equal by performing a recursive comparison.

    Args:
        obj1 (Any): First object for deep comparison.
        obj2 (Any): Second object for deep comparison.
        check_exact (bool): If True, enforce exact matching in comparisons.
        debug (bool): If True, output warning messages on mismatches.
        only_types (bool): If True, only compare the types of the objects.
        **kwargs: Keyword arguments for underlying comparison functions.

    Returns:
        bool: True if the objects are deeply equal, False otherwise.
    """

    def _select_kwargs(_method, _kwargs):
        __kwargs = dict()
        if len(kwargs) > 0:
            for k, v in _kwargs.items():
                if func_accepts_arg(_method, k):
                    __kwargs[k] = v
        return __kwargs

    def _check_array(assert_method):
        __kwargs = _select_kwargs(assert_method, kwargs)
        if obj1.dtype != obj2.dtype:
            raise AssertionError(f"Dtypes {obj1.dtype} and {obj2.dtype} do not match")
        if obj1.dtype.fields is not None:
            for field in obj1.dtype.names:
                try:
                    assert_method(obj1[field], obj2[field], **__kwargs)
                except Exception as e:
                    raise AssertionError(f"Dtype field '{field}'") from e
        else:
            assert_method(obj1, obj2, **__kwargs)

    try:
        if only_types:
            if type(obj1) != type(obj2):
                raise AssertionError(f"Types {type(obj1)} and {type(obj2)} do not match")
            return True
        if id(obj1) == id(obj2):
            return True
        if isinstance(obj1, Comparable):
            return obj1.equals(
                obj2,
                check_exact=check_exact,
                debug=debug,
                only_types=only_types,
                _key=_key,
                **kwargs,
            )
        if type(obj1) != type(obj2):
            raise AssertionError(f"Types {type(obj1)} and {type(obj2)} do not match")
        if attr.has(type(obj1)):
            return is_deep_equal(
                attr.asdict(obj1),
                attr.asdict(obj2),
                check_exact=check_exact,
                debug=debug,
                only_types=only_types,
                _key=_key,
                **kwargs,
            )
        if isinstance(obj1, pd.Series):
            _kwargs = _select_kwargs(pd.testing.assert_series_equal, kwargs)
            pd.testing.assert_series_equal(obj1, obj2, check_exact=check_exact, **_kwargs)
        elif isinstance(obj1, pd.DataFrame):
            _kwargs = _select_kwargs(pd.testing.assert_frame_equal, kwargs)
            pd.testing.assert_frame_equal(obj1, obj2, check_exact=check_exact, **_kwargs)
        elif isinstance(obj1, pd.Index):
            _kwargs = _select_kwargs(pd.testing.assert_index_equal, kwargs)
            pd.testing.assert_index_equal(obj1, obj2, check_exact=check_exact, **_kwargs)
        elif isinstance(obj1, np.ndarray):
            try:
                _check_array(np.testing.assert_array_equal)
            except Exception as e:
                if check_exact:
                    raise e
                _check_array(np.testing.assert_allclose)
        elif isinstance(obj1, (tuple, list)):
            for i in range(len(obj1)):
                if not is_deep_equal(
                    obj1[i],
                    obj2[i],
                    check_exact=check_exact,
                    debug=debug,
                    only_types=only_types,
                    _key=f"[{i}]" if _key is None else _key + f"[{i}]",
                    **kwargs,
                ):
                    return False
        elif isinstance(obj1, dict):
            for k in obj1.keys():
                if not is_deep_equal(
                    obj1[k],
                    obj2[k],
                    check_exact=check_exact,
                    debug=debug,
                    only_types=only_types,
                    _key=f"['{k}']" if _key is None else _key + f"['{k}']",
                    **kwargs,
                ):
                    return False
        else:
            try:
                if obj1 == obj2:
                    return True
            except Exception:
                pass
            try:
                import dill

                _kwargs = _select_kwargs(dill.dumps, kwargs)
                if dill.dumps(obj1, **_kwargs) == dill.dumps(obj2, **_kwargs):
                    return True
            except Exception:
                pass
            if debug:
                warn(f"\n############### {_key} ###############\nObjects do not match")
            return False
    except Exception:
        if debug:
            if _key is None:
                warn(traceback.format_exc())
            else:
                warn(f"\n############### {_key} ###############\n" + traceback.format_exc())
        return False
    return True


def is_class(obj: type, types: tp.TypeLike) -> bool:
    """Return whether the given class matches the specified type descriptor.

    Args:
        obj (type): Class to check.
        types (TypeLike): Type, string, or `vectorbtpro.utils.parsing.Regex` pattern
            (or tuple of such) representing the superclass.

    Returns:
        bool: True if the class matches the type descriptor, False otherwise.
    """
    from vectorbtpro.utils.parsing import Regex

    if isinstance(types, str):
        return str(obj) == types or obj.__name__ == types
    if isinstance(types, Regex):
        return types.matches(str(obj)) or types.matches(obj.__name__)
    if isinstance(types, tuple):
        for t in types:
            if is_class(obj, t):
                return True
        return False
    return obj is types


def is_subclass_of(obj: tp.Any, types: tp.TypeLike) -> bool:
    """Return whether the object is a subclass of the specified type descriptor.

    Args:
        obj (Any): Class to verify for subclassing.
        types (TypeLike): Type, string, or `vectorbtpro.utils.parsing.Regex` pattern
            (or tuple of such) representing the superclass.

    Returns:
        bool: True if the object is a subclass of the specified type descriptor, False otherwise.
    """
    try:
        return issubclass(obj, types)
    except TypeError:
        pass
    if isinstance(types, str):
        if types.lower() == "args":
            if is_namedtuple(obj):
                return False
            return issubclass(obj, tuple)
        for base_t in getmro(obj):
            if str(base_t) == types or base_t.__name__ == types:
                return True
    from vectorbtpro.utils.parsing import Regex

    if isinstance(types, Regex):
        for base_t in getmro(obj):
            if types.matches(str(base_t)) or types.matches(base_t.__name__):
                return True
    if isinstance(types, tuple):
        for t in types:
            if is_subclass_of(obj, t):
                return True
    return False


def is_instance_of(obj: tp.Any, types: tp.TypeLike) -> bool:
    """Return True if the object is an instance of the specified type(s).

    Args:
        obj (Any): Object to check.
        types (TypeLike): Type, string, or `vectorbtpro.utils.parsing.Regex` pattern
            (or tuple of such) representing the superclass.

    Returns:
        bool: True if the object is an instance of the specified type(s), False otherwise.
    """
    return is_subclass_of(type(obj), types)


def is_attrs_class(obj: tp.Any) -> bool:
    """Return True if the object is an `attrs`-decorated class.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is an `attrs`-decorated class, False otherwise.
    """
    return isinstance(obj, type) and attr.has(obj)


def is_attrs_subclass(obj: tp.Any) -> bool:
    """Return True if the object is a subclass of an `attrs`-decorated class.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a subclass of an `attrs`-decorated class, False otherwise.
    """
    return isinstance(obj, type) and any(attr.has(cls) for cls in obj.__mro__)


def is_attrs_instance(obj: tp.Any) -> bool:
    """Return True if the object is an instance of an `attrs`-decorated class.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is an instance of an `attrs`-decorated class, False otherwise.
    """
    return not isinstance(obj, type) and any(attr.has(cls) for cls in obj.__class__.__mro__)


def is_mapping(obj: tp.Any) -> bool:
    """Return True if the object is a mapping.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is a mapping, False otherwise.
    """
    return isinstance(obj, Mapping)


def is_mapping_like(obj: tp.Any) -> bool:
    """Return True if the object is mapping-like.

    An object is considered mapping-like if it is a mapping, a Series, an Index, or a NamedTuple.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if the object is mapping-like, False otherwise.
    """
    return is_mapping(obj) or is_series(obj) or is_index(obj) or is_namedtuple(obj)


def is_valid_variable_name(obj: str) -> bool:
    """Return True if the object is a valid variable name.

    Args:
        obj (str): String representing the variable name.

    Returns:
        bool: True if the string is a valid variable name, False otherwise.
    """
    return obj.isidentifier() and not iskeyword(obj)


def in_notebook() -> bool:
    """Return True if executing in a Jupyter notebook environment.

    This function checks the IPython configuration to determine if the code is running in a notebook.

    Returns:
        bool: True if executing in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


# ############# Asserts ############# #


def safe_assert(obj: bool, msg: tp.Optional[str] = None) -> None:
    """Assert that a condition is True.

    Args:
        obj (bool): Condition to evaluate.
        msg (Optional[str]): Error message to use if the assertion fails.

    Returns:
        None

    Raises:
        AssertionError: If the condition is False.
    """
    if not obj:
        raise AssertionError(msg)


def assert_in(obj1: tp.Any, obj2: tp.Sequence, arg_name: tp.Optional[str] = None) -> None:
    """Assert that `obj1` is present in `obj2`.

    Args:
        obj1 (Any): Element to search for.
        obj2 (Sequence): Sequence in which to search.
        arg_name (Optional[str]): Name of the argument for error messaging.

    Returns:
        None

    Raises:
        AssertionError: If `obj1` is not found in `obj2`.
    """
    if arg_name is None:
        x = ""
    else:
        x = f"for '{arg_name}'"
    if obj1 not in obj2:
        raise AssertionError(f"{obj1} not found in {obj2}{x}")


def assert_numba_func(func: tp.Callable) -> None:
    """Assert that the function is Numba-compiled.

    Args:
        func (Callable): Function to check for Numba compilation.

    Returns:
        None

    Raises:
        AssertionError: If the function is not Numba-compiled.
    """
    if not is_numba_func(func):
        raise AssertionError(f"Function {func} must be Numba compiled")


def assert_not_none(obj: tp.Any, arg_name: tp.Optional[str] = None) -> None:
    """Assert that the object is not None.

    Args:
        obj (Any): Value to check.
        arg_name (Optional[str]): Name of the argument for error messaging.

    Returns:
        None

    Raises:
        AssertionError: If the object is None.
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if obj is None:
        raise AssertionError(f"{x} cannot be None")


def assert_instance_of(obj: tp.Any, types: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Assert that object is an instance of the specified type(s).

    Args:
        obj (Any): Object to validate.
        types (TypeLike): Type, string, or `vectorbtpro.utils.parsing.Regex` pattern
            (or tuple of such) representing the superclass.
        arg_name (Optional[str]): Name of the argument for error messaging.

    Returns:
        None

    Raises:
        AssertionError: If the object is not an instance of the specified type(s).
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if not is_instance_of(obj, types):
        if isinstance(types, tuple):
            raise AssertionError(f"{x} must be of one of types {types}, not {type(obj)}")
        else:
            raise AssertionError(f"{x} must be of type {types}, not {type(obj)}")


def assert_not_instance_of(
    obj: tp.Any, types: tp.TypeLike, arg_name: tp.Optional[str] = None
) -> None:
    """Assert that object is not an instance of the specified type(s).

    Args:
        obj (Any): Object to validate.
        types (TypeLike): Type, string, or `vectorbtpro.utils.parsing.Regex` pattern
            (or tuple of such) representing the superclass.
        arg_name (Optional[str]): Name of the argument for error messaging.

    Returns:
        None

    Raises:
        AssertionError: If the object is an instance of the specified type(s).
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if is_instance_of(obj, types):
        if isinstance(types, tuple):
            raise AssertionError(f"{x} cannot be of one of types {types}")
        else:
            raise AssertionError(f"{x} cannot be of type {types}")


def assert_subclass_of(
    obj: tp.Type, classes: tp.TypeLike, arg_name: tp.Optional[str] = None
) -> None:
    """Assert that object is a subclass of the specified class(es).

    Args:
        obj (Type): Type to check.
        classes (TypeLike): Class or tuple of classes for validation.
        arg_name (Optional[str]): Name of the argument for error messaging.

    Returns:
        None

    Raises:
        AssertionError: If the object is not a subclass of the specified class(es).
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if not is_subclass_of(obj, classes):
        if isinstance(classes, tuple):
            raise AssertionError(f"{x} must be a subclass of one of types {classes}")
        else:
            raise AssertionError(f"{x} must be a subclass of type {classes}")


def assert_not_subclass_of(
    obj: tp.Type, classes: tp.TypeLike, arg_name: tp.Optional[str] = None
) -> None:
    """Assert that object is not a subclass of the specified class(es).

    Args:
        obj (Type): Type to check.
        classes (TypeLike): Class or tuple of classes for validation.
        arg_name (Optional[str]): Name of the argument for error messaging.

    Returns:
        None

    Raises:
        AssertionError: If the object is a subclass of the specified class(es).
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if is_subclass_of(obj, classes):
        if isinstance(classes, tuple):
            raise AssertionError(f"{x} cannot be a subclass of one of types {classes}")
        else:
            raise AssertionError(f"{x} cannot be a subclass of type {classes}")


def assert_type_equal(obj1: tp.Any, obj2: tp.Any) -> None:
    """Assert that `obj1` and `obj2` have the same type.

    Args:
        obj1 (Any): First object to compare.
        obj2 (Any): Second object to compare.

    Returns:
        None

    Raises:
        AssertionError: If the types of `obj1` and `obj2` do not match.
    """
    if type(obj1) != type(obj2):
        raise AssertionError(f"Types {type(obj1)} and {type(obj2)} do not match")


def assert_dtype(
    obj: tp.ArrayLike, dtype: tp.MaybeTuple[tp.DTypeLike], arg_name: tp.Optional[str] = None
) -> None:
    """Assert that the data type of object matches the specified dtype.

    For a DataFrame, each column's data type is validated.

    Args:
        obj (ArrayLike): Array or DataFrame to validate.
        dtype (MaybeTuple[DTypeLike]): Expected data type or a tuple of data types.
        arg_name (Optional[str]): Name of the argument for error messaging.

    Returns:
        None

    Raises:
        AssertionError: If the object's data type does not match the expected dtype.
    """
    if arg_name is None:
        x = "Data type"
    else:
        x = f"Data type of '{arg_name}'"
    obj = to_any_array(obj)
    if isinstance(dtype, tuple):
        if isinstance(obj, pd.DataFrame):
            for i, col_dtype in enumerate(obj.dtypes):
                if not any([col_dtype == _dtype for _dtype in dtype]):
                    raise AssertionError(
                        f"{x} (column {i}) must be one of {dtype}, not {col_dtype}"
                    )
        else:
            if not any([obj.dtype == _dtype for _dtype in dtype]):
                raise AssertionError(f"{x} must be one of {dtype}, not {obj.dtype}")
    else:
        if isinstance(obj, pd.DataFrame):
            for i, col_dtype in enumerate(obj.dtypes):
                if col_dtype != dtype:
                    raise AssertionError(f"{x} (column {i}) must be {dtype}, not {col_dtype}")
        else:
            if obj.dtype != dtype:
                raise AssertionError(f"{x} must be {dtype}, not {obj.dtype}")


def assert_subdtype(
    obj: tp.ArrayLike, dtype: tp.MaybeTuple[tp.DTypeLike], arg_name: tp.Optional[str] = None
) -> None:
    """Assert that the data type of object is a subtype of the specified dtype.

    For a DataFrame, each column's data type is validated.

    Args:
        obj (ArrayLike): Array or DataFrame to validate.
        dtype (MaybeTuple[DTypeLike]): Expected data type or a tuple of data types.
        arg_name (Optional[str]): Name of the argument for error messaging.

    Returns:
        None

    Raises:
        AssertionError: If the object's data type is not a subtype of the expected dtype.
    """
    if arg_name is None:
        x = "Data type"
    else:
        x = f"Data type of '{arg_name}'"
    obj = to_any_array(obj)
    if isinstance(dtype, tuple):
        if isinstance(obj, pd.DataFrame):
            for i, col_dtype in enumerate(obj.dtypes):
                if not any([np.issubdtype(col_dtype, _dtype) for _dtype in dtype]):
                    raise AssertionError(
                        f"{x} (column {i}) must be one of {dtype}, not {col_dtype}"
                    )
        else:
            if not any([np.issubdtype(obj.dtype, _dtype) for _dtype in dtype]):
                raise AssertionError(f"{x} must be one of {dtype}, not {obj.dtype}")
    else:
        if isinstance(obj, pd.DataFrame):
            for i, col_dtype in enumerate(obj.dtypes):
                if not np.issubdtype(col_dtype, dtype):
                    raise AssertionError(f"{x} (column {i}) must be {dtype}, not {col_dtype}")
        else:
            if not np.issubdtype(obj.dtype, dtype):
                raise AssertionError(f"{x} must be {dtype}, not {obj.dtype}")


def assert_dtype_equal(obj1: tp.ArrayLike, obj2: tp.ArrayLike) -> None:
    """Assert that the data types of `obj1` and `obj2` are equal.

    Args:
        obj1 (ArrayLike): First array or DataFrame to compare.
        obj2 (ArrayLike): Second array or DataFrame to compare.

    Returns:
        None

    Raises:
        AssertionError: If the data types of `obj1` and `obj2` do not match.
    """
    obj1 = to_any_array(obj1)
    obj2 = to_any_array(obj2)
    if isinstance(obj1, pd.DataFrame):
        dtypes1 = obj1.dtypes.to_numpy()
    else:
        dtypes1 = np.array([obj1.dtype])
    if isinstance(obj2, pd.DataFrame):
        dtypes2 = obj2.dtypes.to_numpy()
    else:
        dtypes2 = np.array([obj2.dtype])
    if len(dtypes1) == len(dtypes2):
        if (dtypes1 == dtypes2).all():
            return
    elif len(np.unique(dtypes1)) == 1 and len(np.unique(dtypes2)) == 1:
        if np.all(np.unique(dtypes1) == np.unique(dtypes2)):
            return
    raise AssertionError(f"Data types {dtypes1} and {dtypes2} do not match")


def assert_ndim(obj: tp.ArrayLike, ndims: tp.MaybeTuple[int]) -> None:
    """Assert that the number of dimensions of `obj` matches the specified `ndims`.

    Args:
        obj (ArrayLike): Array-like object to be checked.
        ndims (MaybeTuple[int]): Expected number of dimensions or acceptable dimension values.

    Returns:
        None

    Raises:
        AssertionError: If the number of dimensions of `obj` does not match `ndims`.
    """
    obj = to_any_array(obj)
    if isinstance(ndims, tuple):
        if obj.ndim not in ndims:
            raise AssertionError(f"Number of dimensions must be one of {ndims}, not {obj.ndim}")
    else:
        if obj.ndim != ndims:
            raise AssertionError(f"Number of dimensions must be {ndims}, not {obj.ndim}")


def assert_len_equal(obj1: tp.Sized, obj2: tp.Sized) -> None:
    """Assert that the lengths of `obj1` and `obj2` are equal.

    Args:
        obj1 (Sized): First object whose length is compared.
        obj2 (Sized): Second object whose length is compared.

    Returns:
        None

    Raises:
        AssertionError: If the lengths of `obj1` and `obj2` do not match.

    !!! note
        The objects are not converted to NumPy arrays.
    """
    if len(obj1) != len(obj2):
        raise AssertionError(f"Lengths of {obj1} and {obj2} do not match")


def assert_shape_equal(
    obj1: tp.ArrayLike,
    obj2: tp.ArrayLike,
    axis: tp.Optional[tp.Union[int, tp.Tuple[int, int]]] = None,
) -> None:
    """Assert that the shapes of `obj1` and `obj2` are equal along the specified axis.

    If `axis` is None, the entire shapes are compared.
    If `axis` is a tuple, the first element corresponds to `obj1` and the second to `obj2`.
    If `axis` is an integer, that axis index is compared for both arrays.

    Args:
        obj1 (ArrayLike): First array-like object.
        obj2 (ArrayLike): Second array-like object.
        axis (Optional[Union[int, Tuple[int, int]]): Axis or axes along which to compare shapes.

    Returns:
        None

    Raises:
        AssertionError: If the shapes of `obj1` and `obj2` do not match along the specified axis.
    """
    obj1 = to_any_array(obj1)
    obj2 = to_any_array(obj2)
    if axis is None:
        if obj1.shape != obj2.shape:
            raise AssertionError(f"Shapes {obj1.shape} and {obj2.shape} do not match")
    else:
        if isinstance(axis, tuple):
            if axis[0] >= obj1.ndim and axis[1] >= obj2.ndim:
                return
            if obj1.shape[axis[0]] != obj2.shape[axis[1]]:
                raise AssertionError(
                    f"Axis {axis[0]} of {obj1.shape} and axis {axis[1]} of {obj2.shape} do not match"
                )
        else:
            if axis >= obj1.ndim and axis >= obj2.ndim:
                return
            if obj1.shape[axis] != obj2.shape[axis]:
                raise AssertionError(f"Axis {axis} of {obj1.shape} and {obj2.shape} do not match")


def assert_index_equal(obj1: tp.Index, obj2: tp.Index, check_names: bool = True) -> None:
    """Assert that the indexes of `obj1` and `obj2` are equal.

    Args:
        obj1 (Index): First index to compare.
        obj2 (Index): Second index to compare.
        check_names (bool): Whether to check index names in addition to values.

    Returns:
        None

    Raises:
        AssertionError: If the indexes of `obj1` and `obj2` do not match.
    """
    if not is_index_equal(obj1, obj2, check_names=check_names):
        raise AssertionError(f"Indexes {obj1} and {obj2} do not match")


def assert_columns_equal(obj1: tp.Index, obj2: tp.Index, check_names: bool = True) -> None:
    """Assert that the columns of `obj1` and `obj2` are equal.

    Args:
        obj1 (Index): First columns index to compare.
        obj2 (Index): Second columns index to compare.
        check_names (bool): Whether to check index names in addition to values.

    Returns:
        None

    Raises:
        AssertionError: If the columns of `obj1` and `obj2` do not match.
    """
    if not is_index_equal(obj1, obj2, check_names=check_names):
        raise AssertionError(f"Columns {obj1} and {obj2} do not match")


def assert_meta_equal(
    obj1: tp.ArrayLike, obj2: tp.ArrayLike, axis: tp.Optional[int] = None
) -> None:
    """Assert that the metadata of `obj1` and `obj2` are equal.

    The function validates type and shape equality. For Pandas objects, it additionally compares
    indexes and, when applicable, columns or series names.

    Args:
        obj1 (ArrayLike): First array-like object.
        obj2 (ArrayLike): Second array-like object.
        axis (Optional[int]): Axis along which to compare metadata.

    Returns:
        None

    Raises:
        AssertionError: If the metadata of `obj1` and `obj2` do not match.
    """
    obj1 = to_any_array(obj1)
    obj2 = to_any_array(obj2)
    assert_type_equal(obj1, obj2)
    if axis is not None:
        assert_shape_equal(obj1, obj2, axis=axis)
    else:
        assert_shape_equal(obj1, obj2)
    if is_pandas(obj1) and is_pandas(obj2):
        if axis is None or axis == 0:
            assert_index_equal(obj1.index, obj2.index)
        if axis is None or axis == 1:
            if is_series(obj1) and is_series(obj2):
                assert_columns_equal(pd.Index([obj1.name]), pd.Index([obj2.name]))
            else:
                assert_columns_equal(obj1.columns, obj2.columns)


def assert_array_equal(obj1: tp.ArrayLike, obj2: tp.ArrayLike) -> None:
    """Assert that the values of `obj1` and `obj2` are equal.

    The function first compares metadata using `assert_meta_equal`, then checks actual data equality using:

    * Pandas equality check if both objects are Pandas.
    * NumPy array equality check otherwise.

    Args:
        obj1 (ArrayLike): First array-like object.
        obj2 (ArrayLike): Second array-like object.

    Returns:
        None

    Raises:
        AssertionError: If the metadata or values of `obj1` and `obj2` do not match.
    """
    obj1 = to_any_array(obj1)
    obj2 = to_any_array(obj2)
    assert_meta_equal(obj1, obj2)
    if is_pandas(obj1) and is_pandas(obj2):
        if obj1.equals(obj2):
            return
    elif not is_pandas(obj1) and not is_pandas(obj2):
        if np.array_equal(obj1, obj2):
            return
    raise AssertionError(f"Arrays {obj1} and {obj2} do not match")


def assert_level_not_exists(obj: tp.Index, level_name: str) -> None:
    """Assert that the specified level name does not exist in the index.

    Args:
        obj (Index): Index to check.
        level_name (str): Name of the level that must not exist.

    Returns:
        None

    Raises:
        AssertionError: If the level name already exists in the index.
    """
    if isinstance(obj, pd.MultiIndex):
        names = obj.names
    else:
        names = [obj.name]
    if level_name in names:
        raise AssertionError(f"Level {level_name} already exists in {names}")


def assert_equal(obj1: tp.Any, obj2: tp.Any, deep: bool = False) -> None:
    """Assert that `obj1` and `obj2` are equal.

    If `deep` is True, a deep equality check is performed.

    Args:
        obj1 (Any): First object to compare.
        obj2 (Any): Second object to compare.
        deep (bool): If True, perform a deep equality check.

    Returns:
        None

    Raises:
        AssertionError: If `obj1` and `obj2` are not equal.
    """
    if deep:
        if not is_deep_equal(obj1, obj2):
            raise AssertionError(f"{obj1} and {obj2} do not match (deep check)")
    else:
        if not is_equal(obj1, obj2):
            raise AssertionError(f"{obj1} and {obj2} do not match")


def assert_dict_valid(obj: tp.DictLike, lvl_keys: tp.Sequence[tp.MaybeSequence[str]]) -> None:
    """Assert that the dictionary `obj` contains only valid keys.

    `lvl_keys` should be a sequence of sequences, each corresponding to the valid keys
    for a level of the dictionary.

    Args:
        obj (DictLike): Dictionary to validate.
        lvl_keys (Sequence[MaybeSequence[str]]): Sequence of valid key sequences for each level.

    Returns:
        None

    Raises:
        AssertionError: If the object contains keys not present in `lvl_keys`.
    """
    if obj is None:
        obj = {}
    if len(lvl_keys) == 0:
        return
    if isinstance(lvl_keys[0], str):
        lvl_keys = [lvl_keys]
    set1 = set(obj.keys())
    set2 = set(lvl_keys[0])
    if not set1.issubset(set2):
        raise AssertionError(
            f"Invalid keys {list(set1.difference(set2))}, possible keys are {list(set2)}"
        )
    for k, v in obj.items():
        if isinstance(v, dict):
            assert_dict_valid(v, lvl_keys[1:])


def assert_dict_sequence_valid(
    obj: tp.DictLikeSequence, lvl_keys: tp.Sequence[tp.MaybeSequence[str]]
) -> None:
    """Assert that the dictionary and any dictionary within a sequence contains keys present in `lvl_keys`.

    Args:
        obj (DictLikeSequence): Dictionary or a sequence of dictionaries to validate.
        lvl_keys (Sequence[MaybeSequence[str]]): Sequence of valid key sequences for each level.

    Returns:
        None

    Raises:
        AssertionError: If the object contains keys not present in `lvl_keys`.
    """
    if obj is None:
        obj = {}
    if isinstance(obj, dict):
        assert_dict_valid(obj, lvl_keys)
    else:
        for o in obj:
            assert_dict_valid(o, lvl_keys)


def assert_sequence(obj: tp.Any) -> None:
    """Raise a `ValueError` if the object is not a sequence.

    Args:
        obj (Any): Object to test for sequence behavior.

    Returns:
        None
    """
    if not is_sequence(obj):
        raise ValueError(f"{obj} must be a sequence")


def assert_iterable(obj: tp.Any) -> None:
    """Raise a `ValueError` if the object is not an iterable.

    Args:
        obj (Any): Object to test for iterability.

    Returns:
        None
    """
    if not is_iterable(obj):
        raise ValueError(f"{obj} must be an iterable")
