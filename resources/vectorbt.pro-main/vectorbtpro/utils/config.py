# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utility functions for configuration management."""

import inspect
import uuid
from copy import copy, deepcopy

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import MISSING
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.caching import Cacheable
from vectorbtpro.utils.chaining import Chainable
from vectorbtpro.utils.checks import Comparable, assert_in, assert_instance_of, is_deep_equal
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.formatting import Prettified, prettify_dict, prettify_inited
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.pickling import Pickleable, RecState, pdict
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "hdict",
    "atomic_dict",
    "unsetkey",
    "merge_dicts",
    "flat_merge_dicts",
    "child_dict",
    "Config",
    "FrozenConfig",
    "ReadonlyConfig",
    "HybridConfig",
    "Configured",
    "AtomicConfig",
]

__pdoc__ = {}


class hdict(dict, Base):
    """Hashable dictionary subclass that computes its hash based on its items,
    enabling its use in sets or as dictionary keys."""

    def __hash__(self):
        return hash(frozenset(self.items()))


def resolve_dict(dct: tp.DictLikeSequence, i: tp.Optional[int] = None) -> dict:
    """Resolve and return a dictionary from a dictionary-like input.

    Args:
        dct (Union[dict, Sequence[dict]]): Dictionary or sequence of dictionaries.
        i (Optional[int]): Index to select a dictionary from the sequence, if applicable.

    Returns:
        dict: Shallow copy of the resolved dictionary.
    """
    if dct is None:
        dct = {}
    if isinstance(dct, dict):
        return dict(dct)
    if i is not None:
        _dct = dct[i]
        if _dct is None:
            _dct = {}
        return dict(_dct)
    raise ValueError("Cannot resolve dict")


class atomic_dict(pdict):
    """Dictionary subclass that is treated as a single atomic value during merge operations."""

    pass


def convert_to_dict(dct: tp.DictLike, nested: bool = True) -> dict:
    """Convert a configuration object to a dictionary.

    Args:
        dct (DictLike): Configuration input to convert.
        nested (bool): If True, recursively convert nested dictionaries.

    Returns:
        dict: Dictionary representing the configuration.

    !!! note
        When the input is an instance of `AtomicConfig`, it is converted to an `atomic_dict`.
    """
    if dct is None:
        dct = {}
    if isinstance(dct, Config):
        if isinstance(dct, AtomicConfig):
            dct = atomic_dict(dct)
        else:
            dct = dict(dct)
    else:
        dct = type(dct)(dct)
    if not nested:
        return dct
    for k, v in dct.items():
        if isinstance(v, dict):
            dct[k] = convert_to_dict(v, nested=nested)
        else:
            dct[k] = v
    return dct


def get_dict_item(dct: dict, k: tp.PathLikeKey, populate: bool = False) -> tp.Any:
    """Retrieve an item from a dictionary using a nested key.

    Args:
        dct (dict): Dictionary from which to retrieve the item.
        k (PathLikeKey): Key that may use dot notation, a `pathlib.Path`, or a tuple for nested access.
        populate (bool): If True, create an empty dictionary for missing keys.

    Returns:
        Any: Value corresponding to the resolved key.
    """
    if k in dct:
        return dct[k]
    from vectorbtpro.utils.search_ import resolve_pathlike_key

    k = resolve_pathlike_key(k)
    if len(k) == 1:
        k = k[0]
    else:
        return get_dict_item(get_dict_item(dct, k[0], populate=populate), k[1:], populate=populate)
    if k not in dct and populate:
        dct[k] = dict()
    return dct[k]


def set_dict_item(dct: dict, k: tp.Any, v: tp.Any, force: bool = False) -> None:
    """Set an item in a dictionary with an optional force flag.

    Args:
        dct (dict): Dictionary to update.
        k (Any): Key to set.
        v (Any): Value to assign.
        force (bool): If True, and if `dct` is an instance of `Config`, override blocking flags.

    Returns:
        None: Dictionary is updated in place.
    """
    if isinstance(dct, Config):
        dct.__setitem__(k, v, force=force)
    else:
        dct[k] = v


def del_dict_item(dct: dict, k: tp.Any, force: bool = False) -> None:
    """Delete an item from a dictionary with an optional force flag.

    Args:
        dct (dict): Dictionary from which to delete the item.
        k (Any): Key of the item to delete.
        force (bool): If True, and if `dct` is an instance of `Config`, override blocking flags.

    Returns:
        None: Dictionary is updated in place.
    """
    if isinstance(dct, Config):
        dct.__delitem__(k, force=force)
    else:
        del dct[k]


def copy_dict(dct: tp.DictLike, copy_mode: str = "shallow", nested: bool = True) -> dict:
    """Copy a dictionary based on the specified copy mode.

    Args:
        dct (DictLike): Input configuration dictionary.
        copy_mode (str): Copying mode.

            Supported modes are:

            * 'none': No copy is performed.
            * 'shallow': Only the dictionary structure is copied.
            * 'hybrid': Keys are copied and values are shallow copied using `copy.copy`.
            * 'deep': Deep copy is performed using `copy.deepcopy`.
        nested (bool): If True, recursively copy nested dictionaries.

    Returns:
        dict: Copy of the input dictionary according to the specified mode.
    """
    if dct is None:
        return {}
    copy_mode = copy_mode.lower()
    if copy_mode not in {"none", "shallow", "hybrid", "deep"}:
        raise ValueError(f"Copy mode '{copy_mode}' is not supported")

    if copy_mode == "none":
        return dct
    if copy_mode == "deep":
        return deepcopy(dct)
    if isinstance(dct, Config):
        return dct.copy(copy_mode=copy_mode, nested=nested)
    dct_copy = copy(dct)  # copy structure using shallow copy
    for k, v in dct_copy.items():
        if nested and isinstance(v, dict):
            _v = copy_dict(v, copy_mode=copy_mode, nested=nested)
        else:
            if copy_mode == "hybrid":
                _v = copy(v)  # copy values using shallow copy
            else:
                _v = v
        set_dict_item(dct_copy, k, _v, force=True)
    return dct_copy


def update_dict(
    x: tp.DictLike,
    y: tp.DictLike,
    nested: bool = True,
    force: bool = False,
    same_keys: bool = False,
) -> None:
    """Update a dictionary with keys and values from another dictionary.

    Args:
        x (DictLike): Target dictionary to update.
        y (DictLike): Source dictionary with values to update.
        nested (bool): If True, recursively update nested dictionaries.
        force (bool): If True, override blocking flags when updating.
        same_keys (bool): If True, only update keys that already exist in the target dictionary.

    Returns:
        None: Dictionary is updated in place.

    !!! note
        When updating a nested configuration (an instance of `Configured`), if the corresponding
        value in `y` is a dictionary, the method `Configured.replace` is invoked. Additionally,
        if the child dictionary is not atomic, only its values are copied, not its metadata.
    """
    if x is None:
        return
    if y is None:
        return
    assert_instance_of(x, dict)
    assert_instance_of(y, dict)

    for k, v in y.items():
        if (
            nested
            and k in x
            and isinstance(x[k], (dict, Configured))
            and isinstance(v, dict)
            and not isinstance(v, atomic_dict)
        ):
            if isinstance(x[k], Configured):
                set_dict_item(x, k, x[k].replace(**v), force=force)
            else:
                update_dict(x[k], v, force=force)
        else:
            if same_keys and k not in x:
                continue
            set_dict_item(x, k, v, force=force)


def reorder_dict(
    dct: dict, keys: tp.Iterable[tp.Union[tp.Hashable, type(...)]], skip_missing: bool = False
) -> dict:
    """Reorder a dictionary based on a list of keys.

    Args:
        dct (dict): Dictionary to reorder.
        keys (Iterable[Union[Hashable, Ellipsis]]): List of keys specifying the new order.

            A single Ellipsis (`...`) can be used to indicate remaining keys.
        skip_missing (bool): If True, ignore keys not present in `dct`.

    Returns:
        dict: New dictionary with items reordered.
    """
    if not isinstance(dct, dict):
        dct = dict(dct)
    if not isinstance(keys, list):
        keys = list(keys)
    ellipsis_count = keys.count(...)
    if ellipsis_count > 1:
        raise ValueError("Keys list can contain at most one Ellipsis")
    if skip_missing:
        specified_keys = [k for k in keys if k is not Ellipsis and k in dct]
    else:
        specified_keys = [k for k in keys if k is not Ellipsis]
        missing_keys = [k for k in specified_keys if k not in dct]
        if missing_keys:
            raise KeyError(f"Keys not found in dictionary: {missing_keys}")
    remaining_keys = [k for k in dct if k not in specified_keys]
    final_order = []
    for key in keys:
        if key is Ellipsis:
            final_order.extend(remaining_keys)
        elif key in dct:
            final_order.append(key)
    if ellipsis_count == 0 and not skip_missing or ellipsis_count == 0 and skip_missing:
        final_order.extend(k for k in dct if k not in final_order)
    return {k: dct[k] for k in final_order}


def reorder_list(
    lst: list, keys: tp.Iterable[tp.Union[int, type(...)]], skip_missing: bool = False
) -> list:
    """Reorder a list based on a list of integer indices.

    Args:
        lst (list): List to reorder.
        keys (Iterable[Union[int, Ellipsis]]): List of indices specifying the new order.

            A single Ellipsis (`...`) can be used to indicate positions for any remaining elements.
        skip_missing (bool): If True, ignore indices not present in `lst`.

    Returns:
        list: New list with elements reordered.
    """
    if not isinstance(lst, list):
        lst = list(lst)
    if not isinstance(keys, list):
        keys = list(keys)
    ellipsis_count = keys.count(...)
    if ellipsis_count > 1:
        raise ValueError("Keys list can contain at most one Ellipsis (...)")
    specified_keys = [k for k in keys if k is not Ellipsis]
    if not all(isinstance(k, int) for k in specified_keys):
        raise TypeError("All keys must be integers or Ellipsis (...)")
    if skip_missing:
        seen = set()
        valid_specified = []
        for k in specified_keys:
            if 0 <= k < len(lst) and k not in seen:
                valid_specified.append(k)
                seen.add(k)
        specified_keys = valid_specified
    else:
        invalid_keys = [k for k in specified_keys if not (0 <= k < len(lst))]
        if invalid_keys:
            raise IndexError(f"Indices out of range: {invalid_keys}")
        if len(specified_keys) != len(set(specified_keys)):
            duplicates = set(k for k in specified_keys if specified_keys.count(k) > 1)
            raise ValueError(f"Duplicate indices in keys list: {duplicates}")
    remaining_indices = [i for i in range(len(lst)) if i not in specified_keys]
    final_order = []
    for key in keys:
        if key is Ellipsis:
            final_order.extend(remaining_indices)
        else:
            if skip_missing:
                if key in specified_keys:
                    final_order.append(key)
            else:
                final_order.append(key)
    if not skip_missing:
        if ... not in keys:
            if len(final_order) != len(lst):
                raise ValueError(
                    "Reordered list does not include all elements from the original list"
                )
        if set(final_order) != set(range(len(lst))):
            missing = set(range(len(lst))) - set(final_order)
            extra = set(final_order) - set(range(len(lst)))
            if missing:
                raise ValueError(f"Missing indices in reordered list: {missing}")
            if extra:
                raise ValueError(f"Invalid indices in reordered list: {extra}")
    return [lst[i] for i in final_order]


class _unsetkey:
    """Sentinel class for unsetting dictionary keys."""


unsetkey = _unsetkey()
"""Sentinel value indicating that a key should be removed.

It can still be overridden by another dictionary.
"""


def unset_keys(
    dct: tp.DictLike,
    nested: bool = True,
    force: bool = False,
) -> None:
    """Unset keys in a dictionary that have the value `unsetkey`.

    Args:
        dct (DictLike): Dictionary in which keys may be unset.
        nested (bool): If True, unset keys in nested dictionaries recursively.
        force (bool): If True, force the removal of keys.

    Returns:
        None: Dictionary is updated in place.
    """
    if dct is None:
        return
    assert_instance_of(dct, dict)

    for k, v in list(dct.items()):
        if isinstance(v, _unsetkey):
            del_dict_item(dct, k, force=force)
        elif nested and isinstance(v, dict) and not isinstance(v, atomic_dict):
            unset_keys(v, nested=nested, force=force)


def merge_dicts(
    *dicts: tp.DictLike,
    to_dict: bool = True,
    copy_mode: str = "shallow",
    nested: tp.Optional[bool] = None,
    same_keys: bool = False,
) -> dict:
    """Merge multiple dictionaries into one.

    Merge provided dictionaries with optional conversion and copying, and optionally perform
    recursive merging of nested dictionaries.

    Args:
        *dicts (DictLike): Dictionaries to merge.
        to_dict (bool): Whether to convert each dictionary using `convert_to_dict` before merging.
        copy_mode (str): Copying mode.

            See `copy_dict`.
        nested (Optional[bool]): Whether to recursively merge nested dictionaries.

            If None, the function determines automatically if any dictionary is nested.
        same_keys (bool): Whether to merge only overlapping keys.

    Returns:
        dict: Merged dictionary.
    """
    if len(dicts) == 1:
        dicts = (None, dicts[0])

    if dicts[0] is None and dicts[1] is None:
        if len(dicts) > 2:
            return merge_dicts(
                None,
                *dicts[2:],
                to_dict=to_dict,
                copy_mode=copy_mode,
                nested=nested,
                same_keys=same_keys,
            )
        return {}

    if nested is None:
        for dct in dicts:
            if dct is not None:
                for v in dct.values():
                    if isinstance(v, dict) and not isinstance(v, atomic_dict):
                        nested = True
                        break
            if nested:
                break

    if to_dict:
        if not nested and not same_keys and copy_mode in {"none", "shallow"}:
            out = {}
            for dct in dicts:
                if dct is not None:
                    out.update(dct)
            for k, v in list(out.items()):
                if isinstance(v, _unsetkey):
                    del out[k]
            return out
        dicts = tuple([convert_to_dict(dct, nested=True) for dct in dicts])

    if not to_dict or copy_mode not in {"none", "shallow"}:
        dicts = tuple([copy_dict(dct, copy_mode=copy_mode, nested=nested) for dct in dicts])

    x, y = dicts[0], dicts[1]
    should_update = True
    if type(x) is dict and type(y) is dict and len(x) == 0:
        x = y
        should_update = False
    if isinstance(x, atomic_dict) or isinstance(y, atomic_dict):
        x = y
        should_update = False
    if should_update:
        update_dict(x, y, nested=nested, force=True, same_keys=same_keys)

    unset_keys(x, nested=nested, force=True)

    if len(dicts) > 2:
        return merge_dicts(
            x,
            *dicts[2:],
            to_dict=False,  # executed only once
            copy_mode="none",  # executed only once
            nested=nested,
            same_keys=same_keys,
        )
    return x


def flat_merge_dicts(*dicts: tp.DictLike, **kwargs) -> dict:
    """Merge multiple dictionaries with flat (non-recursive) merging.

    Wrapper around `merge_dicts` that forces `nested` to False while applying default arguments.

    Args:
        *dicts (DictLike): Dictionaries to merge.
        **kwargs: Keyword arguments for `merge_dicts`.

    Returns:
        dict: Merged dictionary.
    """
    return merge_dicts(*dicts, nested=False, **kwargs)


class child_dict(pdict):
    """Child dictionary class.

    Subclass of `dict` representing a nested child dictionary.
    """

    pass


ConfigT = tp.TypeVar("ConfigT", bound="Config")


class Config(pdict):
    """Configuration class that extends a pickleable dictionary with enhanced configuration features
    including nested updates, freezing, and resetting.

    Args:
        *args: Positional arguments for `dict`.
        options_ (KwargsLike): Configuration options.

            See details below.
        **kwargs: Keyword arguments for `dict`.

    Options:
        copy_kwargs (dict): Keyword arguments used by `copy_dict` when copying the main
            dictionary and `reset_dct`.

            Copy mode defaults to 'none'.
        reset_dct (dict): Fallback dictionary used for resetting.

            If None, the main dictionary is copied using `reset_dct_copy_kwargs`.

            Defaults to None.

            !!! note
                When `readonly` is True, defaults to the main dictionary if set to None.
        reset_dct_copy_kwargs (dict): Keyword arguments that override those in `copy_kwargs`
            for creating `reset_dct`.

            Copy mode defaults to 'none' if `readonly` is True, otherwise 'hybrid'.
        pickle_reset_dct (bool): Determines whether `reset_dct` is pickled.

            Defaults to False.
        frozen_keys (bool): Denies updates to configuration keys when True.

            Defaults to False.
        readonly (bool): Denies updates to both keys and values when True.

            Defaults to False.
        nested (bool): Applies operations such as copying, updating, and merging recursively to
            each child dictionary when True.

            Disable this to treat each child dictionary as a single value.

            Defaults to True.
        convert_children (Union[bool, type]): Converts child dictionaries of type `child_dict` to
            configurations with the same settings if True or if set to a `Config` subclass.

            This triggers a waterfall conversion across all child dictionaries.
            Existing configurations are not converted. Requires `nested` to be True.

            Defaults to False.
        as_attrs (bool): Enables accessing dictionary keys via dot notation when True.

            This provides autocompletion at runtime but raises an error in the event of naming conflicts.
            To allow nested dictionaries to be accessed via dot notation, wrap them with `child_dict`
            and ensure both `convert_children` and `nested` are True.

            Defaults to True if `frozen_keys` or `readonly` is True, otherwise False.
        override_keys (Set[str]): Specifies keys that can override attribute names when `as_attrs` is True.

    !!! info
        For default settings, see `options` in `vectorbtpro._settings.config`.

        If another configuration is provided, its properties will be copied, but they can be overridden
        by arguments passed during initialization.

    !!! note
        All arguments are applied only once during initialization.
    """

    def __init__(self, *args, options_: tp.KwargsLike = None, **kwargs) -> None:
        try:
            from vectorbtpro._settings import settings

            options_cfg = settings["config"]["options"]
        except ImportError:
            options_cfg = {}

        if len(args) > 0 and isinstance(args[0], Config):
            cfg = args[0]
        else:
            cfg = None
        dct = dict(*args, **kwargs)
        if "options_" in dct:
            raise ValueError("options_ is an argument reserved for configs")
        if options_ is None:
            options_ = dict()
        else:
            options_ = dict(options_)

        def _resolve_setting(pname: str, default: tp.Any, merge: bool = False) -> tp.Any:
            cfg_default = options_cfg.get(pname)
            if cfg is None:
                dct_p = None
            else:
                dct_p = cfg.get_option(pname)
            option = options_.pop(pname, None)

            if merge and isinstance(default, dict):
                return merge_dicts(default, cfg_default, dct_p, option)
            if option is not None:
                return option
            if dct_p is not None:
                return dct_p
            if cfg_default is not None:
                return cfg_default
            return default

        options_["reset_dct_copy_kwargs"] = merge_dicts(
            options_.get("copy_kwargs"),
            options_.get("reset_dct_copy_kwargs"),
        )
        reset_dct = _resolve_setting("reset_dct", None)
        pickle_reset_dct = _resolve_setting("pickle_reset_dct", False)
        frozen_keys = _resolve_setting("frozen_keys", False)
        readonly = _resolve_setting("readonly", False)
        nested = _resolve_setting("nested", True)
        convert_children = _resolve_setting("convert_children", False)
        as_attrs = _resolve_setting("as_attrs", frozen_keys or readonly)
        override_keys = _resolve_setting("override_keys", set())
        copy_kwargs = _resolve_setting(
            "copy_kwargs",
            dict(copy_mode="none", nested=nested),
            merge=True,
        )
        reset_dct_copy_kwargs = _resolve_setting(
            "reset_dct_copy_kwargs",
            dict(copy_mode="none" if readonly else "hybrid", nested=nested),
            merge=True,
        )
        if len(options_) > 0:
            raise ValueError(f"Unexpected config options: {options_}")

        dct = copy_dict(dict(dct), **copy_kwargs)

        if convert_children and nested:
            for k, v in dct.items():
                if isinstance(v, child_dict):
                    if isinstance(convert_children, bool):
                        config_cls = type(self)
                    elif issubclass(convert_children, Config):
                        config_cls = convert_children
                    else:
                        raise TypeError(
                            "Option 'convert_children' must be either boolean or a subclass of Config"
                        )
                    dct[k] = config_cls(
                        v,
                        options_=dict(
                            copy_kwargs=copy_kwargs,
                            reset_dct=None,
                            reset_dct_copy_kwargs=reset_dct_copy_kwargs,
                            pickle_reset_dct=pickle_reset_dct,
                            frozen_keys=frozen_keys,
                            readonly=readonly,
                            nested=nested,
                            convert_children=convert_children,
                            as_attrs=as_attrs,
                            override_keys=override_keys,
                        ),
                    )

        if reset_dct is None:
            reset_dct = dct
        reset_dct = copy_dict(dict(reset_dct), **reset_dct_copy_kwargs)

        dict.__init__(self, dct)

        self._options_ = dict(
            copy_kwargs=copy_kwargs,
            reset_dct=reset_dct,
            reset_dct_copy_kwargs=reset_dct_copy_kwargs,
            pickle_reset_dct=pickle_reset_dct,
            frozen_keys=frozen_keys,
            readonly=readonly,
            nested=nested,
            convert_children=convert_children,
            as_attrs=as_attrs,
            override_keys=override_keys,
        )

        if as_attrs:
            self_dir = set(self.__dir__())
            for k, v in dct.items():
                if k in self_dir and (
                    k not in override_keys or (k.startswith("__") and k.endswith("__"))
                ):
                    raise ValueError(
                        f"Key '{k}' shadows an attribute of the config. "
                        f"Disable option 'as_attrs' or put the key to 'override_keys'."
                    )

    @property
    def options_(self) -> dict:
        """Configuration options dictionary.

        This dictionary contains various settings that control the behavior of the configuration.

        Returns:
            dict: Configuration options dictionary.
        """
        return self._options_

    def get_option(self, k: str) -> tp.Any:
        """Return the configuration option associated with the provided key.

        Args:
            k (str): Key of the option.

        Returns:
            Any: Value of the configuration option.
        """
        return self._options_[k]

    def set_option(self, k: str, v: tp.Any) -> None:
        """Set the configuration option associated with the provided key.

        Args:
            k (str): Key of the option.
            v (Any): Value to set for the option.

        Returns:
            None: Option is set in place.
        """
        self._options_[k] = v

    def __getattribute__(self, k: str) -> tp.Any:
        if k.startswith("__") and k.endswith("__"):
            return object.__getattribute__(self, k)
        as_attrs = object.__getattribute__(self, "_options_")["as_attrs"]  # error -> __getattr__
        if as_attrs:
            try:
                return self.__getitem__(k)  # error -> __getattr__
            except KeyError:
                raise AttributeError
        return object.__getattribute__(self, k)

    def __getattr__(self, k: str) -> tp.Any:
        return object.__getattribute__(self, k)

    def __setattr__(self, k: str, v: tp.Any, force: bool = False) -> None:
        try:
            as_attrs = object.__getattribute__(self, "_options_")["as_attrs"]
        except AttributeError:
            return object.__setattr__(self, k, v)
        if as_attrs:
            return self.__setitem__(k, v, force=force)
        return object.__setattr__(self, k, v)

    def __delattr__(self, k: str, force: bool = False) -> None:
        try:
            as_attrs = object.__getattribute__(self, "_options_")["as_attrs"]
        except AttributeError:
            return object.__delattr__(self, k)
        if as_attrs:
            return self.__delitem__(k, force=force)
        return object.__delattr__(self, k)

    def __setitem__(self, k: str, v: tp.Any, force: bool = False) -> None:
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            if k not in self:
                raise KeyError(f"Config keys are frozen: key '{k}' not found")
        dict.__setitem__(self, k, v)

    def __delitem__(self, k: str, force: bool = False) -> None:
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            raise KeyError("Config keys are frozen")
        dict.__delitem__(self, k)

    def pop(self, k: str, v: tp.Any = MISSING, force: bool = False) -> tp.Any:
        """Remove and return the key-value pair associated with a specified key.

        Args:
            k (str): Key of the item to remove.
            v (Any): Default value if the key is not found.
            force (bool): Bypass configuration restrictions if True.

        Returns:
            Any: Removed value.
        """
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            raise KeyError("Config keys are frozen")
        if v is MISSING:
            result = dict.pop(self, k)
        else:
            result = dict.pop(self, k, v)
        return result

    def popitem(self, force: bool = False) -> tp.Tuple[str, tp.Any]:
        """Remove and return an arbitrary key-value pair from the config.

        Args:
            force (bool): Bypass configuration restrictions if True.

        Returns:
            Tuple[str, Any]: Removed key-value pair.
        """
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            raise KeyError("Config keys are frozen")
        result = dict.popitem(self)
        return result

    def clear(self, force: bool = False) -> None:
        """Remove all items from the config.

        Args:
            force (bool): Bypass configuration restrictions if True.

        Returns:
            None: Config is cleared in place.
        """
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            raise KeyError("Config keys are frozen")
        dict.clear(self)

    def update(
        self, *args, nested: tp.Optional[bool] = None, force: bool = False, **kwargs
    ) -> None:
        """Update the config with the provided key-value pairs using `update_dict`.

        Args:
            *args: Positional arguments for `dict`.
            nested (Optional[bool]): Whether to perform a nested update.
            force (bool): Bypass configuration restrictions if True.
            **kwargs: Keyword arguments for `dict`.

        Returns:
            None: Config is updated in place.
        """
        other = dict(*args, **kwargs)
        if nested is None:
            nested = self.get_option("nested")
        update_dict(self, other, nested=nested, force=force)

    def __copy__(self: ConfigT) -> ConfigT:
        """Make a shallow copy of the config instance.

        This copy does not apply custom copy settings.

        Returns:
            Config: Shallow copied config instance.
        """
        cls = type(self)
        self_copy = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k not in self_copy:  # otherwise copies dict keys twice
                self_copy.__dict__[k] = copy(v)
        self_copy.clear(force=True)
        self_copy.update(copy(dict(self)), nested=False, force=True)
        return self_copy

    def __deepcopy__(self: ConfigT, memo: tp.DictLike = None) -> ConfigT:
        """Make a deep copy of the config instance.

        This copy does not incorporate custom copy settings.

        Args:
            memo (DictLike): Memo dictionary for tracking copied objects.

        Returns:
            Config: Deep copied config instance.
        """
        if memo is None:
            memo = {}
        cls = type(self)
        self_copy = cls.__new__(cls)
        memo[id(self)] = self_copy
        for k, v in self.__dict__.items():
            if k not in self_copy:  # otherwise copies dict keys twice
                self_copy.__dict__[k] = deepcopy(v, memo)
        self_copy.clear(force=True)
        self_copy.update(deepcopy(dict(self), memo), nested=False, force=True)
        return self_copy

    def copy(
        self: ConfigT,
        reset_dct_copy_kwargs: tp.KwargsLike = None,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
    ) -> ConfigT:
        """Create a copy of the config instance.

        By default, the copy is performed using the initialization copy settings.

        Args:
            reset_dct_copy_kwargs (KwargsLike): Additional parameters for copying the reset dictionary.
            copy_mode (Optional[str]): Copying mode.

                See `copy_dict`.
            nested (Optional[bool]): Whether to perform a nested copy.

        Returns:
            Config: Copied config instance.
        """
        if copy_mode is None:
            copy_mode = self.get_option("copy_kwargs")["copy_mode"]
            reset_dct_copy_mode = self.get_option("reset_dct_copy_kwargs")["copy_mode"]
        else:
            reset_dct_copy_mode = copy_mode
        if nested is None:
            nested = self.get_option("copy_kwargs")["nested"]
            reset_dct_nested = self.get_option("reset_dct_copy_kwargs")["nested"]
        else:
            reset_dct_nested = nested
        reset_dct_copy_kwargs = resolve_dict(reset_dct_copy_kwargs)
        if "copy_mode" in reset_dct_copy_kwargs:
            if reset_dct_copy_kwargs["copy_mode"] is not None:
                reset_dct_copy_mode = reset_dct_copy_kwargs["copy_mode"]
        if "nested" in reset_dct_copy_kwargs:
            if reset_dct_copy_kwargs["nested"] is not None:
                reset_dct_nested = reset_dct_copy_kwargs["nested"]

        self_copy = self.__copy__()
        reset_dct = copy_dict(
            dict(self_copy.get_option("reset_dct")),
            copy_mode=reset_dct_copy_mode,
            nested=reset_dct_nested,
        )
        self_copy.set_option("reset_dct", reset_dct)
        dct = copy_dict(dict(self_copy), copy_mode=copy_mode, nested=nested)
        self_copy.update(dct, nested=False, force=True)
        return self_copy

    def merge_with(
        self: ConfigT,
        other: tp.DictLike,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """Merge the current config with another dictionary, combining entries into one dictionary.

        Args:
            other (DictLike): Dictionary to merge.
            copy_mode (Optional[str]): Copying mode.

                See `copy_dict`.
            nested (Optional[bool]): Whether to perform a nested merge.
            **kwargs: Keyword arguments for `merge_dicts`.

        Returns:
            dict: Merged dictionary.
        """
        if copy_mode is None:
            copy_mode = "shallow"
        if nested is None:
            nested = self.get_option("nested")
        return merge_dicts(self, other, copy_mode=copy_mode, nested=nested, **kwargs)

    def to_dict(self, nested: tp.Optional[bool] = None) -> dict:
        """Convert the config instance to a Python dictionary.

        Args:
            nested (Optional[bool]): Whether to apply nested conversion.

        Returns:
            dict: Configuration dictionary.
        """
        return convert_to_dict(self, nested=nested)

    def reset(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """Clear the config and restore it to its initial state.

        Args:
            force (bool): Bypass configuration restrictions if True.
            **reset_dct_copy_kwargs: Keyword arguments for copying the reset dictionary.

        Returns:
            None: Config is reset in place.
        """
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(
            self.get_option("reset_dct_copy_kwargs"), reset_dct_copy_kwargs
        )
        reset_dct = copy_dict(dict(self.get_option("reset_dct")), **reset_dct_copy_kwargs)
        self.clear(force=True)
        self.update(self.get_option("reset_dct"), nested=False, force=True)
        self.set_option("reset_dct", reset_dct)

    def make_checkpoint(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """Update the reset dictionary to reflect the current config state.

        Args:
            force (bool): Bypass configuration restrictions if True.
            **reset_dct_copy_kwargs: Keyword arguments for copying the reset dictionary.

        Returns:
            None: Reset dictionary is updated in place.
        """
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(
            self.get_option("reset_dct_copy_kwargs"), reset_dct_copy_kwargs
        )
        reset_dct = copy_dict(dict(self), **reset_dct_copy_kwargs)
        self.set_option("reset_dct", reset_dct)

    def load_update(
        self,
        path: tp.Optional[tp.PathLike] = None,
        clear: bool = False,
        update_options: bool = False,
        nested: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Load configuration data from a file and update the instance in-place.

        Args:
            path (Optional[PathLike]): File path to load the configuration.
            clear (bool): Clear the current config before updating if True.
            update_options (bool): Update configuration options if True.
            nested (Optional[bool]): Whether to apply a nested update.
            **kwargs: Keyword arguments for `Config.load`.

        Returns:
            None: Config is updated in place.
        """
        loaded = self.load(path=path, **kwargs)
        if clear:
            self.clear(force=True)
            if update_options:
                self.__dict__.clear()
        if nested is None:
            nested = self.get_option("nested")
        self.update(loaded, nested=nested, force=True)
        if update_options:
            self.__dict__.update(loaded.__dict__)

    def prettify(
        self,
        with_options: bool = False,
        replace: tp.DictLike = None,
        path: tp.PathLikeKey = None,
        htchar: str = "    ",
        lfchar: str = "\n",
        indent: int = 0,
        indent_head: bool = True,
        repr_: tp.Optional[tp.Callable] = None,
    ) -> str:
        """Prettify the configuration.

        Args:
            with_options (bool): Whether to include options in the prettified output.
            replace (DictLike): Mapping for value replacement.
            path (str): Current path in the object hierarchy.
            htchar (str): String used for horizontal indentation.
            lfchar (str): Line feed character.
            indent (int): Current indentation level.
            indent_head (bool): Whether to indent the head line.
            repr_ (Optional[Callable]): Function to get the representation of an object.

                Defaults to `repr`.

        Returns:
            str: Prettified string representation of the configuration.
        """
        dct = dict(self)
        if with_options:
            dct["options_"] = self.options_
        if all([isinstance(k, str) and k.isidentifier() for k in dct]):
            return prettify_inited(
                type(self),
                dct,
                replace=replace,
                path=path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent,
                indent_head=indent_head,
                repr_=repr_,
            )
        return prettify_dict(
            self,
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            indent_head=indent_head,
            repr_=repr_,
        )

    def equals(
        self,
        other: tp.Any,
        check_types: bool = True,
        check_options: bool = False,
        _key: tp.Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Check if the current configuration equals another configuration.

        Args:
            other (Any): Configuration to compare against.
            check_types (bool): Whether to verify types during comparison.
            check_options (bool): Whether to compare configuration options.
            **kwargs: Keyword arguments for `vectorbtpro.utils.checks.is_deep_equal`.

        Returns:
            bool: True if the configurations are equal, False otherwise.
        """
        if _key is None:
            _key = type(self).__name__
        if "only_types" in kwargs:
            del kwargs["only_types"]
        if check_types and not is_deep_equal(
            self,
            other,
            _key=_key,
            only_types=True,
            **kwargs,
        ):
            return False
        if check_options and not is_deep_equal(
            self.options_,
            other.options_,
            _key=_key + ".options_",
            **kwargs,
        ):
            return False
        return is_deep_equal(
            dict(self),
            dict(other),
            _key=_key,
            **kwargs,
        )

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        init_kwargs = dict(self)
        init_kwargs["options_"] = dict(self.options_)
        if not self.get_option("pickle_reset_dct"):
            init_kwargs["options_"]["reset_dct"] = None
        return RecState(init_kwargs=init_kwargs)


class AtomicConfig(Config, atomic_dict):
    """Configuration class that behaves like a single value during merge operations."""

    pass


class FrozenConfig(Config):
    """Configuration class with the `frozen_keys` flag enabled.

    Args:
        *args: Positional arguments for `Config`.
        **kwargs: Keyword arguments for `Config`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        options_["frozen_keys"] = True
        Config.__init__(self, *args, options_=options_, **kwargs)


class ReadonlyConfig(Config):
    """Configuration class with the `readonly` flag enabled.

    Args:
        *args: Positional arguments for `Config`.
        **kwargs: Keyword arguments for `Config`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        options_["readonly"] = True
        Config.__init__(self, *args, options_=options_, **kwargs)


class HybridConfig(Config):
    """Configuration class with `copy_kwargs` configured to use `copy_mode='hybrid'`.

    Args:
        *args: Positional arguments for `Config`.
        **kwargs: Keyword arguments for `Config`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        copy_kwargs = options_.pop("copy_kwargs", None)
        if copy_kwargs is None:
            copy_kwargs = {}
        copy_kwargs["copy_mode"] = "hybrid"
        options_["copy_kwargs"] = copy_kwargs
        Config.__init__(self, *args, options_=options_, **kwargs)


ConfiguredT = tp.TypeVar("ConfiguredT", bound="Configured")


class SettingsNotFoundError(KeyError):
    """Exception raised when settings are not found."""

    pass


class SettingNotFoundError(KeyError):
    """Exception raised when a setting is not found."""

    pass


HasSettingsT = tp.TypeVar("HasSettingsT", bound="HasSettings")


ext_settings_paths_config = HybridConfig()
"""_"""

__pdoc__[
    "ext_settings_paths_config"
] = f"""Configuration for currently active extensional settings paths.

Stores tuples of class names and their associated settings paths by unique identifiers.

```python
{ext_settings_paths_config.prettify_doc()}
```
"""


class ExtSettingsPath(Base):
    """Context manager to temporarily add extensional settings paths.

    Args:
        ext_settings_paths (ExtSettingsPaths): Dictionary of extensional settings paths.
    """

    def __init__(self, ext_settings_paths: tp.ExtSettingsPaths) -> None:
        self._unique_id = str(uuid.uuid4())
        self._ext_settings_paths = ext_settings_paths

    @property
    def unique_id(self) -> str:
        """Unique identifier for this extensional settings path instance.

        Returns:
            str: Unique identifier.
        """
        return self._unique_id

    @property
    def ext_settings_paths(self) -> tp.ExtSettingsPaths:
        """Dictionary containing extensional settings paths.

        Returns:
            ExtSettingsPaths: Dictionary of extensional settings paths.
        """
        return self._ext_settings_paths

    def __enter__(self) -> tp.Self:
        ext_settings_paths_config[self.unique_id] = self.ext_settings_paths
        return self

    def __exit__(self, *args) -> None:
        del ext_settings_paths_config[self.unique_id]


spec_settings_paths_config = HybridConfig()
"""_"""

__pdoc__[
    "spec_settings_paths_config"
] = f"""Configuration for currently active specialized settings paths.

Stores dictionaries of settings paths keyed by unique identifiers.

In each dictionary, each key represents a path that may point to one or more other paths.
For instance, a relationship `knowledge` -> `pages` will also consider `pages` settings when
`knowledge` settings are requested.

```python
{spec_settings_paths_config.prettify_doc()}
```
"""


class SpecSettingsPath(Base):
    """Context manager to temporarily add specialized settings paths.

    Args:
        spec_settings_paths (SpecSettingsPaths): Dictionary of specialized settings paths.
    """

    def __init__(self, spec_settings_paths: tp.SpecSettingsPaths) -> None:
        self._unique_id = str(uuid.uuid4())
        self._spec_settings_paths = spec_settings_paths

    @property
    def unique_id(self) -> str:
        """Unique identifier for this specialized settings path instance.

        Returns:
            str: Unique identifier.
        """
        return self._unique_id

    @property
    def spec_settings_paths(self) -> tp.SpecSettingsPaths:
        """Dictionary containing specialized settings paths.

        Returns:
            SpecSettingsPaths: Dictionary of specialized settings paths.
        """
        return self._spec_settings_paths

    def __enter__(self) -> tp.Self:
        spec_settings_paths_config[self.unique_id] = self.spec_settings_paths
        return self

    def __exit__(self, *args) -> None:
        del spec_settings_paths_config[self.unique_id]


class HasSettings(Base):
    """Class for managing settings from `vectorbtpro._settings`."""

    _settings_path: tp.SettingsPath = None
    """Path(s) that locate settings for this class in `vectorbtpro._settings`.

    Must be provided as a single path, a list of paths ordered by specialization,
    or a dictionary mapping path IDs to paths.

    Lookup is performed using `get_dict_item`.
    """

    _specializable: tp.ClassVar[bool] = True
    """Boolean flag indicating if the settings for this class can be specialized."""

    _extendable: tp.ClassVar[bool] = True
    """Boolean flag indicating if the settings for this class can be extended."""

    @classmethod
    def get_path_settings(
        cls,
        path: tp.PathLikeKey,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> dict:
        """Return the settings dictionary located under a specified path.

        Args:
            path (PathLikeKey): Primary settings path.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.

        Returns:
            dict: Settings associated with the given path, optionally merged with sub-path settings.
        """
        from vectorbtpro._settings import settings

        sub_path_settings = None
        if sub_path is not None:
            from vectorbtpro.utils.search_ import combine_pathlike_keys

            sub_path = combine_pathlike_keys(path, sub_path)
            try:
                sub_path_settings = cls.get_path_settings(sub_path)
            except SettingsNotFoundError:
                if sub_path_only:
                    raise SettingsNotFoundError(f"Found no settings under the path '{sub_path}'")
        try:
            path_settings = get_dict_item(settings, path)
        except KeyError:
            raise SettingsNotFoundError(f"Found no settings under the path '{path}'")
        if sub_path_settings is not None:
            return merge_dicts(path_settings, sub_path_settings)
        return path_settings

    @classmethod
    def resolve_settings_paths(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        super_first: bool = True,
        unique_only: bool = True,
    ) -> tp.List[tp.Tuple[tp.Type[HasSettingsT], tp.PathLikeKey]]:
        """Return a list of tuples associating classes with their resolved settings paths from this
        class and its superclasses.

        Args:
            path_id (Optional[Hashable]): Identifier for the settings path.
            inherit (bool): Whether to include settings from superclasses.
            super_first (bool): If True, resolve superclass settings first.
            unique_only (bool): Whether to return only unique settings paths.

        Returns:
            List[Tuple[Type[HasSettings], PathLikeKey]]: List of tuples with classes and their
                resolved settings paths.
        """
        from vectorbtpro.utils.search_ import combine_pathlike_keys, resolve_pathlike_key

        paths = []
        unique_paths = set()

        def _add_path(cls_, path):
            if path not in unique_paths or not unique_only:
                paths.append((cls_, path))
                unique_paths.add(path)

                if cls_._specializable and spec_settings_paths_config:
                    path_ = resolve_pathlike_key(path)
                    for spec_settings_paths in spec_settings_paths_config.values():
                        for from_path, to_path in spec_settings_paths.items():
                            from_path_ = resolve_pathlike_key(from_path)

                            if path_ == from_path_:
                                if not isinstance(to_path, list):
                                    to_path = [to_path]
                                for to_path_ in to_path:
                                    if to_path_ not in unique_paths:
                                        paths.append((cls_, to_path_))
                                        unique_paths.add(to_path_)

                            elif (
                                len(path_) > len(from_path_)
                                and path_[: len(from_path_)] == from_path_
                            ):
                                if not isinstance(to_path, list):
                                    to_path = [to_path]
                                for to_path_ in to_path:
                                    to_path_ = combine_pathlike_keys(
                                        to_path_, path_[len(from_path_) :]
                                    )
                                    if to_path_ not in unique_paths:
                                        paths.append((cls_, to_path_))
                                        unique_paths.add(to_path_)

        def _process_path(cls_, path):
            if path is not None:
                if isinstance(path, dict):
                    if path_id is None:
                        raise ValueError("Must specify path id")
                    if path_id not in path:
                        return
                    path = path[path_id]
                    if path is None:
                        return
                    _add_path(cls_, path)
                elif isinstance(path, list):
                    for p in path:
                        _add_path(cls_, p)
                else:
                    _add_path(cls_, path)

        if inherit:
            classes = cls.__mro__[::-1]
        else:
            classes = [cls]
        for i, cls_ in enumerate(classes):
            if issubclass(cls_, HasSettings):
                _process_path(cls_, cls_._settings_path)
                if cls_._extendable and ext_settings_paths_config:
                    for ext_settings_paths in ext_settings_paths_config.values():
                        for ext_cls, ext_path in ext_settings_paths:
                            if ext_cls is cls_:
                                _process_path(ext_cls, ext_path)
        if not super_first:
            return paths[::-1]
        return paths

    @classmethod
    def get_settings(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> dict:
        """Return the merged settings dictionary associated with this class and its superclasses.

        Args:
            path_id (Optional[Hashable]): Identifier for the settings path.
            inherit (bool): Whether to include settings from superclasses.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.

        Returns:
            dict: Merged settings dictionary.
        """
        paths = cls.resolve_settings_paths(
            path_id=path_id,
            inherit=inherit,
            super_first=True,
        )
        if len(paths) == 0:
            if path_id is not None:
                raise SettingsNotFoundError(
                    f"Found no settings associated with the path id '{path_id}'"
                )
            else:
                raise SettingsNotFoundError(
                    f"Found no settings associated with the class {cls.__name__}"
                )
        setting_dicts = []
        for cls_, path in paths:
            try:
                path_settings = cls_.get_path_settings(
                    path, sub_path=sub_path, sub_path_only=sub_path_only
                )
                setting_dicts.append(path_settings)
            except SettingsNotFoundError:
                pass
        if len(setting_dicts) == 0:
            if path_id is not None:
                raise SettingsNotFoundError(
                    f"Found no settings associated with the path id '{path_id}'"
                )
            else:
                raise SettingsNotFoundError(
                    f"Found no settings associated with the class {cls.__name__}"
                )
        if len(setting_dicts) == 1:
            return setting_dicts[0]
        return merge_dicts(*setting_dicts)

    @classmethod
    def has_path_settings(
        cls,
        path: tp.PathLikeKey,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> bool:
        """Return True if the settings exist under the specified path; otherwise, False.

        Args:
            path (PathLikeKey): Primary settings path.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.

        Returns:
            bool: True if settings exist under the specified path; otherwise, False.
        """
        try:
            cls.get_path_settings(path, sub_path=sub_path, sub_path_only=sub_path_only)
            return True
        except SettingsNotFoundError:
            return False

    @classmethod
    def has_settings(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> bool:
        """Return True if settings exist for this class and its superclasses; otherwise, False.

        Args:
            path_id (Optional[Hashable]): Identifier for the settings path.
            inherit (bool): Whether to include settings from superclasses.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.

        Returns:
            bool: True if settings exist; otherwise, False.
        """
        try:
            cls.get_settings(
                path_id=path_id,
                inherit=inherit,
                sub_path=sub_path,
                sub_path_only=sub_path_only,
            )
            return True
        except SettingsNotFoundError:
            return False

    @classmethod
    def get_path_setting(
        cls,
        path: tp.PathLikeKey,
        key: tp.PathLikeKey,
        default: tp.Any = MISSING,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> tp.Any:
        """Return the value associated with a specified key from the settings located at a given path.

        Args:
            path (PathLikeKey): Primary settings path.
            key (PathLikeKey): Key for which to retrieve the setting value.
            default (Any): Default value to return if the key is not found.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.

        Returns:
            Any: Value corresponding to the specified key, or the default value if not found.
        """
        from vectorbtpro._settings import settings

        if sub_path is not None:
            from vectorbtpro.utils.search_ import combine_pathlike_keys

            sub_path = combine_pathlike_keys(path, sub_path)
            try:
                sub_path_settings = cls.get_path_settings(sub_path)
                try:
                    return get_dict_item(sub_path_settings, key)
                except KeyError:
                    if sub_path_only:
                        raise SettingNotFoundError(
                            f"Found no key '{key}' in the settings under the path '{sub_path}'"
                        )
            except SettingsNotFoundError:
                if sub_path_only:
                    raise SettingsNotFoundError(f"Found no settings under the path '{sub_path}'")
        try:
            path_settings = get_dict_item(settings, path)
        except KeyError:
            raise SettingsNotFoundError(f"Found no settings under the path '{path}'")
        try:
            return get_dict_item(path_settings, key)
        except KeyError:
            if default is MISSING:
                if sub_path is not None:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' in the settings under the paths '{path}' and '{sub_path}'"
                    )
                else:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' in the settings under the path '{path}'"
                    )
        return default

    @classmethod
    def get_setting(
        cls,
        key: tp.PathLikeKey,
        default: tp.Any = MISSING,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
        merge: bool = False,
    ) -> tp.Any:
        """Return the setting value from the settings associated with this class and
        its superclasses (if `inherit` is True).

        Args:
            key (PathLikeKey): Key identifying the setting.
            default (Any): Value to return if the setting is not found.
            path_id (Optional[Hashable]): Identifier for the settings path.
            inherit (bool): Whether to include settings from superclasses.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.
            merge (bool): Whether to merge settings from multiple sources.

        Returns:
            Any: Resolved setting value.
        """
        paths = cls.resolve_settings_paths(
            path_id=path_id,
            inherit=inherit,
            super_first=merge,
        )
        if len(paths) == 0:
            if path_id is not None:
                raise SettingsNotFoundError(
                    f"Found no settings associated with the path id '{path_id}'"
                )
            else:
                raise SettingsNotFoundError(
                    f"Found no settings associated with the class {cls.__name__}"
                )
        merged_setting = None
        found_setting = False
        for cls_, path in paths:
            try:
                setting = cls_.get_path_setting(
                    path, key, sub_path=sub_path, sub_path_only=sub_path_only
                )
                if merge:
                    if setting is None or isinstance(setting, dict):
                        if merged_setting is None or isinstance(merged_setting, dict):
                            merged_setting = merge_dicts(setting, merged_setting)
                else:
                    return setting
                found_setting = True
            except (SettingsNotFoundError, SettingNotFoundError):
                continue
        if found_setting:
            return merged_setting
        if default is MISSING:
            if path_id is not None:
                if sub_path is not None:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' under the settings associated with the path id '{path_id}' "
                        f"and sub-path '{sub_path}'"
                    )
                else:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' under the settings associated with the path id '{path_id}'"
                    )
            else:
                if sub_path is not None:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' under the settings associated with the class {cls.__name__} "
                        f"and sub-path '{sub_path}'"
                    )
                else:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' under the settings associated with the class {cls.__name__}"
                    )
        return default

    @classmethod
    def has_path_setting(
        cls,
        path: tp.PathLikeKey,
        key: tp.PathLikeKey,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> bool:
        """Return whether a setting exists under the specified path.

        Args:
            path (PathLikeKey): Primary settings path.
            key (PathLikeKey): Key identifying the setting.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.

        Returns:
            bool: True if the setting exists; otherwise, False.
        """
        try:
            cls.get_path_setting(path, key, sub_path=sub_path, sub_path_only=sub_path_only)
            return True
        except (SettingsNotFoundError, SettingNotFoundError):
            return False

    @classmethod
    def has_setting(
        cls,
        key: tp.PathLikeKey,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> bool:
        """Return whether the setting exists in the settings for this class and
        its superclasses (if `inherit` is True).

        Args:
            key (PathLikeKey): Key identifying the setting.
            path_id (Optional[Hashable]): Identifier for the settings path.
            inherit (bool): Whether to include settings from superclasses.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.

        Returns:
            bool: True if the setting exists; otherwise, False.
        """
        try:
            cls.get_setting(
                key,
                path_id=path_id,
                inherit=inherit,
                sub_path=sub_path,
                sub_path_only=sub_path_only,
            )
            return True
        except (SettingsNotFoundError, SettingNotFoundError):
            return False

    @classmethod
    def resolve_setting(
        cls,
        value: tp.Optional[tp.Any],
        key: tp.PathLikeKey,
        default: tp.Any = MISSING,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
        merge: bool = False,
    ) -> tp.Any:
        """Return the resolved setting value for the provided key from
        `vectorbtpro._settings` associated with this class.

        Args:
            value (Any): Input value to resolve or override the setting.

                If the provided value is None, fetch the setting.
            key (PathLikeKey): Key identifying the setting.
            default (Any): Value to return if the setting is not found.
            path_id (Optional[Hashable]): Identifier for the settings path.
            inherit (bool): Whether to include settings from superclasses.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            sub_path_only (bool): Whether to consider only the combined sub-path.
            merge (bool): Whether to merge dictionaries if applicable.

        Returns:
            Any: Resolved setting value.
        """
        if merge:
            setting = cls.get_setting(
                key,
                default=default,
                path_id=path_id,
                inherit=inherit,
                sub_path=sub_path,
                sub_path_only=sub_path_only,
                merge=True,
            )
            if setting is None or isinstance(setting, dict):
                if value is None or isinstance(value, dict):
                    return merge_dicts(setting, value)
            return value
        if value is None:
            return cls.get_setting(
                key,
                default=default,
                path_id=path_id,
                inherit=inherit,
                sub_path=sub_path,
                sub_path_only=sub_path_only,
            )
        return value

    @classmethod
    def set_settings(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        populate_: bool = False,
        **kwargs,
    ) -> None:
        """Update the settings in `vectorbtpro._settings` associated with this class.

        Args:
            path_id (Optional[Hashable]): Identifier for the settings path.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.
            populate_ (bool): Indicates if the settings should be populated.

                If the settings do not exist, pass `populate_=True` to initialize them.
            **kwargs: Additional key-value pairs to update the settings.

        Returns:
            None: Settings are updated in place.
        """
        from vectorbtpro._settings import settings

        if isinstance(cls._settings_path, dict):
            if path_id is None:
                raise ValueError("Must specify path id")
            if path_id not in cls._settings_path:
                raise SettingsNotFoundError(
                    f"Found no settings associated with the path id '{path_id}'"
                )
            path = cls._settings_path[path_id]
        elif isinstance(cls._settings_path, list):
            path = cls._settings_path[-1]
        else:
            path = cls._settings_path
        if path is None:
            raise SettingsNotFoundError(
                f"Found no settings associated with the class {cls.__name__}"
            )
        if sub_path is not None:
            from vectorbtpro.utils.search_ import combine_pathlike_keys

            path = combine_pathlike_keys(path, sub_path)
        cls_cfg = get_dict_item(settings, path, populate=populate_)
        for k, v in kwargs.items():
            if populate_:
                cls_cfg[k] = v
            else:
                if k not in cls_cfg:
                    raise SettingNotFoundError(
                        f"Found no key '{k}' in the settings under the path '{path}'"
                    )
                if isinstance(cls_cfg[k], dict) and isinstance(v, dict):
                    cls_cfg[k] = merge_dicts(cls_cfg[k], v)
                else:
                    cls_cfg[k] = v

    @classmethod
    def reset_settings(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
    ) -> None:
        """Reset the settings in `vectorbtpro._settings` associated with this class.

        Args:
            path_id (Optional[Hashable]): Identifier for the settings path.
            sub_path (Optional[PathLikeKey]): Sub-path to extend the settings path.

                The sub-path is appended to the resolved path to give it higher priority.

        Returns:
            None: Settings are reset in place.
        """
        from vectorbtpro._settings import settings

        if isinstance(cls._settings_path, dict):
            if path_id is None:
                raise ValueError("Must specify path id")
            if path_id not in cls._settings_path:
                raise SettingsNotFoundError(
                    f"Found no settings associated with the path id '{path_id}'"
                )
            path = cls._settings_path[path_id]
        elif isinstance(cls._settings_path, list):
            path = cls._settings_path[-1]
        else:
            path = cls._settings_path
        if path is None:
            raise SettingsNotFoundError(
                f"Found no settings associated with the class {cls.__name__}"
            )
        if sub_path is not None:
            from vectorbtpro.utils.search_ import combine_pathlike_keys

            path = combine_pathlike_keys(path, sub_path)
        if not cls.has_path_settings(path):
            raise SettingsNotFoundError(f"Found no settings under the path '{path}'")
        cls_cfg = get_dict_item(settings, path)
        cls_cfg.reset(force=True)


class MetaConfigured(type):
    """Metaclass for `Configured` classes.

    This metaclass automatically configures expected keys based on the `_expected_keys_mode`
    attribute. It aggregates expected keys from inherited classes and from the class's `__init__`
    parameters.

    Args:
        name (str): Name of the class being created.
        bases (Tuple[Type, ...]): Base classes of the class being created.
        attrs (dict): Attribute dictionary defined in the class body.
    """

    def __init__(cls, name: str, bases: tp.Tuple[tp.Type, ...], attrs: dict) -> None:
        super().__init__(name, bases, attrs)

        if hasattr(cls, "_expected_keys_mode"):
            _expected_keys_mode = cls._expected_keys_mode
            if _expected_keys_mode.lower() == "auto":
                _expected_keys = set()
                for base in bases:
                    if hasattr(base, "_expected_keys_mode"):
                        base_expected_keys_mode = base._expected_keys_mode
                        if base_expected_keys_mode.lower() == "disable":
                            _expected_keys = None
                            break
                        if hasattr(base, "_expected_keys"):
                            _expected_keys |= base._expected_keys
                if _expected_keys is None:
                    cls._expected_keys_mode = "disable"
                    cls._expected_keys = None
                else:
                    _expected_keys |= set(get_func_arg_names(cls.__init__)).difference({"self"})
                    cls._expected_keys = _expected_keys
            elif _expected_keys_mode.lower() == "inherit":
                _expected_keys_mode = cls._expected_keys_mode
                if _expected_keys_mode.lower() == "auto":
                    _expected_keys = set()
                    for base in bases:
                        if hasattr(base, "_expected_keys_mode"):
                            base_expected_keys_mode = base._expected_keys_mode
                            if base_expected_keys_mode.lower() == "disable":
                                _expected_keys = None
                                break
                            if hasattr(base, "_expected_keys"):
                                _expected_keys |= base._expected_keys
                    if _expected_keys is None:
                        cls._expected_keys_mode = "disable"
                        cls._expected_keys = None
                    else:
                        cls._expected_keys = _expected_keys
            elif _expected_keys_mode.lower() == "disable":
                cls._expected_keys = None
            elif not _expected_keys_mode.lower() == "custom":
                raise ValueError(f"Invalid expected keys mode: '{_expected_keys_mode}'")


class Configured(
    HasSettings, Cacheable, Comparable, Pickleable, Prettified, Chainable, metaclass=MetaConfigured
):
    """Class representing a configured object.

    All subclasses of `Configured` are initialized using `Config`, which facilitates
    pickling and configuration merging.

    Args:
        **config: Keyword arguments for initialization configuration.

    !!! info
        For default settings, see `vectorbtpro._settings.configured`.

    !!! warning
        If any attribute is overwritten that is not listed in `Configured._writeable_attrs`,
        or if any `Configured` argument depends on global defaults, those values will
        not be copied. Pass them explicitly to ensure the saved, loaded, or copied instance
        remains resilient to changes in globals.
    """

    _expected_keys_mode: tp.ExpectedKeysMode = "auto"
    """Mode of expected keys.

    Accepted values are:

    * "auto": Combines keys from bases and signature. Disabled if any base is disabled.
    * "inherit": Combines keys from bases only. Disabled if any base is disabled.
    * "disable": Disables key checking.
    * "custom": Expects custom provided keys.
    """

    _expected_keys: tp.ExpectedKeys = None
    """Set of expected configuration keys."""

    _writeable_attrs: tp.WriteableAttrs = None
    """Set of writable attribute names to be saved or copied with the configuration."""

    def __init__(self, **config) -> None:
        from vectorbtpro._settings import settings

        configured_cfg = settings["configured"]

        check_expected_keys_ = config.get("check_expected_keys_")
        if self._expected_keys is None:
            check_expected_keys_ = False
        if check_expected_keys_ is None:
            check_expected_keys_ = configured_cfg["check_expected_keys_"]
        if check_expected_keys_:
            if isinstance(check_expected_keys_, bool):
                check_expected_keys_ = "raise"
            keys_diff = list(set(config.keys()).difference(self._expected_keys))
            if len(keys_diff) > 0:
                assert_in(check_expected_keys_, ("warn", "raise"))
                if check_expected_keys_ == "warn":
                    warn(f"{type(self).__name__} doesn't expect arguments {keys_diff}")
                else:
                    raise ValueError(f"{type(self).__name__} doesn't expect arguments {keys_diff}")

        self._config = Config(config, options_=configured_cfg["config"]["options"])

        Cacheable.__init__(self)

    @property
    def config(self) -> Config:
        """Configuration instance set during initialization.

        Returns:
            Config: Configuration instance.
        """
        return self._config

    @hybrid_method
    def get_writeable_attrs(cls_or_self) -> tp.Optional[tp.Set[str]]:
        """Return the writable attribute names for this class and its base classes.

        Returns:
            Optional[Set[str]]: Set of writable attribute names if defined, otherwise None.
        """
        if isinstance(cls_or_self, type):
            cls = cls_or_self
        else:
            cls = type(cls_or_self)
        writeable_attrs = set()
        for cls in inspect.getmro(cls):
            if issubclass(cls, Configured) and cls._writeable_attrs is not None:
                writeable_attrs |= cls._writeable_attrs
        return writeable_attrs

    @classmethod
    def resolve_merge_kwargs(
        cls,
        *configs: tp.MaybeSequence[ConfigT],
        on_merge_conflict: tp.Union[str, dict] = "error",
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing a `Configured` instance after merging configurations.

        Args:
            *configs (MaybeSequence[Config]): Configuration objects to merge.
            on_merge_conflict (Union[str, dict]): Strategy for handling merge conflicts.
            **kwargs: Keyword arguments for initialization.

        Returns:
            Kwargs: Resolved keyword arguments for initialization.
        """
        if len(configs) == 1:
            configs = configs[0]
        configs = list(configs)

        all_keys = set()
        for config in configs:
            all_keys = all_keys.union(set(config.keys()))

        for k in all_keys:
            if k not in kwargs:
                v = None
                for i in range(len(configs)):
                    config = configs[i]
                    if isinstance(on_merge_conflict, dict):
                        if k in on_merge_conflict:
                            _on_merge_conflict = on_merge_conflict[k]
                        elif "_def" in on_merge_conflict:
                            _on_merge_conflict = on_merge_conflict["_def"]
                        else:
                            _on_merge_conflict = "error"
                    else:
                        _on_merge_conflict = on_merge_conflict
                    if _on_merge_conflict.lower() == "error":
                        if i == 0:
                            continue
                        same_k = True
                        try:
                            if k in config:
                                if not is_deep_equal(configs[0][k], config[k]):
                                    same_k = False
                            else:
                                same_k = False
                        except KeyError:
                            same_k = False
                        if not same_k:
                            raise ValueError(
                                f"Objects to be merged must have compatible '{k}'. Pass to override."
                            )
                        else:
                            v = config[k]
                    elif _on_merge_conflict.lower() == "first":
                        if k in config:
                            v = config[k]
                            break
                    elif _on_merge_conflict.lower() == "last":
                        if k in config:
                            v = config[k]
                    else:
                        raise ValueError(f"Invalid on_merge_conflict: '{_on_merge_conflict}'")
                kwargs[k] = v
        return kwargs

    def replace(
        self: ConfiguredT,
        copy_mode_: tp.Optional[str] = None,
        nested_: tp.Optional[bool] = None,
        cls_: tp.Optional[type] = None,
        copy_writeable_attrs_: tp.Optional[bool] = None,
        **new_config,
    ) -> ConfiguredT:
        """Create a new instance with a modified configuration.

        Args:
            copy_mode_ (str): Copy mode for copying the configuration and attributes.
            nested_ (bool): Whether to copy nested objects.
            cls_ (type): Class to instantiate for the new instance.
            copy_writeable_attrs_ (bool): Whether to copy writable attributes.
            **new_config: Additional configuration parameters to update the instance.

        Returns:
            Configured: New instance with the updated configuration.

        !!! warning
            This method returns a new instance initialized with the same configuration and
            writable attributes (or their copies, depending on `copy_mode_`) rather than a direct
            copy of the current instance.
        """
        if cls_ is None:
            cls_ = type(self)
        if copy_writeable_attrs_ is None:
            copy_writeable_attrs_ = cls_ is type(self)
        new_config = self.config.merge_with(new_config, copy_mode=copy_mode_, nested=nested_)
        new_instance = cls_(**new_config)
        if copy_writeable_attrs_:
            for attr in self.get_writeable_attrs():
                attr_obj = getattr(self, attr)
                if isinstance(attr_obj, Config):
                    attr_obj = attr_obj.copy(copy_mode=copy_mode_, nested=nested_)
                else:
                    if copy_mode_ is not None:
                        if copy_mode_ == "hybrid":
                            attr_obj = copy(attr_obj)
                        elif copy_mode_ == "deep":
                            attr_obj = deepcopy(attr_obj)
                setattr(new_instance, attr, attr_obj)
        return new_instance

    def copy(
        self: ConfiguredT,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
        cls: tp.Optional[type] = None,
    ) -> ConfiguredT:
        """Create a new instance by copying the configuration.

        Delegates to `Configured.replace`.

        Args:
            copy_mode (str): Copying mode.

                See `copy_dict`.
            nested (bool): Whether nested objects should be copied.
            cls (type): Class to instantiate for the new instance.

        Returns:
            Configured: New instance with a copied configuration.
        """
        return self.replace(copy_mode_=copy_mode, nested_=nested, cls_=cls)

    def equals(
        self,
        other: tp.Any,
        check_types: bool = True,
        check_attrs: bool = True,
        check_options: bool = False,
        _key: tp.Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Check if the current object equals another object.

        Args:
            other (Any): Object to compare against.
            check_types (bool): Whether to verify types during comparison.
            check_attrs (bool): Whether to compare writable attributes.
            check_options (bool): Whether to compare configuration options.
            **kwargs: Keyword arguments for `vectorbtpro.utils.checks.is_deep_equal` and `Config.equals`.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if _key is None:
            _key = type(self).__name__
        if "only_types" in kwargs:
            del kwargs["only_types"]
        if check_types and not is_deep_equal(
            self,
            other,
            _key=_key,
            only_types=True,
            **kwargs,
        ):
            return False
        if check_attrs:
            if not is_deep_equal(
                self.get_writeable_attrs(),
                other.get_writeable_attrs(),
                _key=_key + ".get_writeable_attrs()",
                **kwargs,
            ):
                return False
            for attr in self.get_writeable_attrs():
                if not is_deep_equal(
                    getattr(self, attr),
                    getattr(other, attr),
                    _key=_key + f".{attr}",
                    **kwargs,
                ):
                    return False
        return self.config.equals(
            other.config,
            check_types=check_types,
            check_options=check_options,
            _key=_key + ".config",
            **kwargs,
        )

    def update_config(self, *args, **kwargs) -> None:
        """Force-update the configuration.

        Args:
            *args: Positional arguments for `Config.update`.
            **kwargs: Keyword arguments for `Config.update`.

        Returns:
            None: Configuration is updated in place.
        """
        self.config.update(*args, **kwargs, force=True)

    def prettify(self, **kwargs) -> str:
        return "%s(%s)" % (
            type(self).__name__,
            self.config.prettify(**kwargs)[len(type(self.config).__name__) + 1 : -1],
        )

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        if self._writeable_attrs is not None:
            attr_dct = {k: getattr(self, k) for k in self._writeable_attrs}
        else:
            attr_dct = {}
        return RecState(init_kwargs=dict(self.config), attr_dct=attr_dct)
