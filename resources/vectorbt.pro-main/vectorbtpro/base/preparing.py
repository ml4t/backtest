# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes for preparing arguments."""

import inspect
import string
from collections import defaultdict
from datetime import time, timedelta
from functools import cached_property as cachedproperty
from pathlib import Path

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.decorators import attach_arg_properties, override_arg_config
from vectorbtpro.base.indexes import repeat_index
from vectorbtpro.base.indexing import IdxRecords, IdxSetter, IdxSetterFactory, index_dict
from vectorbtpro.base.merging import column_stack_arrays, concat_arrays
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.reshaping import BCO, Default, Ref, broadcast
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.config import Config, Configured, HybridConfig, ReadonlyConfig, merge_dicts
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.module_ import import_module_from_path
from vectorbtpro.utils.params import Param
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.path_ import remove_dir
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.source import cut_and_save_func, suggest_module_path
from vectorbtpro.utils.template import CustomTemplate, RepFunc, substitute_templates

__all__ = [
    "BasePreparer",
]

__pdoc__ = {}

base_arg_config = ReadonlyConfig(
    dict(
        broadcast_named_args=dict(is_dict=True),
        broadcast_kwargs=dict(is_dict=True),
        template_context=dict(is_dict=True),
        seed=dict(),
        jitted=dict(),
        chunked=dict(),
        staticized=dict(),
        records=dict(),
    )
)
"""_"""

__pdoc__["base_arg_config"] = f"""Argument configuration for `BasePreparer`.

```python
{base_arg_config.prettify_doc()}
```
"""


class MetaBasePreparer(type(Configured)):
    """Metaclass for `BasePreparer` that provides class-level argument configuration."""

    @property
    def arg_config(cls) -> Config:
        """Class-level argument configuration.

        Returns:
            Config: Class-level argument configuration.
        """
        return cls._arg_config


@attach_arg_properties
@override_arg_config(base_arg_config)
class BasePreparer(Configured, metaclass=MetaBasePreparer):
    """Base class for preparing target functions and arguments.

    Args:
        arg_config (KwargsLike): Optional configuration for target function arguments.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! warning
        Most properties are force-cached - create a new instance to override any attribute.
    """

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _writeable_attrs: tp.WriteableAttrs = {"_arg_config"}

    _settings_path: tp.SettingsPath = None

    _arg_config: tp.ClassVar[Config] = HybridConfig()

    @property
    def arg_config(self) -> Config:
        """Argument configuration of `BasePreparer`.

        ```python
        ${arg_config}
        ```

        Returns:
            Config: Argument configuration of `BasePreparer`.
        """
        return self._arg_config

    def __init__(self, arg_config: tp.KwargsLike = None, **kwargs) -> None:
        Configured.__init__(self, arg_config=arg_config, **kwargs)

        # Copy writeable attrs
        self._arg_config = type(self)._arg_config.copy()
        if arg_config is not None:
            self._arg_config = merge_dicts(self._arg_config, arg_config)

    @classmethod
    def map_enum_value(
        cls, value: tp.ArrayLike, look_for_type: tp.Optional[type] = None, **kwargs
    ) -> tp.ArrayLike:
        """Map enumerated value(s) from the input.

        Args:
            value (ArrayLike): Input value or container of values to map.
            look_for_type (Optional[type]): Type to search for within value to apply mapping.
            **kwargs: Keyword arguments for `vectorbtpro.utils.enum_.map_enum_fields` or
                `BasePreparer.map_enum_value`.

        Returns:
            ArrayLike: Mapped value(s).
        """
        if look_for_type is not None:
            if isinstance(value, look_for_type):
                return map_enum_fields(value, **kwargs)
            return value
        if isinstance(value, (CustomTemplate, Ref)):
            return value
        if isinstance(value, (Param, BCO, Default)):
            attr_dct = value.asdict()
            if isinstance(value, Param) and attr_dct["map_template"] is None:
                attr_dct["map_template"] = RepFunc(
                    lambda values: cls.map_enum_value(values, **kwargs)
                )
            elif not isinstance(value, Param):
                attr_dct["value"] = cls.map_enum_value(attr_dct["value"], **kwargs)
            return type(value)(**attr_dct)
        if isinstance(value, index_dict):
            return index_dict({k: cls.map_enum_value(v, **kwargs) for k, v in value.items()})
        if isinstance(value, IdxSetterFactory):
            value = value.get()
            if not isinstance(value, IdxSetter):
                raise ValueError("Index setter factory must return exactly one index setter")
        if isinstance(value, IdxSetter):
            return IdxSetter([(k, cls.map_enum_value(v, **kwargs)) for k, v in value.idx_items])
        return map_enum_fields(value, **kwargs)

    @classmethod
    def prepare_td_obj(cls, td_obj: object, old_as_keys: bool = True) -> object:
        """Prepare a timedelta-like object for broadcasting.

        Args:
            td_obj (object): Input timedelta-like object, which can be a string,
                timedelta, DateOffset, or Timedelta.
            old_as_keys (bool): Flag indicating whether to use parameter values as keys if not provided.

        Returns:
            object: Processed timedelta object for broadcasting.
        """
        if isinstance(td_obj, Param):
            return td_obj.map_value(cls.prepare_td_obj, old_as_keys=old_as_keys)

        if isinstance(td_obj, (str, timedelta, pd.DateOffset, pd.Timedelta)):
            td_obj = dt.to_timedelta64(td_obj)
        elif isinstance(td_obj, pd.Index):
            td_obj = td_obj.values
        return td_obj

    @classmethod
    def prepare_dt_obj(
        cls,
        dt_obj: object,
        old_as_keys: bool = True,
        last_before: tp.Optional[bool] = None,
    ) -> object:
        """Prepare a datetime-like object for broadcasting.

        Args:
            dt_obj (object): Input datetime-like object, which can be a string, time,
                timedelta, DateOffset, or Timedelta.
            old_as_keys (bool): Flag indicating whether to use parameter values as keys if not provided.
            last_before (Optional[bool]): Flag indicating if the last valid index before the target should be used.

        Returns:
            object: Processed datetime object for broadcasting.
        """
        if isinstance(dt_obj, Param):
            return dt_obj.map_value(cls.prepare_dt_obj, old_as_keys=old_as_keys)

        if isinstance(dt_obj, (str, time, timedelta, pd.DateOffset, pd.Timedelta)):

            def _apply_last_before(source_index, target_index, source_freq):
                resampler = Resampler(source_index, target_index, source_freq=source_freq)
                last_indices = resampler.last_before_target_index(incl_source=False)
                source_rbound_ns = resampler.source_rbound_index.vbt.to_ns()
                return np.where(last_indices != -1, source_rbound_ns[last_indices], -1)

            def _to_dt(wrapper, _dt_obj=dt_obj, _last_before=last_before):
                if _last_before is None:
                    _last_before = False
                _dt_obj = dt.try_align_dt_to_index(_dt_obj, wrapper.index)
                source_index = wrapper.index[wrapper.index < _dt_obj]
                target_index = repeat_index(pd.Index([_dt_obj]), len(source_index))
                if _last_before:
                    target_ns = _apply_last_before(source_index, target_index, wrapper.freq)
                else:
                    target_ns = target_index.vbt.to_ns()
                if len(target_ns) < len(wrapper.index):
                    target_ns = concat_arrays(
                        (target_ns, np.full(len(wrapper.index) - len(target_ns), -1))
                    )
                return target_ns

            def _to_td(wrapper, _dt_obj=dt_obj, _last_before=last_before):
                if _last_before is None:
                    _last_before = True
                target_index = wrapper.index.vbt.to_period_ts(dt.to_freq(_dt_obj), shift=True)
                if _last_before:
                    return _apply_last_before(wrapper.index, target_index, wrapper.freq)
                return target_index.vbt.to_ns()

            def _to_time(wrapper, _dt_obj=dt_obj, _last_before=last_before):
                if _last_before is None:
                    _last_before = False
                index = wrapper.index.tz_localize(None)
                floor_index = index.floor("1d") + dt.time_to_timedelta(_dt_obj)
                target_index = floor_index.where(
                    index < floor_index, floor_index + pd.Timedelta(days=1)
                )
                if wrapper.index.tz is not None:
                    target_index = target_index.tz_localize(wrapper.index.tz)
                if _last_before:
                    return _apply_last_before(wrapper.index, target_index, wrapper.freq)
                return target_index.vbt.to_ns()

            dt_obj_dt_template = RepFunc(_to_dt)
            dt_obj_td_template = RepFunc(_to_td)
            dt_obj_time_template = RepFunc(_to_time)
            if isinstance(dt_obj, str):
                try:
                    time.fromisoformat(dt_obj)
                    dt_obj = dt_obj_time_template
                except Exception:
                    try:
                        dt.to_freq(dt_obj)
                        dt_obj = dt_obj_td_template
                    except Exception:
                        dt_obj = dt_obj_dt_template
            elif isinstance(dt_obj, time):
                dt_obj = dt_obj_time_template
            else:
                dt_obj = dt_obj_td_template
        elif isinstance(dt_obj, pd.Index):
            dt_obj = dt_obj.values
        return dt_obj

    def get_raw_arg_default(self, arg_name: str, is_dict: bool = False) -> tp.Any:
        """Get the raw default value of an argument from settings.

        Args:
            arg_name (str): Name of the argument.
            is_dict (bool): Flag indicating if the argument is expected to be a dictionary.

        Returns:
            Any: Default value for the argument, or an empty dictionary
                if `is_dict` is True and no default is set.
        """
        if self._settings_path is None:
            if is_dict:
                return {}
            return None
        value = self.get_setting(arg_name)
        if is_dict and value is None:
            return {}
        return value

    def get_raw_arg(self, arg_name: str, is_dict: bool = False, has_default: bool = True) -> tp.Any:
        """Retrieve the raw value of an argument from the configuration.

        Args:
            arg_name (str): Name of the argument.
            is_dict (bool): Flag indicating if the argument is expected to be a dictionary.
            has_default (bool): Flag indicating if a default value should be used when the argument is not present.

        Returns:
            Any: Raw value of the argument, merged with defaults if applicable.
        """
        value = self.config.get(arg_name, None)
        if is_dict:
            if has_default:
                return merge_dicts(self.get_raw_arg_default(arg_name), value)
            if value is None:
                return {}
            return value
        if value is None and has_default:
            return self.get_raw_arg_default(arg_name)
        return value

    @cachedproperty
    def idx_setters(self) -> tp.Optional[tp.Dict[tp.Label, IdxSetter]]:
        """Index setters resolved from the `records` argument.

        Returns:
            Optional[Dict[Label, IdxSetter]]: Mapping of record keys to their corresponding
                index setters, or None if records are not provided.
        """
        arg_config = self.arg_config["records"]
        records = self.get_raw_arg(
            "records",
            is_dict=arg_config.get("is_dict", False),
            has_default=arg_config.get("has_default", True),
        )
        if records is None:
            return None
        if not isinstance(records, IdxRecords):
            records = IdxRecords(records)
        idx_setters = records.get()
        for k in idx_setters:
            if k in self.arg_config and not self.arg_config[k].get("broadcast", False):
                raise ValueError(
                    f"Field {k} is not broadcastable and cannot be included in records"
                )
        rename_fields = arg_config.get("rename_fields", {})
        new_idx_setters = {}
        for k, v in idx_setters.items():
            if k in rename_fields:
                k = rename_fields[k]
            new_idx_setters[k] = v
        return new_idx_setters

    def get_arg_default(self, arg_name: str) -> tp.Any:
        """Return the default value for the specified argument based on its configuration.

        Args:
            arg_name (str): Name of the argument.

        Returns:
            Any: Processed default value for the argument.
        """
        arg_config = self.arg_config[arg_name]
        arg = self.get_raw_arg_default(
            arg_name,
            is_dict=arg_config.get("is_dict", False),
        )
        if arg is not None:
            if len(arg_config.get("map_enum_kwargs", {})) > 0:
                arg = self.map_enum_value(arg, **arg_config["map_enum_kwargs"])
            if arg_config.get("is_td", False):
                arg = self.prepare_td_obj(
                    arg,
                    old_as_keys=arg_config.get("old_as_keys", True),
                )
            if arg_config.get("is_dt", False):
                arg = self.prepare_dt_obj(
                    arg,
                    old_as_keys=arg_config.get("old_as_keys", True),
                    last_before=arg_config.get("last_before", None),
                )
        return arg

    def get_arg(
        self, arg_name: str, use_idx_setter: bool = True, use_default: bool = True
    ) -> tp.Any:
        """Return the mapped argument value based on its configuration.

        Args:
            arg_name (str): Name of the argument.
            use_idx_setter (bool): Whether to use the index setter if available.
            use_default (bool): Whether to use the default value from the configuration
                if the argument is missing.

        Returns:
            Any: Processed argument value.
        """
        arg_config = self.arg_config[arg_name]
        if use_idx_setter and self.idx_setters is not None and arg_name in self.idx_setters:
            arg = self.idx_setters[arg_name]
        else:
            arg = self.get_raw_arg(
                arg_name,
                is_dict=arg_config.get("is_dict", False),
                has_default=arg_config.get("has_default", True) if use_default else False,
            )
        if arg is not None:
            if len(arg_config.get("map_enum_kwargs", {})) > 0:
                arg = self.map_enum_value(arg, **arg_config["map_enum_kwargs"])
            if arg_config.get("is_td", False):
                arg = self.prepare_td_obj(arg)
            if arg_config.get("is_dt", False):
                arg = self.prepare_dt_obj(arg, last_before=arg_config.get("last_before", None))
        return arg

    def __getitem__(self, arg_name) -> tp.Any:
        return self.get_arg(arg_name)

    def __iter__(self):
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    @classmethod
    def prepare_td_arr(cls, td_arr: tp.ArrayLike) -> tp.ArrayLike:
        """Return a prepared timedelta array.

        Args:
            td_arr (ArrayLike): Input array of timedelta-like elements.

        Returns:
            ArrayLike: Processed timedelta array.
        """
        if td_arr.dtype == object:
            if td_arr.ndim in (0, 1):
                td_arr = pd.to_timedelta(td_arr)
                if isinstance(td_arr, pd.Timedelta):
                    td_arr = td_arr.to_timedelta64()
                else:
                    td_arr = td_arr.values
            else:
                td_arr_cols = []
                for col in range(td_arr.shape[1]):
                    td_arr_col = pd.to_timedelta(td_arr[:, col])
                    td_arr_cols.append(td_arr_col.values)
                td_arr = column_stack_arrays(td_arr_cols)
        return td_arr

    @classmethod
    def prepare_dt_arr(cls, dt_arr: tp.ArrayLike) -> tp.ArrayLike:
        """Return a prepared datetime array.

        Args:
            dt_arr (ArrayLike): Input array of datetime-like elements.

        Returns:
            ArrayLike: Processed datetime array.
        """
        if dt_arr.dtype == object:
            if dt_arr.ndim in (0, 1):
                dt_arr = pd.to_datetime(dt_arr).tz_localize(None)
                if isinstance(dt_arr, pd.Timestamp):
                    dt_arr = dt_arr.to_datetime64()
                else:
                    dt_arr = dt_arr.values
            else:
                dt_arr_cols = []
                for col in range(dt_arr.shape[1]):
                    dt_arr_col = pd.to_datetime(dt_arr[:, col]).tz_localize(None)
                    dt_arr_cols.append(dt_arr_col.values)
                dt_arr = column_stack_arrays(dt_arr_cols)
        return dt_arr

    @classmethod
    def td_arr_to_ns(cls, td_arr: tp.ArrayLike) -> tp.ArrayLike:
        """Convert a prepared timedelta array to its nanoseconds representation.

        Args:
            td_arr (ArrayLike): Input array of timedelta-like elements.

        Returns:
            ArrayLike: Array of nanoseconds.
        """
        return dt.to_ns(cls.prepare_td_arr(td_arr))

    @classmethod
    def dt_arr_to_ns(cls, dt_arr: tp.ArrayLike) -> tp.ArrayLike:
        """Convert a prepared datetime array to its nanoseconds representation.

        Args:
            dt_arr (ArrayLike): Input array of datetime-like elements.

        Returns:
            ArrayLike: Array of nanoseconds.
        """
        return dt.to_ns(cls.prepare_dt_arr(dt_arr))

    def prepare_post_arg(self, arg_name: str, value: tp.Optional[tp.ArrayLike] = None) -> object:
        """Return the processed argument after broadcasting and template substitution.

        Args:
            arg_name (str): Name of the argument.
            value (Optional[ArrayLike]): Raw value to prepare; if None,
                the default post-argument is used.

        Returns:
            object: Processed argument.
        """
        if value is None:
            if arg_name in self.post_args:
                arg = self.post_args[arg_name]
            else:
                arg = getattr(self, "pre__" + arg_name)
        else:
            arg = value
        if arg is not None:
            arg_config = self.arg_config[arg_name]
            if arg_config.get("substitute_templates", False):
                arg = substitute_templates(arg, self.template_context, eval_id=arg_name)
            if "map_enum_kwargs" in arg_config:
                arg = map_enum_fields(arg, **arg_config["map_enum_kwargs"])
            if arg_config.get("is_td", False):
                arg = self.td_arr_to_ns(arg)
            if arg_config.get("is_dt", False):
                arg = self.dt_arr_to_ns(arg)
            if "type" in arg_config:
                checks.assert_instance_of(arg, arg_config["type"], arg_name=arg_name)
            if "subdtype" in arg_config:
                checks.assert_subdtype(arg, arg_config["subdtype"], arg_name=arg_name)
        return arg

    @classmethod
    def adapt_staticized_to_udf(
        cls, staticized: tp.Kwargs, func: tp.Union[str, tp.Callable], func_name: str
    ) -> None:
        """Adapt a staticized dictionary to a user-defined function (UDF) by updating its import lines.

        Args:
            staticized (Kwargs): Dictionary containing function configuration.
            func (Union[str, Callable]): Function reference, name, or path.
            func_name (str): Target function name.

        Returns:
            None
        """
        target_func_module = inspect.getmodule(staticized["func"])
        if isinstance(func, tuple):
            func, actual_func_name = func
        else:
            actual_func_name = None
        if isinstance(func, (str, Path)):
            if actual_func_name is None:
                actual_func_name = func_name
            if (
                isinstance(func, str)
                and not func.endswith(".py")
                and hasattr(target_func_module, func)
            ):
                staticized[f"{func_name}_block"] = func
                return None
            func = Path(func)
            module_path = func.resolve()
        else:
            if actual_func_name is None:
                actual_func_name = func.__name__
            if inspect.getmodule(func) == target_func_module:
                staticized[f"{func_name}_block"] = actual_func_name
                return None
            module = inspect.getmodule(func)
            if not hasattr(module, "__file__"):
                raise TypeError(f"{func_name} must be defined in a Python file")
            module_path = Path(module.__file__).resolve()
        if "import_lines" not in staticized:
            staticized["import_lines"] = []
        reload = staticized.get("reload", False)
        staticized["import_lines"].extend(
            [
                f'{func_name}_path = r"{module_path}"',
                f"globals().update(vbt.import_module_from_path({func_name}_path).__dict__, reload={reload})",
            ]
        )
        if actual_func_name != func_name:
            staticized["import_lines"].append(f"{func_name} = {actual_func_name}")

    @classmethod
    def find_target_func(cls, target_func_name: str) -> tp.Callable:
        """Find the target function by its name.

        Args:
            target_func_name (str): Name of the target function.

        Returns:
            Callable: Found target function.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @classmethod
    def resolve_dynamic_target_func(
        cls, target_func_name: str, staticized: tp.KwargsLike
    ) -> tp.Callable:
        """Return the dynamic target function based on the provided configuration.

        Args:
            target_func_name (str): Name of the target function.
            staticized (KwargsLike): Dictionary with function configuration or a function reference.

        Returns:
            Callable: Resolved target function.
        """
        if staticized is None:
            func = cls.find_target_func(target_func_name)
        else:
            if isinstance(staticized, dict):
                staticized = dict(staticized)
                module_path = suggest_module_path(
                    staticized.get("suggest_fname", target_func_name),
                    path=staticized.pop("path", None),
                    mkdir_kwargs=staticized.get("mkdir_kwargs"),
                )
                if "new_func_name" not in staticized:
                    staticized["new_func_name"] = target_func_name

                if staticized.pop("override", False) or not module_path.exists():
                    if "skip_func" not in staticized:

                        def _skip_func(out_lines, func_name):
                            to_skip = lambda x: f"def {func_name}" in x or x.startswith(
                                f"{func_name}_path ="
                            )
                            return any(map(to_skip, out_lines))

                        staticized["skip_func"] = _skip_func
                    module_path = cut_and_save_func(path=module_path, **staticized)
                    if staticized.get("clear_cache", True):
                        remove_dir(
                            module_path.parent / "__pycache__", with_contents=True, missing_ok=True
                        )
                reload = staticized.pop("reload", False)
                module = import_module_from_path(module_path, reload=reload)
                func = getattr(module, staticized["new_func_name"])
            else:
                func = staticized
        return func

    def set_seed(self) -> None:
        """Set the random seed using the object's seed attribute.

        Returns:
            None
        """
        seed = self.seed
        if seed is not None:
            set_seed(seed)

    # ############# Before broadcasting ############# #

    @cachedproperty
    def pre__template_context(self) -> tp.Kwargs:
        """Argument `template_context` before broadcasting.

        Returns:
            Kwargs: Template context before broadcasting.
        """
        return merge_dicts(dict(preparer=self), self["template_context"])

    # ############# Broadcasting ############# #

    @cachedproperty
    def pre_args(self) -> tp.Kwargs:
        """Dictionary of pre-broadcast arguments.

        Iterates over `BasePreparer.arg_config` and retrieves each corresponding `pre__` attribute
        for keys with broadcasting enabled.

        Returns:
            Kwargs: Dictionary of pre-broadcast arguments.
        """
        pre_args = dict()
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                pre_args[k] = getattr(self, "pre__" + k)
        return pre_args

    @cachedproperty
    def args_to_broadcast(self) -> dict:
        """Merged dictionary of arguments to broadcast.

        Combines `idx_setters`, pre-broadcast arguments, and broadcast named arguments.

        Returns:
            dict: Dictionary of arguments to broadcast.
        """
        return merge_dicts(self.idx_setters, self.pre_args, self.broadcast_named_args)

    @cachedproperty
    def def_broadcast_kwargs(self) -> tp.Kwargs:
        """Dictionary of default keyword arguments for broadcasting.

        Includes flags for conversion, flexible settings, wrapper configuration,
        and the pre-template context.

        Returns:
            Kwargs: Dictionary of default broadcasting keyword arguments.
        """
        return dict(
            to_pd=False,
            keep_flex=dict(cash_earnings=self.keep_inout_flex, _def=True),
            wrapper_kwargs=dict(
                freq=self.pre__freq,
                group_by=self.group_by,
            ),
            return_wrapper=True,
            template_context=self.pre__template_context,
        )

    @cachedproperty
    def broadcast_kwargs(self) -> tp.Kwargs:
        """Dictionary of keyword arguments for broadcasting.

        Merges default broadcast kwargs, argument-specific broadcast configurations,
        and additional user-provided overrides.

        Returns:
            Kwargs: Dictionary of broadcasting keyword arguments.
        """
        arg_broadcast_kwargs = defaultdict(dict)
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                broadcast_kwargs = v.get("broadcast_kwargs", None)
                if broadcast_kwargs is None:
                    broadcast_kwargs = {}
                for k2, v2 in broadcast_kwargs.items():
                    arg_broadcast_kwargs[k2][k] = v2
        for k in self.args_to_broadcast:
            new_fill_value = None
            if k in self.pre_args:
                fill_default = self.arg_config[k].get("fill_default", True)
                if self.idx_setters is not None and k in self.idx_setters:
                    new_fill_value = self.get_arg(k, use_idx_setter=False, use_default=fill_default)
                elif fill_default and self.arg_config[k].get("has_default", True):
                    new_fill_value = self.get_arg_default(k)
            elif k in self.broadcast_named_args:
                if self.idx_setters is not None and k in self.idx_setters:
                    new_fill_value = self.broadcast_named_args[k]
            if new_fill_value is not None:
                if not np.isscalar(new_fill_value):
                    raise TypeError(
                        f"Argument '{k}' (and its default) must be a scalar when also provided via records"
                    )
                if "reindex_kwargs" not in arg_broadcast_kwargs:
                    arg_broadcast_kwargs["reindex_kwargs"] = {}
                if k not in arg_broadcast_kwargs["reindex_kwargs"]:
                    arg_broadcast_kwargs["reindex_kwargs"][k] = {}
                arg_broadcast_kwargs["reindex_kwargs"][k]["fill_value"] = new_fill_value

        return merge_dicts(
            self.def_broadcast_kwargs,
            dict(arg_broadcast_kwargs),
            self["broadcast_kwargs"],
        )

    @cachedproperty
    def broadcast_result(self) -> tp.Any:
        """Result of the broadcasting process.

        The result is typically a tuple where the first element contains
        the post-broadcast arguments and the second element is the array wrapper.

        Returns:
            Any: Result of the broadcasting process.
        """
        return broadcast(self.args_to_broadcast, **self.broadcast_kwargs)

    @cachedproperty
    def post_args(self) -> tp.Kwargs:
        """Dictionary of arguments after broadcasting.

        Extracts the first element from the broadcasting result.

        Returns:
            Kwargs: Dictionary of post-broadcast arguments.
        """
        return self.broadcast_result[0]

    @cachedproperty
    def post_broadcast_named_args(self) -> tp.Kwargs:
        """Dictionary of custom broadcast arguments.

        Filters the post-broadcast arguments to include only those specified as named broadcast arguments,
        or those from index setters not present in the pre-broadcast arguments.

        Returns:
            Kwargs: Dictionary of post-broadcast named arguments.
        """
        if self.broadcast_named_args is None:
            return dict()
        post_broadcast_named_args = dict()
        for k, v in self.post_args.items():
            if k in self.broadcast_named_args or self.idx_setters is not None and k in self.idx_setters and k not in self.pre_args:
                post_broadcast_named_args[k] = v
        return post_broadcast_named_args

    @cachedproperty
    def wrapper(self) -> ArrayWrapper:
        """Array wrapper from the broadcasting process.

        Extracts the second element of the broadcasting result.

        Returns:
            ArrayWrapper: Array wrapper containing the broadcasted data.
        """
        return self.broadcast_result[1]

    @cachedproperty
    def target_shape(self) -> tp.Shape:
        """Target shape from the array wrapper.

        Uses the 2D shape attribute of the wrapper.

        Returns:
            Shape: Target shape of the array wrapper.
        """
        return self.wrapper.shape_2d

    @cachedproperty
    def index(self) -> tp.Array1d:
        """Index in nanosecond format from the array wrapper.

        Returns:
            Array1d: Index of the array wrapper in nanoseconds.
        """
        return self.wrapper.ns_index

    @cachedproperty
    def freq(self) -> int:
        """Frequency in nanosecond format from the array wrapper.

        Returns:
            int: Frequency of the array wrapper in nanoseconds.
        """
        return self.wrapper.ns_freq

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Merges details from the array wrapper (`wrapper`, `target_shape`, `index`, `freq`),
        broadcast arguments from `BasePreparer.arg_config`, post-broadcast named arguments,
        and the pre-template context.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        builtin_args = {}
        for k, v in self.arg_config.items():
            if v.get("broadcast", False):
                builtin_args[k] = getattr(self, k)
        return merge_dicts(
            dict(
                wrapper=self.wrapper,
                target_shape=self.target_shape,
                index=self.index,
                freq=self.freq,
            ),
            builtin_args,
            self.post_broadcast_named_args,
            self.pre__template_context,
        )

    # ############# Result ############# #

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        """Target function to be invoked with broadcasted arguments.

        Returns:
            Optional[Callable]: Target function to be invoked or None if no target function is defined.
        """
        return None

    @cachedproperty
    def target_arg_map(self) -> tp.Kwargs:
        """Mapping of target function parameters to preparer attribute names.

        This mapping aligns broadcasted arguments with the target function's parameters.

        Returns:
            Kwargs: Dictionary mapping target function parameter names to preparer attribute names.
        """
        return dict()

    @cachedproperty
    def target_args(self) -> tp.KwargsLike:
        """Dictionary of arguments to pass to the target function.

        Maps parameter names of `target_func` to corresponding preparer attributes using `target_arg_map`.
        Returns None if no target function is defined.

        Returns:
            KwargsLike: Dictionary of arguments to be passed to the target function.
        """
        if self.target_func is not None:
            target_arg_map = self.target_arg_map
            func_arg_names = get_func_arg_names(self.target_func)
            target_args = {}
            for k in func_arg_names:
                arg_attr = target_arg_map.get(k, k)
                if arg_attr is not None and hasattr(self, arg_attr):
                    target_args[k] = getattr(self, arg_attr)
            return target_args
        return None

    # ############# Docs ############# #

    @classmethod
    def build_arg_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build and return documentation for the argument configuration.

        Uses the docstring from the `arg_config` attribute of the given source class (defaulting to `BasePreparer`)
        and substitutes placeholders with the current class's argument configuration details.

        Args:
            source_cls (Optional[type]): Source class providing the original configuration.

        Returns:
            str: Generated documentation for the argument configuration.
        """
        if source_cls is None:
            source_cls = BasePreparer
        return string.Template(
            inspect.cleandoc(get_dict_attr(source_cls, "arg_config").__doc__)
        ).substitute(
            {"arg_config": cls.arg_config.prettify_doc(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_arg_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Override the class's argument configuration documentation.

        Updates the provided documentation dictionary with generated documentation from `build_arg_config_doc`.

        Args:
            __pdoc__ (dict): Dictionary mapping objects to their documentation strings.
            source_cls (Optional[type]): Source class providing the original configuration.

        Returns:
            None
        """
        __pdoc__[cls.__name__ + ".arg_config"] = cls.build_arg_config_doc(source_cls=source_cls)


BasePreparer.override_arg_config_doc(__pdoc__)
