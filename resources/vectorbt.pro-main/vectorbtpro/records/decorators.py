# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing class decorators for records."""

import keyword
from functools import partial

from vectorbtpro import _typing as tp
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import cacheable_property, cached_property
from vectorbtpro.utils.mapping import to_value_mapping

__all__ = []


def override_field_config(config: Config, merge_configs: bool = True) -> tp.ClassWrapper:
    """Class decorator to override the field configuration of a subclass of `vectorbtpro.records.base.Records`.

    Pass a configuration to this decorator to override the `_field_config` attribute of the class.
    If merging is enabled, the existing configuration is merged with the provided configuration.
    Use merge_configs=False to disable merging and effectively disable field inheritance.

    Args:
        config (Config): Configuration to override the field configuration.
        merge_configs (bool): Whether to merge the existing configuration with the provided one.

            If False, replace it entirely.

    Returns:
        ClassWrapper: Decorator function that returns the decorated class.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Records")
        if merge_configs:
            new_config = merge_dicts(cls.field_config, config)
        else:
            new_config = config
        if not isinstance(new_config, Config):
            new_config = HybridConfig(new_config)
        cls._field_config = new_config
        return cls

    return wrapper


def attach_fields(*args, on_conflict: str = "raise") -> tp.FlexClassWrapper:
    """Class decorator to attach field properties to a subclass of `vectorbtpro.records.base.Records`.

    Extracts `dtype` and additional metadata from the class's `field_config` to generate properties that
    map fields. Additional configuration can be provided via a dictionary to override the default behavior.

    !!! note
        Ensure that `attach_fields` is applied after `override_field_config`.

    The configuration (if provided) should include keys corresponding to field names, each mapping to
    a dictionary that may contain:

    * `attach`: Determines whether to attach the field property.
        A string value may be provided to specify the target attribute name.
    * `defaults`: Dictionary with keyword arguments for `vectorbtpro.records.base.Records.map_field`.
    * `attach_filters`: Specifies whether to attach filters based on field values.
        If provided as a dictionary, it maps filter values to target filter names.
        When True, the mapping defined in `field_config` is used.
    * `filter_defaults`: Dictionary with keyword arguments for `vectorbtpro.records.base.Records.apply_mask`.
        It can be specified globally or per target filter name.
    * `on_conflict`: Overrides the global `on_conflict` setting for both field and filter properties.

    Attribute names are derived by inserting underscores between capital letters and converting to lowercase.
    If an attribute with the same name exists and is not listed in `field_config`:

    * It is overridden if `on_conflict` is "override".
    * It is ignored if `on_conflict` is "ignore".
    * Error is raised if `on_conflict` is "raise".

    Args:
        *args: Positional arguments representing either the decorated class
            or a configuration dictionary.
        on_conflict (str): Strategy for handling attribute name conflicts,
            which can be "raise", "ignore", or "override".

    Returns:
        FlexClassWrapper: Decorator that attaches field properties to the target class.
    """

    def wrapper(cls: tp.Type[tp.T], config: tp.DictLike = None) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Records")

        dtype = cls.field_config.get("dtype", None)
        checks.assert_not_none(dtype.fields)

        if config is None:
            config = {}

        def _prepare_attr_name(attr_name: str) -> str:
            checks.assert_instance_of(attr_name, str)
            attr_name = attr_name.replace("NaN", "Nan")
            startswith_ = attr_name.startswith("_")
            new_attr_name = ""
            for i in range(len(attr_name)):
                if attr_name[i].isupper():
                    if i > 0 and attr_name[i - 1].islower():
                        new_attr_name += "_"
                new_attr_name += attr_name[i]
            attr_name = new_attr_name
            if not startswith_ and attr_name.startswith("_"):
                attr_name = attr_name[1:]
            attr_name = attr_name.lower()
            if keyword.iskeyword(attr_name):
                attr_name += "_"
            return attr_name.replace("__", "_")

        def _check_attr_name(attr_name, _on_conflict: str = on_conflict) -> None:
            if attr_name not in cls.field_config.get("settings", {}):
                # Consider only attributes that are not listed in the field config
                if hasattr(cls, attr_name):
                    if _on_conflict.lower() == "raise":
                        raise ValueError(
                            f"An attribute with the name '{attr_name}' already exists in {cls}"
                        )
                    if _on_conflict.lower() == "ignore":
                        return
                    if _on_conflict.lower() == "override":
                        return
                    raise ValueError(f"Value '{_on_conflict}' is invalid for on_conflict")
                if keyword.iskeyword(attr_name):
                    raise ValueError(
                        f"Name '{attr_name}' is a keyword and cannot be used as an attribute name"
                    )

        if dtype is not None:
            for field_name in dtype.names:
                settings = config.get(field_name, {})
                attach = settings.get("attach", True)
                if not isinstance(attach, bool):
                    target_name = attach
                    attach = True
                else:
                    target_name = field_name
                defaults = settings.get("defaults", None)
                if defaults is None:
                    defaults = {}
                attach_filters = settings.get("attach_filters", False)
                filter_defaults = settings.get("filter_defaults", None)
                if filter_defaults is None:
                    filter_defaults = {}
                _on_conflict = settings.get("on_conflict", on_conflict)

                if attach:
                    target_name = _prepare_attr_name(target_name)
                    _check_attr_name(target_name, _on_conflict)

                    def new_prop(
                        self,
                        _field_name: str = field_name,
                        _defaults: tp.KwargsLike = defaults,
                    ) -> MappedArray:
                        return self.get_map_field(_field_name, **_defaults)

                    new_prop.__name__ = target_name
                    new_prop.__module__ = cls.__module__
                    new_prop.__qualname__ = f"{cls.__name__}.{new_prop.__name__}"
                    new_prop.__doc__ = f"Mapped array of the field `{field_name}`."
                    setattr(cls, target_name, cached_property(new_prop))

                if attach_filters:
                    if isinstance(attach_filters, bool):
                        if not attach_filters:
                            continue
                        mapping = (
                            cls.field_config.get("settings", {})
                            .get(field_name, {})
                            .get("mapping", None)
                        )
                    else:
                        mapping = attach_filters
                    if mapping is None:
                        raise ValueError(
                            f"Field '{field_name}': Mapping is required to attach filters"
                        )
                    mapping = to_value_mapping(mapping)

                    for filter_value, target_filter_name in mapping.items():
                        if target_filter_name is None:
                            continue
                        if isinstance(attach_filters, bool):
                            target_filter_name = field_name + "_" + target_filter_name
                        target_filter_name = _prepare_attr_name(target_filter_name)
                        _check_attr_name(target_filter_name, _on_conflict)
                        if target_filter_name in filter_defaults:
                            __filter_defaults = filter_defaults[target_filter_name]
                        else:
                            __filter_defaults = filter_defaults

                        def new_filter_prop(
                            self,
                            _field_name: str = field_name,
                            _filter_value: tp.Any = filter_value,
                            _filter_defaults: tp.KwargsLike = __filter_defaults,
                        ) -> MappedArray:
                            filter_mask = self.get_field_arr(_field_name) == _filter_value
                            return self.apply_mask(filter_mask, **_filter_defaults)

                        new_filter_prop.__name__ = target_filter_name
                        new_filter_prop.__module__ = cls.__module__
                        new_filter_prop.__qualname__ = f"{cls.__name__}.{new_filter_prop.__name__}"
                        new_filter_prop.__doc__ = (
                            f"Records filtered by `{field_name} == {filter_value}`."
                        )
                        setattr(cls, target_filter_name, cached_property(new_filter_prop))

        return cls

    if len(args) == 0:
        return wrapper
    elif len(args) == 1:
        if isinstance(args[0], type):
            return wrapper(args[0])
        return partial(wrapper, config=args[0])
    elif len(args) == 2:
        return wrapper(args[0], config=args[1])
    raise ValueError("Either class, config, class and config, or keyword arguments must be passed")


def attach_shortcut_properties(config: Config) -> tp.ClassWrapper:
    """Attach shortcut properties to a subclass of `vectorbtpro.records.base.Records`.

    This class decorator adds shortcut properties to a class based on a configuration mapping.

    The configuration mapping `config` must have target property names as keys and their
    corresponding settings as values. Each setting may include the following keys:

    * `method_name`: Name of the source method.
        If omitted, defaults to the target name prefixed with `get_`.
    * `obj_type`: Type of the returned object.
        Supported types are "array" for 2-dimensional arrays, "red_array" for 1-dimensional arrays,
        "records" for record arrays, and "mapped_array" for mapped arrays. Defaults to "records".
    * `group_by_aware`: Whether the returned object is aligned based on current grouping.
        Defaults to True.
    * `method_kwargs`: Keyword arguments for the source method.
    * `decorator`: Decorator function.
        If not specified, defaults to `vectorbtpro.utils.decorators.cached_property` for object types
        "records" and "red_array", and to `vectorbtpro.utils.decorators.cacheable_property` otherwise.
    * `decorator_kwargs`: Keyword arguments for the decorator, which by default include the
        options `obj_type` and `group_by_aware`.
    * `docstring`: Docstring for the generated method.

    Args:
        config (Config): Configuration mapping containing target property names
            and their corresponding settings.

    Returns:
        ClassWrapper: Class decorator that attaches the configured shortcut properties
            to a subclass of `vectorbtpro.records.base.Records`.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Records")

        for target_name, settings in config.items():
            if target_name.startswith("get_"):
                raise ValueError(f"Property names cannot have prefix 'get_' ('{target_name}')")
            method_name = settings.get("method_name", "get_" + target_name)
            obj_type = settings.get("obj_type", "records")
            group_by_aware = settings.get("group_by_aware", True)
            method_kwargs = settings.get("method_kwargs", None)
            method_kwargs = resolve_dict(method_kwargs)
            decorator = settings.get("decorator", None)
            if decorator is None:
                if obj_type in ("red_array", "records"):
                    decorator = cached_property
                else:
                    decorator = cacheable_property
            decorator_kwargs = merge_dicts(
                dict(obj_type=obj_type, group_by_aware=group_by_aware),
                settings.get("decorator_kwargs", None),
            )
            docstring = settings.get("docstring", None)
            if docstring is None:
                if len(method_kwargs) == 0:
                    docstring = f"`{cls.__name__}.{method_name}` with default arguments."
                else:
                    docstring = f"`{cls.__name__}.{method_name}` with arguments `{method_kwargs}`."

            def new_prop(
                self, _method_name: str = method_name, _method_kwargs: tp.Kwargs = method_kwargs
            ) -> tp.Any:
                return getattr(self, _method_name)(**_method_kwargs)

            new_prop.__name__ = target_name
            new_prop.__module__ = cls.__module__
            new_prop.__qualname__ = f"{cls.__name__}.{new_prop.__name__}"
            new_prop.__doc__ = docstring
            setattr(cls, new_prop.__name__, decorator(new_prop, **decorator_kwargs))
        return cls

    return wrapper
