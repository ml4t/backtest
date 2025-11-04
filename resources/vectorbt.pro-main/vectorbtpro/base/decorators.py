# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing class decorators for base classes."""

from functools import cached_property as cachedproperty

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts

__all__ = []


def override_arg_config(config: Config, merge_configs: bool = True) -> tp.ClassWrapper:
    """Class decorator to override the argument configuration of a subclass of
    `vectorbtpro.base.preparing.BasePreparer`.

    Args:
        config (Config): Configuration to override the current argument config.
        merge_configs (bool): Whether to merge the existing configuration with the provided one.

            If False, replace it entirely.

    Returns:
        ClassWrapper: Decorated class with the updated `_arg_config`.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "BasePreparer")
        if merge_configs:
            new_config = merge_dicts(cls.arg_config, config)
        else:
            new_config = config
        if not isinstance(new_config, Config):
            new_config = HybridConfig(new_config)
        cls._arg_config = new_config
        return cls

    return wrapper


def attach_arg_properties(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
    """Class decorator to attach properties for arguments defined in the argument configuration
    of a subclass of `vectorbtpro.base.preparing.BasePreparer`.

    Args:
        cls (Type[T]): Subclass of `vectorbtpro.base.preparing.BasePreparer` for which
            argument properties will be attached.

    Returns:
        Type[T]: Decorated class with the new argument properties attached.
    """
    checks.assert_subclass_of(cls, "BasePreparer")

    for arg_name, settings in cls.arg_config.items():
        attach = settings.get("attach", None)
        broadcast = settings.get("broadcast", False)
        substitute_templates = settings.get("substitute_templates", False)
        if (isinstance(attach, bool) and attach) or (
            attach is None and (broadcast or substitute_templates)
        ):
            if broadcast:
                return_type = tp.ArrayLike
            else:
                return_type = object
            target_pre_name = "pre__" + arg_name
            if not hasattr(cls, target_pre_name):

                def pre_arg_prop(self, _arg_name: str = arg_name) -> return_type:
                    return self.get_arg(_arg_name)

                pre_arg_prop.__name__ = target_pre_name
                pre_arg_prop.__module__ = cls.__module__
                pre_arg_prop.__qualname__ = f"{cls.__name__}.{pre_arg_prop.__name__}"
                if broadcast and substitute_templates:
                    pre_arg_prop.__doc__ = f"Original value of argument `{arg_name}` before broadcasting and template substitution."
                elif broadcast:
                    pre_arg_prop.__doc__ = (
                        f"Original value of argument `{arg_name}` before broadcasting."
                    )
                else:
                    pre_arg_prop.__doc__ = (
                        f"Original value of argument `{arg_name}` before template substitution."
                    )
                setattr(cls, pre_arg_prop.__name__, cachedproperty(pre_arg_prop))
                getattr(cls, pre_arg_prop.__name__).__set_name__(cls, pre_arg_prop.__name__)

            target_post_name = "post__" + arg_name
            if not hasattr(cls, target_post_name):

                def post_arg_prop(self, _arg_name: str = arg_name) -> return_type:
                    return self.prepare_post_arg(_arg_name)

                post_arg_prop.__name__ = target_post_name
                post_arg_prop.__module__ = cls.__module__
                post_arg_prop.__qualname__ = f"{cls.__name__}.{post_arg_prop.__name__}"
                if broadcast and substitute_templates:
                    post_arg_prop.__doc__ = f"Processed value of argument `{arg_name}` after broadcasting and template substitution."
                elif broadcast:
                    post_arg_prop.__doc__ = (
                        f"Processed value of argument `{arg_name}` after broadcasting."
                    )
                else:
                    post_arg_prop.__doc__ = (
                        f"Processed value of argument `{arg_name}` after template substitution."
                    )
                setattr(cls, post_arg_prop.__name__, cachedproperty(post_arg_prop))
                getattr(cls, post_arg_prop.__name__).__set_name__(cls, post_arg_prop.__name__)

            target_name = arg_name
            if not hasattr(cls, target_name):

                def arg_prop(self, _target_post_name: str = target_post_name) -> return_type:
                    return getattr(self, _target_post_name)

                arg_prop.__name__ = target_name
                arg_prop.__module__ = cls.__module__
                arg_prop.__qualname__ = f"{cls.__name__}.{arg_prop.__name__}"
                arg_prop.__doc__ = f"Processed value for argument `{arg_name}`."
                setattr(cls, arg_prop.__name__, cachedproperty(arg_prop))
                getattr(cls, arg_prop.__name__).__set_name__(cls, arg_prop.__name__)
        elif (isinstance(attach, bool) and attach) or attach is None:
            if not hasattr(cls, arg_name):

                def arg_prop(self, _arg_name: str = arg_name) -> tp.Any:
                    return self.get_arg(_arg_name)

                arg_prop.__name__ = arg_name
                arg_prop.__module__ = cls.__module__
                arg_prop.__qualname__ = f"{cls.__name__}.{arg_prop.__name__}"
                arg_prop.__doc__ = f"Processed value for argument `{arg_name}`."
                setattr(cls, arg_prop.__name__, cachedproperty(arg_prop))
                getattr(cls, arg_prop.__name__).__set_name__(cls, arg_prop.__name__)

    return cls
