# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing class decorators for portfolio."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Config, resolve_dict
from vectorbtpro.utils.decorators import cacheable_property, cached_property
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = []


def attach_returns_acc_methods(config: Config) -> tp.ClassWrapper:
    """Class decorator to attach returns accessor methods.

    The decorated class must be a subclass of `vectorbtpro.portfolio.base.Portfolio`.

    Args:
        config (Config): Dictionary mapping target method names to their corresponding settings.

            Each settings dictionary must include:

            * `source_name`: Name of the source method. Defaults to the target method name.
            * `docstring`: Docstring for the generated method.

    Returns:
        ClassWrapper: Decorator that attaches returns accessor methods to the class.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Portfolio")

        for target_name, settings in config.items():
            source_name = settings.get("source_name", target_name)
            docstring = settings.get(
                "docstring", f"See `vectorbtpro.returns.accessors.ReturnsAccessor.{source_name}`."
            )

            def new_method(
                self,
                *,
                returns: tp.Optional[tp.SeriesFrame] = None,
                use_asset_returns: bool = False,
                bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
                log_returns: bool = False,
                daily_returns: bool = False,
                sim_start: tp.Optional[tp.ArrayLike] = None,
                sim_end: tp.Optional[tp.ArrayLike] = None,
                rec_sim_range: bool = False,
                freq: tp.Optional[tp.FrequencyLike] = None,
                year_freq: tp.Optional[tp.FrequencyLike] = None,
                defaults: tp.KwargsLike = None,
                jitted: tp.JittedOption = None,
                chunked: tp.ChunkedOption = None,
                wrapper: tp.Optional[ArrayWrapper] = None,
                group_by: tp.GroupByLike = None,
                _source_name: str = source_name,
                **kwargs,
            ) -> tp.Any:
                returns_acc = self.get_returns_acc(
                    returns=returns,
                    use_asset_returns=use_asset_returns,
                    bm_returns=bm_returns,
                    log_returns=log_returns,
                    daily_returns=daily_returns,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    freq=freq,
                    year_freq=year_freq,
                    defaults=defaults,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
                ret_method = getattr(returns_acc, _source_name)
                if "jitted" in get_func_arg_names(ret_method):
                    kwargs["jitted"] = jitted
                return ret_method(**kwargs)

            new_method.__name__ = "get_" + target_name
            new_method.__module__ = cls.__module__
            new_method.__qualname__ = f"{cls.__name__}.{new_method.__name__}"
            new_method.__doc__ = docstring
            setattr(cls, new_method.__name__, new_method)
        return cls

    return wrapper


def attach_shortcut_properties(config: Config) -> tp.ClassWrapper:
    """Class decorator to attach shortcut properties.

    The decorated class must be a subclass of `vectorbtpro.portfolio.base.Portfolio`.

    Args:
        config (Config): Dictionary mapping target property names to their corresponding settings.

            Each settings dictionary may include:

            * `method_name`: Name of the source method. Defaults to the target name prefixed with "get_".
            * `use_in_outputs`: Boolean indicating whether the property can return an in-place output.
            * `method_kwargs`: Keyword arguments for the source method.
            * `decorator`: Decorator to apply. Defaults to `vectorbtpro.utils.decorators.cached_property`
                for object types 'records' and 'red_array', otherwise
                `vectorbtpro.utils.decorators.cacheable_property`.
            * `docstring`: Docstring for the generated property.
            * Keyword arguments for wrapping, indexing, resampling, stacking, etc.

    Returns:
        ClassWrapper: Decorator that attaches shortcut properties to the class.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Portfolio")

        for target_name, settings in config.items():
            settings = dict(settings)
            if target_name.startswith("get_"):
                raise ValueError(f"Property names cannot have prefix 'get_' ('{target_name}')")
            method_name = settings.pop("method_name", "get_" + target_name)
            use_in_outputs = settings.pop("use_in_outputs", True)
            method_kwargs = settings.pop("method_kwargs", None)
            method_kwargs = resolve_dict(method_kwargs)
            decorator = settings.pop("decorator", None)
            if decorator is None:
                if settings.get("obj_type", "array") in ("red_array", "records"):
                    decorator = cached_property
                else:
                    decorator = cacheable_property
            docstring = settings.pop("docstring", None)
            if docstring is None:
                if len(method_kwargs) == 0:
                    docstring = f"`{cls.__name__}.{method_name}` with default arguments."
                else:
                    docstring = f"`{cls.__name__}.{method_name}` with arguments `{method_kwargs}`."

            def new_prop(
                self,
                _method_name: tp.Optional[str] = method_name,
                _target_name: str = target_name,
                _use_in_outputs: bool = use_in_outputs,
                _method_kwargs: tp.Kwargs = method_kwargs,
            ) -> tp.Any:
                if _use_in_outputs and self.use_in_outputs and self.in_outputs is not None:
                    try:
                        out = self.get_in_output(_target_name)
                        if out is not None:
                            return out
                    except AttributeError:
                        pass

                if _method_name is None:
                    raise ValueError(f"Field '{_target_name}' must be prefilled")
                return getattr(self, _method_name)(**_method_kwargs)

            new_prop.__name__ = target_name
            new_prop.__module__ = cls.__module__
            new_prop.__qualname__ = f"{cls.__name__}.{new_prop.__name__}"
            new_prop.__doc__ = docstring
            setattr(cls, new_prop.__name__, decorator(new_prop, **settings))
        return cls

    return wrapper
