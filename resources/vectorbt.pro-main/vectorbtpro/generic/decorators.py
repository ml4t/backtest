# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing class decorators for generic accessors."""

import inspect

from vectorbtpro import _typing as tp
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Config, merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = []


def attach_nb_methods(config: Config) -> tp.ClassWrapper:
    """Class decorator to attach Numba methods to a class.

    Args:
        config (Config): Dictionary mapping target method names (str) to
            configuration dictionaries with the following keys:

            * `func` (Callable): Function to be wrapped, where the first argument accepts a 2-dim array.
            * `is_reducing` (bool): Specifies if the function performs a reduction.
            * `disable_jitted` (bool): Disables the jitted option when set.
            * `disable_chunked` (bool): Disables the chunked option when set.
            * `replace_signature` (bool): Replaces the target method signature with
                that of the source function.
            * `wrap_kwargs` (KwargsLike): Keyword arguments for wrapping that
                will be merged with user-supplied values.

    Returns:
        ClassWrapper: Decorated class with the new methods attached.

    The decorated class must be a subclass of `vectorbtpro.base.wrapping.Wrapping`.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        from vectorbtpro.base.wrapping import Wrapping

        checks.assert_subclass_of(cls, Wrapping)

        for target_name, settings in config.items():
            func = settings["func"]
            is_reducing = settings.get("is_reducing", False)
            disable_jitted = settings.get("disable_jitted", False)
            disable_chunked = settings.get("disable_chunked", False)
            replace_signature = settings.get("replace_signature", True)
            default_wrap_kwargs = settings.get(
                "wrap_kwargs", dict(name_or_index=target_name) if is_reducing else None
            )

            def new_method(
                self,
                *args,
                _target_name: str = target_name,
                _func: tp.Callable = func,
                _is_reducing: bool = is_reducing,
                _disable_jitted: bool = disable_jitted,
                _disable_chunked: bool = disable_chunked,
                _default_wrap_kwargs: tp.KwargsLike = default_wrap_kwargs,
                jitted: tp.JittedOption = None,
                chunked: tp.ChunkedOption = None,
                wrap_kwargs: tp.KwargsLike = None,
                **kwargs,
            ) -> tp.SeriesFrame:
                args = (self.to_2d_array(),) + args
                inspect.signature(_func).bind(*args, **kwargs)

                if not _disable_jitted:
                    _func = jit_reg.resolve_option(_func, jitted)
                elif jitted is not None:
                    raise ValueError("This method doesn't support jitting")
                if not _disable_chunked:
                    _func = ch_reg.resolve_option(_func, chunked)
                elif chunked is not None:
                    raise ValueError("This method doesn't support chunking")
                a = _func(*args, **kwargs)
                wrap_kwargs = merge_dicts(_default_wrap_kwargs, wrap_kwargs)
                if _is_reducing:
                    return self.wrapper.wrap_reduced(a, **wrap_kwargs)
                return self.wrapper.wrap(a, **wrap_kwargs)

            if replace_signature:
                # Replace the function's signature with the original one
                source_sig = inspect.signature(func)
                new_method_params = tuple(inspect.signature(new_method).parameters.values())
                self_arg = new_method_params[0]
                jitted_arg = new_method_params[-4]
                chunked_arg = new_method_params[-3]
                wrap_kwargs_arg = new_method_params[-2]
                new_parameters = (self_arg,) + tuple(source_sig.parameters.values())[1:]
                if not disable_jitted:
                    new_parameters += (jitted_arg,)
                if not disable_chunked:
                    new_parameters += (chunked_arg,)
                new_parameters += (wrap_kwargs_arg,)
                new_method.__signature__ = source_sig.replace(parameters=new_parameters)

            new_method.__name__ = target_name
            new_method.__module__ = cls.__module__
            new_method.__qualname__ = f"{cls.__name__}.{new_method.__name__}"
            new_method.__doc__ = f"See `{func.__module__ + '.' + func.__name__}`."
            setattr(cls, target_name, new_method)
        return cls

    return wrapper


def attach_transform_methods(config: Config) -> tp.ClassWrapper:
    """Class decorator to add transformation methods to a class.

    Args:
        config (Config): Dictionary mapping target method names (str) to
            configuration dictionaries with the following keys:

            * `transformer` (MaybeType[Transformer]): Transformer class or instance.
            * `docstring` (str): Docstring assigned to the generated method.
            * `replace_signature` (bool): Replaces the target method signature with that of the transformer.

    Returns:
        ClassWrapper: Decorated class with the new methods attached.

    The decorated class must be a subclass of `vectorbtpro.generic.accessors.GenericAccessor`.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "GenericAccessor")

        for target_name, settings in config.items():
            transformer = settings["transformer"]
            docstring = settings.get("docstring", f"See `{transformer.__name__}`.")
            replace_signature = settings.get("replace_signature", True)

            def new_method(
                self,
                _target_name: str = target_name,
                _transformer: tp.MaybeType[tp.TransformerT] = transformer,
                **kwargs,
            ) -> tp.SeriesFrame:
                if inspect.isclass(_transformer):
                    arg_names = get_func_arg_names(_transformer.__init__)
                    transformer_kwargs = dict()
                    for arg_name in arg_names:
                        if arg_name in kwargs:
                            transformer_kwargs[arg_name] = kwargs.pop(arg_name)
                    return self.transform(_transformer(**transformer_kwargs), **kwargs)
                return self.transform(_transformer, **kwargs)

            if replace_signature:
                source_sig = inspect.signature(transformer.__init__)
                new_method_params = tuple(inspect.signature(new_method).parameters.values())
                if inspect.isclass(transformer):
                    transformer_params = tuple(source_sig.parameters.values())
                    source_sig = inspect.Signature(
                        (new_method_params[0],) + transformer_params[1:] + (new_method_params[-1],),
                    )
                    new_method.__signature__ = source_sig
                else:
                    source_sig = inspect.Signature(
                        (new_method_params[0],) + (new_method_params[-1],)
                    )
                    new_method.__signature__ = source_sig

            new_method.__name__ = target_name
            new_method.__module__ = cls.__module__
            new_method.__qualname__ = f"{cls.__name__}.{new_method.__name__}"
            new_method.__doc__ = docstring
            setattr(cls, target_name, new_method)
        return cls

    return wrapper
