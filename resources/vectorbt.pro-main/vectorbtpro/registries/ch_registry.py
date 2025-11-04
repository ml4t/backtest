# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a global registry for chunkable functions.

!!! info
    For default settings, see `vectorbtpro._settings.chunking`.
"""

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.chunking import chunked, resolve_chunked, resolve_chunked_option
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.template import RepEval

__all__ = [
    "ChunkableRegistry",
    "ch_reg",
    "register_chunkable",
]


@define
class ChunkedSetup(DefineMixin):
    """Class representing a chunkable setup.

    !!! note
        The instance is hashed solely using its `setup_id`.
    """

    setup_id: tp.Hashable = define.field()
    """Unique identifier for the setup."""

    func: tp.Callable = define.field()
    """Function registered as chunkable."""

    options: tp.DictLike = define.field(default=None)
    """Dictionary of options for chunked processing."""

    tags: tp.SetLike = define.field(default=None)
    """Set of tags associated with the setup."""

    @staticmethod
    def get_hash(setup_id: tp.Hashable) -> int:
        return hash((setup_id,))

    @property
    def hash_key(self) -> tuple:
        return (self.setup_id,)


class ChunkableRegistry(Base):
    """Class for registering and managing chunkable functions."""

    def __init__(self) -> None:
        self._setups = {}

    @property
    def setups(self) -> tp.Dict[tp.Hashable, ChunkedSetup]:
        """Dictionary mapping setup IDs to registered `ChunkedSetup` instances.

        Returns:
            Dict[Hashable, ChunkedSetup]: Dictionary of registered setups.
        """
        return self._setups

    def register(
        self,
        func: tp.Callable,
        setup_id: tp.Optional[tp.Hashable] = None,
        options: tp.DictLike = None,
        tags: tp.SetLike = None,
    ) -> None:
        """Register a new chunkable setup.

        Args:
            func (Callable): Function to register as chunkable.
            setup_id (Optional[Hashable]): Unique identifier for the setup.

                If omitted, it is derived from the function's module and name.
            options (DictLike): Dictionary of options for chunking.
            tags (SetLike): Tags associated with the function.

        Returns:
            None
        """
        if setup_id is None:
            setup_id = func.__module__ + "." + func.__name__
        setup = ChunkedSetup(setup_id=setup_id, func=func, options=options, tags=tags)
        self.setups[setup_id] = setup

    def match_setups(
        self, expression: tp.Optional[str] = None, context: tp.KwargsLike = None
    ) -> tp.Set[ChunkedSetup]:
        """Return a set of registered setups that match a given expression.

        Args:
            expression (Optional[str]): Expression to filter setups.

                If None, all setups are matched.
            context (KwargsLike): Additional context data used during evaluation.

        Returns:
            Set[ChunkedSetup]: Set of setups that satisfy the specified expression.
        """
        matched_setups = set()
        for setup in self.setups.values():
            if expression is None:
                result = True
            else:
                result = RepEval(expression).substitute(
                    context=merge_dicts(setup.asdict(), context)
                )
                checks.assert_instance_of(result, bool)

            if result:
                matched_setups.add(setup)
        return matched_setups

    def get_setup(
        self, setup_id_or_func: tp.Union[tp.Hashable, tp.Callable]
    ) -> tp.Optional[ChunkedSetup]:
        """Retrieve a chunkable setup by its identifier or function.

        Args:
            setup_id_or_func (Union[Hashable, Callable]): Setup identifier or function.

                If a function is provided, the identifier is constructed from its module and name.

        Returns:
            Optional[ChunkedSetup]: Corresponding chunkable setup if found; otherwise, None.
        """
        if hasattr(setup_id_or_func, "py_func"):
            nb_setup_id = setup_id_or_func.__module__ + "." + setup_id_or_func.__name__
            if nb_setup_id in self.setups:
                setup_id = nb_setup_id
            else:
                setup_id = (
                    setup_id_or_func.py_func.__module__ + "." + setup_id_or_func.py_func.__name__
                )
        elif callable(setup_id_or_func):
            setup_id = setup_id_or_func.__module__ + "." + setup_id_or_func.__name__
        else:
            setup_id = setup_id_or_func
        if setup_id not in self.setups:
            return None
        return self.setups[setup_id]

    def decorate(
        self,
        setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
        target_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Callable:
        """Decorate a chunkable function using the `chunked` decorator.

        The setup is retrieved using `ChunkableRegistry.get_setup` and its options are merged
        with additional keyword arguments.

        Args:
            setup_id_or_func (Union[Hashable, Callable]): Setup identifier or function.

                If a function is provided, the identifier is constructed from its module and name.
            target_func (Optional[Callable]): Alternative function to decorate.

                If omitted, the function from the setup is used.
            **kwargs: Keyword arguments for `vectorbtpro.utils.chunking.chunked`.

                These options override the setup's options.

        Returns:
            Callable: Decorated function with chunked processing applied.
        """
        setup = self.get_setup(setup_id_or_func)
        if setup is None:
            raise KeyError(f"Setup for {setup_id_or_func} not registered")

        if target_func is not None:
            func = target_func
        elif callable(setup_id_or_func):
            func = setup_id_or_func
        else:
            func = setup.func
        return chunked(func, **merge_dicts(setup.options, kwargs))

    def resolve_option(
        self,
        setup_id_or_func: tp.Union[tp.Hashable, tp.Callable],
        option: tp.ChunkedOption,
        target_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Callable:
        """Resolve a chunkable function using the `resolve_chunked` function.

        Similar to `ChunkableRegistry.decorate`, but applies a specific chunking option.

        Args:
            setup_id_or_func (Union[Hashable, Callable]): Setup identifier or function.

                If a function is provided, the identifier is constructed from its module and name.
            option (ChunkedOption): Option to control chunked processing.
            target_func (Optional[Callable]): Alternative function to apply the setup on.
            **kwargs: Keyword arguments for `vectorbtpro.utils.chunking.resolve_chunked`.

                These options override the setup's options.

        Returns:
            Callable: Function with the resolved chunking applied.
        """
        setup = self.get_setup(setup_id_or_func)
        if setup is None:
            if callable(setup_id_or_func):
                option = resolve_chunked_option(option=option)
                if option is None:
                    return setup_id_or_func
            raise KeyError(f"Setup for {setup_id_or_func} not registered")

        if target_func is not None:
            func = target_func
        elif callable(setup_id_or_func):
            func = setup_id_or_func
        else:
            func = setup.func
        return resolve_chunked(func, option=option, **merge_dicts(setup.options, kwargs))


ch_reg = ChunkableRegistry()
"""Default registry instance of `ChunkableRegistry` for chunkable functions."""


def register_chunkable(
    func: tp.Optional[tp.Callable] = None,
    setup_id: tp.Optional[tp.Hashable] = None,
    registry: ChunkableRegistry = ch_reg,
    tags: tp.SetLike = None,
    return_wrapped: bool = False,
    **options,
) -> tp.Callable:
    """Register a new chunkable function.

    Wraps the given function with chunkable behavior using the `chunked` decorator if `return_wrapped` is True;
    otherwise, the function remains unwrapped. Options are merged in the following order:

    * `options` from `vectorbtpro._settings.chunking`
    * `setup_options.{setup_id}` from `vectorbtpro._settings.chunking`
    * provided `options`
    * `override_options` from `vectorbtpro._settings.chunking`
    * `override_setup_options.{setup_id}` from `vectorbtpro._settings.chunking`

    Args:
        func (Optional[Callable]): Function to register.

            If omitted, a decorator is returned.
        setup_id (Optional[Hashable]): Unique identifier for the setup.

            If omitted, it is derived from the function's module and name.
        registry (ChunkableRegistry): Registry to register the function in.
        tags (SetLike): Tags associated with the function.
        return_wrapped (bool): Flag indicating whether to wrap the function using the `chunked` decorator.
        **options: Additional options for configuring the chunkable setup.

    Returns:
        Callable: Registered function, optionally wrapped with chunking.

    !!! note
        Calling the `register_chunkable` decorator before (or below) the
        `vectorbtpro.registries.jit_registry.register_jitted` decorator with `return_wrapped` set
        to True won't work. Using it after (or above) `vectorbtpro.registries.jit_registry.register_jitted`
        works for Python calls but not for Numba. Generally, avoid wrapping immediately and use
        `ChunkableRegistry.decorate` for decoration.

    !!! info
        For default settings, see `vectorbtpro._settings.chunking`.
    """

    def decorator(_func: tp.Callable) -> tp.Callable:
        nonlocal setup_id, options

        from vectorbtpro._settings import settings

        chunking_cfg = settings["chunking"]

        if setup_id is None:
            setup_id = _func.__module__ + "." + _func.__name__
        options = merge_dicts(
            chunking_cfg.get("options", None),
            chunking_cfg.get("setup_options", {}).get(setup_id, None),
            options,
            chunking_cfg.get("override_options", None),
            chunking_cfg.get("override_setup_options", {}).get(setup_id, None),
        )

        registry.register(func=_func, setup_id=setup_id, options=options, tags=tags)
        if return_wrapped:
            return chunked(_func, **options)
        return _func

    if func is None:
        return decorator
    return decorator(func)
