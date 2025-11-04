# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing base asset function classes.

See `vectorbtpro.utils.knowledge` for the toy dataset.
"""

import attr

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks, search_
from vectorbtpro.utils.attr_ import MISSING
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import flat_merge_dicts, merge_dicts, reorder_dict, reorder_list
from vectorbtpro.utils.execution import NoResult
from vectorbtpro.utils.formatting import dump
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc, substitute_templates

__all__ = [
    "AssetFunc",
]


class AssetFunc(Base):
    """Abstract base class for asset functions.

    Provides methods to prepare arguments and execute asset function calls.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Identifier for the asset function's short name used in expressions."""

    _wrap: tp.ClassVar[tp.Optional[bool]] = None
    """Indicates whether the result should be wrapped with `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`."""

    @classmethod
    def prepare(cls, *args, **kwargs) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        return args, kwargs

    @classmethod
    def call(cls, d: tp.Any, *args, **kwargs) -> tp.Any:
        """Call the asset function.

        Args:
            d: Input data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Result of the asset function call.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @classmethod
    def prepare_and_call(cls, d: tp.Any, *args, **kwargs) -> tp.Any:
        """Prepare arguments and invoke the asset function.

        Args:
            d: Input data.
            *args: Positional arguments for `AssetFunc.prepare` and ultimately to `AssetFunc.call`.
            **kwargs: Keyword arguments for `AssetFunc.prepare` and ultimately to `AssetFunc.call`.

        Returns:
            Any: Result returned by the asset function.
        """
        args, kwargs = cls.prepare(*args, **kwargs)
        return cls.call(d, *args, **kwargs)


class GetAssetFunc(AssetFunc):
    """Asset function class for retrieving asset data with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.get`.

    Extracts data based on a specified path and optionally transforms it using a provided template.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "get"

    _wrap: tp.ClassVar[tp.Optional[bool]] = False

    @classmethod
    def prepare(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Optional[tp.CustomTemplateLike] = None,
        template_context: tp.KwargsLike = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to get (e.g. "x.y[0].z").
            keep_path (Optional[bool]): If True, returns results structured as nested dictionaries
                mirroring the specified path.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            source (Optional[CustomTemplateLike]): Template, function, or string for preprocessing;
                in the template, "i" denotes the index, "d" the full data item, and "x" the extracted part.
            template_context (KwargsLike): Additional context for template substitution.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        keep_path = asset_cls.resolve_setting(keep_path, "keep_path")
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        template_context = asset_cls.resolve_setting(
            template_context, "template_context", merge=True
        )
        template_context = flat_merge_dicts({"asset_cls": asset_cls}, template_context)

        if path is not None:
            if isinstance(path, list):
                path = [search_.resolve_pathlike_key(p) for p in path]
            else:
                path = search_.resolve_pathlike_key(path)
        if source is not None:
            if isinstance(source, str):
                source = RepEval(source)
            elif checks.is_function(source):
                if checks.is_builtin_func(source):
                    source = RepFunc(lambda _source=source: _source)
                else:
                    source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(
                    f"Source must be a string, function, or template, not {type(source)}"
                )
        return (), {
            **dict(
                path=path,
                keep_path=keep_path,
                skip_missing=skip_missing,
                source=source,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: bool = False,
        skip_missing: bool = False,
        source: tp.Optional[tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Any:
        x = d
        if path is not None:
            if isinstance(path, list):
                xs = []
                for p in path:
                    try:
                        xs.append(search_.get_pathlike_key(x, p, keep_path=True))
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        continue
                if len(xs) == 0:
                    return NoResult
                x = merge_dicts(*xs)
            else:
                try:
                    x = search_.get_pathlike_key(x, path, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    return NoResult
        if source is not None:
            _template_context = flat_merge_dicts(
                {
                    "d": d,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            new_d = source.substitute(_template_context, eval_id="source")
            if checks.is_function(new_d):
                new_d = new_d(x)
        else:
            new_d = x
        return new_d


class SetAssetFunc(AssetFunc):
    """Asset function class for setting asset data with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.set`.

    Updates the asset data at specified paths with a given value, optionally applying transformations.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "set"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        value: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            value (Any): Value, function, or template to set.

                In templates, "i" represents the index, "d" the full data item, and "x" the targeted part.
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to set (e.g. "x.y[0].z").
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            template_context (KwargsLike): Additional context for template substitution.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")
        template_context = asset_cls.resolve_setting(
            template_context, "template_context", merge=True
        )
        template_context = flat_merge_dicts({"asset_cls": asset_cls}, template_context)

        if checks.is_function(value):
            if checks.is_builtin_func(value):
                value = RepFunc(lambda _value=value: _value)
            else:
                value = RepFunc(value)
        if path is not None:
            if isinstance(path, list):
                paths = [search_.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search_.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                value=value,
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        value: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        template_context: tp.KwargsLike = None,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search_.get_pathlike_key(x, p[:-1])
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            _template_context = flat_merge_dicts(
                {
                    "d": d,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            v = value.substitute(_template_context, eval_id="value")
            if checks.is_function(v):
                v = v(x)
            d = search_.set_pathlike_key(d, p, v, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class RemoveAssetFunc(AssetFunc):
    """Asset function class for removing an asset field from a knowledge asset with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.remove`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "remove"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            path (MaybeList[PathLikeKey]): Path or list of paths indicating the element(s) to remove.

                If an integer is provided, the entire data item at that index is removed.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")

        if isinstance(path, list):
            paths = [search_.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search_.resolve_pathlike_key(path)]
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            try:
                d = search_.remove_pathlike_key(d, p, make_copy=make_copy, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError) as e:
                if not skip_missing:
                    raise e
                continue
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class MoveAssetFunc(AssetFunc):
    """Asset function class for moving an asset field within a knowledge asset with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.move`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "move"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        path: tp.Union[tp.PathMoveDict, tp.MaybeList[tp.PathLikeKey]],
        new_path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            path (Union[PathMoveDict, MaybeList[PathLikeKey]]): Mapping or path(s) within the data item
                to move (e.g. "x.y[0].z").

                When provided as a dictionary, keys are source paths and values are the corresponding new tokens.
            new_path (Optional[MaybeList[PathLikeKey]]): Path(s) for the moved element(s)
                when `path` is not a dictionary.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")

        if new_path is None:
            checks.assert_instance_of(path, dict, arg_name="path")
            new_path = list(path.values())
            path = list(path.keys())
        if isinstance(path, list):
            paths = [search_.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search_.resolve_pathlike_key(path)]
        if isinstance(new_path, list):
            new_paths = [search_.resolve_pathlike_key(p) for p in new_path]
        else:
            new_paths = [search_.resolve_pathlike_key(new_path)]
        if len(paths) != len(new_paths):
            raise ValueError("Number of new paths must match number of paths")
        return (), {
            **dict(
                paths=paths,
                new_paths=new_paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        new_paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
    ) -> tp.Any:
        prev_keys = []
        for i, p in enumerate(paths):
            try:
                x = search_.get_pathlike_key(d, p)
                d = search_.remove_pathlike_key(d, p, make_copy=make_copy, prev_keys=prev_keys)
                d = search_.set_pathlike_key(
                    d, new_paths[i], x, make_copy=make_copy, prev_keys=prev_keys
                )
            except (KeyError, IndexError, AttributeError) as e:
                if not skip_missing:
                    raise e
                continue
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class RenameAssetFunc(MoveAssetFunc):
    """Asset function class for renaming an asset field in a knowledge asset with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.rename`.

    Converts the rename operation into a move operation with token replacement.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "rename"

    @classmethod
    def prepare(
        cls,
        path: tp.Union[tp.PathRenameDict, tp.MaybeList[tp.PathLikeKey]],
        new_token: tp.Optional[tp.MaybeList[tp.PathKeyToken]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            path (Union[PathMoveDict, MaybeList[PathLikeKey]]): Mapping or path(s) within the data item
                to move (e.g. "x.y[0].z").

                When provided as a dictionary, keys are source paths and values are the corresponding new tokens.
            new_path (Optional[MaybeList[PathLikeKey]]): Path(s) for the moved element(s)
                when `path` is not a dictionary.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")

        if new_token is None:
            checks.assert_instance_of(path, dict, arg_name="path")
            new_token = list(path.values())
            path = list(path.keys())
        if isinstance(path, list):
            paths = [search_.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search_.resolve_pathlike_key(path)]
        if isinstance(new_token, list):
            new_tokens = [search_.resolve_pathlike_key(t) for t in new_token]
        else:
            new_tokens = [search_.resolve_pathlike_key(new_token)]
        if len(paths) != len(new_tokens):
            raise ValueError("Number of new tokens must match number of paths")
        new_paths = []
        for i in range(len(paths)):
            if len(new_tokens[i]) != 1:
                raise ValueError("Exactly one token must be provided for each path")
            new_paths.append(paths[i][:-1] + new_tokens[i])
        return (), {
            **dict(
                paths=paths,
                new_paths=new_paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }


class ReorderAssetFunc(AssetFunc):
    """Asset function class for reordering asset data with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.reorder`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "reorder"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        new_order: tp.Union[str, tp.PathKeyTokens],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            new_order (Union[str, PathKeyTokens]): New order specification, which can be:

                * Sequence with tokens and an ellipsis (`...`) to preserve segments (e.g. ["a", ..., "z"]).
                * String "asc", "ascending", "desc", or "descending" indicating the sort order.
                * Function or template that generates an order using variables: `i` for the item index,
                    `d` for the data item, `x` for the value at the specified path, and field names for
                    individual fields.
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to reorder (e.g. "x.y[0].z").
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            template_context (KwargsLike): Additional context for template substitution.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")
        template_context = asset_cls.resolve_setting(
            template_context, "template_context", merge=True
        )
        template_context = flat_merge_dicts({"asset_cls": asset_cls}, template_context)

        if isinstance(new_order, str):
            if new_order.lower() in ("asc", "ascending"):
                new_order = lambda x: (
                    sorted(x)
                    if isinstance(x, dict)
                    else sorted(
                        range(len(x)),
                        key=x.__getitem__,
                    )
                )
            elif new_order.lower() in ("desc", "descending"):
                new_order = lambda x: (
                    sorted(x)
                    if isinstance(x, dict)
                    else sorted(
                        range(len(x)),
                        key=x.__getitem__,
                        reverse=True,
                    )
                )
        if isinstance(new_order, str):
            new_order = RepEval(new_order)
        elif checks.is_function(new_order):
            if checks.is_builtin_func(new_order):
                new_order = RepFunc(lambda _new_order=new_order: _new_order)
            else:
                new_order = RepFunc(new_order)
        if path is not None:
            if isinstance(path, list):
                paths = [search_.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search_.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                new_order=new_order,
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        new_order: tp.Union[tp.PathKeyTokens, tp.CustomTemplate],
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        template_context: tp.KwargsLike = None,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search_.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            if isinstance(new_order, CustomTemplate):
                _template_context = flat_merge_dicts(
                    {
                        "d": d,
                        "x": x,
                        **(x if isinstance(x, dict) else {}),
                    },
                    template_context,
                )
                _new_order = new_order.substitute(_template_context, eval_id="new_order")
                if checks.is_function(_new_order):
                    _new_order = _new_order(x)
            else:
                _new_order = new_order
            if isinstance(x, dict):
                x = reorder_dict(x, _new_order, skip_missing=skip_missing)
            else:
                if checks.is_namedtuple(x):
                    x = type(x)(*reorder_list(x, _new_order, skip_missing=skip_missing))
                else:
                    x = type(x)(reorder_list(x, _new_order, skip_missing=skip_missing))
            d = search_.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class QueryAssetFunc(AssetFunc):
    """Asset function class for querying asset data with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.query`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "query"

    _wrap: tp.ClassVar[tp.Optional[bool]] = False

    @classmethod
    def prepare(
        cls,
        expression: tp.CustomTemplateLike,
        template_context: tp.KwargsLike = None,
        return_type: tp.Optional[str] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            expression (CustomTemplateLike): Query expression or template.
            template_context (KwargsLike): Additional context for template substitution.
            return_type (Optional[str]): If "item", returns the matched data item; if "bool",
                returns a boolean indicating a match.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        template_context = asset_cls.resolve_setting(
            template_context, "template_context", merge=True
        )
        template_context = flat_merge_dicts({"asset_cls": asset_cls}, template_context)
        return_type = asset_cls.resolve_setting(return_type, "return_type")

        if isinstance(expression, str):
            expression = RepEval(expression)
        elif checks.is_function(expression):
            if checks.is_builtin_func(expression):
                expression = RepFunc(lambda _expression=expression: _expression)
            else:
                expression = RepFunc(expression)
        elif not isinstance(expression, CustomTemplate):
            raise TypeError(
                f"Expression must be a string, function, or template, not {type(expression)}"
            )
        return (), {
            **dict(
                expression=expression,
                template_context=template_context,
                return_type=return_type,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        expression: tp.CustomTemplate,
        template_context: tp.KwargsLike = None,
        return_type: str = "item",
    ) -> tp.Any:
        _template_context = flat_merge_dicts(
            {
                "d": d,
                "x": d,
                **search_.search_config,
                **(d if isinstance(d, dict) else {}),
            },
            template_context,
        )
        new_d = expression.substitute(_template_context, eval_id="expression")
        if checks.is_function(new_d):
            new_d = new_d(d)
        if return_type.lower() == "item":
            as_filter = True
        elif return_type.lower() == "bool":
            as_filter = False
        else:
            raise ValueError(f"Invalid return type: '{return_type}'")
        if as_filter and isinstance(new_d, bool):
            if new_d:
                return d
            return NoResult
        return new_d


class FindAssetFunc(AssetFunc):
    """Asset function class for searching in asset data with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.find`.

    Implements logic to locate assets using configurable search parameters.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "find"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Optional[tp.CustomTemplateLike] = None,
        in_dumps: tp.Optional[bool] = None,
        dump_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        return_type: tp.Optional[str] = None,
        return_path: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            target (MaybeList[Any]): Target value(s) or callable(s) to determine if a match occurs.

                Also supports negation using `vectorbtpro.utils.search_.Not`.
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to search (e.g. "x.y[0].z").
            per_path (Optional[bool]): If True, consider targets provided per path.
            find_all (Optional[bool]): Require all targets to be found when multiple targets are provided.
            keep_path (Optional[bool]): If True, returns results structured as nested dictionaries
                mirroring the specified path.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            source (Optional[CustomTemplateLike]): Template or function to preprocess the source data.
            in_dumps (Optional[bool]): If True, converts the entire data item to string for searching.
            dump_kwargs (KwargsLike): Keyword arguments for dumping structured data.

                See `vectorbtpro.utils.formatting.dump`.
            template_context (KwargsLike): Additional context for template substitution.
            return_type (Optional[str]): Indicates the return type: "item", "field", or "bool".
            return_path (Optional[bool]): Specifies whether to include the path in the returned result.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Keyword arguments distributed between `vectorbtpro.utils.search_.find_in_obj`
                and `vectorbtpro.utils.search_.find`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        per_path = asset_cls.resolve_setting(per_path, "per_path")
        find_all = asset_cls.resolve_setting(find_all, "find_all")
        keep_path = asset_cls.resolve_setting(keep_path, "keep_path")
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        in_dumps = asset_cls.resolve_setting(in_dumps, "in_dumps")
        dump_kwargs = asset_cls.resolve_setting(dump_kwargs, "dump_kwargs", merge=True)
        template_context = asset_cls.resolve_setting(
            template_context, "template_context", merge=True
        )
        template_context = flat_merge_dicts({"asset_cls": asset_cls}, template_context)
        return_type = asset_cls.resolve_setting(return_type, "return_type")
        return_path = asset_cls.resolve_setting(return_path, "return_path")

        if path is not None:
            if isinstance(path, list):
                path = [search_.resolve_pathlike_key(p) for p in path]
            else:
                path = search_.resolve_pathlike_key(path)
        if per_path:
            if not isinstance(target, list):
                target = [target]
                if isinstance(path, list):
                    target *= len(path)
            if not isinstance(path, list):
                path = [path]
                if isinstance(target, list):
                    path *= len(target)
            if len(target) != len(path):
                raise ValueError("Number of targets must match number of paths")
        if source is not None:
            if isinstance(source, str):
                source = RepEval(source)
            elif checks.is_function(source):
                if checks.is_builtin_func(source):
                    source = RepFunc(lambda _source=source: _source)
                else:
                    source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(
                    f"Source must be a string, function, or template, not {type(source)}"
                )
        dump_kwargs = DumpAssetFunc.resolve_dump_kwargs(**dump_kwargs)
        contains_arg_names = set(get_func_arg_names(search_.contains_in_obj))
        search_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in contains_arg_names}
        if "excl_types" not in search_kwargs:
            search_kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                path=path,
                per_path=per_path,
                find_all=find_all,
                keep_path=keep_path,
                skip_missing=skip_missing,
                source=source,
                in_dumps=in_dumps,
                dump_kwargs=dump_kwargs,
                search_kwargs=search_kwargs,
                template_context=template_context,
                return_type=return_type,
                return_path=return_path,
            ),
            **kwargs,
        }

    @classmethod
    def match_func(
        cls,
        k: tp.Optional[tp.Hashable],
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        find_all: bool = False,
        **kwargs,
    ) -> bool:
        """Return whether the given data item matches the specified target criteria used in `FindAssetFunc.call`.

        This function evaluates `target` against the provided data `d` using different strategies:

        * For strings, it utilizes `vectorbtpro.utils.search_.find` with `return_type="bool"`.
        * For other types, it performs equality comparisons.

        A `target` may be a callable that takes a key and a value, and returns a boolean or an instance of
        `vectorbtpro.utils.search_.Not` to indicate negation.

        Args:
            k (Optional[Hashable]): Key associated with the current element.
            d (Any): Data item to test.
            target (MaybeList[Any]): Target value(s) or callable(s) to determine if a match occurs.

                Also supports negation using `vectorbtpro.utils.search_.Not`.
            find_all (bool): Flag specifying if all targets should be evaluated.
            **kwargs: Keyword arguments for `vectorbtpro.utils.search_.find`.

        Returns:
            bool: True if the data item matches the target criteria, False otherwise.
        """
        if not isinstance(target, list):
            targets = [target]
        else:
            targets = target
        for target in targets:
            if isinstance(target, search_.Not):
                target = target.value
                negation = True
            else:
                negation = False
            if checks.is_function(target):
                if target(k, d):
                    if (negation and find_all) or (not negation and not find_all):
                        return not negation
                    continue
            elif d is target or d is None and target is None:
                if (negation and find_all) or (not negation and not find_all):
                    return not negation
                continue
            elif checks.is_bool(d) and checks.is_bool(target) or checks.is_number(d) and checks.is_number(target):
                if d == target:
                    if (negation and find_all) or (not negation and not find_all):
                        return not negation
                    continue
            elif isinstance(d, str) and isinstance(target, str):
                if search_.find(target, d, return_type="bool", **kwargs):
                    if (negation and find_all) or (not negation and not find_all):
                        return not negation
                    continue
            elif type(d) is type(target):
                try:
                    if d == target:
                        if (negation and find_all) or (not negation and not find_all):
                            return not negation
                        continue
                except Exception:
                    pass
            if (negation and not find_all) or (not negation and find_all):
                return negation
        if find_all:
            return True
        return False

    @classmethod
    def call(
        cls,
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: bool = True,
        find_all: bool = False,
        keep_path: bool = False,
        skip_missing: bool = False,
        source: tp.Optional[tp.CustomTemplate] = None,
        in_dumps: bool = False,
        dump_kwargs: tp.KwargsLike = None,
        search_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        return_type: str = "item",
        return_path: bool = False,
        **kwargs,
    ) -> tp.Any:
        if dump_kwargs is None:
            dump_kwargs = {}
        if search_kwargs is None:
            search_kwargs = {}
        if per_path:
            new_path_dct = {}
            new_list = []
            for i, p in enumerate(path):
                x = d
                try:
                    x = search_.get_pathlike_key(x, p, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
                if source is not None:
                    _template_context = flat_merge_dicts(
                        {
                            "d": d,
                            "x": x,
                            **(x if isinstance(x, dict) else {}),
                        },
                        template_context,
                    )
                    _x = source.substitute(_template_context, eval_id="source")
                    if checks.is_function(_x):
                        x = _x(x)
                    else:
                        x = _x
                if not isinstance(x, str) and in_dumps:
                    x = dump(x, **dump_kwargs)
                t = target[i]
                if return_type.lower() in ("item", "bool"):
                    if isinstance(t, search_.Not):
                        t = t.value
                        negation = True
                    else:
                        negation = False
                    if search_.contains_in_obj(
                        x,
                        cls.match_func,
                        target=t,
                        find_all=find_all,
                        **search_kwargs,
                        **kwargs,
                    ):
                        if negation:
                            if find_all:
                                return NoResult if return_type.lower() == "item" else False
                            continue
                        else:
                            if not find_all:
                                return d if return_type.lower() == "item" else True
                            continue
                    else:
                        if negation:
                            if not find_all:
                                return d if return_type.lower() == "item" else True
                            continue
                        else:
                            if find_all:
                                return NoResult if return_type.lower() == "item" else False
                            continue
                else:
                    path_dct = search_.find_in_obj(
                        x,
                        cls.match_func,
                        target=t,
                        find_all=find_all,
                        **search_kwargs,
                        **kwargs,
                    )
                    if len(path_dct) == 0:
                        if find_all:
                            return {} if return_path else []
                        continue
                    if isinstance(t, search_.Not):
                        raise TypeError("Target cannot be negated here")
                    if not isinstance(t, str):
                        raise ValueError("Target must be string")
                    for k, v in path_dct.items():
                        if not isinstance(v, str):
                            raise ValueError("Matched value must be string")
                        _return_type = "bool" if return_type.lower() == "field" else return_type
                        matches = search_.find(t, v, return_type=_return_type, **kwargs)
                        if return_path:
                            if k not in new_path_dct:
                                new_path_dct[k] = []
                            if return_type.lower() == "field":
                                if matches:
                                    new_path_dct[k].append(v)
                            else:
                                new_path_dct[k].extend(matches)
                        else:
                            if return_type.lower() == "field":
                                if matches:
                                    new_list.append(v)
                            else:
                                new_list.extend(matches)
            if return_type.lower() in ("item", "bool"):
                if find_all:
                    return d if return_type.lower() == "item" else True
                return NoResult if return_type.lower() == "item" else False
            else:
                if return_path:
                    return new_path_dct
                return new_list
        else:
            x = d
            if path is not None:
                if isinstance(path, list):
                    xs = []
                    for p in path:
                        try:
                            xs.append(search_.get_pathlike_key(x, p, keep_path=True))
                        except (KeyError, IndexError, AttributeError) as e:
                            if not skip_missing:
                                raise e
                            continue
                    if len(xs) == 0:
                        if return_type.lower() == "item":
                            return NoResult
                        if return_type.lower() == "bool":
                            return False
                        return {} if return_path else []
                    x = merge_dicts(*xs)
                else:
                    try:
                        x = search_.get_pathlike_key(x, path, keep_path=keep_path)
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        if return_type.lower() == "item":
                            return NoResult
                        if return_type.lower() == "bool":
                            return False
                        return {} if return_path else []
            if source is not None:
                _template_context = flat_merge_dicts(
                    {
                        "d": d,
                        "x": x,
                        **(x if isinstance(x, dict) else {}),
                    },
                    template_context,
                )
                _x = source.substitute(_template_context, eval_id="source")
                if checks.is_function(_x):
                    x = _x(x)
                else:
                    x = _x
            if not isinstance(x, str) and in_dumps:
                x = dump(x, **dump_kwargs)
            if return_type.lower() == "item":
                if search_.contains_in_obj(
                    x,
                    cls.match_func,
                    target=target,
                    find_all=find_all,
                    **search_kwargs,
                    **kwargs,
                ):
                    return d
                return NoResult
            elif return_type.lower() == "bool":
                return search_.contains_in_obj(
                    x,
                    cls.match_func,
                    target=target,
                    find_all=find_all,
                    **search_kwargs,
                    **kwargs,
                )
            else:
                path_dct = search_.find_in_obj(
                    x,
                    cls.match_func,
                    target=target,
                    find_all=find_all,
                    **search_kwargs,
                    **kwargs,
                )
                if len(path_dct) == 0:
                    return {} if return_path else []
                if not isinstance(target, list):
                    targets = [target]
                else:
                    targets = target
                new_path_dct = {}
                new_list = []
                for target in targets:
                    if isinstance(target, search_.Not):
                        raise TypeError("Target cannot be negated here")
                    if not isinstance(target, str):
                        raise ValueError("Target must be string")
                    for k, v in path_dct.items():
                        if not isinstance(v, str):
                            raise ValueError("Matched value must be string")
                        _return_type = "bool" if return_type.lower() == "field" else return_type
                        matches = search_.find(target, v, return_type=_return_type, **kwargs)
                        if return_path:
                            if k not in new_path_dct:
                                new_path_dct[k] = []
                            if return_type.lower() == "field":
                                if matches:
                                    new_path_dct[k].append(v)
                            else:
                                new_path_dct[k].extend(matches)
                        else:
                            if return_type.lower() == "field":
                                if matches:
                                    new_list.append(v)
                            else:
                                new_list.extend(matches)
                if return_path:
                    return new_path_dct
                return new_list


class FindReplaceAssetFunc(FindAssetFunc):
    """Asset function class for performing asset-level find and replace operations with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.find_replace`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "find_replace"

    @classmethod
    def prepare(
        cls,
        target: tp.Union[dict, tp.MaybeList[tp.Any]],
        replacement: tp.Optional[tp.MaybeList[tp.Any]] = None,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            target (Union[dict, List[Any]]): Data item(s) or pattern(s) to search for.
            replacement (Optional[List[Any]]): Replacement value(s) for matched occurrences.
            path (Optional[List[PathLikeKey]]): Specific path(s) within each data item to target.
            per_path (Optional[bool]): If True, consider targets and replacements provided per path.
            find_all (Optional[bool]): Require all targets to be found when multiple targets are provided.
            keep_path (Optional[bool]): If True, returns results structured as nested dictionaries
                mirroring the specified path.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Keyword arguments distributed between `vectorbtpro.utils.search_.find_in_obj`
                and `vectorbtpro.utils.search_.replace`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        per_path = asset_cls.resolve_setting(per_path, "per_path")
        find_all = asset_cls.resolve_setting(find_all, "find_all")
        keep_path = asset_cls.resolve_setting(keep_path, "keep_path")
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")

        if replacement is None:
            checks.assert_instance_of(target, dict, arg_name="path")
            replacement = list(target.values())
            target = list(target.keys())
        if path is not None:
            if isinstance(path, list):
                paths = [search_.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search_.resolve_pathlike_key(path)]
                if isinstance(target, list):
                    paths *= len(target)
                elif isinstance(replacement, list):
                    paths *= len(replacement)
        else:
            paths = [None]
            if isinstance(target, list):
                paths *= len(target)
            elif isinstance(replacement, list):
                paths *= len(replacement)
        if per_path:
            if not isinstance(target, list):
                target = [target] * len(paths)
            if not isinstance(replacement, list):
                replacement = [replacement] * len(paths)
            if len(target) != len(replacement) != len(paths):
                raise ValueError("Number of targets and replacements must match number of paths")
        find_arg_names = set(get_func_arg_names(search_.find_in_obj))
        find_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in find_arg_names}
        if "excl_types" not in find_kwargs:
            find_kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                replacement=replacement,
                paths=paths,
                per_path=per_path,
                find_all=find_all,
                keep_path=keep_path,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                find_kwargs=find_kwargs,
            ),
            **kwargs,
        }

    @classmethod
    def replace_func(
        cls,
        k: tp.Optional[tp.Hashable],
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        **kwargs,
    ) -> tp.Any:
        """Replace a value based on matching criteria.

        This method is used by `FindReplaceAssetFunc.call` to determine the replacement for
        a matched value. For string inputs, it applies `vectorbtpro.utils.search_.replace` for
        text substitution. For other types, it returns the replacement directly if the specified
        target condition is met. Both `target` and `replacement` may be callables that take a key and a value,
        where `target` returns a boolean indicating a match and `replacement` computes the new value.

        Args:
            k (Optional[Hashable]): Key associated with the current element.
            d (Any): Original value to evaluate for a match.
            target (MaybeList[Any]): Target value(s) or callable(s) to determine if a match occurs.
            replacement (MaybeList[Any]): Replacement value or callable to apply when a match is found.
            **kwargs: Keyword arguments for `vectorbtpro.utils.search_.replace`.

        Returns:
            Any: Resulting value after replacement if a match is found; otherwise, the original value.
        """
        if not isinstance(target, list):
            targets = [target]
        else:
            targets = target
        if not isinstance(replacement, list):
            replacements = [replacement]
            if len(targets) > 1 and len(replacements) == 1:
                replacements *= len(targets)
        else:
            replacements = replacement
        if len(targets) != len(replacements):
            raise ValueError("Number of targets must match number of replacements")
        for i, target in enumerate(targets):
            if isinstance(target, search_.Not):
                raise TypeError("Target cannot be negated here")
            replacement = replacements[i]
            if checks.is_function(replacement):
                replacement = replacement(k, d)
            if checks.is_function(target):
                if target(k, d):
                    return replacement
            elif d is target or d is None and target is None:
                return replacement
            elif checks.is_bool(d) and checks.is_bool(target) or checks.is_number(d) and checks.is_number(target):
                if d == target:
                    return replacement
            elif isinstance(d, str) and isinstance(target, str):
                d = search_.replace(target, replacement, d, **kwargs)
            elif type(d) is type(target):
                try:
                    if d == target:
                        return replacement
                except Exception:
                    pass
        return d

    @classmethod
    def call(
        cls,
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        paths: tp.List[tp.PathLikeKey],
        per_path: bool = True,
        find_all: bool = False,
        keep_path: bool = False,
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        find_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        if find_kwargs is None:
            find_kwargs = {}
        prev_keys = []
        found_all = True
        if find_all:
            for i, p in enumerate(paths):
                x = d
                if p is not None:
                    try:
                        x = search_.get_pathlike_key(x, p, keep_path=keep_path)
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        continue
                path_dct = search_.find_in_obj(
                    x,
                    cls.match_func,
                    target=target[i] if per_path else target,
                    find_all=find_all,
                    **find_kwargs,
                    **kwargs,
                )
                if len(path_dct) == 0:
                    found_all = False
                    break
        if found_all:
            for i, p in enumerate(paths):
                x = d
                if p is not None:
                    try:
                        x = search_.get_pathlike_key(x, p, keep_path=keep_path)
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        continue
                path_dct = search_.find_in_obj(
                    x,
                    cls.match_func,
                    target=target[i] if per_path else target,
                    find_all=find_all,
                    **find_kwargs,
                    **kwargs,
                )
                for k, v in path_dct.items():
                    if p is not None and not keep_path:
                        new_p = search_.combine_pathlike_keys(p, k, minimize=True)
                    else:
                        new_p = k
                    v = cls.replace_func(
                        k,
                        v,
                        target[i] if per_path else target,
                        replacement[i] if per_path else replacement,
                        **kwargs,
                    )
                    d = search_.set_pathlike_key(
                        d, new_p, v, make_copy=make_copy, prev_keys=prev_keys
                    )
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class FindRemoveAssetFunc(FindAssetFunc):
    """Asset function class for executing removal with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.find_remove`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "find_remove"

    @classmethod
    def prepare(
        cls,
        target: tp.Union[dict, tp.MaybeList[tp.Any]],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            target (Union[dict, MaybeList[Any]]): Value or mapping used to identify occurrences for removal.
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to search (e.g. "x.y[0].z").
            per_path (Optional[bool]): If True, consider targets provided per path.
            find_all (Optional[bool]): Require all targets to be found when multiple targets are provided.
            keep_path (Optional[bool]): If True, returns results structured as nested dictionaries
                mirroring the specified path.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Keyword arguments distributed between `vectorbtpro.utils.search_.find_in_obj`
                and `vectorbtpro.utils.search_.find`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        per_path = asset_cls.resolve_setting(per_path, "per_path")
        find_all = asset_cls.resolve_setting(find_all, "find_all")
        keep_path = asset_cls.resolve_setting(keep_path, "keep_path")
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search_.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search_.resolve_pathlike_key(path)]
                if isinstance(target, list):
                    paths *= len(target)
        else:
            paths = [None]
            if isinstance(target, list):
                paths *= len(target)
        if per_path:
            if not isinstance(target, list):
                target = [target] * len(paths)
            if len(target) != len(paths):
                raise ValueError("Number of targets must match number of paths")
        find_arg_names = set(get_func_arg_names(search_.find_in_obj))
        find_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in find_arg_names}
        if "excl_types" not in find_kwargs:
            find_kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                paths=paths,
                per_path=per_path,
                find_all=find_all,
                keep_path=keep_path,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                find_kwargs=find_kwargs,
            ),
            **kwargs,
        }

    @classmethod
    def is_empty_func(
        cls,
        k: tp.Optional[tp.Hashable],
        d: tp.Any,
        skip_keys: tp.Optional[tp.Container[tp.Hashable]] = None,
    ) -> bool:
        """Return whether the given object is empty.

        Args:
            d (Any): Data item to check for emptiness.

        Returns:
            bool: True if the data item is empty, False otherwise.
        """
        if skip_keys and k in skip_keys:
            return False
        if d is None:
            return True
        if checks.is_collection(d) and len(d) == 0:
            return True
        return False

    @classmethod
    def call(
        cls,
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        paths: tp.List[tp.PathLikeKey],
        per_path: bool = True,
        find_all: bool = False,
        keep_path: bool = False,
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        find_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        if find_kwargs is None:
            find_kwargs = {}
        prev_keys = []
        new_p_v_map = {}
        for i, p in enumerate(paths):
            x = d
            if p is not None:
                try:
                    x = search_.get_pathlike_key(x, p, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            path_dct = search_.find_in_obj(
                x,
                cls.match_func,
                target=target[i] if per_path else target,
                find_all=find_all,
                **find_kwargs,
                **kwargs,
            )
            if len(path_dct) == 0 and find_all:
                new_p_v_map = {}
                break
            for k, v in path_dct.items():
                if p is not None and not keep_path:
                    new_p = search_.combine_pathlike_keys(p, k, minimize=True)
                else:
                    new_p = k
                new_p_v_map[new_p] = v
        for new_p, v in new_p_v_map.items():
            d = search_.remove_pathlike_key(d, new_p, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class FlattenAssetFunc(AssetFunc):
    """Asset function class for performing flattening with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.flatten`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "flatten"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to flatten (e.g. "x.y[0].z").
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Keyword arguments for `vectorbtpro.utils.search_.flatten_obj`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search_.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search_.resolve_pathlike_key(path)]
        else:
            paths = [None]
        if "excl_types" not in kwargs:
            kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search_.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            x = search_.flatten_obj(x, **kwargs)
            d = search_.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class UnflattenAssetFunc(AssetFunc):
    """Asset function class for applying the unflatten transformation with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.unflatten`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "unflatten"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to unflatten (e.g. "x.y[0].z").
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Keyword arguments for `vectorbtpro.utils.search_.unflatten_obj`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        skip_missing = asset_cls.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset_cls.resolve_setting(make_copy, "make_copy")
        changed_only = asset_cls.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search_.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search_.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search_.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            x = search_.unflatten_obj(x, **kwargs)
            d = search_.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class DumpAssetFunc(AssetFunc):
    """Asset function class for performing the dump operation with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.dump`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "dump"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def resolve_dump_kwargs(
        cls,
        dump_engine: tp.Optional[str] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve and merge dumping-related keyword arguments based on asset
        settings and the provided dump engine.

        Args:
            dump_engine (Optional[str]): Name of the dump engine.

                See `vectorbtpro.utils.formatting.dump`.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments to merge with the resolved settings.

        Returns:
            Kwargs: Dictionary containing the resolved dumping-related keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        dump_engine = asset_cls.resolve_setting(dump_engine, "dump_engine")
        kwargs = asset_cls.resolve_setting(
            kwargs, f"dump_engine_kwargs.{dump_engine}", default={}, merge=True
        )
        return {"dump_engine": dump_engine, **kwargs}

    @classmethod
    def prepare(
        cls,
        source: tp.Optional[tp.CustomTemplateLike] = None,
        dump_engine: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            source (Optional[CustomTemplateLike]): Template or function to preprocess the source data.
            dump_engine (Optional[str]): Name of the dump engine.

                See `vectorbtpro.utils.formatting.dump`.
            template_context (KwargsLike): Additional context for template substitution.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Keyword arguments for `vectorbtpro.utils.formatting.dump`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        template_context = asset_cls.resolve_setting(
            template_context, "template_context", merge=True
        )
        template_context = flat_merge_dicts({"asset_cls": asset_cls}, template_context)
        dump_kwargs = cls.resolve_dump_kwargs(dump_engine=dump_engine, **kwargs)

        if source is not None:
            if isinstance(source, str):
                source = RepEval(source)
            elif checks.is_function(source):
                if checks.is_builtin_func(source):
                    source = RepFunc(lambda _source=source: _source)
                else:
                    source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(
                    f"Source must be a string, function, or template, not {type(source)}"
                )
        return (), {
            **dict(
                source=source,
                template_context=template_context,
            ),
            **dump_kwargs,
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        source: tp.Optional[CustomTemplate] = None,
        dump_engine: str = "nestedtext",
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        from vectorbtpro.utils.knowledge.chatting import (
            EmbeddedDocument,
            ScoredDocument,
            StoreDocument,
        )

        if source is not None:
            _template_context = flat_merge_dicts(
                {
                    "d": d,
                    "x": d,
                    **(d if isinstance(d, dict) else {}),
                },
                template_context,
            )
            new_d = source.substitute(_template_context, eval_id="source")
            if checks.is_function(new_d):
                new_d = new_d(d)
        else:
            new_d = d
        if isinstance(new_d, StoreDocument):
            return new_d.get_content()
        if isinstance(new_d, (EmbeddedDocument, ScoredDocument)):
            return new_d.document.get_content()
        return dump(new_d, dump_engine=dump_engine, **kwargs)


class ToDocsAssetFunc(AssetFunc):
    """Asset function class for converting asset data into document objects with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.to_documents`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "to_docs"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        document_cls: tp.Optional[tp.Type[tp.StoreDocument]] = None,
        template_context: tp.Union[tp.KwargsLike, tp.CustomTemplate] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **document_kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            document_cls (Optional[Type[StoreDocument]]): Document class to use for creating documents.

                Defaults to `vectorbtpro.utils.knowledge.chatting.TextDocument`.
            template_context (Union[KwargsLike, CustomTemplate]): Additional context for template substitution.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **document_kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.chatting.StoreData.from_data`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset

        document_cls = asset_cls.resolve_setting(document_cls, "document_cls")
        if document_cls is None:
            from vectorbtpro.utils.knowledge.chatting import TextDocument

            document_cls = TextDocument
        template_context = asset_cls.resolve_setting(
            template_context, "template_context", merge=True
        )
        template_context = flat_merge_dicts({"asset_cls": asset_cls}, template_context)

        document_kwargs = {}
        for k, v in document_cls.fields_dict.items():
            if v.default is not MISSING:
                if k in document_kwargs or asset_cls.has_setting(k, sub_path="document_kwargs"):
                    document_kwargs[k] = asset_cls.resolve_setting(
                        document_kwargs.get(k),
                        k,
                        sub_path="document_kwargs",
                        merge=isinstance(v.default, attr.Factory) and v.default.factory is dict,
                    )
                    if k == "template_context":
                        document_kwargs[k] = merge_dicts(template_context, document_kwargs[k])
                    if k == "dump_kwargs":
                        document_kwargs[k] = DumpAssetFunc.resolve_dump_kwargs(**document_kwargs[k])
        document_kwargs = substitute_templates(
            document_kwargs, template_context, eval_id="document_kwargs", strict=False
        )
        return (), {
            **dict(document_cls=document_cls),
            **document_kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        document_cls: tp.Optional[tp.Type[tp.StoreDocument]] = None,
        template_context: tp.KwargsLike = None,
        **document_kwargs,
    ) -> tp.Any:
        if document_cls is None:
            from vectorbtpro.utils.knowledge.chatting import TextDocument

            document_cls = TextDocument

        _template_context = flat_merge_dicts(
            {
                "d": d,
                "x": d,
                **(d if isinstance(d, dict) else {}),
            },
            template_context,
        )
        document_kwargs = substitute_templates(
            document_kwargs, _template_context, eval_id="document_kwargs"
        )
        return document_cls.from_data(d, template_context=_template_context, **document_kwargs)


class SplitTextAssetFunc(AssetFunc):
    """Asset function class for splitting text with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.split_text`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "split_text"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        text_path: tp.Optional[tp.PathLikeKey] = None,
        document_cls: tp.Optional[tp.Type[tp.StoreDocument]] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **split_text_kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            text_path (Optional[PathLikeKey]): Path specifying the location of the text content.
            document_cls (Optional[Type[StoreDocument]]): Document class to use for creating documents.

                Defaults to `vectorbtpro.utils.knowledge.chatting.TextDocument`.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **split_text_kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.chatting.split_text`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        from vectorbtpro.utils.knowledge.chatting import resolve_text_splitter

        text_path = asset_cls.resolve_setting(text_path, "text_path", sub_path="document_kwargs")
        split_text_kwargs = asset_cls.resolve_setting(
            split_text_kwargs, "split_text_kwargs", sub_path="document_kwargs", merge=True
        )

        text_splitter = split_text_kwargs.pop("text_splitter", None)
        text_splitter = resolve_text_splitter(text_splitter=text_splitter)
        if isinstance(text_splitter, type):
            text_splitter = text_splitter(**split_text_kwargs)
        elif split_text_kwargs:
            text_splitter = text_splitter.replace(**split_text_kwargs)
        return (), {
            **dict(
                text_path=text_path,
                document_cls=document_cls,
                text_splitter=text_splitter,
            ),
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        text_path: tp.Optional[tp.PathLikeKey] = None,
        document_cls: tp.Optional[tp.Type[tp.StoreDocument]] = None,
        **split_text_kwargs,
    ) -> tp.Any:
        if document_cls is None:
            from vectorbtpro.utils.knowledge.chatting import TextDocument

            document_cls = TextDocument

        document = document_cls.from_data(
            d, text_path=text_path, split_text_kwargs=split_text_kwargs
        )
        return [document_chunk.data for document_chunk in document.split()]


# ############# Reduce classes ############# #


class ReduceAssetFunc(AssetFunc):
    """Abstract class defining an asset function for reducing data with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.reduce`."""

    _wrap: tp.ClassVar[tp.Optional[bool]] = False

    _initializer: tp.ClassVar[tp.Optional[tp.Any]] = None

    @classmethod
    def call(cls, d1: tp.Any, d2: tp.Any, *args, **kwargs) -> tp.Any:
        raise NotImplementedError

    @classmethod
    def prepare_and_call(cls, d1: tp.Any, d2: tp.Any, *args, **kwargs) -> tp.Any:
        """Prepare arguments and invoke the asset function for reducing data.

        Args:
            d1: First input data.

                Upon the first call, this will be either the first data item or the initializer.
                Later, it will be replaced with the output of the previous call.
            d2: Second input data.

                This will be the next data item to be processed.
            *args: Positional arguments for `ReduceAssetFunc.prepare` and ultimately to `ReduceAssetFunc.call`.
            **kwargs: Keyword arguments for `ReduceAssetFunc.prepare` and ultimately to `ReduceAssetFunc.call`.

        Returns:
            Any: Result returned by the asset function.
        """
        args, kwargs = cls.prepare(*args, **kwargs)
        return cls.call(d1, d2, *args, **kwargs)


class CollectAssetFunc(ReduceAssetFunc):
    """Asset function class for collecting data with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.collect`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "collect"

    _initializer: tp.ClassVar[tp.Optional[tp.Any]] = {}

    @classmethod
    def prepare(
        cls,
        sort_keys: tp.Optional[bool] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
            sort_keys (Optional[bool]): Whether to sort the keys.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        if asset_cls is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset_cls = KnowledgeAsset
        sort_keys = asset_cls.resolve_setting(sort_keys, "sort_keys")

        return (), {**dict(sort_keys=sort_keys), **kwargs}

    @classmethod
    def sort_key(cls, k: tp.Any) -> tuple:
        """Return a tuple used as a sorting key.

        Args:
            k (Any): Key to be sorted.

        Returns:
            tuple: Tuple used for sorting, where the first element is 0 if `k` is a string,
                otherwise 1, and the second element is `k` itself.
        """
        return (0, k) if isinstance(k, str) else (1, k)

    @classmethod
    def call(cls, d1: tp.Any, d2: tp.Any, sort_keys: bool = False) -> tp.Any:
        if isinstance(d1, list):
            d1 = {i: v for i, v in enumerate(d1)}
        if isinstance(d2, list):
            d2 = {i: v for i, v in enumerate(d2)}
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            raise TypeError(
                f"Data items must be either dicts or lists, not {type(d1)} and {type(d2)}"
            )
        new_d1 = dict(d1)
        for k1 in d1:
            if k1 not in new_d1:
                new_d1[k1] = [d1[k1]]
            if k1 in d2:
                new_d1[k1].append(d2[k1])
        for k2 in d2:
            if k2 not in new_d1:
                new_d1[k2] = [d2[k2]]
        if sort_keys:
            return dict(sorted(new_d1.items(), key=lambda x: cls.sort_key(x[0])))
        return new_d1


class MergeDictsAssetFunc(ReduceAssetFunc):
    """Asset function class for merging dictionaries with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.merge_dicts`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "merge_dicts"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    _initializer: tp.ClassVar[tp.Optional[tp.Any]] = {}

    @classmethod
    def call(cls, d1: tp.Any, d2: tp.Any, **kwargs) -> tp.Any:
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            raise TypeError(f"Data items must be dicts, not {type(d1)} and {type(d2)}")
        return merge_dicts(d1, d2, **kwargs)


class MergeListsAssetFunc(ReduceAssetFunc):
    """Asset function class for merging lists with
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.merge_lists`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "merge_lists"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    _initializer: tp.ClassVar[tp.Optional[tp.Any]] = []

    @classmethod
    def call(cls, d1: tp.Any, d2: tp.Any) -> tp.Any:
        if not isinstance(d1, list) or not isinstance(d2, list):
            raise TypeError(f"Data items must be lists, not {type(d1)} and {type(d2)}")
        return d1 + d2
