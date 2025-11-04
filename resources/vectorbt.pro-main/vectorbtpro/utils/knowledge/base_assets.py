# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing base classes for managing knowledge assets.

See `vectorbtpro.utils.knowledge` for the toy dataset.
"""

import hashlib
import json
import re
import textwrap
from collections.abc import MutableSequence
from functools import partial
from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Configured, flat_merge_dicts, merge_dicts
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.execution import NoResult, Task, execute
from vectorbtpro.utils.knowledge.chatting import RankContextable
from vectorbtpro.utils.module_ import get_caller_qualname
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.path_ import check_mkdir, dir_tree_from_paths, remove_dir
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.pickling import decompress, dumps, load, load_bytes, save
from vectorbtpro.utils.search_ import flatten_obj, unflatten_obj
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "AssetCacheManager",
    "KnowledgeAsset",
]


asset_cache: tp.Dict[tp.Hashable, "KnowledgeAsset"] = {}
"""Cache for storing knowledge assets, keyed by a unique identifier."""


class AssetCacheManager(Configured):
    """Class for managing cached knowledge assets.

    Args:
        persist_cache (Optional[bool]): Whether to persist the cache to disk.
        cache_dir (Optional[PathLike]): Directory for saving knowledge assets.
        cache_mkdir_kwargs (KwargsLike): Keyword arguments for cache directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        clear_cache (Optional[bool]): Remove the cache directory before operation if True.
        max_cache_count (Optional[int]): Maximum number of assets to retain, evicting older ones.
        save_cache_kwargs (KwargsLike): Keyword arguments for saving assets to disk.

            See `vectorbtpro.utils.pickling.save`.
        load_cache_kwargs (KwargsLike): Keyword arguments for loading assets from disk.

            See `vectorbtpro.utils.pickling.load`.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge`.
    """

    _settings_path: tp.SettingsPath = "knowledge"

    _specializable: tp.ClassVar[bool] = False

    _extendable: tp.ClassVar[bool] = False

    def __init__(
        self,
        persist_cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Optional[bool] = None,
        max_cache_count: tp.Optional[int] = None,
        save_cache_kwargs: tp.KwargsLike = None,
        load_cache_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            persist_cache=persist_cache,
            cache_dir=cache_dir,
            cache_mkdir_kwargs=cache_mkdir_kwargs,
            clear_cache=clear_cache,
            max_cache_count=max_cache_count,
            save_cache_kwargs=save_cache_kwargs,
            load_cache_kwargs=load_cache_kwargs,
            template_context=template_context,
            **kwargs,
        )

        persist_cache = self.resolve_setting(persist_cache, "cache")
        cache_dir = self.resolve_setting(cache_dir, "asset_cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(
            cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True
        )
        clear_cache = self.resolve_setting(clear_cache, "clear_cache")
        max_cache_count = self.resolve_setting(max_cache_count, "max_cache_count")
        save_cache_kwargs = self.resolve_setting(save_cache_kwargs, "save_cache_kwargs", merge=True)
        load_cache_kwargs = self.resolve_setting(load_cache_kwargs, "load_cache_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        if isinstance(cache_dir, CustomTemplate):
            asset_cache_dir = cache_dir
            cache_dir = self.get_setting("cache_dir")
            if isinstance(cache_dir, CustomTemplate):
                cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
            template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
            asset_cache_dir = asset_cache_dir.substitute(
                template_context, eval_id="asset_cache_dir"
            )
            cache_dir = asset_cache_dir
        cache_dir = Path(cache_dir)
        if cache_dir.exists():
            if clear_cache:
                remove_dir(cache_dir, missing_ok=True, with_contents=True)
        check_mkdir(cache_dir, **cache_mkdir_kwargs)

        self._persist_cache = persist_cache
        self._cache_dir = cache_dir
        self._max_cache_count = max_cache_count
        self._save_cache_kwargs = save_cache_kwargs
        self._load_cache_kwargs = load_cache_kwargs
        self._template_context = template_context

    @property
    def persist_cache(self) -> bool:
        """Whether to persist the cache to disk.

        Returns:
            bool: True if cache persistence is enabled, otherwise False.
        """
        return self._persist_cache

    @property
    def cache_dir(self) -> tp.Path:
        """Directory path for storing cached assets.

        Returns:
            Path: Path of the cache directory.
        """
        return self._cache_dir

    @property
    def max_cache_count(self) -> tp.Optional[int]:
        """Maximum number of assets to retain, evicting older ones.

        Returns:
            Optional[int]: Maximum number of assets to retain in the cache.
        """
        return self._max_cache_count

    @property
    def save_cache_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for saving assets to disk.

        See `vectorbtpro.utils.pickling.save`.

        Returns:
            Kwargs: Keyword arguments used for saving assets to disk.
        """
        return self._save_cache_kwargs

    @property
    def load_cache_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for loading assets from disk.

        See `vectorbtpro.utils.pickling.load`.

        Returns:
            Kwargs: Keyword arguments used for loading assets from disk.
        """
        return self._load_cache_kwargs

    @classmethod
    def generate_cache_key(cls, **kwargs) -> str:
        """Generate a cache key based on the current VectorBT version, asset settings, and provided parameters.

        Args:
            **kwargs: Additional parameters contributing to the cache key.

        Returns:
            str: MD5 hash representing the cache key.
        """
        from vectorbtpro._version import __version__

        bytes_ = b""
        bytes_ += dumps(kwargs)
        bytes_ += dumps(cls.get_settings())
        bytes_ += dumps(__version__)
        return hashlib.md5(bytes_).hexdigest()

    def load_asset(self, cache_key: str) -> tp.Optional[tp.MaybeKnowledgeAsset]:
        """Load a knowledge asset from the cache.

        Args:
            cache_key (str): Unique identifier for the cached asset.

        Returns:
            Optional[MaybeKnowledgeAsset]: Loaded knowledge asset if found, otherwise None.
        """
        if cache_key in asset_cache:
            return asset_cache[cache_key]
        asset_cache_file = self.cache_dir / cache_key
        if asset_cache_file.exists():
            return load(asset_cache_file, **self.load_cache_kwargs)

    def cleanup_cache_dir(self) -> None:
        """Remove older cached assets, retaining only the most recent ones based on modification time.

        Returns:
            None
        """
        if not self.max_cache_count:
            return
        files = [f for f in self.cache_dir.iterdir() if f.is_file()]
        if len(files) <= self.max_cache_count:
            return
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        files_to_delete = files[self.max_cache_count :]
        for file_path in files_to_delete:
            file_path.unlink(missing_ok=True)

    def save_asset(self, asset: tp.MaybeKnowledgeAsset, cache_key: str) -> tp.Optional[tp.Path]:
        """Save a knowledge asset to the cache.

        Caches the asset in memory and, if persistence is enabled, writes it to disk.

        Args:
            asset (MaybeKnowledgeAsset): Knowledge asset to cache.
            cache_key (str): Unique identifier for the cached asset.

        Returns:
            Optional[Path]: File path where the asset was saved if persistence is enabled, otherwise None.
        """
        asset_cache[cache_key] = asset
        if self.persist_cache:
            asset_cache_file = self.cache_dir / cache_key
            path = save(asset, path=asset_cache_file, **self.save_cache_kwargs)
            self.cleanup_cache_dir()
            return path


KnowledgeAssetT = tp.TypeVar("KnowledgeAssetT", bound="KnowledgeAsset")


class MetaKnowledgeAsset(type(Configured), type(MutableSequence)):
    """Metaclass for the `KnowledgeAsset` class."""

    pass


class KnowledgeAsset(RankContextable, Configured, MutableSequence, metaclass=MetaKnowledgeAsset):
    """Class for working with a knowledge asset.

    This class behaves like a mutable sequence.

    Args:
        data (Optional[List[Any]]): List of data items for the asset.

            If more than one item is provided, the asset is not considered a single item.
        single_item (bool): Indicates whether the asset holds a single data item.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge`.
    """

    _settings_path: tp.SettingsPath = "knowledge"

    def __init__(
        self, data: tp.Optional[tp.List[tp.Any]] = None, single_item: bool = True, **kwargs
    ) -> None:
        if data is None:
            data = []
        if not isinstance(data, list):
            data = [data]
        else:
            data = list(data)
        if len(data) > 1:
            single_item = False

        Configured.__init__(
            self,
            data=data,
            single_item=single_item,
            **kwargs,
        )

        self._data = data
        self._single_item = single_item

    @hybrid_method
    def combine(
        cls_or_self: tp.MaybeType[KnowledgeAssetT],
        *objs: tp.MaybeSequence[KnowledgeAssetT],
        **kwargs,
    ) -> KnowledgeAssetT:
        """Combine multiple `KnowledgeAsset` instances into one.

        Args:
            *objs (MaybeSequence[KnowledgeAsset]): (Additional) `KnowledgeAsset` instances to combine.
            **kwargs: Keyword arguments for `KnowledgeAsset.merge_lists` or
                `KnowledgeAsset.merge_dicts` or `KnowledgeAsset`.

        Returns:
            KnowledgeAsset: New asset containing merged data.

        Examples:
            ```pycon
            >>> asset1 = asset[[0, 1]]
            >>> asset2 = asset[[2, 3]]
            >>> asset1.combine(asset2).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]
            ```
        """
        if not isinstance(cls_or_self, type) and len(objs) == 0:
            if isinstance(cls_or_self[0], list):
                return cls_or_self.merge_lists(**kwargs)
            if isinstance(cls_or_self[0], dict):
                return cls_or_self.merge_dicts(**kwargs)
            raise ValueError("Cannot determine type of data items. Use merge_lists or merge_dicts.")
        elif not isinstance(cls_or_self, type) and len(objs) > 0:
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self

        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, KnowledgeAsset):
                raise TypeError("Each object to be combined must be an instance of KnowledgeAsset")
        new_data = []
        new_single_item = True
        for obj in objs:
            new_data.extend(obj.data)
            if not obj.single_item:
                new_single_item = False
        kwargs = cls_or_self.resolve_merge_kwargs(
            *[obj.config for obj in objs],
            single_item=new_single_item,
            data=new_data,
            **kwargs,
        )
        return cls(**kwargs)

    @hybrid_method
    def merge(
        cls_or_self: tp.MaybeType[KnowledgeAssetT],
        *objs: tp.MaybeSequence[KnowledgeAssetT],
        flatten_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Merge multiple `KnowledgeAsset` instances or the data items of a single instance.

        When called as a class method or instance method with additional objects, combine the provided
        `KnowledgeAsset` instances. When called as an instance method without additional objects,
        merge the data items within the instance.

        Args:
            *objs (MaybeSequence[KnowledgeAsset]): (Additional) `KnowledgeAsset` instances to merge.
            flatten_kwargs (KwargsLike): Keyword arguments for flattening data items.

                See `vectorbtpro.utils.search_.flatten_obj`.
            **kwargs: Keyword arguments for `KnowledgeAsset.merge_lists` or
                `KnowledgeAsset.merge_dicts` or `KnowledgeAsset`.

        Returns:
            KnowledgeAsset: New asset containing merged data.

        Examples:
            ```pycon
            >>> asset1 = asset.select(["s"])
            >>> asset2 = asset.select(["b", "d2"])
            >>> asset1.merge(asset2).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}}]
            ```
        """
        if not isinstance(cls_or_self, type) and len(objs) == 0:
            if isinstance(cls_or_self[0], list):
                return cls_or_self.merge_lists(**kwargs)
            if isinstance(cls_or_self[0], dict):
                return cls_or_self.merge_dicts(**kwargs)
            raise ValueError("Cannot determine type of data items. Use merge_lists or merge_dicts.")
        elif not isinstance(cls_or_self, type) and len(objs) > 0:
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self

        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, KnowledgeAsset):
                raise TypeError("Each object to be merged must be an instance of KnowledgeAsset")

        if flatten_kwargs is None:
            flatten_kwargs = {}
        if "annotate_all" not in flatten_kwargs:
            flatten_kwargs["annotate_all"] = True
        if "excl_types" not in flatten_kwargs:
            flatten_kwargs["excl_types"] = (tuple, set, frozenset)
        max_items = 1
        new_single_item = True
        for obj in objs:
            obj_data = obj.data
            if len(obj_data) > max_items:
                max_items = len(obj_data)
            if not obj.single_item:
                new_single_item = False
        flat_data = []
        for obj in objs:
            obj_data = obj.data
            if len(obj_data) == 1:
                obj_data = [obj_data] * max_items
            flat_obj_data = list(map(lambda x: flatten_obj(x, **flatten_kwargs), obj_data))
            flat_data.append(flat_obj_data)
        new_data = []
        for flat_dcts in zip(*flat_data):
            merged_flat_dct = flat_merge_dicts(*flat_dcts)
            new_data.append(unflatten_obj(merged_flat_dct))
        kwargs = cls_or_self.resolve_merge_kwargs(
            *[obj.config for obj in objs],
            single_item=new_single_item,
            data=new_data,
            **kwargs,
        )
        return cls(**kwargs)

    @classmethod
    def from_json_file(
        cls: tp.Type[KnowledgeAssetT],
        path: tp.PathLike,
        compression: tp.CompressionLike = None,
        decompress_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Build a `KnowledgeAsset` instance from a JSON file.

        Args:
            path (PathLike): Path to the JSON file.
            compression (CompressionLike): Compression algorithm.

                See `vectorbtpro.utils.pickling.compress`.
            decompress_kwargs (KwargsLike): Keyword arguments for decompression.
            **kwargs: Keyword arguments for `KnowledgeAsset`.

        Returns:
            KnowledgeAsset: New asset populated with data from the JSON file.

        See:
            `vectorbtpro.utils.pickling.load_bytes`
        """
        bytes_ = load_bytes(path, compression=compression, decompress_kwargs=decompress_kwargs)
        json_str = bytes_.decode("utf-8")
        return cls(data=json.loads(json_str), **kwargs)

    @classmethod
    def from_json_bytes(
        cls: tp.Type[KnowledgeAssetT],
        bytes_: bytes,
        compression: tp.CompressionLike = None,
        decompress_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Build a `KnowledgeAsset` instance from JSON bytes.

        Args:
            bytes_ (bytes): Byte stream containing the JSON object.
            compression (CompressionLike): Compression algorithm.

                See `vectorbtpro.utils.pickling.compress`.
            decompress_kwargs (KwargsLike): Keyword arguments for decompression.
            **kwargs: Keyword arguments for `KnowledgeAsset`.

        Returns:
            KnowledgeAsset: New asset containing data from the JSON bytes.

        See:
            `vectorbtpro.utils.pickling.decompress`
        """
        if decompress_kwargs is None:
            decompress_kwargs = {}
        bytes_ = decompress(bytes_, compression=compression, **decompress_kwargs)
        json_str = bytes_.decode("utf-8")
        return cls(data=json.loads(json_str), **kwargs)

    @property
    def data(self) -> tp.List[tp.Any]:
        """List of data items in the asset.

        Returns:
            List[Any]: Data items contained in the asset.
        """
        return self._data

    @property
    def single_item(self) -> bool:
        """Whether the asset holds a single item.

        Returns:
            bool: True if the asset contains a single item, otherwise False.
        """
        return self._single_item

    def modify_data(self, data: tp.List[tp.Any]) -> None:
        """Update the asset's data in place and synchronize its configuration.

        Returns:
            None
        """
        if len(data) > 1:
            single_item = False
        else:
            single_item = self.single_item
        self._data = data
        self._single_item = single_item
        self.update_config(data=data, single_item=single_item)

    # ############# Item methods ############# #

    def get_items(self, index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]]) -> tp.Any:
        """Get one or more data items from the asset.

        Args:
            index (Union[int, slice, Iterable[Union[bool, int]]]): Index specifying the item(s) to retrieve.

                A boolean iterable selects items by truth value, an integer iterable selects specific
                positions, a slice selects a range, and an integer selects a single item.

        Returns:
            Union[Any, KnowledgeAsset]: Selected data element if an integer is provided,
                or a new asset containing the extracted items otherwise.
        """
        if checks.is_complex_iterable(index):
            if all(checks.is_bool(i) for i in index):
                index = list(index)
                if len(index) != len(self.data):
                    raise IndexError("Boolean index must have the same length as data")
                return self.replace(data=[item for item, flag in zip(self.data, index) if flag])
            if all(checks.is_int(i) for i in index):
                return self.replace(data=[self.data[i] for i in index])
            raise TypeError("Index must contain all integers or all booleans")
        if isinstance(index, slice):
            return self.replace(data=self.data[index])
        return self.data[index]

    def set_items(
        self: KnowledgeAssetT,
        index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]],
        value: tp.Any,
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Set one or more data items in the asset.

        Args:
            index (Union[int, slice, Iterable[Union[bool, int]]]): Index specifying the item(s) to update.

                A boolean iterable selects items by truth value, an integer iterable selects specific
                positions, a slice selects a range, and an integer selects a single item.
            value (Any): New value or iterable of values to assign.
            inplace (bool): If True, modify the asset in place.

        Returns:
            Optional[KnowledgeAsset]: New asset with updated data, or None if modified in place.
        """
        new_data = list(self.data)
        if checks.is_complex_iterable(index):
            index = list(index)
            if all(checks.is_bool(i) for i in index):
                if len(index) != len(new_data):
                    raise IndexError("Boolean index must have the same length as data")
                if checks.is_complex_iterable(value):
                    value = list(value)
                    if len(value) == len(index):
                        for i, (b, v) in enumerate(zip(index, value)):
                            if b:
                                new_data[i] = v
                    else:
                        num_true = sum(index)
                        if len(value) != num_true:
                            raise ValueError(
                                f"Attempting to assign {len(value)} values to {num_true} targets"
                            )
                        it = iter(value)
                        for i, b in enumerate(index):
                            if b:
                                new_data[i] = next(it)
                else:
                    for i, b in enumerate(index):
                        if b:
                            new_data[i] = value
            elif all(checks.is_int(i) for i in index):
                if checks.is_complex_iterable(value):
                    value = list(value)
                    if len(value) != len(index):
                        raise ValueError(
                            f"Attempting to assign {len(value)} values to {len(index)} targets"
                        )
                    for i, v in zip(index, value):
                        new_data[i] = v
                else:
                    for i in index:
                        new_data[i] = value
            else:
                raise TypeError("Index must contain all integers or all booleans")
        else:
            new_data[index] = value
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def delete_items(
        self: KnowledgeAssetT,
        index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]],
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Delete one or more data items from the asset.

        Args:
            index (Union[int, slice, Iterable[Union[bool, int]]]): Index specifying the item(s) to remove.

                A boolean iterable selects items by truth value, an integer iterable selects specific
                positions, a slice selects a range, and an integer selects a single item.
            inplace (bool): If True, delete the items in place.

        Returns:
            Optional[KnowledgeAsset]: New asset with the selected items removed,
                or None if modified in place.
        """
        new_data = list(self.data)
        if checks.is_complex_iterable(index):
            if all(checks.is_bool(i) for i in index):
                index = list(index)
                if len(index) != len(new_data):
                    raise IndexError("Boolean index must have the same length as data")
                new_data = [item for item, flag in zip(new_data, index) if not flag]
            elif all(checks.is_int(i) for i in index):
                indices_to_remove = set(index)
                max_index = len(new_data) - 1
                for i in indices_to_remove:
                    if not -len(new_data) <= i <= max_index:
                        raise IndexError(f"Index {i} out of range")
                new_data = [item for i, item in enumerate(new_data) if i not in indices_to_remove]
            else:
                raise TypeError("Index must contain all integers or all booleans")
        else:
            del new_data[index]
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def append_item(
        self: KnowledgeAssetT,
        d: tp.Any,
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Append a new data item to the asset.

        Args:
            d (Any): Data item to append.
            inplace (bool): If True, modify the asset in place.

        Returns:
            Optional[KnowledgeAsset]: New asset with the appended item, or None if modified in place.
        """
        new_data = list(self.data)
        new_data.append(d)
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def extend_items(
        self: KnowledgeAssetT,
        data: tp.Iterable[tp.Any],
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Extend the asset with additional data items.

        Args:
            data (Iterable[Any]): Iterable of data items to append.
            inplace (bool): If True, modify the asset in place.

        Returns:
            Optional[KnowledgeAsset]: New asset with extended data, or None if modified in place.
        """
        new_data = list(self.data)
        new_data.extend(data)
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def remove_empty(self, inplace: bool = False) -> tp.Optional[KnowledgeAssetT]:
        """Remove empty data items from the asset.

        Args:
            inplace (bool): If True, remove empty items in place.

        Returns:
            Optional[KnowledgeAsset]: New asset with empty items removed,
                or None if modified in place.
        """
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc

        new_data = [d for d in self.data if not FindRemoveAssetFunc.is_empty_func(None, d)]
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def unique(
        self: KnowledgeAssetT,
        *args,
        keep: str = "first",
        inplace: bool = False,
        **kwargs,
    ) -> tp.Optional[KnowledgeAssetT]:
        """De-duplicate data items using keys obtained via `KnowledgeAsset.get`.

        Args:
            *args: Positional arguments for `KnowledgeAsset.get`.
            keep (str): Indicates which duplicate to retain; valid options are "first" or "last".
            inplace (bool): If True, de-duplicate the data in place.
            **kwargs: Keyword arguments for `KnowledgeAsset.get`.

        Returns:
            Optional[KnowledgeAsset]: New asset with duplicates removed,
                or None if modified in place.

        Examples:
            ```pycon
            >>> asset.unique("b").get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]
            ```
        """
        keys = self.get(*args, **kwargs)
        if keep.lower() == "first":
            seen = set()
            new_data = []
            for key, item in zip(keys, self.data):
                if key not in seen:
                    seen.add(key)
                    new_data.append(item)
        elif keep.lower() == "last":
            seen = set()
            new_data_reversed = []
            for key, item in zip(reversed(keys), reversed(self.data)):
                if key not in seen:
                    seen.add(key)
                    new_data_reversed.append(item)
            new_data = list(reversed(new_data_reversed))
        else:
            raise ValueError(f"Invalid keep option: '{keep}'")
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def sort(
        self: KnowledgeAssetT,
        *args,
        keys: tp.Optional[tp.Iterable[tp.Key]] = None,
        ascending: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Sort data items based on keys extracted via `KnowledgeAsset.get`.

        Args:
            *args: Positional arguments for `KnowledgeAsset.get`.
            keys (Optional[Iterable[Key]]): Iterable of keys to sort by.

                If None, keys are obtained by calling `KnowledgeAsset.get`.
            ascending (bool): True for ascending order, False for descending.
            inplace (bool): If True, sort the data in place.
            **kwargs: Keyword arguments for `KnowledgeAsset.get`.

        Returns:
            Optional[KnowledgeAsset]: New asset with sorted data,
                or None if sorted in place.

        Examples:
            ```pycon
            >>> asset.sort("d2.c").get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]
            ```
        """
        if keys is None:
            keys = self.get(*args, **kwargs)
        new_data = [
            x for _, x in sorted(zip(keys, self.data), key=lambda x: x[0], reverse=not ascending)
        ]
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def shuffle(
        self: KnowledgeAssetT,
        seed: tp.Optional[int] = None,
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Shuffle the asset's data items randomly.

        Args:
            seed (Optional[int]): Random seed for deterministic output.
            inplace (bool): If True, shuffle the data in place; otherwise, return a new asset instance.

        Returns:
            KnowledgeAsset: New asset with shuffled data if `inplace` is False; otherwise, None.
        """
        import random

        if seed is not None:
            random.seed(seed)
        new_data = list(self.data)
        random.shuffle(new_data)
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def sample(
        self,
        k: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        wrap: bool = True,
    ) -> tp.Any:
        """Return a random sample of data items from the asset.

        Args:
            k (Optional[int]): Number of items to sample.

                Defaults to 1 if not specified.
            seed (Optional[int]): Random seed for deterministic output.
            wrap (bool): If True, wrap the sampled data in a new asset; otherwise, return raw data items.

        Returns:
            Any: Either a new asset with the sampled data if `wrap` is True, or a single item
                (when sampling one) or a list of items.
        """
        import random

        if k is None:
            k = 1
            single_item = True
        else:
            single_item = False
        if seed is not None:
            random.seed(seed)
        new_data = random.sample(self.data, min(len(self.data), k))
        if wrap:
            return self.replace(data=new_data, single_item=single_item)
        if single_item:
            return new_data[0]
        return new_data

    def print_sample(
        self, k: tp.Optional[int] = None, seed: tp.Optional[int] = None, **kwargs
    ) -> None:
        """Print a random sample of data items.

        Args:
            k (Optional[int]): Number of items to sample.

                Defaults to 1 if not specified.
            seed (Optional[int]): Random seed for deterministic output.
            **kwargs: Keyword arguments for `KnowledgeAsset.print`.

        Returns:
            None
        """
        self.sample(k=k, seed=seed).print(**kwargs)

    # ############# Collection methods ############# #

    def __len__(self) -> int:
        return len(self.data)

    # ############# Sequence methods ############# #

    def __getitem__(self, index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]]) -> tp.Any:
        return self.get_items(index)

    # ############# MutableSequence methods ############# #

    def insert(self, index: int, value: tp.Any) -> None:
        new_data = list(self.data)
        new_data.insert(index, value)
        self.modify_data(new_data)

    def __setitem__(
        self, index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]], value: tp.Any
    ) -> None:
        self.set_items(index, value, inplace=True)

    def __delitem__(self, index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]]) -> None:
        self.delete_items(index, inplace=True)

    def __add__(self: KnowledgeAssetT, other: tp.Any) -> KnowledgeAssetT:
        if not isinstance(other, KnowledgeAsset):
            other = KnowledgeAsset(other)
        mro_self = self.__class__.mro()
        mro_other = other.__class__.mro()
        common_bases = set(mro_self).intersection(mro_other)
        for cls in mro_self:
            if cls in common_bases:
                new_type = cls
                break
        else:
            new_type = KnowledgeAsset
        return new_type.combine(self, other)

    def __iadd__(self: KnowledgeAssetT, other: tp.Any) -> KnowledgeAssetT:
        if isinstance(other, KnowledgeAsset):
            other = other.data
        self.extend_items(other, inplace=True)
        return self

    # ############# Apply methods ############# #

    def apply(
        self,
        func: tp.MaybeList[tp.Union[tp.AssetFuncLike, tp.AssetPipeline]],
        *args,
        execute_kwargs: tp.KwargsLike = None,
        wrap: tp.Optional[bool] = None,
        single_item: tp.Optional[bool] = None,
        return_iterator: bool = False,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Apply a function or pipeline to each data item in the asset.

        The `func` parameter accepts various types:

        * Callable or a tuple containing a callable and its arguments.
        * Instance of `vectorbtpro.utils.execution.Task`.
        * Subclass of `vectorbtpro.utils.knowledge.base_asset_funcs.AssetFunc` or its prefix/full name.
        * List of any of the above, which will use `BasicAssetPipeline`.
        * Valid expression, which will use `ComplexAssetPipeline`.

        Execution is handled by `vectorbtpro.utils.execution.execute`.

        Args:
            func (MaybeList[Union[AssetFuncLike, AssetPipeline]]): Function, pipeline, or expression to apply.
            *args: Positional arguments for the asset pipeline or function.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            wrap (Optional[bool]): If True, return the result wrapped as an asset.
            single_item (Optional[bool]): Determines if a single item should not be wrapped in a list.
            return_iterator (bool): If True, return an iterator instead of executing tasks.
            **kwargs: Keyword arguments for the asset pipeline or function.

        Returns:
            MaybeKnowledgeAsset: New asset with processed data if `wrap` is True; otherwise, raw output.

        Examples:
            ```pycon
            >>> asset.apply(["flatten", ("query", len)])
            [5, 5, 5, 5, 6]

            >>> asset.apply("query(flatten(d), len)")
            [5, 5, 5, 5, 6]
            ```
        """
        from vectorbtpro.utils.knowledge.asset_pipelines import (
            AssetPipeline,
            BasicAssetPipeline,
            ComplexAssetPipeline,
        )

        execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)
        asset_func_meta = {}

        if isinstance(func, list):
            func, args, kwargs = (
                BasicAssetPipeline(
                    func,
                    *args,
                    cond_kwargs=dict(asset_cls=type(self)),
                    asset_func_meta=asset_func_meta,
                    **kwargs,
                ),
                (),
                {},
            )
        elif isinstance(func, str) and not func.isidentifier():
            if len(args) > 0:
                raise ValueError(
                    "No more positional arguments can be applied to ComplexAssetPipeline"
                )
            func, args, kwargs = (
                ComplexAssetPipeline(
                    func,
                    context=kwargs.get("template_context"),
                    cond_kwargs=dict(asset_cls=type(self)),
                    asset_func_meta=asset_func_meta,
                    **kwargs,
                ),
                (),
                {},
            )
        elif not isinstance(func, AssetPipeline):
            func, args, kwargs = AssetPipeline.resolve_task(
                func,
                *args,
                cond_kwargs=dict(asset_cls=type(self)),
                asset_func_meta=asset_func_meta,
                **kwargs,
            )
        else:
            if len(args) > 0:
                raise ValueError("No more positional arguments can be applied to AssetPipeline")
            if len(kwargs) > 0:
                raise ValueError("No more keyword arguments can be applied to AssetPipeline")
        prefix = get_caller_qualname().split(".")[-1]
        if "_short_name" in asset_func_meta:
            prefix += f"[{asset_func_meta['_short_name']}]"
        elif isinstance(func, type):
            prefix += f"[{func.__name__}]"
        else:
            prefix += f"[{type(func).__name__}]"
        execute_kwargs = merge_dicts(
            dict(
                show_progress=False if self.single_item else None,
                pbar_kwargs=dict(
                    bar_id=get_caller_qualname(),
                    prefix=prefix,
                ),
            ),
            execute_kwargs,
        )

        def _get_task_generator():
            for i, d in enumerate(self.data):
                _kwargs = dict(kwargs)
                if "template_context" in _kwargs:
                    _kwargs["template_context"] = flat_merge_dicts(
                        {"i": i},
                        _kwargs["template_context"],
                    )
                if return_iterator:
                    yield Task(func, d, *args, **_kwargs).execute()
                else:
                    yield Task(func, d, *args, **_kwargs)

        tasks = _get_task_generator()
        if return_iterator:
            return tasks
        new_data = execute(tasks, size=len(self.data), **execute_kwargs)
        if new_data is NoResult:
            new_data = []
        if wrap is None and asset_func_meta.get("_wrap") is not None:
            wrap = asset_func_meta["_wrap"]
        if wrap is None:
            wrap = True
        if single_item is None:
            single_item = self.single_item
        if wrap:
            return self.replace(data=new_data, single_item=single_item)
        if single_item:
            if len(new_data) == 1:
                return new_data[0]
            if len(new_data) == 0:
                return None
        return new_data

    def get(
        self,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Optional[tp.CustomTemplateLike] = None,
        template_context: tp.KwargsLike = None,
        single_item: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Return specific data items or subsets of them.

        This method retrieves complete data items or extracts portions specified by a nested path.
        It applies `vectorbtpro.utils.knowledge.base_asset_funcs.GetAssetFunc` via `KnowledgeAsset.apply`.

        Args:
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to get (e.g. "x.y[0].z").
            keep_path (Optional[bool]): If True, returns results structured as nested dictionaries
                mirroring the specified path.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            source (Optional[CustomTemplateLike]): Template, function, or string for preprocessing;
                in the template, "i" denotes the index, "d" the full data item, and "x" the extracted part.
            template_context (KwargsLike): Additional context for template substitution.
            single_item (Optional[bool]): Determines if a single item should not be wrapped in a list.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset containing the selected data.

        Examples:
            ```pycon
            >>> asset.get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.get("d2.l[0]")
            [1, 3, 5, 7, 9]

            >>> asset.get("d2.l", source=lambda x: sum(x))
            [3, 7, 11, 15, 19]

            >>> asset.get("d2.l[0]", keep_path=True)
            [{'d2': {'l': {0: 1}}},
             {'d2': {'l': {0: 3}}},
             {'d2': {'l': {0: 5}}},
             {'d2': {'l': {0: 7}}},
             {'d2': {'l': {0: 9}}}]

            >>> asset.get(["d2.l[0]", "d2.l[1]"])
            [{'d2': {'l': {0: 1, 1: 2}}},
             {'d2': {'l': {0: 3, 1: 4}}},
             {'d2': {'l': {0: 5, 1: 6}}},
             {'d2': {'l': {0: 7, 1: 8}}},
             {'d2': {'l': {0: 9, 1: 10}}}]

            >>> asset.get("xyz", skip_missing=True)
            [123]
            ```
        """
        if path is None and source is None:
            if single_item is None:
                single_item = self.single_item
            if single_item:
                if len(self.data) == 1:
                    return self.data[0]
                if len(self.data) == 0:
                    return None
            return self.data
        return self.apply(
            "get",
            path=path,
            keep_path=keep_path,
            skip_missing=skip_missing,
            source=source,
            template_context=template_context,
            single_item=single_item,
            **kwargs,
        )

    def select(self: KnowledgeAssetT, *args, **kwargs) -> KnowledgeAssetT:
        """Return a new `KnowledgeAsset` instance based on the output of `KnowledgeAsset.get`.

        Args:
            *args: Positional arguments for `KnowledgeAsset.get`.
            **kwargs: Keyword arguments for `KnowledgeAsset.get`.

        Returns:
            KnowledgeAsset: New asset containing the selected data.
        """
        return self.get(*args, wrap=True, **kwargs)

    def set(
        self,
        value: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Set specific data items or their parts.

        This method modifies data items by applying `vectorbtpro.utils.knowledge.base_asset_funcs.SetAssetFunc`
        via `KnowledgeAsset.apply`.

        Args:
            value (Any): Value, function, or template to set.

                In templates, "i" represents the index, "d" the full data item, and "x" the targeted part.
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to set (e.g. "x.y[0].z").
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with the modified data.

        Examples:
            ```pycon
            >>> asset.set(lambda d: sum(d["d2"]["l"])).get()
            [3, 7, 11, 15, 19]

            >>> asset.set(lambda d: sum(d["d2"]["l"]), path="d2.sum").get()
            >>> asset.set(lambda x: sum(x["l"]), path="d2.sum").get()
            >>> asset.set(lambda l: sum(l), path="d2.sum").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2], 'sum': 3}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4], 'sum': 7}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6], 'sum': 11}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8], 'sum': 15}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10], 'sum': 19}, 'xyz': 123}]

            >>> asset.set(lambda l: sum(l), path="d2.l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': 3}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': 7}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': 11}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': 15}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': 19}, 'xyz': 123}]
            ```
        """
        return self.apply(
            "set",
            value=value,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            template_context=template_context,
            **kwargs,
        )

    def remove(
        self,
        path: tp.MaybeList[tp.PathLikeKey],
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Remove data items or parts of them from the asset.

        Leverages `KnowledgeAsset.apply` with
        `vectorbtpro.utils.knowledge.base_asset_funcs.RemoveAssetFunc` to remove either an entire data item
        (when a numeric path is provided) or a specific element within a data item based on a hierarchical path
        (e.g., "x.y[0].z").

        Args:
            path (MaybeList[PathLikeKey]): Path or list of paths indicating the element(s) to remove.

                If an integer is provided, the entire data item at that index is removed.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with the specified data items removed.

        Examples:
            ```pycon
            >>> asset.remove("d2.l[0]").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [10]}, 'xyz': 123}]

            >>> asset.remove("xyz", skip_missing=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}}]
            ```
        """
        return self.apply(
            "remove",
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def move(
        self,
        path: tp.Union[tp.PathMoveDict, tp.MaybeList[tp.PathLikeKey]],
        new_path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Move data items or parts of them within the asset.

        Uses `KnowledgeAsset.apply` with
        `vectorbtpro.utils.knowledge.base_asset_funcs.MoveAssetFunc` to reposition elements within
        data items. Specify the element to move using `path`. When `new_path` is provided, it designates
        the new token for the element; otherwise, `path` must be given as a dictionary mapping original
        paths to new tokens.

        Args:
            path (Union[PathMoveDict, MaybeList[PathLikeKey]]): Mapping or path(s) within the data item
                to move (e.g. "x.y[0].z").

                When provided as a dictionary, keys are source paths and values are the corresponding new tokens.
            new_path (Optional[MaybeList[PathLikeKey]]): Path(s) for the moved element(s)
                when `path` is not a dictionary.
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with the modified data.

        Examples:
            ```pycon
            >>> asset.move("d2.l", "l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red'}, 'l': [1, 2]},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue'}, 'l': [3, 4]},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green'}, 'l': [5, 6]},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow'}, 'l': [7, 8]},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black'}, 'xyz': 123, 'l': [9, 10]}]

            >>> asset.move({"d2.c": "c", "b": "d2.b"}).get()
            >>> asset.move(["d2.c", "b"], ["c", "d2.b"]).get()
            [{'s': 'ABC', 'd2': {'l': [1, 2], 'b': True}, 'c': 'red'},
             {'s': 'BCD', 'd2': {'l': [3, 4], 'b': True}, 'c': 'blue'},
             {'s': 'CDE', 'd2': {'l': [5, 6], 'b': False}, 'c': 'green'},
             {'s': 'DEF', 'd2': {'l': [7, 8], 'b': False}, 'c': 'yellow'},
             {'s': 'EFG', 'd2': {'l': [9, 10], 'b': False}, 'xyz': 123, 'c': 'black'}]
            ```
        """
        return self.apply(
            "move",
            path=path,
            new_path=new_path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def rename(
        self,
        path: tp.Union[tp.PathRenameDict, tp.MaybeList[tp.PathLikeKey]],
        new_token: tp.Optional[tp.MaybeList[tp.PathKeyToken]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Rename data items or parts of them within the asset.

        Leverages `KnowledgeAsset.apply` with
        `vectorbtpro.utils.knowledge.base_asset_funcs.RenameAssetFunc` to change the names of elements within
        data items. This function is similar to `move` but uses `new_token` to specify the new name.

        Args:
            path (Union[PathRenameDict, MaybeList[PathLikeKey]]): Mapping or list of paths indicating
                the element(s) to rename.

                When provided as a dictionary, the keys define the source paths.
            new_token (Optional[MaybeList[PathKeyToken]]): New token or list of tokens for
                renaming the element(s).
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with the modified data.

        Examples:
            ```pycon
            >>> asset.rename("d2.l", "x").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'x': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'x': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'x': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'x': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'x': [9, 10]}, 'xyz': 123}]

            >>> asset.rename("xyz", "zyx", skip_missing=True, changed_only=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'zyx': 123}]
            ```
        """
        return self.apply(
            "rename",
            path=path,
            new_token=new_token,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def reorder(
        self,
        new_order: tp.Union[str, tp.PathKeyTokens],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Reorder data items or parts within each item.

        Uses `KnowledgeAsset.apply` with `vectorbtpro.utils.knowledge.base_asset_funcs.ReorderAssetFunc`
        to reorder data. For dictionaries, keys are reordered using `vectorbtpro.utils.config.reorder_dict`;
        for sequences, ordering follows `vectorbtpro.utils.config.reorder_list`.

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
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with the reordered data.

        Examples:
            ```pycon
            >>> asset.reorder(["xyz", ...], skip_missing=True).get()
            >>> asset.reorder(lambda x: ["xyz", ...] if "xyz" in x else [...]).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'xyz': 123, 's': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}}]

            >>> asset.reorder("descending", path="d2.l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [2, 1]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [4, 3]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [6, 5]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [8, 7]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [10, 9]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            "reorder",
            new_order=new_order,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            template_context=template_context,
            **kwargs,
        )

    def query(
        self,
        expression: tp.CustomTemplateLike,
        query_engine: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        return_type: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Query data items using a specified engine and return matching results.

        Evaluates an expression or template over data items using one of the following engines:

        * "jmespath": Evaluates expressions with the `jmespath` package.
        * "jsonpath", "jsonpath-ng", "jsonpath_ng": Evaluates expressions with the `jsonpath_ng` package.
        * "jsonpath.ext", "jsonpath-ng.ext", "jsonpath_ng.ext": Evaluates expressions with the extended
            `jsonpath_ng` package.
        * None or "template": Evaluates each data item as a template using `KnowledgeAsset.apply` with
            `vectorbtpro.utils.knowledge.base_asset_funcs.QueryAssetFunc`. In the template, use `i`
            for the item index, `d` for the data item, `x` for the value at a specified path, and
            field names for individual fields.
        * "pandas": Evaluates the expression treating data items as rows with their fields as columns.

        Templates can also utilize functions from `vectorbtpro.utils.search_.search_config` and operate
        on both single values and sequences.

        Args:
            expression (CustomTemplateLike): Query expression or template.
            query_engine (Optional[str]): Name of the query engine.
            template_context (KwargsLike): Additional context for template substitution.
            return_type (Optional[str]): If "item", returns the matched data item; if "bool",
                returns a boolean indicating a match.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply` or the query engine.

        Returns:
            MaybeKnowledgeAsset: New asset with the matching data items.

        Examples:
            ```pycon
            >>> asset.query("d['s'] == 'ABC'")
            >>> asset.query("x['s'] == 'ABC'")
            >>> asset.query("s == 'ABC'")
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}}]

            >>> asset.query("x['s'] == 'ABC'", return_type="bool")
            [True, False, False, False, False]

            >>> asset.query("find('BC', s)")
            >>> asset.query(lambda s: "BC" in s)
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.query("[?contains(s, 'BC')].s", query_engine="jmespath")
            ['ABC', 'BCD']

            >>> asset.query("[].d2.c", query_engine="jmespath")
            ['red', 'blue', 'green', 'yellow', 'black']

            >>> asset.query("[?d2.c != `blue`].d2.l", query_engine="jmespath")
            [[1, 2], [5, 6], [7, 8], [9, 10]]

            >>> asset.query("$[*].d2.c", query_engine="jsonpath.ext")
            ['red', 'blue', 'green', 'yellow', 'black']

            >>> asset.query("$[?(@.b == true)].s", query_engine="jsonpath.ext")
            ['ABC', 'BCD']

            >>> asset.query("s[b]", query_engine="pandas")
            ['ABC', 'BCD']
            ```
        """
        query_engine = self.resolve_setting(query_engine, "query_engine")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)
        return_type = self.resolve_setting(return_type, "return_type")

        if query_engine is None or query_engine.lower() == "template":
            new_obj = self.apply(
                "query",
                expression=expression,
                template_context=template_context,
                return_type=return_type,
                **kwargs,
            )
        elif query_engine.lower() == "jmespath":
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jmespath")
            import jmespath

            new_obj = jmespath.search(expression, self.data, **kwargs)
        elif query_engine.lower() in ("jsonpath", "jsonpath-ng", "jsonpath_ng"):
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jsonpath_ng")
            import jsonpath_ng

            jsonpath_expr = jsonpath_ng.parse(expression)
            new_obj = [match.value for match in jsonpath_expr.find(self.data, **kwargs)]
        elif query_engine.lower() in ("jsonpath.ext", "jsonpath-ng.ext", "jsonpath_ng.ext"):
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jsonpath_ng")
            import jsonpath_ng.ext

            jsonpath_expr = jsonpath_ng.ext.parse(expression)
            new_obj = [match.value for match in jsonpath_expr.find(self.data, **kwargs)]
        elif query_engine.lower() == "pandas":
            if isinstance(expression, str):
                expression = RepEval(expression)
            elif checks.is_function(expression):
                if checks.is_builtin_func(expression):
                    expression = RepFunc(lambda _expression=expression: _expression)
                else:
                    expression = RepFunc(expression)
            elif not isinstance(expression, CustomTemplate):
                raise TypeError("Expression must be a template")
            df = pd.DataFrame.from_records(self.data)
            _template_context = flat_merge_dicts(
                {
                    "d": df,
                    "x": df,
                    **df.to_dict(orient="series"),
                },
                template_context,
            )
            result = expression.substitute(_template_context, eval_id="expression", **kwargs)
            if checks.is_function(result):
                result = result(df)
            if return_type.lower() == "item":
                as_filter = True
            elif return_type.lower() == "bool":
                as_filter = False
            else:
                raise ValueError(f"Invalid return type: '{return_type}'")
            if as_filter and isinstance(result, pd.Series) and result.dtype == "bool":
                result = df[result]
            if isinstance(result, pd.Series):
                new_obj = result.tolist()
            elif isinstance(result, pd.DataFrame):
                new_obj = result.to_dict(orient="records")
            else:
                new_obj = result
        else:
            raise ValueError(f"Invalid query engine: '{query_engine}'")
        return new_obj

    def filter(self: KnowledgeAssetT, *args, **kwargs) -> KnowledgeAssetT:
        """Return a new `KnowledgeAsset` instance by calling `KnowledgeAsset.query`.

        Args:
            *args: Positional arguments for `KnowledgeAsset.query`.
            **kwargs: Keyword arguments for `KnowledgeAsset.query`.

        Returns:
            KnowledgeAsset: New asset containing the filtered data.
        """
        return self.query(*args, wrap=True, **kwargs)

    def find(
        self,
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
        merge_matches: tp.Optional[bool] = None,
        merge_fields: tp.Optional[bool] = None,
        unique_matches: tp.Optional[bool] = None,
        unique_fields: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Return a new `KnowledgeAsset` instance with found occurrences based on the target.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.FindAssetFunc`.

        Searches each data item with `vectorbtpro.utils.search_.contains_in_obj` when `return_type`
        is "item", "field", or "bool", and uses `vectorbtpro.utils.search_.find_in_obj` and
        `vectorbtpro.utils.search_.find` for all other return types.

        Target can be one or multiple data items. If there are multiple targets and `find_all` is True,
        the match function will return True only if all targets have been found.

        Use argument `source` instead of `path` or in addition to `path` to also preprocess the source.
        It can be a string or function (will become a template), or any custom template. In this template,
        the index of the data item is represented by "i", the data item itself is represented by "d",
        the data item under the path is represented by "x" while its fields are represented by their names.

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
            merge_matches (Optional[bool]): If False, keeps empty lists when searching for matches.
            merge_fields (Optional[bool]): If False, keeps empty lists when searching for fields.
            unique_matches (Optional[bool]): If False, allows duplicate matches.
            unique_fields (Optional[bool]): If False, allows duplicate fields.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with the found data items.

        Examples:
            ```pycon
            >>> asset.find("BC").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find("BC", return_type="bool").get()
            [True, True, False, False, False]

            >>> asset.find(vbt.Not("BC")).get()
            [{'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find("bc", ignore_case=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find("bl", path="d2.c").get()
            [{'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find(5, path="d2.l[0]").get()
            [{'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find(True, path="d2.l", source=lambda x: sum(x) >= 10).get()
            [{'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find(["A", "B", "C"]).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find(["A", "B", "C"], find_all=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}}]

            >>> asset.find(r"[ABC]+", mode="regex").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find("yenlow", mode="fuzzy").get()
            [{'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]

            >>> asset.find("yenlow", mode="fuzzy", return_type="match").get()
            'yellow'

            >>> asset.find("yenlow", mode="fuzzy", return_type="match", merge_matches=False).get()
            [[], [], [], ['yellow'], []]

            >>> asset.find("yenlow", mode="fuzzy", return_type="match", return_path=True).get()
            [{}, {}, {}, {('d2', 'c'): ['yellow']}, {}]

            >>> asset.find("xyz", in_dumps=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        found_asset = self.apply(
            "find",
            target=target,
            path=path,
            per_path=per_path,
            find_all=find_all,
            keep_path=keep_path,
            skip_missing=skip_missing,
            source=source,
            in_dumps=in_dumps,
            dump_kwargs=dump_kwargs,
            template_context=template_context,
            return_type=return_type,
            return_path=return_path,
            **kwargs,
        )
        return_type = self.resolve_setting(return_type, "return_type")
        merge_matches = self.resolve_setting(merge_matches, "merge_matches")
        merge_fields = self.resolve_setting(merge_fields, "merge_fields")
        unique_matches = self.resolve_setting(unique_matches, "unique_matches")
        unique_fields = self.resolve_setting(unique_fields, "unique_fields")
        if (
            (
                (merge_matches and return_type.lower() == "match")
                or (merge_fields and return_type.lower() == "field")
            )
            and isinstance(found_asset, KnowledgeAsset)
            and len(found_asset) > 0
            and isinstance(found_asset[0], list)
        ):
            found_asset = found_asset.merge()
        if (
            (
                (unique_matches and return_type.lower() == "match")
                or (unique_fields and return_type.lower() == "field")
            )
            and isinstance(found_asset, KnowledgeAsset)
            and len(found_asset) > 0
            and isinstance(found_asset[0], str)
        ):
            found_asset = found_asset.unique()
        return found_asset

    def find_code(
        self,
        target: tp.Optional[tp.MaybeIterable[tp.Any]] = None,
        language: tp.Union[None, bool, tp.MaybeIterable[str]] = None,
        in_blocks: tp.Optional[bool] = None,
        escape_target: bool = True,
        escape_language: bool = True,
        return_type: tp.Optional[str] = "match",
        flags: int = 0,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Return code segments from the asset data that match the specified target pattern and language criteria.

        This method constructs a regular expression based on the provided target, language, and block settings,
        and then uses `KnowledgeAsset.find` to search the asset data.

        Args:
            target (Optional[Iterable[Any]]): Target pattern(s) to locate in the asset.
            language (Union[None, bool, Iterable[str]]): Language specification(s) to filter code blocks.
            in_blocks (Optional[bool]): If True, search within code blocks rather than inline code.
            escape_target (bool): If True, escape regex special characters in the target.
            escape_language (bool): If True, escape regex special characters in the language.
            return_type (Optional[str]): Type of result to return.
            flags (int): Additional flags for compiling the regular expression.
            **kwargs: Keyword arguments for `KnowledgeAsset.find`.

        Returns:
            MaybeKnowledgeAsset: New asset with segments that match the search criteria.

        !!! info
            For default settings, see `code` in `vectorbtpro._settings.knowledge`.
        """
        language = self.resolve_setting(language, "language", sub_path="code")
        in_blocks = self.resolve_setting(in_blocks, "in_blocks", sub_path="code")

        if target is not None:
            if not isinstance(target, (str, list)):
                target = list(target)
        if language is not None:
            if not isinstance(language, (str, list)):
                language = list(language)
            if escape_language:
                if isinstance(language, list):
                    language = list(map(re.escape, language))
                else:
                    language = re.escape(language)
            if isinstance(language, list):
                language = rf"(?:{'|'.join(language)})"

        opt_language = r"[\w+-]+"
        opt_title = r"(?:[ \t]+[^\n`]+)?"

        if target is not None:
            if not isinstance(target, list):
                targets = [target]
                single_target = True
            else:
                targets = target
                single_target = False
            new_target = []
            for t in targets:
                if escape_target:
                    t = re.escape(t)
                if in_blocks:
                    if language is not None and not isinstance(language, bool):
                        new_t = rf"""
                        ```{language}{opt_title}\n
                        (?:(?!```)[\s\S])*?
                        {t}
                        (?:(?!```)[\s\S])*?
                        ```\s*$
                        """
                    elif language is not None and isinstance(language, bool) and language:
                        new_t = rf"""
                        ```{opt_language}{opt_title}\n
                        (?:(?!```)[\s\S])*?
                        {t}
                        (?:(?!```)[\s\S])*?
                        ```\s*$
                        """
                    else:
                        new_t = rf"""
                        ```(?:{opt_language}{opt_title})?\n
                        (?:(?!```)[\s\S])*?
                        {t}
                        (?:(?!```)[\s\S])*?
                        ```\s*$
                        """
                else:
                    new_t = rf"(?<!`)`([^`]*{t}[^`]*)`(?!`)"
                new_target.append(new_t)
            if single_target:
                new_target = new_target[0]
        else:
            if in_blocks:
                if language is not None and not isinstance(language, bool):
                    new_target = rf"```{language}{opt_title}\n([\s\S]*?)```\s*$"
                elif language is not None and isinstance(language, bool) and language:
                    new_target = rf"```{opt_language}{opt_title}\n([\s\S]*?)```\s*$"
                else:
                    new_target = rf"```(?:{opt_language}{opt_title})?\n([\s\S]*?)```\s*$"
            else:
                new_target = r"(?<!`)`([^`]*)`(?!`)"
        if in_blocks:
            flags |= re.DOTALL | re.MULTILINE | re.VERBOSE
        found = self.find(new_target, mode="regex", return_type=return_type, flags=flags, **kwargs)

        def _clean_block(block):
            return textwrap.dedent(block).lstrip("\n").rstrip()

        if isinstance(found, KnowledgeAsset):
            found = found.apply(_clean_block)
        elif isinstance(found, list):
            found = [_clean_block(x) for x in found]
        elif isinstance(found, str):
            found = _clean_block(found)
        return found

    def find_replace(
        self,
        target: tp.Union[dict, tp.MaybeList[tp.Any]],
        replacement: tp.Optional[tp.MaybeList[tp.Any]] = None,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Return a new `KnowledgeAsset` with occurrences replaced according to the specified criteria.

        This method applies a find-and-replace operation on the asset data using
        `vectorbtpro.utils.knowledge.base_asset_funcs.FindReplaceAssetFunc` via `KnowledgeAsset.apply`.
        It uses `vectorbtpro.utils.search_.find_in_obj` to locate occurrences and
        `vectorbtpro.utils.search_.replace_in_obj` to perform the replacements.

        The target can be provided as a single or multiple data items (list or dictionary).
        When multiple targets are used with `find_all` set to True, all targets must be found
        to register a match. The `path` parameter specifies the portion of the data item to search
        (e.g., "x.y[0].z" to access nested elements). If `keep_path` is True, the results will be
        returned as a nested dictionary keyed by the specified paths. Providing multiple paths will
        automatically enable `keep_path` and merge the results. If `skip_missing` is True, any data
        item missing the specified path will be skipped. When `per_path` is True, targets and
        replacements are applied per individual path.

        Setting `make_copy` avoids modifying the original data.
        Enabling `changed_only` will return only data items that were modified.

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
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with the specified replacements applied.

        Examples:
            ```pycon
            >>> asset.find_replace("BC", "XY").get()
            [{'s': 'AXY', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'XYD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find_replace("BC", "XY", changed_only=True).get()
            [{'s': 'AXY', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'XYD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find_replace(r"(D)E(F)", r"\1X\2", mode="regex", changed_only=True).get()
            [{'s': 'DXF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]

            >>> asset.find_replace(True, False, changed_only=True).get()
            [{'s': 'ABC', 'b': False, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': False, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find_replace(3, 30, path="d2.l", changed_only=True).get()
            [{'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [30, 4]}}]

            >>> asset.find_replace({1: 10, 4: 40}, path="d2.l", changed_only=True).get()
            >>> asset.find_replace({1: 10, 4: 40}, path=["d2.l[0]", "d2.l[1]"], changed_only=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [10, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 40]}}]

            >>> asset.find_replace({1: 10, 4: 40}, find_all=True, changed_only=True).get()
            []

            >>> asset.find_replace({1: 10, 2: 20}, find_all=True, changed_only=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [10, 20]}}]

            >>> asset.find_replace("a", "X", path=["s", "d2.c"], ignore_case=True, changed_only=True).get()
            [{'s': 'XBC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'blXck', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find_replace(123, 456, path="xyz", skip_missing=True, changed_only=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 456}]
            ```
        """
        return self.apply(
            "find_replace",
            target=target,
            replacement=replacement,
            path=path,
            per_path=per_path,
            find_all=find_all,
            keep_path=keep_path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def find_remove(
        self,
        target: tp.Union[dict, tp.MaybeList[tp.Any]],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Remove occurrences of a target from the asset data and return a new `KnowledgeAsset` instance.

        This method applies a removal operation on nested data items using `KnowledgeAsset.apply` with
        `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.

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
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with the specified occurrences removed.

        Similar to `KnowledgeAsset.find_replace`.
        """
        return self.apply(
            "find_remove",
            target=target,
            path=path,
            per_path=per_path,
            find_all=find_all,
            keep_path=keep_path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def find_remove_empty(
        self,
        skip_keys: tp.Optional[tp.Container[tp.Hashable]] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Remove empty objects from the asset data.

        This method uses a predefined emptiness check via
        `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc.is_empty_func` to remove empty objects.

        Args:
            **kwargs: Keyword arguments for `KnowledgeAsset.find_remove`.

        Returns:
            MaybeKnowledgeAsset: New asset with empty objects removed.
        """
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc

        return self.find_remove(
            partial(FindRemoveAssetFunc.is_empty_func, skip_keys=skip_keys), **kwargs
        )

    def flatten(
        self,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Flatten nested elements in the asset data into a flat structure.

        This method applies a flattening operation using `KnowledgeAsset.apply` with
        `vectorbtpro.utils.knowledge.base_asset_funcs.FlattenAssetFunc`. Specify the nested
        portion to flatten using the `path` argument. Multiple paths can be provided.
        If `skip_missing` is True and a specified path is missing, the data item will be skipped.

        Args:
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to flatten (e.g. "x.y[0].z").
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with flattened data.

        Examples:
            ```pycon
            >>> asset.flatten().get()
            [{'s': 'ABC',
              'b': True,
              ('d2', 'c'): 'red',
              ('d2', 'l', 0): 1,
              ('d2', 'l', 1): 2},
              ...
             {'s': 'EFG',
              'b': False,
              ('d2', 'c'): 'black',
              ('d2', 'l', 0): 9,
              ('d2', 'l', 1): 10,
              'xyz': 123}]
            ```
        """
        return self.apply(
            "flatten",
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def unflatten(
        self,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Reconstruct nested structures from flattened asset data.

        This method applies an unflattening operation using `KnowledgeAsset.apply` with
        `vectorbtpro.utils.knowledge.base_asset_funcs.UnflattenAssetFunc`. Specify the flattened portion to
        reconstruct using the `path` argument. Multiple paths can be provided. If `skip_missing`
        is True and a specified path is missing, the data item will be skipped.

        Args:
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to unflatten (e.g. "x.y[0].z").
            skip_missing (Optional[bool]): If True, skips data items where the specified path is missing.
            make_copy (Optional[bool]): If True, operates on a copy rather than modifying the original data.
            changed_only (Optional[bool]): If True, returns only data items that were modified.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with unflattened data.

        Examples:
            ```pycon
            >>> asset.flatten().unflatten().get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            "unflatten",
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def dump(
        self,
        source: tp.Optional[tp.CustomTemplateLike] = None,
        dump_engine: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Dump asset data items using a specified dump engine.

        This method applies `KnowledgeAsset.apply` with
        `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc` to format asset data.

        Supported dump engines:

        * "repr": Uses Python's `repr`.
        * "prettify": Uses `vectorbtpro.utils.formatting.prettify`.
        * "nestedtext": Uses NestedText (https://pypi.org/project/nestedtext/).
        * "yaml": Uses YAML formatting.
        * "toml": Uses TOML (https://pypi.org/project/toml/).
        * "json": Uses JSON formatting.

        The `source` argument can be a string, callable, or custom template to preprocess the data.
        In the template, "i" represents the index, "d" represents the data item, and its fields are
        accessible by name.

        Args:
            source (Optional[CustomTemplateLike]): Template or function to preprocess the source data.
            dump_engine (Optional[str]): Name of the dump engine.

                See `vectorbtpro.utils.formatting.dump`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with dumped data.

        Examples:
            ```pycon
            >>> print(asset.dump(source="{i: d}", default_flow_style=True).join())
            {0: {s: ABC, b: true, d2: {c: red, l: [1, 2]}}}
            {1: {s: BCD, b: true, d2: {c: blue, l: [3, 4]}}}
            {2: {s: CDE, b: false, d2: {c: green, l: [5, 6]}}}
            {3: {s: DEF, b: false, d2: {c: yellow, l: [7, 8]}}}
            {4: {s: EFG, b: false, d2: {c: black, l: [9, 10]}, xyz: 123}}
            ```
        """
        return self.apply(
            "dump",
            source=source,
            dump_engine=dump_engine,
            template_context=template_context,
            **kwargs,
        )

    def dump_all(
        self,
        source: tp.Optional[tp.CustomTemplateLike] = None,
        dump_engine: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Dump asset data list into a single asset representation.

        This method uses `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc.prepare_and_call`
        on the asset's data with the provided parameters.

        Args:
            source (Optional[CustomTemplateLike]): Template or function to preprocess the source data.
            dump_engine (Optional[str]): Name of the dump engine.

                See `vectorbtpro.utils.formatting.dump`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc.prepare_and_call`.

        Returns:
            str: Dumped asset data as a string.
        """
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc

        return DumpAssetFunc.prepare_and_call(
            self.data,
            source=source,
            dump_engine=dump_engine,
            template_context=template_context,
            **kwargs,
        )

    def to_documents(
        self,
        document_cls: tp.Optional[tp.Type[tp.StoreDocument]] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Convert asset data items to text documents of type `vectorbtpro.utils.knowledge.chatting.TextDocument`.

        Templates provided via keyword arguments can reference:

        * "i": the index of the data item,
        * "d": the data item,
        * "x": the data item at a specified path, and
        * field names for respective data item fields.

        Args:
            document_cls (Optional[Type[StoreDocument]]): Document class to use for creating documents.

                Defaults to `vectorbtpro.utils.knowledge.chatting.TextDocument`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with data items converted to text documents.
        """
        return self.apply(
            "to_docs",
            document_cls=document_cls,
            template_context=template_context,
            **kwargs,
        )

    def split_text(
        self,
        text_path: tp.Optional[tp.PathLikeKey] = None,
        document_cls: tp.Optional[tp.Type[tp.StoreDocument]] = None,
        merge_chunks: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Split text content from the asset.

        This method applies `vectorbtpro.utils.knowledge.base_asset_funcs.SplitTextAssetFunc`
        via `KnowledgeAsset.apply` to split text content using
        `vectorbtpro.utils.knowledge.chatting.split_text`.

        Args:
            text_path (Optional[PathLikeKey]): Path specifying the location of the text content.
            document_cls (Optional[Type[StoreDocument]]): Document class to use for creating documents.

                Defaults to `vectorbtpro.utils.knowledge.chatting.TextDocument`.
            merge_chunks (Optional[bool]): If True, merge all text chunks into a single list.
            **kwargs: Keyword arguments for `KnowledgeAsset.apply`.

        Returns:
            MaybeKnowledgeAsset: New asset with its text content split into chunks.
        """
        split_asset = self.apply(
            "split_text",
            text_path=text_path,
            document_cls=document_cls,
            **kwargs,
        )
        merge_chunks = self.resolve_setting(merge_chunks, "merge_chunks")
        if (
            merge_chunks
            and isinstance(split_asset, KnowledgeAsset)
            and len(split_asset) > 0
            and isinstance(split_asset[0], list)
        ):
            split_asset = split_asset.merge()
        return split_asset

    # ############# Reduce methods ############# #

    @classmethod
    def get_keys_and_groups(
        cls,
        by: tp.List[tp.Any],
        uniform_groups: bool = False,
    ) -> tp.Tuple[tp.List[tp.Any], tp.List[tp.List[int]]]:
        """Return keys and grouping indices from a list of items.

        When `uniform_groups` is True, consecutive identical items are grouped together.
        Otherwise, groups are formed based on the first occurrence of each unique item.

        Args:
            by (List[Any]): List of items to group.
            uniform_groups (bool): If True, group consecutive identical items; otherwise,
                group all identical items.

        Returns:
            Tuple[List[Any], List[List[int]]]: Tuple containing a list of keys and a corresponding
                list of index groups.
        """
        keys = []
        groups = []
        if uniform_groups:
            for i, item in enumerate(by):
                if len(keys) > 0 and (keys[-1] is item or keys[-1] == item):
                    groups[-1].append(i)
                else:
                    keys.append(item)
                    groups.append([i])
        else:
            groups = []
            representatives = []
            for idx, item in enumerate(by):
                found = False
                for rep_idx, rep_obj in enumerate(representatives):
                    if item is rep_obj or item == rep_obj:
                        groups[rep_idx].append(idx)
                        found = True
                        break
                if not found:
                    representatives.append(item)
                    keys.append(by[idx])
                    groups.append([idx])
        return keys, groups

    def reduce(
        self,
        func: tp.CustomTemplateLike,
        *args,
        initializer: tp.Optional[tp.Any] = None,
        by: tp.Optional[tp.PathLikeKey] = None,
        template_context: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        wrap: tp.Optional[bool] = None,
        return_iterator: bool = False,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Reduce asset data items using a binary operation.

        The reduction function `func` can be a callable, a tuple pairing a function with its arguments,
        a `vectorbtpro.utils.execution.Task` instance, a subclass (or its prefix/full name) of
        `vectorbtpro.utils.knowledge.base_asset_funcs.AssetFunc`, or an expression/template.
        In templates, use "i" for the data item index and "d1"/"d2" (or "x1"/"x2") for operands.

        If an initializer is provided, the reduction starts with `d1` as the initializer and
        `d2` as the first data item. Otherwise, it starts with the first two data items.

        If `by` is specified, see `KnowledgeAsset.groupby_reduce` for grouped reduction.
        If `wrap` is True, the result is returned as a new `KnowledgeAsset` instance.

        Args:
            func (CustomTemplateLike): Reduction function, expression, or template.
            *args: Positional arguments for `KnowledgeAsset.groupby_reduce` or the reduction function.
            initializer (Optional[Any]): Initial value for the reduction.
            by (Optional[PathLikeKey]): Key or path used to group data items.
            template_context (KwargsLike): Additional context for template substitution.
            show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
            pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

                See `vectorbtpro.utils.pbar.ProgressBar`.
            wrap (Optional[bool]): If True, wrap the result in a `KnowledgeAsset` instance.
            return_iterator (bool): If True, return an iterator instead of executing tasks.
            **kwargs: Keyword arguments for `KnowledgeAsset.groupby_reduce` or the reduction function.

        Returns:
            MaybeKnowledgeAsset: New asset with the result of reducing the asset data items.

        Examples:
            ```pycon
            >>> asset.reduce(lambda d1, d2: vbt.merge_dicts(d1, d2))
            >>> asset.reduce(vbt.merge_dicts)
            >>> asset.reduce("{**d1, **d2}")
            {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}

            >>> asset.reduce("{**d1, **d2}", by="b")
            [{'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        if by is not None:
            return self.groupby_reduce(
                func,
                *args,
                by=by,
                initializer=initializer,
                template_context=template_context,
                show_progress=show_progress,
                pbar_kwargs=pbar_kwargs,
                wrap=wrap,
                **kwargs,
            )

        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        asset_func_meta = {}

        if isinstance(func, str) and not func.isidentifier():
            func = RepEval(func)
        elif not isinstance(func, CustomTemplate):
            from vectorbtpro.utils.knowledge.asset_pipelines import AssetPipeline

            func, args, kwargs = AssetPipeline.resolve_task(
                func,
                *args,
                cond_kwargs=dict(asset_cls=type(self)),
                asset_func_meta=asset_func_meta,
                **kwargs,
            )

        it = iter(self.data)
        if initializer is None and asset_func_meta.get("_initializer") is not None:
            initializer = asset_func_meta["_initializer"]
        if initializer is None:
            d1 = next(it)
            total = len(self.data) - 1
            if total == 0:
                raise ValueError("Must provide initializer")
        else:
            d1 = initializer
            total = len(self.data)

        def _get_d1_generator(d1):
            for i, d2 in enumerate(it):
                if isinstance(func, CustomTemplate):
                    _template_context = flat_merge_dicts(
                        {
                            "i": i,
                            "d1": d1,
                            "d2": d2,
                            "x1": d1,
                            "x2": d2,
                        },
                        template_context,
                    )
                    _d1 = func.substitute(_template_context, eval_id="func", **kwargs)
                    if checks.is_function(_d1):
                        d1 = _d1(d1, d2, *args)
                    else:
                        d1 = _d1
                else:
                    _kwargs = dict(kwargs)
                    if "template_context" in _kwargs:
                        _kwargs["template_context"] = flat_merge_dicts(
                            {"i": i},
                            _kwargs["template_context"],
                        )
                    d1 = func(d1, d2, *args, **_kwargs)
                yield d1

        d1s = _get_d1_generator(d1)
        if return_iterator:
            return d1s

        if show_progress is None:
            show_progress = total > 1
        prefix = get_caller_qualname().split(".")[-1]
        if "_short_name" in asset_func_meta:
            prefix += f"[{asset_func_meta['_short_name']}]"
        elif isinstance(func, type):
            prefix += f"[{func.__name__}]"
        else:
            prefix += f"[{type(func).__name__}]"
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                prefix=prefix,
            ),
            pbar_kwargs,
        )
        with ProgressBar(total=total, show_progress=show_progress, **pbar_kwargs) as pbar:
            for d1 in d1s:
                pbar.update()
        if wrap is None and asset_func_meta.get("_wrap") is not None:
            wrap = asset_func_meta["_wrap"]
        if wrap is None:
            wrap = False
        if wrap:
            if not isinstance(d1, list):
                d1 = [d1]
            return self.replace(data=d1, single_item=True)
        return d1

    def groupby_reduce(
        self,
        func: tp.CustomTemplateLike,
        *args,
        by: tp.Optional[tp.PathLikeKey] = None,
        uniform_groups: tp.Optional[bool] = None,
        get_kwargs: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        return_group_keys: bool = False,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Group data items by keys and reduce them.

        Group data items based on keys obtained using the provided `by` parameter via `KnowledgeAsset.get`.
        If `uniform_groups` is True, only contiguous identical key values are grouped.
        For each group, apply `KnowledgeAsset.reduce` with the supplied function and additional arguments.

        Args:
            func (CustomTemplateLike): Reduction function, expression, or template.
            *args: Positional arguments for `KnowledgeAsset.reduce`.
            by (Optional[PathLikeKey]): Key or path used to group data items.
            uniform_groups (Optional[bool]): Whether to group only contiguous identical key values.
            get_kwargs (KwargsLike): Keyword arguments for retrieving keys via `KnowledgeAsset.get`.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            return_group_keys (bool): If True, returns a dictionary mapping group keys to reduction results.
            **kwargs: Keyword arguments for `KnowledgeAsset.reduce`.

        Returns:
            MaybeKnowledgeAsset: New asset with the reduced data items.
        """
        uniform_groups = self.resolve_setting(uniform_groups, "uniform_groups")
        execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)

        if get_kwargs is None:
            get_kwargs = {}
        by = self.get(path=by, **get_kwargs)
        keys, groups = self.get_keys_and_groups(by, uniform_groups=uniform_groups)
        if len(groups) == 0:
            raise ValueError("Groups are empty")
        tasks = []
        for i, group in enumerate(groups):
            group_instance = self.get_items(group)
            tasks.append(Task(group_instance.reduce, func, *args, **kwargs))
        prefix = get_caller_qualname().split(".")[-1]
        execute_kwargs = merge_dicts(
            dict(
                show_progress=False if len(groups) == 1 else None,
                pbar_kwargs=dict(
                    bar_id=get_caller_qualname(),
                    prefix=prefix,
                ),
            ),
            execute_kwargs,
        )
        results = execute(tasks, size=len(groups), **execute_kwargs)
        if return_group_keys:
            return dict(zip(keys, results))
        if len(results) > 0 and isinstance(results[0], type(self)):
            return type(self).combine(results)
        return results

    def merge_dicts(self, **kwargs) -> tp.MaybeKnowledgeAsset:
        """Merge dictionary data items into a single dictionary.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.utils.config.merge_dicts`.

        Returns:
            MaybeKnowledgeAsset: New asset with merged dictionary data.
        """
        return self.reduce("merge_dicts", **kwargs)

    def merge_lists(self, **kwargs) -> tp.MaybeKnowledgeAsset:
        """Merge list data items into a single list.

        Args:
            **kwargs: Keyword arguments for `KnowledgeAsset.reduce`.

        Returns:
            MaybeKnowledgeAsset: New asset with merged list data.
        """
        return self.reduce("merge_lists", **kwargs)

    def collect(
        self,
        sort_keys: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Collect values for each key across all data items.

        Args:
            sort_keys (Optional[bool]): Whether to sort the keys.
            **kwargs: Keyword arguments for `KnowledgeAsset.reduce`.

        Returns:
            MaybeKnowledgeAsset: New asset containing collected values for each key.
        """
        return self.reduce("collect", sort_keys=sort_keys, **kwargs)

    @classmethod
    def describe_lengths(self, lengths: list, **describe_kwargs) -> dict:
        """Describe a list of lengths.

        Compute descriptive statistics for the input lengths, excluding count and standard deviation.

        Args:
            lengths (list): List of numerical lengths.
            **describe_kwargs: Keyword arguments for `pd.Series.describe`.

        Returns:
            dict: Dictionary of descriptive statistics with keys prefixed by "len_".
        """
        len_describe_dict = pd.Series(lengths).describe(**describe_kwargs).to_dict()
        del len_describe_dict["count"]
        del len_describe_dict["std"]
        return {"len_" + k: int(v) if k != "mean" else v for k, v in len_describe_dict.items()}

    def describe(
        self: KnowledgeAssetT,
        ignore_empty: tp.Optional[bool] = None,
        describe_kwargs: tp.KwargsLike = None,
        wrap: bool = False,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, dict]:
        """Collect and describe values for each key in data items.

        Retrieve values using `KnowledgeAsset.collect` and compute descriptive statistics for each
        key using `pd.Series.describe`. For keys containing collection values, additional length
        statistics are computed. If `wrap` is True, the description is wrapped as a single-item
        asset via `KnowledgeAsset.replace`.

        Args:
            ignore_empty (Optional[bool]): Whether to ignore empty values.
            describe_kwargs (KwargsLike): Keyword arguments for `pd.Series.describe`.
            wrap (bool): If True, wraps the description in a single-item asset.
            **kwargs: Keyword arguments for `KnowledgeAsset.collect`.

        Returns:
            Union[KnowledgeAssetT, dict]: Data asset or dictionary containing descriptive statistics.
        """
        ignore_empty = self.resolve_setting(ignore_empty, "ignore_empty")
        describe_kwargs = self.resolve_setting(describe_kwargs, "describe_kwargs", merge=True)

        collected = self.collect(**kwargs)
        description = {}
        for k, v in list(collected.items()):
            all_types = []
            valid_types = []
            valid_x = None
            new_v = []
            for x in v:
                if not ignore_empty or x:
                    new_v.append(x)
                if x is not None:
                    valid_x = x
                    if type(x) not in valid_types:
                        valid_types.append(type(x))
                if type(x) not in all_types:
                    all_types.append(type(x))
            v = new_v
            description[k] = {}
            description[k]["types"] = list(map(lambda x: x.__name__, all_types))
            describe_sr = pd.Series(v)
            if (
                describe_sr.dtype == object
                and len(valid_types) == 1
                and checks.is_complex_collection(valid_x)
            ):
                describe_dict = {"count": len(v)}
            else:
                describe_dict = describe_sr.describe(**describe_kwargs).to_dict()
                if "count" in describe_dict:
                    describe_dict["count"] = int(describe_dict["count"])
                if "unique" in describe_dict:
                    describe_dict["unique"] = int(describe_dict["unique"])
            if pd.api.types.is_integer_dtype(describe_sr.dtype):
                new_describe_dict = {}
                for _k, _v in describe_dict.items():
                    if _k not in {"mean", "std"}:
                        new_describe_dict[_k] = int(_v)
                    else:
                        new_describe_dict[_k] = _v
                describe_dict = new_describe_dict
            if "unique" in describe_dict and describe_dict["unique"] == describe_dict["count"]:
                del describe_dict["top"]
                del describe_dict["freq"]
            if "unique" in describe_dict and describe_dict["count"] == 1:
                del describe_dict["unique"]
            description[k].update(describe_dict)
            if len(valid_types) == 1 and checks.is_collection(valid_x):
                lengths = [len(_v) for _v in v if _v is not None]
                description[k].update(self.describe_lengths(lengths, **describe_kwargs))
        if wrap:
            return self.replace(data=[description], single_item=True)
        return description

    def print_schema(self, **kwargs) -> None:
        """Print the asset schema as a directory tree.

        Keyword arguments are forwarded to `KnowledgeAsset.describe` and
        `vectorbtpro.utils.path_.dir_tree_from_paths` to build the schema structure.

        Examples:
            ```pycon
            >>> asset.print_schema()
            /
            ├── s [5/5, str]
            ├── b [2/5, bool]
            ├── d2 [5/5, dict]
            │   ├── c [5/5, str]
            │   └── l
            │       ├── 0 [5/5, int]
            │       └── 1 [5/5, int]
            └── xyz [1/5, int]

            2 directories, 6 files
            ```
        """
        dir_tree_arg_names = set(get_func_arg_names(dir_tree_from_paths))
        dir_tree_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in dir_tree_arg_names}
        orig_describe_dict = self.describe(wrap=False, **kwargs)
        flat_describe_dict = self.flatten(
            skip_missing=True,
            make_copy=True,
            changed_only=False,
        ).describe(wrap=False, **kwargs)
        describe_dict = flat_merge_dicts(orig_describe_dict, flat_describe_dict)

        paths = []
        display_names = []
        for k, v in describe_dict.items():
            if k is None:
                k = "."
            if not isinstance(k, tuple):
                k = (k,)
            path = Path(*map(str, k))
            path_name = path.name
            path_name += " [" + str(v["count"]) + "/" + str(len(self.data))
            path_name += ", " + ", ".join(v["types"]) + "]"
            paths.append(path)
            display_names.append(path_name)
        if "root_name" not in dir_tree_kwargs:
            dir_tree_kwargs["root_name"] = "/"
        if "sort" not in dir_tree_kwargs:
            dir_tree_kwargs["sort"] = False
        if "display_names" not in dir_tree_kwargs:
            dir_tree_kwargs["display_names"] = display_names
        if "length_limit" not in dir_tree_kwargs:
            dir_tree_kwargs["length_limit"] = None
        print(dir_tree_from_paths(paths, **dir_tree_kwargs))

    def join(self, separator: tp.Optional[str] = None) -> str:
        """Join string data items into a single string.

        If no separator is provided, the method infers one based on the trailing characters of each string:

        * Uses an empty string if all items end with a newline, tab, or space.
        * Uses ', ' if all items end with '}' or ']'.
        * Otherwise, uses two newlines.

        If the resulting string starts with '{' and ends with '}', it is converted to use square brackets.

        Args:
            separator (Optional[str]): Separator to insert between data items.

        Returns:
            str: Resulting concatenated string.
        """
        if len(self.data) == 0:
            return ""
        if len(self.data) == 1:
            return self.data[0]
        if separator is None:
            use_empty_separator = True
            use_comma_separator = True
            for d in self.data:
                if not d.endswith(("\n", "\t", " ")):
                    use_empty_separator = False
                if not d.endswith(("}", "]")):
                    use_comma_separator = False
                if not use_empty_separator and not use_comma_separator:
                    break
            if use_empty_separator:
                separator = ""
            elif use_comma_separator:
                separator = ", "
            else:
                separator = "\n\n"
        joined = separator.join(self.data)
        if joined.startswith("{") and joined.endswith("}"):
            return "[" + joined + "]"
        return joined

    def embed(
        self,
        to_documents_kwargs: tp.KwargsLike = None,
        wrap_documents: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Optional[tp.MaybeKnowledgeAsset]:
        """Embed documents in the asset.

        Converts the asset's data to `vectorbtpro.utils.knowledge.chatting.TextDocument` format using
        `KnowledgeAsset.to_documents` if needed, then embeds them with
        `vectorbtpro.utils.knowledge.chatting.embed_documents` using provided keyword arguments.
        Optionally unwraps the embedded documents if `wrap_documents` is False.

        Args:
            to_documents_kwargs (KwargsLike): Keyword arguments for `KnowledgeAsset.to_documents`.
            wrap_documents (Optional[bool]): Flag indicating whether to preserve the document embedding structure.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.chatting.embed_documents`.

        Returns:
            Optional[MaybeKnowledgeAsset]: New asset with embedded documents, or None if embedding fails.
        """
        from vectorbtpro.utils.knowledge.chatting import (
            EmbeddedDocument,
            StoreDocument,
            embed_documents,
        )

        if self.data and not isinstance(self.data[0], StoreDocument):
            if to_documents_kwargs is None:
                to_documents_kwargs = {}
            documents = self.to_documents(**to_documents_kwargs)
            if wrap_documents is None:
                wrap_documents = False
        else:
            documents = self.data
            if wrap_documents is None:
                wrap_documents = True
        embedded_documents = embed_documents(documents, **kwargs)
        if embedded_documents is None:
            return None
        if not wrap_documents:

            def _unwrap(document):
                if isinstance(document, EmbeddedDocument):
                    return document.replace(
                        document=_unwrap(document.document),
                        child_documents=[_unwrap(d) for d in document.child_documents],
                    )
                if isinstance(document, StoreDocument):
                    return document.data
                return document

            embedded_documents = list(map(_unwrap, embedded_documents))
        return self.replace(data=embedded_documents)

    def rank(
        self,
        query: str,
        to_documents_kwargs: tp.KwargsLike = None,
        wrap_documents: tp.Optional[bool] = None,
        cache_documents: bool = False,
        cache_key: tp.Optional[str] = None,
        asset_cache_manager: tp.Optional[tp.MaybeType[AssetCacheManager]] = None,
        asset_cache_manager_kwargs: tp.KwargsLike = None,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.MaybeKnowledgeAsset:
        """Rank documents by their similarity to a query.

        Converts the asset's data to `vectorbtpro.utils.knowledge.chatting.TextDocument` format using
        `KnowledgeAsset.to_documents` if necessary, then ranks the documents with
        `vectorbtpro.utils.knowledge.chatting.rank_documents` using provided keyword arguments.
        If caching is enabled with `cache_documents` and `cache_key`, the generated text documents are
        stored or loaded via an asset cache manager.

        Args:
            query (str): Query string to rank document relevance.
            to_documents_kwargs (KwargsLike): Keyword arguments for `KnowledgeAsset.to_documents`.
            wrap_documents (Optional[bool]): Flag indicating whether to preserve the document embedding structure.
            cache_documents (bool): If True, will use an asset cache manager to cache the generated
                text documents after conversion.

                Running the same method again will use the cached documents.
            cache_key (Optional[str]): Unique identifier for the cached asset.
            asset_cache_manager (Optional[MaybeType[AssetCacheManager]]): Class or instance of `AssetCacheManager`.
            asset_cache_manager_kwargs (KwargsLike): Keyword arguments to initialize or update `asset_cache_manager`.
            silence_warnings (bool): Flag to suppress warning messages.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.chatting.rank_documents`.

        Returns:
            MaybeKnowledgeAsset: New asset with documents ranked based on similarity to the query.
        """
        from vectorbtpro.utils.knowledge.chatting import (
            ScoredDocument,
            StoreDocument,
            rank_documents,
        )

        if cache_documents:
            if asset_cache_manager is None:
                asset_cache_manager = AssetCacheManager
            if asset_cache_manager_kwargs is None:
                asset_cache_manager_kwargs = {}
            if isinstance(asset_cache_manager, type):
                checks.assert_subclass_of(
                    asset_cache_manager, AssetCacheManager, "asset_cache_manager"
                )
                asset_cache_manager = asset_cache_manager(**asset_cache_manager_kwargs)
            else:
                checks.assert_instance_of(
                    asset_cache_manager, AssetCacheManager, "asset_cache_manager"
                )
                if asset_cache_manager_kwargs:
                    asset_cache_manager = asset_cache_manager.replace(**asset_cache_manager_kwargs)
        documents = None
        if cache_documents and cache_key is not None:
            documents = asset_cache_manager.load_asset(cache_key)
            if documents is not None:
                if wrap_documents is None:
                    wrap_documents = False
            else:
                if not silence_warnings:
                    warn("Caching documents...")
        if documents is None:
            if self.data and not isinstance(self.data[0], StoreDocument):
                if to_documents_kwargs is None:
                    to_documents_kwargs = {}
                documents = self.to_documents(**to_documents_kwargs)
                if (
                    cache_documents
                    and cache_key is not None
                    and isinstance(documents, KnowledgeAsset)
                ):
                    asset_cache_manager.save_asset(documents, cache_key)
                if wrap_documents is None:
                    wrap_documents = False
            else:
                documents = self.data
                if wrap_documents is None:
                    wrap_documents = True
        if "bm25_mirror_store_id" not in kwargs:
            kwargs["bm25_mirror_store_id"] = cache_key
        ranked_documents = rank_documents(query=query, documents=documents, **kwargs)
        if not wrap_documents:

            def _unwrap(document):
                if isinstance(document, ScoredDocument):
                    return document.replace(
                        document=_unwrap(document.document),
                        child_documents=[_unwrap(d) for d in document.child_documents],
                    )
                if isinstance(document, StoreDocument):
                    return document.data
                return document

            ranked_documents = list(map(_unwrap, ranked_documents))
        return self.replace(data=ranked_documents)

    def to_context(
        self,
        *args,
        dump_all: tp.Optional[bool] = None,
        separator: tp.Optional[str] = None,
        **kwargs,
    ) -> str:
        """Convert the asset to a context string.

        Based on the `dump_all` flag, calls either `KnowledgeAsset.dump_all` or `KnowledgeAsset.dump`
        with provided arguments. The dumped data is then joined using `KnowledgeAsset.join` with
        the specified separator.

        Args:
            *args: Positional arguments for the dump function.
            dump_all (Optional[bool]): Flag determining which dump method to use.
            separator (Optional[str]): Separator used for joining dumped data.
            **kwargs: Keyword arguments for the dump function.

        Returns:
            str: Resulting context string.
        """
        from vectorbtpro.utils.knowledge.chatting import (
            EmbeddedDocument,
            ScoredDocument,
            StoreDocument,
        )

        if dump_all is None:
            dump_all = (
                len(self.data) > 1
                and not isinstance(self.data[0], (StoreDocument, EmbeddedDocument, ScoredDocument))
                and separator is None
            )
        if dump_all:
            dumped = self.dump_all(*args, **kwargs)
        else:
            dumped = self.dump(*args, **kwargs)
        if isinstance(dumped, str):
            return dumped
        if not isinstance(dumped, KnowledgeAsset):
            dumped = self.replace(data=dumped)
        return dumped.join(separator=separator)

    def print(self, *args, **kwargs) -> None:
        """Print the asset as a context string.

        Calls `KnowledgeAsset.to_context` with provided arguments to generate a context string,
        which is then printed.

        Args:
            *args: Positional arguments for `KnowledgeAsset.to_context`.
            **kwargs: Keyword arguments for `KnowledgeAsset.to_context`.

        Returns:
            None
        """
        print(self.to_context(*args, **kwargs))
