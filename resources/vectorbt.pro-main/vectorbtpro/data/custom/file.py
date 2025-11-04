# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `FileData` class for handling file data."""

import re
from glob import glob
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import key_dict
from vectorbtpro.data.custom.local import LocalData
from vectorbtpro.utils import checks

__all__ = [
    "FileData",
]

__pdoc__ = {}

FileDataT = tp.TypeVar("FileDataT", bound="FileData")


class FileData(LocalData):
    """Data class for fetching file data from the filesystem via `LocalData`.

    !!! info
        For default settings, see `custom.file` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.file")

    @classmethod
    def is_dir_match(cls, path: tp.PathLike) -> bool:
        """Determine whether the given directory matches the criteria.

        Args:
            path (PathLike): Directory path to evaluate.

        Returns:
            bool: True if the directory matches the criteria, False otherwise.
        """
        return False

    @classmethod
    def is_file_match(cls, path: tp.PathLike) -> bool:
        """Determine whether the given file matches the criteria.

        Args:
            path (PathLike): File path to evaluate.

        Returns:
            bool: True if the file matches the criteria, False otherwise.
        """
        return True

    @classmethod
    def match_path(
        cls,
        path: tp.PathLike,
        match_regex: tp.Optional[str] = None,
        sort_paths: bool = True,
        recursive: bool = True,
        extension: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.List[Path]:
        """Obtain a list of file system paths that match the specified criteria.

        If the provided path is a directory and `FileData.is_dir_match` returns True, the
        directory is returned directly. Otherwise, if the path is a directory, each file within
        it is checked using `FileData.is_file_match`. When a regex pattern is provided via
        `match_regex`, only paths matching the pattern are included.

        Args:
            path (PathLike): Starting path to search for files.
            match_regex (Optional[str]): Regular expression for filtering paths.
            sort_paths (bool): Flag indicating whether to sort the resulting paths.
            recursive (bool): Whether to search subdirectories recursively.
            extension (Optional[str]): Specifies a file extension to filter by (without the dot).
            **kwargs: Additional keyword arguments.

        Returns:
            List[Path]: List of paths that match the specified criteria.
        """
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists():
            if path.is_dir() and not cls.is_dir_match(path):
                sub_paths = []
                for p in path.iterdir():
                    if p.is_dir() and cls.is_dir_match(p):
                        sub_paths.append(p)
                    if p.is_file() and cls.is_file_match(p):
                        if extension is None or p.suffix == "." + extension:
                            sub_paths.append(p)
            else:
                sub_paths = [path]
        else:
            sub_paths = list([Path(p) for p in glob(str(path), recursive=recursive)])
        if match_regex is not None:
            sub_paths = [p for p in sub_paths if re.match(match_regex, str(p))]
        if sort_paths:
            sub_paths = sorted(sub_paths)
        return sub_paths

    @classmethod
    def list_paths(cls, path: tp.PathLike = ".", **match_path_kwargs) -> tp.List[Path]:
        """List all feature or symbol paths under the specified directory.

        Args:
            path (PathLike): Path in which to search for feature or symbol data.
            **match_path_kwargs: Keyword arguments for `FileData.match_path`.

        Returns:
            List[Path]: List of matching paths.
        """
        return cls.match_path(path, **match_path_kwargs)

    @classmethod
    def path_to_key(cls, path: tp.PathLike, **kwargs) -> str:
        """Convert the given file path to a key by extracting its stem.

        Args:
            path (PathLike): File path to convert.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Resulting key derived from the file path.
        """
        return Path(path).stem

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        paths: tp.Any = None,
    ) -> tp.Kwargs:
        return LocalData.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )

    @classmethod
    def pull(
        cls: tp.Type[FileDataT],
        keys: tp.Union[tp.MaybeKeys] = None,
        *,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[tp.MaybeFeatures] = None,
        symbols: tp.Union[tp.MaybeSymbols] = None,
        paths: tp.Any = None,
        match_paths: tp.Optional[bool] = None,
        match_regex: tp.Optional[str] = None,
        sort_paths: tp.Optional[bool] = None,
        match_path_kwargs: tp.KwargsLike = None,
        path_to_key_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> FileDataT:
        """Pull file data from the filesystem based on provided keys or paths.

        Overrides `vectorbtpro.data.base.Data.pull` to handle file system paths.

        Use one of `features`, `symbols`, or `paths` to specify the file location. The file paths
        may be provided as strings, `pathlib.Path` objects, or glob patterns.

        Set `match_paths` to False to bypass path parsing and perform a standard pull operation from
        `vectorbtpro.data.base.Data`.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).
            keys_are_features (Optional[bool]): Flag indicating whether the keys represent features.
            features (MaybeFeatures): Feature identifier(s).
            symbols (MaybeSymbols): Symbol identifier(s).
            paths (Any): One or more file paths or glob patterns.
            match_paths (Optional[bool]): Flag to enable or disable path matching.
            match_regex (Optional[str]): Regular expression for filtering paths.
            sort_paths (Optional[bool]): Indicates if the matched paths should be sorted.
            match_path_kwargs (KwargsLike): Keyword arguments for `FileData.match_path`.
            path_to_key_kwargs (KwargsLike): Keyword arguments for `FileData.path_to_key`.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.file.FileData.pull`.

        Returns:
            FileData: Instance of `FileData` containing the pulled file data.
        """
        keys_meta = cls.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
            paths=paths,
        )
        keys = keys_meta["keys"]
        keys_are_features = keys_meta["keys_are_features"]
        dict_type = keys_meta["dict_type"]

        match_paths = cls.resolve_custom_setting(match_paths, "match_paths")
        match_regex = cls.resolve_custom_setting(match_regex, "match_regex")
        sort_paths = cls.resolve_custom_setting(sort_paths, "sort_paths")

        if match_paths:
            sync = False
            if paths is None:
                paths = keys
                sync = True
            elif keys is None:
                sync = True
            if paths is None:
                if keys_are_features:
                    raise ValueError("At least features or paths must be set")
                else:
                    raise ValueError("At least symbols or paths must be set")
            if match_path_kwargs is None:
                match_path_kwargs = {}
            if path_to_key_kwargs is None:
                path_to_key_kwargs = {}

            single_key = False
            if isinstance(keys, (str, Path)):
                # Single key
                keys = [keys]
                single_key = True

            single_path = False
            if isinstance(paths, (str, Path)):
                # Single path
                paths = [paths]
                single_path = True
                if sync:
                    single_key = True

            cls.check_dict_type(paths, "paths", dict_type=dict_type)
            if isinstance(paths, key_dict):
                # Dict of path per key
                if sync:
                    keys = list(paths.keys())
                elif len(keys) != len(paths):
                    if keys_are_features:
                        raise ValueError("Number of features must be equal to the number of matched paths")
                    else:
                        raise ValueError("Number of symbols must be equal to the number of matched paths")
            elif checks.is_iterable(paths) or checks.is_sequence(paths):
                # Multiple paths
                matched_paths = [
                    p
                    for sub_path in paths
                    for p in cls.match_path(
                        sub_path,
                        match_regex=match_regex,
                        sort_paths=sort_paths,
                        **match_path_kwargs,
                    )
                ]
                if len(matched_paths) == 0:
                    raise FileNotFoundError(f"No paths could be matched with {paths}")
                if sync:
                    keys = []
                    paths = key_dict()
                    for p in matched_paths:
                        s = cls.path_to_key(p, **path_to_key_kwargs)
                        keys.append(s)
                        paths[s] = p
                elif len(keys) != len(matched_paths):
                    if keys_are_features:
                        raise ValueError("Number of features must be equal to the number of matched paths")
                    else:
                        raise ValueError("Number of symbols must be equal to the number of matched paths")
                else:
                    paths = key_dict({s: matched_paths[i] for i, s in enumerate(keys)})
                if len(matched_paths) == 1 and single_path:
                    paths = matched_paths[0]
            else:
                raise TypeError(f"Path '{paths}' is not supported")
            if len(keys) == 1 and single_key:
                keys = keys[0]

        return super(FileData, cls).pull(
            keys,
            keys_are_features=keys_are_features,
            path=paths,
            **kwargs,
        )
