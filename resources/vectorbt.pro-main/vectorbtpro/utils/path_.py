# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with file and directory paths.

!!! info
    For default settings, see `vectorbtpro._settings.path`.
"""

import os
import shutil
from glob import glob
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import humanize

from vectorbtpro import _typing as tp

__all__ = [
    "get_platform_dir",
    "list_any_files",
    "list_files",
    "list_dirs",
    "file_exists",
    "dir_exists",
    "file_size",
    "dir_size",
    "make_file",
    "make_dir",
    "remove_file",
    "remove_dir",
    "print_dir_tree",
]


def get_platform_dir(
    dir_type: tp.Optional[str] = None, per_vbt_version: tp.Optional[bool] = None, **kwargs
) -> str:
    """Return a platform-specific directory.

    Args:
        dir_type (Optional[str]): Type of directory to retrieve (e.g., 'user_data_dir').
        per_vbt_version (Optional[bool]): Whether to create a VBT-version-specific directory.
        **kwargs: Keyword arguments for `platformdirs.PlatformDirs`.

    Returns:
        str: Path to the platform-specific directory.

    !!! info
        For default settings, see `platformdirs` in `vectorbtpro._settings.path`.
    """
    from vectorbtpro._settings import settings
    from vectorbtpro._version import __version__
    from vectorbtpro.utils.config import merge_dicts
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("platformdirs")
    from platformdirs import PlatformDirs

    platformdirs_cfg = settings["path"]["platformdirs"]
    kwargs = merge_dicts(platformdirs_cfg, kwargs)
    def_dir_type = kwargs.pop("dir_type", "user_data_dir")
    if dir_type is None:
        dir_type = def_dir_type
    def_per_vbt_version = kwargs.pop("per_vbt_version", False)
    if per_vbt_version is None:
        per_vbt_version = def_per_vbt_version
    if per_vbt_version:
        kwargs["version"] = __version__

    dirs = PlatformDirs(**kwargs)
    return Path(getattr(dirs, dir_type))


def list_any_files(path: tp.Optional[tp.PathLike] = None, recursive: bool = False) -> tp.List[Path]:
    """Return a list of files and directories matching a given path.

    Args:
        path (Optional[PathLike]): Path or directory to search.

            If omitted, the current working directory is used.
        recursive (bool): Whether to search subdirectories recursively.

    Returns:
        List[Path]: List of matching file and directory paths.
    """
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
    if path.exists() and path.is_dir():
        if recursive:
            path = path / "**" / "*"
        else:
            path = path / "*"
    return [Path(p) for p in glob(str(path), recursive=recursive)]


def list_files(path: tp.Optional[tp.PathLike] = None, recursive: bool = False) -> tp.List[Path]:
    """Return a list of file paths matching a given pattern by filtering results from `list_any_files`.

    Args:
        path (Optional[PathLike]): Path or directory to search.

            If omitted, the current working directory is used.
        recursive (bool): Whether to search subdirectories recursively.

    Returns:
        List[Path]: List of file paths.
    """
    return [p for p in list_any_files(path, recursive=recursive) if p.is_file()]


def list_dirs(path: tp.Optional[tp.PathLike] = None, recursive: bool = False) -> tp.List[Path]:
    """Return a list of directory paths matching a given pattern by filtering results from `list_any_files`.

    Args:
        path (Optional[PathLike]): Path or directory to search.

            If omitted, the current working directory is used.
        recursive (bool): Whether to search subdirectories recursively.

    Returns:
        List[Path]: List of directory paths.
    """
    return [p for p in list_any_files(path, recursive=recursive) if p.is_dir()]


def file_exists(file_path: tp.PathLike) -> bool:
    """Check if the specified file exists.

    Args:
        file_path (PathLike): Path of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    file_path = Path(file_path)
    if file_path.exists() and file_path.is_file():
        return True
    return False


def dir_exists(dir_path: tp.PathLike) -> bool:
    """Check if the specified directory exists.

    Args:
        dir_path (PathLike): Path of the directory to check.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    dir_path = Path(dir_path)
    if dir_path.exists() and dir_path.is_dir():
        return True
    return False


def file_size(file_path: tp.PathLike, readable: bool = True, **kwargs) -> tp.Union[str, int]:
    """Return the size of the specified file, either as a human-readable string or as a number of bytes.

    Args:
        file_path (PathLike): Path of the file.
        readable (bool): Whether to use a human-readable format.
        **kwargs: Keyword arguments for `humanize.naturalsize`.

    Returns:
        Union[str, int]: File size in bytes or as a human-readable string.
    """
    file_path = Path(file_path)
    if not file_exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")
    n_bytes = file_path.stat().st_size
    if readable:
        return humanize.naturalsize(n_bytes, **kwargs)
    return n_bytes


def dir_size(dir_path: tp.PathLike, readable: bool = True, **kwargs) -> tp.Union[str, int]:
    """Return the total size of all files within the specified directory,
    either as a human-readable string or as a number of bytes.

    Args:
        dir_path (PathLike): Path of the directory.
        readable (bool): Whether to use a human-readable format.
        **kwargs: Keyword arguments for `humanize.naturalsize`.

    Returns:
        Union[str, int]: Cumulative size of the directory in bytes or as a human-readable string.
    """
    dir_path = Path(dir_path)
    if not dir_exists(dir_path):
        raise FileNotFoundError(f"Directory '{dir_path}' not found")
    n_bytes = sum(path.stat().st_size for path in dir_path.glob("**/*") if path.is_file())
    if readable:
        return humanize.naturalsize(n_bytes, **kwargs)
    return n_bytes


def check_mkdir(
    dir_path: tp.PathLike,
    mkdir: tp.Optional[bool] = None,
    mode: tp.Optional[int] = None,
    parents: tp.Optional[bool] = None,
    exist_ok: tp.Optional[bool] = None,
) -> None:
    """Ensure that the specified directory exists or create it if necessary.

    Args:
        dir_path (PathLike): Directory path to check.
        mkdir (Optional[bool]): Whether to create the directory if it does not exist.

            If None, the default setting is used.
        mode (Optional[int]): Mode for the directory if created.

            If None, the default setting is used.
        parents (Optional[bool]): Whether to create parent directories if needed.

            If None, the default setting is used.
        exist_ok (Optional[bool]): Whether to ignore an error if the directory already exists.

            If None, the default setting is used.

    Returns:
        None

    !!! info
        For default settings, see `mkdir` in `vectorbtpro._settings.path`.
    """
    from vectorbtpro._settings import settings

    mkdir_cfg = settings["path"]["mkdir"]

    if mkdir is None:
        mkdir = mkdir_cfg["mkdir"]
    if mode is None:
        mode = mkdir_cfg["mode"]
    if parents is None:
        parents = mkdir_cfg["parents"]
    if exist_ok is None:
        exist_ok = mkdir_cfg["exist_ok"]

    dir_path = Path(dir_path)
    if dir_path.exists() and not dir_path.is_dir():
        raise TypeError(f"Path '{dir_path}' is not a directory")
    if not dir_path.exists() and not mkdir:
        raise FileNotFoundError(f"Directory '{dir_path}' not found. Use mkdir=True to proceed.")
    dir_path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)


def make_file(file_path: tp.PathLike, mode: int = 0o666, exist_ok: bool = True, **kwargs) -> Path:
    """Create an empty file at the specified path.

    Args:
        file_path (PathLike): Path of the file to create.
        mode (int): Permission mode for the file.
        exist_ok (bool): Whether to do nothing if the file already exists.
        **kwargs: Keyword arguments for `check_mkdir`.

    Returns:
        Path: Path of the created file.
    """
    file_path = Path(file_path)
    check_mkdir(file_path.parent, **kwargs)
    file_path.touch(mode=mode, exist_ok=exist_ok)
    return file_path


def make_dir(dir_path: tp.PathLike, **kwargs) -> Path:
    """Create a directory at the specified path.

    Args:
        dir_path (PathLike): Path of the directory to create.
        **kwargs: Keyword arguments for `check_mkdir`.

    Returns:
        Path: Path of the created directory.
    """
    check_mkdir(dir_path, mkdir=True, **kwargs)
    return dir_path


def remove_file(file_path: tp.PathLike, missing_ok: bool = False) -> None:
    """Delete the specified file.

    Args:
        file_path (PathLike): Path of the file to delete.
        missing_ok (bool): If True, do not raise an error if the file is not found.

    Returns:
        None
    """
    file_path = Path(file_path)
    if file_exists(file_path):
        file_path.unlink()
    elif not missing_ok:
        raise FileNotFoundError(f"File '{file_path}' not found")


def remove_dir(
    dir_path: tp.PathLike, missing_ok: bool = False, with_contents: bool = False
) -> None:
    """Delete the specified directory.

    Args:
        dir_path (PathLike): Path of the directory to delete.
        missing_ok (bool): If True, do not raise an error if the directory is not found.
        with_contents (bool): If True, delete directories that contain files;
            otherwise, raise an error if the directory is not empty.

    Returns:
        None
    """
    dir_path = Path(dir_path)
    if dir_exists(dir_path):
        if any(dir_path.iterdir()) and not with_contents:
            raise ValueError(
                f"Directory '{dir_path}' has contents. Use with_contents=True to proceed."
            )
        shutil.rmtree(dir_path)
    elif not missing_ok:
        raise FileNotFoundError(f"Directory '{dir_path}' not found")


def get_common_prefix(paths: tp.Iterable[tp.PathLike]) -> str:
    """Return the common prefix shared by a list of URLs or file paths.

    Args:
        paths (Iterable[PathLike]): Iterable of URLs or file paths.

    Returns:
        str: Common prefix as a URL or file path string, or an empty string if no common prefix exists.
    """
    if not paths:
        raise ValueError("Path list is empty")
    paths = [str(path) for path in paths]
    first = paths[0]
    parsed_first = urlparse(first)
    is_url = parsed_first.scheme != ""

    for path in paths:
        parsed = urlparse(path)
        if (parsed.scheme != parsed_first.scheme) or (
            parsed.scheme != "" and parsed.netloc != parsed_first.netloc
        ):
            return ""

    if is_url:
        parsed_urls = [urlparse(p) for p in paths]
        scheme = parsed_urls[0].scheme
        netloc = parsed_urls[0].netloc
        paths_split = [pu.path.strip("/").split("/") for pu in parsed_urls]
        min_length = min(len(p) for p in paths_split)
        common_components = []
        for i in range(min_length):
            current_component = paths_split[0][i]
            if all(p[i] == current_component for p in paths_split):
                common_components.append(current_component)
            else:
                break
        if common_components:
            common_path = "/" + "/".join(common_components) + "/"
        else:
            common_path = "/"
        common_url = urlunparse((scheme, netloc, common_path, "", "", ""))
        return common_url
    else:
        try:
            common_path = os.path.commonpath(paths)
            if not common_path.endswith(os.path.sep):
                common_path += os.path.sep
            return common_path
        except ValueError:
            return ""


def dir_tree_from_paths(
    paths: tp.Iterable[tp.PathLike],
    root: tp.Optional[tp.PathLike] = None,
    root_name: tp.Optional[str] = None,
    show_root: bool = True,
    root_as_item: bool = False,
    display_names: tp.Optional[tp.Iterable[str]] = None,
    name_formatter: tp.Optional[tp.Callable[[str, bool], str]] = None,
    level: int = -1,
    limit_to_dirs: bool = False,
    length_limit: tp.Optional[int] = 1000,
    sort: bool = True,
    space: str = "    ",
    branch: str = "│   ",
    tee: str = "├── ",
    last: str = "└── ",
    directories_name: str = "directories",
    files_name: str = "files",
    print_stats: bool = True,
) -> str:
    """Generate a visual tree structure from provided file system paths.

    This function builds a tree representation based on an iterable of paths.
    The tree is constructed relative to a given root.

    Args:
        paths (Iterable[PathLike]): Iterable of file system paths.
        root (Optional[PathLike]): Root path to which the tree structure should be relative.
        root_name (Optional[str]): Custom name for the root of the tree.
        show_root (bool): If True, show the root name in output.
        root_as_item (bool): If True, treat root as a list item with prefix; if False, show root without prefix.
        display_names (Optional[Iterable[str]]): List of display names corresponding to each path.

            By default, the name of each path is used.
        name_formatter (Optional[Callable[[str, bool], str]]): Mapping or function to format display names.

            Takes (display_name, is_directory) and returns formatted string.
        level (int): Maximum depth level to display.

            A negative value indicates no limit.
        limit_to_dirs (bool): If True, the tree only includes directories.
        length_limit (Optional[int]): Limits the total number of lines in the generated tree.
        sort (bool): If True, sorts tree entries alphabetically.
        space (str): Indentation string used for level spacing.
        branch (str): String representing a branch segment.
        tee (str): String for an intermediate node in the tree.
        last (str): String for the last node in a branch.
        directories_name (str): Name for the directory count in the output.
        files_name (str): Name for the file count in the output.
        print_stats (bool): If True, prints the count of directories and files at the end.

    Returns:
        str: String representing the visual tree structure.
    """
    resolved_paths = []
    for p in paths:
        if not isinstance(p, Path):
            parsed_url = urlparse(str(p))
            p = Path(parsed_url.path)
        resolved_paths.append(p.resolve())
    if display_names is None:
        display_names = [p.name for p in resolved_paths]
    path_display_map = {path: name for path, name in zip(resolved_paths, display_names)}
    if root is None:
        try:
            common_path_str = get_common_prefix(resolved_paths)
            root = Path(common_path_str).resolve()
        except ValueError:
            root = Path(".").resolve()
    else:
        if not isinstance(root, Path):
            parsed_url = urlparse(str(root))
            root = Path(parsed_url.path)
        root = root.resolve()

    dirs = set()
    path_set = set(resolved_paths)
    for path in resolved_paths:
        for parent in path.parents:
            if parent in path_set:
                dirs.add(parent)

    tree = {}
    for path in resolved_paths:
        try:
            relative_path = path.relative_to(root)
        except ValueError:
            continue
        if relative_path == Path("."):
            continue
        parts = relative_path.parts
        if not parts:
            continue
        current_level = tree
        for part in parts[:-1]:
            current_level = current_level.setdefault(part, {})
        last_part = parts[-1]
        if path in dirs:
            current_level.setdefault(last_part, {})
        else:
            current_level[last_part] = None

    files = 0
    dir_count = 0

    def _inner(current_tree, prefix="", current_lvl=-1, current_path=root):
        nonlocal files, dir_count
        if current_lvl == 0:
            return
        entries = list(current_tree.items())
        if sort:
            entries.sort(key=lambda x: (not isinstance(x[1], dict), x[0].lower()))
        if limit_to_dirs:
            entries = [e for e in entries if isinstance(e[1], dict)]
        pointers = [tee] * (len(entries) - 1) + [last] if entries else []

        for pointer, (name, subtree) in zip(pointers, entries):
            child_path = current_path / name
            display_name = path_display_map.get(child_path, name)
            is_directory = isinstance(subtree, dict)
            if name_formatter:
                display_name = name_formatter(display_name, is_directory)
            yield prefix + pointer + display_name

            if isinstance(subtree, dict):
                dir_count += 1
                extension = branch if pointer == tee else space
                yield from _inner(
                    subtree,
                    prefix=prefix + extension,
                    current_lvl=(current_lvl - 1 if current_lvl > 0 else -1),
                    current_path=child_path,
                )
            elif not limit_to_dirs:
                files += 1

    result_lines = []
    if show_root:
        root_display_name = root_name if root_name is not None else root.name
        if name_formatter:
            root_display_name = name_formatter(root_display_name, True)
        if root_as_item:
            result_lines.append(tee + root_display_name)
            root_prefix = space
        else:
            result_lines.append(root_display_name)
            root_prefix = ""
    else:
        root_prefix = ""
    iterator = _inner(tree, prefix=root_prefix, current_lvl=level, current_path=root)
    if length_limit is not None:
        iterator = islice(iterator, length_limit)
    for line in iterator:
        result_lines.append(line)
    if print_stats:
        if next(iterator, None):
            result_lines.append(f"... length_limit {length_limit} reached, counts:")
        result_lines.append(
            f"\n{dir_count} {directories_name}" + (f", {files} {files_name}" if files else "")
        )
    return "\n".join(result_lines)


def dir_tree(dir_path: Path, **kwargs) -> str:
    """Generate a visual tree structure for a directory.

    This function generates a tree representation for a given directory by recursively scanning its contents.
    Internally, it calls `dir_tree_from_paths`.

    Args:
        dir_path (Path): Directory path for which to generate the tree.
        **kwargs: Keyword arguments for `dir_tree_from_paths`.

    Returns:
        str: String representing the visual tree structure.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{dir_path}' not found")
    if not dir_path.is_dir():
        raise TypeError(f"Path '{dir_path}' is not a directory")
    paths = list(dir_path.rglob("*"))
    return dir_tree_from_paths(paths=paths, root=dir_path, **kwargs)


def print_dir_tree(*args, **kwargs) -> None:
    """Print a visual tree structure for a directory.

    This function generates a tree representation for a directory using `dir_tree`
    and prints it to the standard output.

    Args:
        *args: Positional arguments for `dir_tree`.
        **kwargs: Keyword arguments for `dir_tree`.

    Returns:
        None
    """
    print(dir_tree(*args, **kwargs))
