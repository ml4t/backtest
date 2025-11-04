# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing custom asset classes.

See `vectorbtpro.utils.knowledge` for the toy dataset.
"""

import base64
import hashlib
import inspect
import io
import os
import pkgutil
import re
import tempfile
import webbrowser
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse, urlunparse

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import (
    HybridConfig,
    SpecSettingsPath,
    flat_merge_dicts,
    merge_dicts,
    reorder_list,
)
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.knowledge.asset_pipelines import BasicAssetPipeline, EarlyReturn
from vectorbtpro.utils.knowledge.base_assets import AssetCacheManager, KnowledgeAsset
from vectorbtpro.utils.knowledge.formatting import FormatHTML
from vectorbtpro.utils.module_ import get_caller_qualname, prepare_refname
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.path_ import check_mkdir, dir_tree_from_paths, get_common_prefix, remove_dir
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.pickling import suggest_compression
from vectorbtpro.utils.search_ import find, replace
from vectorbtpro.utils.template import CustomTemplate
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "VBTAsset",
    "PagesAsset",
    "MessagesAsset",
    "ExamplesAsset",
    "find_api",
    "find_docs",
    "find_messages",
    "find_examples",
    "find_assets",
    "chat_about",
    "search",
    "quick_search",
    "chat",
    "quick_chat",
]


__pdoc__ = {}

class_abbr_config = HybridConfig(
    dict(
        Accessor={"acc"},
        Array={"arr"},
        ArrayWrapper={"wrapper"},
        Benchmark={"bm"},
        Cacheable={"ca"},
        Chunkable={"ch"},
        Drawdowns={"dd"},
        Jitable={"jit"},
        Figure={"fig"},
        MappedArray={"ma"},
        NumPy={"np"},
        Numba={"nb"},
        Optimizer={"opt"},
        Pandas={"pd"},
        Portfolio={"pf"},
        ProgressBar={"pbar"},
        Registry={"reg"},
        Returns_={"ret"},
        Returns={"rets"},
        QuantStats={"qs"},
        Signals_={"sig"},
    )
)
"""_"""

__pdoc__["class_abbr_config"] = f"""Configuration for class name abbreviations.

```python
{class_abbr_config.prettify_doc()}
```
"""


class NoItemFoundError(Exception):
    """Exception raised when no matching data item is found."""


class MultipleItemsFoundError(Exception):
    """Exception raised when multiple matching data items are found."""


VBTAssetT = tp.TypeVar("VBTAssetT", bound="VBTAsset")


class VBTAsset(KnowledgeAsset):
    """Class for working with VBT content.

    Args:
        *args: Positional arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.
        release_name (Optional[str]): Release name.
        **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`.

    !!! info
        For default settings, see `assets.vbt` in `vectorbtpro._settings.knowledge`.
    """

    _settings_path: tp.SettingsPath = "knowledge.assets.vbt"

    def __init__(self, *args, release_name: tp.Optional[str] = None, **kwargs) -> None:
        KnowledgeAsset.__init__(self, *args, release_name=release_name, **kwargs)

        self._release_name = release_name

    @property
    def release_name(self) -> tp.Optional[str]:
        """Release name of the asset.

        Returns:
            Optional[str]: Release name of the asset, or None if not set.
        """
        return self._release_name

    @classmethod
    def pull(
        cls: tp.Type[VBTAssetT],
        release_name: tp.Optional[str] = None,
        asset_name: tp.Optional[str] = None,
        repo_owner: tp.Optional[str] = None,
        repo_name: tp.Optional[str] = None,
        token: tp.Optional[str] = None,
        token_required: tp.Optional[bool] = None,
        use_pygithub: tp.Optional[bool] = None,
        chunk_size: tp.Optional[int] = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Optional[bool] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> VBTAssetT:
        """Build a `VBTAsset` instance from a JSON asset in a GitHub release.

        Args:
            release_name (Optional[str]): GitHub release specification ('latest', 'current',
                or a specific tag such as 'v2024.12.15').
            asset_name (Optional[str]): Name of the asset file (e.g. 'messages.json.zip').

                You can find all asset file names at https://github.com/polakowo/vectorbt.pro/releases/latest
            repo_owner (Optional[str]): Owner of the GitHub repository.
            repo_name (Optional[str]): Name of the GitHub repository.
            token (Optional[str]): GitHub authentication token.

                It doesn't have to be provided if the asset has already been downloaded.
            token_required (Optional[bool]): Flag indicating whether a GitHub token is required.
            use_pygithub (Optional[bool]): Use the PyGithub library to fetch release data.

                If True, uses https://github.com/PyGithub/PyGithub (otherwise requests)
            chunk_size (Optional[int]): Number of bytes per download chunk.
            cache (Optional[bool]): Flag to determine whether to use the cache directory.
            cache_dir (Optional[PathLike]): Directory for saving JSON asset files (`assets_dir` in settings).
            cache_mkdir_kwargs (KwargsLike): Keyword arguments for cache directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            clear_cache (Optional[bool]): Remove the cache directory before operation if True.
            show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
            pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

                See `vectorbtpro.utils.pbar.ProgressBar`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `VBTAsset.from_json_file` or `VBTAsset.from_json_bytes`.

        Returns:
            VBTAsset: New VBT asset built from the downloaded JSON asset.
        """
        import requests

        release_name = cls.resolve_setting(release_name, "release_name")
        asset_name = cls.resolve_setting(asset_name, "asset_name")
        repo_owner = cls.resolve_setting(repo_owner, "repo_owner")
        repo_name = cls.resolve_setting(repo_name, "repo_name")
        token = cls.resolve_setting(token, "token")
        token_required = cls.resolve_setting(token_required, "token_required")
        use_pygithub = cls.resolve_setting(use_pygithub, "use_pygithub")
        chunk_size = cls.resolve_setting(chunk_size, "chunk_size")
        cache = cls.resolve_setting(cache, "cache")
        assets_dir = cls.resolve_setting(cache_dir, "assets_dir")
        cache_mkdir_kwargs = cls.resolve_setting(
            cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True
        )
        clear_cache = cls.resolve_setting(clear_cache, "clear_cache")
        show_progress = cls.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = cls.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = cls.resolve_setting(template_context, "template_context", merge=True)

        if release_name is None or release_name.lower() == "current":
            from vectorbtpro._version import __release__

            release_name = __release__
        if release_name.lower() == "latest":
            if token is None:
                token = os.environ.get("GITHUB_TOKEN", None)
            if token is None and token_required:
                raise ValueError("GitHub token is required")
            if use_pygithub is None:
                from vectorbtpro.utils.module_ import check_installed

                use_pygithub = check_installed("github")
            if use_pygithub:
                from vectorbtpro.utils.module_ import assert_can_import

                assert_can_import("github")
                from github import Auth, Github
                from github.GithubException import UnknownObjectException

                if token is not None:
                    g = Github(auth=Auth.Token(token))
                else:
                    g = Github()
                try:
                    repo = g.get_repo(f"{repo_owner}/{repo_name}")
                except UnknownObjectException:
                    raise Exception(
                        f"Repository '{repo_owner}/{repo_name}' not found or access denied"
                    )
                try:
                    release = repo.get_latest_release()
                except UnknownObjectException:
                    raise Exception("Latest release not found")
                release_name = release.title
            else:
                headers = {"Accept": "application/vnd.github+json"}
                if token is not None:
                    headers["Authorization"] = f"token {token}"
                release_url = (
                    f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
                )
                response = requests.get(release_url, headers=headers)
                response.raise_for_status()
                release_info = response.json()
                release_name = release_info.get("name")

        template_context = flat_merge_dicts(dict(release_name=release_name), template_context)
        if isinstance(assets_dir, CustomTemplate):
            cache_dir = cls.get_setting("cache_dir")
            if isinstance(cache_dir, CustomTemplate):
                cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
            template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
            release_dir = cls.get_setting("release_dir")
            if isinstance(release_dir, CustomTemplate):
                release_dir = release_dir.substitute(template_context, eval_id="release_dir")
            template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
            assets_dir = assets_dir.substitute(template_context, eval_id="assets_dir")
        if cache:
            if assets_dir.exists():
                if clear_cache:
                    remove_dir(assets_dir, missing_ok=True, with_contents=True)
                else:
                    cache_file = None
                    for file in assets_dir.iterdir():
                        if file.is_file() and file.name == asset_name:
                            cache_file = file
                            break
                    if cache_file is not None:
                        return cls.from_json_file(cache_file, release_name=release_name, **kwargs)

        if token is None:
            token = os.environ.get("GITHUB_TOKEN", None)
        if token is None and token_required:
            raise ValueError("GitHub token is required")
        if use_pygithub is None:
            from vectorbtpro.utils.module_ import check_installed

            use_pygithub = check_installed("github")
        if use_pygithub:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("github")
            from github import Auth, Github
            from github.GithubException import UnknownObjectException

            if token is not None:
                g = Github(auth=Auth.Token(token))
            else:
                g = Github()
            try:
                repo = g.get_repo(f"{repo_owner}/{repo_name}")
            except UnknownObjectException:
                raise Exception(f"Repository '{repo_owner}/{repo_name}' not found or access denied")
            releases = repo.get_releases()
            found_release = None
            for release in releases:
                if release.title == release_name:
                    found_release = release
            if found_release is None:
                raise Exception(f"Release '{release_name}' not found")
            release = found_release
            assets = release.get_assets()
            if asset_name is not None:
                asset = next((a for a in assets if a.name == asset_name), None)
                if asset is None:
                    raise Exception(f"Asset '{asset_name}' not found in release {release_name}")
            else:
                assets_list = list(assets)
                if len(assets_list) == 1:
                    asset = assets_list[0]
                else:
                    raise Exception("Please specify asset_name")
            asset_url = asset.url
        else:
            headers = {"Accept": "application/vnd.github+json"}
            if token is not None:
                headers["Authorization"] = f"token {token}"
            releases_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
            response = requests.get(releases_url, headers=headers)
            response.raise_for_status()
            releases = response.json()
            release_info = None
            for release in releases:
                if release.get("name") == release_name:
                    release_info = release
            if release_info is None:
                raise ValueError(f"Release '{release_name}' not found")
            assets = release_info.get("assets", [])
            if asset_name is not None:
                asset = next((a for a in assets if a["name"] == asset_name), None)
                if asset is None:
                    raise Exception(f"Asset '{asset_name}' not found in release {release_name}")
            else:
                if len(assets) == 1:
                    asset = assets[0]
                else:
                    raise Exception("Please specify asset_name")
            asset_url = asset["url"]

        asset_headers = {"Accept": "application/octet-stream"}
        if token is not None:
            asset_headers["Authorization"] = f"token {token}"
        asset_response = requests.get(asset_url, headers=asset_headers, stream=True)
        asset_response.raise_for_status()
        file_size = int(asset_response.headers.get("Content-Length", 0))
        if file_size == 0:
            file_size = asset.get("size", 0)
        if show_progress is None:
            show_progress = True
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                unit="iB",
                unit_scale=True,
                prefix=f"Downloading {asset_name}",
            ),
            pbar_kwargs,
        )

        if cache:
            check_mkdir(assets_dir, **cache_mkdir_kwargs)
            cache_file = assets_dir / asset_name
            with open(cache_file, "wb") as f:
                with ProgressBar(
                    total=file_size, show_progress=show_progress, **pbar_kwargs
                ) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return cls.from_json_file(cache_file, release_name=release_name, **kwargs)
        else:
            with io.BytesIO() as bytes_io:
                with ProgressBar(
                    total=file_size, show_progress=show_progress, **pbar_kwargs
                ) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            bytes_io.write(chunk)
                            pbar.update(len(chunk))
                bytes_ = bytes_io.getvalue()
            compression = suggest_compression(asset_name)
            if compression is not None and "compression" not in kwargs:
                kwargs["compression"] = compression
            return cls.from_json_bytes(bytes_, release_name=release_name, **kwargs)

    def find_link(
        self: VBTAssetT,
        link: tp.MaybeList[str],
        mode: str = "end",
        per_path: bool = False,
        single_item: bool = True,
        consolidate: bool = True,
        allow_empty: bool = False,
        **kwargs,
    ) -> tp.MaybeVBTAsset:
        """Return asset item(s) matching the provided link(s).

        Args:
            link (MaybeList[str]): Link or list of links to search for.
            mode (str): Search mode.

                See `vectorbtpro.utils.search_.find`.
            per_path (bool): Whether to perform the search per specified path.
            single_item (bool): Indicates if only a single matching item is expected.
            consolidate (bool): If True, consolidates matches sharing the same top parent link.
            allow_empty (bool): If True, returns an empty result when no match is found.
            **kwargs: Keyword arguments for `VBTAsset.find`.

        Returns:
            MaybeVBTAsset: New VBT asset with matching asset item(s).
        """

        def _extend_link(link):
            if not urlparse(link).fragment:
                if link.endswith("/"):
                    return [link, link[:-1]]
                return [link, link + "/"]
            return [link]

        links = link
        if mode.lower() in ("exact", "end"):
            if isinstance(link, str):
                links = _extend_link(link)
            elif isinstance(link, list):
                from itertools import chain

                links = list(chain(*map(_extend_link, link)))
            else:
                raise TypeError("Link must be either string or list")
        found = self.find(
            links, path="link", mode=mode, per_path=per_path, single_item=single_item, **kwargs
        )
        if isinstance(found, (type(self), list)):
            if len(found) == 0:
                if allow_empty:
                    return found
                raise NoItemFoundError(f"No item matching '{link}'")
            if single_item and len(found) > 1:
                if consolidate:
                    top_parents = self.get_top_parent_links(list(found))
                    if len(top_parents) == 1:
                        for i, d in enumerate(found):
                            if d["link"] == top_parents[0]:
                                if isinstance(found, type(self)):
                                    return found.replace(data=[d], single_item=True)
                                return d
                links_block = "\n".join([d["link"] for d in found])
                raise MultipleItemsFoundError(f"Multiple items matching '{link}':\n\n{links_block}")
        elif found is None:
            if allow_empty:
                return found
            raise NoItemFoundError(f"No item matching '{link}'")
        return found

    @classmethod
    def minimize_link(cls, link: str, rules: tp.Optional[tp.Dict[str, str]] = None) -> str:
        """Return a minimized version of the given link by applying regex replacement rules.

        Args:
            link (str): Link to minimize.
            rules (Optional[Dict[str, str]]): Dictionary of regex replacement rules.

        Returns:
            str: Minimized link.
        """
        rules = cls.resolve_setting(rules, "minimize_link_rules", merge=True)

        for k, v in rules.items():
            link = replace(k, v, link, mode="regex")
        return link

    def minimize_metadata(self, keys: tp.Optional[tp.List[str]] = None) -> tp.MaybeVBTAsset:
        """Return a minimized asset with metadata reduced to essential information.

        Args:
            keys (Optional[List[str]]): List of metadata keys to retain.

        Returns:
            MaybeVBTAsset: New VBT asset with minimized metadata.
        """
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc

        keys = self.resolve_setting(keys, "minimize_keys")

        pipeline = BasicAssetPipeline()
        pipeline.add_task(lambda d: EarlyReturn(d) if not isinstance(d, dict) else d)
        pipeline.add_task(
            "find_remove", partial(FindRemoveAssetFunc.is_empty_func, skip_keys=["content"])
        )
        pipeline.add_task("remove", keys, skip_missing=True)
        return self.apply(pipeline, wrap=True)

    def remove_metadata(self) -> VBTAssetT:
        """Remove metadata from the asset.

        Returns:
            VBTAsset: New VBT asset with metadata removed.
        """
        pipeline = BasicAssetPipeline()
        pipeline.add_task(lambda d: EarlyReturn(d) if not isinstance(d, dict) else d)
        pipeline.add_task("query", "content")
        return self.apply(pipeline, wrap=True)

    def minimize_links(self, rules: tp.Optional[tp.Dict[str, str]] = None) -> tp.MaybeVBTAsset:
        """Return asset with minimized links by applying regex replacement rules.

        Args:
            rules (Optional[Dict[str, str]]): Dictionary of regex replacement rules.

        Returns:
            MaybeVBTAsset: New VBT asset with minimized links.
        """
        rules = self.resolve_setting(rules, "minimize_link_rules", merge=True)

        return self.find_replace(rules, mode="regex")

    def minimize(
        self,
        keys: tp.Optional[tp.List[str]] = None,
        links: tp.Optional[bool] = None,
    ) -> tp.MaybeVBTAsset:
        """Return a minimized asset emphasizing essential information.

        Args:
            keys (Optional[List[str]]): List of keys to retain in the asset.
            links (Optional[bool]): If True, apply link minimization to replace redundant URL prefixes.

        Returns:
            MaybeVBTAsset: New minimized VBT asset.
        """
        keys = self.resolve_setting(keys, "minimize_keys")
        links = self.resolve_setting(links, "minimize_links")

        new_instance = self.minimize_metadata(keys=keys)
        if links:
            return new_instance.minimize_links()
        return new_instance

    def select_previous(self: VBTAssetT, link: str, **kwargs) -> VBTAssetT:
        """Return the asset item immediately preceding the item matching the specified link.

        Args:
            link (str): Link identifying the reference item.
            **kwargs: Keyword arguments for `VBTAsset.find_link`.

        Returns:
            VBTAsset: New VBT asset containing the previous data item.
        """
        d = self.find_link(link, wrap=False, **kwargs)
        d_index = self.index(d)
        new_data = []
        if d_index > 0:
            new_data.append(self.data[d_index - 1])
        return self.replace(data=new_data, single_item=True)

    def select_next(self: VBTAssetT, link: str, **kwargs) -> VBTAssetT:
        """Return the asset item immediately following the item matching the specified link.

        Args:
            link (str): Link identifying the reference item.
            **kwargs: Keyword arguments for `VBTAsset.find_link`.

        Returns:
            VBTAsset: New VBT asset containing the next data item.
        """
        d = self.find_link(link, wrap=False, **kwargs)
        d_index = self.index(d)
        new_data = []
        if d_index < len(self.data) - 1:
            new_data.append(self.data[d_index + 1])
        return self.replace(data=new_data, single_item=True)

    def to_markdown(
        self,
        root_metadata_key: tp.Optional[tp.Key] = None,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.MaybeVBTAsset:
        """Convert asset data to Markdown format using a dedicated conversion function.

        Uses `VBTAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc`.

        Args:
            root_metadata_key (Optional[Key]): Key under which to nest metadata.
            minimize_metadata (Optional[bool]): Whether to minimize metadata.
            minimize_keys (Optional[MaybeList[PathLikeKey]]): Keys specifying which metadata to minimize.
            clean_metadata (Optional[bool]): If True, remove empty metadata fields.
            clean_metadata_kwargs (KwargsLike): Keyword arguments for cleaning metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.
            dump_metadata_kwargs (KwargsLike): Keyword arguments for dumping metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.
            metadata_fence (Optional[str]): Metadata fence to use for formatting.

                Options are "code", "frontmatter", or a custom string.
            **kwargs: Keyword arguments for `VBTAsset.apply`.

        Returns:
            MaybeVBTAsset: New VBT asset converted to Markdown.
        """
        return self.apply(
            "to_markdown",
            root_metadata_key=root_metadata_key,
            minimize_metadata=minimize_metadata,
            minimize_keys=minimize_keys,
            clean_metadata=clean_metadata,
            clean_metadata_kwargs=clean_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            metadata_fence=metadata_fence,
            **kwargs,
        )

    @classmethod
    def links_to_paths(
        cls,
        urls: tp.Iterable[str],
        extension: tp.Optional[str] = None,
        allow_fragments: bool = True,
        use_hash: bool = False,
    ) -> tp.List[Path]:
        """Convert a collection of URLs into corresponding filesystem paths.

        Args:
            urls (Iterable[str]): Iterable of URL strings.
            extension (Optional[str]): File extension to append to generated file names.
            allow_fragments (bool): Whether URL fragments are allowed in path generation.
            use_hash (bool): If True, use a hash of the URL as filename to avoid long paths.

        Returns:
            List[Path]: List of filesystem paths generated from the URLs.
        """
        url_paths = []
        for url in urls:
            parsed = urlparse(url, allow_fragments=allow_fragments)

            if use_hash:
                file_name = hashlib.sha1(url.encode("utf-8")).hexdigest()
                if extension is not None:
                    file_name += "." + extension
                url_paths.append(Path(file_name))
                continue

            path_parts = [parsed.netloc]
            url_path = parsed.path.strip("/")
            if url_path:
                parts = url_path.split("/")
                if parsed.fragment:
                    path_parts.extend(parts)
                    if extension is not None:
                        file_name = parsed.fragment + "." + extension
                    else:
                        file_name = parsed.fragment
                    path_parts.append(file_name)
                else:
                    if len(parts) > 1:
                        path_parts.extend(parts[:-1])
                    last_part = parts[-1]
                    if extension is not None:
                        file_name = last_part + "." + extension
                    else:
                        file_name = last_part
                    path_parts.append(file_name)
            else:
                if parsed.fragment:
                    if extension is not None:
                        file_name = parsed.fragment + "." + extension
                    else:
                        file_name = parsed.fragment
                    path_parts.append(file_name)
                else:
                    if extension is not None:
                        path_parts.append("index." + extension)
                    else:
                        path_parts.append("index")
            url_paths.append(Path(os.path.join(*path_parts)))
        return url_paths

    def save_to_markdown(
        self,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Optional[bool] = None,
        use_hash: bool = False,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        """Save content as Markdown files.

        Args:
            cache (Optional[bool]): Flag to determine whether to use the cache directory.

                Otherwise, creates a temporary directory.
            cache_dir (Optional[PathLike]): Directory for saving Markdown files (`markdown_dir` in settings).
            cache_mkdir_kwargs (KwargsLike): Keyword arguments for cache directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            clear_cache (Optional[bool]): Remove the cache directory before operation if True.
            use_hash (bool): If True, use a hash of the URL as filename to avoid long paths.
            show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
            pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

                See `vectorbtpro.utils.pbar.ProgressBar`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc`
                and forwarded to `vectorbtpro.utils.knowledge.formatting.to_markdown`.

        Returns:
            Path: Path to the directory where Markdown files are stored.
        """
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc

        cache = self.resolve_setting(cache, "cache")
        markdown_dir = self.resolve_setting(cache_dir, "markdown_dir")
        cache_mkdir_kwargs = self.resolve_setting(
            cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True
        )
        clear_cache = self.resolve_setting(clear_cache, "clear_cache")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        if cache:
            template_context = flat_merge_dicts(
                dict(release_name=self.release_name), template_context
            )
            if isinstance(markdown_dir, CustomTemplate):
                cache_dir = self.get_setting("cache_dir")
                if isinstance(cache_dir, CustomTemplate):
                    cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
                template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
                release_dir = self.get_setting("release_dir")
                if isinstance(release_dir, CustomTemplate):
                    release_dir = release_dir.substitute(template_context, eval_id="release_dir")
                template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
                markdown_dir = markdown_dir.substitute(template_context, eval_id="markdown_dir")
            if markdown_dir.exists():
                if clear_cache:
                    remove_dir(markdown_dir, missing_ok=True, with_contents=True)
            check_mkdir(markdown_dir, **cache_mkdir_kwargs)
        else:
            markdown_dir = Path(tempfile.mkdtemp(prefix=get_caller_qualname() + "_"))
        link_map = {d["link"]: dict(d) for d in self.data}
        url_paths = self.links_to_paths(link_map.keys(), extension="md", use_hash=use_hash)
        url_file_map = dict(zip(link_map.keys(), [markdown_dir / p for p in url_paths]))
        _, kwargs = ToMarkdownAssetFunc.prepare(**kwargs)

        if show_progress is None:
            show_progress = not self.single_item
        prefix = get_caller_qualname().split(".")[-1]
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                prefix=prefix,
            ),
            pbar_kwargs,
        )
        with ProgressBar(total=len(self.data), show_progress=show_progress, **pbar_kwargs) as pbar:
            for d in self.data:
                if not url_file_map[d["link"]].exists():
                    markdown_content = ToMarkdownAssetFunc.call(d, **kwargs)
                    check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                    with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                pbar.update()

        return markdown_dir

    def to_html(
        self: VBTAssetT,
        root_metadata_key: tp.Optional[tp.Key] = None,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeVBTAsset:
        """Convert asset content to HTML.

        Uses `VBTAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc`.

        Args:
            root_metadata_key (Optional[Key]): Key under which to nest metadata.
            minimize_metadata (Optional[bool]): Whether to minimize metadata.
            minimize_keys (Optional[MaybeList[PathLikeKey]]): Keys specifying which metadata to minimize.
            clean_metadata (Optional[bool]): If True, remove empty metadata fields.
            clean_metadata_kwargs (KwargsLike): Keyword arguments for cleaning metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.
            dump_metadata_kwargs (KwargsLike): Keyword arguments for dumping metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.
            metadata_fence (Optional[str]): Metadata fence to use for formatting.

                Options are "code", "frontmatter", or a custom string.
            to_markdown_kwargs (KwargsLike): Keyword arguments for markdown conversion.

                See `vectorbtpro.utils.knowledge.formatting.to_markdown`.
            format_html_kwargs (KwargsLike): Keyword arguments for HTML formatting.

                See `vectorbtpro.utils.knowledge.formatting.format_html`.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.formatting.to_html`.

        Returns:
            MaybeVBTAsset: New VBT asset converted into HTML format.
        """
        return self.apply(
            "to_html",
            root_metadata_key=root_metadata_key,
            minimize_metadata=minimize_metadata,
            minimize_keys=minimize_keys,
            clean_metadata=clean_metadata,
            clean_metadata_kwargs=clean_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            metadata_fence=metadata_fence,
            to_markdown_kwargs=to_markdown_kwargs,
            format_html_kwargs=format_html_kwargs,
            **kwargs,
        )

    @classmethod
    def get_top_parent_links(cls, data: tp.List) -> tp.List[str]:
        """Return links of top-level parent entries.

        Args:
            data (List): List of dictionaries representing asset data; each dictionary should
                contain a "link" key and may include a "parent" key.

        Returns:
            List[str]: List of links for items that have no parent or whose parent is not present
                in the input data.
        """
        link_map = {d["link"]: dict(d) for d in data}
        top_parents = []
        for d in data:
            if d.get("parent", None) is None or d["parent"] not in link_map:
                top_parents.append(d["link"])
        return top_parents

    @property
    def top_parent_links(self) -> tp.List[str]:
        """Top-level parent links.

        Returns:
            List[str]: Top-level parent links derived from the asset data.
        """
        return self.get_top_parent_links(self.data)

    @classmethod
    def replace_urls_in_html(cls, html: str, url_map: dict) -> str:
        """Replace URLs in HTML anchor tags based on a provided mapping.

        Args:
            html (str): HTML content containing `<a href="...">` attributes.
            url_map (dict): Mapping from original URLs to replacement URLs.

        Returns:
            str: Modified HTML with updated anchor tag URLs.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("bs4")

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            original_href = a_tag["href"]
            if original_href in url_map:
                a_tag["href"] = url_map[original_href]
            else:
                try:
                    parsed_href = urlparse(original_href)
                    base_url = urlunparse(parsed_href._replace(fragment=""))
                    if base_url in url_map:
                        new_base_url = url_map[base_url]
                        new_parsed = urlparse(new_base_url)
                        new_parsed = new_parsed._replace(fragment=parsed_href.fragment)
                        new_href = urlunparse(new_parsed)
                        a_tag["href"] = new_href
                except ValueError:
                    pass
        return str(soup)

    def save_to_html(
        self,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Optional[bool] = None,
        use_hash: bool = False,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        return_url_map: bool = False,
        **kwargs,
    ) -> tp.Union[Path, tp.Tuple[Path, dict]]:
        """Save asset content as HTML files and open them in a web browser.

        Args:
            cache (Optional[bool]): Flag to determine whether to use the cache directory.

                Otherwise, creates a temporary directory.
            cache_dir (Optional[PathLike]): Directory for saving HTML files (`html_dir` in settings).
            cache_mkdir_kwargs (KwargsLike): Keyword arguments for cache directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            clear_cache (Optional[bool]): Remove the cache directory before operation if True.
            use_hash (bool): If True, use a hash of the URL as filename to avoid long paths.
            show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
            pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

                See `vectorbtpro.utils.pbar.ProgressBar`.
            template_context (KwargsLike): Additional context for template substitution.
            return_url_map (bool): If True, also return a mapping of links to file paths along
                with the HTML directory path.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc`.

        Returns:
            Union[Path, Tuple[Path, dict]]: Directory where HTML files are stored, and optionally
                a mapping of links to file paths.

        !!! note
            An index page is created if there are multiple top-level parent entries.
        """
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToHTMLAssetFunc

        cache = self.resolve_setting(cache, "cache")
        html_dir = self.resolve_setting(cache_dir, "html_dir")
        cache_mkdir_kwargs = self.resolve_setting(
            cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True
        )
        clear_cache = self.resolve_setting(clear_cache, "clear_cache")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        if cache:
            template_context = flat_merge_dicts(
                dict(release_name=self.release_name), template_context
            )
            if isinstance(html_dir, CustomTemplate):
                cache_dir = self.get_setting("cache_dir")
                if isinstance(cache_dir, CustomTemplate):
                    cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
                template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
                release_dir = self.get_setting("release_dir")
                if isinstance(release_dir, CustomTemplate):
                    release_dir = release_dir.substitute(template_context, eval_id="release_dir")
                template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
                html_dir = html_dir.substitute(template_context, eval_id="html_dir")
            if html_dir.exists():
                if clear_cache:
                    remove_dir(html_dir, missing_ok=True, with_contents=True)
            check_mkdir(html_dir, **cache_mkdir_kwargs)
        else:
            html_dir = Path(tempfile.mkdtemp(prefix=get_caller_qualname() + "_"))
        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = self.top_parent_links
        if len(top_parents) > 1:
            link_map["/"] = {}
        url_paths = self.links_to_paths(link_map.keys(), extension="html", use_hash=use_hash)
        url_file_map = dict(zip(link_map.keys(), [html_dir / p for p in url_paths]))
        url_map = {k: "file://" + str(v.resolve()) for k, v in url_file_map.items()}
        _, kwargs = ToHTMLAssetFunc.prepare(**kwargs)

        if len(top_parents) > 1:
            entry_link = "/"
            if not url_file_map[entry_link].exists():
                html = ToHTMLAssetFunc.call([link_map[link] for link in top_parents], **kwargs)
                html = self.replace_urls_in_html(html, url_map)
                check_mkdir(url_file_map[entry_link].parent, mkdir=True)
                with open(url_file_map[entry_link], "w", encoding="utf-8") as f:
                    f.write(html)

        if show_progress is None:
            show_progress = not self.single_item
        prefix = get_caller_qualname().split(".")[-1]
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                prefix=prefix,
            ),
            pbar_kwargs,
        )
        with ProgressBar(total=len(self.data), show_progress=show_progress, **pbar_kwargs) as pbar:
            for d in self.data:
                if not url_file_map[d["link"]].exists():
                    html = ToHTMLAssetFunc.call(d, **kwargs)
                    html = self.replace_urls_in_html(html, url_map)
                    check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                    with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                        f.write(html)
                pbar.update()

        if return_url_map:
            return html_dir, url_map
        return html_dir

    def browse(
        self,
        entry_link: tp.Optional[str] = None,
        find_kwargs: tp.KwargsLike = None,
        open_browser: tp.Optional[bool] = None,
        **kwargs,
    ) -> Path:
        """Browse one or more HTML pages.

        Opens the default web browser with the specified entry page and returns the directory
        path where HTML files are stored.

        Args:
            entry_link (Optional[str]): Link of the page to display first.

                If None and there are multiple top-level parents, displays them as an index.
                If not None, will be matched using `VBTAsset.find_link`.
            find_kwargs (KwargsLike): Keyword arguments for `VBTAsset.find_link`.
            open_browser (Optional[bool]): Flag indicating whether to open the web browser.
            **kwargs: Keyword arguments for `VBTAsset.save_to_html`.

        Returns:
            Path: Directory path where the HTML files are stored.
        """
        open_browser = self.resolve_setting(open_browser, "open_browser")

        if entry_link is None:
            if len(self.data) == 1:
                entry_link = self.data[0]["link"]
            else:
                top_parents = self.top_parent_links
                if len(top_parents) == 1:
                    entry_link = top_parents[0]
                else:
                    entry_link = "/"
        else:
            if find_kwargs is None:
                find_kwargs = {}
            d = self.find_link(entry_link, wrap=False, **find_kwargs)
            entry_link = d["link"]
        html_dir, url_map = self.save_to_html(return_url_map=True, **kwargs)
        if open_browser:
            webbrowser.open(url_map[entry_link])
        return html_dir

    def display(
        self,
        link: tp.Optional[str] = None,
        find_kwargs: tp.KwargsLike = None,
        open_browser: tp.Optional[bool] = None,
        html_template: tp.Optional[tp.CustomTemplateLike] = None,
        style_extras: tp.Optional[tp.MaybeList[str]] = None,
        head_extras: tp.Optional[tp.MaybeList[str]] = None,
        body_extras: tp.Optional[tp.MaybeList[str]] = None,
        invert_colors: tp.Optional[bool] = None,
        title: str = "",
        **kwargs,
    ) -> Path:
        """Display asset(s) as an HTML page.

        If multiple HTML pages exist, displays them as iframes within a parent HTML page with
        pagination using `vectorbtpro.utils.knowledge.formatting.FormatHTML`. Opens the default web
        browser and returns the file path of the generated HTML page.

        Args:
            link (Optional[str]): Link identifier of the page to display.

                If provided, it is used to locate a target page.
            find_kwargs (KwargsLike): Keyword arguments for `VBTAsset.find_link` when locating the page.
            open_browser (Optional[bool]): Flag indicating whether to open the web browser.
            html_template (Optional[CustomTemplateLike]): Template for HTML formatting,
                as a string, function, or custom template.
            style_extras (Optional[MaybeList[str]]): Extra CSS rules for the `<style>` element.
            head_extras (Optional[MaybeList[str]]): Extra HTML elements to inject into the `<head>` section.
            body_extras (Optional[MaybeList[str]]): Extra content to insert at the end of the `<body>` section.
            invert_colors (Optional[bool]): Flag to enable color inversion.
            title (str): Title of the HTML page.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `VBTAsset.to_html`.

        Returns:
            Path: File path of the generated HTML file.

        !!! note
            The file __won't__ be deleted automatically.
        """
        open_browser = self.resolve_setting(open_browser, "open_browser", sub_path="display")

        if link is not None:
            if find_kwargs is None:
                find_kwargs = {}
            instance = self.find_link(link, **find_kwargs)
        else:
            instance = self
        html = instance.to_html(wrap=False, single_item=True, **kwargs)
        if len(instance) > 1:
            from vectorbtpro.utils.config import ExtSettingsPath

            encoded_pages = map(lambda x: base64.b64encode(x.encode("utf-8")).decode("ascii"), html)
            pages = "[\n" + ",\n".join(f'    "{page}"' for page in encoded_pages) + "\n]"
            ext_settings_paths = []
            for cls_ in type(self).__mro__[::-1]:
                if issubclass(cls_, VBTAsset):
                    if not isinstance(cls_._settings_path, str):
                        raise TypeError(
                            "_settings_path for VBTAsset and its subclasses must be a string"
                        )
                    ext_settings_paths.append((FormatHTML, cls_._settings_path + ".display"))
            with ExtSettingsPath(ext_settings_paths):
                html = FormatHTML(
                    html_template=html_template,
                    style_extras=style_extras,
                    head_extras=head_extras,
                    body_extras=body_extras,
                    invert_colors=invert_colors,
                    auto_scroll=False,
                ).format_html(title=title, pages=pages)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=get_caller_qualname() + "_",
            suffix=".html",
            delete=False,
        ) as f:
            f.write(html)
            file_path = Path(f.name)
        if open_browser:
            webbrowser.open("file://" + str(file_path.resolve()))
        return file_path

    @classmethod
    def prepare_mention_target(
        cls,
        target: str,
        as_code: bool = False,
        as_regex: bool = True,
        allow_prefix: bool = False,
        allow_suffix: bool = False,
    ) -> str:
        """Prepare a mention target for pattern matching.

        Transforms the input target into a regex pattern if `as_regex` is True by escaping special
        characters and optionally adding boundary assertions based on the `allow_prefix` and
        `allow_suffix` flags. Returns the original target if `as_regex` is False.

        Args:
            target (str): Target string to process.
            as_code (bool): Indicates if the target represents code.
            as_regex (bool): Indicates whether to prepare the target as a regex pattern.
            allow_prefix (bool): If True, do not prepend a non-word boundary.
            allow_suffix (bool): If True, do not append a non-word boundary.

        Returns:
            str: Processed mention target.
        """
        if as_regex:
            escaped_target = re.escape(target)
            new_target = ""
            if not allow_prefix and re.match(r"\w", target[0]):
                new_target += r"(?<!\w)"
            new_target += escaped_target
            if not allow_suffix and re.match(r"\w", target[-1]):
                new_target += r"(?!\w)"
            elif not as_code and target[-1] == ".":
                new_target += r"(?=\w)"
            return new_target
        return target

    @classmethod
    def split_class_name(cls, name: str) -> tp.List[str]:
        """Split a CamelCase class name into its constituent parts.

        Args:
            name (str): Class name in CamelCase.

        Returns:
            List[str]: List of substrings representing the parts of the class name.
        """
        return re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z][a-z]+", name)

    @classmethod
    def get_class_abbrs(cls, name: str) -> tp.List[str]:
        """Generate `snake_case` and abbreviated versions of a class name.

        Uses the constituent parts of the class name to produce variations in `snake_case` format,
        including abbreviations based on a predefined configuration.

        Args:
            name (str): Class name.

        Returns:
            List[str]: List of `snake_case` and abbreviated class name variants.
        """
        from itertools import product

        parts = cls.split_class_name(name)

        replacement_lists = []
        for i, part in enumerate(parts):
            replacements = [part.lower()]
            if i == 0 and f"{part}_" in class_abbr_config:
                replacements.extend(class_abbr_config[f"{part}_"])
            if part in class_abbr_config:
                replacements.extend(class_abbr_config[part])
            replacement_lists.append(replacements)
        all_combinations = list(product(*replacement_lists))
        snake_case_names = ["_".join(combo) for combo in all_combinations]

        return snake_case_names

    @classmethod
    def generate_refname_targets(
        cls,
        refname: str,
        resolve: bool = True,
        incl_shortcuts: tp.Optional[bool] = None,
        incl_shortcut_access: tp.Optional[bool] = None,
        incl_shortcut_call: tp.Optional[bool] = None,
        incl_instances: tp.Optional[bool] = None,
        as_code: tp.Optional[bool] = None,
        as_regex: tp.Optional[bool] = None,
        allow_prefix: tp.Optional[bool] = None,
        allow_suffix: tp.Optional[bool] = None,
    ) -> tp.List[str]:
        """Generate reference name targets based on a reference string.

        This method constructs a list of targets representing a module, class, attribute, or callable.
        It optionally includes:

        * Shortcuts from `vectorbtpro as vbt`.
        * Attribute access versions for classes or modules.
        * Callable versions for callables.
        * Common class abbreviation forms via mapped name parts from `class_abbr_config`.

        Each target is formatted using `VBTAsset.prepare_mention_target`.

        Args:
            refname (str): Dot-separated reference name.
            resolve (bool): Whether to resolve annotated reference name parts.
            incl_shortcuts (Optional[bool]): Include shortcuts from `import vectorbtpro as vbt`.
            incl_shortcut_access (Optional[bool]): Include attribute access forms when applicable.
            incl_shortcut_call (Optional[bool]): Include callable forms when applicable.
            incl_instances (Optional[bool]): Include common class abbreviation forms.
            as_code (Optional[bool]): Format the target as code.
            as_regex (Optional[bool]): Format the target as a regular expression.

                For instance, `vbt.PF` may match `vbt.PFO` if RegEx is not used.
            allow_prefix (Optional[bool]): Allow a prefix in target formatting.
            allow_suffix (Optional[bool]): Allow a suffix in target formatting.

        Returns:
            List[str]: Sorted list of generated reference name targets.
        """
        import vectorbtpro as vbt
        from vectorbtpro.utils.module_ import annotate_refname_parts

        incl_shortcuts = cls.resolve_setting(incl_shortcuts, "incl_shortcuts")
        incl_shortcut_access = cls.resolve_setting(incl_shortcut_access, "incl_shortcut_access")
        incl_shortcut_call = cls.resolve_setting(incl_shortcut_call, "incl_shortcut_call")
        incl_instances = cls.resolve_setting(incl_instances, "incl_instances")
        as_code = cls.resolve_setting(as_code, "as_code")
        as_regex = cls.resolve_setting(as_regex, "as_regex")
        allow_prefix = cls.resolve_setting(allow_prefix, "allow_prefix")
        allow_suffix = cls.resolve_setting(allow_suffix, "allow_suffix")

        def _prepare_target(
            target,
            _as_code=as_code,
            _as_regex=as_regex,
            _allow_prefix=allow_prefix,
            _allow_suffix=allow_suffix,
        ):
            return cls.prepare_mention_target(
                target,
                as_code=_as_code,
                as_regex=_as_regex,
                allow_prefix=_allow_prefix,
                allow_suffix=_allow_suffix,
            )

        targets = set()
        new_target = _prepare_target(refname)
        targets.add(new_target)
        refname_parts = refname.split(".")
        if resolve:
            annotated_parts = annotate_refname_parts(refname)
            if len(annotated_parts) >= 2 and isinstance(annotated_parts[-2]["obj"], type):
                cls_refname = ".".join(refname_parts[:-1])
                cls_aliases = {annotated_parts[-2]["name"]}
                attr_aliases = set()
                for k, v in vbt.__dict__.items():
                    v_refname = prepare_refname(v, raise_error=False)
                    if v_refname is not None:
                        if v_refname == cls_refname:
                            cls_aliases.add(k)
                        elif v_refname == refname:
                            attr_aliases.add(k)
                            if incl_shortcuts:
                                new_target = _prepare_target("vbt." + k)
                                targets.add(new_target)
                if incl_shortcuts:
                    for cls_alias in cls_aliases:
                        new_target = _prepare_target(cls_alias + "." + annotated_parts[-1]["name"])
                        targets.add(new_target)
                    for attr_alias in attr_aliases:
                        if incl_shortcut_call and callable(annotated_parts[-1]["obj"]):
                            new_target = _prepare_target(attr_alias + "(")
                            targets.add(new_target)
                if incl_instances:
                    for cls_alias in cls_aliases:
                        for class_abbr in cls.get_class_abbrs(cls_alias):
                            new_target = _prepare_target(
                                class_abbr + "." + annotated_parts[-1]["name"]
                            )
                            targets.add(new_target)
            else:
                if len(refname_parts) >= 2:
                    module_name = ".".join(refname_parts[:-1])
                    attr_name = refname_parts[-1]
                    new_target = _prepare_target(f"from {module_name} import {attr_name}")
                    targets.add(new_target)
                aliases = {annotated_parts[-1]["name"]}
                for k, v in vbt.__dict__.items():
                    v_refname = prepare_refname(v, raise_error=False)
                    if v_refname is not None:
                        if v_refname == refname:
                            aliases.add(k)
                            if incl_shortcuts:
                                new_target = _prepare_target("vbt." + k)
                                targets.add(new_target)
                if incl_shortcuts:
                    for alias in aliases:
                        if incl_shortcut_access and isinstance(
                            annotated_parts[-1]["obj"], (type, ModuleType)
                        ):
                            new_target = _prepare_target(alias + ".")
                            targets.add(new_target)
                        if incl_shortcut_call and callable(annotated_parts[-1]["obj"]):
                            new_target = _prepare_target(alias + "(")
                            targets.add(new_target)
                if incl_instances and isinstance(annotated_parts[-1]["obj"], type):
                    for alias in aliases:
                        for class_abbr in cls.get_class_abbrs(alias):
                            new_target = _prepare_target(class_abbr + " =")
                            targets.add(new_target)
                            new_target = _prepare_target(class_abbr + ".")
                            targets.add(new_target)
        return sorted(targets)

    def generate_mention_targets(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        incl_base_attr: tp.Optional[bool] = None,
        incl_shortcuts: tp.Optional[bool] = None,
        incl_shortcut_access: tp.Optional[bool] = None,
        incl_shortcut_call: tp.Optional[bool] = None,
        incl_instances: tp.Optional[bool] = None,
        as_code: tp.Optional[bool] = None,
        as_regex: tp.Optional[bool] = None,
        allow_prefix: tp.Optional[bool] = None,
        allow_suffix: tp.Optional[bool] = None,
    ) -> tp.List[str]:
        """Generate mention targets for an object.

        This method generates a list of mention targets for a given object or list of objects.
        It first resolves the object reference using `vectorbtpro.utils.module_.prepare_refname`.
        If an attribute is specified, the method checks whether it is defined on the object
        itself or on a base class. When the attribute belongs to a base class and `incl_base_attr` is True,
        targets are generated for both the object attribute and the corresponding base class attribute.
        Final targets are produced via `VBTAsset.generate_refname_targets`.

        Args:
            obj (MaybeList): Object or list of objects to generate mention targets for.
            attr (Optional[str]): Attribute name to target on the object.
            module (Union[None, str, ModuleType]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the object's reference name.
            incl_base_attr (Optional[bool]): Include targets for base class attributes if applicable.
            incl_shortcuts (Optional[bool]): Include shortcuts from `import vectorbtpro as vbt`.
            incl_shortcut_access (Optional[bool]): Include attribute access forms when applicable.
            incl_shortcut_call (Optional[bool]): Include callable forms when applicable.
            incl_instances (Optional[bool]): Include common class abbreviation forms.
            as_code (Optional[bool]): Format the target as code.
            as_regex (Optional[bool]): Format the target as a regular expression.

                For instance, `vbt.PF` may match `vbt.PFO` if RegEx is not used.
            allow_prefix (Optional[bool]): Allow a prefix in target formatting.
            allow_suffix (Optional[bool]): Allow a suffix in target formatting.

        Returns:
            List[str]: List of generated mention targets.
        """
        from vectorbtpro.utils.module_ import prepare_refname

        incl_base_attr = self.resolve_setting(incl_base_attr, "incl_base_attr")

        targets = []
        if not isinstance(obj, list):
            objs = [obj]
        else:
            objs = obj
        for obj in objs:
            obj_refname = prepare_refname(obj, module=module, resolve=resolve)
            if attr is not None:
                checks.assert_instance_of(attr, str, arg_name="attr")
                if isinstance(obj, tuple):
                    attr_obj = (*obj, attr)
                else:
                    attr_obj = (obj, attr)
                base_attr_refname = prepare_refname(attr_obj, module=module, resolve=resolve)
                obj_refname += "." + attr
                if base_attr_refname == obj_refname:
                    obj_refname = base_attr_refname
                    base_attr_refname = None
            else:
                base_attr_refname = None
            targets.extend(
                self.generate_refname_targets(
                    obj_refname,
                    resolve=resolve,
                    incl_shortcuts=incl_shortcuts,
                    incl_shortcut_access=incl_shortcut_access,
                    incl_shortcut_call=incl_shortcut_call,
                    incl_instances=incl_instances,
                    as_code=as_code,
                    as_regex=as_regex,
                    allow_prefix=allow_prefix,
                    allow_suffix=allow_suffix,
                )
            )
            if incl_base_attr and base_attr_refname is not None:
                targets.extend(
                    self.generate_refname_targets(
                        base_attr_refname,
                        resolve=resolve,
                        incl_shortcuts=incl_shortcuts,
                        incl_shortcut_access=incl_shortcut_access,
                        incl_shortcut_call=incl_shortcut_call,
                        incl_instances=incl_instances,
                        as_code=as_code,
                        as_regex=as_regex,
                        allow_prefix=allow_prefix,
                        allow_suffix=allow_suffix,
                    )
                )
        seen = set()
        targets = [x for x in targets if not (x in seen or seen.add(x))]
        return targets

    @classmethod
    def merge_mention_targets(cls, targets: tp.List[str], as_regex: bool = True) -> str:
        """Merge a list of mention target strings into a single regular expression pattern.

        Args:
            targets (List[str]): List of mention target strings.
            as_regex (bool): If True, construct the pattern using regular expressions;
                otherwise, escape the targets.

        Returns:
            str: Combined regular expression pattern.
        """
        if as_regex:
            prefixed_targets = []
            non_prefixed_targets = []
            common_prefix = r"(?<!\w)"
            for target in targets:
                if target.startswith(common_prefix):
                    prefixed_targets.append(target[len(common_prefix) :])
                else:
                    non_prefixed_targets.append(target)
            combined_targets = []
            if prefixed_targets:
                combined_prefixed = "|".join(f"(?:{p})" for p in prefixed_targets)
                combined_targets.append(f"{common_prefix}(?:{combined_prefixed})")
            if non_prefixed_targets:
                combined_non_prefixed = "|".join(f"(?:{p})" for p in non_prefixed_targets)
                combined_targets.append(f"(?:{combined_non_prefixed})")
            if len(combined_targets) == 1:
                return combined_targets[0]
        else:
            combined_targets = [re.escape(target) for target in targets]
        combined_target = "|".join(combined_targets)
        return f"(?:{combined_target})"

    def find_obj_mentions(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        incl_shortcuts: tp.Optional[bool] = None,
        incl_shortcut_access: tp.Optional[bool] = None,
        incl_shortcut_call: tp.Optional[bool] = None,
        incl_instances: tp.Optional[bool] = None,
        incl_custom: tp.Optional[tp.MaybeList[str]] = None,
        is_custom_regex: bool = False,
        as_code: tp.Optional[bool] = None,
        as_regex: tp.Optional[bool] = None,
        allow_prefix: tp.Optional[bool] = None,
        allow_suffix: tp.Optional[bool] = None,
        merge_targets: tp.Optional[bool] = None,
        per_path: bool = False,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = "content",
        return_type: tp.Optional[str] = "item",
        **kwargs,
    ) -> tp.MaybeVBTAsset:
        """Find and return mentions associated with a VBT object.

        Uses `VBTAsset.generate_mention_targets` to obtain mention targets from the given object.
        Custom mention targets can be provided via `incl_custom`, and if these are regular expressions,
        set `is_custom_regex` to True. Depending on the settings for `as_code` and `as_regex`, the search
        is performed using either `VBTAsset.find_code` or `VBTAsset.find`. If `merge_targets` is True, the
        mention targets are consolidated using `VBTAsset.merge_mention_targets`.

        Args:
            obj (MaybeList): Object or list of objects to find mentions for.
            attr (Optional[str]): Attribute name to target on the object.
            module (Union[None, str, ModuleType]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the object's reference name.
            incl_shortcuts (Optional[bool]): Include shortcuts from `import vectorbtpro as vbt`.
            incl_shortcut_access (Optional[bool]): Include attribute access forms when applicable.
            incl_shortcut_call (Optional[bool]): Include callable forms when applicable.
            incl_instances (Optional[bool]): Include common class abbreviation forms.
            incl_custom (Optional[MaybeList[str]]): Additional custom mention targets.
            is_custom_regex (bool): Treat custom mention targets as regular expressions.
            as_code (Optional[bool]): Format the target as code.
            as_regex (Optional[bool]): Format the target as a regular expression.

                For instance, `vbt.PF` may match `vbt.PFO` if RegEx is not used.
            allow_prefix (Optional[bool]): Allow a prefix in target formatting.
            allow_suffix (Optional[bool]): Allow a suffix in target formatting.
            merge_targets (Optional[bool]): Merge mention targets to simplify the search.
            per_path (bool): Whether to perform the search per specified path.
            path (Optional[MaybeList[PathLikeKey]]): Path(s) within the data item to
                search (e.g. "x.y[0].z").
            return_type (Optional[str]): Type of result to return.
            **kwargs: Keyword arguments for `VBTAsset.find` or `VBTAsset.find_code`.

        Returns:
            MaybeVBTAsset: New VBT asset containing the found mentions.
        """
        as_code = self.resolve_setting(as_code, "as_code")
        as_regex = self.resolve_setting(as_regex, "as_regex")
        allow_prefix = self.resolve_setting(allow_prefix, "allow_prefix")
        allow_suffix = self.resolve_setting(allow_suffix, "allow_suffix")
        merge_targets = self.resolve_setting(merge_targets, "merge_targets")

        mention_targets = self.generate_mention_targets(
            obj,
            attr=attr,
            module=module,
            resolve=resolve,
            incl_shortcuts=incl_shortcuts,
            incl_shortcut_access=incl_shortcut_access,
            incl_shortcut_call=incl_shortcut_call,
            incl_instances=incl_instances,
            as_code=as_code,
            as_regex=as_regex,
            allow_prefix=allow_prefix,
            allow_suffix=allow_suffix,
        )
        if incl_custom:

            def _prepare_target(
                target,
                _as_code=as_code,
                _as_regex=as_regex,
                _allow_prefix=allow_prefix,
                _allow_suffix=allow_suffix,
            ):
                return self.prepare_mention_target(
                    target,
                    as_code=_as_code,
                    as_regex=_as_regex,
                    allow_prefix=_allow_prefix,
                    allow_suffix=_allow_suffix,
                )

            if isinstance(incl_custom, str):
                incl_custom = [incl_custom]
            for custom in incl_custom:
                new_target = _prepare_target(custom, _as_regex=is_custom_regex)
                if new_target not in mention_targets:
                    mention_targets.append(new_target)
        if merge_targets:
            mention_targets = self.merge_mention_targets(mention_targets, as_regex=as_regex)
            as_regex = True
        if as_code:
            mentions_asset = self.find_code(
                mention_targets,
                escape_target=not as_regex,
                path=path,
                per_path=per_path,
                return_type=return_type,
                **kwargs,
            )
        elif as_regex:
            mentions_asset = self.find(
                mention_targets,
                mode="regex",
                path=path,
                per_path=per_path,
                return_type=return_type,
                **kwargs,
            )
        else:
            mentions_asset = self.find(
                mention_targets,
                path=path,
                per_path=per_path,
                return_type=return_type,
                **kwargs,
            )
        return mentions_asset

    @classmethod
    def resolve_spec_settings_path(cls) -> dict:
        """Resolve specialized settings paths based on the inheritance hierarchy of `VBTAsset`.

        Iterates through the reversed method resolution order of the class, collecting the `_settings_path`
        attribute from each subclass of `VBTAsset`. The base path is added under the key "knowledge", and the
        same path with a ".chat" suffix is added under "knowledge.chat".

        Returns:
            dict: Dictionary containing specialized settings paths for 'knowledge' and 'knowledge.chat'.
        """
        spec_settings_path = {}
        for cls_ in cls.__mro__[::-1]:
            if issubclass(cls_, VBTAsset):
                if not isinstance(cls_._settings_path, str):
                    raise TypeError(
                        "_settings_path for VBTAsset and its subclasses must be a string"
                    )
                if "knowledge" not in spec_settings_path:
                    spec_settings_path["knowledge"] = []
                spec_settings_path["knowledge"].append(cls_._settings_path)
                if "knowledge.chat" not in spec_settings_path:
                    spec_settings_path["knowledge.chat"] = []
                spec_settings_path["knowledge.chat"].append(cls_._settings_path + ".chat")
        return spec_settings_path

    def embed(
        self, *args, template_context: tp.KwargsLike = None, **kwargs
    ) -> tp.Optional[tp.MaybeVBTAsset]:
        """Embed the instance's documents.

        Args:
            *args: Positional arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.embed`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.embed`.

        Returns:
            Optional[Rankable]: Updated instance with embedded documents, if available.
        """
        template_context = flat_merge_dicts(dict(release_name=self.release_name), template_context)
        spec_settings_path = self.resolve_spec_settings_path()
        if spec_settings_path:
            with SpecSettingsPath(spec_settings_path):
                return KnowledgeAsset.embed(
                    self, *args, template_context=template_context, **kwargs
                )
        return KnowledgeAsset.embed(self, *args, template_context=template_context, **kwargs)

    def rank(self, *args, template_context: tp.KwargsLike = None, **kwargs) -> tp.MaybeVBTAsset:
        """Rank documents based on their relevance to a provided query.

        Args:
            *args: Positional arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.rank`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.rank`.

        Returns:
            Rankable: Updated instance with ranked documents.
        """
        template_context = flat_merge_dicts(dict(release_name=self.release_name), template_context)
        spec_settings_path = self.resolve_spec_settings_path()
        if spec_settings_path:
            with SpecSettingsPath(spec_settings_path):
                return KnowledgeAsset.rank(self, *args, template_context=template_context, **kwargs)
        return KnowledgeAsset.rank(self, *args, template_context=template_context, **kwargs)

    def create_chat(
        self, *args, template_context: tp.KwargsLike = None, **kwargs
    ) -> tp.Completions:
        """Create a chat interface using the generated context.

        Args:
            *args: Positional arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.create_chat`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.create_chat`.

        Returns:
            Completions: Instance of `vectorbtpro.utils.knowledge.chatting.Completions`
                configured with the generated context.
        """
        template_context = flat_merge_dicts(dict(release_name=self.release_name), template_context)
        spec_settings_path = self.resolve_spec_settings_path()
        if spec_settings_path:
            with SpecSettingsPath(spec_settings_path):
                return KnowledgeAsset.create_chat(
                    self, *args, template_context=template_context, **kwargs
                )
        return KnowledgeAsset.create_chat(self, *args, template_context=template_context, **kwargs)

    @hybrid_method
    def chat(
        cls_or_self, *args, template_context: tp.KwargsLike = None, **kwargs
    ) -> tp.MaybeChatOutput:
        """Chat with a language model using the instance as context.

        Args:
            *args: Positional arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.chat`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.chat`.

        Returns:
            MaybeChatOutput: Completion response or a tuple of the response and the chat instance.
        """
        if not isinstance(cls_or_self, type):
            template_context = flat_merge_dicts(
                dict(release_name=cls_or_self.release_name), template_context
            )
        spec_settings_path = cls_or_self.resolve_spec_settings_path()
        if spec_settings_path:
            with SpecSettingsPath(spec_settings_path):
                return KnowledgeAsset.chat.__func__(
                    cls_or_self, *args, template_context=template_context, **kwargs
                )
        return KnowledgeAsset.chat.__func__(
            cls_or_self, *args, template_context=template_context, **kwargs
        )


PagesAssetT = tp.TypeVar("PagesAssetT", bound="PagesAsset")


class PagesAsset(VBTAsset):
    """Class for handling website pages and headings.

    Fields:
        link (str): URL of the page without fragment, such as "https://vectorbt.pro/features/data/",
            or heading with fragment, such as "https://vectorbt.pro/features/data/#trading-view".
        parent (Optional[str]): URL of the parent page or heading.

            For example, a heading 1 is a parent of a heading 2.
        children (List[str]): List of URLs of the child pages and/or headings.

            For example, a heading 2 is a child of a heading 1.
        name (str): Name of the page or heading representing the API object's name, such as "Portfolio.from_signals".
        type (str): Type of the page or heading (e.g., "page", "heading 1").
        icon (Optional[str]): Icon identifier (e.g., "material-brain").
        tags (List[str]): List of tags associated with the page or heading.
        content (Optional[str]): Content of the page or heading; may be None if it solely redirects.
        obj_type (Optional[str]): API type of the represented object (e.g., "property").
        github_link (Optional[str]): URL to the source code of the represented object.

    !!! info
        For default settings, see `assets.pages` in `vectorbtpro._settings.knowledge`.
    """

    _settings_path: tp.SettingsPath = "knowledge.assets.pages"

    def descend_links(self: PagesAssetT, links: tp.List[str], **kwargs) -> PagesAssetT:
        """Return a new asset with redundant links removed via descendant headings.

        Processes the provided list of links and removes links that appear as descendant headings.

        Args:
            links (List[str]): List of link strings to descend.
            **kwargs: Keyword arguments for `replace`.

        Returns:
            PagesAsset: New pages asset with redundant links removed.
        """
        redundant_links = set()
        new_data = {}
        for link in links:
            if link in redundant_links:
                continue
            descendant_headings = self.select_descendant_headings(link, incl_link=True)
            for d in descendant_headings:
                if d["link"] != link:
                    redundant_links.add(d["link"])
                new_data[d["link"]] = d
        for link in links:
            if link in redundant_links and link in new_data:
                del new_data[link]
        return self.replace(data=list(new_data.values()), **kwargs)

    def aggregate_links(
        self: PagesAssetT,
        links: tp.List[str],
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PagesAssetT:
        """Return a new asset with aggregated heading links by removing redundant entries.

        Processes the list of links by aggregating descendant headings and eliminating redundant links.

        Args:
            links (List[str]): List of link strings to aggregate.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            **kwargs: Keyword arguments for `replace`.

        Returns:
            PagesAsset: New pages asset with aggregated links.
        """
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        redundant_links = set()
        new_data = {}
        for link in links:
            if link in redundant_links:
                continue
            descendant_headings = self.select_descendant_headings(link, incl_link=True)
            for d in descendant_headings:
                if d["link"] != link:
                    redundant_links.add(d["link"])
            descendant_headings = descendant_headings.aggregate(**aggregate_kwargs)
            new_data[link] = descendant_headings[0]
        for link in links:
            if link in redundant_links and link in new_data:
                del new_data[link]
        return self.replace(data=list(new_data.values()), **kwargs)

    def find_page(
        self: PagesAssetT,
        link: tp.MaybeList[str],
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        incl_descendants: bool = False,
        single_item: bool = True,
        **kwargs,
    ) -> tp.MaybePagesAsset:
        """Return the page or pages corresponding to the given link(s).

        Searches for pages using the specified link(s) with options to aggregate or include descendant headings.
        Keyword arguments are passed to `VBTAsset.find_link`.

        Args:
            link (MaybeList[str]): Link or list of links to search for.
            aggregate (bool): Whether to aggregate headings into pages.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            incl_descendants (bool): Whether to include descendant headings.
            single_item (bool): Whether to return a single item.
            **kwargs: Keyword arguments for `VBTAsset.find_link`.

        Returns:
            MaybePagesAsset: New pages asset with the found pages or headings.
        """
        found = self.find_link(link, single_item=single_item, **kwargs)
        if not isinstance(found, (type(self), list)):
            return found
        if aggregate:
            return self.aggregate_links(
                [d["link"] for d in found],
                aggregate_kwargs=aggregate_kwargs,
                single_item=single_item,
            )
        if incl_descendants:
            return self.descend_links(
                [d["link"] for d in found],
                single_item=single_item,
            )
        return found

    def find_refname(
        self,
        refname: tp.MaybeList[str],
        **kwargs,
    ) -> tp.MaybePagesAsset:
        """Return the page asset corresponding to the given reference name.

        Transforms the reference name(s) into a link format and searches for the matching page asset.

        Args:
            refname (MaybeList[str]): Reference name or list of reference names.
            **kwargs: Keyword arguments for `PagesAsset.find_page`.

        Returns:
            MaybePagesAsset: New pages asset corresponding to the reference name.
        """
        if isinstance(refname, list):
            link = list(map(lambda x: f"#({re.escape(x)})$", refname))
        else:
            link = f"#({re.escape(refname)})$"
        return self.find_page(link, mode="regex", **kwargs)

    def find_obj(
        self,
        obj: tp.Any,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        **kwargs,
    ) -> tp.MaybePagesAsset:
        """Return the page corresponding to an object or its reference name.

        If an attribute is provided, it is appended to the object reference.
        The object reference is prepared using `vectorbtpro.utils.module_.prepare_refname`.

        Args:
            obj (Any): Object to search for.
            attr (Optional[str]): Attribute name to target on the object.
            module (Union[None, str, ModuleType]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the object's reference name.
            **kwargs: Keyword arguments for `PagesAsset.find_refname`.

        Returns:
            MaybePagesAsset: New pages asset corresponding to the object or reference name.
        """
        if attr is not None:
            checks.assert_instance_of(attr, str, arg_name="attr")
            if isinstance(obj, tuple):
                obj = (*obj, attr)
            else:
                obj = (obj, attr)
        refname = prepare_refname(obj, module=module, resolve=resolve)
        return self.find_refname(refname, **kwargs)

    @classmethod
    def parse_content_links(cls, content: str) -> tp.List[str]:
        """Return all links extracted from the provided content string.

        Parses Markdown-style links from the content and returns the URL components.

        Args:
            content (str): Content string to parse.

        Returns:
            List[str]: List of extracted link URLs.
        """
        link_pattern = r'(?<!\!)\[[^\]]+\]\((\S+?)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\)'
        return re.findall(link_pattern, content)

    @classmethod
    def parse_link_refname(cls, link: str) -> tp.Optional[str]:
        """Return the reference name extracted from the given link, if available.

        Examines the link for an API reference. If a fragment is present and starts with "vectorbtpro",
        returns the fragment as the reference name. Otherwise, constructs a reference name from the link.

        Args:
            link (str): URL to parse.

        Returns:
            Optional[str]: Extracted reference name or None.
        """
        if "/api/" not in link:
            return None
        if "#" in link:
            refname = link.split("#")[1]
            if refname.startswith("vectorbtpro"):
                return refname
            return None
        return "vectorbtpro." + ".".join(link.split("/api/")[1].strip("/").split("/"))

    @classmethod
    def is_link_module(cls, link: str) -> bool:
        """Return True if the given link represents a module, otherwise False.

        Determines if the link is structured as a module reference based on its URL format.

        Args:
            link (str): URL to evaluate.

        Returns:
            bool: True if the link is a module reference, otherwise False.
        """
        if "/api/" not in link:
            return False
        if "#" not in link:
            return True
        refname = link.split("#")[1]
        if "/".join(refname.split(".")) in link:
            return True
        return False

    def find_obj_api(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        only_obj: bool = False,
        use_parent: tp.Optional[bool] = None,
        use_base_parents: tp.Optional[bool] = None,
        use_ref_parents: tp.Optional[bool] = None,
        incl_bases: tp.Union[None, bool, int] = None,
        incl_ancestors: tp.Union[None, bool, int] = None,
        incl_base_ancestors: tp.Union[None, bool, int] = None,
        incl_refs: tp.Union[None, bool, int] = None,
        incl_descendants: tp.Optional[bool] = None,
        incl_ancestor_descendants: tp.Optional[bool] = None,
        incl_ref_descendants: tp.Optional[bool] = None,
        aggregate: tp.Optional[bool] = None,
        aggregate_ancestors: tp.Optional[bool] = None,
        aggregate_refs: tp.Optional[bool] = None,
        aggregate_kwargs: tp.KwargsLike = None,
        topo_sort: tp.Optional[bool] = None,
        return_refname_graph: bool = False,
    ) -> tp.Union[PagesAssetT, tp.Tuple[PagesAssetT, dict]]:
        """Return API pages and headings relevant to the provided object(s).

        Prepares the object reference using `vectorbtpro.utils.module_.prepare_refname` and extends the asset
        with related pages based on various options. This includes incorporating base classes/attributes,
        ancestors, reference descendants, and aggregation of links.

        Args:
            obj (MaybeList): Object or list of objects to find API pages for.
            attr (Optional[str]): Attribute name to target on the object.
            module (Union[None, str, ModuleType]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the object's reference name.
            only_obj (bool): If True, disable all extensions and return only the object itself.
            use_parent (Optional[bool]): Include the object's parent page.
            use_base_parents (Optional[bool]): Include base classes/attributes of the parent.
            use_ref_parents (Optional[bool]): Include reference parent pages.
            incl_bases (Union[None, bool, int]): Include base classes/attributes or specify maximum inheritance level.

                For instance, `vectorbtpro.portfolio.base.Portfolio` has `vectorbtpro.generic.analyzable.Analyzable`
                as one of its base classes. It can also be an integer indicating the maximum inheritance level.
                If `obj` is a module, then bases are sub-modules.
            incl_ancestors (Union[None, bool, int]): Include ancestor pages or specify maximum level.

                For instance, `vectorbtpro.portfolio.base.Portfolio` has `vectorbtpro.portfolio.base` as its ancestor.
                It can also be an integer indicating the maximum level. Provide `incl_base_ancestors`
                to override `incl_ancestors` for base classes/attributes.
            incl_base_ancestors (Union[None, bool, int]): Override `incl_ancestors` for base classes/attributes.
            incl_refs (Union[None, bool, int]): Extend the asset with references from the object's content.

                If True, include reference names; if an integer, limit to the specified maximum reference level.
                Defaults to False for modules and classes, and True otherwise. When reference name resolution
                is disabled, defaults to False.
            incl_descendants (Optional[bool]): Extend the asset with descendant headings.

                Provide `incl_ancestor_descendants` and `incl_ref_descendants` to override `incl_descendants`
                for ancestors and references respectively.
            incl_ancestor_descendants (Optional[bool]): Override descendant inclusion for ancestor headings.
            incl_ref_descendants (Optional[bool]): Override descendant inclusion for reference headings.
            aggregate (Optional[bool]): Aggregate descendant headings into pages for the object and its
                base classes or attributes.

                Provide `aggregate_ancestors` and `aggregate_refs` to override `aggregate` for ancestors
                and references respectively.
            aggregate_ancestors (Optional[bool]): Override aggregation for ancestor headings.
            aggregate_refs (Optional[bool]): Override aggregation for reference headings.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            topo_sort (Optional[bool]): Create a topological graph from reference names and sort pages and headings.

                Set `return_refname_graph` to True to also return the graph.
            return_refname_graph (bool): Return a tuple of the asset and reference name graph if True.

        Returns:
            Union[PagesAsset, Tuple[PagesAsset, dict]]: Asset with relevant API pages and headings,
                optionally accompanied by a reference name graph.
        """
        from vectorbtpro.utils.module_ import annotate_refname_parts, prepare_refname

        incl_bases = self.resolve_setting(incl_bases, "incl_bases")
        incl_ancestors = self.resolve_setting(incl_ancestors, "incl_ancestors")
        incl_base_ancestors = self.resolve_setting(incl_base_ancestors, "incl_base_ancestors")
        incl_refs = self.resolve_setting(incl_refs, "incl_refs")
        incl_descendants = self.resolve_setting(incl_descendants, "incl_descendants")
        incl_ancestor_descendants = self.resolve_setting(
            incl_ancestor_descendants, "incl_ancestor_descendants"
        )
        incl_ref_descendants = self.resolve_setting(incl_ref_descendants, "incl_ref_descendants")
        aggregate = self.resolve_setting(aggregate, "aggregate")
        aggregate_ancestors = self.resolve_setting(aggregate_ancestors, "aggregate_ancestors")
        aggregate_refs = self.resolve_setting(aggregate_refs, "aggregate_refs")
        topo_sort = self.resolve_setting(topo_sort, "topo_sort")

        if only_obj:
            incl_bases = False
            incl_ancestors = False
            incl_base_ancestors = False
            incl_refs = False
            incl_descendants = False
            incl_ancestor_descendants = False
            incl_ref_descendants = False

        base_refnames = []
        base_refnames_set = set()
        if not isinstance(obj, list):
            objs = [obj]
        else:
            objs = obj
        for obj in objs:
            if attr is not None:
                checks.assert_instance_of(attr, str, arg_name="attr")
                if isinstance(obj, tuple):
                    obj = (*obj, attr)
                else:
                    obj = (obj, attr)
            obj_refname = prepare_refname(obj, module=module, resolve=resolve)
            refname_graph = defaultdict(list)
            if resolve:
                annotated_parts = annotate_refname_parts(obj_refname)
                if isinstance(annotated_parts[-1]["obj"], ModuleType):
                    _module = annotated_parts[-1]["obj"]
                    _cls = None
                    _attr = None
                elif isinstance(annotated_parts[-1]["obj"], type):
                    _module = None
                    _cls = annotated_parts[-1]["obj"]
                    _attr = None
                elif len(annotated_parts) >= 2 and isinstance(annotated_parts[-2]["obj"], type):
                    _module = None
                    _cls = annotated_parts[-2]["obj"]
                    _attr = annotated_parts[-1]["name"]
                else:
                    _module = None
                    _cls = None
                    _attr = None
                if use_parent is None:
                    use_parent = _cls is not None and _attr is None
                if not aggregate and not incl_descendants:
                    use_parent = False
                    use_base_parents = False
                if incl_refs is None:
                    incl_refs = _module is None and _cls is None
                if _cls is not None and incl_bases:
                    level_classes = defaultdict(set)
                    visited = set()
                    queue = deque([(_cls, 0)])
                    while queue:
                        current_cls, current_level = queue.popleft()
                        if current_cls in visited:
                            continue
                        visited.add(current_cls)
                        level_classes[current_level].add(current_cls)
                        for base in current_cls.__bases__:
                            queue.append((base, current_level + 1))
                    mro = inspect.getmro(_cls)
                    classes = []
                    levels = list(level_classes.keys())
                    if not isinstance(incl_bases, bool):
                        if isinstance(incl_bases, int):
                            levels = levels[: incl_bases + 1]
                        else:
                            raise TypeError(f"Invalid incl_bases: {incl_bases}")
                    for level in levels:
                        classes.extend([_cls for _cls in mro if _cls in level_classes[level]])
                    for c in classes:
                        if c.__module__.split(".")[0] != "vectorbtpro":
                            continue
                        if _attr is not None:
                            if not hasattr(c, _attr):
                                continue
                            refname = prepare_refname((c, _attr))
                        else:
                            refname = prepare_refname(c)
                        if (use_parent and refname == obj_refname) or use_base_parents:
                            refname = ".".join(refname.split(".")[:-1])
                        if refname not in base_refnames_set:
                            base_refnames.append(refname)
                            base_refnames_set.add(refname)
                            for b in c.__bases__:
                                if b.__module__.split(".")[0] == "vectorbtpro":
                                    if _attr is not None:
                                        if not hasattr(b, _attr):
                                            continue
                                        b_refname = prepare_refname((b, _attr))
                                    else:
                                        b_refname = prepare_refname(b)
                                    if use_base_parents:
                                        b_refname = ".".join(b_refname.split(".")[:-1])
                                    if refname != b_refname:
                                        refname_graph[refname].append(b_refname)
                elif _module is not None and hasattr(_module, "__path__") and incl_bases:
                    base_refnames.append(_module.__name__)
                    base_refnames_set.add(_module.__name__)
                    refname_level = {}
                    refname_level[_module.__name__] = 0
                    for _, refname, _ in pkgutil.walk_packages(
                        _module.__path__, prefix=f"{_module.__name__}."
                    ):
                        if refname not in base_refnames_set:
                            parent_refname = ".".join(refname.split(".")[:-1])
                            if not isinstance(incl_bases, bool):
                                if isinstance(incl_bases, int):
                                    if refname_level[parent_refname] + 1 > incl_bases:
                                        continue
                                else:
                                    raise TypeError(f"Invalid incl_bases: {incl_bases}")
                            base_refnames.append(refname)
                            base_refnames_set.add(refname)
                            refname_level[refname] = refname_level[parent_refname] + 1
                            if parent_refname != refname:
                                refname_graph[parent_refname].append(refname)
                else:
                    base_refnames.append(obj_refname)
                    base_refnames_set.add(obj_refname)
            else:
                if incl_refs is None:
                    incl_refs = False
                base_refnames.append(obj_refname)
                base_refnames_set.add(obj_refname)
        api_asset = self.find_refname(
            base_refnames,
            single_item=False,
            incl_descendants=incl_descendants,
            aggregate=aggregate,
            aggregate_kwargs=aggregate_kwargs,
            allow_empty=True,
            wrap=True,
        )
        if len(api_asset) == 0:
            return api_asset
        if not topo_sort:
            refname_indices = {refname: [] for refname in base_refnames}
            remaining_indices = []
            for i, d in enumerate(api_asset):
                refname = self.parse_link_refname(d["link"])
                if refname is not None:
                    while refname not in refname_indices:
                        if not refname:
                            break
                        refname = ".".join(refname.split(".")[:-1])
                if refname:
                    refname_indices[refname].append(i)
                else:
                    remaining_indices.append(i)
            get_indices = [i for v in refname_indices.values() for i in v] + remaining_indices
            api_asset = api_asset.get_items(get_indices)

        if incl_ancestors or incl_refs:
            refnames_aggregated = {}
            for d in api_asset:
                refname = self.parse_link_refname(d["link"])
                if refname is not None:
                    refnames_aggregated[refname] = aggregate
            to_ref_api_asset = api_asset
            if incl_ancestors:
                anc_refnames = []
                anc_refnames_set = set(refnames_aggregated.keys())
                for d in api_asset:
                    child_refname = refname = self.parse_link_refname(d["link"])
                    if refname is not None:
                        if incl_base_ancestors or refname == obj_refname:
                            refname = ".".join(refname.split(".")[:-1])
                            anc_level = 1
                            while refname:
                                if isinstance(incl_base_ancestors, bool) or refname == obj_refname:
                                    if not isinstance(incl_ancestors, bool):
                                        if isinstance(incl_ancestors, int):
                                            if anc_level > incl_ancestors:
                                                break
                                        else:
                                            raise TypeError(
                                                f"Invalid incl_ancestors: {incl_ancestors}"
                                            )
                                else:
                                    if not isinstance(incl_base_ancestors, bool):
                                        if isinstance(incl_base_ancestors, int):
                                            if anc_level > incl_base_ancestors:
                                                break
                                        else:
                                            raise TypeError(
                                                f"Invalid incl_base_ancestors: {incl_base_ancestors}"
                                            )
                                if refname not in anc_refnames_set:
                                    anc_refnames.append(refname)
                                    anc_refnames_set.add(refname)
                                    if refname != child_refname:
                                        refname_graph[refname].append(child_refname)
                                child_refname = refname
                                refname = ".".join(refname.split(".")[:-1])
                                anc_level += 1
                anc_api_asset = self.find_refname(
                    anc_refnames,
                    single_item=False,
                    incl_descendants=incl_ancestor_descendants,
                    aggregate=aggregate_ancestors,
                    aggregate_kwargs=aggregate_kwargs,
                    allow_empty=True,
                    wrap=True,
                )
                if aggregate_ancestors or incl_ancestor_descendants:
                    obj_index = None
                    for i, d in enumerate(api_asset):
                        d_refname = self.parse_link_refname(d["link"])
                        if d_refname == obj_refname:
                            obj_index = i
                            break
                    if obj_index is not None:
                        del api_asset[obj_index]
                for d in anc_api_asset:
                    refname = self.parse_link_refname(d["link"])
                    if refname is not None:
                        refnames_aggregated[refname] = aggregate_ancestors
                api_asset = anc_api_asset + api_asset

            if incl_refs:
                if not aggregate and not incl_descendants:
                    use_ref_parents = False
                main_ref_api_asset = None
                ref_api_asset = to_ref_api_asset
                while incl_refs:
                    content_refnames = []
                    content_refnames_set = set()
                    for d in ref_api_asset:
                        d_refname = self.parse_link_refname(d["link"])
                        if d_refname is not None:
                            for link in self.parse_content_links(d["content"]):
                                if "/api/" in link:
                                    refname = self.parse_link_refname(link)
                                    if refname is not None:
                                        if use_ref_parents and not self.is_link_module(link):
                                            refname = ".".join(refname.split(".")[:-1])
                                        if refname not in content_refnames_set:
                                            content_refnames.append(refname)
                                            content_refnames_set.add(refname)
                                            if (
                                                d_refname != refname
                                                and refname not in refname_graph
                                            ):
                                                refname_graph[d_refname].append(refname)
                    ref_refnames = []
                    ref_refnames_set = set(refnames_aggregated.keys()) | content_refnames_set
                    for refname in content_refnames:
                        if refname in refnames_aggregated and (
                            refnames_aggregated[refname] or not aggregate_refs
                        ):
                            continue
                        _refname = refname
                        while _refname:
                            _refname = ".".join(_refname.split(".")[:-1])
                            if _refname in ref_refnames_set and refnames_aggregated.get(
                                _refname, aggregate_refs
                            ):
                                break
                        if not _refname:
                            ref_refnames.append(refname)
                    if len(ref_refnames) == 0:
                        break
                    ref_api_asset = self.find_refname(
                        ref_refnames,
                        single_item=False,
                        incl_descendants=incl_ref_descendants,
                        aggregate=aggregate_refs,
                        aggregate_kwargs=aggregate_kwargs,
                        allow_empty=True,
                        wrap=True,
                    )
                    for d in ref_api_asset:
                        refname = self.parse_link_refname(d["link"])
                        if refname is not None:
                            refnames_aggregated[refname] = aggregate_refs
                    if main_ref_api_asset is None:
                        main_ref_api_asset = ref_api_asset
                    else:
                        main_ref_api_asset += ref_api_asset
                    incl_refs -= 1
                if main_ref_api_asset is not None:
                    api_asset += main_ref_api_asset
                    aggregated_refnames_set = set()
                    for refname, aggregated in refnames_aggregated.items():
                        if aggregated:
                            aggregated_refnames_set.add(refname)
                    delete_indices = []
                    for i, d in enumerate(api_asset):
                        refname = self.parse_link_refname(d["link"])
                        if refname is not None:
                            if (
                                not refnames_aggregated[refname]
                                and refname in aggregated_refnames_set
                            ):
                                delete_indices.append(i)
                                continue
                            while refname:
                                refname = ".".join(refname.split(".")[:-1])
                                if refname in aggregated_refnames_set:
                                    break
                        if refname:
                            delete_indices.append(i)
                    if len(delete_indices) > 0:
                        api_asset.delete_items(delete_indices, inplace=True)

        if topo_sort:
            from graphlib import TopologicalSorter

            refname_topo_graph = defaultdict(set)
            refname_topo_sorter = TopologicalSorter(refname_topo_graph)
            for parent_node, child_nodes in refname_graph.items():
                for child_node in child_nodes:
                    refname_topo_sorter.add(child_node, parent_node)
            refname_topo_order = refname_topo_sorter.static_order()
            refname_indices = {refname: [] for refname in refname_topo_order}
            remaining_indices = []
            for i, d in enumerate(api_asset):
                refname = self.parse_link_refname(d["link"])
                if refname is not None:
                    while refname not in refname_indices:
                        if not refname:
                            break
                        refname = ".".join(refname.split(".")[:-1])
                    if refname:
                        refname_indices[refname].append(i)
                    else:
                        remaining_indices.append(i)
                else:
                    remaining_indices.append(i)
            get_indices = [i for v in refname_indices.values() for i in v] + remaining_indices
            api_asset = api_asset.get_items(get_indices)

        if return_refname_graph:
            return api_asset, refname_graph
        return api_asset

    def find_obj_docs(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        incl_pages: tp.Optional[tp.MaybeIterable[str]] = None,
        excl_pages: tp.Optional[tp.MaybeIterable[str]] = None,
        page_find_mode: tp.Optional[str] = None,
        up_aggregate: tp.Optional[bool] = None,
        up_aggregate_th: tp.Union[None, int, float] = None,
        up_aggregate_pages: tp.Optional[bool] = None,
        aggregate: tp.Optional[bool] = None,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybePagesAsset:
        """Find documentation for the given object(s).

        Search for documentation links related to the specified object(s) by applying inclusion and
        exclusion criteria. This method filters documentation pages by checking if their links contain
        parts specified in `incl_pages` and excludes those matching `excl_pages`. Matching is performed
        using `vectorbtpro.utils.search_.find` with `page_find_mode` to determine relevance. For example,
        specifying `excl_pages=["release-notes"]` will omit documentation related to release notes.

        When upward aggregation is enabled via `up_aggregate`, the method aggregates child headings into
        their parent if the count exceeds the threshold defined by `up_aggregate_th` (an integer for an
        absolute count or a float for a relative count). For example, `up_aggregate_th=2/3` means this
        method must find 2 headings out of 3 in order to replace it by the full parent heading/page.
        Similarly, if `up_aggregate_pages` is True, page aggregation is applied. For example, if 2
        tutorial pages out of 3 are matched, the whole tutorial series is used. This strategy
        consolidates related documentation sections.

        If `aggregate` is True, aggregates any descendant headings into pages for this object
        and all base classes/attributes using `PagesAsset.aggregate_links`.

        Args:
            obj (MaybeList): Object or list of objects to find documentation for.
            attr (Optional[str]): Attribute name to target on the object.
            module (Union[None, str, ModuleType]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the object's reference name.
            incl_pages (Optional[MaybeIterable[str]]): Iterable of page identifiers or parts to include.
            excl_pages (Optional[MaybeIterable[str]]): Iterable of page identifiers or parts to exclude.
            page_find_mode (Optional[str]): Mode used for matching pages in `vectorbtpro.utils.search_.find`.
            up_aggregate (Optional[bool]): Whether to aggregate child headings into their parent.
            up_aggregate_th (Union[None, int, float]): Threshold for upward aggregation of headings.
            up_aggregate_pages (Optional[bool]): Whether to aggregate pages.
            aggregate (Optional[bool]): Whether to perform aggregation on matched items.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            **kwargs: Keyword arguments for `PagesAsset.find_obj_mentions`.

        Returns:
            MaybePagesAsset: New pages asset with the found documentation pages or headings.
        """
        incl_pages = self.resolve_setting(incl_pages, "incl_pages")
        excl_pages = self.resolve_setting(excl_pages, "excl_pages")
        page_find_mode = self.resolve_setting(page_find_mode, "page_find_mode")
        up_aggregate = self.resolve_setting(up_aggregate, "up_aggregate")
        up_aggregate_th = self.resolve_setting(up_aggregate_th, "up_aggregate_th")
        up_aggregate_pages = self.resolve_setting(up_aggregate_pages, "up_aggregate_pages")
        aggregate = self.resolve_setting(aggregate, "aggregate")

        if incl_pages is None:
            incl_pages = ()
        elif isinstance(incl_pages, str):
            incl_pages = (incl_pages,)
        if excl_pages is None:
            excl_pages = ()
        elif isinstance(excl_pages, str):
            excl_pages = (excl_pages,)

        def _filter_func(x):
            if "link" not in x:
                return False
            if "/api/" in x["link"]:
                return False
            if excl_pages:
                for page in excl_pages:
                    if find(page, x["link"], mode=page_find_mode):
                        return False
            if incl_pages:
                for page in incl_pages:
                    if find(page, x["link"], mode=page_find_mode):
                        return True
                return False
            return True

        docs_asset = self.filter(_filter_func)
        mentions_asset = docs_asset.find_obj_mentions(
            obj,
            attr=attr,
            module=module,
            resolve=resolve,
            **kwargs,
        )
        if (
            isinstance(mentions_asset, PagesAsset)
            and len(mentions_asset) > 0
            and isinstance(mentions_asset[0], dict)
            and "link" in mentions_asset[0]
        ):
            if up_aggregate:
                link_map = {d["link"]: dict(d) for d in docs_asset.data}
                new_links = {d["link"] for d in mentions_asset}
                while True:
                    parent_map = defaultdict(list)
                    without_parent = set()
                    for link in new_links:
                        if link_map[link]["parent"] is not None:
                            parent_map[link_map[link]["parent"]].append(link)
                        else:
                            without_parent.add(link)
                    _new_links = set()
                    for parent, children in parent_map.items():
                        headings = set()
                        non_headings = set()
                        for child in children:
                            if link_map[child]["type"].startswith("heading"):
                                headings.add(child)
                            else:
                                non_headings.add(child)
                        if up_aggregate_pages:
                            _children = children
                        else:
                            _children = headings
                        if checks.is_float(up_aggregate_th) and 0 <= abs(up_aggregate_th) <= 1:
                            _up_aggregate_th = int(
                                up_aggregate_th * len(link_map[parent]["children"])
                            )
                        elif checks.is_number(up_aggregate_th):
                            if (
                                checks.is_float(up_aggregate_th)
                                and not up_aggregate_th.is_integer()
                            ):
                                raise TypeError(
                                    f"Up-aggregation threshold ({up_aggregate_th}) must be between 0 and 1"
                                )
                            _up_aggregate_th = int(up_aggregate_th)
                        else:
                            raise TypeError("Up-aggregation threshold must be a number")
                        if 0 < len(_children) >= _up_aggregate_th:
                            _new_links.add(parent)
                        else:
                            _new_links |= headings
                        _new_links |= non_headings
                    if _new_links == new_links:
                        break
                    new_links = _new_links | without_parent
                return docs_asset.find_page(
                    list(new_links),
                    single_item=False,
                    aggregate=aggregate,
                    aggregate_kwargs=aggregate_kwargs,
                )
            if aggregate:
                return docs_asset.aggregate_links(
                    [d["link"] for d in mentions_asset],
                    aggregate_kwargs=aggregate_kwargs,
                )
        return mentions_asset

    def browse(
        self,
        entry_link: tp.Optional[str] = None,
        descendants_only: bool = False,
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        """Browse one or more HTML pages.

        Args:
            entry_link (Optional[str]): Link of the page to display first.

                If None and there are multiple top-level parents, displays them as an index.
                If not None, will be matched using `VBTAsset.find_link`.
            descendants_only (bool): If True, only descendants of the entry page are displayed.
            aggregate (bool): Whether to aggregate headings into pages.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            **kwargs: Keyword arguments for `VBTAsset.browse`.

        Returns:
            Path: Directory path where the HTML files are stored.
        """
        new_instance = self
        if entry_link is not None and entry_link != "/" and descendants_only:
            new_instance = new_instance.select_descendants(entry_link, incl_link=True)
        if aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            new_instance = new_instance.aggregate(**aggregate_kwargs)
        return VBTAsset.browse(new_instance, entry_link=entry_link, **kwargs)

    def display(
        self,
        link: tp.Optional[str] = None,
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        """Display page(s) as an HTML page.

        Args:
            link (Optional[str]): Link identifier of the page to display.

                If provided, it is used to locate a target page.
            aggregate (bool): Whether to aggregate headings into pages.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            **kwargs: Keyword arguments for `VBTAsset.display`.

        Returns:
            Path: File path of the generated HTML file.
        """
        new_instance = self
        if link is not None:
            new_instance = new_instance.find_page(
                link,
                aggregate=aggregate,
                aggregate_kwargs=aggregate_kwargs,
            )
        elif aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            new_instance = new_instance.aggregate(**aggregate_kwargs)
        return VBTAsset.display(new_instance, **kwargs)

    def aggregate(
        self: PagesAssetT,
        append_obj_type: tp.Optional[bool] = None,
        append_github_link: tp.Optional[bool] = None,
    ) -> PagesAssetT:
        """Aggregate pages by merging descendant headings into their parent page content.

        This method converts each heading's content into markdown and concatenates it with its
        parent's content, resulting in pages that contain aggregated information. Only parent pages
        and headings without a designated parent remain, while aggregated headings are merged.

        Args:
            append_obj_type (Optional[bool]): If True, append the object type to the heading.
            append_github_link (Optional[bool]): If True, append the GitHub source link to the heading.

        Returns:
            PagesAsset: New pages asset with aggregated page content.
        """
        append_obj_type = self.resolve_setting(append_obj_type, "append_obj_type")
        append_github_link = self.resolve_setting(append_github_link, "append_github_link")

        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = self.top_parent_links
        aggregated_links = set()

        def _aggregate_content(link):
            node = link_map[link]
            content = node["content"]
            if content is None:
                content = ""
            if node["type"].startswith("heading"):
                level = int(node["type"].split(" ")[1])
                heading_markdown = "#" * level + " " + node["name"]
                if append_obj_type and node.get("obj_type", None) is not None:
                    heading_markdown += f" | {node['obj_type']}"
                if append_github_link and node.get("github_link", None) is not None:
                    heading_markdown += f" | [source]({node['github_link']})"
                if content == "":
                    content = heading_markdown
                else:
                    content = f"{heading_markdown}\n\n{content}"

            children = list(node["children"])
            for child in list(children):
                if child in link_map:
                    child_node = link_map[child]
                    child_content = _aggregate_content(child)
                    if child_node["type"].startswith("heading"):
                        if child_content.startswith("# "):
                            content = child_content
                        else:
                            content += f"\n\n{child_content}"
                        children.remove(child)
                        aggregated_links.add(child)

            if content != "":
                node["content"] = content
            node["children"] = children
            return content

        for top_parent in top_parents:
            _aggregate_content(top_parent)

        new_data = [link_map[link] for link in link_map if link not in aggregated_links]
        return self.replace(data=new_data)

    def select_api(self: PagesAssetT) -> PagesAssetT:
        """Return API documentation pages and headings.

        Filters the pages asset to include only those that are part of the API documentation.

        Returns:
            PagesAsset: New pages asset containing only API documentation pages and headings.
        """
        return self.filter(lambda x: "link" in x and "/api/" in x["link"])

    def select_docs(self: PagesAssetT) -> PagesAssetT:
        """Return general documentation pages and headings.

        Filters the pages asset to include only those that are part of the general documentation.

        Returns:
            PagesAsset: New pages asset containing only general documentation pages and headings.
        """
        return self.filter(lambda x: "link" in x and "/api/" not in x["link"])

    def select_parent(
        self: PagesAssetT, link: str, incl_link: bool = False, **kwargs
    ) -> PagesAssetT:
        """Select the parent page of a given link.

        Fetches the parent page of the specified link using `PagesAsset.find_page` and
        returns a new pages asset. If `incl_link` is True, the original page is also included.

        Args:
            link (str): Link of the page.
            incl_link (bool): Indicates whether to include the page corresponding to `link`.
            **kwargs: Keyword arguments for `PagesAsset.find_page`.

        Returns:
            PagesAsset: New pages asset containing the parent page (and optionally the original page).
        """
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        if d.get("parent", None):
            if d["parent"] in link_map:
                new_data.append(link_map[d["parent"]])
        return self.replace(data=new_data, single_item=True)

    def select_children(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Select the child pages of a given link.

        Fetches the child pages of the given link using `PagesAsset.find_page` and
        returns them as a new pages asset. If `incl_link` is True, the original page is also included.

        Args:
            link (str): Link of the page.
            incl_link (bool): Indicates whether to include the page corresponding to `link`.
            **kwargs: Keyword arguments for `PagesAsset.find_page`.

        Returns:
            PagesAsset: New pages asset containing the child pages (and optionally the original page).
        """
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        if d.get("children", []):
            for child in d["children"]:
                if child in link_map:
                    new_data.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_siblings(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Return the sibling pages for the specified page link.

        Args:
            link (str): Unique identifier of the page.
            incl_link (bool): Indicates whether to include the page corresponding to `link`.
            **kwargs: Keyword arguments for `PagesAsset.find_page`.

        Returns:
            PagesAsset: New pages asset containing the sibling pages.
        """
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        if d.get("parent", None):
            if d["parent"] in link_map:
                parent_d = link_map[d["parent"]]
                if parent_d.get("children", []):
                    for child in parent_d["children"]:
                        if incl_link or child != d["link"]:
                            if child in link_map:
                                new_data.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_descendants(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Return all descendant pages of the specified page link.

        Args:
            link (str): Link identifying the starting page.
            incl_link (bool): Indicates whether to include the page corresponding to `link`.
            **kwargs: Keyword arguments for `PagesAsset.find_page`.

        Returns:
            PagesAsset: New pages asset containing the descendant pages.
        """
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        descendants = set()
        stack = [d]
        while stack:
            d = stack.pop()
            children = d.get("children", [])
            for child in children:
                if child in link_map and child not in descendants:
                    descendants.add(child)
                    new_data.append(link_map[child])
                    stack.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_branch(self, link: str, **kwargs) -> PagesAssetT:
        """Return the branch of pages starting from the specified link, including the page itself
        and all its descendants.

        Args:
            link (str): Link identifying the starting page.
            **kwargs: Keyword arguments for `PagesAsset.select_descendants`.

        Returns:
            PagesAsset: New pages asset containing the branch pages.
        """
        return self.select_descendants(link, incl_link=True, **kwargs)

    def select_ancestors(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Return all ancestor pages of the specified page link.

        Args:
            link (str): Link identifying the target page.
            incl_link (bool): Indicates whether to include the page corresponding to `link`.
            **kwargs: Keyword arguments for `PagesAsset.find_page`.

        Returns:
            PagesAsset: New pages asset containing the ancestor pages.
        """
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        ancestors = set()
        parent = d.get("parent", None)
        while parent and parent in link_map:
            if parent in ancestors:
                break
            ancestors.add(parent)
            new_data.append(link_map[parent])
            parent = link_map[parent].get("parent", None)
        return self.replace(data=new_data, single_item=False)

    def select_parent_page(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Return the parent page for the specified link, searching upward until a page
        of type 'page' is encountered.

        Args:
            link (str): Link identifying the target page.
            incl_link (bool): Indicates whether to include the page corresponding to `link`.
            **kwargs: Keyword arguments for `PagesAsset.find_page`.

        Returns:
            PagesAsset: New pages asset containing the parent page information.
        """
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        ancestors = set()
        parent = d.get("parent", None)
        while parent and parent in link_map:
            if parent in ancestors:
                break
            ancestors.add(parent)
            new_data.append(link_map[parent])
            if link_map[parent]["type"] == "page":
                break
            parent = link_map[parent].get("parent", None)
        return self.replace(data=new_data, single_item=False)

    def select_descendant_headings(
        self, link: str, incl_link: bool = False, **kwargs
    ) -> PagesAssetT:
        """Return descendant heading pages for the specified link.

        Args:
            link (str): Link identifying the base page.
            incl_link (bool): Indicates whether to include the page corresponding to `link`.
            **kwargs: Keyword arguments for `PagesAsset.find_page`.

        Returns:
            PagesAsset: New pages asset containing the descendant heading pages.
        """
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        descendants = set()
        stack = [d]
        while stack:
            d = stack.pop()
            children = d.get("children", [])
            for child in children:
                if child in link_map and child not in descendants:
                    if link_map[child]["type"].startswith("heading"):
                        descendants.add(child)
                        new_data.append(link_map[child])
                        stack.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def print_site_schema(
        self,
        append_type: bool = False,
        append_obj_type: bool = False,
        structure_fragments: bool = True,
        split_fragments: bool = True,
        **dir_tree_kwargs,
    ) -> None:
        """Print the site schema in a formatted tree layout.

        The schema displays page links with optional appended types and object types.
        When `structure_fragments` is True, link fragments are organized into a hierarchical structure,
        and if `split_fragments` is enabled, fragments are shown as continuations of their parent links.

        Args:
            append_type (bool): Append the page type to each displayed name.
            append_obj_type (bool): Append the object type to each displayed name when available.
            structure_fragments (bool): Organize link fragments into a hierarchical structure.
            split_fragments (bool): Display fragments as continuations of their parent links.
            **dir_tree_kwargs: Keyword arguments for `vectorbtpro.utils.path_.dir_tree_from_paths`.

        Returns:
            None
        """
        link_map = {d["link"]: dict(d) for d in self.data}
        links = []
        for link, d in link_map.items():
            if not structure_fragments:
                links.append(link)
                continue
            x = d
            link_base = None
            link_fragments = []
            while x["type"].startswith("heading") and "#" in x["link"]:
                link_parts = x["link"].split("#")
                if link_base is None:
                    link_base = link_parts[0]
                link_fragments.append("#" + link_parts[1])
                if not x.get("parent", None) or x["parent"] not in link_map:
                    if x["type"].startswith("heading"):
                        level = int(x["type"].split()[1])
                        for i in range(level - 1):
                            link_fragments.append("?")
                    break
                x = link_map[x["parent"]]
            if link_base is None:
                links.append(link)
            else:
                if split_fragments and len(link_fragments) > 1:
                    link_fragments = link_fragments[::-1]
                    new_link_fragments = [link_fragments[0]]
                    for i in range(1, len(link_fragments)):
                        link_fragment1 = link_fragments[i - 1]
                        link_fragment2 = link_fragments[i]
                        if link_fragment2.startswith(link_fragment1 + "."):
                            new_link_fragments.append(
                                "." + link_fragment2[len(link_fragment1 + ".") :]
                            )
                        else:
                            new_link_fragments.append(link_fragment2)
                    link_fragments = new_link_fragments
                links.append(link_base + "/".join(link_fragments))
        paths = self.links_to_paths(links, allow_fragments=not structure_fragments)

        display_names = []
        for i, d in enumerate(link_map.values()):
            path_name = paths[i].name
            brackets = []
            if append_type:
                brackets.append(d["type"])
            if append_obj_type and d["obj_type"]:
                brackets.append(d["obj_type"])
            if brackets:
                path_name += f" [{', '.join(brackets)}]"
            display_names.append(path_name)
        if "root_name" not in dir_tree_kwargs:
            root_name = get_common_prefix(link_map.keys())
            if not root_name:
                root_name = "/"
            dir_tree_kwargs["root_name"] = root_name
        if "sort" not in dir_tree_kwargs:
            dir_tree_kwargs["sort"] = False
        if "display_names" not in dir_tree_kwargs:
            dir_tree_kwargs["display_names"] = display_names
        if "length_limit" not in dir_tree_kwargs:
            dir_tree_kwargs["length_limit"] = None
        print(dir_tree_from_paths(paths, **dir_tree_kwargs))

    def generate_llmstxt_api(
        self,
        aggregate: bool = True,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Generate API documentation in llms.txt format.

        Args:
            aggregate (bool): Whether to aggregate headings into pages.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            **kwargs: Keyword arguments for `PagesAsset.to_markdown`.

        Returns:
            str: API documentation formatted as llms.txt.
        """
        api_asset = self.select_api().filter("not not content")
        if aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            api_asset = api_asset.aggregate(**aggregate_kwargs)
        return api_asset.to_markdown(**kwargs).join()

    def generate_llmstxt_docs(
        self,
        aggregate: bool = True,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Generate general documentation in llms.txt format.

        Args:
            aggregate (bool): Whether to aggregate headings into pages.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            **kwargs: Keyword arguments for `PagesAsset.to_markdown`.

        Returns:
            str: General documentation formatted as llms.txt.
        """
        docs_asset = self.select_docs().filter("not not content")
        if aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            docs_asset = docs_asset.aggregate(**aggregate_kwargs)
        return docs_asset.to_markdown(**kwargs).join()

    def generate_llmstxt_full(
        self,
        aggregate: bool = True,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Generate full documentation in llms.txt format.

        Args:
            aggregate (bool): Whether to aggregate headings into pages.
            aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
            **kwargs: Keyword arguments for `PagesAsset.to_markdown`.

        Returns:
            str: Full documentation formatted as llms.txt.
        """
        api_asset = self.select_api().filter("not not content")
        if aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            api_asset = api_asset.aggregate(**aggregate_kwargs)
        api_asset = api_asset.to_markdown(**kwargs)
        docs_asset = self.select_docs().filter("not not content")
        if aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            docs_asset = docs_asset.aggregate(**aggregate_kwargs)
        docs_asset = docs_asset.to_markdown(**kwargs)
        return (api_asset + docs_asset).join()


MessagesAssetT = tp.TypeVar("MessagesAssetT", bound="MessagesAsset")


class MessagesAsset(VBTAsset):
    """Class for managing Discord messages.

    Fields:
        link (str): URL of the message, e.g. "https://discord.com/channels/918629562441695344/919715148896301067/923327319882485851".
        block (str): URL of the first message in a block.

            A block is a sequence of messages by the same author that either reference another
            author's message or reference none.
        thread (str): URL of the first message in a thread.

            A thread is a bunch of blocks that reference each other in a chain, such as questions,
            answers, follow-up questions, etc.
        reference (Optional[str]): URL of the referenced message.
        replies (List[str]): URLs of messages that reference this message.
        channel (str): Name of the message channel, e.g. "support".
        timestamp (str): Timestamp of the message in ISO 8601 format, e.g. "2025-06-23".
        author (str): Message author:

            * "@user_n": participant n in this thread
            * "@ext_user_n": (external) mentioned user n who hasn't posted in this thread
            * "@maintainer": project maintainer
        content (str): Text content of the message.
        mentions (List[str]): Discord usernames mentioned in the message, e.g. ["@maintainer"].
        attachments (List[dict]): Attachments with fields "file_name" (e.g. "some_image.png") and
            "content" (extracted file content).
        reactions (int): Total number of reactions received.

    !!! info
        For default settings, see `assets.messages` in `vectorbtpro._settings.knowledge`.
    """

    _settings_path: tp.SettingsPath = "knowledge.assets.messages"

    def latest_first(self, **kwargs) -> tp.MaybeMessagesAsset:
        """Return messages sorted in reverse chronological order.

        Args:
            **kwargs: Keyword arguments for `MessagesAsset.sort`.

        Returns:
            MaybeMessagesAsset: New messages asset sorted with the latest messages first.
        """

        def _sort_key(x):
            path = urlparse(x["link"]).path.rstrip("/")
            parts = [p for p in path.split("/") if p]
            return int(parts[-1])

        return self.sort(source=_sort_key, ascending=False, **kwargs)

    def aggregate_messages(
        self: MessagesAssetT,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
        """Aggregate attachments and metadata for each message.

        Applies `MessagesAsset.apply` with `vectorbtpro.utils.knowledge.custom_asset_funcs.AggMessageAssetFunc`
        to aggregate attachments on a per-message basis. For additional keyword arguments,
        see `MessagesAsset.to_markdown`.

        Args:
            minimize_metadata (Optional[bool]): Whether to minimize metadata.
            minimize_keys (Optional[MaybeList[PathLikeKey]]): Keys specifying which metadata to minimize.
            clean_metadata (Optional[bool]): If True, remove empty metadata fields.
            clean_metadata_kwargs (KwargsLike): Keyword arguments for cleaning metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.
            dump_metadata_kwargs (KwargsLike): Keyword arguments for dumping metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.
            metadata_fence (Optional[str]): Metadata fence to use for formatting.

                Options are "code", "frontmatter", or a custom string.
            to_markdown_kwargs (KwargsLike): Keyword arguments for markdown conversion.

                See `vectorbtpro.utils.knowledge.formatting.to_markdown`.
            **kwargs: Keyword arguments for `MessagesAsset.apply`.

        Returns:
            MaybeMessagesAsset: New messages asset with aggregated messages.
        """
        return self.apply(
            "agg_message",
            minimize_metadata=minimize_metadata,
            minimize_keys=minimize_keys,
            clean_metadata=clean_metadata,
            clean_metadata_kwargs=clean_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            metadata_fence=metadata_fence,
            to_markdown_kwargs=to_markdown_kwargs,
            **kwargs,
        )

    def aggregate_blocks(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeIterable[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
        """Aggregate messages by block.

        Collects messages grouped by the "block" field using `MessagesAsset.collect`
        (with `uniform_groups` defaulting to True) and then applies
        `vectorbtpro.utils.knowledge.custom_asset_funcs.AggBlockAssetFunc` to aggregate each group.

        Args:
            collect_kwargs (KwargsLike): Keyword arguments for `MessagesAsset.collect`.

                If not provided, an empty dict with `uniform_groups` set to True is used.
            aggregate_fields (Union[None, bool, Iterable[str]]): Fields to aggregate instead of
                including in child metadata; True aggregates all lists; False aggregates none.
            parent_links_only (Optional[bool]): If True, excludes links from the metadata.
            minimize_metadata (Optional[bool]): Whether to minimize metadata.
            minimize_keys (Optional[MaybeList[PathLikeKey]]): Keys specifying which metadata to minimize.
            clean_metadata (Optional[bool]): If True, remove empty metadata fields.
            clean_metadata_kwargs (KwargsLike): Keyword arguments for cleaning metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.
            dump_metadata_kwargs (KwargsLike): Keyword arguments for dumping metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.
            metadata_fence (Optional[str]): Metadata fence to use for formatting.

                Options are "code", "frontmatter", or a custom string.
            to_markdown_kwargs (KwargsLike): Keyword arguments for markdown conversion.

                See `vectorbtpro.utils.knowledge.formatting.to_markdown`.
            **kwargs: Keyword arguments for `MessagesAsset.apply`.

        Returns:
            MaybeMessagesAsset: New messages asset with messages aggregated by block.
        """
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by="block", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_block",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            minimize_metadata=minimize_metadata,
            minimize_keys=minimize_keys,
            clean_metadata=clean_metadata,
            clean_metadata_kwargs=clean_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            metadata_fence=metadata_fence,
            to_markdown_kwargs=to_markdown_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_threads(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeIterable[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
        """Aggregate messages by thread.

        Collects messages grouped by the "thread" field using `MessagesAsset.collect`
        (with `uniform_groups` defaulting to True) and then applies
        `vectorbtpro.utils.knowledge.custom_asset_funcs.AggThreadAssetFunc` to aggregate each group.

        Args:
            collect_kwargs (KwargsLike): Keyword arguments for `MessagesAsset.collect`.

                If not provided, an empty dict with `uniform_groups` set to True is used.
            aggregate_fields (Union[None, bool, Iterable[str]]): Fields to aggregate instead of
                including in child metadata; True aggregates all lists; False aggregates none.
            parent_links_only (Optional[bool]): If True, excludes links from the metadata.
            minimize_metadata (Optional[bool]): Whether to minimize metadata.
            minimize_keys (Optional[MaybeList[PathLikeKey]]): Keys specifying which metadata to minimize.
            clean_metadata (Optional[bool]): If True, remove empty metadata fields.
            clean_metadata_kwargs (KwargsLike): Keyword arguments for cleaning metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.
            dump_metadata_kwargs (KwargsLike): Keyword arguments for dumping metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.
            metadata_fence (Optional[str]): Metadata fence to use for formatting.

                Options are "code", "frontmatter", or a custom string.
            to_markdown_kwargs (KwargsLike): Keyword arguments for markdown conversion.

                See `vectorbtpro.utils.knowledge.formatting.to_markdown`.
            **kwargs: Keyword arguments for `MessagesAsset.apply`.

        Returns:
            MaybeMessagesAsset: New messages asset with messages aggregated by thread.
        """
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by="thread", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_thread",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            minimize_metadata=minimize_metadata,
            minimize_keys=minimize_keys,
            clean_metadata=clean_metadata,
            clean_metadata_kwargs=clean_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            metadata_fence=metadata_fence,
            to_markdown_kwargs=to_markdown_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_channels(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeIterable[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
        """Aggregate messages by channel.

        Collects messages grouped by the "channel" field using `MessagesAsset.collect`
        (with `uniform_groups` defaulting to True) and then applies
        `vectorbtpro.utils.knowledge.custom_asset_funcs.AggChannelAssetFunc` to aggregate each group.

        Args:
            collect_kwargs (KwargsLike): Keyword arguments for `MessagesAsset.collect`.

                If not provided, an empty dict with `uniform_groups` set to True is used.
            aggregate_fields (Union[None, bool, Iterable[str]]): Fields to aggregate instead of
                including in child metadata; True aggregates all lists; False aggregates none.
            parent_links_only (Optional[bool]): If True, excludes links from the metadata.
            minimize_metadata (Optional[bool]): Whether to minimize metadata.
            minimize_keys (Optional[MaybeList[PathLikeKey]]): Keys specifying which metadata to minimize.
            clean_metadata (Optional[bool]): If True, remove empty metadata fields.
            clean_metadata_kwargs (KwargsLike): Keyword arguments for cleaning metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.
            dump_metadata_kwargs (KwargsLike): Keyword arguments for dumping metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.
            metadata_fence (Optional[str]): Metadata fence to use for formatting.

                Options are "code", "frontmatter", or a custom string.
            to_markdown_kwargs (KwargsLike): Keyword arguments for markdown conversion.

                See `vectorbtpro.utils.knowledge.formatting.to_markdown`.
            **kwargs: Keyword arguments for `MessagesAsset.apply`.

        Returns:
            MaybeMessagesAsset: New messages asset with messages aggregated by channel.
        """
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by="channel", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_channel",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            minimize_metadata=minimize_metadata,
            minimize_keys=minimize_keys,
            clean_metadata=clean_metadata,
            clean_metadata_kwargs=clean_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            metadata_fence=metadata_fence,
            to_markdown_kwargs=to_markdown_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    @property
    def lowest_aggregate_by(self) -> tp.Optional[str]:
        """Lowest aggregation level for the messages.

        Checks the following fields in order:

        * If `attachments` is present, returns "message".
        * If all messages share the same `block`, returns "block".
        * If all messages share the same `thread`, returns "thread".
        * If all messages share the same `channel`, returns "channel".

        Returns:
            Optional[str]: Lowest level by which messages are aggregated,
                or None if no level is consistently available.
        """
        try:
            if self.get("attachments", single_item=False):
                return "message"
        except KeyError:
            pass
        try:
            if len(set(self.get("block", single_item=False))) == 1:
                return "block"
        except KeyError:
            pass
        try:
            if len(set(self.get("thread", single_item=False))) == 1:
                return "thread"
        except KeyError:
            pass
        try:
            if len(set(self.get("channel", single_item=False))) == 1:
                return "channel"
        except KeyError:
            pass

    @property
    def highest_aggregate_by(self) -> tp.Optional[str]:
        """Highest aggregation level for messages.

        Checks if all messages share a uniform "channel", "thread", or "block" identifier,
        or if messages contain attachments. Returns the corresponding level name if found.

        Returns:
            Optional[str]: Aggregation level ("channel", "thread", "block", or "message")
                if a uniform attribute is found; otherwise, None.
        """
        try:
            if len(set(self.get("channel", single_item=False))) == 1:
                return "channel"
        except KeyError:
            pass
        try:
            if len(set(self.get("thread", single_item=False))) == 1:
                return "thread"
        except KeyError:
            pass
        try:
            if len(set(self.get("block", single_item=False))) == 1:
                return "block"
        except KeyError:
            pass
        try:
            if self.get("attachments", single_item=False):
                return "message"
        except KeyError:
            pass

    def aggregate(self, by: str = "lowest", **kwargs) -> tp.MaybeMessagesAsset:
        """Aggregate messages based on a specified level.

        Determines the aggregation level among "message", "block", "thread", or "channel" by processing
        the input `by`. If `by` is "lowest" or "highest", it is replaced with the corresponding property.
        The level name is then pluralized and used to call the appropriate aggregation method.

        Args:
            by (str): Aggregation level or a special keyword ("lowest" or "highest").
            **kwargs: Keyword arguments for the aggregation method.

        Returns:
            MessagesAsset: New messages asset with aggregated messages.
        """
        if by.lower() == "lowest":
            by = self.lowest_aggregate_by
        elif by.lower() == "highest":
            by = self.highest_aggregate_by
        if by is None:
            raise ValueError("Must provide by")
        if not by.lower().endswith("s"):
            by += "s"
        return getattr(self, "aggregate_" + by.lower())(**kwargs)

    def select_reference(self: MessagesAssetT, link: str, **kwargs) -> MessagesAssetT:
        """Return the reference message corresponding to the specified link.

        Locates a message using `VBTAsset.find_link`, retrieves its "reference" field, and returns
        the first message matching that reference from the asset data.

        Args:
            link (str): Link used to identify the base message.
            **kwargs: Keyword arguments for `VBTAsset.find_link`.

        Returns:
            MessagesAsset: New messages asset containing the reference message.
        """
        d = self.find_link(link, wrap=False, **kwargs)
        reference = d.get("reference", None)
        new_data = []
        if reference:
            for d2 in self.data:
                if d2["reference"] == reference:
                    new_data.append(d2)
                    break
        return self.replace(data=new_data, single_item=True)

    def select_replies(self: MessagesAssetT, link: str, **kwargs) -> MessagesAssetT:
        """Return the reply messages associated with the specified link.

        Uses `VBTAsset.find_link` to locate a base message and extracts its "replies".
        The method then collects the reply messages from the asset data and returns them as a new asset.

        Args:
            link (str): Link used to find the base message.
            **kwargs: Keyword arguments for `VBTAsset.find_link`.

        Returns:
            MessagesAsset: New messages asset containing the reply messages.
        """
        d = self.find_link(link, wrap=False, **kwargs)
        replies = d.get("replies", [])
        new_data = []
        if replies:
            reply_data = dict.fromkeys(replies)
            replies_found = 0
            for d2 in self.data:
                if d2["link"] in reply_data:
                    reply_data[d2["link"]] = d2
                    replies_found += 1
                    if replies_found == len(replies):
                        break
            new_data = list(reply_data.values())
        return self.replace(data=new_data, single_item=True)

    def select_block(
        self: MessagesAssetT, link: str, incl_link: bool = True, **kwargs
    ) -> MessagesAssetT:
        """Return messages belonging to the same block as the specified link.

        Locates a message using `VBTAsset.find_link` to determine the target block,
        then selects all messages with the same block identifier. The original message
        is included if `incl_link` is True.

        Args:
            link (str): Link used to determine the target block.
            incl_link (bool): Indicates whether to include the message corresponding to `link`.
            **kwargs: Keyword arguments for `VBTAsset.find_link`.

        Returns:
            MessagesAsset: New messages asset containing messages from the same block.
        """
        d = self.find_link(link, wrap=False, **kwargs)
        new_data = []
        for d2 in self.data:
            if d2["block"] == d["block"] and (incl_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)

    def select_thread(
        self: MessagesAssetT, link: str, incl_link: bool = True, **kwargs
    ) -> MessagesAssetT:
        """Return messages belonging to the same thread as the specified link.

        Uses `VBTAsset.find_link` to locate a message and determines its thread.
        It then collects all messages sharing the same thread identifier.
        The originating message is excluded if `incl_link` is False.

        Args:
            link (str): Link used to determine the target thread.
            incl_link (bool): Indicates whether to include the message corresponding to `link`.
            **kwargs: Keyword arguments for `VBTAsset.find_link`.

        Returns:
            MessagesAsset: New messages asset containing messages from the same thread.
        """
        d = self.find_link(link, wrap=False, **kwargs)
        new_data = []
        for d2 in self.data:
            if d2["thread"] == d["thread"] and (incl_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)

    def select_channel(
        self: MessagesAssetT, name_or_link: str, incl_link: bool = True, **kwargs
    ) -> MessagesAssetT:
        """Return messages belonging to the same channel as the specified link.

        If `name_or_link` is a link, finds a message using `VBTAsset.find_link` to determine the channel,
        then selects all messages with the same channel identifier from the asset. The originating message
        is included based on the `incl_link` flag.

        Args:
            name_or_link (str): Channel name or link used to identify the channel.
            incl_link (bool): Indicates whether to include the message corresponding to `link`.
            **kwargs: Keyword arguments for `VBTAsset.find_link`.

        Returns:
            MessagesAsset: New messages asset containing messages from the same channel.
        """
        try:
            d = self.find_link(name_or_link, wrap=False, **kwargs)
        except NoItemFoundError:
            if name_or_link in set(self.get("channel", single_item=False)):
                new_data = []
                for d2 in self.data:
                    if d2["channel"] == name_or_link:
                        new_data.append(d2)
                return self.replace(data=new_data, single_item=False)
            raise NoItemFoundError(f"No channel or link matching '{name_or_link}'")
        new_data = []
        for d2 in self.data:
            if d2["channel"] == d["channel"] and (incl_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)

    def find_obj_messages(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
        """Return messages relevant to the specified object(s).

        Delegates the search to `MessagesAsset.find_obj_mentions` to locate messages associated
        with the provided object(s). Any additional keyword arguments are forwarded to the method.

        Args:
            obj (MaybeList): Object or list of objects to search for.
            attr (Optional[str]): Attribute name to target on the object.
            module (Union[None, str, ModuleType]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the object's reference name.
            **kwargs: Keyword arguments for `MessagesAsset.find_obj_mentions`.

        Returns:
            MaybeMessagesAsset: New messages asset containing messages related to the specified object(s).
        """
        return self.find_obj_mentions(obj, attr=attr, module=module, resolve=resolve, **kwargs)


ExamplesAssetT = tp.TypeVar("ExamplesAssetT", bound="ExamplesAsset")


class ExamplesAsset(VBTAsset):
    """Class for managing code examples (i.e., extracted and annotated code snippets from other assets).

    Fields:
        link (str): URL of the page or message, e.g. "https://vectorbt.pro/features/data/".
        title (str): Title of the code example, e.g. "Definition of `<function_name>`".
        description (str): Description of the code example, e.g. "Demonstrates ...".
        content (str): Actual code example wrapped in a fenced code block, e.g. "```python\n...\n```".
        verified (bool): Whether the code example was posted by project maintainer or has at least one upvote.

    !!! info
        For default settings, see `assets.examples` in `vectorbtpro._settings.knowledge`.
    """

    _settings_path: tp.SettingsPath = "knowledge.assets.examples"

    def select_verified(self, **kwargs) -> tp.MaybeExamplesAsset:
        """Return code examples that are verified.

        Filters the code examples asset to include only those where the `verified` field is True.

        Args:
            **kwargs: Keyword arguments for `ExamplesAsset.filter`.

        Returns:
            MaybeExamplesAsset: New examples asset containing only verified code examples.
        """
        return self.filter(lambda x: x.get("verified", False), **kwargs)

    def latest_first(self, **kwargs) -> tp.MaybeExamplesAsset:
        """Return code examples sorted in reverse chronological order.

        Documentation links appear first, followed by code examples from Discord messages.
        Sorts the code examples asset by the Discord message ID extracted from the `link` field.

        Args:
            **kwargs: Keyword arguments for `ExamplesAsset.sort`.

        Returns:
            MaybeExamplesAsset: New code examples asset sorted with the latest code examples first.
        """

        def _extract_discord_message_id(url):
            path = urlparse(url).path.rstrip("/")
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
                return int(parts[-1])
            return None

        def _sort_key(x):
            msg_id = _extract_discord_message_id(x["link"])
            return (0, 0) if msg_id is None else (1, -msg_id)

        return self.sort(source=_sort_key, ascending=True, **kwargs)

    def find_obj_examples(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        **kwargs,
    ) -> tp.MaybeExamplesAsset:
        """Return code examples relevant to the specified object(s).

        Delegates the search to `ExamplesAsset.find_obj_mentions` to locate code examples associated
        with the provided object(s). Any additional keyword arguments are forwarded to the method.

        Args:
            obj (MaybeList): Object or list of objects to search for.
            attr (Optional[str]): Attribute name to target on the object.
            module (Union[None, str, ModuleType]): Module context used in reference resolution.
            resolve (bool): Whether to resolve the object's reference name.
            **kwargs: Keyword arguments for `ExamplesAsset.find_obj_mentions`.

        Returns:
            MaybeExamplesAsset: New examples asset containing code examples related to the specified object(s).
        """
        return self.find_obj_mentions(obj, attr=attr, module=module, resolve=resolve, **kwargs)


def is_obj_or_query_ref(obj_or_query: tp.MaybeList) -> bool:
    """Return True if the input is a valid object reference; otherwise, False.

    For string inputs, verifies that each segment separated by a dot is a valid identifier.
    For non-string inputs, returns True.

    Args:
        obj_or_query (MaybeList): Object or query to evaluate.

    Returns:
        bool: True if the input is a valid object reference; otherwise, False.
    """
    if isinstance(obj_or_query, str):
        return all(segment.isidentifier() for segment in obj_or_query.split("."))
    return True


def find_api(
    obj_or_query: tp.Optional[tp.MaybeList] = None,
    *,
    as_query: tp.Optional[bool] = None,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    aggregate: tp.Optional[bool] = None,
    aggregate_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybePagesAsset:
    """Return API pages relevant to the specified object(s) or query.

    If `obj_or_query` is None, returns all API pages. When `obj_or_query` is a reference
    to an object, uses `PagesAsset.find_obj_api` to locate matching pages; otherwise, uses
    `PagesAsset.rank` to rank pages based on the query.

    Args:
        obj_or_query (Optional[MaybeList]): Object reference, query, or list of such.
        as_query (Optional[bool]): Flag indicating whether to treat `obj_or_query` as a query.
        attr (Optional[str]): Attribute name to target on the object.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the object's reference name.
        pages_asset (Optional[MaybeType[PagesAsset]]): Class or instance representing pages assets.
        pull_kwargs (KwargsLike): Keyword arguments for `PagesAsset.pull`.
        aggregate (Optional[bool]): Whether to aggregate headings into pages.

            If None, defaults to False.
        aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
        **kwargs: Keyword arguments for `PagesAsset.find_obj_api` or `PagesAsset.rank`.

    Returns:
        MaybePagesAsset: New pages asset containing API pages and headings relevant to the input.
    """
    if pages_asset is None:
        pages_asset = PagesAsset
    if isinstance(pages_asset, type):
        checks.assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        pages_asset = pages_asset.pull(**pull_kwargs)
    else:
        checks.assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")

    if as_query is None:
        as_query = obj_or_query is not None and not is_obj_or_query_ref(obj_or_query)
    if obj_or_query is not None and not as_query:
        return pages_asset.find_obj_api(
            obj_or_query,
            attr=attr,
            module=module,
            resolve=resolve,
            aggregate=aggregate,
            aggregate_kwargs=aggregate_kwargs,
            **kwargs,
        )
    if aggregate is None:
        aggregate = False
    if aggregate:
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        pages_asset = pages_asset.aggregate(**aggregate_kwargs)
    pages_asset = pages_asset.select_api()
    if obj_or_query is None:
        return pages_asset
    return pages_asset.rank(obj_or_query, **kwargs)


def find_docs(
    obj_or_query: tp.Optional[tp.MaybeList] = None,
    *,
    as_query: tp.Optional[bool] = None,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    aggregate: tp.Optional[bool] = None,
    aggregate_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybePagesAsset:
    """Return documentation pages relevant to the specified object(s) or query.

    If `obj_or_query` is None, returns all documentation pages. When `obj_or_query` is a reference
    to an object, uses `PagesAsset.find_obj_docs` to retrieve related pages; otherwise, employs
    `PagesAsset.rank` to assess page relevance.

    Args:
        obj_or_query (Optional[MaybeList]): Object reference, query, or list of such.
        as_query (Optional[bool]): Flag indicating whether to treat `obj_or_query` as a query.
        attr (Optional[str]): Attribute name to target on the object.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the object's reference name.
        pages_asset (Optional[MaybeType[PagesAsset]]): Class or instance representing pages assets.
        pull_kwargs (KwargsLike): Keyword arguments for `PagesAsset.pull`.
        aggregate (Optional[bool]): Whether to aggregate headings into pages.

            If None, defaults to False.
        aggregate_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
        **kwargs: Keyword arguments for `PagesAsset.find_obj_docs` or `PagesAsset.rank`.

    Returns:
        MaybePagesAsset: New pages asset containing documentation pages and headings relevant to the input.
    """
    if pages_asset is None:
        pages_asset = PagesAsset
    if isinstance(pages_asset, type):
        checks.assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        pages_asset = pages_asset.pull(**pull_kwargs)
    else:
        checks.assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")

    if as_query is None:
        as_query = obj_or_query is not None and not is_obj_or_query_ref(obj_or_query)
    if obj_or_query is not None and not as_query:
        return pages_asset.find_obj_docs(
            obj_or_query,
            attr=attr,
            module=module,
            resolve=resolve,
            aggregate=aggregate,
            aggregate_kwargs=aggregate_kwargs,
            **kwargs,
        )
    if aggregate is None:
        aggregate = False
    if aggregate:
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        pages_asset = pages_asset.aggregate(**aggregate_kwargs)
    pages_asset = pages_asset.select_docs()
    if obj_or_query is None:
        return pages_asset
    return pages_asset.rank(obj_or_query, **kwargs)


def find_messages(
    obj_or_query: tp.Optional[tp.MaybeList] = None,
    *,
    as_query: tp.Optional[bool] = None,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    aggregate: tp.Union[bool, str] = "messages",
    aggregate_kwargs: tp.KwargsLike = None,
    latest_first: bool = False,
    shuffle: bool = False,
    **kwargs,
) -> tp.MaybeMessagesAsset:
    """Find messages associated with an object or query.

    Args:
        obj_or_query (Optional[MaybeList]): Object reference, query, or list of such.

            If None, all messages are returned.
        as_query (Optional[bool]): Flag indicating whether to treat `obj_or_query` as a query.
        attr (Optional[str]): Attribute name to target on the object.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the object's reference name.
        messages_asset (Optional[MaybeType[MessagesAsset]]): Class or instance representing messages assets.
        pull_kwargs (KwargsLike): Keyword arguments for `MessagesAsset.pull`.
        aggregate (Union[bool, str]): Option to aggregate messages; if a string, it specifies the aggregation key.
        aggregate_kwargs (KwargsLike): Keyword arguments for `MessagesAsset.aggregate`.
        latest_first (bool): If True, sorts messages in reverse chronological order.
        shuffle (bool): If True, shuffles the order of messages.
        **kwargs: Keyword arguments for `MessagesAsset.find_obj_messages` or `MessagesAsset.rank`.

    Returns:
        MaybeMessagesAsset: New messages asset of messages processed according to the specified parameters.

    !!! note
        If `obj_or_query` is provided and not treated as a query, messages are retrieved using
        `MessagesAsset.find_obj_messages`. Otherwise, messages are filtered by ranking via `MessagesAsset.rank`.
    """
    if messages_asset is None:
        messages_asset = MessagesAsset
    if isinstance(messages_asset, type):
        checks.assert_subclass_of(messages_asset, MessagesAsset, arg_name="messages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        messages_asset = messages_asset.pull(**pull_kwargs)
    else:
        checks.assert_instance_of(messages_asset, MessagesAsset, arg_name="messages_asset")
    if aggregate:
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        if isinstance(aggregate, str) and "by" not in aggregate_kwargs:
            aggregate_kwargs["by"] = aggregate
        messages_asset = messages_asset.aggregate(**aggregate_kwargs)
    if latest_first:
        messages_asset = messages_asset.latest_first()
    elif shuffle:
        messages_asset = messages_asset.shuffle()

    if as_query is None:
        as_query = obj_or_query is not None and not is_obj_or_query_ref(obj_or_query)
    if obj_or_query is not None and not as_query:
        return messages_asset.find_obj_messages(
            obj_or_query, attr=attr, module=module, resolve=resolve, **kwargs
        )
    if obj_or_query is None:
        return messages_asset
    return messages_asset.rank(obj_or_query, **kwargs)


def find_examples(
    obj_or_query: tp.Optional[tp.MaybeList] = None,
    *,
    as_query: tp.Optional[bool] = None,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    examples_asset: tp.Optional[tp.MaybeType[ExamplesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    latest_first: bool = False,
    verified_only: bool = False,
    shuffle: bool = False,
    **kwargs,
) -> tp.MaybeExamplesAsset:
    """Find code examples associated with an object or query.

    Args:
        obj_or_query (Optional[MaybeList]): Object reference, query, or list of such.

            If None, all code examples are returned.
        as_query (Optional[bool]): Flag indicating whether to treat `obj_or_query` as a query.
        attr (Optional[str]): Attribute name to target on the object.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the object's reference name.
        examples_asset (Optional[MaybeType[ExamplesAsset]]): Class or instance representing examples assets.
        pull_kwargs (KwargsLike): Keyword arguments for `ExamplesAsset.pull`.
        latest_first (bool): If True, sorts code examples in reverse chronological order.
        verified_only (bool): If True, only returns verified code examples.
        shuffle (bool): If True, shuffles the order of code examples.
        **kwargs: Keyword arguments for `ExamplesAsset.find_obj_examples` or `ExamplesAsset.rank`.

    Returns:
        MaybeExamplesAsset: New examples asset of code examples processed according to the specified parameters.

    !!! note
        If `obj_or_query` is provided and not treated as a query, examples are retrieved using
        `ExamplesAsset.find_obj_examples`. Otherwise, examples are filtered by ranking via `ExamplesAsset.rank`.
    """
    if examples_asset is None:
        examples_asset = ExamplesAsset
    if isinstance(examples_asset, type):
        checks.assert_subclass_of(examples_asset, ExamplesAsset, arg_name="examples_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        examples_asset = examples_asset.pull(**pull_kwargs)
    else:
        checks.assert_instance_of(examples_asset, ExamplesAsset, arg_name="examples_asset")
    if verified_only:
        examples_asset = examples_asset.select_verified()
    if latest_first:
        examples_asset = examples_asset.latest_first()
    elif shuffle:
        examples_asset = examples_asset.shuffle()

    if as_query is None:
        as_query = obj_or_query is not None and not is_obj_or_query_ref(obj_or_query)
    if obj_or_query is not None and not as_query:
        return examples_asset.find_obj_examples(
            obj_or_query, attr=attr, module=module, resolve=resolve, **kwargs
        )
    if obj_or_query is None:
        return examples_asset
    return examples_asset.rank(obj_or_query, **kwargs)


def find_assets(
    obj_or_query: tp.Optional[tp.MaybeList] = None,
    *,
    as_query: tp.Optional[bool] = None,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    asset_names: tp.MaybeIterable[str] = "all",
    pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
    messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
    examples_asset: tp.Optional[tp.MaybeType[ExamplesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    aggregate_pages: bool = False,
    aggregate_pages_kwargs: tp.KwargsLike = None,
    aggregate_messages: tp.Union[bool, str] = "messages",
    aggregate_messages_kwargs: tp.KwargsLike = None,
    latest_first: bool = False,
    shuffle: bool = False,
    api_kwargs: tp.KwargsLike = None,
    docs_kwargs: tp.KwargsLike = None,
    messages_kwargs: tp.KwargsLike = None,
    examples_kwargs: tp.KwargsLike = None,
    minimize: tp.Optional[bool] = None,
    minimize_pages: tp.Optional[bool] = None,
    minimize_messages: tp.Optional[bool] = None,
    minimize_examples: tp.Optional[bool] = None,
    minimize_kwargs: tp.KwargsLike = None,
    minimize_pages_kwargs: tp.KwargsLike = None,
    minimize_messages_kwargs: tp.KwargsLike = None,
    minimize_examples_kwargs: tp.KwargsLike = None,
    combine: bool = True,
    combine_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybeDict[tp.VBTAsset]:
    """Return a dictionary of assets relevant to a given object(s) or query.

    Args:
        obj_or_query (Optional[MaybeList]): Object reference, query, or list of such.
        as_query (Optional[bool]): Flag indicating whether to treat `obj_or_query` as a query.
        attr (Optional[str]): Attribute name to target on the object.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the object's reference name.
        asset_names (Optional[MaybeIterable[str]]): List specifying the order and selection of assets.

            May include ellipsis (`...`) to adjust ordering. Allowed asset names are:

            * `api`: Retrieved via `find_api` with `api_kwargs`.
            * `docs`: Retrieved via `find_docs` with `docs_kwargs`.
            * `messages`: Retrieved via `find_messages` with `messages_kwargs`.
            * `examples`: Retrieved via `find_examples` with `examples_kwargs`.
            * `all`: Includes all supported asset types.

            For example, `["examples", ...]` puts "examples" at the beginning and all other assets
            in their usual order at the end.
        pages_asset (Optional[MaybeType[PagesAsset]]): Class or instance representing pages assets.
        messages_asset (Optional[MaybeType[MessagesAsset]]): Class or instance representing messages assets.
        examples_asset (Optional[MaybeType[ExamplesAsset]]): Class or instance representing examples assets.
        pull_kwargs (KwargsLike): Keyword arguments for `PagesAsset.pull` or `MessagesAsset.pull`.
        aggregate_pages (bool): Whether to aggregate the pages asset.
        aggregate_pages_kwargs (KwargsLike): Keyword arguments for `PagesAsset.aggregate`.
        aggregate_messages (Union[bool, str]): Option to aggregate messages;
            if a string, it specifies the aggregation key.
        aggregate_messages_kwargs (KwargsLike): Keyword arguments for `MessagesAsset.aggregate`.

            Key `minimize_metadata` is set to True by default.
        latest_first (bool): Whether to order messages and code examples with the most recent first.
        shuffle (bool): Whether to shuffle the order of messages and code examples.
        api_kwargs (KwargsLike): Keyword arguments for `find_api`.
        docs_kwargs (KwargsLike): Keyword arguments for `find_docs`.
        messages_kwargs (KwargsLike): Keyword arguments for `find_messages`.
        examples_kwargs (KwargsLike): Keyword arguments for `find_examples`.
        minimize (Optional[bool]): Whether to remove fields that are not relevant for chatting
            using `VBTAsset.minimize`.

            It defaults to True if `combine` is True, otherwise, it defaults to False.
        minimize_pages (Optional[bool]): Whether to remove non-chat-relevant fields from the pages asset.
        minimize_messages (Optional[bool]): Whether to remove non-chat-relevant fields from the messages asset.
        minimize_examples (Optional[bool]): Whether to remove non-chat-relevant fields from the examples asset.
        minimize_kwargs (KwargsLike): Keyword arguments for `VBTAsset.minimize`.

            Arguments `minimize_pages_kwargs` and `minimize_messages_kwargs` are merged over `minimize_kwargs`.
        minimize_pages_kwargs (KwargsLike): Keyword arguments for `PagesAsset.minimize`.
        minimize_messages_kwargs (KwargsLike): Keyword arguments for `MessagesAsset.minimize`.
        minimize_examples_kwargs (KwargsLike): Keyword arguments for `ExamplesAsset.minimize`.
        combine (bool): Whether to combine all found assets into a single asset.
        combine_kwargs (KwargsLike): Keyword arguments for `VBTAsset.combine`.
        **kwargs: Keyword arguments for asset functions (except for `find_api`
            when `obj_or_query` is an object; if both `combine` and `as_query` are True, these are
            instead passed to `VBTAsset.rank`).

    Returns:
        Dict[VBTAsset]: Dictionary mapping asset names to their corresponding assets;
            entries with no content are omitted.

    !!! note
        Keyword arguments are passed to all functions (except for `find_api` when `obj_or_query` is an object
        since it doesn't share common arguments with other three functions), unless `combine` and `as_query`
        are both True; in this case they are passed to `VBTAsset.rank`. Use specialized arguments like
        `api_kwargs` to provide keyword arguments to the respective function.

        If `obj_or_query` is a query, will rank the combined asset. Otherwise, will rank each individual asset.
    """
    if as_query is None:
        as_query = obj_or_query is not None and not is_obj_or_query_ref(obj_or_query)
    if combine and as_query and obj_or_query is not None:
        if api_kwargs is None:
            api_kwargs = {}
        if docs_kwargs is None:
            docs_kwargs = {}
        if messages_kwargs is None:
            messages_kwargs = {}
        if examples_kwargs is None:
            examples_kwargs = {}
    else:
        if as_query:
            api_kwargs = merge_dicts(kwargs, api_kwargs)
        else:
            if api_kwargs is None:
                api_kwargs = {}
        docs_kwargs = merge_dicts(kwargs, docs_kwargs)
        messages_kwargs = merge_dicts(kwargs, messages_kwargs)
        examples_kwargs = merge_dicts(kwargs, examples_kwargs)
    if "aggregate" not in api_kwargs:
        api_kwargs["aggregate"] = False
    if "aggregate" not in docs_kwargs:
        docs_kwargs["aggregate"] = False
    if "aggregate" not in messages_kwargs:
        messages_kwargs["aggregate"] = False
    if "latest_first" not in messages_kwargs:
        messages_kwargs["latest_first"] = False
    if "latest_first" not in examples_kwargs:
        examples_kwargs["latest_first"] = False

    all_asset_names = ["api", "docs", "messages", "examples"]
    if isinstance(asset_names, str) and asset_names.lower() == "all":
        asset_names = all_asset_names
    else:
        if isinstance(asset_names, (str, type(Ellipsis))):
            asset_names = [asset_names]
        asset_keys = []
        for asset_name in asset_names:
            if asset_name is Ellipsis or asset_name == "...":
                asset_keys.append(Ellipsis)
            else:
                asset_key = all_asset_names.index(asset_name.lower())
                if asset_key == -1:
                    raise ValueError(f"Invalid asset name: '{asset_name}'")
                asset_keys.append(asset_key)
        new_asset_names = reorder_list(all_asset_names, asset_keys, skip_missing=True)
        asset_names = new_asset_names

    if "api" in asset_names or "docs" in asset_names:
        if pages_asset is None:
            pages_asset = PagesAsset
        if isinstance(pages_asset, type):
            checks.assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
            if pull_kwargs is None:
                pull_kwargs = {}
            pages_asset = pages_asset.pull(**pull_kwargs)
        else:
            checks.assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if aggregate_pages:
            if aggregate_pages_kwargs is None:
                aggregate_pages_kwargs = {}
            pages_asset = pages_asset.aggregate(**aggregate_pages_kwargs)
    else:
        pages_asset = None
    if "messages" in asset_names:
        if messages_asset is None:
            messages_asset = MessagesAsset
        if isinstance(messages_asset, type):
            checks.assert_subclass_of(messages_asset, MessagesAsset, arg_name="messages_asset")
            if pull_kwargs is None:
                pull_kwargs = {}
            messages_asset = messages_asset.pull(**pull_kwargs)
        else:
            checks.assert_instance_of(messages_asset, MessagesAsset, arg_name="messages_asset")
        if aggregate_messages:
            if aggregate_messages_kwargs is None:
                aggregate_messages_kwargs = {}
            else:
                aggregate_messages_kwargs = dict(aggregate_messages_kwargs)
            if isinstance(aggregate_messages, str) and "by" not in aggregate_messages_kwargs:
                aggregate_messages_kwargs["by"] = aggregate_messages
            if "minimize_metadata" not in aggregate_messages_kwargs:
                aggregate_messages_kwargs["minimize_metadata"] = True
            messages_asset = messages_asset.aggregate(**aggregate_messages_kwargs)
        if latest_first:
            messages_asset = messages_asset.latest_first()
        elif shuffle:
            messages_asset = messages_asset.shuffle()
    else:
        messages_asset = None
    if "examples" in asset_names:
        if examples_asset is None:
            examples_asset = ExamplesAsset
        if isinstance(examples_asset, type):
            checks.assert_subclass_of(examples_asset, ExamplesAsset, arg_name="examples_asset")
            if pull_kwargs is None:
                pull_kwargs = {}
            examples_asset = examples_asset.pull(**pull_kwargs)
        else:
            checks.assert_instance_of(examples_asset, ExamplesAsset, arg_name="examples_asset")
        if latest_first:
            examples_asset = examples_asset.latest_first()
        elif shuffle:
            examples_asset = examples_asset.shuffle()
    else:
        examples_asset = None

    asset_dict = {}
    for asset_name in asset_names:
        if asset_name == "api":
            asset = find_api(
                None if combine and as_query else obj_or_query,
                as_query=as_query,
                attr=attr,
                module=module,
                resolve=resolve,
                pages_asset=pages_asset,
                **api_kwargs,
            )
            if len(asset) > 0:
                asset_dict[asset_name] = asset
        elif asset_name == "docs":
            asset = find_docs(
                None if combine and as_query else obj_or_query,
                as_query=as_query,
                attr=attr,
                module=module,
                resolve=resolve,
                pages_asset=pages_asset,
                **docs_kwargs,
            )
            if len(asset) > 0:
                asset_dict[asset_name] = asset
        elif asset_name == "messages":
            asset = find_messages(
                None if combine and as_query else obj_or_query,
                as_query=as_query,
                attr=attr,
                module=module,
                resolve=resolve,
                messages_asset=messages_asset,
                **messages_kwargs,
            )
            if len(asset) > 0:
                asset_dict[asset_name] = asset
        elif asset_name == "examples":
            if examples_kwargs is None:
                examples_kwargs = {}
            asset = find_examples(
                None if combine and as_query else obj_or_query,
                as_query=as_query,
                attr=attr,
                module=module,
                resolve=resolve,
                examples_asset=examples_asset,
                **examples_kwargs,
            )
            if len(asset) > 0:
                asset_dict[asset_name] = asset

    if minimize is None:
        minimize = combine and not as_query
    if minimize:
        if minimize_kwargs is None:
            minimize_kwargs = {}
        for k, v in asset_dict.items():
            if (
                isinstance(v, VBTAsset)
                and not isinstance(v, (PagesAsset, MessagesAsset, ExamplesAsset))
                and len(v) > 0
                and not isinstance(v[0], str)
            ):
                asset_dict[k] = v.minimize(**minimize_kwargs)
    if minimize_pages is None:
        minimize_pages = minimize
    if minimize_pages:
        minimize_pages_kwargs = merge_dicts(minimize_kwargs, minimize_pages_kwargs)
        for k, v in asset_dict.items():
            if isinstance(v, PagesAsset) and len(v) > 0 and not isinstance(v[0], str):
                asset_dict[k] = v.minimize(**minimize_pages_kwargs)
    if minimize_messages is None:
        minimize_messages = minimize
    if minimize_messages:
        minimize_messages_kwargs = merge_dicts(minimize_kwargs, minimize_messages_kwargs)
        for k, v in asset_dict.items():
            if isinstance(v, MessagesAsset) and len(v) > 0 and not isinstance(v[0], str):
                asset_dict[k] = v.minimize(**minimize_messages_kwargs)
    if minimize_examples is None:
        minimize_examples = minimize
    if minimize_examples:
        minimize_examples_kwargs = merge_dicts(minimize_kwargs, minimize_examples_kwargs)
        for k, v in asset_dict.items():
            if isinstance(v, ExamplesAsset) and len(v) > 0 and not isinstance(v[0], str):
                asset_dict[k] = v.minimize(**minimize_examples_kwargs)
    if combine:
        if len(asset_dict) >= 2:
            if combine_kwargs is None:
                combine_kwargs = {}
            combined_asset = VBTAsset.combine(*asset_dict.values(), **combine_kwargs)
        elif len(asset_dict) == 1:
            combined_asset = list(asset_dict.values())[0]
        else:
            combined_asset = VBTAsset()
        if combined_asset and as_query and obj_or_query is not None:
            combined_asset = combined_asset.rank(obj_or_query, **kwargs)
        return combined_asset
    return asset_dict


def chat_about(
    obj: tp.MaybeList,
    message: str,
    chat_history: tp.ChatHistory = None,
    *,
    asset_names: tp.MaybeIterable[str] = "examples",
    latest_first: bool = True,
    shuffle: tp.Optional[bool] = None,
    find_assets_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybeChatOutput:
    """Initiate a chat session for the given object(s) and return the resulting chat output.

    Uses `find_assets` with `combine=True` and `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.chat`
    to retrieve assets (defaulting to the asset name `examples`) and execute the chat. Keyword arguments are
    distributed automatically between the asset search and chat methods based on parameter names, unless
    some keys cannot be found in both signatures. In such a case, the key will be used for chatting.

    Args:
        obj (MaybeList): Object or list of objects to chat about.
        message (str): Initial message to start the chat.
        chat_history (ChatHistory): Chat history, a list of dictionaries with defined roles.
        asset_names (Optional[MaybeIterable[str]]): List specifying the order and selection of assets.

            May include ellipsis (`...`) to adjust ordering. Allowed asset names are:

            * `api`: Retrieved via `find_api` with `api_kwargs`.
            * `docs`: Retrieved via `find_docs` with `docs_kwargs`.
            * `messages`: Retrieved via `find_messages` with `messages_kwargs`.
            * `examples`: Retrieved via `find_examples` with `examples_kwargs`.
            * `all`: Includes all supported asset types.

            For example, `["examples", ...]` puts "examples" at the beginning and all other assets
            in their usual order at the end.
        latest_first (bool): Whether to order messages and code examples with the most recent first.
        shuffle (Optional[bool]): If True, shuffles the combined asset.

            If None, shuffles messages and code examples by default, but not the combined asset.
        find_assets_kwargs (KwargsLike): Keyword arguments for `find_assets`.
        **kwargs: Keyword arguments for `find_assets` or `VBTAsset.chat`.

    Returns:
        MaybeChatOutput: Output of the chat session, which may include the chat history and other information.
    """
    find_arg_names = set(get_func_arg_names(find_assets))
    if find_assets_kwargs is None:
        find_assets_kwargs = {}
    else:
        find_assets_kwargs = dict(find_assets_kwargs)
    chat_kwargs = {}
    for k, v in kwargs.items():
        if k in find_arg_names:
            if k not in find_assets_kwargs:
                find_assets_kwargs[k] = v
        else:
            chat_kwargs[k] = v
    asset = find_assets(
        obj,
        as_query=False,
        asset_names=asset_names,
        combine=True,
        latest_first=latest_first,
        shuffle=shuffle is None,
        **find_assets_kwargs,
    )
    if shuffle:
        asset = asset.shuffle()
    return asset.chat(message, chat_history, **chat_kwargs)


def search(
    query: str,
    cache_documents: bool = True,
    cache_key: tp.Optional[str] = None,
    asset_cache_manager: tp.Optional[tp.MaybeType[AssetCacheManager]] = None,
    asset_cache_manager_kwargs: tp.KwargsLike = None,
    aggregate_messages: tp.Union[bool, str] = "threads",
    find_assets_kwargs: tp.KwargsLike = None,
    display: tp.Union[bool, int] = 20,
    display_kwargs: tp.KwargsLike = None,
    silence_warnings: bool = False,
    **kwargs,
) -> tp.Union[tp.MaybeVBTAsset, tp.Path]:
    """Search for assets relevant to the provided query and return a ranked asset or display output.

    Uses `find_assets` with `combine=True` and `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.rank`.
    Keyword arguments are distributed among these two methods automatically, unless some keys cannot be
    found in both signatures. In such a case, the key will be used for ranking. If this is not wanted,
    specify the `find_assets`-related arguments explicitly with `find_assets_kwargs`.

    Metadata when aggregating messages will be minimized by default.

    Args:
        query (str): Search query string.
        cache_documents (bool): If True, will use an asset cache manager to cache the generated
            text documents after conversion.

            Running the same method again will use the cached documents.
        cache_key (Optional[str]): Unique identifier for the cached asset.
        asset_cache_manager (Optional[MaybeType[AssetCacheManager]]): Class or instance of
            `vectorbtpro.utils.knowledge.base_assets.AssetCacheManager`.
        asset_cache_manager_kwargs (KwargsLike): Keyword arguments to initialize or update `asset_cache_manager`.
        aggregate_messages (Union[bool, str]): Option to aggregate messages;
            if a string, it specifies the aggregation key.
        find_assets_kwargs (KwargsLike): Keyword arguments for `find_assets`.
        display (Union[bool, int]): If True, displays the top results as static HTML pages with `VBTAsset.display`.

            Pass an integer to display n top results. Will return the path to the temporary file.
        display_kwargs (KwargsLike): Keyword arguments for `VBTAsset.display`.
        silence_warnings (bool): Flag to suppress warning messages.
        **kwargs: Keyword arguments for `find_assets` or `VBTAsset.rank`.

    Returns:
        Union[MaybeVBTAsset, Path]: Ranked asset or the path to the temporary file with displayed results.
    """
    find_arg_names = set(get_func_arg_names(find_assets))
    if find_assets_kwargs is None:
        find_assets_kwargs = {}
    else:
        find_assets_kwargs = dict(find_assets_kwargs)
    rank_kwargs = {}
    for k, v in kwargs.items():
        if k in find_arg_names:
            if k not in find_assets_kwargs:
                find_assets_kwargs[k] = v
        else:
            rank_kwargs[k] = v
    find_assets_kwargs["aggregate_messages"] = aggregate_messages
    if cache_documents:
        if asset_cache_manager is None:
            asset_cache_manager = AssetCacheManager
        if asset_cache_manager_kwargs is None:
            asset_cache_manager_kwargs = {}
        if isinstance(asset_cache_manager, type):
            checks.assert_subclass_of(asset_cache_manager, AssetCacheManager, "asset_cache_manager")
            asset_cache_manager = asset_cache_manager(**asset_cache_manager_kwargs)
        else:
            checks.assert_instance_of(asset_cache_manager, AssetCacheManager, "asset_cache_manager")
            if asset_cache_manager_kwargs:
                asset_cache_manager = asset_cache_manager.replace(**asset_cache_manager_kwargs)
        asset_cache_manager_kwargs = {}
        if cache_key is None:
            cache_key = asset_cache_manager.generate_cache_key(**find_assets_kwargs)
        asset = asset_cache_manager.load_asset(cache_key)
        if asset is None:
            if not silence_warnings:
                warn("Caching documents...")
                silence_warnings = True
    else:
        asset = None
    if asset is None:
        asset = find_assets(None, as_query=True, **find_assets_kwargs)
    found_asset = asset.rank(
        query,
        cache_documents=cache_documents,
        cache_key=cache_key,
        asset_cache_manager=asset_cache_manager,
        asset_cache_manager_kwargs=asset_cache_manager_kwargs,
        silence_warnings=silence_warnings,
        **rank_kwargs,
    )
    if display:
        if display_kwargs is None:
            display_kwargs = {}
        else:
            display_kwargs = dict(display_kwargs)
        if "title" not in display_kwargs:
            display_kwargs["title"] = query
        if isinstance(display, bool):
            display_asset = found_asset
        else:
            display_asset = found_asset[:display]
        return display_asset.display(**display_kwargs)
    return found_asset


def quick_search(*args, **kwargs) -> tp.Union[tp.MaybeVBTAsset, tp.Path]:
    """Invoke `search` with `search_method` preset to "bm25".

    Args:
        *args: Positional arguments for `search`.
        **kwargs: Keyword arguments for `search`.

    Returns:
        Union[MaybeVBTAsset, Path]: Ranked asset or the path to the temporary file with displayed results.
    """
    return search(*args, search_method="bm25", **kwargs)


def chat(
    query: str,
    chat_history: tp.ChatHistory = None,
    *,
    cache_documents: bool = True,
    cache_key: tp.Optional[str] = None,
    asset_cache_manager: tp.Optional[tp.MaybeType[AssetCacheManager]] = None,
    asset_cache_manager_kwargs: tp.KwargsLike = None,
    aggregate_messages: tp.Union[bool, str] = "threads",
    find_assets_kwargs: tp.KwargsLike = None,
    rank: tp.Optional[bool] = True,
    top_k: tp.TopKLike = "elbow",
    min_top_k: tp.TopKLike = 20,
    max_top_k: tp.TopKLike = 100,
    cutoff: tp.Optional[float] = None,
    return_chunks: tp.Optional[bool] = True,
    rank_kwargs: tp.KwargsLike = None,
    wrap_documents: tp.Optional[bool] = True,
    silence_warnings: bool = False,
    **kwargs,
) -> tp.MaybeChatOutput:
    """Process a query and generate a chat response.

    Distribute keyword arguments between the internal `find_assets` function and the chat method of
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`. The function calls `find_assets` with
    `obj_or_query=None`, `as_query=True`, and `combine=True`. Any extra keyword arguments that match the
    parameters of `find_assets` are assigned there unless already provided via `find_assets_kwargs`, while the
    rest are passed to the chat method. Metadata in aggregated messages is minimized by default, and if
    `cache_documents` is enabled, the generated text documents are cached locally for reuse.

    Args:
        query (str): Query string to process.
        chat_history (ChatHistory): Chat history, a list of dictionaries with defined roles.
        cache_documents (bool): If True, will use an asset cache manager to cache the generated
            text documents after conversion.

            Running the same method again will use the cached documents.
        cache_key (Optional[str]): Unique identifier for the cached asset.
        asset_cache_manager (Optional[MaybeType[AssetCacheManager]]): Class or instance of
            `vectorbtpro.utils.knowledge.base_assets.AssetCacheManager`.
        asset_cache_manager_kwargs (KwargsLike): Keyword arguments to initialize or update `asset_cache_manager`.
        aggregate_messages (Union[bool, str]): Option to aggregate messages;
            if a string, it specifies the aggregation key.
        find_assets_kwargs (KwargsLike): Keyword arguments for `find_assets`.
        rank (Optional[bool]): Flag indicating whether to apply ranking.
        top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
        min_top_k (TopKLike): Minimum limit for determining top documents.
        max_top_k (TopKLike): Maximum limit for determining top documents.
        cutoff (Optional[float]): Score threshold to filter documents.
        return_chunks (Optional[bool]): Whether to return document chunks.
        rank_kwargs (KwargsLike): Keyword arguments for `VBTAsset.rank`.
        wrap_documents (Optional[bool]): Flag indicating whether to preserve the document embedding structure.
        silence_warnings (bool): Flag to suppress warning messages.
        **kwargs: Keyword arguments for `find_assets` or `VBTAsset.chat`.

    Returns:
        MaybeChatOutput: Chat response generated by the asset's chat method.
    """
    find_arg_names = set(get_func_arg_names(find_assets))
    if find_assets_kwargs is None:
        find_assets_kwargs = {}
    else:
        find_assets_kwargs = dict(find_assets_kwargs)
    chat_kwargs = {}
    for k, v in kwargs.items():
        if k in find_arg_names:
            if k not in find_assets_kwargs:
                find_assets_kwargs[k] = v
        else:
            chat_kwargs[k] = v
    find_assets_kwargs["aggregate_messages"] = aggregate_messages
    if cache_documents:
        if asset_cache_manager is None:
            asset_cache_manager = AssetCacheManager
        if asset_cache_manager_kwargs is None:
            asset_cache_manager_kwargs = {}
        if isinstance(asset_cache_manager, type):
            checks.assert_subclass_of(asset_cache_manager, AssetCacheManager, "asset_cache_manager")
            asset_cache_manager = asset_cache_manager(**asset_cache_manager_kwargs)
        else:
            checks.assert_instance_of(asset_cache_manager, AssetCacheManager, "asset_cache_manager")
            if asset_cache_manager_kwargs:
                asset_cache_manager = asset_cache_manager.replace(**asset_cache_manager_kwargs)
            asset_cache_manager_kwargs = {}
        if cache_key is None:
            cache_key = asset_cache_manager.generate_cache_key(**find_assets_kwargs)
        asset = asset_cache_manager.load_asset(cache_key)
        if asset is None:
            if not silence_warnings:
                warn("Caching documents...")
                silence_warnings = True
    else:
        asset = None
    if asset is None:
        asset = find_assets(None, as_query=True, **find_assets_kwargs)
    if rank_kwargs is None:
        rank_kwargs = {}
    else:
        rank_kwargs = dict(rank_kwargs)
    rank_kwargs["cache_documents"] = cache_documents
    rank_kwargs["cache_key"] = cache_key
    rank_kwargs["asset_cache_manager"] = asset_cache_manager
    rank_kwargs["asset_cache_manager_kwargs"] = asset_cache_manager_kwargs
    rank_kwargs["silence_warnings"] = silence_warnings
    if "wrap_documents" not in rank_kwargs:
        rank_kwargs["wrap_documents"] = wrap_documents
    return asset.chat(
        query,
        chat_history,
        rank=rank,
        top_k=top_k,
        min_top_k=min_top_k,
        max_top_k=max_top_k,
        cutoff=cutoff,
        return_chunks=return_chunks,
        rank_kwargs=rank_kwargs,
        **chat_kwargs,
    )


def quick_chat(
    *args,
    min_top_k: tp.TopKLike = 10,
    max_top_k: tp.TopKLike = 50,
    rank_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybeChatOutput:
    """Call `chat` with preset parameters for a quick response.

    Invoke `chat` using `search_method="bm25"` in `rank_kwargs` and enable quick mode by setting
    `quick_mode=True`. Positional and keyword arguments are forwarded to `chat`.

    Args:
        *args: Positional arguments for `chat`.
        min_top_k (TopKLike): Minimum limit for determining top documents.
        max_top_k (TopKLike): Maximum limit for determining top documents.
        rank_kwargs (KwargsLike): Keyword arguments for `VBTAsset.rank`.
        **kwargs: Keyword arguments for `chat`.

    Returns:
        MaybeChatOutput: Chat response generated by `chat`.
    """
    if rank_kwargs is None:
        rank_kwargs = {}
    else:
        rank_kwargs = dict(rank_kwargs)
    rank_kwargs["search_method"] = "bm25"
    return chat(
        *args,
        min_top_k=min_top_k,
        max_top_k=max_top_k,
        rank_kwargs=rank_kwargs,
        quick_mode=True,
        **kwargs,
    )
