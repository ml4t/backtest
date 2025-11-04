# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing custom asset function classes.

See `vectorbtpro.utils.knowledge` for the toy dataset.
"""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import flat_merge_dicts
from vectorbtpro.utils.knowledge.base_asset_funcs import AssetFunc, RemoveAssetFunc
from vectorbtpro.utils.knowledge.formatting import format_html, to_html, to_markdown

__all__ = []


class ToMarkdownAssetFunc(AssetFunc):
    """Asset function class for formatting asset metadata and content as Markdown with
    `vectorbtpro.utils.knowledge.custom_assets.VBTAsset.to_markdown`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "to_markdown"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        root_metadata_key: tp.Optional[tp.Key] = None,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **to_markdown_kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

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
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.custom_assets.VBTAsset`.
            **to_markdown_kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.formatting.to_markdown`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc, FindRemoveAssetFunc

        if asset_cls is None:
            from vectorbtpro.utils.knowledge.custom_assets import VBTAsset

            asset_cls = VBTAsset
        root_metadata_key = asset_cls.resolve_setting(root_metadata_key, "root_metadata_key")
        minimize_metadata = asset_cls.resolve_setting(minimize_metadata, "minimize_metadata")
        minimize_keys = asset_cls.resolve_setting(minimize_keys, "minimize_keys")
        clean_metadata = asset_cls.resolve_setting(clean_metadata, "clean_metadata")
        clean_metadata_kwargs = asset_cls.resolve_setting(
            clean_metadata_kwargs, "clean_metadata_kwargs", merge=True
        )
        dump_metadata_kwargs = asset_cls.resolve_setting(
            dump_metadata_kwargs, "dump_metadata_kwargs", merge=True
        )
        metadata_fence = asset_cls.resolve_setting(metadata_fence, "metadata_fence")
        to_markdown_kwargs = asset_cls.resolve_setting(
            to_markdown_kwargs, "to_markdown_kwargs", merge=True
        )

        clean_metadata_kwargs = flat_merge_dicts(
            dict(target=FindRemoveAssetFunc.is_empty_func), clean_metadata_kwargs
        )
        _, clean_metadata_kwargs = FindRemoveAssetFunc.prepare(**clean_metadata_kwargs)
        _, dump_metadata_kwargs = DumpAssetFunc.prepare(**dump_metadata_kwargs)
        return (), {
            **dict(
                minimize_metadata=minimize_metadata,
                minimize_keys=minimize_keys,
                root_metadata_key=root_metadata_key,
                clean_metadata=clean_metadata,
                clean_metadata_kwargs=clean_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                metadata_fence=metadata_fence,
            ),
            **to_markdown_kwargs,
        }

    @classmethod
    def get_markdown_metadata(
        cls,
        d: dict,
        root_metadata_key: tp.Optional[tp.Key] = None,
        allow_empty: tp.Optional[bool] = None,
        minimize_metadata: bool = False,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: bool = True,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: str = "frontmatter",
        **to_markdown_kwargs,
    ) -> str:
        """Return Markdown formatted metadata by converting data to markdown using
        `vectorbtpro.utils.knowledge.formatting.to_markdown`.

        Args:
            d (dict): Asset data dictionary.
            root_metadata_key (Optional[Key]): Key under which to nest metadata.
            allow_empty (Optional[bool]): Whether to allow empty metadata.
            minimize_metadata (bool): If True, remove specified keys to minimize metadata.
            minimize_keys (Optional[Union[PathLikeKey, list]]): Key or list of keys to remove during minimization.
            clean_metadata (bool): If True, clean the metadata to remove empty or irrelevant values.
            clean_metadata_kwargs (KwargsLike): Keyword arguments for cleaning metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.
            dump_metadata_kwargs (KwargsLike): Keyword arguments for dumping metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.
            metadata_fence (str): Metadata fence to use for formatting.

                Options are "code", "frontmatter", or a custom string.
            **to_markdown_kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.formatting.to_markdown`.

        Returns:
            str: Markdown formatted metadata string.
        """
        from vectorbtpro.utils.formatting import get_dump_frontmatter, get_dump_language
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc, FindRemoveAssetFunc

        if allow_empty is None:
            allow_empty = root_metadata_key is not None
        if clean_metadata_kwargs is None:
            clean_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        metadata = dict(d)
        if "content" in metadata:
            del metadata["content"]
        if metadata and minimize_metadata and minimize_keys:
            metadata = RemoveAssetFunc.call(metadata, minimize_keys, skip_missing=True)
        if metadata and clean_metadata:
            metadata = FindRemoveAssetFunc.call(metadata, **clean_metadata_kwargs)
        if not metadata and not allow_empty:
            return ""
        if root_metadata_key is not None:
            if not metadata:
                metadata = None
            metadata = {root_metadata_key: metadata}
        text = DumpAssetFunc.call(metadata, **dump_metadata_kwargs).strip()
        if metadata_fence.lower() == "frontmatter":
            dump_engine = dump_metadata_kwargs.get("dump_engine", "yaml")
            metadata_fence = get_dump_frontmatter(dump_engine)
            if not metadata_fence:
                metadata_fence = "code"
        if metadata_fence.lower() == "code":
            dump_engine = dump_metadata_kwargs.get("dump_engine", "yaml")
            dump_language = get_dump_language(dump_engine)
            text = f"```{dump_language}\n{text}\n```"
        else:
            text = f"{metadata_fence}\n{text}\n{metadata_fence}"
        to_markdown_kwargs["remove_code_title"] = False
        to_markdown_kwargs["even_indentation"] = False
        to_markdown_kwargs["newline_before_list"] = False
        return to_markdown(text, **to_markdown_kwargs)

    @classmethod
    def get_markdown_content(cls, d: dict, **kwargs) -> str:
        """Return Markdown formatted content by converting data to markdown using
        `vectorbtpro.utils.knowledge.formatting.to_markdown`.

        Args:
            d (dict): Asset data dictionary.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.formatting.to_markdown`.

        Returns:
            str: Markdown formatted content string.
        """
        if d["content"] is None:
            return ""
        return to_markdown(d["content"], **kwargs)

    @classmethod
    def call(
        cls,
        d: tp.Any,
        root_metadata_key: tp.Optional[tp.Key] = None,
        minimize_metadata: bool = False,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: bool = True,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: str = "frontmatter",
        **to_markdown_kwargs,
    ) -> tp.Any:
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc

        if not isinstance(d, (str, dict)):
            d = DumpAssetFunc.call(d)
        if isinstance(d, str):
            d = dict(content=d)
        markdown_metadata = cls.get_markdown_metadata(
            d,
            root_metadata_key=root_metadata_key,
            minimize_metadata=minimize_metadata,
            minimize_keys=minimize_keys,
            clean_metadata=clean_metadata,
            clean_metadata_kwargs=clean_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            metadata_fence=metadata_fence,
            **to_markdown_kwargs,
        )
        markdown_content = cls.get_markdown_content(d, **to_markdown_kwargs)
        if markdown_metadata and markdown_content:
            markdown_content = markdown_metadata + "\n\n" + markdown_content
        elif markdown_metadata:
            markdown_content = markdown_metadata
        return markdown_content


class ToHTMLAssetFunc(ToMarkdownAssetFunc):
    """Asset function class for converting asset data to HTML with
    `vectorbtpro.utils.knowledge.custom_assets.VBTAsset.to_html`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "to_html"

    @classmethod
    def prepare(
        cls,
        root_metadata_key: tp.Optional[tp.Key] = None,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **to_html_kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

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
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.custom_assets.VBTAsset`.
            **to_html_kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.formatting.to_html`.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc, FindRemoveAssetFunc

        if asset_cls is None:
            from vectorbtpro.utils.knowledge.custom_assets import VBTAsset

            asset_cls = VBTAsset
        root_metadata_key = asset_cls.resolve_setting(root_metadata_key, "root_metadata_key")
        minimize_metadata = asset_cls.resolve_setting(minimize_metadata, "minimize_metadata")
        minimize_keys = asset_cls.resolve_setting(minimize_keys, "minimize_keys")
        clean_metadata = asset_cls.resolve_setting(clean_metadata, "clean_metadata")
        clean_metadata_kwargs = asset_cls.resolve_setting(
            clean_metadata_kwargs, "clean_metadata_kwargs", merge=True
        )
        dump_metadata_kwargs = asset_cls.resolve_setting(
            dump_metadata_kwargs, "dump_metadata_kwargs", merge=True
        )
        metadata_fence = asset_cls.resolve_setting(metadata_fence, "metadata_fence")
        to_markdown_kwargs = asset_cls.resolve_setting(
            to_markdown_kwargs, "to_markdown_kwargs", merge=True
        )
        format_html_kwargs = asset_cls.resolve_setting(
            format_html_kwargs, "format_html_kwargs", merge=True
        )
        to_html_kwargs = asset_cls.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True)

        clean_metadata_kwargs = flat_merge_dicts(
            dict(target=FindRemoveAssetFunc.is_empty_func), clean_metadata_kwargs
        )
        _, clean_metadata_kwargs = FindRemoveAssetFunc.prepare(**clean_metadata_kwargs)
        _, dump_metadata_kwargs = DumpAssetFunc.prepare(**dump_metadata_kwargs)
        return (), {
            **dict(
                root_metadata_key=root_metadata_key,
                minimize_metadata=minimize_metadata,
                minimize_keys=minimize_keys,
                clean_metadata=clean_metadata,
                clean_metadata_kwargs=clean_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                metadata_fence=metadata_fence,
                to_markdown_kwargs=to_markdown_kwargs,
                format_html_kwargs=format_html_kwargs,
            ),
            **to_html_kwargs,
        }

    @classmethod
    def get_html_metadata(
        cls,
        d: dict,
        root_metadata_key: tp.Optional[tp.Key] = None,
        allow_empty: tp.Optional[bool] = None,
        minimize_metadata: bool = False,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: bool = True,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: str = "frontmatter",
        to_markdown_kwargs: tp.KwargsLike = None,
        **to_html_kwargs,
    ) -> str:
        """Return HTML formatted metadata by converting data to markdown using
        `ToHTMLAssetFunc.get_markdown_metadata` and then to HTML using
        `vectorbtpro.utils.knowledge.formatting.to_html`.

        Args:
            d (dict): Asset data dictionary.
            root_metadata_key (Optional[Key]): Key under which to nest metadata.
            allow_empty (Optional[bool]): Whether to allow empty metadata.
            minimize_metadata (bool): If True, remove specified keys to minimize metadata.
            minimize_keys (Optional[List[PathLikeKey]]): Keys to minimize in the metadata.
            clean_metadata (bool): If True, clean the metadata to remove empty or irrelevant values.
            clean_metadata_kwargs (KwargsLike): Keyword arguments for cleaning metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.
            dump_metadata_kwargs (KwargsLike): Keyword arguments for dumping metadata.

                See `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.
            metadata_fence (str): Metadata fence to use for formatting.

                Options are "code", "frontmatter", or a custom string.
            to_markdown_kwargs (KwargsLike): Keyword arguments for markdown conversion.

                See `vectorbtpro.utils.knowledge.formatting.to_markdown`.
            **to_html_kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.formatting.to_html`.

        Returns:
            str: HTML formatted metadata.
        """
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}
        metadata = cls.get_markdown_metadata(
            d,
            root_metadata_key=root_metadata_key,
            allow_empty=allow_empty,
            minimize_metadata=minimize_metadata,
            minimize_keys=minimize_keys,
            clean_metadata=clean_metadata,
            clean_metadata_kwargs=clean_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            metadata_fence=metadata_fence,
            **to_markdown_kwargs,
        )
        if not metadata:
            return ""
        return to_html(metadata, **to_html_kwargs)

    @classmethod
    def get_html_content(cls, d: dict, to_markdown_kwargs: tp.KwargsLike = None, **kwargs) -> str:
        """Return HTML formatted content by converting data to markdown using
        `ToHTMLAssetFunc.get_markdown_content` and then to HTML using
        `vectorbtpro.utils.knowledge.formatting.to_html`.

        Args:
            d (dict): Asset data dictionary.
            to_markdown_kwargs (KwargsLike): Keyword arguments for markdown conversion.

                See `vectorbtpro.utils.knowledge.formatting.to_markdown`.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.formatting.to_html`.

        Returns:
            str: HTML formatted content.
        """
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}
        content = cls.get_markdown_content(d, **to_markdown_kwargs)
        return to_html(content, **kwargs)

    @classmethod
    def call(
        cls,
        d: tp.Any,
        root_metadata_key: tp.Optional[tp.Key] = None,
        minimize_metadata: bool = False,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: bool = True,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: str = "frontmatter",
        to_markdown_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        **to_html_kwargs,
    ) -> tp.Any:
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc

        if not isinstance(d, (str, dict, list)):
            d = DumpAssetFunc.call(d)
        if isinstance(d, str):
            d = dict(content=d)
        if isinstance(d, list):
            html_metadata = []
            for _d in d:
                if not isinstance(_d, (str, dict)):
                    _d = DumpAssetFunc.call(_d)
                if isinstance(_d, str):
                    _d = dict(content=_d)
                html_metadata.append(
                    cls.get_html_metadata(
                        _d,
                        root_metadata_key=root_metadata_key,
                        minimize_metadata=minimize_metadata,
                        minimize_keys=minimize_keys,
                        clean_metadata=clean_metadata,
                        clean_metadata_kwargs=clean_metadata_kwargs,
                        dump_metadata_kwargs=dump_metadata_kwargs,
                        metadata_fence=metadata_fence,
                        to_markdown_kwargs=to_markdown_kwargs,
                        **to_html_kwargs,
                    )
                )
            html = format_html(
                title="/",
                html_metadata="\n".join(html_metadata),
                **format_html_kwargs,
            )
        else:
            html_metadata = cls.get_html_metadata(
                d,
                root_metadata_key=root_metadata_key,
                minimize_metadata=minimize_metadata,
                minimize_keys=minimize_keys,
                clean_metadata=clean_metadata,
                clean_metadata_kwargs=clean_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                metadata_fence=metadata_fence,
                to_markdown_kwargs=to_markdown_kwargs,
                **to_html_kwargs,
            )
            html_content = cls.get_html_content(
                d,
                to_markdown_kwargs=to_markdown_kwargs,
                **to_html_kwargs,
            )
            html = format_html(
                title=d["link"] if "link" in d else "",
                html_metadata=html_metadata,
                html_content=html_content,
                **format_html_kwargs,
            )
        return html


class AggMessageAssetFunc(AssetFunc):
    """Asset function class for aggregating messages with
    `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_messages`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_message"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

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
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc, FindRemoveAssetFunc

        if asset_cls is None:
            from vectorbtpro.utils.knowledge.custom_assets import MessagesAsset

            asset_cls = MessagesAsset
        minimize_metadata = asset_cls.resolve_setting(minimize_metadata, "minimize_metadata")
        minimize_keys = asset_cls.resolve_setting(minimize_keys, "minimize_keys")
        clean_metadata = asset_cls.resolve_setting(clean_metadata, "clean_metadata")
        clean_metadata_kwargs = asset_cls.resolve_setting(
            clean_metadata_kwargs, "clean_metadata_kwargs", merge=True
        )
        dump_metadata_kwargs = asset_cls.resolve_setting(
            dump_metadata_kwargs, "dump_metadata_kwargs", merge=True
        )
        metadata_fence = asset_cls.resolve_setting(metadata_fence, "metadata_fence")

        clean_metadata_kwargs = flat_merge_dicts(
            dict(target=FindRemoveAssetFunc.is_empty_func), clean_metadata_kwargs
        )
        _, clean_metadata_kwargs = FindRemoveAssetFunc.prepare(**clean_metadata_kwargs)
        _, dump_metadata_kwargs = DumpAssetFunc.prepare(**dump_metadata_kwargs)
        return (), {
            **dict(
                minimize_metadata=minimize_metadata,
                minimize_keys=minimize_keys,
                clean_metadata=clean_metadata,
                clean_metadata_kwargs=clean_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                metadata_fence=metadata_fence,
                to_markdown_kwargs=to_markdown_kwargs,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        minimize_metadata: bool = False,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: bool = True,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: str = "frontmatter",
        to_markdown_kwargs: tp.KwargsLike = None,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if "attachments" not in d:
            return dict(d)
        if clean_metadata_kwargs is None:
            clean_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}

        new_d = dict(d)
        new_d["content"] = new_d["content"].strip()
        attachments = new_d.pop("attachments", [])
        for attachment in attachments:
            content = attachment["content"].strip()
            if new_d["content"]:
                new_d["content"] += "\n\n"
            metadata = ToMarkdownAssetFunc.get_markdown_metadata(
                attachment,
                root_metadata_key="attachment",
                allow_empty=not content,
                minimize_metadata=minimize_metadata,
                minimize_keys=minimize_keys,
                clean_metadata=clean_metadata,
                clean_metadata_kwargs=clean_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                metadata_fence=metadata_fence,
                **to_markdown_kwargs,
            )
            new_d["content"] += metadata
            if content:
                new_d["content"] += "\n\n" + content
        return new_d


class AggBlockAssetFunc(AssetFunc):
    """Asset function class for aggregating block messages with
    `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_blocks`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_block"

    _wrap: tp.ClassVar[tp.Optional[bool]] = True

    @classmethod
    def prepare(
        cls,
        aggregate_fields: tp.Union[None, bool, tp.MaybeIterable[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        minimize_metadata: tp.Optional[bool] = None,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: tp.Optional[bool] = None,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: tp.Optional[str] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
        asset_cls: tp.Optional[tp.Type[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments for an asset function call.

        Args:
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
            link_map (Optional[Dict[str, dict]]): Mapping of links to their corresponding data items.
            asset_cls (Optional[Type[KnowledgeAsset]]): Asset class to use for resolving settings.

                Defaults to `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset`.
            **kwargs: Additional keyword arguments.

        Returns:
            ArgsKwargs: Tuple containing the positional arguments and keyword arguments.
        """
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc, FindRemoveAssetFunc

        if asset_cls is None:
            from vectorbtpro.utils.knowledge.custom_assets import MessagesAsset

            asset_cls = MessagesAsset
        aggregate_fields = asset_cls.resolve_setting(aggregate_fields, "aggregate_fields")
        parent_links_only = asset_cls.resolve_setting(parent_links_only, "parent_links_only")
        minimize_metadata = asset_cls.resolve_setting(minimize_metadata, "minimize_metadata")
        minimize_keys = asset_cls.resolve_setting(minimize_keys, "minimize_keys")
        clean_metadata = asset_cls.resolve_setting(clean_metadata, "clean_metadata")
        clean_metadata_kwargs = asset_cls.resolve_setting(
            clean_metadata_kwargs, "clean_metadata_kwargs", merge=True
        )
        dump_metadata_kwargs = asset_cls.resolve_setting(
            dump_metadata_kwargs, "dump_metadata_kwargs", merge=True
        )
        metadata_fence = asset_cls.resolve_setting(metadata_fence, "metadata_fence")

        clean_metadata_kwargs = flat_merge_dicts(
            dict(target=FindRemoveAssetFunc.is_empty_func), clean_metadata_kwargs
        )
        _, clean_metadata_kwargs = FindRemoveAssetFunc.prepare(**clean_metadata_kwargs)
        _, dump_metadata_kwargs = DumpAssetFunc.prepare(**dump_metadata_kwargs)
        return (), {
            **dict(
                aggregate_fields=aggregate_fields,
                parent_links_only=parent_links_only,
                minimize_metadata=minimize_metadata,
                minimize_keys=minimize_keys,
                clean_metadata=clean_metadata,
                clean_metadata_kwargs=clean_metadata_kwargs,
                dump_metadata_kwargs=dump_metadata_kwargs,
                metadata_fence=metadata_fence,
                to_markdown_kwargs=to_markdown_kwargs,
                link_map=link_map,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        aggregate_fields: tp.Union[bool, tp.MaybeIterable[str]] = False,
        parent_links_only: bool = True,
        minimize_metadata: bool = False,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: bool = True,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: str = "frontmatter",
        to_markdown_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if isinstance(aggregate_fields, bool):
            if aggregate_fields:
                aggregate_fields = {"mentions", "attachments", "reactions"}
            else:
                aggregate_fields = set()
        elif isinstance(aggregate_fields, str):
            aggregate_fields = {aggregate_fields}
        elif not isinstance(aggregate_fields, set):
            aggregate_fields = set(aggregate_fields)
        if clean_metadata_kwargs is None:
            clean_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}

        new_d = {}
        metadata_keys = []
        for k, v in d.items():
            if k == "link":
                new_d[k] = d["block"][0]
            if k == "block":
                continue
            if k == "timestamp":
                new_d[k] = v[0]
            if k in {"thread", "channel", "author"}:
                new_d[k] = v[0]
                continue
            if k == "reference" and link_map is not None:
                found_missing = False
                new_v = []
                for _v in v:
                    if _v:
                        if _v in link_map:
                            _v = link_map[_v]["block"]
                        else:
                            found_missing = True
                            break
                    if _v not in new_v:
                        new_v.append(_v)
                if found_missing or len(new_v) > 1:
                    new_d[k] = "?"
                else:
                    new_d[k] = new_v[0]
            if k == "replies" and link_map is not None:
                new_v = []
                for _v in v:
                    for __v in _v:
                        if __v and __v in link_map:
                            __v = link_map[__v]["block"]
                            if __v not in new_v:
                                new_v.append(__v)
                        else:
                            new_v.append("?")
                new_d[k] = new_v
            if k == "content":
                new_d[k] = []
                continue
            if k in aggregate_fields and isinstance(v[0], list):
                new_v = []
                for _v in new_v:
                    for __v in _v:
                        if __v not in new_v:
                            new_v.append(__v)
                new_d[k] = new_v
                continue
            if k == "reactions" and k in aggregate_fields:
                new_d[k] = sum(v)
                continue
            if parent_links_only:
                if k in ("link", "block", "thread", "reference", "replies"):
                    continue
            metadata_keys.append(k)
        if len(metadata_keys) > 0:
            for i in range(len(d[metadata_keys[0]])):
                content = d["content"][i].strip()
                metadata = {}
                for k in metadata_keys:
                    metadata[k] = d[k][i]
                if len(new_d["content"]) > 0:
                    new_d["content"].append("\n\n")
                metadata = ToMarkdownAssetFunc.get_markdown_metadata(
                    metadata,
                    root_metadata_key="message",
                    allow_empty=not content,
                    minimize_metadata=minimize_metadata,
                    minimize_keys=minimize_keys,
                    clean_metadata=clean_metadata,
                    clean_metadata_kwargs=clean_metadata_kwargs,
                    dump_metadata_kwargs=dump_metadata_kwargs,
                    metadata_fence=metadata_fence,
                    **to_markdown_kwargs,
                )
                new_d["content"].append(metadata)
                if content:
                    new_d["content"].append("\n\n" + content)
        new_d["content"] = "".join(new_d["content"])
        return new_d


class AggThreadAssetFunc(AggBlockAssetFunc):
    """Asset function class for aggregating thread messages with
    `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_threads`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_thread"

    @classmethod
    def call(
        cls,
        d: tp.Any,
        aggregate_fields: tp.Union[bool, tp.MaybeIterable[str]] = False,
        parent_links_only: bool = True,
        minimize_metadata: bool = False,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: bool = True,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: str = "frontmatter",
        to_markdown_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if isinstance(aggregate_fields, bool):
            if aggregate_fields:
                aggregate_fields = {"mentions", "attachments", "reactions"}
            else:
                aggregate_fields = set()
        elif isinstance(aggregate_fields, str):
            aggregate_fields = {aggregate_fields}
        elif not isinstance(aggregate_fields, set):
            aggregate_fields = set(aggregate_fields)
        if clean_metadata_kwargs is None:
            clean_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}

        new_d = {}
        metadata_keys = []
        for k, v in d.items():
            if k == "link":
                new_d[k] = d["thread"][0]
            if k == "thread":
                continue
            if k == "timestamp":
                new_d[k] = v[0]
            if k == "channel":
                new_d[k] = v[0]
                continue
            if k == "content":
                new_d[k] = []
                continue
            if k in aggregate_fields and isinstance(v[0], list):
                new_v = []
                for _v in new_v:
                    for __v in _v:
                        if __v not in new_v:
                            new_v.append(__v)
                new_d[k] = new_v
                continue
            if k == "reactions" and k in aggregate_fields:
                new_d[k] = sum(v)
                continue
            if parent_links_only:
                if k in ("link", "block", "thread", "reference", "replies"):
                    continue
            metadata_keys.append(k)
        if len(metadata_keys) > 0:
            for i in range(len(d[metadata_keys[0]])):
                content = d["content"][i].strip()
                metadata = {}
                for k in metadata_keys:
                    metadata[k] = d[k][i]
                if len(new_d["content"]) > 0:
                    new_d["content"].append("\n\n")
                metadata = ToMarkdownAssetFunc.get_markdown_metadata(
                    metadata,
                    root_metadata_key="message",
                    allow_empty=not content,
                    minimize_metadata=minimize_metadata,
                    minimize_keys=minimize_keys,
                    clean_metadata=clean_metadata,
                    clean_metadata_kwargs=clean_metadata_kwargs,
                    dump_metadata_kwargs=dump_metadata_kwargs,
                    metadata_fence=metadata_fence,
                    **to_markdown_kwargs,
                )
                new_d["content"].append(metadata)
                if content:
                    new_d["content"].append("\n\n" + content)
        new_d["content"] = "".join(new_d["content"])
        return new_d


class AggChannelAssetFunc(AggThreadAssetFunc):
    """Asset function class for aggregating channel messages with
    `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_channels`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_channel"

    @classmethod
    def get_channel_link(cls, link: str) -> str:
        """Return the channel link extracted from a message link.

        Args:
            link (str): Message link to process.

        Returns:
            str: Extracted channel link.
        """
        if link.startswith("$discord/"):
            link = link[len("$discord/") :]
            link_parts = link.split("/")
            channel_id = link_parts[0]
            return "$discord/" + channel_id
        if link.startswith("https://discord.com/channels/"):
            link = link[len("https://discord.com/channels/") :]
            link_parts = link.split("/")
            guild_id = link_parts[0]
            channel_id = link_parts[1]
            return f"https://discord.com/channels/{guild_id}/{channel_id}"
        raise ValueError(f"Invalid link: '{link}'")

    @classmethod
    def call(
        cls,
        d: tp.Any,
        aggregate_fields: tp.Union[bool, tp.MaybeIterable[str]] = False,
        parent_links_only: bool = True,
        minimize_metadata: bool = False,
        minimize_keys: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        clean_metadata: bool = True,
        clean_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        metadata_fence: str = "frontmatter",
        to_markdown_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if isinstance(aggregate_fields, bool):
            if aggregate_fields:
                aggregate_fields = {"mentions", "attachments", "reactions"}
            else:
                aggregate_fields = set()
        elif isinstance(aggregate_fields, str):
            aggregate_fields = {aggregate_fields}
        elif not isinstance(aggregate_fields, set):
            aggregate_fields = set(aggregate_fields)
        if clean_metadata_kwargs is None:
            clean_metadata_kwargs = {}
        if dump_metadata_kwargs is None:
            dump_metadata_kwargs = {}
        if to_markdown_kwargs is None:
            to_markdown_kwargs = {}

        new_d = {}
        metadata_keys = []
        for k, v in d.items():
            if k == "link":
                new_d[k] = cls.get_channel_link(v[0])
            if k == "timestamp":
                new_d[k] = v[0]
            if k == "channel":
                new_d[k] = v[0]
                continue
            if k == "content":
                new_d[k] = []
                continue
            if k in aggregate_fields and isinstance(v[0], list):
                new_v = []
                for _v in new_v:
                    for __v in _v:
                        if __v not in new_v:
                            new_v.append(__v)
                new_d[k] = new_v
                continue
            if k == "reactions" and k in aggregate_fields:
                new_d[k] = sum(v)
                continue
            if parent_links_only:
                if k in ("link", "block", "thread", "reference", "replies"):
                    continue
            metadata_keys.append(k)
        if len(metadata_keys) > 0:
            for i in range(len(d[metadata_keys[0]])):
                content = d["content"][i].strip()
                metadata = {}
                for k in metadata_keys:
                    metadata[k] = d[k][i]
                if len(new_d["content"]) > 0:
                    new_d["content"].append("\n\n")
                metadata = ToMarkdownAssetFunc.get_markdown_metadata(
                    metadata,
                    root_metadata_key="message",
                    allow_empty=not content,
                    minimize_metadata=minimize_metadata,
                    minimize_keys=minimize_keys,
                    clean_metadata=clean_metadata,
                    clean_metadata_kwargs=clean_metadata_kwargs,
                    dump_metadata_kwargs=dump_metadata_kwargs,
                    metadata_fence=metadata_fence,
                    **to_markdown_kwargs,
                )
                new_d["content"].append(metadata)
                if content:
                    new_d["content"].append("\n\n" + content)
        new_d["content"] = "".join(new_d["content"])
        return new_d
