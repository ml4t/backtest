# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the base class."""

from vectorbtpro import _typing as tp

__all__ = ["Base"]


class Base:
    """Base class for all VBT classes.

    Provides utility methods to retrieve related assets such as API pages, documentation,
    messages, code examples, and more via `vectorbtpro.utils.knowledge.custom_assets`.
    """

    @classmethod
    def find_api(cls, **kwargs) -> tp.MaybePagesAsset:
        """Return API pages and headings relevant to the class or its attributes.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.custom_assets.find_api`.

        Returns:
            MaybePagesAsset: API pages and headings relevant to this class.
        """
        from vectorbtpro.utils.knowledge.custom_assets import find_api

        return find_api(cls, **kwargs)

    @classmethod
    def find_docs(cls, **kwargs) -> tp.MaybePagesAsset:
        """Return documentation pages and headings relevant to the class or its attributes.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.custom_assets.find_docs`.

        Returns:
            MaybePagesAsset: Documentation pages and headings relevant to this class.
        """
        from vectorbtpro.utils.knowledge.custom_assets import find_docs

        return find_docs(cls, **kwargs)

    @classmethod
    def find_messages(cls, **kwargs) -> tp.MaybeMessagesAsset:
        """Return messages relevant to the class or its attributes.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.custom_assets.find_messages`.

        Returns:
            MaybeMessagesAsset: Messages relevant to this class.
        """
        from vectorbtpro.utils.knowledge.custom_assets import find_messages

        return find_messages(cls, **kwargs)

    @classmethod
    def find_examples(cls, **kwargs) -> tp.MaybeVBTAsset:
        """Return code examples relevant to the class or its attributes.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.custom_assets.find_examples`.

        Returns:
            MaybeVBTAsset: Code examples relevant to this class.
        """
        from vectorbtpro.utils.knowledge.custom_assets import find_examples

        return find_examples(cls, **kwargs)

    @classmethod
    def find_assets(cls, **kwargs) -> tp.MaybeDict[tp.VBTAsset]:
        """Return all assets relevant to the class or its attributes.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.custom_assets.find_assets`.

        Returns:
            MaybeDict[VBTAsset]: All assets relevant to this class.
        """
        from vectorbtpro.utils.knowledge.custom_assets import find_assets

        return find_assets(cls, **kwargs)

    @classmethod
    def chat(cls, message: str, chat_history: tp.ChatHistory = None, **kwargs) -> tp.ChatOutput:
        """Process a chat message for the class or its attributes.

        Args:
            message (str): Chat message.
            chat_history (ChatHistory): Chat history, a list of dictionaries with defined roles.
            **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.custom_assets.chat_about`.

        Returns:
            ChatOutput: Output data from the chat process.
        """
        from vectorbtpro.utils.knowledge.custom_assets import chat_about

        return chat_about(cls, message, chat_history=chat_history, **kwargs)
