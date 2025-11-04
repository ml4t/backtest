# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for processing chat interactions.

See `vectorbtpro.utils.knowledge` for the toy dataset.
"""

import ast
import hashlib
import inspect
import re
import sys
from collections.abc import MutableMapping
from pathlib import Path

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import (
    Configured,
    ExtSettingsPath,
    HasSettings,
    flat_merge_dicts,
    merge_dicts,
)
from vectorbtpro.utils.decorators import hybrid_method, memoized_method
from vectorbtpro.utils.knowledge.formatting import (
    ContentFormatter,
    HTMLFileFormatter,
    resolve_formatter,
)
from vectorbtpro.utils.parsing import get_forward_args, get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.template import CustomTemplate, RepFunc, SafeSub
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from tiktoken import Encoding as EncodingT
else:
    EncodingT = "tiktoken.Encoding"
if tp.TYPE_CHECKING:
    from openai import OpenAI as OpenAIT
    from openai import Stream as StreamT
    from openai.types.chat.chat_completion import ChatCompletion as ChatCompletionT
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as ChatCompletionChunkT
else:
    OpenAIT = "openai.OpenAI"
    StreamT = "openai.Stream"
    ChatCompletionT = "openai.types.chat.chat_completion.ChatCompletion"
    ChatCompletionChunkT = "openai.types.chat.chat_completion_chunk.ChatCompletionChunk"
if tp.TYPE_CHECKING:
    from litellm import CustomStreamWrapper as CustomStreamWrapperT
    from litellm import ModelResponse as ModelResponseT
else:
    ModelResponseT = "litellm.ModelResponse"
    CustomStreamWrapperT = "litellm.CustomStreamWrapper"
if tp.TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding as BaseEmbeddingT
    from llama_index.core.llms import LLM as LLMT
    from llama_index.core.llms import ChatResponse as ChatResponseT
    from llama_index.core.node_parser import NodeParser as NodeParserT
else:
    BaseEmbeddingT = "llama_index.core.embeddings.BaseEmbedding"
    LLMT = "llama_index.core.llms.LLM"
    ChatMessageT = "llama_index.core.llms.ChatMessage"
    ChatResponseT = "llama_index.core.llms.ChatResponse"
    NodeParserT = "llama_index.core.node_parser.NodeParser"
if tp.TYPE_CHECKING:
    from lmdbm import Lmdb as LmdbT
else:
    LmdbT = "lmdbm.Lmdb"
if tp.TYPE_CHECKING:
    from bm25s import BM25 as BM25T
    from bm25s.tokenization import Tokenizer as BM25TokenizerT
else:
    BM25TokenizerT = "bm25s.tokenization.Tokenizer"
    BM25T = "bm25s.BM25"

__all__ = [
    "Tokenizer",
    "TikTokenizer",
    "tokenize",
    "detokenize",
    "Embeddings",
    "OpenAIEmbeddings",
    "LiteLLMEmbeddings",
    "LlamaIndexEmbeddings",
    "embed",
    "Completions",
    "OpenAICompletions",
    "LiteLLMCompletions",
    "LlamaIndexCompletions",
    "complete",
    "completed",
    "TextSplitter",
    "TokenSplitter",
    "SegmentSplitter",
    "SourceSplitter",
    "PythonSplitter",
    "MarkdownSplitter",
    "LlamaIndexSplitter",
    "split_text",
    "StoreObject",
    "StoreData",
    "StoreDocument",
    "TextDocument",
    "StoreEmbedding",
    "ObjectStore",
    "DictStore",
    "MemoryStore",
    "FileStore",
    "LMDBStore",
    "EmbeddedDocument",
    "ScoredDocument",
    "DocumentRanker",
    "embed_documents",
    "rank_documents",
    "Rankable",
    "Contextable",
    "RankContextable",
]


# ############# Tokenizers ############# #


class Tokenizer(Configured):
    """Abstract class for tokenizers.

    Args:
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.tokenizer_config`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _settings_path: tp.SettingsPath = [
        "knowledge",
        "knowledge.chat",
        "knowledge.chat.tokenizer_config",
    ]

    def __init__(self, template_context: tp.KwargsLike = None, **kwargs) -> None:
        Configured.__init__(self, template_context=template_context, **kwargs)

        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._template_context = template_context

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    def encode(self, text: str) -> tp.Tokens:
        """Return a list of tokens corresponding to the given text.

        Args:
            text (str): Text to encode.

        Returns:
            list: List of tokens representing the input text.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def decode(self, tokens: tp.Tokens) -> str:
        """Return the text obtained by decoding the given list of tokens.

        Args:
            tokens (list): List of tokens to decode.

        Returns:
            str: Decoded text.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @memoized_method
    def encode_single(self, text: str) -> tp.Token:
        """Return a single token encoded from the given text.

        Args:
            text (str): Text to encode.

        Returns:
            Token: Single token representing the input text.

        Raises:
            ValueError: If the text contains multiple tokens.
        """
        tokens = self.encode(text)
        if len(tokens) > 1:
            raise ValueError("Text contains multiple tokens")
        return tokens[0]

    @memoized_method
    def decode_single(self, token: tp.Token) -> str:
        """Return the text decoded from the provided single token.

        Args:
            token: Token to decode.

        Returns:
            str: Decoded text.
        """
        return self.decode([token])

    def count_tokens(self, text: str) -> int:
        """Return the total number of tokens in the provided text.

        Args:
            text (str): Text for token counting.

        Returns:
            int: Number of tokens.
        """
        return len(self.encode(text))

    def count_tokens_in_messages(self, messages: tp.ChatMessages) -> int:
        """Return the total number of tokens across the provided messages.

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.

        Returns:
            int: Total token count.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


class TikTokenizer(Tokenizer):
    """Tokenizer class for tiktoken.

    Encoding can be a model name, an encoding name, or an encoding object for tokenization.

    Args:
        encoding (Union[None, str, Encoding]): Encoding specification as a model name,
            encoding name, or encoding object.
        model (Optional[str]): Model identifier used to determine the encoding.
        tokens_per_message (Optional[int]): Number of tokens charged per message.
        tokens_per_name (Optional[int]): Additional token count for message names.
        **kwargs: Keyword arguments for `Tokenizer`.

    !!! info
        For default settings, see `chat.tokenizer_configs.tiktoken` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "tiktoken"

    _settings_path: tp.SettingsPath = "knowledge.chat.tokenizer_configs.tiktoken"

    def __init__(
        self,
        encoding: tp.Union[None, str, EncodingT] = None,
        model: tp.Optional[str] = None,
        tokens_per_message: tp.Optional[int] = None,
        tokens_per_name: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        Tokenizer.__init__(
            self,
            encoding=encoding,
            model=model,
            tokens_per_message=tokens_per_message,
            tokens_per_name=tokens_per_name,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tiktoken")
        from tiktoken import Encoding, encoding_for_model, get_encoding

        encoding = self.resolve_setting(encoding, "encoding")
        model = self.resolve_setting(model, "model")
        tokens_per_message = self.resolve_setting(tokens_per_message, "tokens_per_message")
        tokens_per_name = self.resolve_setting(tokens_per_name, "tokens_per_name")

        if isinstance(encoding, str):
            if encoding.startswith("model_or_"):
                try:
                    if model is None:
                        raise KeyError
                    encoding = encoding_for_model(model)
                except KeyError:
                    encoding = encoding[len("model_or_") :]
                    encoding = (
                        get_encoding(encoding)
                        if "k_base" in encoding
                        else encoding_for_model(encoding)
                    )
            elif isinstance(encoding, str):
                encoding = (
                    get_encoding(encoding) if "k_base" in encoding else encoding_for_model(encoding)
                )
        checks.assert_instance_of(encoding, Encoding, arg_name="encoding")

        self._encoding = encoding
        self._tokens_per_message = tokens_per_message
        self._tokens_per_name = tokens_per_name

    @property
    def encoding(self) -> EncodingT:
        """Token encoding object used for tokenization.

        Returns:
            Encoding: Encoding object.
        """
        return self._encoding

    @property
    def tokens_per_message(self) -> int:
        """Token count charged per message.

        Returns:
            int: Number of tokens charged per message.
        """
        return self._tokens_per_message

    @property
    def tokens_per_name(self) -> int:
        """Additional token count for message names.

        Returns:
            int: Number of tokens charged for message names.
        """
        return self._tokens_per_name

    def encode(self, text: str) -> tp.Tokens:
        return self.encoding.encode(text)

    def decode(self, tokens: tp.Tokens) -> str:
        return self.encoding.decode(tokens)

    def count_tokens_in_messages(self, messages: tp.ChatMessages) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                num_tokens += self.count_tokens(value)
                if key == "name":
                    num_tokens += self.tokens_per_name
        num_tokens += 3
        return num_tokens


def resolve_tokenizer(tokenizer: tp.TokenizerLike = None) -> tp.MaybeType[Tokenizer]:
    """Resolve a `Tokenizer` subclass or instance.

    Args:
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

            Supported identifiers:

            * "tiktoken" for `TikTokenizer`

    Returns:
        Tokenizer: Resolved tokenizer type or instance.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.
    """
    if tokenizer is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        tokenizer = chat_cfg["tokenizer"]
    if isinstance(tokenizer, str):
        curr_module = sys.modules[__name__]
        found_tokenizer = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Tokenizer"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == tokenizer.lower():
                    found_tokenizer = cls
                    break
        if found_tokenizer is None:
            raise ValueError(f"Invalid tokenizer: '{tokenizer}'")
        tokenizer = found_tokenizer
    if isinstance(tokenizer, type):
        checks.assert_subclass_of(tokenizer, Tokenizer, arg_name="tokenizer")
    else:
        checks.assert_instance_of(tokenizer, Tokenizer, arg_name="tokenizer")
    return tokenizer


def tokenize(text: str, tokenizer: tp.TokenizerLike = None, **kwargs) -> tp.Tokens:
    """Tokenize text using a resolved `Tokenizer`.

    Args:
        text (str): Text to tokenize.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

            Resolved using `resolve_tokenizer`.
        **kwargs: Keyword arguments to initialize or update `tokenizer`.

    Returns:
        Tokens: List of tokens representing the input text.
    """
    tokenizer = resolve_tokenizer(tokenizer=tokenizer)
    if isinstance(tokenizer, type):
        tokenizer = tokenizer(**kwargs)
    elif kwargs:
        tokenizer = tokenizer.replace(**kwargs)
    return tokenizer.encode(text)


def detokenize(tokens: tp.Tokens, tokenizer: tp.TokenizerLike = None, **kwargs) -> str:
    """Detokenize tokens into text using a resolved `Tokenizer`.

    Args:
        tokens (Tokens): List of tokens to decode.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

            Resolved using `resolve_tokenizer`.
        **kwargs: Keyword arguments to initialize or update `tokenizer`.

    Returns:
        str: Decoded text.
    """
    tokenizer = resolve_tokenizer(tokenizer=tokenizer)
    if isinstance(tokenizer, type):
        tokenizer = tokenizer(**kwargs)
    elif kwargs:
        tokenizer = tokenizer.replace(**kwargs)
    return tokenizer.decode(tokens)


# ############# Embeddings ############# #


class Embeddings(Configured):
    """Abstract class for embedding providers.

    Args:
        batch_size (Optional[int]): Batch size for processing queries.

            Use None to disable batching.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (Kwargs): Keyword arguments for configuring the progress bar.
        template_context (Kwargs): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.embeddings_config`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = [
        "knowledge",
        "knowledge.chat",
        "knowledge.chat.embeddings_config",
    ]

    def __init__(
        self,
        batch_size: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            batch_size=batch_size,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

        batch_size = self.resolve_setting(batch_size, "batch_size")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._batch_size = batch_size
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

    @property
    def batch_size(self) -> tp.Optional[int]:
        """Batch size used for processing queries.

        Use None to disable batching.

        Returns:
            Optional[int]: Batch size.
        """
        return self._batch_size

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to display a progress bar.

        Returns:
            Optional[bool]: True if progress bar is shown, False otherwise.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for configuring `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Keyword arguments for the progress bar.
        """
        return self._pbar_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def model(self) -> tp.Optional[str]:
        """Model identifier.

        Returns:
            Optional[str]: Model identifier; None by default.
        """
        return None

    def get_embedding(self, query: str) -> tp.List[float]:
        """Return the embedding vector for the given query.

        Args:
            query (str): Query text.

        Returns:
            List[float]: Embedding vector.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        """Return a batch of embedding vectors for a list of queries.

        Args:
            batch (List[str]): List of query texts.

        Returns:
            List[List[float]]: List containing an embedding vector for each query.
        """
        return [self.get_embedding(query) for query in batch]

    def iter_embedding_batches(self, queries: tp.List[str]) -> tp.Iterator[tp.List[tp.List[float]]]:
        """Return an iterator over batches of embeddings.

        Args:
            queries (List[str]): List of query texts.

        Returns:
            Iterator[List[List[float]]]: Iterator yielding batches of embedding vectors.
        """
        from vectorbtpro.utils.pbar import ProgressBar

        if self.batch_size is not None:
            batches = [
                queries[i : i + self.batch_size] for i in range(0, len(queries), self.batch_size)
            ]
        else:
            batches = [queries]
        pbar_kwargs = merge_dicts(dict(prefix="get_embeddings"), self.pbar_kwargs)
        with ProgressBar(
            total=len(queries), show_progress=self.show_progress, **pbar_kwargs
        ) as pbar:
            for batch in batches:
                yield self.get_embedding_batch(batch)
                pbar.update(len(batch))

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[tp.List[float]]:
        """Return embeddings for multiple queries.

        Args:
            queries (List[str]): List of query texts.

        Returns:
            List[List[float]]: List containing an embedding vector for each query.
        """
        return [embedding for batch in self.iter_embedding_batches(queries) for embedding in batch]


class OpenAIEmbeddings(Embeddings):
    """Embeddings class for OpenAI.

    Args:
        model (Optional[str]): OpenAI model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `openai.OpenAI`.
        embeddings_kwargs (KwargsLike): Keyword arguments for `openai.resources.embeddings.Embeddings.create`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embeddings_kwargs`.

    !!! info
        For default settings, see `chat.embeddings_configs.openai` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.openai"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embeddings_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embeddings_kwargs=embeddings_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        super_arg_names = set(get_func_arg_names(Embeddings.__init__))
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = openai_config.pop("model", None)
        def_client_kwargs = openai_config.pop("client_kwargs", None)
        def_embeddings_kwargs = openai_config.pop("embeddings_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_kwargs = get_func_kwargs(type(self).__init__)
        for k in list(openai_config.keys()):
            if k in init_kwargs:
                openai_config.pop(k)

        client_arg_names = set(get_func_arg_names(OpenAI.__init__))
        _client_kwargs = {}
        _embeddings_kwargs = {}
        for k, v in openai_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embeddings_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embeddings_kwargs = merge_dicts(
            _embeddings_kwargs, def_embeddings_kwargs, embeddings_kwargs
        )
        client = OpenAI(**client_kwargs)

        self._model = model
        self._client = client
        self._embeddings_kwargs = embeddings_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OpenAIT:
        """OpenAI client instance.

        Returns:
            OpenAI: OpenAI client instance.
        """
        return self._client

    @property
    def embeddings_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `openai.resources.embeddings.Embeddings.create`.

        Returns:
            Kwargs: Keyword arguments for creating embeddings.
        """
        return self._embeddings_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        response = self.client.embeddings.create(
            input=query, model=self.model, **self.embeddings_kwargs
        )
        return response.data[0].embedding

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        response = self.client.embeddings.create(
            input=batch, model=self.model, **self.embeddings_kwargs
        )
        return [embedding.embedding for embedding in response.data]


class LiteLLMEmbeddings(Embeddings):
    """Embeddings class for LiteLLM.

    Args:
        model (Optional[str]): LiteLLM model identifier.
        embedding_kwargs (KwargsLike): Keyword arguments for `litellm.embedding`.
        **kwargs: Keyword arguments for `Embeddings` or used as `embedding_kwargs`.

    !!! info
        For default settings, see `chat.embeddings_configs.litellm` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.litellm"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        embedding_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            embedding_kwargs=embedding_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        super_arg_names = set(get_func_arg_names(Embeddings.__init__))
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        litellm_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = litellm_config.pop("model", None)
        def_embedding_kwargs = litellm_config.pop("embedding_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_kwargs = get_func_kwargs(type(self).__init__)
        for k in list(litellm_config.keys()):
            if k in init_kwargs:
                litellm_config.pop(k)
        embedding_kwargs = merge_dicts(litellm_config, def_embedding_kwargs, embedding_kwargs)

        self._model = model
        self._embedding_kwargs = embedding_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `litellm.embedding`.

        Returns:
            Kwargs: Keyword arguments for creating embeddings.
        """
        return self._embedding_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        from litellm import embedding

        response = embedding(self.model, input=query, **self.embedding_kwargs)
        return response.data[0]["embedding"]

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        from litellm import embedding

        response = embedding(self.model, input=batch, **self.embedding_kwargs)
        return [embedding["embedding"] for embedding in response.data]


class LlamaIndexEmbeddings(Embeddings):
    """Embeddings class for LlamaIndex.

    This class initializes embeddings for LlamaIndex using a specified identifier or instance.
    It combines configuration from `vectorbtpro._settings.knowledge` with provided parameters.

    Args:
        embedding (Union[None, str, BaseEmbedding]): Embedding identifier or instance.

            If None, a default from settings is used.
        embedding_kwargs (KwargsLike): Keyword arguments for embedding initialization.
        **kwargs: Keyword arguments for `Embeddings` or used as `embedding_kwargs`.

    !!! info
        For default settings, see `chat.embeddings_configs.llama_index` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.llama_index"

    def __init__(
        self,
        embedding: tp.Union[None, str, tp.MaybeType[BaseEmbeddingT]] = None,
        embedding_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            embedding=embedding,
            embedding_kwargs=embedding_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.embeddings import BaseEmbedding

        super_arg_names = set(get_func_arg_names(Embeddings.__init__))
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_embedding = llama_index_config.pop("embedding", None)
        def_embedding_kwargs = llama_index_config.pop("embedding_kwargs", None)

        if embedding is None:
            embedding = def_embedding
        if embedding is None:
            raise ValueError("Must provide an embedding name or path")
        init_kwargs = get_func_kwargs(type(self).__init__)
        for k in list(llama_index_config.keys()):
            if k in init_kwargs:
                llama_index_config.pop(k)

        if isinstance(embedding, str):
            import llama_index.embeddings

            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, BaseEmbedding):
                    if "." in embedding:
                        if k.endswith(embedding):
                            return True
                    else:
                        if k.split(".")[-1].lower() == embedding.lower():
                            return True
                        if k.split(".")[-1].replace(
                            "Embedding", ""
                        ).lower() == embedding.lower().replace("_", ""):
                            return True
                return False

            found_embedding = search_package(
                llama_index.embeddings,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_embedding is None:
                raise ValueError(f"Embedding '{embedding}' not found")
            embedding = found_embedding
        if isinstance(embedding, type):
            checks.assert_subclass_of(embedding, BaseEmbedding, arg_name="embedding")
            embedding_name = embedding.__name__.replace("Embedding", "").lower()
            module_name = embedding.__module__
        else:
            checks.assert_instance_of(embedding, BaseEmbedding, arg_name="embedding")
            embedding_name = type(embedding).__name__.replace("Embedding", "").lower()
            module_name = type(embedding).__module__
        embedding_configs = llama_index_config.pop("embedding_configs", {})
        if embedding_name in embedding_configs:
            llama_index_config = merge_dicts(llama_index_config, embedding_configs[embedding_name])
        elif module_name in embedding_configs:
            llama_index_config = merge_dicts(llama_index_config, embedding_configs[module_name])
        embedding_kwargs = merge_dicts(llama_index_config, def_embedding_kwargs, embedding_kwargs)
        model_name = embedding_kwargs.get("model_name", None)
        if model_name is None:
            func_kwargs = get_func_kwargs(type(embedding).__init__)
            model_name = func_kwargs.get("model_name", None)
        if isinstance(embedding, type):
            embedding = embedding(**embedding_kwargs)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized embedding")

        self._model = model_name
        self._embedding = embedding

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def embedding(self) -> BaseEmbeddingT:
        """Underlying embedding instance.

        Returns:
            BaseEmbedding: Embedding instance.
        """
        return self._embedding

    def get_embedding(self, query: str) -> tp.List[float]:
        return self.embedding.get_text_embedding(query)

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[tp.List[float]]:
        return [embedding for embedding in self.embedding.get_text_embedding_batch(batch)]


def resolve_embeddings(embeddings: tp.EmbeddingsLike = None) -> tp.MaybeType[Embeddings]:
    """Return a subclass or instance of `Embeddings` based on the provided identifier or object.

    Args:
        embeddings (EmbeddingsLike): Identifier, subclass, or instance of `Embeddings`.

            Supported identifiers:

            * "openai" for `OpenAIEmbeddings`
            * "litellm" for `LiteLLMEmbeddings`
            * "llama_index" for `LlamaIndexEmbeddings`
            * "auto" to select the first available option

            If None, configuration from `vectorbtpro._settings` is used.

    Returns:
        Embeddings: Resolved embeddings subclass or instance.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.
    """
    if embeddings is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        embeddings = chat_cfg["embeddings"]
    if isinstance(embeddings, str):
        if embeddings.lower() == "auto":
            from vectorbtpro.utils.module_ import check_installed

            if check_installed("openai"):
                embeddings = "openai"
            elif check_installed("litellm"):
                embeddings = "litellm"
            elif check_installed("llama_index"):
                embeddings = "llama_index"
            else:
                raise ValueError("No packages for embeddings installed")
        curr_module = sys.modules[__name__]
        found_embeddings = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Embeddings"):
                _short_name: tp.ClassVar[tp.Optional[str]] = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == embeddings.lower():
                    found_embeddings = cls
                    break
        if found_embeddings is None:
            raise ValueError(f"Invalid embeddings: '{embeddings}'")
        embeddings = found_embeddings
    if isinstance(embeddings, type):
        checks.assert_subclass_of(embeddings, Embeddings, arg_name="embeddings")
    else:
        checks.assert_instance_of(embeddings, Embeddings, arg_name="embeddings")
    return embeddings


def embed(
    query: tp.MaybeList[str], embeddings: tp.EmbeddingsLike = None, **kwargs
) -> tp.MaybeList[tp.List[float]]:
    """Return embedding(s) for one or more queries.

    Args:
        query (MaybeList[str]): Query string or a list of query strings to embed.
        embeddings (EmbeddingsLike): Identifier, subclass, or instance of `Embeddings`.

            Resolved using `resolve_embeddings`.
        **kwargs: Keyword arguments to initialize or update `embeddings`.

    Returns:
        MaybeList[List[float]]: Embedding vector(s) corresponding to the input query or queries.
    """
    embeddings = resolve_embeddings(embeddings=embeddings)
    if isinstance(embeddings, type):
        embeddings = embeddings(**kwargs)
    elif kwargs:
        embeddings = embeddings.replace(**kwargs)
    if isinstance(query, str):
        return embeddings.get_embedding(query)
    return embeddings.get_embeddings(query)


# ############# Completions ############# #


class Completions(Configured):
    """Abstract class for completion providers.

    Args:
        context (str): Context string to be used as a user message.
        chat_history (Optional[ChatHistory]): Chat history, a list of dictionaries with defined roles.

            After a response is generated, the assistant message is appended to this history.
        stream (Optional[bool]): Boolean indicating whether responses are streamed.

            In streaming mode, chunks are appended and displayed incrementally; otherwise,
            the entire message is displayed.
        max_tokens (Union[None, bool, int]): Maximum token limit configured for messages.

            If False, the limit is disabled.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

            Resolved using `resolve_tokenizer`.
        tokenizer_kwargs (KwargsLike): Keyword arguments to initialize or update `tokenizer`.
        system_prompt (Optional[str]): System prompt that precedes the context prompt.

            This prompt is used to set the system's behavior or context for the conversation.
        system_as_user (Optional[bool]): Boolean indicating whether to use the user role for the system message.

            This is mainly used for experimental models where a dedicated system role is not available.
        context_template (Optional[str]): Context template requiring a 'context' variable.

            The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.
        formatter (ContentFormatterLike): Identifier, subclass, or instance of
            `vectorbtpro.utils.knowledge.formatting.ContentFormatter`.

            Resolved using `vectorbtpro.utils.knowledge.formatting.resolve_formatter`.

            This formatter is used to format the content of the response.
        formatter_kwargs (KwargsLike): Keyword arguments to initialize or update `formatter`.
        minimal_format (Optional[bool]): Boolean indicating if the input is minimally formatted.
        quick_mode (Optional[bool]): Boolean indicating whether quick mode is enabled.
        silence_warnings (Optional[bool]): Flag to suppress warning messages.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.completions_config`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = [
        "knowledge",
        "knowledge.chat",
        "knowledge.chat.completions_config",
    ]

    def __init__(
        self,
        context: str = "",
        chat_history: tp.Optional[tp.ChatHistory] = None,
        stream: tp.Optional[bool] = None,
        max_tokens: tp.Union[None, bool, int] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_template: tp.Optional[str] = None,
        formatter: tp.ContentFormatterLike = None,
        formatter_kwargs: tp.KwargsLike = None,
        minimal_format: tp.Optional[bool] = None,
        quick_mode: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            context=context,
            chat_history=chat_history,
            stream=stream,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            system_prompt=system_prompt,
            system_as_user=system_as_user,
            context_template=context_template,
            formatter=formatter,
            formatter_kwargs=formatter_kwargs,
            minimal_format=minimal_format,
            quick_mode=quick_mode,
            silence_warnings=silence_warnings,
            template_context=template_context,
            **kwargs,
        )

        if chat_history is None:
            chat_history = []
        stream = self.resolve_setting(stream, "stream")
        max_tokens_set = max_tokens is not None
        max_tokens = self.resolve_setting(max_tokens, "max_tokens")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(
            tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True
        )
        system_prompt = self.resolve_setting(system_prompt, "system_prompt")
        system_as_user = self.resolve_setting(system_as_user, "system_as_user")
        context_template = self.resolve_setting(context_template, "context_template")
        formatter = self.resolve_setting(formatter, "formatter", default=None)
        formatter_kwargs = self.resolve_setting(
            formatter_kwargs, "formatter_kwargs", default=None, merge=True
        )
        minimal_format = self.resolve_setting(minimal_format, "minimal_format", default=None)
        quick_mode = self.resolve_setting(quick_mode, "quick_mode")
        silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        tokenizer = resolve_tokenizer(tokenizer)
        formatter = resolve_formatter(formatter)

        self._context = context
        self._chat_history = chat_history
        self._stream = stream
        self._max_tokens_set = max_tokens_set
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._system_prompt = system_prompt
        self._system_as_user = system_as_user
        self._context_template = context_template
        self._formatter = formatter
        self._formatter_kwargs = formatter_kwargs
        self._minimal_format = minimal_format
        self._quick_mode = quick_mode
        self._silence_warnings = silence_warnings
        self._template_context = template_context

    @property
    def context(self) -> str:
        """Context string to be used as a user message.

        Returns:
            str: Context string used for expression evaluation.
        """
        return self._context

    @property
    def chat_history(self) -> tp.ChatHistory:
        """Chat history, a list of dictionaries with defined roles.

        After a response is generated, the assistant message is appended to this history.

        Returns:
            ChatHistory: List of dictionaries representing the chat history.
        """
        return self._chat_history

    @property
    def stream(self) -> bool:
        """Boolean indicating whether responses are streamed.

        In streaming mode, chunks are appended and displayed incrementally; otherwise,
        the entire message is displayed.

        Returns:
            bool: True if streaming is enabled, False otherwise.
        """
        return self._stream

    @property
    def max_tokens_set(self) -> tp.Optional[int]:
        """Boolean indicating if `Completions.max_tokens` was explicitly provided by the user.

        Returns:
            Optional[int]: Maximum token limit set by the user; None if not set.
        """
        return self._max_tokens_set

    @property
    def max_tokens(self) -> tp.Union[bool, int]:
        """Maximum token limit configured for messages.

        Returns:
            Union[bool, int]: Maximum token limit; False if disabled.
        """
        return self._max_tokens

    @property
    def tokenizer(self) -> tp.MaybeType[Tokenizer]:
        """Subclass or instance of `Tokenizer`.

        Resolved using `resolve_tokenizer`.

        Returns:
            MaybeType[Tokenizer]: Resolved tokenizer instance or subclass.
        """
        return self._tokenizer

    @property
    def tokenizer_kwargs(self) -> tp.Kwargs:
        """Keyword arguments to initialize or update `Completions.tokenizer`.

        Returns:
            Kwargs: Keyword arguments for tokenizer initialization or update.
        """
        return self._tokenizer_kwargs

    @property
    def system_prompt(self) -> str:
        """System prompt that precedes the context prompt.

        This prompt is used to set the system's behavior or context for the conversation.

        Returns:
            str: System prompt.
        """
        return self._system_prompt

    @property
    def system_as_user(self) -> bool:
        """Boolean indicating whether to use the user role for the system message.

        This is mainly used for experimental models where a dedicated system role is not available.

        Returns:
            bool: True if the system message is treated as a user message, False otherwise.
        """
        return self._system_as_user

    @property
    def context_template(self) -> str:
        """Context prompt template requiring a 'context' variable.

        The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

        This prompt is used to provide context for the conversation.

        Returns:
            str: Context prompt template.
        """
        return self._context_template

    @property
    def formatter(self) -> tp.MaybeType[ContentFormatter]:
        """Content formatter subclass or instance.

        Resolved using `vectorbtpro.utils.knowledge.formatting.resolve_formatter`.

        This formatter is used to format the content of the response.

        Returns:
            MaybeType[ContentFormatter]: Resolved content formatter instance or subclass.
        """
        return self._formatter

    @property
    def formatter_kwargs(self) -> tp.Kwargs:
        """Keyword arguments to initialize or update `Completions.formatter`.

        Returns:
            Kwargs: Keyword arguments for the content formatter.
        """
        return self._formatter_kwargs

    @property
    def minimal_format(self) -> bool:
        """Boolean indicating if the input is minimally formatted.

        Returns:
            bool: True if the input is minimally formatted, False otherwise.
        """
        return self._minimal_format

    @property
    def quick_mode(self) -> bool:
        """Boolean indicating whether quick mode is enabled.

        Returns:
            bool: True if quick mode is enabled, False otherwise.
        """
        return self._quick_mode

    @property
    def silence_warnings(self) -> bool:
        """Boolean indicating whether warnings are suppressed.

        Returns:
            bool: True if warnings are suppressed, False otherwise.
        """
        return self._silence_warnings

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def model(self) -> tp.Optional[str]:
        """Model name.

        Returns:
            Optional[str]: Model name if specified; otherwise, None.
        """
        return None

    def get_chat_response(self, messages: tp.ChatMessages, **kwargs) -> tp.Any:
        """Return a chat response based on the provided messages

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Chat response generated from the provided messages.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_message_content(self, response: tp.Any) -> tp.Optional[str]:
        """Return the content extracted from a chat response.

        Args:
            response (Any): Chat response object.

        Returns:
            Optional[str]: Content extracted from the chat response.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_stream_response(self, messages: tp.ChatMessages, **kwargs) -> tp.Any:
        """Return a streaming response generated from the provided messages.

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Streaming response generated from the provided messages.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_delta_content(self, response_chunk: tp.Any) -> tp.Optional[str]:
        """Return the content extracted from a streaming response chunk.

        Args:
            response (Any): Streaming response object.

        Returns:
            Optional[str]: Content extracted from the streaming response chunk.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def prepare_messages(self, message: str) -> tp.ChatMessages:
        """Return a list of chat messages formatted for a completion request.

        Args:
            message (str): User message to process.

        Returns:
            ChatMessages: List of dictionaries representing the conversation history.
        """
        context = self.context
        chat_history = self.chat_history
        max_tokens_set = self.max_tokens_set
        max_tokens = self.max_tokens
        tokenizer = self.tokenizer
        tokenizer_kwargs = self.tokenizer_kwargs
        system_prompt = self.system_prompt
        system_as_user = self.system_as_user
        context_template = self.context_template
        template_context = self.template_context
        silence_warnings = self.silence_warnings

        if isinstance(tokenizer, type):
            tokenizer_kwargs = dict(tokenizer_kwargs)
            tokenizer_kwargs["template_context"] = merge_dicts(
                template_context, tokenizer_kwargs.get("template_context")
            )
            if issubclass(tokenizer, TikTokenizer) and "model" not in tokenizer_kwargs:
                tokenizer_kwargs["model"] = self.model
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)

        if context:
            if isinstance(context_template, str):
                context_template = SafeSub(context_template)
            elif checks.is_function(context_template):
                context_template = RepFunc(context_template)
            elif not isinstance(context_template, CustomTemplate):
                raise TypeError("Context prompt must be a string, function, or template")
            if max_tokens not in (None, False):
                if max_tokens is True:
                    raise ValueError("max_tokens cannot be True")
                empty_context_template = context_template.substitute(
                    flat_merge_dicts(dict(context=""), template_context),
                    eval_id="context_template",
                )
                empty_messages = [
                    dict(role="user" if system_as_user else "system", content=system_prompt),
                    dict(role="user", content=empty_context_template),
                    *chat_history,
                    dict(role="user", content=message),
                ]
                num_tokens = tokenizer.count_tokens_in_messages(empty_messages)
                max_context_tokens = max(0, max_tokens - num_tokens)
                encoded_context = tokenizer.encode(context)
                if len(encoded_context) > max_context_tokens:
                    context = tokenizer.decode(encoded_context[:max_context_tokens])
                    if not max_tokens_set and not silence_warnings:
                        warn(
                            f"Context is too long ({len(encoded_context)}). "
                            f"Truncating to {max_context_tokens} tokens."
                        )
            template_context = flat_merge_dicts(dict(context=context), template_context)
            context_template = context_template.substitute(
                template_context, eval_id="context_template"
            )
            return [
                dict(role="user" if system_as_user else "system", content=system_prompt),
                dict(role="user", content=context_template),
                *chat_history,
                dict(role="user", content=message),
            ]
        else:
            return [
                dict(role="user" if system_as_user else "system", content=system_prompt),
                *chat_history,
                dict(role="user", content=message),
            ]

    def get_completion(
        self,
        message: str,
        return_response: bool = False,
    ) -> tp.ChatOutput:
        """Return the formatted completion output for a provided message.

        Args:
            message (str): User message to generate a completion for.
            return_response (bool): Flag to return the raw response along with the file path.

        Returns:
            ChatOutput: File path for the formatted output; if `return_response` is True,
                a tuple containing the file path and raw response.
        """
        chat_history = self.chat_history
        stream = self.stream
        formatter = self.formatter
        formatter_kwargs = self.formatter_kwargs
        template_context = self.template_context

        messages = self.prepare_messages(message)
        if self.stream:
            response = self.get_stream_response(messages)
        else:
            response = self.get_chat_response(messages)

        if isinstance(formatter, type):
            formatter_kwargs = dict(formatter_kwargs)
            if "minimal_format" not in formatter_kwargs:
                formatter_kwargs["minimal_format"] = self.minimal_format
            formatter_kwargs["template_context"] = merge_dicts(
                template_context, formatter_kwargs.get("template_context")
            )
            if issubclass(formatter, HTMLFileFormatter):
                if "page_title" not in formatter_kwargs:
                    formatter_kwargs["page_title"] = message
                if "cache_dir" not in formatter_kwargs:
                    chat_dir = self.get_setting("chat_dir", default=None)
                    if isinstance(chat_dir, CustomTemplate):
                        cache_dir = self.get_setting("cache_dir", default=None)
                        if cache_dir is not None:
                            if isinstance(cache_dir, CustomTemplate):
                                cache_dir = cache_dir.substitute(
                                    template_context, eval_id="cache_dir"
                                )
                            template_context = flat_merge_dicts(
                                dict(cache_dir=cache_dir), template_context
                            )
                        release_dir = self.get_setting("release_dir", default=None)
                        if release_dir is not None:
                            if isinstance(release_dir, CustomTemplate):
                                release_dir = release_dir.substitute(
                                    template_context, eval_id="release_dir"
                                )
                            template_context = flat_merge_dicts(
                                dict(release_dir=release_dir), template_context
                            )
                        chat_dir = chat_dir.substitute(template_context, eval_id="chat_dir")
                    chat_dir = Path(chat_dir) / "html"
                    formatter_kwargs["dir_path"] = chat_dir
            formatter = formatter(**formatter_kwargs)
        elif formatter_kwargs:
            formatter = formatter.replace(**formatter_kwargs)
        if stream:
            with formatter:
                for i, response_chunk in enumerate(response):
                    new_content = self.get_delta_content(response_chunk)
                    if new_content is not None:
                        formatter.append(new_content)
                content = formatter.content
        else:
            content = self.get_message_content(response)
            if content is None:
                content = ""
            formatter.append_once(content)

        chat_history.append(dict(role="user", content=message))
        chat_history.append(dict(role="assistant", content=content))
        if isinstance(formatter, HTMLFileFormatter) and formatter.file_handle is not None:
            file_path = Path(formatter.file_handle.name)
        else:
            file_path = None
        if return_response:
            return file_path, response
        return file_path

    def get_completion_content(self, message: str) -> str:
        """Return the text content of a completion for a given message.

        Args:
            message (str): User message to complete.

        Returns:
            str: Generated completion text.
        """
        chat_history = self.chat_history

        messages = self.prepare_messages(message)
        response = self.get_chat_response(messages)
        content = self.get_message_content(response)
        if content is None:
            content = ""
        chat_history.append(dict(role="user", content=message))
        chat_history.append(dict(role="assistant", content=content))
        return content


class OpenAICompletions(Completions):
    """Completions class for OpenAI.

    Args:
        model (Optional[str]): Identifier for the model to use.
        client_kwargs (KwargsLike): Keyword arguments for `openai.OpenAI`.
        completions_kwargs (KwargsLike): Keyword arguments for `openai.Completions.create`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `completions_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.openai` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.openai"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        completions_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            completions_kwargs=completions_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        super_arg_names = set(get_func_arg_names(Completions.__init__))
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = openai_config.pop("model", None)
        def_quick_model = openai_config.pop("quick_model", None)
        def_client_kwargs = openai_config.pop("client_kwargs", None)
        def_completions_kwargs = openai_config.pop("completions_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_kwargs = get_func_kwargs(type(self).__init__)
        for k in list(openai_config.keys()):
            if k in init_kwargs:
                openai_config.pop(k)

        client_arg_names = set(get_func_arg_names(OpenAI.__init__))
        _client_kwargs = {}
        _completions_kwargs = {}
        for k, v in openai_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _completions_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        completions_kwargs = merge_dicts(
            _completions_kwargs, def_completions_kwargs, completions_kwargs
        )
        client = OpenAI(**client_kwargs)

        self._model = model
        self._client = client
        self._completions_kwargs = completions_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OpenAIT:
        """OpenAI client instance used for API calls.

        Returns:
            OpenAI: OpenAI client instance.
        """
        return self._client

    @property
    def completions_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `openai.Completions.create`.

        Returns:
            Kwargs: Keyword arguments for the completion API call.
        """
        return self._completions_kwargs

    def get_chat_response(self, messages: tp.ChatMessages) -> ChatCompletionT:
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=False,
            **self.completions_kwargs,
        )

    def get_message_content(self, response: ChatCompletionT) -> tp.Optional[str]:
        return response.choices[0].message.content

    def get_stream_response(self, messages: tp.ChatMessages) -> StreamT:
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            **self.completions_kwargs,
        )

    def get_delta_content(self, response_chunk: ChatCompletionChunkT) -> tp.Optional[str]:
        return response_chunk.choices[0].delta.content


class LiteLLMCompletions(Completions):
    """Completions class for LiteLLM.

    Args:
        model (Optional[str]): Identifier for the model to use.
        completion_kwargs (KwargsLike): Keyword arguments for `litellm.completion`.
        **kwargs: Keyword arguments for `Completions` or used as `completion_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.litellm` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.litellm"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        completion_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            completion_kwargs=completion_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        super_arg_names = set(get_func_arg_names(Completions.__init__))
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        litellm_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = litellm_config.pop("model", None)
        def_quick_model = litellm_config.pop("quick_model", None)
        def_completion_kwargs = litellm_config.pop("completion_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        completion_kwargs = merge_dicts(litellm_config, def_completion_kwargs, completion_kwargs)

        self._model = model
        self._completion_kwargs = completion_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for the `litellm.completion` API call.

        Returns:
            Kwargs: Keyword arguments for the completion API call.
        """
        return self._completion_kwargs

    def get_chat_response(self, messages: tp.ChatMessages) -> ModelResponseT:
        from litellm import completion

        return completion(
            messages=messages,
            model=self.model,
            stream=False,
            **self.completion_kwargs,
        )

    def get_message_content(self, response: ModelResponseT) -> tp.Optional[str]:
        return response.choices[0].message.content

    def get_stream_response(self, messages: tp.ChatMessages) -> CustomStreamWrapperT:
        from litellm import completion

        return completion(
            messages=messages,
            model=self.model,
            stream=True,
            **self.completion_kwargs,
        )

    def get_delta_content(self, response_chunk: ModelResponseT) -> tp.Optional[str]:
        return response_chunk.choices[0].delta.content


class LlamaIndexCompletions(Completions):
    """Completions class for LlamaIndex.

    LLM can be provided via `llm`, which can be either the name of the class (case doesn't matter),
    the path or its suffix to the class (case matters), or a subclass or an instance of
    `llama_index.core.llms.LLM`.

    Args:
        llm (Union[None, str, MaybeType[LLM]]): Identifier, class path, subclass, or instance of
            `llama_index.core.llms.LLM`.
        llm_kwargs (KwargsLike): Additional parameters for LLM initialization.
        **kwargs: Keyword arguments for `Completions` or used as `llm_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.llama_index` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.llama_index"

    def __init__(
        self,
        llm: tp.Union[None, str, tp.MaybeType[LLMT]] = None,
        llm_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            llm=llm,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.llms import LLM

        super_arg_names = set(get_func_arg_names(Completions.__init__))
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_llm = llama_index_config.pop("llm", None)
        def_llm_kwargs = llama_index_config.pop("llm_kwargs", None)

        if llm is None:
            llm = def_llm
        if llm is None:
            raise ValueError("Must provide an LLM name or path")
        init_kwargs = get_func_kwargs(type(self).__init__)
        for k in list(llama_index_config.keys()):
            if k in init_kwargs:
                llama_index_config.pop(k)

        if isinstance(llm, str):
            import llama_index.llms

            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, LLM):
                    if "." in llm:
                        if k.endswith(llm):
                            return True
                    else:
                        if k.split(".")[-1].lower() == llm.lower():
                            return True
                        if k.split(".")[-1].replace("LLM", "").lower() == llm.lower().replace(
                            "_", ""
                        ):
                            return True
                return False

            found_llm = search_package(
                llama_index.llms,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_llm is None:
                raise ValueError(f"LLM '{llm}' not found")
            llm = found_llm
        if isinstance(llm, type):
            checks.assert_subclass_of(llm, LLM, arg_name="llm")
            llm_name = llm.__name__.replace("LLM", "").lower()
            module_name = llm.__module__
        else:
            checks.assert_instance_of(llm, LLM, arg_name="llm")
            llm_name = type(llm).__name__.replace("LLM", "").lower()
            module_name = type(llm).__module__
        llm_configs = llama_index_config.pop("llm_configs", {})
        if llm_name in llm_configs:
            llama_index_config = merge_dicts(llama_index_config, llm_configs[llm_name])
        elif module_name in llm_configs:
            llama_index_config = merge_dicts(llama_index_config, llm_configs[module_name])
        llm_kwargs = merge_dicts(llama_index_config, def_llm_kwargs, llm_kwargs)
        def_model = llm_kwargs.pop("model", None)
        quick_model = llm_kwargs.pop("quick_model", None)
        model = quick_model if self.quick_mode else def_model
        if model is None:
            func_kwargs = get_func_kwargs(type(llm).__init__)
            model = func_kwargs.get("model", None)
        else:
            llm_kwargs["model"] = model
        if isinstance(llm, type):
            llm = llm(**llm_kwargs)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized LLM")

        self._model = model
        self._llm = llm

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def llm(self) -> LLMT:
        """Initialized LLM instance used for generating completions.

        Returns:
            LLM: Initialized LLM instance.
        """
        return self._llm

    def get_chat_response(self, messages: tp.ChatMessages) -> ChatResponseT:
        from llama_index.core.llms import ChatMessage

        return self.llm.chat(list(map(lambda x: ChatMessage(**dict(x)), messages)))

    def get_message_content(self, response: ChatResponseT) -> tp.Optional[str]:
        return response.message.content

    def get_stream_response(self, messages: tp.ChatMessages) -> tp.Iterator[ChatResponseT]:
        from llama_index.core.llms import ChatMessage

        return self.llm.stream_chat(list(map(lambda x: ChatMessage(**dict(x)), messages)))

    def get_delta_content(self, response_chunk: ChatResponseT) -> tp.Optional[str]:
        return response_chunk.delta


def resolve_completions(completions: tp.CompletionsLike = None) -> tp.MaybeType[Completions]:
    """Resolve and return a `Completions` subclass or instance.

    Args:
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Supported identifiers:

            * "openai" for `OpenAICompletions`
            * "litellm" for `LiteLLMCompletions`
            * "llama_index" for `LlamaIndexCompletions`
            * "auto" to select the first available option

    Returns:
        Completions: Resolved completions class or instance.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.
    """
    if completions is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        completions = chat_cfg["completions"]
    if isinstance(completions, str):
        if completions.lower() == "auto":
            from vectorbtpro.utils.module_ import check_installed

            if check_installed("openai"):
                completions = "openai"
            elif check_installed("litellm"):
                completions = "litellm"
            elif check_installed("llama_index"):
                completions = "llama_index"
            else:
                raise ValueError("No packages for completions installed")
        curr_module = sys.modules[__name__]
        found_completions = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Completions"):
                _short_name: tp.ClassVar[tp.Optional[str]] = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == completions.lower():
                    found_completions = cls
                    break
        if found_completions is None:
            raise ValueError(f"Invalid completions: '{completions}'")
        completions = found_completions
    if isinstance(completions, type):
        checks.assert_subclass_of(completions, Completions, arg_name="completions")
    else:
        checks.assert_instance_of(completions, Completions, arg_name="completions")
    return completions


def complete(message: str, completions: tp.CompletionsLike = None, **kwargs) -> tp.ChatOutput:
    """Get and return the chat completion for a provided message.

    Args:
        message (str): Input message for which to generate a completion.
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Resolved using `resolve_completions`.
        **kwargs: Keyword arguments to initialize or update `completions`.

    Returns:
        ChatOutput: Completion output generated by the resolved completions.
    """
    completions = resolve_completions(completions=completions)
    if isinstance(completions, type):
        completions = completions(**kwargs)
    elif kwargs:
        completions = completions.replace(**kwargs)
    return completions.get_completion(message)


def completed(message: str, completions: tp.CompletionsLike = None, **kwargs) -> str:
    """Return completion content for a given message using the provided completions configuration.

    Args:
        message (str): Input message.
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Resolved using `resolve_completions`.
        **kwargs: Keyword arguments to initialize or update `completions`.

    Returns:
        str: Completion content based on the input message.
    """
    completions = resolve_completions(completions=completions)
    if isinstance(completions, type):
        completions = completions(**kwargs)
    elif kwargs:
        completions = completions.replace(**kwargs)
    return completions.get_completion_content(message)


# ############# Splitting ############# #


class TextSplitter(Configured):
    """Abstract class for text splitters.

    Args:
        chunk_template (Optional[CustomTemplateLike]): Template used to format each text chunk.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.text_splitter_config`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the text splitter class."""

    _settings_path: tp.SettingsPath = [
        "knowledge",
        "knowledge.chat",
        "knowledge.chat.text_splitter_config",
    ]

    def __init__(
        self,
        chunk_template: tp.Optional[tp.CustomTemplateLike] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            chunk_template=chunk_template,
            template_context=template_context,
            **kwargs,
        )

        chunk_template = self.resolve_setting(chunk_template, "chunk_template")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._chunk_template = chunk_template
        self._template_context = template_context

    @property
    def chunk_template(self) -> tp.Kwargs:
        """Template used for formatting text chunks.

        Can use the following context: `chunk_idx`, `chunk_start`, `chunk_end`, `chunk_text`, and `text`.

        The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

        Returns:
            Kwargs: Context mapping used for expression evaluation.
        """
        return self._chunk_template

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    def split(self, text: str) -> tp.TSSpanChunks:
        """Yield the start and end character indices for each text chunk in the given text.

        Args:
            text (str): Input text to split.

        Yields:
            Tuple[int, int]: Tuple representing the start and end indices of a text chunk.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def split_text(self, text: str) -> tp.TSTextChunks:
        """Yield formatted text chunks generated from the input text by applying the chunk template.

        The method substitutes the chunk template with context derived from each chunk's position and text.

        Args:
            text (str): Text to split.

        Yields:
            str: Formatted text chunk.
        """
        for chunk_idx, (chunk_start, chunk_end) in enumerate(self.split(text)):
            chunk_text = text[chunk_start:chunk_end]
            chunk_template = self.chunk_template
            if isinstance(chunk_template, str):
                chunk_template = SafeSub(chunk_template)
            elif checks.is_function(chunk_template):
                chunk_template = RepFunc(chunk_template)
            elif not isinstance(chunk_template, CustomTemplate):
                raise TypeError("Chunk template must be a string, function, or template")
            template_context = flat_merge_dicts(
                dict(
                    chunk_idx=chunk_idx,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    chunk_text=chunk_text,
                    text=text,
                ),
                self.template_context,
            )
            yield chunk_template.substitute(template_context, eval_id="chunk_template")


class TokenSplitter(TextSplitter):
    """Splitter class for tokens.

    Args:
        chunk_size (Optional[int]): Maximum number of tokens per chunk; None if disabled.
        chunk_overlap (Union[None, int, float]): Number or fraction of tokens
            overlapping between consecutive chunks.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

            Resolved using `resolve_tokenizer`.
        tokenizer_kwargs (KwargsLike): Keyword arguments to initialize or update `tokenizer`.
        **kwargs: Keyword arguments for `TextSplitter`.

    !!! info
        For default settings, see `chat.text_splitter_configs.token` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "token"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.token"

    def __init__(
        self,
        chunk_size: tp.Optional[int] = None,
        chunk_overlap: tp.Union[None, int, float] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        TextSplitter.__init__(
            self,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

        chunk_size = self.resolve_setting(chunk_size, "chunk_size")
        chunk_overlap = self.resolve_setting(chunk_overlap, "chunk_overlap")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(
            tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True
        )

        tokenizer = resolve_tokenizer(tokenizer)
        if isinstance(tokenizer, type):
            tokenizer_kwargs = dict(tokenizer_kwargs)
            tokenizer_kwargs["template_context"] = merge_dicts(
                self.template_context, tokenizer_kwargs.get("template_context")
            )
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)
        if chunk_size is not None:
            if checks.is_float(chunk_overlap):
                if 0 <= abs(chunk_overlap) <= 1:
                    chunk_overlap = chunk_overlap * chunk_size
                elif not chunk_overlap.is_integer():
                    raise ValueError("Floating number for chunk_overlap must be between 0 and 1")
                chunk_overlap = int(chunk_overlap)
            if chunk_overlap >= chunk_size:
                raise ValueError("Chunk overlap must be less than the chunk size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = tokenizer

    @property
    def chunk_size(self) -> tp.Optional[int]:
        """Maximum number of tokens per chunk.

        Returns:
            int: Maximum number of tokens allowed in each chunk; None if disabled.
        """
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Number of overlapping tokens between chunks.

        If specified as a float between 0 and 1, it is scaled by `TokenSplitter.chunk_size`.

        Returns:
            int: Number of overlapping tokens between chunks.
        """
        return self._chunk_overlap

    @property
    def tokenizer(self) -> Tokenizer:
        """`Tokenizer` instance used to tokenize input text.

        Returns:
            Tokenizer: Tokenizer instance used for encoding and decoding.
        """
        return self._tokenizer

    def split_into_tokens(self, text: str) -> tp.TSSpanChunks:
        """Yield start and end indices for each token in the given text.

        The method encodes the text into tokens and decodes each token to determine its character span.

        Args:
            text (str): Text to tokenize.

        Yields:
            Tuple[int, int]: Start and end indices of each token.
        """
        tokens = self.tokenizer.encode(text)
        last_end = 0
        for token in tokens:
            _text = self.tokenizer.decode_single(token)
            start = last_end
            end = start + len(_text)
            yield start, end
            last_end = end

    def split(self, text: str) -> tp.TSSpanChunks:
        if self.chunk_size is None:
            yield from self.split_into_tokens(text)

        tokens = list(self.split_into_tokens(text))
        total_tokens = len(tokens)
        if not tokens:
            return

        token_count = 0
        while token_count < total_tokens:
            chunk_tokens = tokens[token_count : token_count + self.chunk_size]
            chunk_start = chunk_tokens[0][0]
            chunk_end = chunk_tokens[-1][1]
            yield chunk_start, chunk_end

            if token_count + self.chunk_size >= total_tokens:
                break
            token_count += self.chunk_size - self.chunk_overlap


class SegmentSplitter(TokenSplitter):
    """Splitter class for segments based on specified separators.

    This class iteratively splits text by applying nested layers of separators.
    If a segment exceeds the allowed size and no valid previous chunk exists or the token
    count falls below the minimum, the next layer of separators is used. To split into tokens,
    set a separator to None; to split into individual characters, use an empty string.

    Args:
        separators (List[List[Optional[str]]]): Nested list of separators grouped by layers used
            for splitting text.
        min_chunk_size (Union[int, float]): Minimum number of tokens required per chunk.

            If provided as a float between 0 and 1, it is interpreted relative to the chunk size.
        fixed_overlap (bool): Indicates whether fixed overlap is applied.
        **kwargs: Keyword arguments for `TokenSplitter`.

    !!! info
        For default settings, see `chat.text_splitter_configs.segment` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "segment"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.segment"

    def __init__(
        self,
        separators: tp.MaybeList[tp.MaybeList[tp.Optional[str]]] = None,
        min_chunk_size: tp.Union[None, int, float] = None,
        fixed_overlap: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        TokenSplitter.__init__(
            self,
            separators=separators,
            min_chunk_size=min_chunk_size,
            fixed_overlap=fixed_overlap,
            **kwargs,
        )

        separators = self.resolve_setting(separators, "separators")
        min_chunk_size = self.resolve_setting(min_chunk_size, "min_chunk_size")
        fixed_overlap = self.resolve_setting(fixed_overlap, "fixed_overlap")

        if not isinstance(separators, list):
            separators = [separators]
        else:
            separators = list(separators)
        for layer in range(len(separators)):
            if not isinstance(separators[layer], list):
                separators[layer] = [separators[layer]]
            else:
                separators[layer] = list(separators[layer])
        if self.chunk_size is not None:
            if checks.is_float(min_chunk_size):
                if 0 <= abs(min_chunk_size) <= 1:
                    min_chunk_size = min_chunk_size * self.chunk_size
                elif not min_chunk_size.is_integer():
                    raise ValueError("Floating number for min_chunk_size must be between 0 and 1")
                min_chunk_size = int(min_chunk_size)

        self._separators = separators
        self._min_chunk_size = min_chunk_size
        self._fixed_overlap = fixed_overlap

    @property
    def separators(self) -> tp.List[tp.List[tp.Optional[str]]]:
        """Nested list of separators grouped by layers.

        Returns:
            List[List[Optional[str]]]: (Nested) list of separators used for splitting text.
        """
        return self._separators

    @property
    def min_chunk_size(self) -> int:
        """Minimum number of tokens per chunk. If provided as a float, it is interpreted relative to
        `SegmentSplitter.chunk_size`.

        Returns:
            int: Minimum number of tokens required per chunk.
        """
        return self._min_chunk_size

    @property
    def fixed_overlap(self) -> bool:
        """Whether fixed overlap is applied.

        Returns:
            bool: True if fixed overlap is applied, False otherwise.
        """
        return self._fixed_overlap

    def split_into_segments(
        self, text: str, separator: tp.Optional[str] = None
    ) -> tp.TSSegmentChunks:
        """Split text into segments using the provided separator.

        If `separator` is None, split the text into tokens using `SegmentSplitter.split_into_tokens`.
        If `separator` is an empty string, split the text into individual characters; otherwise,
        split the text at each occurrence of `separator`.

        Args:
            text (str): Text to be split.
            separator (Optional[str]): Separator to insert between data items.

        Yields:
            Tuple[int, int, bool]: Tuple containing the segment's start index, end index, and
                a flag indicating if the segment is a separator.
        """
        if not separator:
            if separator is None:
                for start, end in self.split_into_tokens(text):
                    yield start, end, False
            else:
                for i in range(len(text)):
                    yield i, i + 1, False
        else:
            last_end = 0

            for match in re.finditer(separator, text):
                start, end = match.span()
                if start > last_end:
                    _text = text[last_end:start]
                    yield last_end, start, False

                _text = text[start:end]
                yield start, end, True
                last_end = end

            if last_end < len(text):
                _text = text[last_end:]
                yield last_end, len(text), False

    def split(self, text: str) -> tp.TSSpanChunks:
        if not text:
            yield 0, 0
            return None
        if self.chunk_size is None:
            yield 0, len(text)
            return None
        total_tokens = self.tokenizer.count_tokens(text)
        if total_tokens <= self.chunk_size:
            yield 0, len(text)
            return None

        layer = 0
        chunk_start = 0
        chunk_continue = 0
        chunk_tokens = []
        stable_token_count = 0
        stable_char_count = 0
        remaining_text = text
        overlap_segments = []
        token_offset_map = {}

        while remaining_text:
            if layer == 0:
                if chunk_continue:
                    curr_start = chunk_continue
                else:
                    curr_start = chunk_start
                curr_text = remaining_text
                curr_segments = list(overlap_segments)
                curr_tokens = list(chunk_tokens)
                curr_stable_token_count = stable_token_count
                curr_stable_char_count = stable_char_count
                sep_curr_segments = None
                sep_curr_tokens = None
                sep_curr_stable_token_count = None
                sep_curr_stable_char_count = None

            for separator in self.separators[layer]:
                segments = self.split_into_segments(curr_text, separator=separator)
                curr_text = ""
                finished = False

                for segment in segments:
                    segment_start = curr_start + segment[0]
                    segment_end = curr_start + segment[1]
                    segment_is_separator = segment[2]

                    if not curr_tokens:
                        segment_text = text[segment_start:segment_end]
                        new_curr_tokens = self.tokenizer.encode(segment_text)
                        new_curr_stable_token_count = 0
                        new_curr_stable_char_count = 0
                    elif not curr_stable_token_count:
                        chunk_text = text[chunk_start:segment_end]
                        new_curr_tokens = self.tokenizer.encode(chunk_text)
                        new_curr_stable_token_count = 0
                        new_curr_stable_char_count = 0
                        min_token_count = min(len(curr_tokens), len(new_curr_tokens))
                        for i in range(min_token_count):
                            if curr_tokens[i] == new_curr_tokens[i]:
                                new_curr_stable_token_count += 1
                                new_curr_stable_char_count += len(
                                    self.tokenizer.decode_single(curr_tokens[i])
                                )
                            else:
                                break
                    else:
                        stable_tokens = curr_tokens[:curr_stable_token_count]
                        unstable_start = chunk_start + curr_stable_char_count
                        partial_text = text[unstable_start:segment_end]
                        partial_tokens = self.tokenizer.encode(partial_text)
                        new_curr_tokens = stable_tokens + partial_tokens
                        new_curr_stable_token_count = curr_stable_token_count
                        new_curr_stable_char_count = curr_stable_char_count
                        min_token_count = min(len(curr_tokens), len(new_curr_tokens))
                        for i in range(curr_stable_token_count, min_token_count):
                            if curr_tokens[i] == new_curr_tokens[i]:
                                new_curr_stable_token_count += 1
                                new_curr_stable_char_count += len(
                                    self.tokenizer.decode_single(curr_tokens[i])
                                )
                            else:
                                break

                    if len(new_curr_tokens) > self.chunk_size:
                        if segment_is_separator:
                            if (
                                sep_curr_segments
                                and len(sep_curr_tokens) >= self.min_chunk_size
                                and not (
                                    self.chunk_overlap
                                    and len(sep_curr_tokens) <= self.chunk_overlap
                                )
                            ):
                                curr_segments = list(sep_curr_segments)
                                curr_tokens = list(sep_curr_tokens)
                                curr_stable_token_count = sep_curr_stable_token_count
                                curr_stable_char_count = sep_curr_stable_char_count
                                segment_start = curr_segments[-1][0]
                                segment_end = curr_segments[-1][1]
                        curr_text = text[segment_start:segment_end]
                        curr_start = segment_start
                        finished = False
                        break
                    else:
                        curr_segments.append((segment_start, segment_end, segment_is_separator))
                        token_offset_map[segment_start] = len(curr_tokens)
                        curr_tokens = new_curr_tokens
                        curr_stable_token_count = new_curr_stable_token_count
                        curr_stable_char_count = new_curr_stable_char_count
                        if segment_is_separator:
                            sep_curr_segments = list(curr_segments)
                            sep_curr_tokens = list(curr_tokens)
                            sep_curr_stable_token_count = curr_stable_token_count
                            sep_curr_stable_char_count = curr_stable_char_count
                        finished = True

                if finished:
                    break

            if (
                curr_segments
                and len(curr_tokens) >= self.min_chunk_size
                and not (self.chunk_overlap and len(curr_tokens) <= self.chunk_overlap)
            ):
                chunk_start = curr_segments[0][0]
                chunk_end = curr_segments[-1][1]
                yield chunk_start, chunk_end

                if chunk_end == len(text):
                    break
                if self.chunk_overlap:
                    fixed_overlap = True
                    if not self.fixed_overlap:
                        for segment in curr_segments:
                            if not segment[2]:
                                token_offset = token_offset_map[segment[0]]
                                if token_offset > curr_stable_token_count:
                                    break
                                if len(curr_tokens) - token_offset <= self.chunk_overlap:
                                    chunk_tokens = curr_tokens[token_offset:]
                                    new_chunk_start = segment[0]
                                    chunk_offset = new_chunk_start - chunk_start
                                    chunk_start = new_chunk_start
                                    chunk_continue = chunk_end
                                    fixed_overlap = False
                                    break
                    if fixed_overlap:
                        chunk_tokens = curr_tokens[-self.chunk_overlap :]
                        token_offset = len(curr_tokens) - len(chunk_tokens)
                        new_chunk_start = chunk_end - len(self.tokenizer.decode(chunk_tokens))
                        chunk_offset = new_chunk_start - chunk_start
                        chunk_start = new_chunk_start
                        chunk_continue = chunk_end
                    stable_token_count = max(0, curr_stable_token_count - token_offset)
                    stable_char_count = max(0, curr_stable_char_count - chunk_offset)
                    overlap_segments = [(chunk_start, chunk_end, False)]
                    token_offset_map[chunk_start] = 0
                else:
                    chunk_tokens = []
                    chunk_start = chunk_end
                    chunk_continue = 0
                    stable_token_count = 0
                    stable_char_count = 0
                    overlap_segments = []
                    token_offset_map = {}

                if chunk_continue:
                    remaining_text = text[chunk_continue:]
                else:
                    remaining_text = text[chunk_start:]
                layer = 0
            else:
                layer += 1
                if layer == len(self.separators):
                    if curr_segments and curr_segments[-1][1] == len(text):
                        chunk_start = curr_segments[0][0]
                        chunk_end = curr_segments[-1][1]
                        yield chunk_start, chunk_end
                        break
                    remaining_tokens = self.tokenizer.encode(remaining_text)
                    if len(remaining_tokens) > self.chunk_size:
                        raise ValueError(
                            "Total number of tokens in the last chunk is greater than the chunk size. "
                            "Increase chunk_size or the separator granularity."
                        )
                    yield curr_start, len(text)
                    break


class SourceSplitter(TokenSplitter):
    """Splitter class for source code.

    This class is used to split source code into chunks by parsing the structure of the code.
    It divides nodes of the code into levels and performs splitting based on the specified chunk size and overlap.

    Args:
        uniform_chunks (Optional[bool]): Whether each chunk should start and end at the same base level.

            If nested chunks (with level > base) are present, includes them only if they fit as a whole.
        **kwargs: Keyword arguments for `TokenSplitter`.

    !!! info
        For default settings, see `chat.text_splitter_configs.source` in `vectorbtpro._settings.knowledge`.
    """

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.source"

    def __init__(
        self,
        uniform_chunks: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        TokenSplitter.__init__(
            self,
            uniform_chunks=uniform_chunks,
            **kwargs,
        )

        uniform_chunks = self.resolve_setting(uniform_chunks, "uniform_chunks")

        self._uniform_chunks = uniform_chunks

    @property
    def uniform_chunks(self) -> bool:
        """Whether each chunk should start and end at the same base level.

        If nested chunks (with level > base) are present, includes them only if they fit as a whole.

        Returns:
            bool: True if uniform chunks are enabled, False otherwise.
        """
        return self._uniform_chunks

    def split_source(self, source: str) -> tp.TSSourceChunks:
        """Split the source code into chunks.

        Args:
            source (str): Source code to be split.

        Yields:
            Tuple[str, int]: Tuple containing the source code chunk and its base level.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def split_text(self, text: str, debug: bool = False) -> tp.TSTextChunks:
        source_nodes = list(self.split_source(text))

        if self.chunk_size is None:
            for code, _ in source_nodes:
                yield code
            return

        count_tokens = self.tokenizer.count_tokens
        max_chunk_tokens = self.chunk_size
        max_overlap_tokens = self.chunk_overlap

        total_nodes = len(source_nodes)
        current_node_index = 0
        last_overlap_start_idx = last_overlap_end_idx = None

        def _last_index_non_uniform(start_idx):
            used_tokens = 0
            idx = start_idx
            while idx < total_nodes:
                node_text = source_nodes[idx][0]
                node_tokens = count_tokens(node_text)
                if node_tokens > max_chunk_tokens:
                    return idx if idx == start_idx else idx - 1
                if used_tokens + node_tokens > max_chunk_tokens:
                    return idx - 1
                used_tokens += node_tokens
                idx += 1
            return idx - 1

        def _last_index_uniform(start_idx):
            base_level = source_nodes[start_idx][1]
            used_tokens = 0
            last_base_idx = start_idx - 1
            idx = start_idx
            while idx < total_nodes and source_nodes[idx][1] >= base_level:
                node_text, node_level = source_nodes[idx]
                node_tokens = count_tokens(node_text)
                if node_tokens > max_chunk_tokens:
                    return last_base_idx if last_base_idx >= start_idx else idx
                if used_tokens + node_tokens > max_chunk_tokens:
                    return last_base_idx
                used_tokens += node_tokens
                if node_level == base_level:
                    last_base_idx = idx
                idx += 1
                if idx == total_nodes or source_nodes[idx][1] < base_level:
                    return last_base_idx
            return idx - 1

        while current_node_index < total_nodes:
            chunk_end_idx = (
                _last_index_uniform(current_node_index)
                if self.uniform_chunks
                else _last_index_non_uniform(current_node_index)
            )

            if (
                last_overlap_start_idx is not None
                and current_node_index == last_overlap_start_idx
                and chunk_end_idx == last_overlap_end_idx
            ):
                current_node_index = chunk_end_idx + 1
                last_overlap_start_idx = last_overlap_end_idx = None
                continue

            node_slice = source_nodes[current_node_index : chunk_end_idx + 1]
            chunk_text = "".join(code for code, _ in node_slice)

            if debug:
                print("=" * 20, count_tokens(chunk_text), "=" * 20)
                for code, level in node_slice:
                    print("-" * 10, level, count_tokens(code), "-" * 10)
                    print(code, end="")

            yield chunk_text

            if max_overlap_tokens > 0:
                overlap_tokens = 0
                overlap_start_idx = chunk_end_idx
                while overlap_start_idx >= current_node_index:
                    node_tokens = count_tokens(source_nodes[overlap_start_idx][0])
                    if overlap_tokens + node_tokens > max_overlap_tokens:
                        break
                    overlap_tokens += node_tokens
                    overlap_start_idx -= 1
                overlap_start_idx += 1

                if chunk_end_idx - overlap_start_idx >= 1:
                    last_overlap_start_idx = overlap_start_idx
                    last_overlap_end_idx = chunk_end_idx
                    current_node_index = overlap_start_idx
                else:
                    last_overlap_start_idx = last_overlap_end_idx = None
                    current_node_index = chunk_end_idx + 1
            else:
                current_node_index = chunk_end_idx + 1


class PythonSplitter(SourceSplitter):
    """Splitter class for Python source code.

    This class is used to split Python source code using the `ast` module. All module-level statements
    become the zero level, which can be split into nested levels. The class supports splitting
    statements based on a whitelist and blacklist of statement types. It also allows for limiting
    the maximum statement level.

    Args:
        stmt_whitelist (Optional[Iterable[str]]): Statement types to include in the split.

            Effective only if `max_stmt_level` is met.
        stmt_blacklist (Optional[Iterable[str]]): Statement types to exclude from the split.
        max_stmt_level (Optional[int]): Maximum level of statements to include in the split.

            If None, all levels are included.
        **kwargs: Keyword arguments for `SourceSplitter`.

    !!! info
        For default settings, see `chat.text_splitter_configs.python` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "python"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.python"

    def __init__(
        self,
        stmt_whitelist: tp.Optional[tp.Iterable[str]] = None,
        stmt_blacklist: tp.Optional[tp.Iterable[str]] = None,
        max_stmt_level: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        SourceSplitter.__init__(
            self,
            stmt_whitelist=stmt_whitelist,
            stmt_blacklist=stmt_blacklist,
            max_stmt_level=max_stmt_level,
            **kwargs,
        )

        stmt_whitelist = self.resolve_setting(stmt_whitelist, "stmt_whitelist")
        stmt_blacklist = self.resolve_setting(stmt_blacklist, "stmt_blacklist")
        max_stmt_level = self.resolve_setting(max_stmt_level, "max_stmt_level")

        self._stmt_whitelist = tuple(stmt_whitelist or ())
        self._stmt_blacklist = tuple(stmt_blacklist or ())
        self._max_stmt_level = max_stmt_level

    @property
    def stmt_whitelist(self) -> tp.Tuple[str, ...]:
        """Statement types to include in the split.

        Effective only if `max_stmt_level` is met.

        Returns:
            Tuple[str, ...]: Tuple of statement types.
        """
        return self._stmt_whitelist

    @property
    def stmt_blacklist(self) -> tp.Tuple[str, ...]:
        """Statement types to exclude from the split.

        Returns:
            Tuple[str, ...]: Tuple of statement types.
        """
        return self._stmt_blacklist

    @property
    def max_stmt_level(self) -> tp.Optional[int]:
        """Maximum level of statements to include in the split.

        Returns:
            Optional[int]: Maximum statement level; None if all levels are included.
        """
        return self._max_stmt_level

    def should_split_stmt(self, stmt: ast.stmt, level: int) -> bool:
        """Check if the statement should be split based on its type and level.

        Args:
            stmt (ast.stmt): Statement to check.
            level (int): Level of the statement.

        Returns:
            bool: True if the statement should be split, False otherwise.
        """
        if self.max_stmt_level is not None and level >= self.max_stmt_level:
            return False
        if self.stmt_blacklist and checks.is_instance_of(stmt, self.stmt_blacklist):
            return False
        if self.stmt_whitelist and not checks.is_instance_of(stmt, self.stmt_whitelist):
            return False
        return True

    def split_source(self, source: str) -> tp.TSSourceChunks:
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source, type_comments=True)

        def _stmt_span(node):
            start = min(
                (d.lineno for d in getattr(node, "decorator_list", ())), default=node.lineno
            )
            end = getattr(node, "end_lineno", node.lineno)
            return start, end

        def _header_end(first_line):
            for idx in range(first_line - 1, len(lines)):
                code = lines[idx].split("#", 1)[0].rstrip()
                if code and code.endswith(":"):
                    return idx + 1
            return first_line

        def _split_block(body, start_line, end_line, level):
            body = list(body)
            if not body:
                yield (start_line, end_line, level)
                return

            cursor = start_line
            i = 0
            n = len(body)

            while i < n:
                stmt = body[i]
                stmt_start, stmt_end = _stmt_span(stmt)
                if stmt_start > cursor:
                    yield (cursor, stmt_start - 1, level)
                if (
                    isinstance(stmt, (ast.Assign, ast.AnnAssign))
                    and i + 1 < n
                    and isinstance(body[i + 1], ast.Expr)
                    and isinstance(body[i + 1].value, ast.Constant)
                    and isinstance(body[i + 1].value.value, str)
                ):
                    _, next_end = _stmt_span(body[i + 1])
                    stmt_end = next_end
                    i += 1
                if self.should_split_stmt(stmt, level) and isinstance(
                    stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    yield from _split_node(stmt, level + 1)
                else:
                    yield (stmt_start, stmt_end, level)
                cursor = stmt_end + 1
                i += 1
            if cursor <= end_line:
                yield (cursor, end_line, level)

        def _split_node(node, level):
            start, end = _stmt_span(node)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                hdr_end = node.body[0].end_lineno
                body_stmts = node.body[1:]
            else:
                hdr_end = _header_end(start)
                body_stmts = node.body

            yield (start, hdr_end, level)
            yield from _split_block(body_stmts, hdr_end + 1, end, level)

        for s, e, lvl in _split_block(tree.body, 1, len(lines), 0):
            yield ("".join(lines[s - 1 : e]), lvl)


class MarkdownSplitter(SourceSplitter):
    """Splitter class for Markdown source code.

    This class is responsible for splitting Markdown source code into chunks
    based on headers and paragraphs. It uses a custom algorithm to identify headers
    and split the content accordingly.

    Args:
        split_by (Optional[str]): Method to split the source code.

            Options are "header" or "paragraph".
        max_section_level (Optional[int]): Maximum level of sections to include in the split.

            If None, all levels are included.
        **kwargs: Keyword arguments for `SourceSplitter`.

    !!! info
        For default settings, see `chat.text_splitter_configs.markdown` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "markdown"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.markdown"

    def __init__(
        self,
        split_by: tp.Optional[str] = None,
        max_section_level: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        SourceSplitter.__init__(
            self,
            split_by=split_by,
            max_section_level=max_section_level,
            **kwargs,
        )

        split_by = self.resolve_setting(split_by, "split_by")
        max_section_level = self.resolve_setting(max_section_level, "max_section_level")

        self._split_by = split_by
        self._max_section_level = max_section_level

    @property
    def split_by(self) -> str:
        """Method to split the source code.

        Options are "header" or "paragraph".

        Returns:
            str: Method used to split the source code.
        """
        return self._split_by

    @property
    def max_section_level(self) -> tp.Optional[int]:
        """Maximum level of sections to include in the split.

        Returns:
            Optional[int]: Maximum section level; None if all levels are included.
        """
        return self._max_section_level

    def should_split_section(self, section: str, level: int) -> bool:
        """Determine whether to split the given section.

        Args:
            section: Section to evaluate.
            level: Current level of the section.

        Returns:
            bool: True if the section should be split; False otherwise.
        """
        if self.max_section_level is not None and level >= self.max_section_level:
            return False
        return True

    def split_source(self, source: str) -> tp.TSSourceChunks:
        lines = source.splitlines(True)

        chunks, buf = [], []
        level = 0
        header_pending = para_started = False
        in_code = in_html = False
        fence = html_tag = ""
        code_closed = html_closed = False
        i, n = 0, len(lines)

        def _is_header(txt):
            return txt.lstrip().startswith("#")

        while i < n:
            line = lines[i]
            stripped = line.rstrip("\n")
            lstripped = stripped.lstrip()

            if in_code:
                buf.append(line)
                if re.match(r"\s*" + re.escape(fence), lstripped):
                    in_code = False
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    code_closed = True
                i += 1
                continue

            if in_html:
                buf.append(line)
                if re.search(r"</" + html_tag + r"\s*>", lstripped, re.I):
                    in_html = False
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    html_closed = True
                i += 1
                continue

            if re.match(r"\s*<div\b", lstripped, re.I):
                if header_pending and not para_started:
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    header_pending = False
                buf.append(line)
                html_tag = "div"
                if re.search(r"</div\s*>", lstripped, re.I):
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    html_closed = True
                else:
                    in_html = True
                i += 1
                continue

            if lstripped.startswith("```") or lstripped.startswith("~~~"):
                if header_pending and not para_started:
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    header_pending = False
                buf.append(line)
                fence, in_code = lstripped[:3], True
                i += 1
                continue

            if _is_header(lstripped):
                if buf:
                    chunks.append(("".join(buf), level))
                    buf.clear()
                level = len(lstripped.split(" ")[0])
                buf.append(line)
                header_pending = True
                para_started = False
                i += 1
                continue

            if not lstripped:
                blanks = []
                while i < n and not lines[i].strip():
                    blanks.append(lines[i])
                    i += 1
                if code_closed or html_closed:
                    t, lvl = chunks[-1]
                    chunks[-1] = (t + "".join(blanks), lvl)
                    code_closed = html_closed = False
                    continue
                if i == n:
                    buf.extend(blanks)
                    break
                next_line = lines[i]
                prev_idx = i - 1
                while prev_idx >= 0 and not lines[prev_idx].strip():
                    prev_idx -= 1
                prev_line = lines[prev_idx] if prev_idx >= 0 else ""
                if (
                    len(prev_line) - len(prev_line.lstrip(" ")) >= 4
                    and len(next_line) - len(next_line.lstrip(" ")) >= 4
                ):
                    buf.extend(blanks)
                    continue
                buf.extend(blanks)
                if not (header_pending and not para_started):
                    chunks.append(("".join(buf), level))
                    buf.clear()
                    header_pending = para_started = False
                continue

            if header_pending and not para_started:
                para_started = True
            buf.append(line)
            code_closed = html_closed = False
            i += 1

        if buf:
            chunks.append(("".join(buf), level))

        split_by = self.split_by.lower()
        if split_by == "paragraph":
            final_chunks = []
            header_flag = [_is_header(c[0]) for c in chunks]
            i, m = 0, len(chunks)

            while i < m:
                text, lvl = chunks[i]
                if not header_flag[i]:
                    final_chunks.append((text, lvl))
                    i += 1
                    continue
                j = i + 1
                while j < m and not (header_flag[j] and chunks[j][1] <= lvl):
                    j += 1
                section_text = "".join(c[0] for c in chunks[i:j])
                if self.should_split_section(section_text, lvl):
                    final_chunks.append((text, lvl))
                    i += 1
                    continue
                final_chunks.append((text, lvl))
                k = i + 1
                while k < j:
                    ctext, clvl = chunks[k]
                    if header_flag[k] and clvl > lvl:
                        l = k + 1
                        while l < j and not (header_flag[l] and chunks[l][1] <= lvl):
                            l += 1
                        final_chunks.append(("".join(c[0] for c in chunks[k:l]), lvl))
                        k = l
                    else:
                        final_chunks.append((ctext, lvl))
                        k += 1
                i = j

            for chunk in final_chunks:
                yield chunk
            return

        if split_by == "header":
            sections, i, m = [], 0, len(chunks)
            while i < m:
                text, lvl = chunks[i]
                if not _is_header(text):
                    tail = text
                    i += 1
                    while i < m and not _is_header(chunks[i][0]):
                        tail += chunks[i][0]
                        i += 1
                    sections.append((tail, 0))
                    continue
                sec, header_lvl = text, lvl
                i += 1
                while i < m and not _is_header(chunks[i][0]):
                    sec += chunks[i][0]
                    i += 1
                sections.append((sec, header_lvl))

            final_chunks, i, s = [], 0, len(sections)
            while i < s:
                txt, lvl = sections[i]
                if self.should_split_section(txt, lvl):
                    final_chunks.append((txt, lvl))
                    i += 1
                    continue
                merged = txt
                i += 1
                while i < s and not (_is_header(sections[i][0]) and sections[i][1] <= lvl):
                    merged += sections[i][0]
                    i += 1
                final_chunks.append((merged, lvl))

            for chunk in final_chunks:
                yield chunk
            return

        raise ValueError(f"Invalid split_by: '{self.split_by}'")


class LlamaIndexSplitter(TextSplitter):
    """Splitter class based on a node parser from LlamaIndex that divides text into chunks using nodes.

    Args:
        node_parser (Union[None, str, NodeParser]): Node parser to use,
            specified as a string key, class, or instance.
        node_parser_kwargs (KwargsLike): Keyword arguments to node parser initialization.
        **kwargs: Keyword arguments for `TextSplitter` or used as `node_parser_kwargs`.

    !!! info
        For default settings, see `chat.text_splitter_configs.llama_index` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.llama_index"

    def __init__(
        self,
        node_parser: tp.Union[None, str, NodeParserT] = None,
        node_parser_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        TextSplitter.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.node_parser import NodeParser

        super_arg_names = set(get_func_arg_names(TextSplitter.__init__))
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_node_parser = llama_index_config.pop("node_parser", None)
        def_node_parser_kwargs = llama_index_config.pop("node_parser_kwargs", None)

        if node_parser is None:
            node_parser = def_node_parser
        init_kwargs = get_func_kwargs(type(self).__init__)
        for k in list(llama_index_config.keys()):
            if k in init_kwargs:
                llama_index_config.pop(k)

        if isinstance(node_parser, str):
            import llama_index.core.node_parser

            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, NodeParser):
                    if "." in node_parser:
                        if k.endswith(node_parser):
                            return True
                    else:
                        if k.split(".")[-1].lower() == node_parser.lower():
                            return True
                        if k.split(".")[-1].replace("Splitter", "").replace(
                            "NodeParser", ""
                        ).lower() == node_parser.lower().replace("_", ""):
                            return True
                return False

            found_node_parser = search_package(
                llama_index.core.node_parser,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_node_parser is None:
                raise ValueError(f"Node parser '{node_parser}' not found")
            node_parser = found_node_parser
        if isinstance(node_parser, type):
            checks.assert_subclass_of(node_parser, NodeParser, arg_name="node_parser")
            node_parser_name = (
                node_parser.__name__.replace("Splitter", "").replace("NodeParser", "").lower()
            )
            module_name = node_parser.__module__
        else:
            checks.assert_instance_of(node_parser, NodeParser, arg_name="node_parser")
            node_parser_name = (
                type(node_parser).__name__.replace("Splitter", "").replace("NodeParser", "").lower()
            )
            module_name = type(node_parser).__module__
        node_parser_configs = llama_index_config.pop("node_parser_configs", {})
        if node_parser_name in node_parser_configs:
            llama_index_config = merge_dicts(
                llama_index_config, node_parser_configs[node_parser_name]
            )
        elif module_name in node_parser_configs:
            llama_index_config = merge_dicts(llama_index_config, node_parser_configs[module_name])
        node_parser_kwargs = merge_dicts(
            llama_index_config, def_node_parser_kwargs, node_parser_kwargs
        )
        model_name = node_parser_kwargs.get("model_name", None)
        if model_name is None:
            func_kwargs = get_func_kwargs(type(node_parser).__init__)
            model_name = func_kwargs.get("model_name", None)
        if isinstance(node_parser, type):
            node_parser = node_parser(**node_parser_kwargs)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized node parser")

        self._model = model_name
        self._node_parser = node_parser

    @property
    def node_parser(self) -> NodeParserT:
        """LlamaIndex node parser instance used for splitting text.

        Returns:
            NodeParser: Node parser instance used for splitting text.
        """
        return self._node_parser

    def split_text(self, text: str) -> tp.TSTextChunks:
        from llama_index.core.schema import Document

        nodes = self.node_parser.get_nodes_from_documents([Document(text=text)])
        for node in nodes:
            yield node.text


def resolve_text_splitter(text_splitter: tp.TextSplitterLike = None) -> tp.MaybeType[TextSplitter]:
    """Resolve a `TextSplitter` subclass or instance.

    Args:
        text_splitter (TextSplitterLike): Identifier, subclass, or instance of `TextSplitter`.

            Supported identifiers:

            * "token" for `TokenSplitter`
            * "segment" for `SegmentSplitter`
            * "llama_index" for `LlamaIndexSplitter`

    Returns:
        TextSplitter: Resolved text splitter subclass or instance.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.
    """
    if text_splitter is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        text_splitter = chat_cfg["text_splitter"]
    if isinstance(text_splitter, str):
        curr_module = sys.modules[__name__]
        found_text_splitter = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Splitter"):
                _short_name: tp.ClassVar[tp.Optional[str]] = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == text_splitter.lower():
                    found_text_splitter = cls
                    break
        if found_text_splitter is None:
            raise ValueError(f"Invalid text splitter: '{text_splitter}'")
        text_splitter = found_text_splitter
    if isinstance(text_splitter, type):
        checks.assert_subclass_of(text_splitter, TextSplitter, arg_name="text_splitter")
    else:
        checks.assert_instance_of(text_splitter, TextSplitter, arg_name="text_splitter")
    return text_splitter


def split_text(text: str, text_splitter: tp.TextSplitterLike = None, **kwargs) -> tp.List[str]:
    """Split text into chunks using a specified text splitter.

    Args:
        text (str): Input text to be split.
        text_splitter (TextSplitterLike): Identifier, subclass, or instance of `TextSplitter`.

            Resolved using `resolve_text_splitter`.
        **kwargs: Keyword arguments to initialize or update `text_splitter`.

    Returns:
        List[str]: List of text chunks.
    """
    text_splitter = resolve_text_splitter(text_splitter=text_splitter)
    if isinstance(text_splitter, type):
        text_splitter = text_splitter(**kwargs)
    elif kwargs:
        text_splitter = text_splitter.replace(**kwargs)
    return list(text_splitter.split_text(text))


# ############# Storing ############# #


StoreObjectT = tp.TypeVar("StoreObjectT", bound="StoreObject")


@define
class StoreObject(DefineMixin):
    """Class representing an object managed by a store."""

    id_: str = define.field()
    """Object identifier."""

    @property
    def hash_key(self) -> tuple:
        return (self.id_,)


StoreDataT = tp.TypeVar("StoreDataT", bound="StoreData")


@define
class StoreData(StoreObject, DefineMixin):
    """Class for any data to be stored.

    Accepts the same arguments as in `StoreObject` + the ones listed below.
    """

    data: tp.Any = define.field()
    """Stored data."""

    @classmethod
    def id_from_data(cls, data: tp.Any) -> str:
        """Return a unique identifier computed from the given data.

        Args:
            data (Any): Data from which to generate the identifier.

        Returns:
            str: MD5 hash of the serialized data.
        """
        from vectorbtpro.utils.pickling import dumps

        return hashlib.md5(dumps(data)).hexdigest()

    @classmethod
    def from_data(
        cls: tp.Type[StoreDataT],
        data: tp.Any,
        id_: tp.Optional[str] = None,
        **kwargs,
    ) -> StoreDataT:
        """Return a new instance of `StoreData` derived from the provided data.

        Args:
            data (Any): Data to store.
            id_ (Optional[str]): Optional identifier; if not provided, one is generated.
            **kwargs: Keyword arguments for `StoreData`.

        Returns:
            StoreData: New instance of `StoreData`.
        """
        if id_ is None:
            id_ = cls.id_from_data(data)
        return cls(id_, data, **kwargs)

    def __attrs_post_init__(self):
        if self.id_ is None:
            new_id = self.id_from_data(self.data)
            object.__setattr__(self, "id_", new_id)


StoreDocumentT = tp.TypeVar("StoreDocumentT", bound="StoreDocument")


@define
class StoreDocument(StoreData, DefineMixin):
    """Abstract class for documents to be stored."""

    template_context: tp.KwargsLike = define.field(factory=dict)
    """Context for substituting template variables."""

    def get_content(self, for_embed: bool = False) -> tp.Optional[str]:
        """Return the document content.

        Returns:
            Optional[str]: Content if available, otherwise None.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def split(self: StoreDocumentT) -> tp.List[StoreDocumentT]:
        """Return a list of document instances resulting from splitting the current document.

        Returns:
            List[StoreDocument]: List of document chunks.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_content()


TextDocumentT = tp.TypeVar("TextDocumentT", bound="TextDocument")


def def_metadata_template(metadata_content: str) -> str:
    """Return a formatted metadata template string.

    Args:
        metadata_content (str): Metadata content to include.

    Returns:
        str: Formatted metadata template string with front matter delimiters.
    """
    if metadata_content.endswith("\n"):
        return f"---\n{metadata_content}---\n\n"
    return f"---\n{metadata_content}\n---\n\n"


@define
class TextDocument(StoreDocument, DefineMixin):
    """Class for text documents."""

    text_path: tp.Optional[tp.PathLikeKey] = define.field(default=None)
    """Path to the text field within the data."""

    split_text_kwargs: tp.KwargsLike = define.field(factory=dict)
    """Keyword arguments for `split_text`."""

    excl_metadata: tp.Union[bool, tp.MaybeList[tp.PathLikeKey]] = define.field(default=False)
    """Indicates whether to exclude metadata or specify fields to exclude.

    If False, metadata includes all fields except text.
    """

    excl_embed_metadata: tp.Union[None, bool, tp.MaybeList[tp.PathLikeKey]] = define.field(
        default=None
    )
    """Indicates whether to exclude metadata for embeddings; if None, defaults to `excl_metadata`."""

    skip_missing: bool = define.field(default=True)
    """Determines whether missing text or metadata returns None instead of raising an error."""

    dump_kwargs: tp.KwargsLike = define.field(factory=dict)
    """Keyword arguments for the dump formatting function."""

    metadata_template: tp.CustomTemplateLike = define.field(
        default=RepFunc(def_metadata_template, eval_id="metadata_template")
    )
    """Template for formatting metadata via the `format()` method."""

    content_template: tp.CustomTemplateLike = define.field(
        default=SafeSub("${metadata_content}${text}", eval_id="content_template")
    )
    """Template for formatting content via the `format()` method."""

    def get_text(self) -> tp.Optional[str]:
        """Return the text content of the document.

        Returns:
            Optional[str]: Document's text, or None if not available.
        """
        from vectorbtpro.utils.search_ import get_pathlike_key

        if self.data is None:
            return None
        if isinstance(self.data, str):
            return self.data
        if self.text_path is not None:
            try:
                text = get_pathlike_key(self.data, self.text_path, keep_path=False)
            except (KeyError, IndexError, AttributeError) as e:
                if not self.skip_missing:
                    raise e
                return None
            if text is None:
                return None
            if not isinstance(text, str):
                raise TypeError(f"Text field must be a string, not {type(text)}")
            return text
        raise TypeError(
            f"If text path is not provided, data item must be a string, not {type(self.data)}"
        )

    def get_metadata(self, for_embed: bool = False) -> tp.Optional[tp.Any]:
        """Return the metadata extracted from the document's data.

        Args:
            for_embed (bool): Flag indicating if metadata for embeddings should be retrieved.

        Returns:
            Optional[Any]: Metadata if available, otherwise None.
        """
        from vectorbtpro.utils.search_ import remove_pathlike_key

        if self.data is None or isinstance(self.data, str) or self.text_path is None:
            return None
        prev_keys = []
        data = self.data
        try:
            data = remove_pathlike_key(data, self.text_path, make_copy=True, prev_keys=prev_keys)
        except (KeyError, IndexError, AttributeError) as e:
            if not self.skip_missing:
                raise e
        excl_metadata = self.excl_metadata
        if for_embed:
            excl_embed_metadata = self.excl_embed_metadata
            if excl_embed_metadata is None:
                excl_embed_metadata = excl_metadata
            excl_metadata = excl_embed_metadata
        if isinstance(excl_metadata, bool):
            if excl_metadata:
                return None
            return data
        if not excl_metadata:
            return data
        if not isinstance(excl_metadata, list):
            excl_metadata = [excl_metadata]
        for p in excl_metadata:
            try:
                data = remove_pathlike_key(data, p, make_copy=True, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError):
                continue
        return data

    def get_metadata_content(self, for_embed: bool = False) -> tp.Optional[str]:
        """Return the metadata content as a formatted string.

        Args:
            for_embed (bool): Flag indicating if metadata for embeddings should be retrieved.

        Returns:
            Optional[str]: Formatted metadata content, or None if metadata is missing.
        """
        from vectorbtpro.utils.formatting import dump

        metadata = self.get_metadata(for_embed=for_embed)
        if metadata is None:
            return None
        return dump(metadata, **self.dump_kwargs)

    def get_content(self, for_embed: bool = False) -> tp.Optional[str]:
        text = self.get_text()
        metadata_content = self.get_metadata_content(for_embed=for_embed)
        if text is None and metadata_content is None:
            return None
        if text is None:
            text = ""
        if metadata_content is None:
            metadata_content = ""
        if metadata_content:
            metadata_template = self.metadata_template
            if isinstance(metadata_template, str):
                metadata_template = SafeSub(metadata_template)
            elif checks.is_function(metadata_template):
                metadata_template = RepFunc(metadata_template)
            elif not isinstance(metadata_template, CustomTemplate):
                raise TypeError("Metadata template must be a string, function, or template")
            template_context = flat_merge_dicts(
                dict(metadata_content=metadata_content),
                self.template_context,
            )
            metadata_content = metadata_template.substitute(
                template_context, eval_id="metadata_template"
            )
        content_template = self.content_template
        if isinstance(content_template, str):
            content_template = SafeSub(content_template)
        elif checks.is_function(content_template):
            content_template = RepFunc(content_template)
        elif not isinstance(content_template, CustomTemplate):
            raise TypeError("Content template must be a string, function, or template")
        template_context = flat_merge_dicts(
            dict(metadata_content=metadata_content, text=text),
            self.template_context,
        )
        return content_template.substitute(template_context, eval_id="content_template")

    def split(self: TextDocumentT) -> tp.List[TextDocumentT]:
        from vectorbtpro.utils.search_ import set_pathlike_key

        text = self.get_text()
        if text is None:
            return [self]
        text_chunks = split_text(text, **self.split_text_kwargs)
        document_chunks = []
        for text_chunk in text_chunks:
            if not isinstance(self.data, str) and self.text_path is not None:
                data_chunk = set_pathlike_key(
                    self.data,
                    self.text_path,
                    text_chunk,
                    make_copy=True,
                )
            else:
                data_chunk = text_chunk
            document_chunks.append(self.replace(data=data_chunk, id_=None))
        return document_chunks


@define
class StoreEmbedding(StoreObject, DefineMixin):
    """Class for embeddings to be stored."""

    parent_id: tp.Optional[str] = define.field(default=None)
    """Identifier of the parent object."""

    child_ids: tp.List[str] = define.field(factory=list)
    """List of identifiers for the child objects."""

    embedding: tp.Optional[tp.List[int]] = define.field(
        default=None, repr=lambda x: f"List[{len(x)}]" if x else None
    )
    """Embedding vector."""


class MetaObjectStore(type(Configured), type(MutableMapping)):
    """Metaclass for `ObjectStore`.

    Serves as the metaclass combining configuration from `Configured` and mutable mapping behavior.
    """

    pass


class ObjectStore(Configured, MutableMapping, metaclass=MetaObjectStore):
    """Class for managing an object store.

    Args:
        store_id (Optional[str]): Identifier for the store.
        purge_on_open (Optional[bool]): Indicates if the store should be purged upon opening.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.obj_store_config`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name identifier for the store class."""

    _settings_path: tp.SettingsPath = [
        "knowledge",
        "knowledge.chat",
        "knowledge.chat.obj_store_config",
    ]

    def __init__(
        self,
        store_id: tp.Optional[str] = None,
        purge_on_open: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            store_id=store_id,
            purge_on_open=purge_on_open,
            template_context=template_context,
            **kwargs,
        )

        store_id = self.resolve_setting(store_id, "store_id")
        purge_on_open = self.resolve_setting(purge_on_open, "purge_on_open")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._store_id = store_id
        self._purge_on_open = purge_on_open
        self._template_context = template_context

        self._opened = False
        self._enter_calls = 0

    @property
    def store_id(self) -> str:
        """Store identifier.

        Returns:
            str: Unique identifier of the store.
        """
        return self._store_id

    @property
    def purge_on_open(self) -> bool:
        """Flag indicating whether the store should be purged upon opening.

        Returns:
            bool: True if the store will be purged on open; otherwise, False.
        """
        return self._purge_on_open

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def opened(self) -> bool:
        """Indicates whether the store is currently open.

        Returns:
            bool: True if the store is open; otherwise, False.
        """
        return self._opened

    @property
    def enter_calls(self) -> int:
        """Number of times the store has been entered.

        Returns:
            int: Count of how many times the store's context has been entered.
        """
        return self._enter_calls

    @property
    def mirror_store_id(self) -> tp.Optional[str]:
        """Mirror store identifier.

        Returns:
            Optional[str]: Mirror store ID if applicable; otherwise, None.
        """
        return None

    def open(self) -> None:
        """Open the store.

        If already open, close it first; purge if `purge_on_open` is True.

        Returns:
            None
        """
        if self.opened:
            self.close()
        if self.purge_on_open:
            self.purge()
        self._opened = True

    def check_opened(self) -> None:
        """Ensure the store is open; raise an exception if it is not.

        Returns:
            None
        """
        if not self.opened:
            raise Exception(f"{type(self)} must be opened first")

    def commit(self) -> None:
        """Commit any pending changes to the store.

        Returns:
            None
        """
        pass

    def close(self) -> None:
        """Close the store by committing changes and marking it as closed.

        Returns:
            None
        """
        self.commit()
        self._opened = False

    def purge(self) -> None:
        """Purge the store by closing it.

        Returns:
            None
        """
        self.close()

    def __getitem__(self, id_: str) -> StoreObjectT:
        """Retrieve an object from the store using its identifier.

        Args:
            id_ (str): Identifier of the object to retrieve.

        Returns:
            StoreObject: Object associated with the given identifier.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        """Store an object in the store using its identifier.

        Args:
            id_ (str): Identifier for the object to store.
            obj (StoreObject): Object to store.

        Returns:
            None

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def __delitem__(self, id_: str) -> None:
        """Delete an object from the store using its identifier.

        Args:
            id_ (str): Identifier of the object to delete.

        Returns:
            None

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def __iter__(self) -> tp.Iterator[str]:
        """Return an iterator over the identifiers of the objects in the store.

        Returns:
            Iterator[str]: Iterator over the identifiers of the objects in the store.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of objects in the store.

        Returns:
            int: Number of objects in the store.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def __enter__(self) -> tp.Self:
        if not self.opened:
            self.open()
        self._enter_calls += 1
        return self

    def __exit__(self, *args) -> None:
        if self.enter_calls == 1:
            self.close()
            self._close_on_exit = False
        self._enter_calls -= 1
        if self.enter_calls < 0:
            self._enter_calls = 0


class DictStore(ObjectStore):
    """Store class based on a dictionary that holds objects in memory.

    Args:
        **kwargs: Keyword arguments for `ObjectStore`.

    !!! info
        For default settings, see `chat.obj_store_configs.dict` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "dict"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.dict"

    def __init__(self, **kwargs) -> None:
        ObjectStore.__init__(self, **kwargs)

        self._store = {}

    @property
    def store(self) -> tp.Dict[str, StoreObjectT]:
        """Underlying dictionary storing the objects.

        Returns:
            Dict[str, StoreObject]: Dictionary holding the objects.
        """
        return self._store

    def purge(self) -> None:
        ObjectStore.purge(self)
        self.store.clear()

    def __getitem__(self, id_: str) -> StoreObjectT:
        self.check_opened()
        return self.store[id_]

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        self.check_opened()
        self.store[id_] = obj

    def __delitem__(self, id_: str) -> None:
        self.check_opened()
        del self.store[id_]

    def __iter__(self) -> tp.Iterator[str]:
        self.check_opened()
        return iter(self.store)

    def __len__(self) -> int:
        self.check_opened()
        return len(self.store)


memory_store: tp.Dict[str, tp.Dict[str, StoreObjectT]] = {}
"""Dictionary mapping store identifiers to their corresponding object dictionaries used by `MemoryStore`."""


class MemoryStore(DictStore):
    """Store class for in-memory object storage that commits changes to `memory_store`.

    Args:
        **kwargs: Keyword arguments for `DictStore`.

    !!! info
        For default settings, see `chat.obj_store_configs.memory` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "memory"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.memory"

    def __init__(self, **kwargs) -> None:
        DictStore.__init__(self, **kwargs)

    @property
    def store(self) -> tp.Dict[str, StoreObjectT]:
        return self._store

    def store_exists(self) -> bool:
        """Return whether a store exists for the current store identifier in `memory_store`.

        Returns:
            bool: True if the store exists, otherwise False.
        """
        return self.store_id in memory_store

    def open(self) -> None:
        DictStore.open(self)
        if self.store_exists():
            self._store = dict(memory_store[self.store_id])

    def commit(self) -> None:
        DictStore.commit(self)
        memory_store[self.store_id] = dict(self.store)

    def purge(self) -> None:
        DictStore.purge(self)
        if self.store_exists():
            del memory_store[self.store_id]


class FileStore(DictStore):
    """Store class based on files.

    This class manages file-based storage. It either commits all changes to a single file
    (with the file name corresponding to the index id) or applies an initial commit to a base
    file and subsequent modifications as patch files (with the directory name serving as the index id).

    Args:
        dir_path (Optional[PathLike]): Directory path used for file storage.
        compression (CompressionLike): Compression algorithm.

            See `vectorbtpro.utils.pickling.compress`.
        save_kwargs (KwargsLike): Keyword arguments for saving objects.

            See `vectorbtpro.utils.pickling.save`.
        load_kwargs (KwargsLike): Keyword arguments for loading objects.

            See `vectorbtpro.utils.pickling.load`.
        use_patching (Optional[bool]): Whether patch files are used instead of a single file.
        consolidate (Optional[bool]): Whether patch files should be consolidated.
        **kwargs: Keyword arguments for `DictStore`.

    !!! info
        For default settings, see `chat.obj_store_configs.file` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "file"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.file"

    def __init__(
        self,
        dir_path: tp.Optional[tp.PathLike] = None,
        compression: tp.Union[None, bool, str] = None,
        save_kwargs: tp.KwargsLike = None,
        load_kwargs: tp.KwargsLike = None,
        use_patching: tp.Optional[bool] = None,
        consolidate: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        DictStore.__init__(
            self,
            dir_path=dir_path,
            compression=compression,
            save_kwargs=save_kwargs,
            load_kwargs=load_kwargs,
            use_patching=use_patching,
            consolidate=consolidate,
            **kwargs,
        )

        dir_path = self.resolve_setting(dir_path, "dir_path")
        template_context = self.template_context
        if isinstance(dir_path, CustomTemplate):
            cache_dir = self.get_setting("cache_dir", default=None)
            if cache_dir is not None:
                if isinstance(cache_dir, CustomTemplate):
                    cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
                template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
            release_dir = self.get_setting("release_dir", default=None)
            if release_dir is not None:
                if isinstance(release_dir, CustomTemplate):
                    release_dir = release_dir.substitute(template_context, eval_id="release_dir")
                template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
            dir_path = dir_path.substitute(template_context, eval_id="dir_path")
        compression = self.resolve_setting(compression, "compression")
        save_kwargs = self.resolve_setting(save_kwargs, "save_kwargs", merge=True)
        load_kwargs = self.resolve_setting(load_kwargs, "load_kwargs", merge=True)
        use_patching = self.resolve_setting(use_patching, "use_patching")
        consolidate = self.resolve_setting(consolidate, "consolidate")

        self._dir_path = dir_path
        self._compression = compression
        self._save_kwargs = save_kwargs
        self._load_kwargs = load_kwargs
        self._use_patching = use_patching
        self._consolidate = consolidate

        self._store_changes = {}
        self._new_keys = set()

    @property
    def dir_path(self) -> tp.Optional[tp.Path]:
        """Directory path used for file storage.

        Returns:
            Optional[Path]: Directory path, or None if not set.
        """
        return self._dir_path

    @property
    def compression(self) -> tp.CompressionLike:
        """Compression setting used for file operations.

        Returns:
            CompressionLike: Compression configuration used (e.g., None, True, or a specific compression type).
        """
        return self._compression

    @property
    def save_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for saving objects.

        See `vectorbtpro.utils.pickling.save`.

        Returns:
            Kwargs: Dictionary of parameters used when saving objects.
        """
        return self._save_kwargs

    @property
    def load_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for loading objects.

        See `vectorbtpro.utils.pickling.load`.

        Returns:
            Kwargs: Dictionary of parameters used when loading objects.
        """
        return self._load_kwargs

    @property
    def use_patching(self) -> bool:
        """Whether patch files are used instead of a single file.

        Returns:
            bool: True if patch files are used, otherwise False.
        """
        return self._use_patching

    @property
    def consolidate(self) -> bool:
        """Whether patch files should be consolidated.

        Returns:
            bool: True if patch consolidation is enabled, otherwise False.
        """
        return self._consolidate

    @property
    def store_changes(self) -> tp.Dict[str, StoreObjectT]:
        """Dictionary of newly added or modified objects.

        Returns:
            Dict[str, StoreObject]: Mapping of object keys to their associated updated objects.
        """
        return self._store_changes

    @property
    def new_keys(self) -> tp.Set[str]:
        """Keys representing objects not yet added to the main store.

        Returns:
            Set[str]: Set of new object keys.
        """
        return self._new_keys

    def reset_state(self) -> None:
        """Reset the internal state tracking modifications and new keys.

        This method clears any tracked changes and resets consolidation status.

        Returns:
            None
        """
        self._consolidate = False
        self._store_changes = {}
        self._new_keys = set()

    @property
    def store_path(self) -> tp.Path:
        """Filesystem path to the store.

        If patching is used, the path points to the directory containing patch files;
        otherwise, it points to a single file.

        Returns:
            Path: Complete filesystem path for the store.
        """
        dir_path = self.dir_path
        if dir_path is None:
            dir_path = "."
        dir_path = Path(dir_path)
        return dir_path / self.store_id

    @property
    def mirror_store_id(self) -> str:
        return str(self.store_path.resolve())

    def get_next_patch_path(self) -> tp.Path:
        """Return the path for the next patch file to be saved, using an incremented index.

        Returns:
            Path: Path for the next patch file.
        """
        indices = []
        for file in self.store_path.glob("patch_*"):
            indices.append(int(file.stem.split("_")[1]))
        next_index = max(indices) + 1 if indices else 0
        return self.store_path / f"patch_{next_index}"

    def open(self) -> None:
        DictStore.open(self)
        if self.store_path.exists():
            from vectorbtpro.utils.pickling import load

            if self.store_path.is_dir():
                store = {}
                store.update(
                    load(
                        path=self.store_path / "base",
                        compression=self.compression,
                        **self.load_kwargs,
                    )
                )
                patch_paths = sorted(
                    self.store_path.glob("patch_*"), key=lambda f: int(f.stem.split("_")[1])
                )
                for patch_path in patch_paths:
                    store.update(
                        load(
                            path=patch_path,
                            compression=self.compression,
                            **self.load_kwargs,
                        )
                    )
            else:
                store = load(
                    path=self.store_path,
                    compression=self.compression,
                    **self.load_kwargs,
                )
            self._store = store
        self.reset_state()

    def commit(self) -> tp.Optional[tp.Path]:
        DictStore.commit(self)
        from vectorbtpro.utils.pickling import save

        file_path = None
        if self.use_patching:
            base_path = self.store_path / "base"
            if self.consolidate:
                self.purge()
                file_path = save(
                    self.store,
                    path=base_path,
                    compression=self.compression,
                    **self.save_kwargs,
                )
            elif self.store_changes:
                if self.store_path.exists() and self.store_path.is_file():
                    self.purge()
                if not base_path.exists():
                    file_path = save(
                        self.store_changes,
                        path=base_path,
                        compression=self.compression,
                        **self.save_kwargs,
                    )
                else:
                    file_path = save(
                        self.store_changes,
                        path=self.get_next_patch_path(),
                        compression=self.compression,
                        **self.save_kwargs,
                    )
        else:
            if self.consolidate or self.store_changes:
                if self.store_path.exists() and self.store_path.is_dir():
                    self.purge()
                file_path = save(
                    self.store,
                    path=self.store_path,
                    compression=self.compression,
                    **self.save_kwargs,
                )

        self.reset_state()
        return file_path

    def close(self) -> None:
        DictStore.close(self)
        self.reset_state()

    def purge(self) -> None:
        DictStore.purge(self)
        from vectorbtpro.utils.path_ import remove_dir, remove_file

        if self.store_path.exists():
            if self.store_path.is_dir():
                remove_dir(self.store_path, with_contents=True)
            else:
                remove_file(self.store_path)
        self.reset_state()

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        if obj.id_ not in self:
            self.new_keys.add(obj.id_)
        self.store_changes[obj.id_] = obj
        DictStore.__setitem__(self, id_, obj)

    def __delitem__(self, id_: str) -> None:
        if id_ in self.new_keys:
            del self.store_changes[id_]
            self.new_keys.remove(id_)
        else:
            if id_ in self.store_changes:
                del self.store_changes[id_]
        DictStore.__delitem__(self, id_)


class LMDBStore(ObjectStore):
    """Store class based on LMDB (Lightning Memory-Mapped Database) using the `lmdbm` package.

    Args:
        dir_path (Optional[PathLike]): Directory path used for the LMDB store.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        dumps_kwargs (KwargsLike): Keyword arguments used for serializing objects.

            See `vectorbtpro.utils.pickling.dumps`.
        loads_kwargs (KwargsLike): Keyword arguments used for deserializing objects.

            See `vectorbtpro.utils.pickling.loads`.
        open_kwargs (KwargsLike): Keyword arguments used when opening the LMDB database via `Lmdb.open`.
        **kwargs: Keyword arguments for `ObjectStore`.

    !!! info
        For default settings, see `chat.obj_store_configs.lmdb` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "lmdb"

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.lmdb"

    def __init__(
        self,
        dir_path: tp.Optional[tp.PathLike] = None,
        mkdir_kwargs: tp.KwargsLike = None,
        dumps_kwargs: tp.KwargsLike = None,
        loads_kwargs: tp.KwargsLike = None,
        open_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        ObjectStore.__init__(
            self,
            dir_path=dir_path,
            mkdir_kwargs=mkdir_kwargs,
            dumps_kwargs=dumps_kwargs,
            loads_kwargs=loads_kwargs,
            open_kwargs=open_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("lmdbm")

        dir_path = self.resolve_setting(dir_path, "dir_path")
        template_context = self.template_context
        if isinstance(dir_path, CustomTemplate):
            cache_dir = self.get_setting("cache_dir", default=None)
            if cache_dir is not None:
                if isinstance(cache_dir, CustomTemplate):
                    cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
                template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
            release_dir = self.get_setting("release_dir", default=None)
            if release_dir is not None:
                if isinstance(release_dir, CustomTemplate):
                    release_dir = release_dir.substitute(template_context, eval_id="release_dir")
                template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
            dir_path = dir_path.substitute(template_context, eval_id="dir_path")
        mkdir_kwargs = self.resolve_setting(mkdir_kwargs, "mkdir_kwargs", merge=True)
        dumps_kwargs = self.resolve_setting(dumps_kwargs, "dumps_kwargs", merge=True)
        loads_kwargs = self.resolve_setting(loads_kwargs, "loads_kwargs", merge=True)
        lmdb_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        for arg_name in get_func_arg_names(ObjectStore.__init__) + get_func_arg_names(
            type(self).__init__
        ):
            if arg_name in lmdb_config:
                del lmdb_config[arg_name]
        if "mirror" in lmdb_config:
            del lmdb_config["mirror"]
        def_open_kwargs = lmdb_config.pop("open_kwargs", None)
        open_kwargs = merge_dicts(lmdb_config, def_open_kwargs, open_kwargs)

        self._dir_path = dir_path
        self._mkdir_kwargs = mkdir_kwargs
        self._dumps_kwargs = dumps_kwargs
        self._loads_kwargs = loads_kwargs
        self._open_kwargs = open_kwargs

        self._db = None

    @property
    def dir_path(self) -> tp.Optional[tp.Path]:
        """Directory path used for the LMDB store.

        Returns:
            Optional[Path]: Directory path for the LMDB store, or None if not set.
        """
        return self._dir_path

    @property
    def mkdir_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for directory creation.

        See `vectorbtpro.utils.path_.check_mkdir`.

        Returns:
            Kwargs: Dictionary of parameters for directory creation.
        """
        return self._mkdir_kwargs

    @property
    def dumps_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for serializing objects.

        See `vectorbtpro.utils.pickling.dumps`.

        Returns:
            Kwargs: Dictionary of parameters for object serialization.
        """
        return self._dumps_kwargs

    @property
    def loads_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used for deserializing objects.

        See `vectorbtpro.utils.pickling.loads`.

        Returns:
            Kwargs: Dictionary of parameters for object deserialization.
        """
        return self._loads_kwargs

    @property
    def open_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used when opening the LMDB database via `Lmdb.open`.

        Returns:
            Kwargs: Dictionary of parameters for opening the LMDB database.
        """
        return self._open_kwargs

    @property
    def db_path(self) -> tp.Path:
        """File system path to the LMDB database.

        Constructs the path by combining the directory (defaulting to "." if not set) with the store identifier.

        Returns:
            Path: Complete file system path pointing to the LMDB database.
        """
        dir_path = self.dir_path
        if dir_path is None:
            dir_path = "."
        dir_path = Path(dir_path)
        return dir_path / self.store_id

    @property
    def mirror_store_id(self) -> str:
        return str(self.db_path.resolve())

    @property
    def db(self) -> tp.Optional[LmdbT]:
        """LMDB database instance.

        Returns:
            Optional[Lmdb]: LMDB database instance if the store is open; otherwise, None.
        """
        return self._db

    def open(self) -> None:
        ObjectStore.open(self)
        from lmdbm import Lmdb

        from vectorbtpro.utils.path_ import check_mkdir

        check_mkdir(self.db_path.parent, **self.mkdir_kwargs)
        self._db = Lmdb.open(str(self.db_path.resolve()), **self.open_kwargs)

    def close(self) -> None:
        ObjectStore.close(self)
        if self.db:
            self.db.close()
        self._db = None

    def purge(self) -> None:
        ObjectStore.purge(self)
        from vectorbtpro.utils.path_ import remove_dir

        remove_dir(self.db_path, missing_ok=True, with_contents=True)

    def encode(self, obj: StoreObjectT) -> bytes:
        """Encode the given object to a bytes representation using the configured serialization settings.

        Args:
            obj (StoreObject): Object to encode.

        Returns:
            bytes: Serialized bytes of the object.
        """
        from vectorbtpro.utils.pickling import dumps

        return dumps(obj, **self.dumps_kwargs)

    def decode(self, bytes_: bytes) -> StoreObjectT:
        """Decode the given bytes into an object using the configured deserialization settings.

        Args:
            bytes_ (bytes): Byte stream containing the serialized object.

        Returns:
            StoreObject: Deserialized object.
        """
        from vectorbtpro.utils.pickling import loads

        return loads(bytes_, **self.loads_kwargs)

    def __getitem__(self, id_: str) -> StoreObjectT:
        self.check_opened()
        return self.decode(self.db[id_])

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        self.check_opened()
        self.db[id_] = self.encode(obj)

    def __delitem__(self, id_: str) -> None:
        self.check_opened()
        del self.db[id_]

    def __iter__(self) -> tp.Iterator[str]:
        self.check_opened()
        return iter(self.db)

    def __len__(self) -> int:
        self.check_opened()
        return len(self.db)


class CachedStore(DictStore):
    """Store class acting as a temporary cache for another store.

    Args:
        obj_store (ObjectStore): Underlying object store to cache.
        lazy_open (Optional[bool]): Flag indicating whether to open the store lazily.
        mirror (Optional[bool]): Flag indicating whether to mirror the store in `memory_store`.
        **kwargs: Keyword arguments for `DictStore`.

    !!! info
        For default settings, see `chat.obj_store_configs.cached` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "cached"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.cached"

    def __init__(
        self,
        obj_store: ObjectStore,
        lazy_open: tp.Optional[bool] = None,
        mirror: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        DictStore.__init__(
            self,
            obj_store=obj_store,
            lazy_open=lazy_open,
            mirror=mirror,
            **kwargs,
        )

        lazy_open = self.resolve_setting(lazy_open, "lazy_open")
        mirror = obj_store.resolve_setting(mirror, "mirror", default=None)
        mirror = self.resolve_setting(mirror, "mirror")
        if mirror and obj_store.mirror_store_id is None:
            mirror = False

        self._obj_store = obj_store
        self._lazy_open = lazy_open
        self._mirror = mirror

        self._force_open = False

    @property
    def obj_store(self) -> ObjectStore:
        """Underlying object store.

        Returns:
            ObjectStore: Object store instance being cached.
        """
        return self._obj_store

    @property
    def lazy_open(self) -> bool:
        """Whether the store opens lazily.

        Returns:
            bool: True if the store opens lazily; otherwise, False.
        """
        return self._lazy_open

    @property
    def mirror(self) -> bool:
        """Whether the store is mirrored in `memory_store`.

        Returns:
            bool: True if the store is mirrored; otherwise, False.
        """
        return self._mirror

    @property
    def force_open(self) -> bool:
        """Whether the store is forced open.

        Returns:
            bool: True if the store is forced open; otherwise, False.
        """
        return self._force_open

    def open(self) -> None:
        DictStore.open(self)
        if self.mirror and self.obj_store.mirror_store_id in memory_store:
            self.store.update(memory_store[self.obj_store.mirror_store_id])
        elif not self.lazy_open or self.force_open:
            self.obj_store.open()

    def check_opened(self) -> None:
        if self.lazy_open and not self.obj_store.opened:
            self._force_open = True
            self.obj_store.open()
        DictStore.check_opened(self)

    def commit(self) -> None:
        DictStore.commit(self)
        self.check_opened()
        self.obj_store.commit()
        if self.mirror:
            memory_store[self.obj_store.mirror_store_id] = dict(self.store)

    def close(self) -> None:
        DictStore.close(self)
        self.obj_store.close()
        self._force_open = False

    def purge(self) -> None:
        DictStore.purge(self)
        self.obj_store.purge()
        if self.mirror and self.obj_store.mirror_store_id in memory_store:
            del memory_store[self.obj_store.mirror_store_id]

    def __getitem__(self, id_: str) -> StoreObjectT:
        if id_ in self.store:
            return self.store[id_]
        self.check_opened()
        obj = self.obj_store[id_]
        self.store[id_] = obj
        return obj

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        self.check_opened()
        self.store[id_] = obj
        self.obj_store[id_] = obj

    def __delitem__(self, id_: str) -> None:
        self.check_opened()
        if id_ in self.store:
            del self.store[id_]
        del self.obj_store[id_]

    def __iter__(self) -> tp.Iterator[str]:
        self.check_opened()
        return iter(self.obj_store)

    def __len__(self) -> int:
        self.check_opened()
        return len(self.obj_store)


def resolve_obj_store(obj_store: tp.ObjectStoreLike = None) -> tp.MaybeType[ObjectStore]:
    """Resolve a subclass or an instance of `ObjectStore`.

    Args:
        obj_store (ObjectStoreLike): Identifier, subclass, or instance of `ObjectStore`.

            Supported identifiers:

            * "dict" for `DictStore`
            * "memory" for `MemoryStore`
            * "file" for `FileStore`
            * "lmdb" for `LMDBStore`
            * "cached" for `CachedStore`

    Returns:
        ObjectStore: Resolved object store.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.
    """
    if obj_store is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        obj_store = chat_cfg["obj_store"]
    if isinstance(obj_store, str):
        curr_module = sys.modules[__name__]
        found_obj_store = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Store"):
                _short_name: tp.ClassVar[tp.Optional[str]] = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == obj_store.lower():
                    found_obj_store = cls
                    break
        if found_obj_store is None:
            raise ValueError(f"Invalid object store: '{obj_store}'")
        obj_store = found_obj_store
    if isinstance(obj_store, type):
        checks.assert_subclass_of(obj_store, ObjectStore, arg_name="obj_store")
    else:
        checks.assert_instance_of(obj_store, ObjectStore, arg_name="obj_store")
    return obj_store


# ############# Ranking ############# #


@define
class EmbeddedDocument(DefineMixin):
    """Define an abstract class for embedded documents."""

    document: StoreDocument = define.field()
    """Primary document content."""

    embedding: tp.Optional[tp.List[float]] = define.field(default=None)
    """List of floats representing the document's embedding."""

    child_documents: tp.List["EmbeddedDocument"] = define.field(factory=list)
    """List of embedded child documents."""


@define
class ScoredDocument(DefineMixin):
    """Define an abstract class for scored documents with an associated numerical score."""

    document: StoreDocument = define.field()
    """Primary document content."""

    score: float = define.field(default=float("nan"))
    """Numeric score assigned to the document."""

    child_documents: tp.List["ScoredDocument"] = define.field(factory=list)
    """List of scored child documents."""


class FallbackError(Exception):
    """Exception raised when a fallback is triggered."""

    pass


class DocumentRanker(Configured):
    """Class for embedding, scoring, and ranking documents.

    Args:
        dataset_id (Optional[str]): Identifier for the dataset.
        embeddings (EmbeddingsLike): Identifier, subclass, or instance of `Embeddings`.

            Resolved using `resolve_embeddings`.
        embeddings_kwargs (KwargsLike): Keyword arguments to initialize or update `embeddings`.
        doc_store (ObjectStoreLike): Identifier, subclass, or instance of `ObjectStore` for documents.

            Resolved using `resolve_obj_store`.
        doc_store_kwargs (KwargsLike): Keyword arguments to initialize or update `doc_store`.
        cache_doc_store (Optional[bool]): Flag to indicate if `doc_store` should be cached.
        emb_store (ObjectStoreLike): Identifier, subclass, or instance of `ObjectStore` for embeddings.

            Resolved using `resolve_obj_store`.
        emb_store_kwargs (KwargsLike): Keyword arguments to initialize or update `emb_store`.
        cache_emb_store (Optional[bool]): Flag to indicate if `emb_store` should be cached.
        search_method (Optional[str]): Strategy for document search.

            Supported strategies:

            * "bm25": Use BM25 for document search.
            * "embeddings": Use embeddings for document search.
                Embeds documents that don't have embeddings, which can be time-consuming.
            * "hybrid": Use a combination of embeddings and BM25 for document search.
                Embeds documents that don't have embeddings, which can be time-consuming.
            * "embeddings_fallback": Use "embeddings" if all documents have embeddings, otherwise use "bm25".
            * "hybrid_fallback": Use "hybrid" if all documents have embeddings, otherwise use "bm25".
        bm25_tokenizer (Optional[BM25Tokenizer]): BM25 tokenizer instance or type for processing text.

            Resolved using `DocumentRanker.resolve_bm25_tokenizer`.
        bm25_tokenizer_kwargs (KwargsLike): Keyword arguments to initialize `bm25_tokenizer`.
        bm25_retriever (Optional[MaybeType[BM25]]): BM25 retriever instance or type for document retrieval.

            Resolved using `DocumentRanker.resolve_bm25_retriever`.
        bm25_retriever_kwargs (KwargsLike): Keyword arguments to initialize `bm25_retriever`.
        bm25_mirror_store_id (Optional[str]): Identifier for the BM25 mirror store.
        rrf_k (Optional[int]): K parameter for RRF (Reciprocal Rank Fusion).
        rrf_bm25_weight (Optional[float]): BM25 weight for RRF (Reciprocal Rank Fusion).

            The embedding weight is computed as 1 minus this value.
        score_func (Union[None, str, Callable]): Function or identifier for scoring documents.

            See `DocumentRanker.compute_score`.
        score_agg_func (Union[None, str, Callable]): Function or identifier for aggregating scores.
        normalize_scores (Optional[bool]): Whether scores should be normalized before filtering.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

            See `vectorbtpro.utils.pbar.ProgressBar`.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.doc_ranker_config`.
    """

    _settings_path: tp.SettingsPath = [
        "knowledge",
        "knowledge.chat",
        "knowledge.chat.doc_ranker_config",
    ]

    def __init__(
        self,
        dataset_id: tp.Optional[str] = None,
        embeddings: tp.EmbeddingsLike = None,
        embeddings_kwargs: tp.KwargsLike = None,
        doc_store: tp.ObjectStoreLike = None,
        doc_store_kwargs: tp.KwargsLike = None,
        cache_doc_store: tp.Optional[bool] = None,
        emb_store: tp.ObjectStoreLike = None,
        emb_store_kwargs: tp.KwargsLike = None,
        cache_emb_store: tp.Optional[bool] = None,
        search_method: tp.Optional[str] = None,
        bm25_tokenizer: tp.Optional[tp.MaybeType[BM25TokenizerT]] = None,
        bm25_tokenizer_kwargs: tp.KwargsLike = None,
        bm25_retriever: tp.Optional[tp.MaybeType[BM25T]] = None,
        bm25_retriever_kwargs: tp.KwargsLike = None,
        bm25_mirror_store_id: tp.Optional[str] = None,
        rrf_k: tp.Optional[int] = None,
        rrf_bm25_weight: tp.Optional[float] = None,
        score_func: tp.Union[None, str, tp.Callable] = None,
        score_agg_func: tp.Union[None, str, tp.Callable] = None,
        normalize_scores: tp.Optional[bool] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            dataset_id=dataset_id,
            embeddings=embeddings,
            embeddings_kwargs=embeddings_kwargs,
            doc_store=doc_store,
            doc_store_kwargs=doc_store_kwargs,
            cache_doc_store=cache_doc_store,
            emb_store=emb_store,
            emb_store_kwargs=emb_store_kwargs,
            cache_emb_store=cache_emb_store,
            search_method=search_method,
            bm25_tokenizer=bm25_tokenizer,
            bm25_tokenizer_kwargs=bm25_tokenizer_kwargs,
            bm25_retriever=bm25_retriever,
            bm25_retriever_kwargs=bm25_retriever_kwargs,
            bm25_mirror_store_id=bm25_mirror_store_id,
            rrf_k=rrf_k,
            rrf_bm25_weight=rrf_bm25_weight,
            score_func=score_func,
            score_agg_func=score_agg_func,
            normalize_scores=normalize_scores,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

        dataset_id = self.resolve_setting(dataset_id, "dataset_id")
        embeddings = self.resolve_setting(embeddings, "embeddings", default=None)
        embeddings_kwargs = self.resolve_setting(
            embeddings_kwargs, "embeddings_kwargs", default=None, merge=True
        )
        doc_store = self.resolve_setting(doc_store, "doc_store", default=None)
        doc_store_kwargs = self.resolve_setting(
            doc_store_kwargs, "doc_store_kwargs", default=None, merge=True
        )
        cache_doc_store = self.resolve_setting(cache_doc_store, "cache_doc_store")
        emb_store = self.resolve_setting(emb_store, "emb_store", default=None)
        emb_store_kwargs = self.resolve_setting(
            emb_store_kwargs, "emb_store_kwargs", default=None, merge=True
        )
        cache_emb_store = self.resolve_setting(cache_emb_store, "cache_emb_store")
        search_method = self.resolve_setting(search_method, "search_method")
        bm25_mirror_store_id = self.resolve_setting(bm25_mirror_store_id, "bm25_mirror_store_id")
        rrf_k = self.resolve_setting(rrf_k, "rrf_k")
        rrf_bm25_weight = self.resolve_setting(rrf_bm25_weight, "rrf_bm25_weight")
        score_func = self.resolve_setting(score_func, "score_func")
        score_agg_func = self.resolve_setting(score_agg_func, "score_agg_func")
        normalize_scores = self.resolve_setting(normalize_scores, "normalize_scores")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        obj_store = self.get_setting("obj_store", default=None)
        obj_store_kwargs = self.get_setting("obj_store_kwargs", default=None, merge=True)
        if doc_store is None:
            doc_store = obj_store
        doc_store_kwargs = merge_dicts(obj_store_kwargs, doc_store_kwargs)
        if emb_store is None:
            emb_store = obj_store
        emb_store_kwargs = merge_dicts(obj_store_kwargs, emb_store_kwargs)

        search_method = search_method.lower()
        checks.assert_in(
            search_method,
            ("bm25", "embeddings", "hybrid", "embeddings_fallback", "hybrid_fallback"),
            arg_name="search_method",
        )
        if search_method in ("embeddings", "hybrid", "embeddings_fallback", "hybrid_fallback"):
            try:
                embeddings = resolve_embeddings(embeddings)
                if isinstance(embeddings, type):
                    embeddings_kwargs = dict(embeddings_kwargs)
                    embeddings_kwargs["template_context"] = merge_dicts(
                        template_context, embeddings_kwargs.get("template_context")
                    )
                    embeddings = embeddings(**embeddings_kwargs)
                elif embeddings_kwargs:
                    embeddings = embeddings.replace(**embeddings_kwargs)
            except Exception as e:
                if search_method in ("embeddings_fallback", "hybrid_fallback"):
                    warn(f'Failed to resolve embeddings: "{e}"')
                    embeddings = None
                else:
                    raise e
        else:
            embeddings = None

        if isinstance(self._settings_path, list):
            if not isinstance(self._settings_path[-1], str):
                raise TypeError(
                    "_settings_path[-1] for DocumentRanker and its subclasses must be a string"
                )
            target_settings_path = self._settings_path[-1]
        elif isinstance(self._settings_path, str):
            target_settings_path = self._settings_path
        else:
            raise TypeError(
                "_settings_path for DocumentRanker and its subclasses must be a list or string"
            )

        doc_store = resolve_obj_store(doc_store)
        if not isinstance(doc_store._settings_path, str):
            raise TypeError("_settings_path for ObjectStore and its subclasses must be a string")
        doc_store_cls = doc_store if isinstance(doc_store, type) else type(doc_store)
        doc_store_settings_path = doc_store._settings_path
        doc_store_settings_path = doc_store_settings_path.replace(
            "knowledge.chat", target_settings_path
        )
        doc_store_settings_path = doc_store_settings_path.replace("obj_store", "doc_store")
        with ExtSettingsPath([(doc_store_cls, doc_store_settings_path)]):
            if isinstance(doc_store, type):
                doc_store_kwargs = dict(doc_store_kwargs)
                if dataset_id is not None and "store_id" not in doc_store_kwargs:
                    doc_store_kwargs["store_id"] = dataset_id
                doc_store_kwargs["template_context"] = merge_dicts(
                    template_context, doc_store_kwargs.get("template_context")
                )
                doc_store = doc_store(**doc_store_kwargs)
            elif doc_store_kwargs:
                doc_store = doc_store.replace(**doc_store_kwargs)
        if cache_doc_store and not isinstance(doc_store, CachedStore):
            doc_store = CachedStore(doc_store)

        emb_store = resolve_obj_store(emb_store)
        if not isinstance(emb_store._settings_path, str):
            raise TypeError("_settings_path for ObjectStore and its subclasses must be a string")
        emb_store_cls = emb_store if isinstance(emb_store, type) else type(emb_store)
        emb_store_settings_path = emb_store._settings_path
        emb_store_settings_path = emb_store_settings_path.replace(
            "knowledge.chat", target_settings_path
        )
        emb_store_settings_path = emb_store_settings_path.replace("obj_store", "emb_store")
        with ExtSettingsPath([(emb_store_cls, emb_store_settings_path)]):
            if isinstance(emb_store, type):
                emb_store_kwargs = dict(emb_store_kwargs)
                if dataset_id is not None and "store_id" not in emb_store_kwargs:
                    emb_store_kwargs["store_id"] = dataset_id
                emb_store_kwargs["template_context"] = merge_dicts(
                    template_context, emb_store_kwargs.get("template_context")
                )
                emb_store = emb_store(**emb_store_kwargs)
            elif emb_store_kwargs:
                emb_store = emb_store.replace(**emb_store_kwargs)
        if cache_emb_store and not isinstance(emb_store, CachedStore):
            emb_store = CachedStore(emb_store)

        if search_method in ("bm25", "hybrid", "embeddings_fallback", "hybrid_fallback"):
            if bm25_tokenizer_kwargs is None:
                bm25_tokenizer_kwargs = {}
            if bm25_retriever_kwargs is None:
                bm25_retriever_kwargs = {}
            if bm25_mirror_store_id is not None:
                with MemoryStore(store_id=bm25_mirror_store_id) as bm25_memory_store:
                    if bm25_memory_store.store_exists():
                        bm25_tokenizer = bm25_memory_store["bm25_tokenizer"].data
                        bm25_retriever = bm25_memory_store["bm25_retriever"].data
            bm25_tokenizer, bm25_tokenize_kwargs = self.resolve_bm25_tokenizer(
                bm25_tokenizer=bm25_tokenizer, **bm25_tokenizer_kwargs
            )
            bm25_retriever, bm25_retrieve_kwargs = self.resolve_bm25_retriever(
                bm25_retriever=bm25_retriever, **bm25_retriever_kwargs
            )
            if bm25_mirror_store_id is not None:
                with MemoryStore(store_id=bm25_mirror_store_id) as bm25_memory_store:
                    bm25_memory_store["bm25_tokenizer"] = StoreData(
                        "bm25_tokenizer", bm25_tokenizer
                    )
                    bm25_memory_store["bm25_retriever"] = StoreData(
                        "bm25_retriever", bm25_retriever
                    )
        else:
            bm25_tokenizer = None
            bm25_tokenize_kwargs = {}
            bm25_retriever = None
            bm25_retrieve_kwargs = {}

        if isinstance(score_agg_func, str):
            score_agg_func = getattr(np, score_agg_func)

        self._embeddings = embeddings
        self._doc_store = doc_store
        self._emb_store = emb_store
        self._search_method = search_method
        self._bm25_tokenizer = bm25_tokenizer
        self._bm25_tokenize_kwargs = bm25_tokenize_kwargs
        self._bm25_retriever = bm25_retriever
        self._bm25_retrieve_kwargs = bm25_retrieve_kwargs
        self._rrf_k = rrf_k
        self._rrf_bm25_weight = rrf_bm25_weight
        self._score_func = score_func
        self._score_agg_func = score_agg_func
        self._normalize_scores = normalize_scores
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

    @property
    def embeddings(self) -> tp.Optional[Embeddings]:
        """Instance of `Embeddings`.

        Returns:
            Embeddings: Embeddings engine or class used for processing document embeddings; None if not set.
        """
        return self._embeddings

    @property
    def doc_store(self) -> ObjectStore:
        """Instance of `ObjectStore` used for documents.

        Returns:
            ObjectStore: Document store instance used for managing documents.
        """
        return self._doc_store

    @property
    def emb_store(self) -> ObjectStore:
        """Instance of `ObjectStore` used for embeddings.

        Returns:
            ObjectStore: Embedding store instance used for managing embeddings.
        """
        return self._emb_store

    @property
    def search_method(self) -> str:
        """Strategy for document search.

        Supported strategies:

        * "bm25": Use BM25 for document search.
        * "embeddings": Use embeddings for document search.
            Embeds documents that don't have embeddings, which can be time-consuming.
        * "hybrid": Use a combination of embeddings and BM25 for document search.
            Embeds documents that don't have embeddings, which can be time-consuming.
        * "embeddings_fallback": Use "embeddings" if all documents have embeddings, otherwise use "bm25".
        * "hybrid_fallback": Use "hybrid" if all documents have embeddings, otherwise use "bm25".

        Returns:
            str: Search method used for document retrieval.
        """
        return self._search_method

    @property
    def bm25_tokenizer(self) -> tp.Optional[BM25TokenizerT]:
        """BM25 tokenizer instance from `bm25s.tokenization.Tokenizer`.

        Returns:
            Optional[BM25Tokenizer]: BM25 tokenizer instance used for processing text; None if not set.
        """
        return self._bm25_tokenizer

    @property
    def bm25_tokenize_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for the `tokenize` method of `bm25s.tokenization.Tokenizer`.

        Returns:
            Kwargs: Dictionary of parameters for the tokenization process.
        """
        return self._bm25_tokenize_kwargs

    @property
    def bm25_retriever(self) -> tp.Optional[BM25T]:
        """BM25 retriever instance from `bm25s.BM25`.

        Returns:
            Optional[BM25]: BM25 retriever instance used for document retrieval; None if not set.
        """
        return self._bm25_retriever

    @property
    def bm25_retrieve_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for the `retrieve` method of `bm25s.BM25`.

        Returns:
            Kwargs: Dictionary of parameters for the retrieval process.
        """
        return self._bm25_retrieve_kwargs

    @property
    def rrf_k(self) -> int:
        """K parameter for RRF (Reciprocal Rank Fusion).

        Returns:
            int: K parameter used in RRF.
        """
        return self._rrf_k

    @property
    def rrf_bm25_weight(self) -> float:
        """BM25 weight for RRF (Reciprocal Rank Fusion).

        The embedding weight is computed as 1 minus this value.

        Returns:
            float: BM25 weight used in RRF.
        """
        return self._rrf_bm25_weight

    @property
    def score_func(self) -> tp.Union[str, tp.Callable]:
        """Score function or its name used for computing document scores.

        See `DocumentRanker.compute_score`.

        Returns:
            Union[str, Callable]: Score function used for computing document scores.
        """
        return self._score_func

    @property
    def score_agg_func(self) -> tp.Callable:
        """Function used to aggregate scores.

        Returns:
            Callable: Function used for aggregating scores.
        """
        return self._score_agg_func

    @property
    def normalize_scores(self) -> bool:
        """Whether scores should be normalized before filtering.

        Returns:
            bool: True if scores should be normalized; otherwise, False.
        """
        return self._normalize_scores

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to display a progress bar.

        Returns:
            Optional[bool]: True if a progress bar should be shown; otherwise, False.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for configuring the progress bar.

        See `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Dictionary of parameters for the progress bar.
        """
        return self._pbar_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    def resolve_bm25_tokenizer(
        cls,
        bm25_tokenizer: tp.Optional[tp.MaybeType[BM25TokenizerT]] = None,
        **kwargs,
    ) -> tp.Tuple[BM25TokenizerT, tp.Kwargs]:
        """Return a tuple containing a resolved instance of `bm25s.tokenization.Tokenizer` and
        tokenization keyword arguments.

        Args:
            bm25_tokenizer (Optional[BM25Tokenizer]): BM25 tokenizer instance or type.
            **kwargs: Keyword arguments for initializing `bm25_tokenizer` and tokenization.

        Returns:
            Tuple[BM25TokenizerT, Kwargs]: Resolved BM25 tokenizer and the tokenization keyword arguments.
        """
        from vectorbtpro.utils.module_ import assert_can_import, check_installed

        assert_can_import("bm25s")

        from bm25s.tokenization import Tokenizer

        bm25_tokenizer = cls.resolve_setting(bm25_tokenizer, "bm25_tokenizer")
        kwargs = cls.resolve_setting(kwargs, "bm25_tokenizer_kwargs", merge=True)

        if bm25_tokenizer is None:
            bm25_tokenizer = Tokenizer
        if isinstance(bm25_tokenizer, type):
            checks.assert_subclass_of(bm25_tokenizer, Tokenizer, arg_name="bm25_tokenizer")
            bm25_tokenizer_type = bm25_tokenizer
        else:
            checks.assert_instance_of(bm25_tokenizer, Tokenizer, arg_name="bm25_tokenizer")
            bm25_tokenizer_type = type(bm25_tokenizer)
        bm25_tokenize_kwargs = {}
        if kwargs:
            bm25_tokenize_arg_names = get_func_arg_names(bm25_tokenizer_type.tokenize)
            for k in bm25_tokenize_arg_names:
                if k in kwargs:
                    bm25_tokenize_kwargs[k] = kwargs.pop(k)
        if isinstance(bm25_tokenizer, type):
            if "splitter" not in kwargs:
                kwargs["splitter"] = cls.bm25_splitter
                if "lower" not in kwargs:
                    kwargs["lower"] = False
            if "stemmer" not in kwargs and check_installed("Stemmer"):
                import Stemmer

                kwargs["stemmer"] = Stemmer.Stemmer("english")
            bm25_tokenizer = bm25_tokenizer(**kwargs)
        return bm25_tokenizer, bm25_tokenize_kwargs

    def resolve_bm25_retriever(
        cls,
        bm25_retriever: tp.Optional[tp.MaybeType[BM25T]] = None,
        **kwargs,
    ) -> tp.Tuple[BM25T, tp.Kwargs]:
        """Return a tuple containing a resolved instance of `bm25s.BM25` and retrieval keyword arguments.

        Args:
            bm25_retriever (Optional[BM25]): BM25 retriever instance or type.
            **kwargs: Keyword arguments for initializing `bm25_retriever` and retrieval.

        Returns:
            Tuple[BM25T, Kwargs]: Resolved BM25 retriever and the retrieval keyword arguments.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("bm25s")

        from bm25s import BM25

        bm25_retriever = cls.resolve_setting(bm25_retriever, "bm25_retriever")
        kwargs = cls.resolve_setting(kwargs, "bm25_retriever_kwargs", merge=True)

        if bm25_retriever is None:
            bm25_retriever = BM25
        if isinstance(bm25_retriever, type):
            checks.assert_subclass_of(bm25_retriever, BM25, arg_name="bm25_retriever")
            bm25_retriever_type = bm25_retriever
        else:
            checks.assert_instance_of(bm25_retriever, BM25, arg_name="bm25_retriever")
            bm25_retriever_type = type(bm25_retriever)
        bm25_retrieve_kwargs = {}
        if kwargs:
            bm25_retrieve_arg_names = get_func_arg_names(bm25_retriever_type.retrieve)
            for k in bm25_retrieve_arg_names:
                if k in kwargs:
                    bm25_retrieve_kwargs[k] = kwargs.pop(k)
        if isinstance(bm25_retriever, type):
            bm25_retriever = bm25_retriever(**kwargs)
        return bm25_retriever, bm25_retrieve_kwargs

    def embed_documents(
        self,
        documents: tp.Iterable[StoreDocument],
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_embeddings: bool = False,
        return_documents: bool = False,
        with_fallback: bool = False,
    ) -> tp.Optional[tp.EmbeddedDocuments]:
        """Embed documents by optionally refreshing stored documents and embeddings.

        Without refreshing, persisted objects from the respective stores are used.

        Args:
            documents (Iterable[StoreDocument]): Collection of documents to embed.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_embeddings (bool): Flag indicating whether to return embeddings.
            return_documents (bool): If True, include original document objects in the output.
            with_fallback (bool): If True, raise `FallbackError` if new embeddings are needed.

        Returns:
            Optional[EmbeddedDocuments]: Embedded documents or embeddings based on the specified return flags.

                Returns None if both return flags are False.
        """
        if refresh_documents is None:
            refresh_documents = refresh
        if refresh_embeddings is None:
            refresh_embeddings = refresh
        with self.doc_store, self.emb_store:
            documents = list(documents)
            documents_to_split = []
            document_splits = {}
            for document in documents:
                refresh_document = (
                    refresh_documents
                    or refresh_embeddings
                    or document.id_ not in self.doc_store
                    or document.id_ not in self.emb_store
                )
                if not refresh_document:
                    obj = self.emb_store[document.id_]
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            if child_id not in self.doc_store or child_id not in self.emb_store:
                                refresh_document = True
                                break
                if refresh_document:
                    if with_fallback:
                        raise FallbackError("Some documents need to be refreshed")
                    documents_to_split.append(document)
            if documents_to_split:
                from vectorbtpro.utils.pbar import ProgressBar

                pbar_kwargs = merge_dicts(dict(prefix="split_documents"), self.pbar_kwargs)
                with ProgressBar(
                    total=len(documents_to_split),
                    show_progress=self.show_progress,
                    **pbar_kwargs,
                ) as pbar:
                    for document in documents_to_split:
                        document_splits[document.id_] = document.split()
                        pbar.update()

            obj_contents = {}
            for document in documents:
                if refresh_documents or document.id_ not in self.doc_store:
                    self.doc_store[document.id_] = document
                if document.id_ in document_splits:
                    document_chunks = document_splits[document.id_]
                    obj = StoreEmbedding(document.id_)
                    for document_chunk in document_chunks:
                        if document_chunk.id_ != document.id_:
                            if refresh_documents or document_chunk.id_ not in self.doc_store:
                                self.doc_store[document_chunk.id_] = document_chunk
                            if refresh_embeddings or document_chunk.id_ not in self.emb_store:
                                child_obj = StoreEmbedding(
                                    document_chunk.id_, parent_id=document.id_
                                )
                                self.emb_store[child_obj.id_] = child_obj
                            else:
                                child_obj = self.emb_store[document_chunk.id_]
                            obj.child_ids.append(child_obj.id_)
                    if (
                        refresh_documents
                        or refresh_embeddings
                        or document.id_ not in self.emb_store
                    ):
                        self.emb_store[obj.id_] = obj
                else:
                    obj = self.emb_store[document.id_]
                if not obj.embedding:
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            child_obj = self.emb_store[child_id]
                            if not child_obj.embedding:
                                child_document = self.doc_store[child_id]
                                content = child_document.get_content(for_embed=True)
                                if content:
                                    obj_contents[child_id] = content
                                    if with_fallback:
                                        raise FallbackError("Some documents need to be embedded")
                    else:
                        content = document.get_content(for_embed=True)
                        if content:
                            obj_contents[obj.id_] = content
                            if with_fallback:
                                raise FallbackError("Some documents need to be embedded")

            if obj_contents:
                if self.embeddings is None:
                    if with_fallback:
                        raise FallbackError("Embeddings engine is not set")
                    raise ValueError("Embeddings engine is not set")
                total = 0
                for batch in self.embeddings.iter_embedding_batches(list(obj_contents.values())):
                    batch_keys = list(obj_contents.keys())[total : total + len(batch)]
                    obj_embeddings = dict(zip(batch_keys, batch))
                    for obj_id, embedding in obj_embeddings.items():
                        obj = self.emb_store[obj_id]
                        new_obj = obj.replace(embedding=embedding)
                        self.emb_store[new_obj.id_] = new_obj
                    total += len(batch)

            if return_embeddings or return_documents:
                embeddings = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if obj.embedding:
                        if return_documents:
                            embeddings.append(EmbeddedDocument(document, embedding=obj.embedding))
                        else:
                            embeddings.append(obj.embedding)
                    elif obj.child_ids:
                        child_embeddings = []
                        for child_id in obj.child_ids:
                            child_obj = self.emb_store[child_id]
                            if child_obj.embedding:
                                if return_documents:
                                    child_document = self.doc_store[child_id]
                                    child_embeddings.append(
                                        EmbeddedDocument(
                                            child_document, embedding=child_obj.embedding
                                        )
                                    )
                                else:
                                    child_embeddings.append(child_obj.embedding)
                            else:
                                if return_documents:
                                    child_document = self.doc_store[child_id]
                                    child_embeddings.append(EmbeddedDocument(child_document))
                                else:
                                    child_embeddings.append(None)
                        if return_documents:
                            embeddings.append(
                                EmbeddedDocument(document, child_documents=child_embeddings)
                            )
                        else:
                            embeddings.append(child_embeddings)
                    else:
                        if return_documents:
                            embeddings.append(EmbeddedDocument(document))
                        else:
                            embeddings.append(None)

                return embeddings

    def compute_score(
        self,
        emb1: tp.Union[tp.MaybeIterable[tp.List[float]], np.ndarray],
        emb2: tp.Union[tp.MaybeIterable[tp.List[float]], np.ndarray],
    ) -> tp.Union[float, np.ndarray]:
        """Compute similarity or distance scores between embeddings.

        Compute scores between embedding vectors using the configured scoring function.
        Supported functions include "cosine", "euclidean", and "dot". Alternatively, a callable
        metric can be supplied that accepts two arrays and returns a 2-dimensional ndarray.

        Args:
            emb1 (Union[MaybeIterable[List[float]], ndarray]): First embedding or collection of embeddings.
            emb2 (Union[MaybeIterable[List[float]], ndarray]): Second embedding or collection of embeddings.

        Returns:
            Union[float, ndarray]: Computed score or score matrix between the embeddings.
        """
        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        emb1_single = emb1.ndim == 1
        emb2_single = emb2.ndim == 1
        if emb1_single:
            emb1 = emb1.reshape(1, -1)
        if emb2_single:
            emb2 = emb2.reshape(1, -1)

        if isinstance(self.score_func, str):
            if self.score_func.lower() == "cosine":
                emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
                emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
                emb1_norm = np.nan_to_num(emb1_norm)
                emb2_norm = np.nan_to_num(emb2_norm)
                score_matrix = np.dot(emb1_norm, emb2_norm.T)
            elif self.score_func.lower() == "euclidean":
                diff = emb1[:, np.newaxis, :] - emb2[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=2)
                score_matrix = np.divide(
                    1, distances, where=distances != 0, out=np.full_like(distances, np.inf)
                )
            elif self.score_func.lower() == "dot":
                score_matrix = np.dot(emb1, emb2.T)
            else:
                raise ValueError(f"Invalid distance function: '{self.score_func}'")
        else:
            score_matrix = self.score_func(emb1, emb2)

        if emb1_single and emb2_single:
            return float(score_matrix[0, 0])
        if emb1_single or emb2_single:
            return score_matrix.flatten()
        return score_matrix

    def score_documents(
        self,
        query: str,
        documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_chunks: bool = False,
        return_documents: bool = False,
        with_fallback: bool = False,
    ) -> tp.ScoredDocuments:
        """Score documents by relevance to a query.

        Optionally refresh and embed documents before scoring their relevance to a query.
        If no documents are provided, the document store is used. When `return_chunks` is True,
        document chunks are scored instead of parent documents. The query is embedded and compared
        against document embeddings to compute relevance scores.

        Args:
            query (str): Query string for scoring relevance.
            documents (Optional[Iterable[StoreDocument]]): Collection of documents to score.

                If None, documents from the document store are used.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_chunks (bool): Whether to return document chunks.
            return_documents (bool): If True, include original document objects in the output.
            with_fallback (bool): If True, raise `FallbackError` if new embeddings are needed.

        Returns:
            ScoredDocuments: Collection of documents with their computed relevance scores.
        """
        with self.doc_store, self.emb_store:
            if documents is None:
                if self.doc_store is None:
                    raise ValueError("Must provide at least documents or doc_store")
                documents = self.doc_store.values()
                documents_provided = False
            else:
                documents_provided = True
            documents = list(documents)
            if not documents:
                return []
            self.embed_documents(
                documents,
                refresh=refresh,
                refresh_documents=refresh_documents,
                refresh_embeddings=refresh_embeddings,
                with_fallback=with_fallback,
            )
            if return_chunks:
                document_chunks = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            document_chunk = self.doc_store[child_id]
                            document_chunks.append(document_chunk)
                    elif not obj.parent_id or obj.parent_id not in self.doc_store:
                        document_chunk = self.doc_store[obj.id_]
                        document_chunks.append(document_chunk)
                documents = document_chunks
            elif not documents_provided:
                document_parents = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if not obj.parent_id or obj.parent_id not in self.doc_store:
                        document_parent = self.doc_store[obj.id_]
                        document_parents.append(document_parent)
                documents = document_parents

            obj_embeddings = {}
            for document in documents:
                obj = self.emb_store[document.id_]
                if obj.embedding:
                    obj_embeddings[obj.id_] = obj.embedding
                elif obj.child_ids:
                    for child_id in obj.child_ids:
                        child_obj = self.emb_store[child_id]
                        if child_obj.embedding:
                            obj_embeddings[child_id] = child_obj.embedding
            if obj_embeddings:
                if self.embeddings is None:
                    if with_fallback:
                        raise FallbackError("Embeddings engine is not set")
                    raise ValueError("Embeddings engine is not set")
                query_embedding = self.embeddings.get_embedding(query)
                scores = self.compute_score(query_embedding, list(obj_embeddings.values()))
                obj_scores = dict(zip(obj_embeddings.keys(), scores))
            else:
                obj_scores = {}

            scores = []
            for document in documents:
                obj = self.emb_store[document.id_]
                child_scores = []
                if obj.child_ids:
                    for child_id in obj.child_ids:
                        if child_id in obj_scores:
                            child_score = obj_scores[child_id]
                            if return_documents:
                                child_document = self.doc_store[child_id]
                                child_scores.append(
                                    ScoredDocument(child_document, score=child_score)
                                )
                            else:
                                child_scores.append(child_score)
                    if child_scores:
                        if return_documents:
                            doc_score = self.score_agg_func(
                                [document.score for document in child_scores]
                            )
                        else:
                            doc_score = self.score_agg_func(child_scores)
                    else:
                        doc_score = float("nan")
                else:
                    if obj.id_ in obj_scores:
                        doc_score = obj_scores[obj.id_]
                    else:
                        doc_score = float("nan")
                if return_documents:
                    scores.append(
                        ScoredDocument(document, score=doc_score, child_documents=child_scores)
                    )
                else:
                    scores.append(doc_score)
            return scores

    SPLIT_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])|_")
    """Regular expression pattern used by `DocumentRanker.bm25_splitter` to split text at
    transitions between lowercase and uppercase letters or underscores."""

    TOKEN_PATTERN = re.compile(r"(?u)\b\w{2,}\b")
    """Regular expression pattern used by `DocumentRanker.bm25_splitter` to extract tokens with
    at least two characters."""

    @classmethod
    def bm25_splitter(cls, text: str) -> tp.List[str]:
        """Return a list of lowercase tokens extracted from the input text using BM25 tokenization.

        Args:
            text (str): Text to tokenize.

        Returns:
            list[str]: Lowercase tokens extracted from the input text.
        """
        spaced_text = cls.SPLIT_PATTERN.sub(" ", text)
        tokens = cls.TOKEN_PATTERN.findall(spaced_text)
        return [token.lower() for token in tokens]

    def bm25_score_documents(
        self,
        query: str,
        documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        return_chunks: bool = False,
        return_documents: bool = False,
    ) -> tp.ScoredDocuments:
        """Return BM25 relevance scores for documents matching a query.

        Args:
            query (str): Query string for relevance scoring.
            documents (Optional[Iterable[StoreDocument]]): Collection of documents to score.

                If None, documents from the document store are used.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            return_chunks (bool): Whether to return document chunks.
            return_documents (bool): If True, include original document objects in the output.

        Returns:
            ScoredDocuments: Computed BM25 scores for each document, as either numeric scores or
                `ScoredDocument` objects.
        """
        with self.doc_store, self.emb_store:
            if refresh_documents is None:
                refresh_documents = refresh
            if documents is None:
                if self.doc_store is None:
                    raise ValueError("Must provide at least documents or doc_store")
                documents = self.doc_store.values()
            documents = list(documents)

            if return_chunks:
                documents_to_split = []
                document_splits = {}
                for document in documents:
                    refresh_document = (
                        refresh_documents
                        or document.id_ not in self.doc_store
                        or document.id_ not in self.emb_store
                    )
                    if not refresh_document:
                        obj = self.emb_store[document.id_]
                        if obj.child_ids:
                            for child_id in obj.child_ids:
                                if child_id not in self.doc_store or child_id not in self.emb_store:
                                    refresh_document = True
                                    break
                    if refresh_document:
                        documents_to_split.append(document)
                if documents_to_split:
                    from vectorbtpro.utils.pbar import ProgressBar

                    pbar_kwargs = merge_dicts(dict(prefix="split_documents"), self.pbar_kwargs)
                    with ProgressBar(
                        total=len(documents_to_split),
                        show_progress=self.show_progress,
                        **pbar_kwargs,
                    ) as pbar:
                        for document in documents_to_split:
                            document_splits[document.id_] = document.split()
                            pbar.update()

                for document in documents:
                    if refresh_documents or document.id_ not in self.doc_store:
                        self.doc_store[document.id_] = document
                    if document.id_ in document_splits:
                        document_chunks = document_splits[document.id_]
                        obj = StoreEmbedding(document.id_)
                        for document_chunk in document_chunks:
                            if document_chunk.id_ != document.id_:
                                if refresh_documents or document_chunk.id_ not in self.doc_store:
                                    self.doc_store[document_chunk.id_] = document_chunk
                                if document_chunk.id_ not in self.emb_store:
                                    child_obj = StoreEmbedding(
                                        document_chunk.id_, parent_id=document.id_
                                    )
                                    self.emb_store[child_obj.id_] = child_obj
                                else:
                                    child_obj = self.emb_store[document_chunk.id_]
                                obj.child_ids.append(child_obj.id_)
                        if refresh_documents or document.id_ not in self.emb_store:
                            self.emb_store[obj.id_] = obj

                document_chunks = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            document_chunk = self.doc_store[child_id]
                            document_chunks.append(document_chunk)
                    elif not obj.parent_id or obj.parent_id not in self.doc_store:
                        document_chunk = self.doc_store[obj.id_]
                        document_chunks.append(document_chunk)
                documents = document_chunks

            bm25_tokenizer = self.bm25_tokenizer
            bm25_retriever = self.bm25_retriever
            bm25_tokenize_kwargs = dict(self.bm25_tokenize_kwargs)
            bm25_retrieve_kwargs = dict(self.bm25_retrieve_kwargs)
            if (
                refresh_documents
                or not bm25_tokenizer.get_vocab_dict()
                or not hasattr(bm25_retriever, "scores")
                or not bm25_retriever.scores
                or bm25_retriever.scores["num_docs"] != len(documents)
            ):
                texts = []
                for document in documents:
                    content = document.get_content(for_embed=True)
                    if not content:
                        content = ""
                    texts.append(content)
                tokenized_documents = bm25_tokenizer.tokenize(
                    texts,
                    return_as="string",
                    **bm25_tokenize_kwargs,
                )
                bm25_retriever.index(tokenized_documents, show_progress=False)
            if "update_vocab" in bm25_tokenize_kwargs:
                del bm25_tokenize_kwargs["update_vocab"]
            if "show_progress" in bm25_tokenize_kwargs:
                del bm25_tokenize_kwargs["show_progress"]
            tokenized_queries = bm25_tokenizer.tokenize(
                [query],
                return_as="string",
                update_vocab=False,
                show_progress=False,
                **bm25_tokenize_kwargs,
            )
            indices, scores = bm25_retriever.retrieve(
                tokenized_queries,
                k=len(documents),
                sorted=False,
                **bm25_retrieve_kwargs,
            )
            obj_scores = {}
            for i in range(scores.shape[1]):
                obj_scores[documents[indices[0, i]].id_] = scores[0, i]

            scores = []
            for document in documents:
                if return_chunks:
                    obj = self.emb_store[document.id_]
                    child_scores = []
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            if child_id in obj_scores:
                                child_score = obj_scores[child_id]
                                if return_documents:
                                    child_document = self.doc_store[child_id]
                                    child_scores.append(
                                        ScoredDocument(child_document, score=child_score)
                                    )
                                else:
                                    child_scores.append(child_score)
                        if child_scores:
                            if return_documents:
                                doc_score = self.score_agg_func(
                                    [document.score for document in child_scores]
                                )
                            else:
                                doc_score = self.score_agg_func(child_scores)
                        else:
                            doc_score = float("nan")
                    else:
                        if obj.id_ in obj_scores:
                            doc_score = obj_scores[obj.id_]
                        else:
                            doc_score = float("nan")
                else:
                    if document.id_ in obj_scores:
                        doc_score = obj_scores[document.id_]
                    else:
                        doc_score = float("nan")
                    child_scores = []
                if return_documents:
                    scores.append(
                        ScoredDocument(document, score=doc_score, child_documents=child_scores)
                    )
                else:
                    scores.append(doc_score)
            return scores

    @classmethod
    def resolve_top_k(
        cls, scores: tp.Iterable[float], top_k: tp.TopKLike = None
    ) -> tp.Optional[int]:
        """Resolve the `top_k` value from sorted scores.

        Args:
            scores (Iterable[float]): Sorted document scores.
            top_k (TopKLike): Parameter specifying the `top_k` selection method, which can be an integer,
                a float percentage, a string ('elbow' or 'kmeans'), or a callable.

        Returns:
            Optional[int]: Resolved `top_k` value, or None if `top_k` is not provided.
        """
        if top_k is None:
            return None
        scores = np.asarray(scores)
        scores = scores[~np.isnan(scores)]

        if isinstance(top_k, str):
            if top_k.lower() == "elbow":
                if scores.size == 0:
                    return 0
                diffs = np.diff(scores)
                top_k = np.argmax(-diffs) + 1
            elif top_k.lower() == "kmeans":
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=2, random_state=0).fit(scores.reshape(-1, 1))
                high_score_cluster = np.argmax(kmeans.cluster_centers_)
                top_k_indices = np.where(kmeans.labels_ == high_score_cluster)[0]
                top_k = max(top_k_indices) + 1
            else:
                raise ValueError(f"Invalid top_k method: '{top_k}'")
        elif callable(top_k):
            top_k = top_k(scores)
        if checks.is_float(top_k):
            top_k = int(top_k * len(scores))
        return top_k

    @classmethod
    def top_k_from_cutoff(
        cls, scores: tp.Iterable[float], cutoff: tp.Optional[float] = None
    ) -> tp.Optional[int]:
        """Determine the number of top documents based on a cutoff threshold from sorted scores.

        Args:
            scores (Iterable[float]): Sorted document scores.
            cutoff (Optional[float]): Score threshold to filter documents.

        Returns:
            Optional[int]: Count of scores greater than or equal to the cutoff, or None if cutoff is None.
        """
        if cutoff is None:
            return None
        scores = np.asarray(scores)
        scores = scores[~np.isnan(scores)]
        return len(scores[scores >= cutoff])

    @classmethod
    def extract_doc_scores(cls, scored_documents: tp.List[ScoredDocument]) -> tp.List[float]:
        """Recursively extract scores from a list of scored documents.

        Args:
            scored_documents (List[ScoredDocument]): Documents with existing scores.

        Returns:
            List[float]: Scores extracted from each document and its child documents.
        """
        scores = []
        for document in scored_documents:
            scores.append(document.score)
            if document.child_documents:
                scores.extend(cls.extract_doc_scores(document.child_documents))
        return scores

    @classmethod
    def normalize_doc_scores(cls, scores: tp.Iterable[float]) -> np.ndarray:
        """Normalize a collection of scores using min-max scaling.

        Args:
            scores (Iterable[float]): Iterable of scores to normalize.

        Returns:
            ndarray: Array of normalized scores.
        """
        scores = np.array(scores, dtype=float)
        min_score, max_score = np.nanmin(scores), np.nanmax(scores)
        return (
            (scores - min_score) / (max_score - min_score)
            if max_score != min_score
            else scores - min_score
        )

    @classmethod
    def replace_doc_scores(
        cls,
        scored_documents: tp.List[ScoredDocument],
        new_scores: tp.List[float],
    ) -> tp.List[ScoredDocument]:
        """Recursively replace scores in documents with new scores.

        Args:
            scored_documents (List[ScoredDocument]): Documents with existing scores.
            new_scores (List[float]): New scores to assign, consumed in order.

        Returns:
            List[ScoredDocument]: Updated documents with replaced scores.
        """
        new_scored_documents = []
        for i in range(len(scored_documents)):
            doc = scored_documents[i]
            document = doc.document
            score = new_scores.pop(0)
            if doc.child_documents:
                child_documents = cls.replace_doc_scores(doc.child_documents, new_scores)
            else:
                child_documents = []
            new_scored_documents.append(
                ScoredDocument(document, score=score, child_documents=child_documents)
            )
        return new_scored_documents

    @classmethod
    def normalize_scored_documents(
        cls, scored_documents: tp.List[ScoredDocument]
    ) -> tp.List[ScoredDocument]:
        """Normalize the scores of scored documents using min-max scaling.

        Args:
            scored_documents (List[ScoredDocument]): Documents with existing scores.

        Returns:
            List[ScoredDocument]: Documents with normalized scores.
        """
        scores = cls.extract_doc_scores(scored_documents)
        new_scores = cls.normalize_doc_scores(scores).tolist()
        return cls.replace_doc_scores(scored_documents, new_scores)

    @classmethod
    def extract_doc_pair_scores(
        cls,
        emb_scored_documents: tp.List[ScoredDocument],
        bm25_scored_documents: tp.List[ScoredDocument],
    ) -> tp.List[tp.Tuple[float, float]]:
        """Recursively extract paired scores from embedding and BM25 scored documents.

        Args:
            emb_scored_documents (List[ScoredDocument]): Documents scored using embeddings.
            bm25_scored_documents (List[ScoredDocument]): Documents scored using BM25.

        Returns:
            List[Tuple[float, float]]: Pairs of scores from corresponding documents and their child documents.
        """

        def _score(doc):
            return doc.score if doc is not None else float("nan")

        def _children(doc):
            return doc.child_documents if doc is not None else []

        emb_map = {}
        bm25_map = {}
        order = []
        for d in emb_scored_documents:
            doc_id = d.document.id_
            emb_map[doc_id] = d
            order.append(doc_id)
        for d in bm25_scored_documents:
            doc_id = d.document.id_
            bm25_map[doc_id] = d
            if doc_id not in emb_map:
                order.append(doc_id)

        pair_scores = []
        for doc_id in order:
            emb_doc = emb_map.get(doc_id)
            bm25_doc = bm25_map.get(doc_id)
            pair_scores.append((_score(emb_doc), _score(bm25_doc)))
            if _children(emb_doc) or _children(bm25_doc):
                child_pair_scores = cls.extract_doc_pair_scores(
                    _children(emb_doc), _children(bm25_doc)
                )
                pair_scores.extend(child_pair_scores)
        return pair_scores

    def fuse_doc_pair_scores(
        self, doc_pair_scores: tp.Iterable[tp.Tuple[float, float]]
    ) -> np.ndarray:
        """Fuse paired (embedding, BM25) scores with Reciprocal-Rank Fusion (RRF).

        Args:
            doc_pair_scores (Iterable[Tuple[float, float]]): Paired scores (embedding, BM25) to fuse.

        Returns:
            ndarray: Array of fused scores.
        """
        pair_scores = np.asarray(doc_pair_scores, dtype=float)
        emb_scores = pair_scores[:, 0]
        bm25_scores = pair_scores[:, 1]

        emb_tmp = np.where(np.isnan(emb_scores), -np.inf, emb_scores)
        bm25_tmp = np.where(np.isnan(bm25_scores), -np.inf, bm25_scores)
        emb_order = np.argsort(-emb_tmp, kind="mergesort")
        bm25_order = np.argsort(-bm25_tmp, kind="mergesort")

        emb_rank = np.empty(len(pair_scores), dtype=np.int32)
        bm25_rank = np.empty(len(pair_scores), dtype=np.int32)
        emb_rank[emb_order] = np.arange(1, len(pair_scores) + 1)
        bm25_rank[bm25_order] = np.arange(1, len(pair_scores) + 1)

        new_emb_scores = (1 - self.rrf_bm25_weight) / (self.rrf_k + emb_rank)
        new_bm25_scores = self.rrf_bm25_weight / (self.rrf_k + bm25_rank)
        return new_emb_scores + new_bm25_scores

    @classmethod
    def replace_doc_pair_scores(
        cls,
        emb_scored_documents: tp.List[ScoredDocument],
        bm25_scored_documents: tp.List[ScoredDocument],
        new_scores: tp.List[float],
    ) -> tp.List[ScoredDocument]:
        """Recursively replace scores in paired embedding and BM25 documents with new scores.

        Args:
            emb_scored_documents (List[ScoredDocument]): Documents scored using embeddings.
            bm25_scored_documents (List[ScoredDocument]): Documents scored using BM25.
            new_scores (List[float]): New scores to assign, consumed in order.

        Returns:
            List[ScoredDocument]: Updated documents with replaced paired scores.
        """

        def _children(doc):
            return doc.child_documents if doc is not None else []

        emb_map = {}
        bm25_map = {}
        order = []
        for d in emb_scored_documents:
            doc_id = d.document.id_
            emb_map[doc_id] = d
            order.append(doc_id)
        for d in bm25_scored_documents:
            doc_id = d.document.id_
            bm25_map[doc_id] = d
            if doc_id not in emb_map:
                order.append(doc_id)

        scored_documents = []
        for doc_id in order:
            emb_doc = emb_map.get(doc_id)
            bm25_doc = bm25_map.get(doc_id)
            if emb_doc is not None:
                document = emb_doc.document
            else:
                document = bm25_doc.document
            score = new_scores.pop(0)
            if _children(emb_doc) or _children(bm25_doc):
                child_documents = cls.replace_doc_pair_scores(
                    _children(emb_doc), _children(bm25_doc), new_scores
                )
            else:
                child_documents = []
            scored_documents.append(
                ScoredDocument(document, score=score, child_documents=child_documents)
            )
        return scored_documents

    def fuse_scored_documents(
        self,
        emb_scored_documents: tp.List[ScoredDocument],
        bm25_scored_documents: tp.List[ScoredDocument],
    ) -> tp.List[ScoredDocument]:
        """Fuse embedding and BM25 scored documents by merging and updating their scores.

        Args:
            emb_scored_documents (List[ScoredDocument]): Documents scored using embeddings.
            bm25_scored_documents (List[ScoredDocument]): Documents scored using BM25.

        Returns:
            List[ScoredDocument]: Fused scored documents with updated scores.
        """
        doc_pair_scores = self.extract_doc_pair_scores(emb_scored_documents, bm25_scored_documents)
        new_scores = self.fuse_doc_pair_scores(doc_pair_scores).tolist()
        return self.replace_doc_pair_scores(emb_scored_documents, bm25_scored_documents, new_scores)

    def rank_documents(
        self,
        query: str,
        documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
        top_k: tp.TopKLike = None,
        min_top_k: tp.TopKLike = None,
        max_top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_chunks: bool = False,
        return_scores: bool = False,
    ) -> tp.RankedDocuments:
        """Rank documents based on their relevance to a query.

        The method retrieves scored documents using embedding and BM25 strategies (or both in hybrid mode),
        fuses and normalizes their scores, and then sorts them to identify the most relevant documents.
        Top-k parameters and score cutoff are resolved using `DocumentRanker.resolve_top_k` and
        `DocumentRanker.top_k_from_cutoff`.

        Args:
            query (str): Query string to evaluate document relevance.
            documents (Optional[Iterable[StoreDocument]]): Collection of documents to rank.

                If None, documents from the document store are used.
            top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
            min_top_k (TopKLike): Minimum limit for determining top documents.
            max_top_k (TopKLike): Maximum limit for determining top documents.
            cutoff (Optional[float]): Score threshold to filter documents.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_chunks (bool): Whether to return document chunks.
            return_scores (bool): Whether to return scored documents with their scores.

        Returns:
            RankedDocuments: Documents ranked by relevance to the query.
        """
        if documents is not None:
            documents = list(documents)
        if self.search_method in ("embeddings", "hybrid", "embeddings_fallback", "hybrid_fallback"):
            try:
                emb_scored_documents = self.score_documents(
                    query,
                    documents=documents,
                    refresh=refresh,
                    refresh_documents=refresh_documents,
                    refresh_embeddings=refresh_embeddings,
                    return_chunks=return_chunks,
                    return_documents=True,
                    with_fallback=self.search_method in ("embeddings_fallback", "hybrid_fallback"),
                )
            except FallbackError as e:
                warn(f'Fallback triggered: "{e}"')
                emb_scored_documents = None
        else:
            emb_scored_documents = None
        if self.search_method in ("bm25", "hybrid") or (
            emb_scored_documents is None
            and self.search_method in ("embeddings_fallback", "hybrid_fallback")
        ):
            bm25_scored_documents = self.bm25_score_documents(
                query,
                documents=documents,
                refresh=refresh,
                refresh_documents=refresh_documents,
                return_chunks=return_chunks,
                return_documents=True,
            )
        else:
            bm25_scored_documents = None
        if emb_scored_documents is not None and bm25_scored_documents is not None:
            scored_documents = self.fuse_scored_documents(
                emb_scored_documents, bm25_scored_documents
            )
        elif emb_scored_documents is not None:
            scored_documents = emb_scored_documents
        elif bm25_scored_documents is not None:
            scored_documents = bm25_scored_documents
        else:
            raise NotImplementedError
        if self.normalize_scores:
            scored_documents = self.normalize_scored_documents(scored_documents)
        scored_documents = sorted(
            scored_documents, key=lambda x: (not np.isnan(x.score), x.score), reverse=True
        )
        scores = [document.score for document in scored_documents]

        int_top_k = top_k is not None and checks.is_int(top_k)
        top_k = self.resolve_top_k(scores, top_k=top_k)
        min_top_k = self.resolve_top_k(scores, top_k=min_top_k)
        max_top_k = self.resolve_top_k(scores, top_k=max_top_k)
        cutoff = self.top_k_from_cutoff(scores, cutoff=cutoff)
        if not int_top_k and min_top_k is not None and min_top_k > top_k:
            top_k = min_top_k
        if not int_top_k and max_top_k is not None and max_top_k < top_k:
            top_k = max_top_k
        if cutoff is not None and min_top_k is not None and min_top_k > cutoff:
            cutoff = min_top_k
        if cutoff is not None and max_top_k is not None and max_top_k < cutoff:
            cutoff = max_top_k
        if top_k is None:
            top_k = len(scores)
        if cutoff is None:
            cutoff = len(scores)
        top_k = min(top_k, cutoff)
        if top_k == 0:
            raise ValueError("No documents selected after ranking. Change top_k or cutoff.")
        scored_documents = scored_documents[:top_k]
        if return_scores:
            return scored_documents
        return [document.document for document in scored_documents]


def embed_documents(
    documents: tp.Iterable[StoreDocument],
    refresh: bool = False,
    refresh_documents: tp.Optional[bool] = None,
    refresh_embeddings: tp.Optional[bool] = None,
    return_embeddings: bool = False,
    return_documents: bool = False,
    doc_ranker: tp.Optional[tp.MaybeType[DocumentRanker]] = None,
    **kwargs,
) -> tp.Optional[tp.EmbeddedDocuments]:
    """Embed the provided documents using a `DocumentRanker`.

    Args:
        documents (Iterable[StoreDocument]): Collection of documents to embed.
        refresh (bool): Flag to refresh both documents and embeddings.
        refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
        refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
        return_embeddings (bool): Flag indicating whether to return embeddings.
        return_documents (bool): If True, include original document objects in the output.
        doc_ranker (Optional[MaybeType[DocumentRanker]]): `DocumentRanker` class or instance.
        **kwargs: Keyword arguments to initialize or update `doc_ranker`.

    Returns:
        Optional[EmbeddedDocuments]: Embedded documents output.
    """
    if doc_ranker is None:
        doc_ranker = DocumentRanker
    if isinstance(doc_ranker, type):
        checks.assert_subclass_of(doc_ranker, DocumentRanker, "doc_ranker")
        doc_ranker = doc_ranker(**kwargs)
    else:
        checks.assert_instance_of(doc_ranker, DocumentRanker, "doc_ranker")
        if kwargs:
            doc_ranker = doc_ranker.replace(**kwargs)
    return doc_ranker.embed_documents(
        documents,
        refresh=refresh,
        refresh_documents=refresh_documents,
        refresh_embeddings=refresh_embeddings,
        return_embeddings=return_embeddings,
        return_documents=return_documents,
    )


def rank_documents(
    query: str,
    documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
    top_k: tp.TopKLike = None,
    min_top_k: tp.TopKLike = None,
    max_top_k: tp.TopKLike = None,
    cutoff: tp.Optional[float] = None,
    refresh: bool = False,
    refresh_documents: tp.Optional[bool] = None,
    refresh_embeddings: tp.Optional[bool] = None,
    return_chunks: bool = False,
    return_scores: bool = False,
    doc_ranker: tp.Optional[tp.MaybeType[DocumentRanker]] = None,
    **kwargs,
) -> tp.RankedDocuments:
    """Rank documents based on their relevance to a query using a `DocumentRanker`.

    Args:
        query (str): Query string for ranking.
        documents (Optional[Iterable[StoreDocument]]): Collection of documents to rank.

            If None, documents from the document store are used.
        top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
        min_top_k (TopKLike): Minimum limit for determining top documents.
        max_top_k (TopKLike): Maximum limit for determining top documents.
        cutoff (Optional[float]): Score threshold to filter documents.
        refresh (bool): Flag to refresh both documents and embeddings.
        refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
        refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
        return_chunks (bool): Whether to return document chunks.
        return_scores (bool): Whether to return scored documents with their scores.
        doc_ranker (Optional[MaybeType[DocumentRanker]]): `DocumentRanker` class or instance.
        **kwargs: Keyword arguments to initialize or update `doc_ranker`.

    Returns:
        RankedDocuments: Ranked documents based on the query relevance.
    """
    if doc_ranker is None:
        doc_ranker = DocumentRanker
    if isinstance(doc_ranker, type):
        checks.assert_subclass_of(doc_ranker, DocumentRanker, "doc_ranker")
        doc_ranker = doc_ranker(**kwargs)
    else:
        checks.assert_instance_of(doc_ranker, DocumentRanker, "doc_ranker")
        if kwargs:
            doc_ranker = doc_ranker.replace(**kwargs)
    return doc_ranker.rank_documents(
        query,
        documents=documents,
        top_k=top_k,
        min_top_k=min_top_k,
        max_top_k=max_top_k,
        cutoff=cutoff,
        refresh=refresh,
        refresh_documents=refresh_documents,
        refresh_embeddings=refresh_embeddings,
        return_chunks=return_chunks,
        return_scores=return_scores,
    )


RankableT = tp.TypeVar("RankableT", bound="Rankable")


class Rankable(HasSettings):
    """Abstract class representing an entity that supports embedding and ranking operations.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and its sub-configuration `chat`.
    """

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def embed(
        self: RankableT,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_embeddings: bool = False,
        return_documents: bool = False,
        **kwargs,
    ) -> tp.Optional[RankableT]:
        """Embed the instance's documents.

        Args:
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_embeddings (bool): Flag indicating whether to return embeddings.
            return_documents (bool): If True, include original document objects in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[Rankable]: Updated instance with embedded documents, if available.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def rank(
        self: RankableT,
        query: str,
        top_k: tp.TopKLike = None,
        min_top_k: tp.TopKLike = None,
        max_top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        refresh: bool = False,
        refresh_documents: tp.Optional[bool] = None,
        refresh_embeddings: tp.Optional[bool] = None,
        return_chunks: bool = False,
        return_scores: bool = False,
        **kwargs,
    ) -> RankableT:
        """Rank documents based on their relevance to a provided query.

        Args:
            query (str): Query string to evaluate document relevance.
            top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
            min_top_k (TopKLike): Minimum limit for determining top documents.
            max_top_k (TopKLike): Maximum limit for determining top documents.
            cutoff (Optional[float]): Score threshold to filter documents.
            refresh (bool): Flag to refresh both documents and embeddings.
            refresh_documents (Optional[bool]): Flag to refresh documents; defaults to `refresh`.
            refresh_embeddings (Optional[bool]): Flag to refresh embeddings; defaults to `refresh`.
            return_chunks (bool): Whether to return document chunks.
            return_scores (bool): Whether to return scored documents with their scores.
            **kwargs: Additional keyword arguments.

        Returns:
            Rankable: Updated instance with ranked documents.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


# ############# Contexting ############# #


class Contextable(HasSettings):
    """Abstract class that provides functionality to generate a textual context.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and its sub-configuration `chat`.
    """

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def to_context(self, *args, **kwargs) -> str:
        """Convert the instance into a textual context.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Textual context representation.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def count_tokens(
        self,
        to_context_kwargs: tp.KwargsLike = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
    ) -> int:
        """Count the number of tokens in the generated context.

        Args:
            to_context_kwargs (KwargsLike): Keyword arguments for `Contextable.to_context`.
            tokenizer (TokenizerLike): Identifier, subclass, or instance of `Tokenizer`.

                Resolved using `resolve_tokenizer`.
        tokenizer_kwargs (KwargsLike): Keyword arguments to initialize or update `tokenizer`.

        Returns:
            int: Number of tokens in the context.
        """
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True)
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(
            tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True
        )

        context = self.to_context(**to_context_kwargs)
        tokenizer = resolve_tokenizer(tokenizer)
        if isinstance(tokenizer, type):
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)
        return len(tokenizer.encode(context))

    def create_chat(
        self,
        to_context_kwargs: tp.KwargsLike = None,
        completions: tp.CompletionsLike = None,
        **kwargs,
    ) -> tp.Completions:
        """Create a chat interface using the generated context.

        Args:
            to_context_kwargs (KwargsLike): Keyword arguments for `Contextable.to_context`.
            completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

                Resolved using `resolve_completions`.
            **kwargs: Keyword arguments to initialize or update `completions`.

        Returns:
            Completions: Instance of `Completions` configured with the generated context.

        Examples:
            ```pycon
            >>> chat = asset.create_chat()

            >>> chat.get_completion("What's the value under 'xyz'?")
            The value under 'xyz' is 123.

            >>> chat.get_completion("Are you sure?")
            Yes, I am sure. The value under 'xyz' is 123 for the entry where `s` is "EFG".
            ```
        """
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True)
        context = self.to_context(**to_context_kwargs)
        completions = resolve_completions(completions=completions)
        if isinstance(completions, type):
            completions = completions(context=context, **kwargs)
        else:
            completions = completions.replace(context=context, **kwargs)
        return completions

    @hybrid_method
    def chat(
        cls_or_self,
        message: str,
        chat_history: tp.Optional[tp.ChatHistory] = None,
        *,
        return_chat: bool = False,
        **kwargs,
    ) -> tp.MaybeChatOutput:
        """Chat with a language model using the instance as context.

        Args:
            message (str): Message to send to the language model.
            chat_history (Optional[ChatHistory]): Chat history, a list of dictionaries with defined roles.
            return_chat (bool): Flag indicating whether to return both the completion and the chat instance.
            **kwargs: Keyword arguments for `Contextable.create_chat`.

        Returns:
            MaybeChatOutput: Completion response or a tuple of the response and the chat instance.

        !!! note
            Context is recalculated each time this method is invoked. For multiple turns,
            it's more efficient to use `Contextable.create_chat`.

        Examples:
            ```pycon
            >>> asset.chat("What's the value under 'xyz'?")
            The value under 'xyz' is 123.

            >>> chat_history = []
            >>> asset.chat("What's the value under 'xyz'?", chat_history=chat_history)
            The value under 'xyz' is 123.

            >>> asset.chat("Are you sure?", chat_history=chat_history)
            Yes, I am sure. The value under 'xyz' is 123 for the entry where `s` is "EFG".
            ```
        """
        if isinstance(cls_or_self, type):
            args, kwargs = get_forward_args(super().chat, locals())
            return super().chat(*args, **kwargs)

        completions = cls_or_self.create_chat(chat_history=chat_history, **kwargs)
        if return_chat:
            return completions.get_completion(message), completions
        return completions.get_completion(message)


class RankContextable(Rankable, Contextable):
    """Abstract class combining `Rankable` and `Contextable` functionalities.

    This abstract class integrates ranking with contextual chat processing by applying
    ranking methods to chat queries when ranking parameters are provided.
    """

    @hybrid_method
    def chat(
        cls_or_self,
        message: str,
        chat_history: tp.Optional[tp.ChatHistory] = None,
        *,
        incl_past_queries: tp.Optional[bool] = None,
        rank: tp.Optional[bool] = None,
        top_k: tp.TopKLike = None,
        min_top_k: tp.TopKLike = None,
        max_top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        return_chunks: tp.Optional[bool] = None,
        rank_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeChatOutput:
        """Return the chat output with optional ranking applied.

        If `rank` is True, or if `rank` is None and any ranking parameter (`top_k`, `min_top_k`,
        `max_top_k`, `cutoff`, or `return_chunks`) is specified, process the query using
        `Rankable.rank` before delegating to `Contextable.chat`.

        Args:
            message (str): Message to send to the language model.
            chat_history (Optional[ChatHistory]): Chat history, a list of dictionaries with defined roles.
            incl_past_queries (Optional[bool]): Whether to include past queries in the ranking process.
            rank (Optional[bool]): Flag indicating whether to apply ranking.
            top_k (TopKLike): Number or percentage of top documents to return, or a method to determine it.
            min_top_k (TopKLike): Minimum limit for determining top documents.
            max_top_k (TopKLike): Maximum limit for determining top documents.
            cutoff (Optional[float]): Score threshold to filter documents.
            return_chunks (Optional[bool]): Whether to return document chunks.
            rank_kwargs (KwargsLike): Keyword arguments for `Rankable.rank`.
            **kwargs: Keyword arguments for `Contextable.chat`.

        Returns:
            MaybeChatOutput: Completion response or a tuple of the response and the chat instance.
        """
        if isinstance(cls_or_self, type):
            args, kwargs = get_forward_args(super().chat, locals())
            return super().chat(*args, **kwargs)

        incl_past_queries = cls_or_self.resolve_setting(incl_past_queries, "incl_past_queries")
        rank = cls_or_self.resolve_setting(rank, "rank")
        rank_kwargs = cls_or_self.resolve_setting(rank_kwargs, "rank_kwargs", merge=True)
        def_top_k = rank_kwargs.pop("top_k")
        if top_k is None:
            top_k = def_top_k
        def_min_top_k = rank_kwargs.pop("min_top_k")
        if min_top_k is None:
            min_top_k = def_min_top_k
        def_max_top_k = rank_kwargs.pop("max_top_k")
        if max_top_k is None:
            max_top_k = def_max_top_k
        def_cutoff = rank_kwargs.pop("cutoff")
        if cutoff is None:
            cutoff = def_cutoff
        def_return_chunks = rank_kwargs.pop("return_chunks")
        if return_chunks is None:
            return_chunks = def_return_chunks
        if rank or (rank is None and (top_k or min_top_k or max_top_k or cutoff or return_chunks)):
            if incl_past_queries and chat_history is not None:
                queries = []
                for message_dct in chat_history:
                    if "role" in message_dct and message_dct["role"] == "user":
                        queries.append(message_dct["content"])
                queries.append(message)
                if len(queries) > 1:
                    query = "\n\n".join(queries)
                else:
                    query = queries[0]
            else:
                query = message
            _cls_or_self = cls_or_self.rank(
                query,
                top_k=top_k,
                min_top_k=min_top_k,
                max_top_k=max_top_k,
                cutoff=cutoff,
                return_chunks=return_chunks,
                **rank_kwargs,
            )
        else:
            _cls_or_self = cls_or_self
        return Contextable.chat.__func__(_cls_or_self, message, chat_history, **kwargs)
