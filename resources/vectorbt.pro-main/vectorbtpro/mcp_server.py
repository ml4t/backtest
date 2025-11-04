# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing MCP server tools.

The module is meant to be executed as a script using the command:

```bash
python -m vectorbtpro.mcp_server
```
"""

import argparse

from vectorbtpro import _typing as tp

__all__ = []


def auto_cast(value: tp.Any) -> tp.Any:
    """Automatically cast a string to an appropriate Python literal type."""
    import ast

    if value is not None and isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def search(
    query: str,
    asset_names: tp.Union[str, tp.List[str]] = "all",
    search_method: str = "hybrid",
    return_chunks: bool = True,
    return_metadata: str = "none",
    max_tokens: tp.Optional[int] = 5_000,
    n: tp.Optional[int] = None,
    page: int = 1,
) -> str:
    """Search for VectorBT PRO (vectorbtpro, VBT) assets relevant to the provided
    (natural language) query and return the results as a context string.

    !!! note
        This tool is designed to search for general information about VBT.
        For specific information about a specific object (such as `vbt.Portfolio`),
        use tools that take a reference name. They operate on the actual objects.

        Also, running this tool on any combination of assets for the first time may take a while,
        as it prepares and caches the documents. If the tool times out repeatedly,
        it's recommended to call `vbt.search("")` directly in your code to build the index
        and then use the MCP tool to search the index.

    Args:
        query (str): Search query.

            Do not reinstate the name "VectorBT PRO" in the query, as it is already implied.
        asset_names (Union[str, List[str]]): One or more asset names to search. Supported names:

            * "api": API reference. Best for specific API queries.
            * "docs": Regular documentation, including getting started, features, tutorials, guides,
                recipes, and legal information. Best for general queries.
            * "messages": Discord messages and discussions. Best for support queries.
            * "examples": Code examples across all assets. Best for practical implementation queries.
            * "all": All of the above. Best for comprehensive queries.

            Order doesn't matter.

            Defaults to "all".
        search_method (str): Strategy for document search. Supported strategies:

            * "bm25": Uses BM25 for lexical search. Best for specific keywords.
            * "embeddings": Uses embeddings for semantic search. Best for general queries.
            * "hybrid": Combines both embeddings and BM25. Best for balanced search.

            Defaults to "hybrid". If embeddings are not available, it falls back to "bm25".
        return_chunks (bool): Whether to return the chunks of the results; otherwise, returns the full results.

            Defaults to True.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as title and URL.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "none".
        max_tokens (Optional[int]): Maximum number of tokens to return in the results.

            Defaults to 5,000.
        n (Optional[int]): Number of results to return per page.

            If specified, it will return the first `n` results that fit within the `max_tokens` limit.
            If None, returns all results that fit within the `max_tokens` limit. Defaults to None.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.

    Returns:
        str: Context string containing the search results.
    """
    from vectorbtpro.utils.knowledge.chatting import detokenize, tokenize
    from vectorbtpro.utils.knowledge.custom_assets import search
    from vectorbtpro.utils.pbar import ProgressHidden

    with ProgressHidden():
        query = auto_cast(query)
        asset_names = auto_cast(asset_names)
        search_method = auto_cast(search_method)
        return_chunks = auto_cast(return_chunks)
        return_metadata = auto_cast(return_metadata)
        max_tokens = auto_cast(max_tokens)
        n = auto_cast(n)
        page = auto_cast(page)

        if search_method == "embeddings":
            search_method = "embeddings_fallback"
        elif search_method == "hybrid":
            search_method = "hybrid_fallback"

        results = search(
            query,
            search_method=search_method,
            return_chunks=return_chunks,
            find_assets_kwargs=dict(
                asset_names=asset_names,
                minimize=False,
            ),
            display=False,
        )
        if n is not None:
            if page < 1:
                raise ValueError("Page number must be greater than or equal to 1")
            results = results[(page - 1) * n : page * n]
        elif page > 1:
            raise ValueError("Page number must be 1 when n is not specified")
        if return_metadata.lower() == "minimal":
            results = results.minimize_metadata()
        elif return_metadata.lower() == "none":
            results = results.remove_metadata()
        elif return_metadata.lower() != "full":
            raise ValueError(f"Invalid return_metadata: '{return_metadata}'")

        dumps = results.dump()
        context = "==== Result 1 ====\n\n"
        context += dumps[0]
        for i, dump in enumerate(dumps[1:], start=2):
            context += "\n\n==== Result " + str(i) + " ====\n\n"
            context += dump
        if not context:
            context = "No results found"
        if max_tokens is not None:
            tokens = tokenize(context)
            if len(tokens) > max_tokens:
                info_tokens = tokenize("\n\n... Truncated to fit max_tokens limit.")
                context = detokenize(tokens[: max_tokens - len(info_tokens)])
                context += detokenize(info_tokens)
        return context


def find(
    refname: tp.Union[str, tp.List[str]],
    resolve: bool = True,
    asset_names: tp.Union[str, tp.List[str]] = "all",
    aggregate_api: bool = False,
    aggregate_messages: bool = False,
    return_metadata: str = "none",
    max_tokens: tp.Optional[int] = 5_000,
    n: tp.Optional[int] = None,
    page: int = 1,
) -> str:
    """Find VectorBT PRO (vectorbtpro, VBT) assets relevant to a specific object and
    return the results as a context string.

    This can be used to find assets mentioning specific VBT objects, such as modules, classes,
    functions, and instances. For example, searching for "Portfolio" will generate
    targets such as `vbt.Portfolio`, `Portfolio(...)`, `pf = ...`, etc.

    If any of the mentioned targets are found in an asset, it will be returned.

    Args:
        refname (Union[str, List[str]]): One or more references to the object(s).

            A reference can be a fully-qualified dotted name (e.g., "vectorbtpro.data.base.Data")
            or a short name (e.g., "Data", "vbt.Portfolio") that uniquely identifies the object.

            If multiple references are provided, returns a code example if any of the references
            are found in the code example.
        resolve (bool): Whether to resolve the object's reference name.

            Set to False to find any string, not just VBT objects, such as "SQLAlchemy".
            In this case, `refname` becomes a simple string to match against.
            Defaults to True.
        asset_names (Union[str, List[str]]): One or more asset names to search. Supported names:

            * "api": API reference.
            * "docs": Regular documentation, including getting started, features, tutorials, guides,
                recipes, and legal information.
            * "messages": Discord messages and discussions.
            * "examples": Code examples across all assets.
            * "all": All of the above, in the order specified above.

            Do not use "examples" with other assets, as examples are already included in those assets.
            Use them separately to get only examples.

            Order matters. May also include ellipsis. For example, `["messages", "..."]` puts
            "messages" at the beginning and all other assets in their usual order at the end.

            Defaults to "all".
        aggregate_api (bool): Whether to aggregate all children of the object into a single context.

            If True, the context will contain all the children of the object, such as methods,
            properties, and attributes, in a single context string. Note that this might result in
            a large context string, especially for modules and classes. If False, the context
            will contain only the object description.

            Applies only to API documentation. Defaults to False.
        aggregate_messages (bool): Whether to aggregate messages belonging to the same thread (question-reply chain).

            If True, finding an object in a message will return the question and all replies
            in a single context string, not just the isolated message containing the object.

            Applies only to Discord messages. Defaults to False.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as title and URL.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "none".
        max_tokens (Optional[int]): Maximum number of tokens to return in the results.

            Defaults to 5,000.
        n (Optional[int]): Number of results to return per page.

            If specified, it will return the first `n` results that fit within the `max_tokens` limit.
            If None, returns all results that fit within the `max_tokens` limit. Defaults to None.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.

    Returns:
        str: Context string containing the search results.
    """
    from vectorbtpro.utils.knowledge.chatting import detokenize, tokenize
    from vectorbtpro.utils.knowledge.custom_assets import find_assets
    from vectorbtpro.utils.pbar import ProgressHidden

    with ProgressHidden():
        refname = auto_cast(refname)
        resolve = auto_cast(resolve)
        asset_names = auto_cast(asset_names)
        aggregate_api = auto_cast(aggregate_api)
        aggregate_messages = auto_cast(aggregate_messages)
        return_metadata = auto_cast(return_metadata)
        max_tokens = auto_cast(max_tokens)
        n = auto_cast(n)
        page = auto_cast(page)

        results = find_assets(
            refname,
            resolve=resolve,
            asset_names=asset_names,
            api_kwargs=dict(
                only_obj=True,
                aggregate=aggregate_api,
            ),
            docs_kwargs=dict(
                aggregate=False,
                up_aggregate=False,
            ),
            messages_kwargs=dict(
                aggregate="threads" if aggregate_messages else "messages",
                latest_first=True,
            ),
            examples_kwargs=dict(
                return_type="match" if return_metadata.lower() == "none" else "item",
                latest_first=True,
            ),
            minimize=False,
        )
        if n is not None:
            if page < 1:
                raise ValueError("Page number must be greater than or equal to 1")
            results = results[(page - 1) * n : page * n]
        elif page > 1:
            raise ValueError("Page number must be 1 when n is not specified")
        if return_metadata.lower() == "minimal":
            results = results.minimize_metadata()
        elif return_metadata.lower() == "none":
            results = results.remove_metadata()
        elif return_metadata.lower() != "full":
            raise ValueError(f"Invalid return_metadata: '{return_metadata}'")

        dumps = results.dump()
        context = "==== Result 1 ====\n\n"
        context += dumps[0]
        for i, dump in enumerate(dumps[1:], start=2):
            context += "\n\n==== Result " + str(i) + " ====\n\n"
            context += dump
        if not context:
            context = "No results found"
        if max_tokens is not None:
            tokens = tokenize(context)
            if len(tokens) > max_tokens:
                info_tokens = tokenize("\n\n... Truncated to fit max_tokens limit.")
                context = detokenize(tokens[: max_tokens - len(info_tokens)])
                context += detokenize(info_tokens)
        return context


def get_attrs(refname: str, own_only: bool = False, incl_private: bool = False) -> str:
    """Get a list of attributes of an object (similar to `dir()`) with their types and reference names.

    Can be used to discover the API of VectorBT PRO (vectorbtpro, VBT). For example, use it to
    find out what methods and properties are available on a specific class, or to explore the
    objects defined in a module.

    Each line is formatted as `<name> [<type>] (@ <refname>)`, where the `@ <refname>` suffix
    is shown only when the attribute is not defined directly on the object.

    Args:
        refname (str): Reference to the object.

            A reference can be a fully-qualified dotted name (e.g., "vectorbtpro.data.base.Data")
            or a short name (e.g., "Data", "vbt.Portfolio") that uniquely identifies the object.

            Pass "vbt" to get all the attributes of the `vectorbtpro` module.
        own_only (bool): If True, include only attributes that are defined directly on the object
            (i.e., attributes defined elsewhere, such as inherited attributes, will be excluded).
        incl_private (bool): If True, include private attributes (those starting with an underscore).

    Returns:
        str: String containing the list of attributes, each on a new line.
    """
    from vectorbtpro.utils.attr_ import get_attrs
    from vectorbtpro.utils.module_ import get_refname_obj, resolve_refname

    refname = auto_cast(refname)
    resolved_refname = resolve_refname(refname)
    if not resolved_refname:
        raise ValueError(f"Reference name '{refname}' cannot be resolved to an object")
    obj = get_refname_obj(resolved_refname)
    df = get_attrs(obj=obj, own_only=own_only, incl_private=incl_private)

    display_lines = []
    for attr_name, attr_type, attr_refname in df.itertuples():
        line = attr_name
        if attr_type != "?":
            line += f" [{attr_type}]"
        if attr_refname != "?" and attr_refname != resolved_refname + "." + attr_name:
            line += f" @ {attr_refname}"
        display_lines.append(line)
    return "\n".join(display_lines)


def get_source(refname: tp.Union[str, tp.List[str]]) -> str:
    """Get the source code of any object.

    This can be used to inspect the implementation of VectorBT PRO (vectorbtpro, VBT) objects,
    such as modules, classes, functions, and instances. It uses AST parsing to retrieve the source code
    of any object, including named tuples, class variables, dataclasses, and other objects that
    may not have a traditional source code representation.

    Args:
        refname (Union[str, List[str]]): One or more references to the object(s).

            A reference can be a fully-qualified dotted name (e.g., "vectorbtpro.data.base.Data")
            or a short name (e.g., "Data", "vbt.Portfolio") that uniquely identifies the object.

    Returns:
        str: Source code of the object.

            Multiple references can be provided, in which case the source code of each object
            is concatenated together, separated by two newlines.
    """
    from vectorbtpro.utils.module_ import resolve_refname
    from vectorbtpro.utils.source import get_source

    refname = auto_cast(refname)
    if isinstance(refname, str):
        refname = [refname]

    sources = []
    for name in refname:
        resolved_name = resolve_refname(name)
        if not resolved_name:
            raise ValueError(f"Reference name '{name}' cannot be resolved to an object")
        sources.append(get_source(resolved_name))
    return "\n\n".join(sources)


current_kernel = None
"""Currently running Jupyter kernel for executing code snippets."""


def run_code(code: str, restart: bool = False, exec_timeout: tp.Optional[float] = None) -> str:
    """Run a code snippet.

    This spins up a Jupyter kernel if it is not already running, and automatically imports
    `from vectorbtpro import *`, which includes `vbt`, `pd` (Pandas), `np` (NumPy), `njit` (from Numba),
    and other commonly used modules from the documentation. Running this the second time will
    reuse the existing kernel and all variables defined earlier.

    Use this tool to develop and test code snippets in the VectorBT PRO (vectorbtpro, VBT) environment,
    similar to a Jupyter notebook. You can backtest a strategy, debug a function, or explore data interactively.

    !!! note
        VBT is centered around easy-to-use APIs and high-performance computing, thus you should
        put priority on discovering and using VBT APIs. Before running this tool, use other MCP tools to
        search for relevant information, such as existing code examples, API references, and documentation.
        This will help you understand how to use VBT effectively and avoid reinventing the wheel.
        If a custom implementation is needed, consider extending existing VBT functionality or
        using high-performance libraries such as Numba.

    !!! warning
        Ensure that the code is safe, as this tool can execute arbitrary code in the current environment.
        Do not run code that has side effects, such as installing new dependencies, modifying global state,
        or performing I/O operations, unless explicitly granted permission!

        Be aware of long-running and resource-intensive code snippets, as they may block the kernel
        and prevent further execution. Use minimal code snippets whenever possible.

        Use this tool for development and testing purposes related to VBT only.

    Args:
        code (str): Code snippet to run.
        restart (bool): Whether to restart the kernel before running the code.

            Defaults to False.
        exec_timeout (Optional[float]): Timeout for the code execution in seconds.

            None means no timeout. Defaults to None.

    Returns:
        str: Output of the executed code.
    """
    from vectorbtpro.utils.eval_ import VBTKernel

    code = auto_cast(code)
    restart = auto_cast(restart)
    exec_timeout = auto_cast(exec_timeout)

    global current_kernel
    if current_kernel is None:
        current_kernel = VBTKernel()
        current_kernel.start()
    if restart:
        current_kernel.restart()
    return current_kernel.execute(code, exec_timeout=exec_timeout)


def main() -> None:
    """Run the MCP server for VectorBT PRO."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("mcp")
    from mcp.server.fastmcp import FastMCP

    parser = argparse.ArgumentParser(description="Run the MCP server for VectorBT PRO.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    args = parser.parse_args()

    mcp = FastMCP("VectorBT PRO")
    mcp.tool()(search)
    mcp.tool()(find)
    mcp.tool()(get_attrs)
    mcp.tool()(get_source)
    mcp.tool()(run_code)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
