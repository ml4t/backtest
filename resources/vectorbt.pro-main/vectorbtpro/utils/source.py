# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with source."""

import ast
import importlib
import inspect
import os
import re
import tempfile
import textwrap
import webbrowser
from collections import defaultdict
from difflib import HtmlDiff, SequenceMatcher
from pathlib import Path
from types import FunctionType, ModuleType

from vectorbtpro import _typing as tp
from vectorbtpro.utils.checks import is_complex_iterable, is_numba_func
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.formatting import dump, get_dump_language
from vectorbtpro.utils.knowledge.chatting import completed, split_text
from vectorbtpro.utils.knowledge.custom_assets import search
from vectorbtpro.utils.module_ import assert_can_import
from vectorbtpro.utils.path_ import check_mkdir, get_common_prefix
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.template import CustomTemplate, RepEval

__all__ = [
    "cut_and_save_module",
    "cut_and_save_func",
    "refactor_source",
    "refactor_docstrings",
    "refactor_markdown",
    "refactor_docs",
]


def get_source(
    qualname: str, *, clean_indent: bool = True, return_meta: bool = False
) -> tp.Union[str, dict]:
    """Return the exact source that defines *qualname* by parsing the module AST.

    Handles modules, classes, (async) functions, methods, top-level assignments
    and annotated class variables (together with their inline docstrings).

    Args:
        qualname: Fully-qualified dotted name (e.g. "pkg.mod.Class.attr").
        clean_indent: If True, the leading indent of the snippet is removed and
            the block is dedented to column 0.
        return_meta: If True, a dictionary with the code and its file/line metadata
            is returned instead of plain code.

    Returns:
        * When `return_meta=False`, returns the source text as `str`.
        * When `return_meta=True`, returns a dictionary with the code, file path,
            start line, and end line (both 1-based).

    Raises:
        ImportError: No component of *qualname* can be imported.
        FileNotFoundError: The discovered module has no readable *.py* file.
        ValueError: The object cannot be located in the module AST.
    """

    def _find(scope, chain):
        if not chain:
            return scope
        name = chain[0]
        for child in getattr(scope, "body", []):
            if (
                isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == name
            ):
                return child if len(chain) == 1 else _find(child, chain[1:])
        for child in getattr(scope, "body", []):
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    return child if len(chain) == 1 else None
        return None

    def _strip(lines, spaces):
        if spaces == 0:
            return lines
        pad = " " * spaces
        return [l[len(pad) :] if l.startswith(pad) else l for l in lines]

    parts = qualname.split(".")
    module = None
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            chain = parts[i:]
            break
        except ModuleNotFoundError:
            continue
    if module is None:
        raise ImportError(f"Could not import any part of '{qualname}'")

    filepath = inspect.getsourcefile(module) or inspect.getfile(module)
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"No pure-Python source for '{module.__name__}'")

    with open(filepath, encoding="utf-8") as fh:
        source = fh.read()

    if not chain:
        code = source if not clean_indent else textwrap.dedent(source)
        if return_meta:
            return {
                "code": code,
                "file": filepath,
                "start_line": 1,
                "end_line": len(source.splitlines()),
            }
        return code

    tree = ast.parse(source, filename=filepath)
    target = _find(tree, chain)
    if target is None:
        raise ValueError(f"Could not locate '{'.'.join(chain)}' in '{module.__name__}'")

    if (
        isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and target.decorator_list
    ):
        start = min(d.lineno for d in target.decorator_list) - 1
    else:
        start = target.lineno - 1

    end = getattr(target, "end_lineno", len(source.splitlines()))
    if isinstance(target, (ast.Assign, ast.AnnAssign)):

        def _body(root, needle):
            if hasattr(root, "body"):
                for itm in root.body:
                    if itm is needle:
                        return root.body
                    sub = _body(itm, needle)
                    if sub is not None:
                        return sub
            return None

        body = _body(tree, target)
        if body is not None:
            idx = body.index(target)
            if idx + 1 < len(body):
                nxt = body[idx + 1]
                if (
                    isinstance(nxt, ast.Expr)
                    and isinstance(getattr(nxt, "value", None), ast.Constant)
                    and isinstance(nxt.value.value, str)
                ):
                    end = max(end, getattr(nxt, "end_lineno", nxt.lineno))

    lines = source.splitlines(True)[start:end]
    if clean_indent:
        lines = _strip(lines, getattr(target, "col_offset", 0) or 0)
        snippet = textwrap.dedent("".join(lines))
    else:
        snippet = "".join(lines)

    if return_meta:
        return {"code": snippet, "file": filepath, "start_line": start + 1, "end_line": end}
    return snippet


def collect_blocks(lines: tp.Iterable[str]) -> tp.Dict[str, tp.List[str]]:
    """Collect block sections from source code lines.

    Scans through the provided lines and groups lines into blocks defined by
    markers starting with `# % <block block_name>` and ending with `# % </block>`.

    Args:
        lines (Iterable[str]): Lines of source code.

    Returns:
        Dict[str, List[str]]: Mapping from block names to lists of lines for each block.
    """
    blocks = {}
    block_name = None

    for line in lines:
        sline = line.strip()

        if sline.startswith("# % <block") and sline.endswith(">"):
            block_name = sline[len("# % <block") : -1].strip()
            if len(block_name) == 0:
                raise ValueError("Missing block name")
            blocks[block_name] = []
        elif sline.startswith("# % </block>"):
            block_name = None
        elif block_name is not None:
            blocks[block_name].append(line)

    return blocks


def cut_from_source(
    source: str,
    section_name: str,
    prepend_lines: tp.Optional[tp.Iterable[str]] = None,
    append_lines: tp.Optional[tp.Iterable[str]] = None,
    out_lines_callback: tp.Union[None, tp.Callable, CustomTemplate] = None,
    return_lines: bool = False,
    **kwargs,
) -> tp.Union[str, tp.List[str]]:
    """Extract an annotated section from the source code.

    Processes the source code string to extract a section defined by markers. The section is delimited
    by a starting marker `# % <section section_name>` and an ending marker `# % </section>`. Within the
    section, block subsections can be defined using markers `# % <block block_name>` and `# % </block>`.

    The function also handles skip and uncomment operations:

    * Lines between `# % <skip [expression]>` and `# % </skip>` are omitted.
    * Lines between `# % <uncomment [expression]>` and `# % </uncomment>` have their comment prefix removed.

    Any line containing `# %` outside these blocks is interpreted as a Python expression. The evaluation
    result of the expression directs the output as follows:

    * None: Skip the line.
    * `str`: Insert a single line.
    * `Iterable[str]`: Insert multiple lines into the output.

    Args:
        source (str): Python source code.
        section_name (str): Name of the section to extract.
        prepend_lines (Optional[Iterable[str]]): Lines to prepend to the extracted section.
        append_lines (Optional[Iterable[str]]): Lines to append to the extracted section.
        out_lines_callback (Union[None, Callable, CustomTemplate]): Callback or template
            to process the output lines.
        return_lines (bool): If True, return the output as a list of lines.
        **kwargs: Additional context variables for expression evaluation.

    Returns:
        Union[str, List[str]]: Processed section as a cleaned string, or as a list
            of lines if `return_lines` is True.
    """
    lines = source.split("\n")
    blocks = collect_blocks(lines)

    out_lines = []
    if prepend_lines is not None:
        out_lines.extend(list(prepend_lines))
    section_found = False
    uncomment = False
    skip = False
    i = 0

    while i < len(lines):
        line = lines[i]
        sline = line.strip()

        if sline.startswith("# % <section") and sline.endswith(">"):
            if section_found:
                raise ValueError("Missing </section>")
            found_name = sline[len("# % <section") : -1].strip()
            if len(found_name) == 0:
                raise ValueError("Missing section name")
            section_found = found_name == section_name
        elif section_found:
            context = {
                "lines": lines,
                "blocks": blocks,
                "section_name": section_name,
                "line": line,
                "out_lines": out_lines,
                **kwargs,
            }
            if sline.startswith("# % </section>"):
                if append_lines is not None:
                    out_lines.extend(list(append_lines))
                if out_lines_callback is not None:
                    if isinstance(out_lines_callback, CustomTemplate):
                        out_lines_callback = out_lines_callback.substitute(
                            context=context, strict=True
                        )
                    out_lines = out_lines_callback(out_lines)
                if return_lines:
                    return out_lines
                return inspect.cleandoc("\n".join(out_lines))
            if sline.startswith("# % <skip") and sline.endswith(">"):
                if skip:
                    raise ValueError("Missing </skip>")
                expression = sline[len("# % <skip") : -1].strip()
                if len(expression) == 0:
                    skip = True
                else:
                    if expression.startswith("?"):
                        expression = expression[1:]
                        strict = False
                    else:
                        strict = True
                    eval_skip = RepEval(expression).substitute(context=context, strict=strict)
                    if not isinstance(eval_skip, RepEval):
                        skip = eval_skip
            elif sline.startswith("# % </skip>"):
                skip = False
            elif not skip:
                if sline.startswith("# % <uncomment") and sline.endswith(">"):
                    if uncomment:
                        raise ValueError("Missing </uncomment>")
                    expression = sline[len("# % <uncomment") : -1].strip()
                    if len(expression) == 0:
                        uncomment = True
                    else:
                        if expression.startswith("?"):
                            expression = expression[1:]
                            strict = False
                        else:
                            strict = True
                        eval_uncomment = RepEval(expression).substitute(
                            context=context, strict=strict
                        )
                        if not isinstance(eval_uncomment, RepEval):
                            uncomment = eval_uncomment
                elif sline.startswith("# % </uncomment>"):
                    uncomment = False
                elif "# %" in line:
                    expression = line.split("# %")[1].strip()
                    if expression.startswith("?"):
                        expression = expression[1:]
                        strict = False
                    else:
                        strict = True
                    line_woc = line.split("# %")[0].rstrip()
                    context["line"] = line_woc
                    eval_line = RepEval(expression).substitute(context=context, strict=strict)
                    if eval_line is not None:
                        if not isinstance(eval_line, RepEval):
                            if isinstance(eval_line, str):
                                out_lines.append(eval_line)
                            else:
                                lines[i + 1 : i + 1] = eval_line
                        else:
                            out_lines.append(line)
                elif uncomment:
                    if sline.startswith("# "):
                        out_lines.append(sline[2:])
                    elif sline.startswith("#"):
                        out_lines.append(sline[1:])
                    else:
                        out_lines.append(line)
                else:
                    out_lines.append(line)

        i += 1
    if section_found:
        raise ValueError(f"Code section '{section_name}' not closed")
    raise ValueError(f"Code section '{section_name}' not found")


def suggest_module_path(
    section_name: str,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
) -> Path:
    """Suggest a file path for the target module.

    Determines a suitable file path using the given section name and an optional base path.
    If the provided path is a directory or lacks a file extension, uses the section name to form a filename
    with a `.py` extension. This function also ensures that the target directory exists.

    Args:
        section_name (str): Name of the section.
        path (Optional[PathLike]): Base directory or file path.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.

    Returns:
        Path: Determined file path.
    """
    if path is None:
        path = Path(".")
    else:
        path = Path(path)
    if not path.is_file() and path.suffix == "":
        path = (path / section_name).with_suffix(".py")
    if mkdir_kwargs is None:
        mkdir_kwargs = {}
    check_mkdir(path.parent, **mkdir_kwargs)
    return path


def cut_and_save(
    source: str,
    section_name: str,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> Path:
    """Extract an annotated section from the source code and save it to a file.

    Extracts a section, identified by an annotation, from the given source code using `cut_from_source`
    and saves it to a file determined by `suggest_module_path`.

    Args:
        source (str): Python source code.
        section_name (str): Name of the section to extract.
        path (Optional[PathLike]): File path or directory in which to save the extracted section.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        **kwargs: Keyword arguments for `cut_from_source`.

    Returns:
        Path: File path where the extracted section is saved.
    """
    parsed_source = cut_from_source(source, section_name, **kwargs)
    path = suggest_module_path(section_name, path=path, mkdir_kwargs=mkdir_kwargs)
    with open(path, "w") as f:
        f.write(parsed_source)
    return path


def cut_and_save_module(module: tp.Union[str, ModuleType], *args, **kwargs) -> Path:
    """Extract an annotated section from a module's source code and save it to a file.

    If a module is provided as an import path string, it is imported prior to processing.
    The source code is retrieved using `inspect.getsource`, after which the specified section
    is extracted and saved using `cut_and_save`.

    Args:
        module (Union[str, ModuleType]): Target module or its import path.
        *args: Positional arguments for `cut_and_save`.
        **kwargs: Keyword arguments for `cut_and_save`.

    Returns:
        Path: File path where the extracted module section is saved.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)
    source = inspect.getsource(module)
    return cut_and_save(source, *args, **kwargs)


def cut_and_save_func(func: tp.Union[str, FunctionType], *args, **kwargs) -> Path:
    """Extract a function's annotated section from its module and save it to a file.

    If `func` is provided as a fully qualified name string, the containing module is imported and
    the function is retrieved before extraction. The source code is then obtained and processed
    using `cut_and_save`.

    Args:
        func (Union[str, FunctionType]): Function or its fully qualified name.

            If provided as a string, the module will be imported and the function will be retrieved.
        *args: Positional arguments for `cut_and_save`.
        **kwargs: Keyword arguments for `cut_and_save`.

    Returns:
        Path: File path where the extracted function section is saved.
    """
    if isinstance(func, str):
        module = importlib.import_module(".".join(func.split(".")[:-1]))
        func = getattr(module, func.split(".")[-1])
    else:
        module = inspect.getmodule(func)
    source = inspect.getsource(module)
    return cut_and_save(source, section_name=func.__name__, *args, **kwargs)


def get_source_indent(source: str) -> int:
    """Return the minimum indentation, in spaces, of all non-empty lines in the source code.

    Tabs are treated as 4 spaces.

    Args:
        source (str): Python source code.

    Returns:
        int: Minimum indentation in spaces.
    """
    lines = source.splitlines(keepends=True)
    indentations = []
    for line in lines:
        if line.strip():
            line_expanded = line.replace("\t", " " * 4)
            match = re.match(r"^( *)", line_expanded)
            if match:
                indentations.append(len(match.group(1)))
    return min(indentations) if indentations else 0


def remove_source_indent(source: str, indent: int) -> str:
    """Remove a fixed number of leading spaces from all non-empty lines in the source code.

    Tabs are treated as 4 spaces.

    Args:
        source (str): Python source code.
        indent (int): Number of leading spaces to remove from each non-empty line.

    Returns:
        str: Source code with the specified indentation removed.
    """
    dedented_lines = []
    for line in source.splitlines(keepends=True):
        line_expanded = line.replace("\t", " " * 4)
        if line.strip():
            dedented_lines.append(line_expanded[indent:])
        else:
            dedented_lines.append(line_expanded)
    return "".join(dedented_lines)


def add_source_indent(source: str, indent: int) -> str:
    """Add spaces to each non-empty line in a source string.

    Args:
        source (str): Python source code.
        indent (int): Number of leading spaces to add to each non-empty line.

    Returns:
        str: Resulting source code with added indentation.
    """
    indent_str = " " * indent
    indented_lines = []
    for line in source.splitlines(keepends=True):
        if line.strip():
            indented_lines.append(indent_str + line)
        else:
            indented_lines.append(line)
    return "".join(indented_lines)


def get_source_imports(source: str, global_only: bool = False) -> str:
    """Extract, normalize, deduplicate, and sort import statements from the source code.

    Args:
        source (str): Python source code.
        global_only (bool): If True, only extract top-level (global) import statements.

    Returns:
        str: Sorted string of normalized import statements, separated by newlines.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    imports = set()

    def _process_import_node(node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imports.add(f"import {alias.name} as {alias.asname}")
                else:
                    imports.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = "." * node.level
            for alias in node.names:
                if alias.asname:
                    imports.add(f"from {level}{module} import {alias.name} as {alias.asname}")
                else:
                    imports.add(f"from {level}{module} import {alias.name}")

    if global_only:
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _process_import_node(node)
    else:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _process_import_node(node)

    return "\n".join(sorted(imports)) if imports else ""


def get_source_map(source: str) -> dict:
    """Generate a high-level map of top-level variables, functions, and classes defined in the source code.

    Args:
        source (str): Python source code.

    Returns:
        dict: Dictionary summarizing the top-level code structure.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    code_map = {
        "variables": [],
        "functions": [],
        "classes": defaultdict(lambda: {"methods": [], "attributes": []}),
    }

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    code_map["variables"].append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                code_map["variables"].append(node.target.id)
        elif isinstance(node, ast.FunctionDef):
            code_map["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            for class_body_item in node.body:
                if isinstance(class_body_item, ast.FunctionDef):
                    code_map["classes"][class_name]["methods"].append(class_body_item.name)
                elif isinstance(class_body_item, (ast.Assign, ast.AnnAssign)):
                    targets = []
                    if isinstance(class_body_item, ast.Assign):
                        targets = class_body_item.targets
                    elif isinstance(class_body_item, ast.AnnAssign):
                        targets = [class_body_item.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            code_map["classes"][class_name]["attributes"].append(target.id)

    code_map["classes"] = dict(code_map["classes"])
    return code_map


LINE_NUMBER_RE = re.compile(r"^(\d+):\s?", re.MULTILINE)
"""Regular expression to match line numbers."""


def add_line_numbers(source: str, start_line: int = 1) -> str:
    """Prefix every line in the source with `N: `.

    Args:
        source (str): Source to which line numbers will be added.
        start_line (int): Starting line number for the source (1-based).

    Returns:
        str: Source with each line prefixed by its line number.
    """
    lines = source.splitlines(keepends=True)
    out = []
    for idx, line in enumerate(lines, start_line):
        out.append(f"{idx}: " + line)
    return "".join(out)


def remove_line_numbers(source: str) -> str:
    """Remove leading `N: ` from every line, if present.

    Args:
        source (str): Source from which line numbers will be removed.

    Returns:
        str: Source with line numbers removed.
    """
    return LINE_NUMBER_RE.sub("", source)


def find_source(target: str, source: str, start_line: int = 1) -> tp.List[tp.Tuple[int, int]]:
    """Find all occurrences of a target string in the source.

    All lines in the source and target are stripped of leading and trailing whitespace,
    and all empty lines are ignored.

    Args:
        target (str): Target string to search for in the source.
        source (str): Source string in which to search for the target.
        start_line (int): Starting line number for the source (1-based).

    Returns:
        List[Tuple[int, int]]: List of tuples where each tuple contains the start and end line numbers
            (1-based) of the found occurrences of the target in the source.
    """
    tgt = [ln.strip() for ln in target.splitlines() if ln.strip()]
    if not tgt:
        return []
    src = [(i + start_line, ln.strip()) for i, ln in enumerate(source.splitlines()) if ln.strip()]
    res = []
    t_len = len(tgt)
    s_len = len(src)
    i = 0
    while i <= s_len - t_len:
        if all(src[i + k][1] == tgt[k] for k in range(t_len)):
            res.append((src[i][0], src[i + t_len - 1][0]))
            i += t_len
        else:
            i += 1
    return res


def find_source_fuzzy(
    target: str,
    source: str,
    start_line: int = 1,
    threshold: float = 0.8,
) -> tp.List[tp.Tuple[int, int]]:
    """Find all occurrences of a target string in the source using fuzzy search.

    All lines in the source and target are stripped of leading and trailing whitespace,
    and all empty lines are ignored.

    Uses `rapidfuzz` if available, otherwise falls back to `difflib.SequenceMatcher`.

    Args:
        target (str): Target string to search for in the source.
        source (str): Source string in which to search for the target.
        start_line (int): Starting line number for the source (1-based).

    Returns:
        List[Tuple[int, int]]: List of tuples where each tuple contains the start and end line numbers
            (1-based) of the found occurrences of the target in the source.
    """
    from vectorbtpro.utils.module_ import check_installed

    tgt = [ln.strip() for ln in target.splitlines() if ln.strip()]
    if not tgt:
        return []
    src = [(i + start_line, ln.strip()) for i, ln in enumerate(source.splitlines()) if ln.strip()]
    t_len = len(tgt)
    s_len = len(src)

    if check_installed("rapidfuzz"):
        from rapidfuzz import fuzz

        def _ratio(a, b):
            return fuzz.ratio(a, b) / 100

    else:

        def _ratio(a, b):
            return SequenceMatcher(None, a, b).ratio()

    res = []
    i = 0
    while i <= s_len - t_len:
        if all(_ratio(src[i + k][1], tgt[k]) >= threshold for k in range(t_len)):
            res.append((src[i][0], src[i + t_len - 1][0]))
            i += t_len
        else:
            i += 1
    return res


def find_source_fuzzy_window(
    target: str,
    source: str,
    start_line: int = 1,
    threshold: float = 0.8,
) -> tp.List[tp.Tuple[int, int]]:
    """Find all occurrences of a target string in the source using fuzzy search with a sliding window.

    All lines in the source and target are stripped of leading and trailing whitespace,
    and all empty lines are ignored.

    Uses `rapidfuzz` if available, otherwise falls back to `difflib.SequenceMatcher`.

    Args:
        target (str): Target string to search for in the source.
        source (str): Source string in which to search for the target.
        start_line (int): Starting line number for the source (1-based).

    Returns:
        List[Tuple[int, int]]: List of tuples where each tuple contains the start and end line numbers
            (1-based) of the found occurrences of the target in the source.
    """
    from vectorbtpro.utils.module_ import check_installed

    tgt_norm = " ".join(ln.strip() for ln in target.splitlines() if ln.strip())
    if not tgt_norm:
        return []
    win_len = len(tgt_norm)
    src_norm = " ".join(ln.strip() for ln in source.splitlines())
    src_len = len(src_norm)

    if check_installed("rapidfuzz"):
        from rapidfuzz import fuzz

        def _ratio(a, b):
            return fuzz.ratio(a, b) / 100

    else:

        def _ratio(a, b):
            return SequenceMatcher(None, a, b).ratio()

    res = []
    i = 0
    while i <= src_len - win_len:
        window = src_norm[i : i + win_len]
        if _ratio(window, tgt_norm) >= threshold:
            start_char = i
            end_char = i + win_len - 1
            start_ln = source.count("\n", 0, start_char) + start_line
            end_ln = source.count("\n", 0, end_char + 1) + start_line
            res.append((start_ln, end_ln))
            i += win_len
        else:
            i += 1
    return res


ANY_CODE_FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*\n(.*?)\n?```$", re.DOTALL)
"""Regular expression to match code fences around code blocks with any language in the output."""

PATCH_BLOCK_RE = re.compile(
    r"<<<PATCH(?::(\d+))?>>>\s*\r?\n"
    r"<<<<<<< SEARCH\s*\r?\n"
    r"(.*?)\r?\n"
    r"=======\s*\r?\n"
    r"(.*?)\r?\n"
    r">>>>>>> REPLACE\s*\r?\n"
    r"<<<END PATCH>>>",
    re.DOTALL,
)
"""Regular expression to match patch blocks in the output."""

NO_PATCHES_SENTINEL = "<<<NO PATCHES>>>"
"""Sentinel value indicating that no patch blocks are in the output."""


def apply_patches(
    source: str, patch_output: str, start_line: int = 1, fuzzy_kwargs: tp.KwargsLike = None
) -> str:
    """Apply anchor-style patches to the source; exact match first, fuzzy-search fallback.

    If the output contains no patch blocks (`NO_PATCHES_SENTINEL`), the function returns
    the original text unchanged.

    Args:
        source (str): Original source to which patches will be applied.
        patch_output (str): Output containing patch blocks.
        start_line (int): Starting line number for the source (1-based).
        fuzzy_kwargs (KwargsLike): Keyword arguments for `vectorbtpro.utils.search_.find_fuzzy`.

    Returns:
        str: Source with applied patches.

    !!! note
        Source should not contain line numbers.
    """
    m = ANY_CODE_FENCE_RE.match(patch_output.strip())
    if m:
        patch_output = m.group(1)
    if patch_output.strip() == NO_PATCHES_SENTINEL:
        return source
    blocks = []
    for ln, search_bl, replace_bl in PATCH_BLOCK_RE.findall(patch_output):
        ln = int(ln) if ln else None
        if LINE_NUMBER_RE.match(search_bl):
            search_bl = remove_line_numbers(search_bl)
        if LINE_NUMBER_RE.match(replace_bl):
            replace_bl = remove_line_numbers(replace_bl)
        if search_bl and not search_bl.endswith(("\n", "\r")):
            search_bl += "\n"
        if replace_bl and not replace_bl.endswith(("\n", "\r")):
            replace_bl += "\n"
        blocks.append((ln, search_bl, replace_bl))
    if not blocks:
        raise ValueError("No PATCH blocks found")
    if fuzzy_kwargs is None:
        fuzzy_kwargs = {}

    if all(ln is not None for ln, _, _ in blocks):
        if all(ln < start_line for ln, _, _ in blocks):
            blocks = [
                (ln + start_line - 1, search_bl, replace_bl) for ln, search_bl, replace_bl in blocks
            ]
        blocks.sort(key=lambda x: x[0])
        lines_present = True
    else:
        lines_present = False

    all_matches = []
    for i, (_, search_bl, replace_bl) in enumerate(blocks):
        matches = find_source(search_bl, source, start_line=start_line)
        if not matches:
            matches = find_source_fuzzy(search_bl, source, start_line=start_line, **fuzzy_kwargs)
        if not matches:
            matches = find_source_fuzzy_window(
                search_bl, source, start_line=start_line, **fuzzy_kwargs
            )
        if len(matches) == 1:
            all_matches.append((*matches[0], replace_bl))
            continue
        if len(matches) > 1 and all_matches:
            found_match = False
            for s, e in matches:
                if s > all_matches[-1][1]:
                    found_match = True
                    all_matches.append((s, e, replace_bl))
                    break
            if found_match:
                continue
            if not lines_present:
                all_matches.append((*matches[0], replace_bl))
        raise ValueError(f"Original block not found:\n{search_bl}")

    if not lines_present:
        all_matches.sort(key=lambda x: x[0])
    for s, e, replace_bl in reversed(all_matches):
        source_lines = source.splitlines(keepends=True)
        repl_lines = replace_bl.splitlines(keepends=True)
        source_lines[s - start_line : e - start_line + 1] = repl_lines
        source = "".join(source_lines)
    return source


DIFF_CODE_FENCE_RE = re.compile(r"^\s*```(?:diff)?\s*\n(.*?)\n?```$", re.DOTALL)
"""Regular expression to match code fences around unified diffs in the output."""

HUNK_HEADER_RE = re.compile(r"^@@[^\n]*\n", re.MULTILINE)
"""Regular expression to split unified diff hunks by their headers."""

HUNK_HEADER_PATH_RE = re.compile(r"^(---|\+\+\+).*?$", re.MULTILINE)
"""Regular expression to match hunk header paths in unified diffs."""

HUNK_HEADER_LO_RE = re.compile(r"^@@\s*-(\d+)")
"""Regular expression to match the original line number in hunk headers."""

HUNK_PREFIXES = {" ": "both", "-": "search_bl", "+": "replace_bl"}
"""Mapping of diff prefixes to their roles in hunks."""

NO_DIFF_SENTINEL = "<<<NO DIFF>>>"
"""Sentinel value indicating that no hunks are in the output."""


def udiff_to_patches(udiff_output: str, start_line: int = 1) -> str:
    """Convert unified diff hunks to the anchor-style patch blocks expected by `apply_patches`.

    Args:
        udiff_output (str): Unified diff output containing hunks.
        start_line (int): Starting line number for the source (1-based).

    Returns:
        str: Anchor-style patch blocks.
    """
    m = DIFF_CODE_FENCE_RE.match(udiff_output.strip())
    if m:
        udiff_output = m.group(1)
    if udiff_output.strip() == NO_DIFF_SENTINEL:
        return NO_PATCHES_SENTINEL
    udiff_output = HUNK_HEADER_PATH_RE.sub("", udiff_output).lstrip()
    if not udiff_output.strip():
        return NO_PATCHES_SENTINEL

    headers = list(HUNK_HEADER_RE.finditer(udiff_output))
    hunks = []
    for idx, hdr in enumerate(headers):
        m = HUNK_HEADER_LO_RE.search(hdr.group(0))
        ln = int(m.group(1)) if m else None
        hunk_start = hdr.end()
        hunk_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(udiff_output)
        hunk_body = udiff_output[hunk_start:hunk_end]
        hunks.append((ln, hunk_body))

    if all(ln is not None for ln, _ in hunks):
        if all(ln < start_line for ln, _ in hunks):
            hunks = [(ln + start_line - 1, hunk) for ln, hunk in hunks]
        hunks.sort(key=lambda x: x[0])

    patch_blocks = []
    for ln, hunk in hunks:
        orig_lines, repl_lines = [], []
        for line in hunk.splitlines(keepends=True):
            if not line:
                continue
            tag, body = line[0], line[1:]
            role = HUNK_PREFIXES.get(tag, "both")
            if role in ("both", "search_bl"):
                orig_lines.append(body)
            if role in ("both", "replace_bl"):
                repl_lines.append(body)
        if orig_lines == repl_lines:
            continue
        search_bl = "".join(orig_lines)
        replace_bl = "".join(repl_lines)
        if search_bl and not search_bl.endswith(("\n", "\r")):
            search_bl += "\n"
        if replace_bl and not replace_bl.endswith(("\n", "\r")):
            replace_bl += "\n"
        patch_blocks.append(
            f"<<<PATCH{f':{ln}' if ln is not None else ''}>>>\n"
            "<<<<<<< SEARCH\n"
            f"{search_bl}"
            "=======\n"
            f"{replace_bl}"
            ">>>>>>> REPLACE\n"
            "<<<END PATCH>>>"
        )

    return "\n".join(patch_blocks) if patch_blocks else NO_PATCHES_SENTINEL


FULL_FORMAT_PROMPT = """\
Emit **only** the complete, modified file—nothing else.

Strict formatting rules
-----------------------
1. **Single stream:**
    * First character you output is the file's first character.
    * Last character is the final newline (if one existed).
    * No prologue, epilogue, or commentary.
2. **No line numbers.** Never insert numeric prefixes.
3. **No fences or back-ticks.** Do not wrap output in ``` or any delimiter.
4. **Line integrity:**
    * Unchanged lines must stay character-for-character identical (whitespace included).
    * Change, insert, or delete **only** what the task demands.
5. **Newline convention:** Mirror the original newline sequence (`\n`, `\r\n`, or `\r`) everywhere.
6. **No-change case:** If nothing changes, reproduce the input exactly—still following every rule.

**Any deviation from these rules is a failure.**"""
"""Default prompt for formatting the output in `refactor_source` when `output_format` is "full"."""

PATCH_FORMAT_PROMPT = """\
Output exactly one of:

* <<<NO PATCHES>>> if no changes are needed, or
* One or more patch blocks, each in this Git-style form:

<<<PATCH:<LN>>>
<<<<<<< SEARCH
<original text EXACTLY as in input>
=======
<replacement text>
>>>>>>> REPLACE
<<<END PATCH>>>

Strict formatting rules
-----------------------
1. <LN> = 1-based line number of the first ORIGINAL line; blocks appear in strictly increasing order.
2. ORIGINAL lines are copied **verbatim**, including their indentation and newline style; use the same style in REPLACEMENT.
3. Start a new block if ≥ 4 unchanged lines separate two edits.
4. Blocks must not overlap or touch the same line twice.
5. **No other output:** No markdown fences, comments, prose, or full-file echo.

**Any deviation from these rules is a failure.**

Example input (line numbers added for clarity):

1: def example():
2:     x = 1
3:     y = 2
4:     sum = x + y
5:     print(sum)
6:
7: def greet():
8:     print("Hello, world!")
9:     return True

Example output:

<<<PATCH:2>>>
<<<<<<< SEARCH
    x = 1
=======
    x = 10
>>>>>>> REPLACE
<<<END PATCH>>>

<<<PATCH:8>>>
<<<<<<< SEARCH
    print("Hello, world!")
=======
    print("Hi there!")
>>>>>>> REPLACE
<<<END PATCH>>>"""
"""Default prompt for formatting the output in `refactor_source` when `output_format` is "patch"."""

UDIFF_FORMAT_PROMPT = """\
Your ONLY permitted output is a valid *unified diff* patch that applies with `patch -p0`.

Strict formatting rules
----------------------
1. The first two lines **must be**:
    --- <old-path>
    +++ <new-path>
    (no timestamps; paths may match)
2. Each hunk header: @@ -<old-start>,<old-len> +<new-start>,<new-len> @@
3. Within each hunk:
    * ' '-prefixed lines are context.
    * '-'-prefixed lines are removals.
    * '+'-prefixed lines are additions.
    * Preserve all whitespace.
4. Start a new hunk if ≥ 4 unchanged lines separate two edits.
5. Show ≤ 3 context lines before/after each hunk (`-U3` behavior).
6. Hunks appear **in the same order** they occur in the input and never overlap.
7. Do **not** abbreviate long diffs; output them completely.
8. Unix LF endings only. Diff ends with a single LF.
9. Do **not** echo the whole file; never wrap the diff in markdown fences.
10. Do **not** add explanations, comments, or prose outside the diff itself.
11. For binary diffs, output **exactly** <<<NO DIFF>>>.
12. If nothing changes, output **exactly** <<<NO DIFF>>>.

**Any deviation from these rules is a failure.**

Example input (line numbers added for clarity):

1: def example():
2:     x = 1
3:     y = 2
4:     sum = x + y
5:     print(sum)
6:
7: def greet():
8:     print("Hello, world!")
9:     return True

Example output:

--- example.py
+++ example.py
@@ -1,5 +1,5 @@
 def example():
-    x = 1
+    x = 10
     y = 2
     sum = x + y
     print(sum)
@@ -6,4 +6,4 @@

 def greet():
-    print("Hello, world!")
+    print("Hi there!")
     return True"""
"""Default prompt for formatting the output in `refactor_source` when `output_format` is "udiff"."""


REFACTOR_SOURCE_PROMPT = """You are a code-refactoring assistant.

1. Your goal is to **address any detected code smells or issues** in the given chunk of Python code.
2. If the code contains a TODO or FIXME comment, **follow the instructions in the comment**.
3. **Do not add any comments** to the code, unless explicitly requested."""
"""Default system prompt for `refactor_source`."""


def refactor_source(
    source: tp.Any,
    *,
    source_name: tp.Optional[str] = None,
    as_package: bool = True,
    glob_pattern: str = "*.py",
    start_line: tp.Optional[int] = None,
    end_line: tp.Optional[int] = None,
    line_numbers: bool = False,
    system_prompt: tp.Optional[str] = None,
    output_format: str = "full",
    fuzzy_kwargs: tp.KwargsLike = None,
    format_prompt: tp.Optional[str] = None,
    context: tp.Optional[str] = None,
    attach_metadata: bool = True,
    attach_imports: tp.Optional[bool] = None,
    attach_map: tp.Optional[bool] = None,
    attach_knowledge: bool = False,
    search_kwargs: tp.KwargsLike = None,
    to_context_kwargs: tp.KwargsLike = None,
    dump_engine: str = "yaml",
    dump_kwargs: tp.KwargsLike = None,
    split: bool = True,
    split_text_kwargs: tp.KwargsLike = None,
    keep_history: tp.Union[bool, int, slice] = False,
    show_progress: tp.Optional[bool] = None,
    pbar_kwargs: tp.KwargsLike = None,
    mult_show_progress: tp.Optional[bool] = None,
    mult_pbar_kwargs: tp.KwargsLike = None,
    modify: bool = False,
    write_chunks: bool = False,
    copy_to_clipboard: bool = False,
    show_diff: bool = False,
    open_browser: bool = True,
    return_path: bool = False,
    **kwargs,
) -> tp.MaybeRefactorSourceOutput:
    """Refactor the source by splitting it into manageable chunks and applying completion methods.

    Args:
        source (Any): Source(s) or object(s) from which to extract the source.

            A source may be:

            * a string containing any text, such as Python code (e.g. "import vectorbtpro as vbt ..."),
            * a file path (e.g. "./strategies/sma_crossover.py"),
            * a directory path (e.g. "./strategies"),
            * a Python object (e.g. `pipeline_nb`),
            * a module name or object (e.g. `vectorbtpro.utils`),
            * a package name or object (e.g. `vectorbtpro`),
            * an iterable of the above.

            When a directory or package is provided, all contained files matching `glob_pattern`
            are processed as separate sources.
        source_name (Optional[str]): Name displayed in the progress bar and/or HTML file name.
        as_package (bool): Whether to process a package as multiple sources.
        glob_pattern (str): Glob pattern for matching files in a directory.
        start_line (Optional[int]): Inclusive starting line number in the source.

            !!! note
                Counting starts at 1.
        end_line (Optional[int]): Inclusive ending line number in the source.

            !!! note
                Counting starts at 1.
        line_numbers (bool): Whether to add line numbers to the source.
        system_prompt (Optional[str]): System prompt that precedes the context prompt.

            This prompt is used to set the system's behavior or context for the conversation.
            If None, defaults to `REFACTOR_SOURCE_PROMPT`.
        output_format (str): Format of the output.

            Can be "full", "patch", or "udiff".
        fuzzy_kwargs (KwargsLike): Keyword arguments for `vectorbtpro.utils.search_.find_fuzzy`.
        format_prompt (Optional[str]): Custom prompt for formatting the output.

            If None, defaults to `FULL_FORMAT_PROMPT`, `PATCH_FORMAT_PROMPT`, or `UDIFF_FORMAT_PROMPT`
            depending on `output_format`.
        context (Optional[str]): Custom context.
        attach_metadata (bool): Whether to attach (dumped) metadata to the context.
        attach_imports (Optional[bool]): Whether to attach global source imports to the context.

            If None, becomes True if `split` is True.
        attach_map (Optional[bool]): Whether to attach (dumped) source map to the context.

            If None, becomes True if `split` is True.
        attach_knowledge (bool): Whether to attach relevant knowledge to the context.
        search_kwargs (KwargsLike): Keyword arguments for searching for knowledge.

            By default, uses the source as the search query and top 20 results.
            See `vectorbtpro.utils.knowledge.custom_assets.search`.
        to_context_kwargs (KwargsLike): Keyword arguments for converting the search results to context.

            See `vectorbtpro.utils.knowledge.custom_assets.VBTAsset.to_context`.
        dump_engine (str): Name of the dump engine.

            See `vectorbtpro.utils.formatting.dump`.
        dump_kwargs (KwargsLike): Keyword arguments for dumping structured data.

            See `vectorbtpro.utils.formatting.dump`.
        split (bool): Whether to split the source into chunks.
        split_text_kwargs (KwargsLike): Keyword arguments for splitting the source.

            By default, uses "python" as `text_splitter`, 2000 as `chunk_size`, and 0 as `chunk_overlap`.
            See `vectorbtpro.utils.knowledge.chatting.split_text`.
        keep_history (Union[bool, int, slice]): Whether to keep the history of the conversation.

            If True, keeps the history of the conversation.
            If an integer, it specifies the number of last messages to keep in the history.
            If a slice, it specifies the range of messages to keep in the history
            (e.g., `slice(1, None, 2)` keeps every completion).
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

            See `vectorbtpro.utils.pbar.ProgressBar`.
        mult_show_progress (Optional[bool]): Whether to display progress during processing multiple sources.

            If not provided, defaults to `show_progress`.
        mult_pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar for multiple sources.

            See `vectorbtpro.utils.pbar.ProgressBar`.
        modify (bool): Whether to update the source file with the refactored source.
        write_chunks (bool): Whether to write each chunk instead of the entire source.
        copy_to_clipboard (bool): Whether to copy the refactored source to the clipboard.

            Does not apply when processing multiple sources.
        show_diff (bool): Whether to generate and display an HTML diff file using `difflib`.

            Does not apply when processing multiple sources.
        open_browser (bool): Whether to open the HTML diff in a web browser.

            Does not apply when processing multiple sources.
        return_path (bool): Whether to return the path to the updated source file or the HTML diff file.
        **kwargs: Keyword arguments for `vectorbtpro.utils.knowledge.chatting.completed`.

    Returns:
        Union[RefactorSourceOutput, RefactorSourceOutputs]: Result of the refactoring process.

            * Returns the refactored source if neither `modify` nor `copy_to_clipboard` is True.
            * Returns the path to the updated source file if `modify` is True.
            * Returns the path to the HTML diff file if `show_diff` is True.
            * For multiple sources, returns a zipped list of sources and their corresponding outputs.
    """
    pbar_kwargs = merge_dicts(
        dict(desc_kwargs=dict(refresh=True)),
        pbar_kwargs,
    )

    if isinstance(source, str):
        try:
            if Path(source).exists():
                source = Path(source)
        except Exception:
            pass
    if isinstance(source, ModuleType) and hasattr(source, "__path__") and as_package:
        package_path = source.__path__
        if isinstance(package_path, list):
            source = [Path(path) for path in package_path]
        else:
            source = Path(package_path)
    if isinstance(source, Path) and source.is_dir():
        source = list(source.rglob(glob_pattern))
    if is_complex_iterable(source):
        sources = source
        new_sources = []
        for source in sources:
            if isinstance(source, Path) and source.is_dir():
                new_sources.extend(list(source.rglob(glob_pattern)))
            else:
                new_sources.append(source)
        sources = new_sources
        source_names = []
        paths = []
        all_paths = True
        for i, source in enumerate(sources):
            if isinstance(source, str):
                try:
                    if Path(source).exists():
                        source = Path(source)
                except Exception:
                    pass
            if isinstance(source, str):
                source_name = "<string>"
                all_paths = False
            elif isinstance(source, Path):
                source_name = source.name
                paths.append(source.resolve())
            else:
                if is_numba_func(source):
                    source = source.py_func
                source_path = inspect.getsourcefile(source)
                if source_path:
                    source_path = Path(source_path)
                if not source_path or not source_path.is_file():
                    raise ValueError(f"Cannot determine a valid source file for object {source}")
                source_name = source_path.name
                paths.append(source_path.resolve())
            source_names.append(source_name)
        if all_paths:
            common_path = Path(get_common_prefix(paths)).resolve()
            same_file = True
            for path in paths:
                if path.relative_to(common_path) != Path():
                    same_file = False
                    break
            if not same_file:
                source_names = [str(path.relative_to(common_path)) for path in paths]

        outputs = []
        if mult_show_progress is None:
            mult_show_progress = show_progress
        if mult_show_progress is None:
            mult_show_progress = True
        mult_pbar_kwargs = merge_dicts(pbar_kwargs, mult_pbar_kwargs)
        with ProgressBar(
            total=len(source_names), show_progress=mult_show_progress, **mult_pbar_kwargs
        ) as pbar:
            for i, source in enumerate(sources):
                pbar.set_description(dict(source=source_names[i]))
                output = refactor_source(
                    source=source,
                    source_name=source_names[i],
                    as_package=False,
                    glob_pattern=glob_pattern,
                    start_line=start_line,
                    end_line=end_line,
                    line_numbers=line_numbers,
                    system_prompt=system_prompt,
                    output_format=output_format,
                    fuzzy_kwargs=fuzzy_kwargs,
                    format_prompt=format_prompt,
                    context=context,
                    attach_metadata=attach_metadata,
                    attach_imports=attach_imports,
                    attach_map=attach_map,
                    attach_knowledge=attach_knowledge,
                    search_kwargs=search_kwargs,
                    to_context_kwargs=to_context_kwargs,
                    dump_engine=dump_engine,
                    dump_kwargs=dump_kwargs,
                    split=split,
                    split_text_kwargs=split_text_kwargs,
                    keep_history=keep_history,
                    show_progress=show_progress,
                    pbar_kwargs=pbar_kwargs,
                    modify=modify,
                    write_chunks=write_chunks,
                    copy_to_clipboard=False,
                    show_diff=False,
                    open_browser=False,
                    return_path=return_path,
                    **kwargs,
                )
                outputs.append(output)
                pbar.update()
        return list(zip(sources, outputs))

    if isinstance(source, str):
        source_path = None
        source_lines = source.splitlines(keepends=True)
        source_start_line = 1
        if source_name is None:
            source_name = "<string>"
    elif isinstance(source, Path):
        source_path = source
        with source_path.open("r", encoding="utf-8") as f:
            source_lines = f.readlines()
        source_start_line = 1
        if source_name is None:
            source_name = source_path.name
    else:
        if is_numba_func(source):
            source = source.py_func
        source_path = inspect.getsourcefile(source)
        if source_path:
            source_path = Path(source_path)
        if not source_path or not source_path.is_file():
            raise ValueError(f"Cannot determine a valid source file for object {source}")
        source_lines, source_start_line = inspect.getsourcelines(source)
        if source_start_line == 0:
            source_start_line = 1
        if source_name is None:
            source_name = source_path.name

    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = len(source_lines)
    start_index = start_line - 1
    end_index = end_line
    source_lines = source_lines[start_index:end_index]

    source_end_line = source_start_line + end_line - 1
    source_start_line = source_start_line + start_line - 1
    source_start_index = source_start_line - 1
    source_end_index = source_end_line
    source = "".join(source_lines)
    source_name = f"{source_name}#L{source_start_line}-L{source_end_line}"

    if system_prompt is None:
        system_prompt = REFACTOR_SOURCE_PROMPT
    if format_prompt is None:
        if output_format.lower() == "full":
            format_prompt = FULL_FORMAT_PROMPT
        elif output_format.lower() == "patch":
            format_prompt = PATCH_FORMAT_PROMPT
        elif output_format.lower() == "udiff":
            format_prompt = UDIFF_FORMAT_PROMPT
        else:
            raise ValueError(f"Invalid output format: '{output_format}'")
    if system_prompt:
        system_prompt += "\n\n====\n\n"
    system_prompt += f"Output format:\n\n{format_prompt}"

    if context is None:
        context = ""
    if dump_kwargs is None:
        dump_kwargs = {}
    if attach_metadata:
        source_metadata = dict(
            source=str(source_path.resolve()) if source_path else "<string>",
            start_line=source_start_line,
            end_line=source_end_line,
        )
        source_metadata = dump(source_metadata, dump_engine=dump_engine, **dump_kwargs).strip()
        source_metadata_language = get_dump_language(dump_engine=dump_engine)
        if context:
            context += "\n\n====\n\n"
        context += f"Metadata of the current code context:\n\n```{source_metadata_language}\n{source_metadata}\n```"
    if attach_imports is None:
        attach_imports = split
    if attach_imports:
        source_imports = get_source_imports(source, global_only=True)
        if source_imports:
            if context:
                context += "\n\n====\n\n"
            context += f"Global imports available to the current code context:\n\n```python\n{source_imports}\n```"
    if attach_map is None:
        attach_map = split
    if attach_map:
        source_map = get_source_map(source)
        if source_map:
            source_map = dump(source_map, dump_engine=dump_engine, **dump_kwargs).strip()
            source_map_language = get_dump_language(dump_engine=dump_engine)
            if context:
                context += "\n\n====\n\n"
            context += f"Top-level objects available to the current code context:\n\n```{source_map_language}\n{source_map}\n```"
    if attach_knowledge:
        if search_kwargs is None:
            search_kwargs = {}
        if to_context_kwargs is None:
            to_context_kwargs = {}
        if "query" not in search_kwargs:
            search_kwargs["query"] = source
        if "top_k" not in search_kwargs:
            search_kwargs["top_k"] = 20
        asset = search(display=False, **search_kwargs)
        if asset:
            if context:
                context += "\n\n====\n\n"
            context += f"Knowledge relevant to the current code context:\n\n{asset.to_context(**to_context_kwargs)}"

    split_text_kwargs = merge_dicts(
        dict(
            text_splitter="python",
            chunk_size=2000,
            chunk_overlap=0,
        ),
        split_text_kwargs,
    )
    if split:
        source_chunks = split_text(source, **split_text_kwargs)
    else:
        source_chunks = [source]

    if source_path and modify:
        with source_path.open("r", encoding="utf-8") as f:
            file_contents = f.readlines()

    processed = []
    chunk_start_line = source_start_line
    chat_history = []
    if show_progress is None:
        show_progress = len(source_chunks) > 1
    with ProgressBar(total=len(source_chunks), show_progress=show_progress, **pbar_kwargs) as pbar:
        for i in range(len(source_chunks)):
            chunk = source_chunks[i]
            chunk_lines = chunk.splitlines(keepends=True)
            pbar.set_description(
                dict(
                    lines=f"{chunk_start_line}..{chunk_start_line + len(chunk_lines) - 1}"
                )
            )
            indent = get_source_indent(chunk)
            chunk = remove_source_indent(chunk, indent)
            leading_len = len(chunk) - len(chunk.lstrip())
            leading = chunk[:leading_len]
            trailing_len = len(chunk) - len(chunk.rstrip())
            trailing = chunk[-trailing_len:] if trailing_len > 0 else ""
            input = chunk[leading_len : len(chunk) - trailing_len]

            if input:
                if line_numbers:
                    num_input = add_line_numbers(input, start_line=chunk_start_line)
                else:
                    num_input = input
                if not keep_history:
                    chat_history = []
                output = completed(
                    num_input,
                    chat_history=chat_history,
                    system_prompt=system_prompt,
                    context=context,
                    **kwargs,
                ).strip("\n")
                if keep_history and not isinstance(keep_history, bool):
                    if isinstance(keep_history, int):
                        keep_history = slice(-keep_history, None)
                    if isinstance(keep_history, slice):
                        chat_history = chat_history[keep_history]
            else:
                output = ""
            if output:
                if output_format.lower() == "patch":
                    print(output)
                    output = apply_patches(
                        input, output, start_line=chunk_start_line, fuzzy_kwargs=fuzzy_kwargs
                    )
                elif output_format.lower() == "udiff":
                    print(output)
                    output = udiff_to_patches(output, start_line=start_line)
                    print(output)
                    output = apply_patches(
                        input, output, start_line=chunk_start_line, fuzzy_kwargs=fuzzy_kwargs
                    )
                elif output_format.lower() != "full":
                    raise ValueError(f"Invalid output format: '{output_format}'")
            output = add_source_indent(output, indent)
            new_chunk = leading + output + trailing
            processed.append(new_chunk)

            if input and source_path and modify and write_chunks:
                new_processed = processed + source_chunks[i + 1 :]
                new_source = "".join(new_processed)
                new_source_lines = new_source.splitlines(keepends=True)
                if new_source_lines and not new_source_lines[-1].endswith("\n"):
                    new_source_lines.append("\n")
                new_file_contents = file_contents.copy()
                new_file_contents[source_start_index:source_end_index] = new_source_lines
                with source_path.open("w", encoding="utf-8") as f:
                    f.writelines(new_file_contents)
            chunk_start_line += len(chunk_lines)

            pbar.update()
    new_source = "".join(processed)

    if source_path and modify and not write_chunks:
        new_source_lines = new_source.splitlines(keepends=True)
        if new_source_lines and not new_source_lines[-1].endswith("\n"):
            new_source_lines.append("\n")
        new_file_contents = file_contents.copy()
        new_file_contents[source_start_index:source_end_index] = new_source_lines
        with source_path.open("w", encoding="utf-8") as f:
            f.writelines(new_file_contents)

    if copy_to_clipboard:
        assert_can_import("pyperclip")
        import pyperclip

        pyperclip.copy(new_source)

    if show_diff:
        differ = HtmlDiff()
        html_diff = differ.make_file(
            source.splitlines(),
            new_source.splitlines(),
            fromdesc="Original",
            todesc="Modified",
        )
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            prefix=re.sub(r"[\W_]+", "_", source_name).strip("_"),
            suffix=".html",
        ) as f:
            f.write(html_diff)
            diff_path = Path(f.name)
        if open_browser:
            webbrowser.open("file://" + str(diff_path.resolve()))
        if modify and source_path:
            if not return_path:
                return None
            return source_path, diff_path
        if copy_to_clipboard:
            if not return_path:
                return None
            return diff_path
        if not return_path:
            return new_source
        return new_source, diff_path

    if modify and source_path:
        if not return_path:
            return None
        return source_path
    if copy_to_clipboard:
        return None
    return new_source


REFACTOR_DOCSTRINGS_PROMPT = """You are a docstring-refactoring assistant.

1. Your goal is to refactor **only** the docstrings of the given chunk of Python code.
2. **Edit docstrings** for clarity, correctness, and consistent format and wording.
3. **Retain all non-docstring parts of the code** exactly as they are.
4. **If the given chunk contains only text, consider it a docstring**."""
"""System prompt for `refactor_docstrings`."""


def refactor_docstrings(source: tp.Any, **kwargs) -> tp.MaybeRefactorSourceOutput:
    """Call `refactor_source` with the system prompt from `REFACTOR_DOCSTRINGS_PROMPT` to refactor docstrings.

    Args:
        source (Any): Source(s) or object(s) from which to extract the Python code.
        **kwargs: Keyword arguments for `refactor_source`.

    Returns:
        RefactorSourceOutput: Result of the refactoring process.
    """
    return refactor_source(source, system_prompt=REFACTOR_DOCSTRINGS_PROMPT, **kwargs)


REFACTOR_MARKDOWN_PROMPT = """You are a Markdown refactoring assistant."""
"""Default system prompt for `refactor_markdown`."""


def refactor_markdown(
    source: tp.Any,
    glob_pattern: str = "*.md",
    system_prompt: tp.Optional[str] = None,
    split_text_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybeRefactorSourceOutput:
    """Call `refactor_source` with the system prompt from `REFACTOR_MARKDOWN_PROMPT` to refactor Markdown.

    Args:
        source (Any): Source(s) or object(s) from which to extract the Markdown.
        glob_pattern (str): Glob pattern for matching files in a directory.
        split_text_kwargs (KwargsLike): Keyword arguments for splitting the source.

            By default, uses "markdown" as `text_splitter`.
            See `vectorbtpro.utils.knowledge.chatting.split_text`.
        **kwargs: Keyword arguments for `refactor_source`.

    Returns:
        RefactorSourceOutput: Result of the refactoring process.
    """
    if system_prompt is None:
        system_prompt = REFACTOR_MARKDOWN_PROMPT
    split_text_kwargs = merge_dicts(
        dict(
            text_splitter="markdown",
        ),
        split_text_kwargs,
    )
    return refactor_source(
        source,
        glob_pattern=glob_pattern,
        system_prompt=system_prompt,
        split_text_kwargs=split_text_kwargs,
        **kwargs,
    )


REFACTOR_DOCS_PROMPT = """You are a Markdown-documentation refactoring assistant.

1. Your goal is to refactor **only** the prose of the given chunk of Markdown-documentation.
2. Edit prose for **clarity, correctness, and consistent format and wording**.
3. **Do not** introduce new ideas, remove existing ideas, or alter facts. **Only rephrase.**
4. **Retain all non-prose parts of the chunk exactly as they are.**
5. **Preserve Markdown structure, list markers, indentation, and blank lines.**"""
"""Default system prompt for `refactor_docs`."""


def refactor_docs(source: tp.Any, **kwargs) -> tp.MaybeRefactorSourceOutput:
    """Call `refactor_markdown` with the system prompt from `REFACTOR_DOCS_PROMPT` to refactor Markdown documentation.

    Args:
        source (Any): Source(s) or object(s) from which to extract the Markdown documentation.
        **kwargs: Keyword arguments for `refactor_markdown`.

    Returns:
        RefactorSourceOutput: Result of the refactoring process.
    """
    return refactor_source(
        source,
        system_prompt=REFACTOR_DOCS_PROMPT,
        **kwargs,
    )
