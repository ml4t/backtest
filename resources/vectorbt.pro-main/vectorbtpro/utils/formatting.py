# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for formatting."""

import inspect
import io
import re

import attr
import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

__all__ = [
    "prettify",
    "format_func",
    "pprint",
    "ptable",
    "phelp",
    "pdir",
    "dump",
]


def camel_to_snake_case(camel_str: str) -> str:
    """Convert a camel case string to a snake case string.

    Args:
        camel_str (str): String formatted in camel case.

    Returns:
        str: String converted to snake case.
    """
    snake_str = re.sub(r"(?<!^)(?<![A-Z_])([A-Z])", r"_\1", camel_str).lower()
    if snake_str.startswith("_"):
        snake_str = snake_str[1:]
    return snake_str


class Prettified(Base):
    """Abstract class for objects that can be prettified."""

    def prettify(self, **kwargs) -> str:
        """Prettify the object.

        Returns:
            str: Prettified representation of the object.

        !!! warning
            Calling `prettify` can lead to an infinite recursion. Make sure to pre-process this object.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def prettify_doc(self, **kwargs) -> str:
        """Prettify the object for documentation, equivalent to using
        `Prettified.prettify` with `repr_doc` as `repr_`.

        Args:
            **kwargs: Keyword arguments for `prettify`.

        Returns:
            str: Prettified representation of the object.
        """
        return self.prettify(repr_=repr_doc, **kwargs)

    def pprint(self, **kwargs) -> None:
        """Pretty-print the object.

        Args:
            **kwargs: Keyword arguments for `prettify`.

        Returns:
            None
        """
        print(self.prettify(**kwargs))

    def __str__(self) -> str:
        try:
            return self.prettify()
        except NotImplementedError:
            return repr(self)


def prettify_inited(
    cls: type,
    kwargs: tp.Any,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
    indent_head: bool = True,
    repr_: tp.Optional[tp.Callable] = None,
) -> str:
    """Prettify an instance initialized with keyword arguments.

    Args:
        cls (type): Class to instantiate for the new instance.
        kwargs (Any): Dictionary of keyword arguments used for initialization.
        replace (DictLike): Mapping for value replacement.
        path (str): Current path in the object hierarchy.
        htchar (str): String used for horizontal indentation.
        lfchar (str): Line feed character.
        indent (int): Current indentation level.
        indent_head (bool): Whether to indent the head line.
        repr_ (Optional[Callable]): Function to get the representation of an object.

            Defaults to `repr`.

    Returns:
        str: Prettified string representation of the initialized instance.
    """

    def _indent_head(content):
        if indent_head:
            return htchar * indent + content
        return content

    def _indent_tail(content):
        return htchar * indent + content

    if repr_ is None:
        repr_ = repr
    if replace is None:
        replace = {}

    items = []
    for k, v in kwargs.items():
        if path is None:
            new_path = k
        else:
            new_path = f"{path}.{k}"
        if new_path in replace:
            new_v = replace[new_path]
        else:
            new_v = prettify(
                v,
                replace=replace,
                path=new_path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1,
                indent_head=False,
                repr_=repr_,
            )
        k_repr = repr(k)
        if isinstance(k, str):
            k_repr = k_repr[1:-1]
        items.append(lfchar + htchar * (indent + 1) + k_repr + "=" + new_v)
    if len(items) == 0:
        return _indent_head(f"{cls.__name__}()")
    return _indent_head(f"{cls.__name__}(") + ",".join(items) + lfchar + _indent_tail(")")


def prettify_dict(
    obj: dict,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
    indent_head: bool = True,
    repr_: tp.Optional[tp.Callable] = None,
) -> str:
    """Prettify a dictionary.

    Args:
        obj (dict): Dictionary to prettify.
        replace (DictLike): Mapping for value replacement.
        path (str): Current path in the object hierarchy.
        htchar (str): String used for horizontal indentation.
        lfchar (str): Line feed character.
        indent (int): Current indentation level.
        indent_head (bool): Whether to indent the head line.
        repr_ (Optional[Callable]): Function to get the representation of an object.

            Defaults to `repr`.

    Returns:
        str: Prettified string representation of the dictionary.
    """

    def _indent_head(content):
        if indent_head:
            return htchar * indent + content
        return content

    def _indent_tail(content):
        return htchar * indent + content

    if repr_ is None:
        repr_ = repr
    if replace is None:
        replace = {}

    if all(isinstance(k, str) and k.isidentifier() for k in obj):
        return prettify_inited(
            type(obj),
            obj,
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            indent_head=indent_head,
            repr_=repr_,
        )
    items = []
    for k, v in obj.items():
        if path is None:
            new_path = k
        else:
            new_path = f"{path}.{k}"

        if new_path in replace:
            new_v = replace[new_path]
        else:
            new_v = prettify(
                v,
                replace=replace,
                path=new_path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1,
                indent_head=False,
                repr_=repr_,
            )
        items.append(lfchar + htchar * (indent + 1) + repr_(k) + ": " + new_v)
    if type(obj) is dict:
        if len(items) == 0:
            return _indent_head("{}")
        return _indent_head("{") + ",".join(items) + lfchar + _indent_tail("}")
    if len(items) == 0:
        return _indent_head(f"{type(obj).__name__}({{}})")
    return _indent_head(f"{type(obj).__name__}({{") + ",".join(items) + lfchar + _indent_tail("})")


def prettify(
    obj: tp.Any,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
    indent_head: bool = True,
    repr_: tp.Optional[tp.Callable] = None,
) -> str:
    """Prettify an object.

    Unfolds regular Python data structures such as lists, tuples, and dictionaries.

    If `obj` is an instance of `Prettified`, calls its `prettify` method.

    Args:
        obj (Any): Object to prettify.
        replace (DictLike): Mapping for value replacement.
        path (str): Current path in the object hierarchy.
        htchar (str): String used for horizontal indentation.
        lfchar (str): Line feed character.
        indent (int): Current indentation level.
        indent_head (bool): Whether to indent the head line.
        repr_ (Optional[Callable]): Function to get the representation of an object.

            Defaults to `repr`.

    Returns:
        str: Prettified string representation of the object.
    """

    def _indent_head(content):
        if indent_head:
            return htchar * indent + content
        return content

    def _indent_tail(content):
        return htchar * indent + content

    if repr_ is None:
        repr_ = repr
    if replace is None:
        replace = {}

    if isinstance(obj, Prettified):
        return obj.prettify(
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            indent_head=indent_head,
            repr_=repr_,
        )
    if attr.has(type(obj)):
        return prettify_inited(
            type(obj),
            attr.asdict(obj),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            indent_head=indent_head,
            repr_=repr_,
        )
    if isinstance(obj, dict):
        return prettify_dict(
            obj,
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            indent_head=indent_head,
            repr_=repr_,
        )
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return prettify_inited(
            type(obj),
            obj._asdict(),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            indent_head=indent_head,
            repr_=repr_,
        )
    if isinstance(obj, (tuple, list, set, frozenset)):
        items = []
        for v in obj:
            new_v = prettify(
                v,
                replace=replace,
                path=path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1,
                indent_head=False,
                repr_=repr_,
            )
            items.append(lfchar + htchar * (indent + 1) + new_v)
        if isinstance(obj, tuple):
            if len(items) == 0:
                return _indent_head("()")
            return _indent_head("(") + ",".join(items) + lfchar + _indent_tail(")")
        if isinstance(obj, list):
            if len(items) == 0:
                return _indent_head("[]")
            return _indent_head("[") + ",".join(items) + lfchar + _indent_tail("]")
        if isinstance(obj, set):
            if len(items) == 0:
                return _indent_head("set()")
            return _indent_head("{") + ",".join(items) + lfchar + _indent_tail("}")
        if len(items) == 0:
            return _indent_head(f"{type(obj).__name__}([])")
        return (
            _indent_head(f"{type(obj).__name__}([") + ",".join(items) + lfchar + _indent_tail("])")
        )
    if isinstance(obj, np.dtype) and hasattr(obj, "fields"):
        items = []
        for k, v in dict(obj.fields).items():
            items.append(lfchar + htchar * (indent + 1) + repr_((k, str(v[0]))))
        if len(items) == 0:
            return _indent_head("np.dtype([])")
        return _indent_head("np.dtype([") + ",".join(items) + lfchar + _indent_tail("])")
    if hasattr(obj, "shape") and isinstance(obj.shape, tuple) and len(obj.shape) > 0:
        module = type(obj).__module__
        qualname = type(obj).__qualname__
        return _indent_head(
            f"<{module}.{qualname} object at {hex(id(obj))} with shape {obj.shape}>"
        )
    if isinstance(obj, float):
        if np.isnan(obj):
            return "np.nan"
        if np.isposinf(obj):
            return "np.inf"
        if np.isneginf(obj):
            return "-np.inf"
    return "".join(_indent_head(line) for line in repr_(obj).splitlines(keepends=True))


def repr_doc(obj: tp.Any) -> str:
    """Representation function suited for documentation.

    Args:
        obj (Any): Object.

    Returns:
        str: Representation.
    """
    import re

    obj_repr = repr(obj)
    if obj_repr.startswith("environ({") and obj_repr.endswith("})"):
        return "os.environ"
    obj_repr = re.sub(r"\s+from\s+'[^']+'", "", obj_repr)
    obj_repr = re.sub(r"\s+at\s+0x[0-9a-fA-F]+", "", obj_repr)
    return obj_repr


def prettify_doc(*args, **kwargs) -> str:
    """Prettify for documentation, equivalent to using `prettify` with `repr_doc` as `repr_`.

    Args:
        *args: Positional arguments for `prettify`.
        **kwargs: Keyword arguments for `prettify`.

    Returns:
        str: Prettified representation of the object.
    """
    return prettify(*args, repr_=repr_doc, **kwargs)


def pprint(*args, **kwargs) -> None:
    """Print the prettified representation of the given arguments.

    Args:
        *args: Positional arguments for `prettify`.
        **kwargs: Keyword arguments for `prettify`.

    Returns:
        None
    """
    print(prettify(*args, **kwargs))


def format_array(
    array: tp.ArrayLike, tabulate: tp.Optional[bool] = None, html: bool = False, **kwargs
) -> str:
    """Format an array for display.

    Args:
        array (ArrayLike): Array-like object to be formatted.
        tabulate (Optional[bool]): If True, use `tabulate.tabulate` for formatting;
            if False, use Pandas formatting functions (`DataFrame.to_string` or `DataFrame.to_html`).

            If None, auto-detect based on the availability of the `tabulate` library and the `html` parameter.
        html (bool): Format the output in HTML if True.
        **kwargs: Keyword arguments for the formatting function.

    Returns:
        str: Formatted array as a string.
    """
    from vectorbtpro.base.reshaping import to_pd_array

    pd_array = to_pd_array(array)
    if tabulate is None:
        from vectorbtpro.utils.module_ import check_installed

        tabulate = check_installed("tabulate") and not html
    if tabulate:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tabulate")
        from tabulate import tabulate

        if isinstance(pd_array, pd.Series):
            pd_array = pd_array.to_frame()
        if html:
            return tabulate(pd_array, headers="keys", tablefmt="html", **kwargs)
        return tabulate(pd_array, headers="keys", **kwargs)
    if html:
        if isinstance(pd_array, pd.Series):
            pd_array = pd_array.to_frame()
        return pd_array.to_html(**kwargs)
    return pd_array.to_string(**kwargs)


def ptable(*args, display_html: tp.Optional[bool] = None, **kwargs) -> None:
    """Print the formatted array.

    Args:
        *args: Positional arguments for `format_array`.
        display_html (Optional[bool]): Display output in HTML if True.

            If None, auto-detect if running in an IPython notebook.
        **kwargs: Keyword arguments for `format_array`.

    Returns:
        None
    """
    from vectorbtpro.utils.checks import in_notebook

    if display_html is None:
        display_html = in_notebook()
    if display_html:
        from IPython.display import HTML, display

        display(HTML(format_array(*args, html=True, **kwargs)))
    else:
        print(format_array(*args, **kwargs))


def format_parameter(param: inspect.Parameter, annotate: bool = False) -> str:
    """Format a function parameter into a string representation.

    Args:
        param (inspect.Parameter): Parameter to format.
        annotate (bool): Include type annotations if True.

    Returns:
        str: Formatted parameter.
    """
    kind = param.kind
    formatted = param.name

    if annotate and param.annotation is not param.empty:
        formatted = f"{formatted}: {inspect.formatannotation(param.annotation)}"

    if param.default is not param.empty:
        if annotate and param.annotation is not param.empty:
            formatted = f"{formatted} = {repr(param.default)}"
        else:
            formatted = f"{formatted}={repr(param.default)}"

    if kind == param.VAR_POSITIONAL:
        formatted = "*" + formatted
    elif kind == param.VAR_KEYWORD:
        formatted = "**" + formatted

    return formatted


def format_signature(
    signature: inspect.signature,
    annotate: bool = False,
    start: str = "\n    ",
    separator: str = ",\n    ",
    end: str = "\n",
) -> str:
    """Format a function signature.

    Args:
        signature (Signature): Function signature to format.
        annotate (bool): Include type annotations if True.
        start (str): String inserted at the beginning of the parameter list.
        separator (str): String used to separate parameters.
        end (str): String appended after the parameter list.

    Returns:
        str: Formatted signature.
    """
    result = []
    render_pos_only_separator = False
    render_kw_only_separator = True

    for param in signature.parameters.values():
        formatted = format_parameter(param, annotate=annotate)

        kind = param.kind

        if kind == param.POSITIONAL_ONLY:
            render_pos_only_separator = True
        elif render_pos_only_separator:
            result.append("/")
            render_pos_only_separator = False

        if kind == param.VAR_POSITIONAL:
            render_kw_only_separator = False
        elif kind == param.KEYWORD_ONLY and render_kw_only_separator:
            result.append("*")
            render_kw_only_separator = False

        result.append(formatted)

    if render_pos_only_separator:
        result.append("/")

    if len(result) == 0:
        rendered = "()"
    else:
        rendered = f"({start + separator.join(result) + end})"

    if annotate and signature.return_annotation is not inspect._empty:
        anno = inspect.formatannotation(signature.return_annotation)
        rendered += f" -> {anno}"

    return rendered


def format_func(func: tp.Callable, incl_doc: bool = True, **kwargs) -> str:
    """Format a function or class constructor.

    Args:
        func (Callable): Function or class to format.

            If a class, its `__init__` method is used.
        incl_doc (bool): If True, include the function's docstring in the output if available.
        **kwargs: Keyword arguments for `format_signature`.

    Returns:
        str: Formatted function description, including its signature and docstring if available.
    """
    from vectorbtpro.utils.attr_ import DefineMixin
    from vectorbtpro.utils.checks import is_attrs_subclass

    doc = func.__doc__
    if is_attrs_subclass(func):
        if issubclass(func, DefineMixin):
            if func.__init__ is DefineMixin.__init__:
                func_name = func.__name__ + ".__attrs_init__"
                func = func.__attrs_init__
            else:
                func_name = func.__name__ + ".__init__"
                func = func.__init__
        else:
            if hasattr(func, "__attrs_init__"):
                func_name = func.__name__ + ".__attrs_init__"
                func = func.__attrs_init__
            else:
                func_name = func.__name__ + ".__init__"
                func = func.__init__
    elif inspect.isclass(func):
        func_name = func.__name__ + ".__init__"
        func = func.__init__
    elif inspect.ismethod(func) and hasattr(func, "__self__"):
        if isinstance(func.__self__, type):
            func_name = func.__self__.__name__ + "." + func.__name__
        else:
            func_name = type(func.__self__).__name__ + "." + func.__name__
    else:
        func_name = func.__qualname__
    if doc is None or (
        func.__doc__ is not None and not func.__doc__.startswith("Method generated by attrs")
    ):
        doc = func.__doc__
    if incl_doc and doc is not None:
        return "{}{}:\n{}".format(
            func_name,
            format_signature(inspect.signature(func), **kwargs),
            "    " + "\n    ".join(inspect.cleandoc(doc).splitlines()),
        )
    return f"{func_name}{format_signature(inspect.signature(func), **kwargs)}"


def phelp(*args, **kwargs) -> None:
    """Print the formatted representation of a function.

    Args:
        *args: Positional arguments for `format_func`.
        **kwargs: Keyword arguments for `format_func`.

    Returns:
        None
    """
    print(format_func(*args, **kwargs))


def pdir(*args, **kwargs) -> None:
    """Print parsed attributes of an object.

    Args:
        *args: Positional arguments for `vectorbtpro.utils.attr_.get_attrs`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.attr_.get_attrs`.

    Returns:
        None
    """
    from vectorbtpro.utils.attr_ import get_attrs

    ptable(get_attrs(*args, **kwargs))


def dump(obj: tp.Any, dump_engine: str = "prettify", **kwargs) -> str:
    """Dump an object to a string using the specified dump engine.

    Args:
        obj (Any): Object to dump.
        dump_engine (str): Name of the dump engine.

            Options include:

            * "repr": Python's `repr` function
            * "repr_doc": `repr_doc`
            * "prettify": `prettify`
            * "nestedtext": `nestedtext` (https://pypi.org/project/nestedtext/)
            * "pyyaml": `pyyaml` (https://pypi.org/project/PyYAML/)
            * "ruamel" or "ruamel.yaml": `ruamel` (https://pypi.org/project/ruamel.yaml/)
            * "yaml": `pyyaml` or `ruamel`, depending on which is installed
            * "toml": `toml` (https://pypi.org/project/toml/)
            * "json": `json` (https://docs.python.org/3/library/json.html)
        **kwargs: Keyword arguments for the dump engine.

    Returns:
        str: Dumped object as a string.
    """
    if isinstance(obj, str):
        return obj
    if dump_engine.lower() == "repr":
        return repr(obj)
    if dump_engine.lower() == "repr_doc":
        return repr_doc(obj, **kwargs)
    if dump_engine.lower() == "prettify":
        return prettify(obj, **kwargs)
    if dump_engine.lower() == "nestedtext":
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("nestedtext")
        import nestedtext as nt

        return nt.dumps(obj, **kwargs)
    if dump_engine.lower() == "yaml":
        from vectorbtpro.utils.module_ import check_installed

        if check_installed("ruamel"):
            dump_engine = "ruamel"
        else:
            dump_engine = "pyyaml"
    if dump_engine.lower() == "pyyaml":
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("yaml")
        import yaml

        def _multiline_str_representer(dumper, data):
            if isinstance(data, str) and "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_str(data)

        class CustomDumper(yaml.SafeDumper):
            pass

        CustomDumper.add_representer(str, _multiline_str_representer)

        if "Dumper" not in kwargs:
            kwargs["Dumper"] = CustomDumper
        return yaml.dump(obj, **kwargs)
    if dump_engine.lower() in ("ruamel", "ruamel.yaml"):
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ruamel")
        from ruamel.yaml import YAML
        from ruamel.yaml.representer import RoundTripRepresenter

        def _multiline_str_representer(dumper, data):
            if isinstance(data, str) and "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_str(data)

        class CustomRepresenter(RoundTripRepresenter):
            pass

        CustomRepresenter.add_representer(str, _multiline_str_representer)

        yaml = YAML(
            typ=kwargs.pop("typ", None),
            pure=kwargs.pop("pure", False),
            plug_ins=kwargs.pop("plug_ins", None),
        )
        if "Representer" not in kwargs:
            yaml.Representer = CustomRepresenter
        for k, v in kwargs.items():
            if not hasattr(yaml, k):
                raise AttributeError(f"Invalid YAML attribute: '{k}'")
            if isinstance(v, tuple):
                getattr(yaml, k)(*v)
            elif isinstance(v, dict):
                getattr(yaml, k)(**v)
            else:
                setattr(yaml, k, v)
        transform = kwargs.pop("transform", None)
        output = io.StringIO()
        yaml.dump(obj, output, transform=transform)
        return output.getvalue()
    if dump_engine.lower() == "toml":
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("toml")
        import toml

        return toml.dumps(obj, **kwargs)
    if dump_engine.lower() == "json":
        import json

        return json.dumps(obj, **kwargs)
    raise ValueError(f"Invalid dump engine: '{dump_engine}'")


def get_dump_language(dump_engine: str) -> str:
    """Return the language corresponding to the provided dump engine.

    Args:
        dump_engine (str): Name of the dump engine.

            See `vectorbtpro.utils.formatting.dump`.

    Returns:
        str: Corresponding language name, or an empty string if the dump engine is unknown.
    """
    if dump_engine.lower() == "repr":
        return "python"
    if dump_engine.lower() == "prettify":
        return "python"
    if dump_engine.lower() == "nestedtext":
        return "text"
    if dump_engine.lower() == "yaml":
        return "yaml"
    if dump_engine.lower() == "pyyaml":
        return "yaml"
    if dump_engine.lower() in ("ruamel", "ruamel.yaml"):
        return "yaml"
    if dump_engine.lower() == "toml":
        return "toml"
    if dump_engine.lower() == "json":
        return "json"
    return ""


def get_dump_frontmatter(dump_engine: str) -> str:
    """Return the frontmatter corresponding to the provided dump engine.

    Args:
        dump_engine (str): Name of the dump engine.

            See `vectorbtpro.utils.formatting.dump`.

    Returns:
        str: Corresponding frontmatter string, or an empty string if unknown.
    """
    if dump_engine.lower() == "yaml":
        return "---"
    if dump_engine.lower() == "pyyaml":
        return "---"
    if dump_engine.lower() in ("ruamel", "ruamel.yaml"):
        return "---"
    return ""
