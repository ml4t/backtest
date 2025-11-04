# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for modules."""

import importlib
import importlib.util
import inspect
import pkgutil
import sys
import urllib.request
import webbrowser
from functools import cached_property
from pathlib import Path
from types import ModuleType

from vectorbtpro import _typing as tp
from vectorbtpro._opt_deps import opt_dep_config
from vectorbtpro.utils.config import HybridConfig
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "import_module_from_path",
    "get_refname",
    "get_obj",
    "imlucky",
    "get_api_ref",
    "open_api_ref",
]

__pdoc__ = {}

package_shortcut_config = HybridConfig(
    dict(
        vbt="vectorbtpro",
        pd="pandas",
        np="numpy",
        nb="numba",
    )
)
"""_"""

__pdoc__["package_shortcut_config"] = f"""Config for package shortcuts.

```python
{package_shortcut_config.prettify_doc()}
```
"""


def get_module(obj: tp.Any) -> tp.Optional[ModuleType]:
    """Return the module in which the given object is defined.

    Args:
        obj (Any): Object whose module is to be obtained.

    Returns:
        Optional[ModuleType]: Module where the object is defined; None if the module cannot be determined.
    """
    if isinstance(obj, ModuleType):
        return obj
    target = inspect.unwrap(obj) if callable(obj) else obj
    modname = getattr(target, "__module__", None) or getattr(type(target), "__module__", None)
    if not modname:
        return None
    return sys.modules.get(modname)


def is_from_module(obj: tp.Any, module: ModuleType) -> bool:
    """Return True if the provided object is defined in the specified module; otherwise, return False.

    Args:
        obj (Any): Object to verify.
        module (ModuleType): Module to check against.

    Returns:
        bool: True if the object is from the specified module; otherwise, False.
    """
    mod = get_module(obj)
    return mod is None or mod.__name__ == module.__name__


def list_module_keys(
    module_or_name: tp.Union[str, ModuleType],
    whitelist: tp.Optional[tp.List[str]] = None,
    blacklist: tp.Optional[tp.List[str]] = None,
) -> tp.List[str]:
    """Return a list of names for all public functions and classes in the specified module.

    Args:
        module_or_name (Union[str, ModuleType]): Module or its name to inspect.
        whitelist (Optional[List[str]]): Additional names to include.
        blacklist (Optional[List[str]]): Names to exclude from the list.

    Returns:
        List[str]: List of public function and class names.
    """
    if whitelist is None:
        whitelist = []
    if blacklist is None:
        blacklist = []
    if isinstance(module_or_name, str):
        module = sys.modules[module_or_name]
    else:
        module = module_or_name
    return [
        name
        for name, obj in inspect.getmembers(module)
        if (
            not name.startswith("_")
            and is_from_module(obj, module)
            and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))
            and name not in blacklist
        )
        or name in whitelist
    ]


def search_package(
    package: tp.Union[str, ModuleType],
    match_func: tp.Callable,
    blacklist: tp.Optional[tp.Sequence[str]] = None,
    path_attrs: bool = False,
    return_first: bool = False,
    _visited: tp.Optional[tp.Set[str]] = None,
) -> tp.Union[None, tp.Any, tp.Dict[str, tp.Any]]:
    """Search for objects in a package that satisfy a given condition.

    The matching function should accept the name of an object and the object itself, and return a boolean.

    Args:
        package (Union[str, ModuleType]): Package or its name to search.
        match_func (Callable): Function that takes an object's name and the object, returning a boolean.
        blacklist (Optional[Sequence[str]]): Names to exclude from the search.
        path_attrs (bool): If True, use fully qualified names for object attributes.
        return_first (bool): If True, return the first matching object.

    Returns:
        Union[None, Any, Dict[str, Any]]: If `return_first` is True, returns the first matching object or None;
            otherwise, returns a dictionary of matching objects.
    """
    if blacklist is None:
        blacklist = []
    if _visited is None:
        _visited = set()
    results = {}

    if isinstance(package, str):
        package = importlib.import_module(package)
    if package.__name__ not in _visited:
        _visited.add(package.__name__)
        for attr in dir(package):
            if path_attrs:
                path_attr = package.__name__ + "." + attr
            else:
                path_attr = attr
            if not attr.startswith("_") and match_func(path_attr, getattr(package, attr)):
                if return_first:
                    return getattr(package, attr)
                results[path_attr] = getattr(package, attr)

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if ".".join(name.split(".")[:-1]) != package.__name__:
            continue
        try:
            if name in _visited or name in blacklist:
                continue
            _visited.add(name)
            module = importlib.import_module(name)
            for attr in dir(module):
                if path_attrs:
                    path_attr = module.__name__ + "." + attr
                else:
                    path_attr = attr
                if not attr.startswith("_") and match_func(path_attr, getattr(module, attr)):
                    if return_first:
                        return getattr(module, attr)
                    results[path_attr] = getattr(module, attr)
            if is_pkg:
                results.update(
                    search_package(
                        name,
                        match_func,
                        blacklist=blacklist,
                        path_attrs=path_attrs,
                        _visited=_visited,
                    )
                )
        except (ModuleNotFoundError, ImportError):
            pass
    if return_first:
        return None
    return results


def find_class(path: str) -> tp.Optional[tp.Type]:
    """Return a class object based on its fully qualified path.

    Args:
        path (str): Dot-separated path to the class.

    Returns:
        Optional[Type]: Class if found; otherwise, None.
    """
    try:
        path_parts = path.split(".")
        module_path = ".".join(path_parts[:-1])
        class_name = path_parts[-1]
        if module_path.startswith("vectorbtpro.indicators.factory"):
            import vectorbtpro as vbt

            return getattr(vbt, path_parts[-2])(class_name)
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            return getattr(module, class_name)
    except Exception:
        pass
    return None


def check_installed(pkg_name: str) -> bool:
    """Return True if the package with the specified name is installed; otherwise, return False.

    Args:
        pkg_name (str): Name of the package.

    Returns:
        bool: True if the package is installed; otherwise, False.
    """
    return importlib.util.find_spec(pkg_name) is not None


def get_installed_overview() -> tp.Dict[str, bool]:
    """Return a dictionary mapping package names from `opt_dep_config` to their installation status.

    Returns:
        Dict[str, bool]: Mapping where keys are package names and values indicate installation status.
    """
    return {pkg_name: check_installed(pkg_name) for pkg_name in opt_dep_config.keys()}


def get_package_meta(pkg_name: str) -> dict:
    """Return the metadata dictionary for the specified package from `opt_dep_config`.

    Args:
        pkg_name (str): Name of the package.

    Returns:
        dict: Dictionary containing metadata such as 'dist_name', 'version', and 'link'.
    """
    if pkg_name not in opt_dep_config:
        raise KeyError(f"Package '{pkg_name}' not found in opt_dep_config")
    dist_name = opt_dep_config[pkg_name].get("dist_name", pkg_name)
    version = opt_dep_config[pkg_name].get("version", "")
    link = opt_dep_config[pkg_name].get("link", f"https://pypi.org/project/{dist_name}/")
    return dict(dist_name=dist_name, version=version, link=link)


def assert_can_import(pkg_name: str) -> None:
    """Assert that the specified package can be imported.

    The package must be listed in `opt_dep_config`. An `ImportError` is raised if the package
    is not installed or the installed version is incompatible.

    Args:
        pkg_name (str): Name of the package.

    Returns:
        None
    """
    from importlib.metadata import version as get_version

    metadata = get_package_meta(pkg_name)
    dist_name = metadata["dist_name"]
    version = version_str = metadata["version"]
    link = metadata["link"]
    if not check_installed(pkg_name):
        raise ImportError(f"Please install {dist_name}{version_str}. See {link}.")
    if version != "":
        actual_version_parts = get_version(dist_name).split(".")
        actual_version_parts = map(lambda x: x if x.isnumeric() else f"'{x}'", actual_version_parts)
        actual_version = "(" + ",".join(actual_version_parts) + ")"
        if version[0].isdigit():
            operator = "=="
        else:
            operator = version[:2]
            version_parts = version[2:].split(".")
            version_parts = map(lambda x: x if x.isnumeric() else f"'{x}'", version_parts)
            version = "(" + ",".join(version_parts) + ")"
        if not eval(f"{actual_version} {operator} {version}"):
            raise ImportError(f"Please install {dist_name}{version_str}. See {link}.")


def assert_can_import_any(*pkg_names: str) -> None:
    """Assert that at least one of the specified packages can be imported.

    Packages must be listed in `opt_dep_config`. If none of the packages can be imported,
    an ImportError is raised.

    Args:
        *pkg_names (str): Additional package names for checking import.

    Returns:
        None
    """
    if len(pkg_names) == 1:
        return assert_can_import(pkg_names[0])
    for pkg_name in pkg_names:
        try:
            return assert_can_import(pkg_name)
        except ImportError:
            pass
    requirements = []
    for pkg_name in pkg_names:
        metadata = get_package_meta(pkg_name)
        dist_name = metadata["dist_name"]
        version_str = metadata["version"]
        link = metadata["link"]
        requirements.append(f"{dist_name}{version_str} - {link}")
    raise ImportError("Please install any of " + ", ".join(requirements))


def warn_cannot_import(pkg_name: str) -> bool:
    """Warn if the specified package cannot be imported.

    The package must be listed in `opt_dep_config`. If the package cannot be imported,
    a warning is issued and True is returned; otherwise, False is returned.

    Args:
        pkg_name (str): Name of the package.

    Returns:
        bool: True if the package cannot be imported; otherwise, False.
    """
    try:
        assert_can_import(pkg_name)
        return False
    except ImportError as e:
        warn(str(e))
        return True


def import_module_from_path(module_path: tp.PathLike, reload: bool = False) -> ModuleType:
    """Import a Python module from a specified file path.

    Args:
        module_path (PathLike): File system path to the module.
        reload (bool): Whether to force reloading if the module is already imported.

    Returns:
        ModuleType: Imported module.
    """
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path.resolve()))
    module = importlib.util.module_from_spec(spec)
    if module.__name__ in sys.modules and not reload:
        return sys.modules[module.__name__]
    spec.loader.exec_module(module)
    sys.modules[module.__name__] = module
    return module


def get_caller_qualname() -> tp.Optional[str]:
    """Return the qualified name of the calling function or method.

    Returns:
        Optional[str]: Qualified name of the function or method that invoked this function.
    """
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        code = caller_frame.f_code
        func_name = code.co_name
        locals_ = caller_frame.f_locals
        if "self" in locals_:
            cls = locals_["self"].__class__
            return f"{cls.__qualname__}.{func_name}"
        elif "cls" in locals_:
            cls = locals_["cls"]
            return f"{cls.__qualname__}.{func_name}"
        else:
            module = inspect.getmodule(caller_frame)
            if module:
                func = module.__dict__.get(func_name)
                if func and hasattr(func, "__qualname__"):
                    return func.__qualname__
            return func_name
    finally:
        del frame


def get_method_class(meth: tp.Callable) -> tp.Optional[tp.Type]:
    """Return the class associated with the given method, if available.

    Args:
        meth (Callable): Method or function for which to determine the associated class.

    Returns:
        Optional[type]: Class object if found, otherwise None.
    """
    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and getattr(meth, "__self__", None) is not None
        and getattr(meth.__self__, "__class__", None)
    ):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, "__func__", meth)
    if inspect.isfunction(meth):
        cls = getattr(
            get_module(meth), meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0], None
        )
        if isinstance(cls, type):
            return cls
    return getattr(meth, "__objclass__", None)


def parse_refname(obj: tp.Any) -> str:
    """Return the fully qualified reference name for the provided object.

    Args:
        obj (Any): Target object to derive its reference name.

    Returns:
        str: Fully qualified reference name.
    """
    from vectorbtpro.utils.decorators import class_property, custom_property, hybrid_property

    if inspect.ismodule(obj):
        return obj.__name__
    if inspect.isclass(obj):
        return obj.__module__ + "." + obj.__qualname__

    if isinstance(obj, staticmethod):
        return parse_refname(obj.__func__)
    if isinstance(obj, classmethod):
        return parse_refname(obj.__func__)

    if (
        inspect.isdatadescriptor(obj)
        or inspect.ismethoddescriptor(obj)
        or inspect.isgetsetdescriptor(obj)
        or inspect.ismemberdescriptor(obj)
    ):
        cls = getattr(obj, "__objclass__", None)
        name = getattr(obj, "__name__", None)
        if cls and name:
            return parse_refname(cls) + "." + name

    if inspect.ismethod(obj) or inspect.isfunction(obj):
        cls = get_method_class(obj)
        if cls is not None:
            return parse_refname(cls) + "." + obj.__name__
        if hasattr(obj, "func"):
            return parse_refname(obj.func)

    if isinstance(obj, (class_property, hybrid_property, custom_property)):
        return parse_refname(obj.func)
    if isinstance(obj, cached_property) and hasattr(obj, "func"):
        return parse_refname(obj.func)
    if isinstance(obj, property):
        return parse_refname(obj.fget)

    if hasattr(obj, "__name__"):
        module = get_module(obj)
        if module is not None and obj.__name__ in module.__dict__:
            return parse_refname(module) + "." + obj.__name__

    module = get_module(obj)
    if module is not None:
        for k, v in module.__dict__.items():
            if obj is v:
                return parse_refname(module) + "." + k

    return parse_refname(type(obj))


def get_refname_module_and_qualname(
    refname: str,
    module: tp.Optional[ModuleType] = None,
) -> tp.Tuple[tp.Optional[ModuleType], tp.Optional[str]]:
    """Return the module and qualified name extracted from the given reference name.

    Args:
        refname (str): Dot-separated reference name.
        module (Optional[ModuleType]): Module context for extraction.

    Returns:
        Tuple[Optional[ModuleType], Optional[str]]: Tuple containing the module and the qualified name.
    """
    refname_parts = refname.split(".")
    if module is None:
        module = importlib.import_module(refname_parts[0])
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module, None
        return get_refname_module_and_qualname(".".join(refname_parts), module=module)
    elif inspect.ismodule(getattr(module, refname_parts[0])):
        module = getattr(module, refname_parts[0])
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module, None
        return get_refname_module_and_qualname(".".join(refname_parts), module=module)
    else:
        return module, ".".join(refname_parts)


def resolve_refname(
    refname: str, module: tp.Union[None, str, ModuleType] = None
) -> tp.Optional[tp.MaybeList[str]]:
    """Resolve a reference name into its fully qualified form using the provided module context.

    Args:
        refname (str): Dot-separated reference name.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.

    Returns:
        Optional[MaybeList[str]]: Resolved reference name(s) as a string or a list of strings,
            or None if unresolved.
    """
    if refname == "":
        if module is None:
            return None
        if isinstance(module, str):
            return module
        return module.__name__

    refname_parts = refname.split(".")
    if module is None:
        if refname_parts[0] in package_shortcut_config:
            refname_parts[0] = package_shortcut_config[refname_parts[0]]
            module = importlib.import_module(refname_parts[0])
            refname_parts = refname_parts[1:]
        else:
            try:
                module = importlib.import_module(refname_parts[0])
                refname_parts = refname_parts[1:]
            except ImportError:
                module = "vectorbtpro"
    if isinstance(module, str):
        module = importlib.import_module(module)
    if len(refname_parts) == 0:
        return module.__name__
    if refname_parts[0] in package_shortcut_config:
        if package_shortcut_config[refname_parts[0]] == module.__name__:
            refname_parts[0] = package_shortcut_config[refname_parts[0]]
    if refname_parts[0] == module.__name__ and refname_parts[0] not in module.__dict__:
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module.__name__

    if refname_parts[0] in module.__dict__:
        obj = module.__dict__[refname_parts[0]]
        if inspect.ismodule(obj):
            parent_module = ".".join(obj.__name__.split(".")[:-1])
        else:
            parent_module = get_module(obj)
            if parent_module is not None:
                if refname_parts[0] in parent_module.__dict__:
                    parent_module = parent_module.__name__
                else:
                    parent_module = None
        if parent_module is None or parent_module == module.__name__:
            if inspect.ismodule(obj):
                module = getattr(module, refname_parts[0])
                refname_parts = refname_parts[1:]
                return resolve_refname(".".join(refname_parts), module=module)
            if hasattr(obj, "__name__") and obj.__name__ in module.__dict__:
                obj = module.__dict__[obj.__name__]
                refname_parts[0] = obj.__name__
            if len(refname_parts) == 1:
                return module.__name__ + "." + refname_parts[0]
            if not isinstance(obj, type):
                cls = type(obj)
            else:
                cls = obj
            k = refname_parts[1]
            v = inspect.getattr_static(cls, k, None)
            found_super_cls = None
            for i, super_cls in enumerate(inspect.getmro(cls)[1:]):
                if k in dir(super_cls):
                    v2 = inspect.getattr_static(super_cls, k, None)
                    if v2 is not None and v == v2:
                        found_super_cls = super_cls
            if found_super_cls is not None:
                cls_path = found_super_cls.__module__ + "." + found_super_cls.__name__
                return cls_path + "." + ".".join(refname_parts[1:])
            return module.__name__ + "." + ".".join(refname_parts)
        if inspect.ismodule(obj):
            parent_module = obj
            refname_parts = refname_parts[1:]
        return resolve_refname(".".join(refname_parts), module=parent_module)

    refnames = []
    visited_modules = set()
    for k, v in module.__dict__.items():
        if v is not module:
            if (
                inspect.ismodule(v)
                and v.__name__.startswith(module.__name__)
                and v.__name__ not in visited_modules
            ):
                visited_modules.add(v.__name__)
                new_refname = resolve_refname(".".join(refname_parts), module=v)
                if new_refname is not None:
                    if isinstance(new_refname, str):
                        new_refname = [new_refname]
                    for r in new_refname:
                        if r not in refnames:
                            refnames.append(r)
    if len(refnames) > 1:
        return refnames
    if len(refnames) == 1:
        return refnames[0]
    return None


def get_refname(
    obj: tp.Any,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
) -> tp.Optional[tp.MaybeList[str]]:
    """Parse and optionally resolve the reference name(s) for an object.

    Args:
        obj (Any): Object from which to extract the reference name.

            If a tuple is provided, its elements are concatenated.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the parsed reference name.

    Returns:
        Optional[MaybeList[str]]: Resolved reference name as a string,
            a list of strings if multiple names are found, or None.
    """
    if isinstance(obj, tuple):
        if len(obj) == 1:
            obj = obj[0]
        else:
            first_refname = parse_refname(obj[0])
            obj = first_refname + "." + ".".join(obj[1:])
    if isinstance(obj, str):
        refname = obj
    else:
        refname = parse_refname(obj)
    if resolve:
        return resolve_refname(refname, module=module)
    return refname


def get_refname_obj(refname: str) -> tp.Any:
    """Return the object corresponding to a dot-separated reference name.

    Args:
        refname (str): Dot-separated reference name.

    Returns:
        Any: Object obtained by importing modules and accessing attributes.
    """
    refname_parts = refname.split(".")
    obj = None
    for refname_part in refname_parts:
        if obj is None:
            obj = importlib.import_module(refname_part)
        else:
            obj = getattr(obj, refname_part)
    return obj


def get_obj(*args, allow_multiple: bool = False, **kwargs) -> tp.MaybeList:
    """Return the object obtained by resolving its reference name.

    Args:
        *args: Positional arguments for `get_refname`.
        allow_multiple (bool): Whether to allow returning multiple objects
            if more than one reference name is resolved.
        **kwargs: Keyword arguments for `get_refname`.

    Returns:
        MaybeList: Resolved object or a list of objects if multiple reference names are found.
    """
    refname = get_refname(*args, **kwargs)
    if isinstance(refname, list):
        obj = None
        for _refname in refname:
            _obj = get_refname_obj(_refname)
            if obj is None:
                obj = _obj
            elif not isinstance(obj, list):
                if _obj is not obj:
                    if not allow_multiple:
                        raise ValueError(
                            "Multiple reference names found:\n\n* {}".format("\n* ".join(refname))
                        )
                    obj = [obj, _obj]
            else:
                if _obj not in obj:
                    obj.append(_obj)
        return obj
    return get_refname_obj(refname)


def prepare_refname(
    obj: tp.Any,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    vbt_only: bool = False,
    return_parts: bool = False,
    raise_error: bool = True,
) -> tp.Union[None, str, tp.Tuple[str, ModuleType, str]]:
    """Prepare the reference name for an object and optionally extract its module and qualified name.

    Args:
        obj (Any): Object or reference name to prepare.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the object's reference name.
        vbt_only (bool): If True, limit resolution to objects within vectorbtpro.
        return_parts (bool): If True, return a tuple containing the reference name, module, and qualified name.
        raise_error (bool): Whether to raise an error if the reference name cannot be determined.

    Returns:
        Union[None, str, Tuple[str, ModuleType, str]]: Prepared reference name as a string,
            or a tuple of (refname, module, qualified name) if `return_parts` is True; or None
            if the reference name cannot be determined.
    """

    def _raise_error():
        raise ValueError(
            "Couldn't find the reference name, or the object is external. "
            "If the object is internal, please decompose the object or provide a string instead."
        )

    refname = get_refname(obj, module=module, resolve=resolve)
    if refname is None:
        if raise_error:
            _raise_error()
        return None
    if isinstance(refname, list):
        raise ValueError("Multiple reference names found:\n\n* {}".format("\n* ".join(refname)))
    if vbt_only or return_parts or resolve:
        module, qualname = get_refname_module_and_qualname(refname)
        if module.__name__.split(".")[0] != "vectorbtpro" and vbt_only:
            if raise_error:
                _raise_error()
            return None
        if return_parts:
            return refname, module, qualname
        if resolve:
            if qualname is None:
                return module.__name__
            return module.__name__ + "." + qualname
    return refname


def annotate_refname_parts(refname: str) -> tp.Tuple[dict, ...]:
    """Annotate each part of a dot-separated reference name with its corresponding object.

    Args:
        refname (str): Dot-separated reference name.

    Returns:
        Tuple[dict, ...]: Tuple of dictionaries, each containing:

            * `name`: Reference name part.
            * `obj`: Object corresponding to the reference name part.
    """
    refname_parts = refname.split(".")
    obj = None
    annotated_parts = []
    for refname_part in refname_parts:
        if obj is None:
            obj = importlib.import_module(refname_part)
        else:
            obj = getattr(obj, refname_part)
        annotated_parts.append(dict(name=refname_part, obj=obj))
    return tuple(annotated_parts)


def get_imlucky_url(query: str) -> str:
    """Construct a DuckDuckGo "I'm lucky" URL for a query.

    Args:
        query (str): Search query.

    Returns:
        str: DuckDuckGo "I'm lucky" URL based on the query.
    """
    return "https://duckduckgo.com/?q=!ducky+" + urllib.request.pathname2url(query)


def imlucky(query: str, **kwargs) -> bool:
    """Open a DuckDuckGo "I'm lucky" URL for a query in the web browser.

    Args:
        query (str): Search query.
        **kwargs: Keyword arguments for `webbrowser.open`.

    Returns:
        bool: True if the browser was opened successfully, False otherwise.
    """
    return webbrowser.open(get_imlucky_url(query), **kwargs)


def get_api_ref(
    obj: tp.Any,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    vbt_only: bool = False,
) -> str:
    """Return the API reference URL for an object.

    Args:
        obj (Any): Object for which the API reference is constructed.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the object's reference name.
        vbt_only (bool): If True, limit resolution to objects within vectorbtpro.

    Returns:
        str: API reference URL for the given object.
    """
    refname, module, qualname = prepare_refname(
        obj,
        module=module,
        resolve=resolve,
        vbt_only=vbt_only,
        return_parts=True,
    )
    if module.__name__.split(".")[0] == "vectorbtpro":
        api_url = "https://github.com/polakowo/vectorbt.pro/blob/pvt-links/api/"
        md_url = api_url + module.__name__ + ".md/"
        if qualname is None:
            return md_url + "#" + module.__name__.replace(".", "")
        return md_url + "#" + module.__name__.replace(".", "") + qualname.replace(".", "")
    if resolve:
        if qualname is None:
            search_query = module.__name__
        else:
            search_query = module.__name__ + "." + qualname
    else:
        search_query = refname
    return get_imlucky_url(search_query)


def open_api_ref(
    obj: tp.Any,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    **kwargs,
) -> bool:
    """Open the API reference URL for an object in the web browser.

    Args:
        obj (Any): Object whose API reference is to be opened.
        module (Union[None, str, ModuleType]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the object's reference name.
        **kwargs: Keyword arguments for `webbrowser.open`.

    Returns:
        bool: True if the browser was opened successfully, False otherwise.
    """
    return webbrowser.open(get_api_ref(obj, module=module, resolve=resolve), **kwargs)
