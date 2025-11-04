# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes for creating and executing asset pipelines.

See `vectorbtpro.utils.knowledge` for the toy dataset.
"""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.eval_ import evaluate
from vectorbtpro.utils.execution import Task
from vectorbtpro.utils.knowledge.base_asset_funcs import AssetFunc
from vectorbtpro.utils.module_ import package_shortcut_config
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = [
    "AssetPipeline",
    "EarlyReturn",
    "BasicAssetPipeline",
    "ComplexAssetPipeline",
]


class AssetPipeline(Base):
    """Abstract class representing an asset pipeline.

    Provides functionality to resolve and execute tasks in an asset pipeline.
    """

    @classmethod
    def resolve_task(
        cls,
        func: tp.AssetFuncLike,
        *args,
        prepare: bool = True,
        prepare_once: bool = True,
        cond_kwargs: tp.KwargsLike = None,
        asset_func_meta: tp.Union[None, dict, list] = None,
        **kwargs,
    ) -> tp.Task:
        """Return a `vectorbtpro.utils.execution.Task` by resolving the provided asset function and its arguments.

        Args:
            func (AssetFuncLike): Asset function identifier, which may be a tuple,
                `vectorbtpro.utils.execution.Task`, string, or subclass of
                `vectorbtpro.utils.knowledge.base_asset_funcs.AssetFunc`.
            *args: Positional arguments for `vectorbtpro.utils.execution.Task`.
            prepare (bool): Flag indicating whether to prepare the function's arguments before execution.
            prepare_once (bool): Flag indicating whether to prepare the function's arguments only once.
            cond_kwargs (KwargsLike): Keyword arguments for conditional preparation.
            asset_func_meta (Union[None, dict, list]): Metadata for the asset function.
            **kwargs: Keyword arguments for `vectorbtpro.utils.execution.Task`.

        Returns:
            Task: Callable task resolved from the provided definition.
        """
        if isinstance(func, tuple):
            func = Task.from_tuple(func)
        if isinstance(func, Task):
            args = (*func.args, *args)
            kwargs = merge_dicts(func.kwargs, kwargs)
            func = func.func
        if isinstance(func, str):
            from vectorbtpro.utils.knowledge import base_asset_funcs, custom_asset_funcs

            base_keys = dir(base_asset_funcs)
            custom_keys = dir(custom_asset_funcs)
            base_values = [getattr(base_asset_funcs, k) for k in base_keys]
            custom_values = [getattr(custom_asset_funcs, k) for k in custom_keys]
            module_items = dict(zip(base_keys + custom_keys, base_values + custom_values))

            if (
                func in module_items
                and isinstance(module_items[func], type)
                and issubclass(module_items[func], AssetFunc)
            ):
                func = module_items[func]
            elif func.title() + "AssetFunc" in module_items:
                func = module_items[func.title() + "AssetFunc"]
            else:
                found_func = False
                for k, v in module_items.items():
                    if isinstance(v, type) and issubclass(v, AssetFunc):
                        if v._short_name is not None:
                            if func.lower() == v._short_name.lower():
                                func = v
                                found_func = True
                                break
                if not found_func:
                    raise ValueError(f"Function '{func}' not found")
        if isinstance(func, AssetFunc):
            raise TypeError("Function must be a subclass of AssetFunc, not an instance")
        if isinstance(func, type) and issubclass(func, AssetFunc):
            _asset_func_meta = {}
            for var_name, var_type in func.__annotations__.items():
                if var_name.startswith("_") and tp.get_origin(var_type) is tp.ClassVar:
                    _asset_func_meta[var_name] = getattr(func, var_name)
            if asset_func_meta is not None:
                if isinstance(asset_func_meta, dict):
                    asset_func_meta.update(_asset_func_meta)
                else:
                    asset_func_meta.append(_asset_func_meta)
            if prepare:
                if prepare_once:
                    if cond_kwargs is None:
                        cond_kwargs = {}
                    if len(cond_kwargs) > 0:
                        prepare_arg_names = get_func_arg_names(func.prepare)
                        for k, v in cond_kwargs.items():
                            if k in prepare_arg_names:
                                kwargs[k] = v
                    args, kwargs = func.prepare(*args, **kwargs)
                    func = func.call
                else:
                    func = func.prepare_and_call
            else:
                func = func.call
        if not callable(func):
            raise TypeError("Function must be callable")
        return Task(func, *args, **kwargs)

    def run(self, d: tp.Any) -> tp.Any:
        """Execute the asset pipeline on the provided data by applying all tasks sequentially.

        Args:
            d (Any): Data item to be processed.

        Returns:
            Any: Result of executing the pipeline on the data item.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def __call__(self, d: tp.Any) -> tp.Any:
        return self.run(d)


@define
class EarlyReturn(DefineMixin):
    """Class representing an early return value for `BasicAssetPipeline`."""

    value: tp.Any = define.field()
    """Early return value."""


class BasicAssetPipeline(AssetPipeline):
    """Class representing a basic asset pipeline.

    Creates a composite function by resolving and chaining individual asset tasks.

    Args:
        *args: Positional arguments for `BasicAssetPipeline.resolve_task`.

            The first positional argument can be a task or list of tasks;
            subsequent positional arguments are used in task resolution.
        **kwargs: Keyword arguments for `BasicAssetPipeline.resolve_task`.

    Examples:
        ```pycon
        >>> asset_pipeline = vbt.BasicAssetPipeline()
        >>> asset_pipeline.add_task("flatten")
        >>> asset_pipeline.add_task("query", len)
        >>> asset_pipeline.add_task("get")

        >>> asset_pipeline(dataset[0])
        5
        ```
    """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 0:
            tasks = []
        else:
            tasks = args[0]
            args = args[1:]
        if not isinstance(tasks, list):
            tasks = [tasks]
        self._tasks = [self.resolve_task(task, *args, **kwargs) for task in tasks]

    @property
    def tasks(self) -> tp.List[tp.Task]:
        """Tasks that have been added to the pipeline.

        Returns:
            List[Task]: List of tasks in the pipeline.
        """
        return self._tasks

    def add_task(self, func: tp.AssetFuncLike, *args, **kwargs) -> None:
        """Add a task to the pipeline using the provided asset function and arguments.

        Args:
            func (AssetFuncLike): Asset function identifier, which may be a tuple,
                `vectorbtpro.utils.execution.Task`, string, or subclass of
                `vectorbtpro.utils.knowledge.base_asset_funcs.AssetFunc`.
            *args: Positional arguments for `BasicAssetPipeline.resolve_task`.
            **kwargs: Keyword arguments for `BasicAssetPipeline.resolve_task`.

        Returns:
            None
        """
        self.tasks.append(self.resolve_task(func, *args, **kwargs))

    @classmethod
    def compose_tasks(cls, tasks: tp.List[tp.Task]) -> tp.Callable:
        """Compose multiple tasks into a single callable that applies them sequentially.

        Args:
            tasks (List[Task]): List of tasks to be composed.

        Returns:
            Callable: Callable that takes a data item and applies the tasks sequentially.
        """

        def _composed(d):
            result = d
            for func, args, kwargs in tasks:
                result = func(result, *args, **kwargs)
                if isinstance(result, EarlyReturn):
                    return result.value
            return result

        return _composed

    def run(self, d: tp.Any) -> tp.Any:
        return self.compose_tasks(list(self.tasks))(d)


class ComplexAssetPipeline(AssetPipeline):
    """Class representing a complex asset pipeline.

    This pipeline takes an expression string that may contain nested function calls and
    a context mapping. It resolves functions within the expression and evaluates the expression
    using `vectorbtpro.utils.eval_.evaluate`.

    Args:
        expression (str): Expression string to evaluate.
        context (KwargsLike): Mapping of variables for expression evaluation.
        prepare_once (bool): Flag indicating whether to prepare the function's arguments only once.
        **resolve_task_kwargs: Keyword arguments for task resolution.

    Examples:
        ```pycon
        >>> asset_pipeline = vbt.ComplexAssetPipeline("query(flatten(d), len)")
        >>> asset_pipeline(dataset[0])
        5
        ```
    """

    def __init__(
        self,
        expression: str,
        context: tp.KwargsLike = None,
        prepare_once: bool = True,
        **resolve_task_kwargs,
    ) -> None:
        self._expression, self._context = self.resolve_expression_and_context(
            expression,
            context=context,
            prepare_once=prepare_once,
            **resolve_task_kwargs,
        )

    @classmethod
    def resolve_expression_and_context(
        cls,
        expression: str,
        context: tp.KwargsLike = None,
        prepare: bool = True,
        prepare_once: bool = True,
        **resolve_task_kwargs,
    ) -> tp.Tuple[str, tp.Kwargs]:
        """Resolve an expression and update its context.

        Parses the expression string to extract function calls and their arguments,
        then removes the first positional argument from each function call.
        It also builds a new context by merging resolved functions with the existing context.

        Args:
            expression (str): Expression string to process.
            context (KwargsLike): Mapping of context variables.
            prepare (bool): Flag indicating whether to prepare the function's arguments before execution.
            prepare_once (bool): Flag indicating whether to prepare the function's arguments only once.
            **resolve_task_kwargs: Keyword arguments for task resolution.

        Returns:
            Tuple[str, Kwargs]: Tuple containing the modified expression and the updated context.
        """
        import ast
        import builtins
        import importlib
        import sys

        if context is None:
            context = {}
        for k, v in package_shortcut_config.items():
            if k not in context:
                try:
                    context[k] = importlib.import_module(v)
                except ImportError:
                    pass
        tree = ast.parse(expression)
        builtin_functions = set(dir(builtins))
        imported_functions = set()
        imported_modules = set()
        defined_functions = set()
        func_context = {}

        class _FunctionAnalyzer(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split(".")[0]
                    imported_modules.add(name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_functions.add(name)
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                defined_functions.add(node.name)
                self.generic_visit(node)

        analyzer = _FunctionAnalyzer()
        analyzer.visit(tree)

        class _NodeMixin:
            def get_func_name(self, func):
                attrs = []
                while isinstance(func, ast.Attribute):
                    attrs.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    attrs.append(func.id)
                return ".".join(reversed(attrs)) if attrs else "<unknown>"

            def is_function_assigned(self, func):
                func_name = self.get_func_name(func)
                if "." in func_name:
                    func_name = func_name.split(".")[0]
                return (
                    func_name in context
                    or func_name in builtin_functions
                    or func_name in imported_functions
                    or func_name in imported_modules
                    or func_name in defined_functions
                )

        class _FunctionCallVisitor(ast.NodeVisitor, _NodeMixin):
            def process_argument(self, arg):
                if isinstance(arg, ast.Constant):
                    return arg.value
                elif isinstance(arg, ast.Name):
                    var_name = arg.id
                    if var_name in context:
                        return context[var_name]
                    elif var_name in builtin_functions:
                        return getattr(builtins, var_name)
                    else:
                        raise ValueError(f"Variable '{var_name}' is not defined in the context")
                elif isinstance(arg, ast.List):
                    return [self.process_argument(elem) for elem in arg.elts]
                elif isinstance(arg, ast.Tuple):
                    return tuple(self.process_argument(elem) for elem in arg.elts)
                elif isinstance(arg, ast.Dict):
                    return {
                        self.process_argument(k): self.process_argument(v)
                        for k, v in zip(arg.keys, arg.values)
                    }
                elif isinstance(arg, ast.Set):
                    return {self.process_argument(elem) for elem in arg.elts}
                elif isinstance(arg, ast.Call):
                    if self.is_function_assigned(arg.func):
                        return self.get_func_name(arg.func)
                raise ValueError(f"Unsupported or dynamic argument: {ast.dump(arg)}")

            def visit_Call(self, node):
                self.generic_visit(node)
                func_name = self.get_func_name(node.func)
                pos_args = []
                for arg in node.args[1:]:
                    arg_value = self.process_argument(arg)
                    pos_args.append(arg_value)
                kw_args = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        raise ValueError(
                            f"Dynamic keyword argument names are not allowed in '{func_name}'"
                        )
                    kw_name = kw.arg
                    kw_value = self.process_argument(kw.value)
                    kw_args[kw_name] = kw_value
                if not self.is_function_assigned(node.func):
                    task = cls.resolve_task(
                        func_name,
                        *pos_args,
                        **kw_args,
                        prepare=prepare,
                        prepare_once=prepare_once,
                        **resolve_task_kwargs,
                    )
                    if prepare and prepare_once:

                        def func(d, _task=task):
                            return _task.func(d, *_task.args, **_task.kwargs)

                    else:
                        func = task.func

                    func_context[func_name] = func

        visitor = _FunctionCallVisitor()
        visitor.visit(tree)

        if prepare and prepare_once:

            class _ArgumentPruner(ast.NodeTransformer, _NodeMixin):
                def visit_Call(self, node: ast.Call):
                    if not self.is_function_assigned(node.func):
                        if node.args:
                            node.args = [node.args[0]]
                        else:
                            node.args = []
                        node.keywords = []
                    self.generic_visit(node)
                    return node

            pruner = _ArgumentPruner()
            modified_tree = pruner.visit(tree)
            ast.fix_missing_locations(modified_tree)
            if sys.version_info >= (3, 9):
                new_expression = ast.unparse(modified_tree)
            else:
                import astor

                new_expression = astor.to_source(modified_tree).strip()
        else:
            new_expression = expression

        new_context = merge_dicts(func_context, context)
        return new_expression, new_context

    @property
    def expression(self) -> str:
        """Processed expression string for the pipeline.

        Returns:
            str: Expression string after processing.
        """
        return self._expression

    @property
    def context(self) -> tp.Kwargs:
        """Updated context mapping for the pipeline.

        Returns:
            Kwargs: Context mapping used for expression evaluation.
        """
        return self._context

    def run(self, d: tp.Any) -> tp.Any:
        context = merge_dicts({"d": d, "x": d}, self.context)
        return evaluate(self.expression, context=context)
