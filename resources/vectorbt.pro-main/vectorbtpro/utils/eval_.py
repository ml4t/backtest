# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for evaluation and compilation."""

import ast
import builtins
import concurrent.futures as cf
import inspect
import symtable

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import Configured

if tp.TYPE_CHECKING:
    from jupyter_client.client import KernelClient as KernelClientT
    from jupyter_client.manager import KernelManager as KernelManagerT
else:
    KernelManagerT = "jupyter_client.manager.KernelManager"
    KernelClientT = "jupyter_client.client.KernelClient"

__all__ = [
    "evaluate",
    "JupyterKernel",
    "VBTKernel",
]


def evaluate(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate one or multiple lines of Python code and return the result of the final expression.

    Args:
        expr (str): Expression string.

            Must contain valid Python code and can be single-line or multi-line.
        context (KwargsLike): Dictionary representing the execution context.

    Returns:
        Any: Result of evaluating the final expression.
    """
    expr = inspect.cleandoc(expr)
    if context is None:
        context = {}
    if "\n" in expr:
        tree = ast.parse(expr)
        eval_expr = ast.Expression(tree.body[-1].value)
        exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
        exec(compile(exec_expr, "<string>", "exec"), context)
        return eval(compile(eval_expr, "<string>", "eval"), context)
    return eval(compile(expr, "<string>", "eval"), context)


def get_symbols(table: symtable.SymbolTable) -> tp.List[symtable.Symbol]:
    """Recursively retrieve all symbols from a symbol table.

    Args:
        table (symtable.SymbolTable): Symbol table to traverse.

    Returns:
        List[symtable.Symbol]: List of symbols found in the table and its child tables.
    """
    symbols = []
    children = {child.get_name(): child for child in table.get_children()}
    for symbol in table.get_symbols():
        if symbol.is_namespace():
            symbols.extend(get_symbols(children[symbol.get_name()]))
        else:
            symbols.append(symbol)
    return symbols


def get_free_vars(expr: str) -> tp.List[str]:
    """Parse the provided code and return free variable names, excluding built-in names.

    Args:
        expr (str): Expression string.

            Must contain valid Python code and can be single-line or multi-line.

    Returns:
        List[str]: List of free variable names found in the code.
    """
    expr = inspect.cleandoc(expr)
    global_table = symtable.symtable(expr, "<string>", "exec")
    symbols = get_symbols(global_table)
    builtins_set = set(dir(builtins))
    free_vars = []
    free_vars_set = set()
    not_free_vars_set = set()
    for symbol in symbols:
        symbol_name = symbol.get_name()
        if (
            symbol.is_imported()
            or symbol.is_parameter()
            or symbol.is_assigned()
            or symbol_name in builtins_set
        ):
            not_free_vars_set.add(symbol_name)
    for symbol in symbols:
        symbol_name = symbol.get_name()
        if symbol_name not in not_free_vars_set and symbol_name not in free_vars_set:
            free_vars.append(symbol_name)
            free_vars_set.add(symbol_name)
    return free_vars


class Evaluable(Base):
    """Abstract class for objects that can be evaluated.

    This class provides an interface to check whether an instance's evaluation id meets a given evaluation id.
    """

    def meets_eval_id(self, eval_id: tp.Optional[tp.Hashable]) -> bool:
        """Return whether the instance's evaluation id matches the provided evaluation id.

        Args:
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            bool: True if the instance's evaluation id satisfies the given evaluation id, False otherwise.
        """
        if self.eval_id is not None and eval_id is not None:
            if checks.is_complex_sequence(self.eval_id):
                if eval_id not in self.eval_id:
                    return False
            else:
                if eval_id != self.eval_id:
                    return False
        return True


class JupyterKernel(Configured):
    """Lightweight wrapper class around an IPython kernel process.

    Args:
        startup_timeout (int): Seconds to wait for the kernel to become ready.
        **manager_kwargs: Keyword arguments for the kernel manager.

    Example:

        ```pycon
        >>> with vbt.JupyterKernel() as kernel:
        ...     output = kernel.execute("print('Hello, world!')")
        ...     print(output)
        Hello, world!

        ...     output = kernel.execute("a = 2 + 2")
        ...     print(output)

        ...     output = kernel.execute("a")
        ...     print(output)
        4
        ...     output = kernel.execute("1 / 0")
        ...     print(output)
        ZeroDivisionError: division by zero
        ```
    """

    def __init__(self, startup_timeout: int = 60, **manager_kwargs) -> None:
        Configured.__init__(self, startup_timeout=startup_timeout, **manager_kwargs)

        if manager_kwargs is None:
            manager_kwargs = {}

        self._startup_timeout = startup_timeout
        self._manager_kwargs = manager_kwargs

        self._manager = None
        self._client = None

    @property
    def startup_timeout(self) -> int:
        """Seconds to wait for the kernel to become ready.

        Returns:
            int: Timeout in seconds.
        """
        return self._startup_timeout

    @property
    def manager_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for the kernel manager.

        Returns:
            KwargsLike: Dictionary of additional parameters.
        """
        return self._manager_kwargs

    @property
    def manager(self) -> tp.Optional[KernelManagerT]:
        """Kernel manager instance.

        Returns:
            KernelManager: Kernel manager object.
        """
        return self._manager

    @property
    def client(self) -> tp.Optional[KernelClientT]:
        """Kernel client instance.

        Returns:
            KernelClient: Kernel client object.
        """
        return self._client

    def collect_output(self, parent_msg_id: str, msg_timeout: tp.Optional[float]) -> str:
        """Aggregate all messages produced by a single kernel execution.

        Args:
            parent_msg_id (str): `msg_id` returned by `jupyter_client.KernelClient.execute`.

                Only messages whose `parent_header.msg_id` matches this ID are collected.
            msg_timeout (Optional[float]): Seconds to wait between `IOPub` messages.

                None waits indefinitely.

        Returns:
            str: Newline-joined string containing stdout/stderr, `repr` representations, and tracebacks.

                Rich MIME outputs are represented by placeholder stubs such as `<image/png: bytes>`
                so that all content is surfaced as text.
        """
        import queue

        outputs = []
        while True:
            try:
                msg = self.client.get_iopub_msg(timeout=msg_timeout)
            except queue.Empty:
                break
            if msg["parent_header"].get("msg_id") != parent_msg_id:
                continue
            mtype, content = msg["msg_type"], msg["content"]
            if mtype == "status" and content["execution_state"] == "idle":
                break
            if mtype in ("execute_result", "display_data"):
                for mime, value in content["data"].items():
                    if mime.startswith("text/"):
                        outputs.append(value if isinstance(value, str) else str(value))
                    else:
                        outputs.append(f"<{mime}: {type(value).__name__}>")
            elif mtype == "stream":
                outputs.append(content["text"])
            elif mtype == "error":
                outputs.extend(content["traceback"])
        return "\n".join(outputs)

    def execute(
        self,
        code: str,
        silent: bool = False,
        exec_timeout: tp.Optional[float] = None,
        interrupt_grace: float = 5.0,
        msg_timeout: tp.Optional[float] = None,
    ) -> str:
        """Run Python code inside the managed kernel.

        Args:
            code (str): Python source to execute.
            silent (bool): If True, the kernel will not increment its execution counter.

                This does not affect captured output.
            exec_timeout (Optional[float]): Seconds to wait for the code execution to complete.

                None means no timeout.
            interrupt_grace (float): Seconds to wait for the kernel to gracefully interrupt the execution.
            msg_timeout (Optional[float]): Seconds to wait between `IOPub` messages.

                None waits indefinitely.

        Returns:
            str: Combined textual output from the cell, including prints, returned `repr` values,
                and full tracebacks on error.
        """
        msg_id = self.client.execute(code, silent=silent)
        if exec_timeout is None:
            return self.collect_output(msg_id, msg_timeout)

        with cf.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.collect_output, msg_id, msg_timeout)
            try:
                return future.result(timeout=exec_timeout)
            except cf.TimeoutError:
                self.manager.interrupt_kernel()
                try:
                    output = future.result(timeout=interrupt_grace)
                    return output.rstrip("\n") + "\n\nTimeoutError: Kernel interrupted"
                except cf.TimeoutError:
                    self.restart()
                    return "TimeoutError: Kernel restarted"

    def start(self) -> None:
        """Start the underlying kernel process and open ZMQ channels.

        Returns:
            None
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("jupyter_client")
        from jupyter_client import KernelManager

        self._manager = KernelManager(**self.manager_kwargs)

        self.manager.start_kernel()
        self._client = self.manager.client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=self.startup_timeout)

    def restart(self, now: bool = True) -> None:
        """Restart the underlying kernel, clearing all interpreter state.

        Args:
            now (bool): Whether to force an immediate restart.

                If False, the request is politely sent and the method waits for
                the kernel to finish any pending operations first.

        Returns:
            None
        """
        self.manager.restart_kernel(now=now)
        self._client = self.manager.client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=self.startup_timeout)

    def shutdown(self, now: bool = True) -> None:
        """Terminate the kernel process and close all ZMQ channels.

        Args:
            now (bool): If True, the kernel is killed immediately; if False it's asked to shut down gracefully.

        Returns:
            None
        """
        self.client.stop_channels()
        self.manager.shutdown_kernel(now=now)

    def __enter__(self) -> tp.Self:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()


class VBTKernel(JupyterKernel):
    """Jupyter kernel that automatically star-imports `vectorbtpro` on startup."""

    def start(self) -> None:
        JupyterKernel.start(self)
        self.execute("from vectorbtpro import *")
