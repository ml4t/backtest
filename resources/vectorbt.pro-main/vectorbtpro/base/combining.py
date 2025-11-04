# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing functions for combining NumPy arrays.

The functions in this module combine two or more arrays using a custom function and
stack the results horizontally. This is essential for concatenating outputs from various
hyper-parameter combinations in the vectorbtpro package. All functions are available
in both Python and Numba-compiled forms.
"""

import numpy as np
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import jit_reg, register_jitted
from vectorbtpro.utils.execution import Task, execute
from vectorbtpro.utils.template import RepFunc

__all__ = []


@register_jitted
def custom_apply_and_concat_none_nb(
    indices: tp.Array1d,
    apply_func_nb: tp.R0ApplyFunc,
    *args,
) -> None:
    """Run the JIT-compiled function for each index for in-place operations.

    Args:
        indices (Array1d): 1D array of indices to iterate over.
        apply_func_nb (R0ApplyFunc): Callback function that accepts an index and additional arguments,
            and returns nothing.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        None
    """
    for i in indices:
        apply_func_nb(i, *args)


@register_jitted
def apply_and_concat_none_nb(
    ntimes: int,
    apply_func_nb: tp.R0ApplyFunc,
    *args,
) -> None:
    """Execute the JIT-compiled function multiple times for in-place operations.

    Args:
        ntimes (int): Number of times to execute `apply_func_nb`.
        apply_func_nb (R0ApplyFunc): Callback function that accepts an index and additional arguments,
            and returns nothing.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        None

    See:
        `custom_apply_and_concat_none_nb`
    """
    custom_apply_and_concat_none_nb(np.arange(ntimes), apply_func_nb, *args)


@register_jitted
def to_2d_one_nb(a: tp.Array) -> tp.Array2d:
    """Expand the input array to a 2D array along axis 1.

    Args:
        a (Array): Input array.

    Returns:
        Array2d: Array expanded along axis 1 if it was originally one-dimensional;
            otherwise, the original array.
    """
    if a.ndim > 1:
        return a
    return np.expand_dims(a, axis=1)


@register_jitted
def custom_apply_and_concat_one_nb(
    indices: tp.Array1d,
    apply_func_nb: tp.R1ApplyFunc,
    *args,
) -> tp.Array2d:
    """Execute a JIT-compiled function that returns a single array per index and
    horizontally concatenate the results.

    Args:
        indices (Array1d): 1D array of indices to iterate over.
        apply_func_nb (R1ApplyFunc): Callback function that accepts an index and additional arguments,
            and returns an array.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        Array2d: 2D array created by horizontally concatenating the arrays returned for each index.
    """
    output_0 = to_2d_one_nb(apply_func_nb(indices[0], *args))
    output = np.empty((output_0.shape[0], len(indices) * output_0.shape[1]), dtype=output_0.dtype)
    for i in range(len(indices)):
        if i == 0:
            outputs_i = output_0
        else:
            outputs_i = to_2d_one_nb(apply_func_nb(indices[i], *args))
        output[:, i * outputs_i.shape[1] : (i + 1) * outputs_i.shape[1]] = outputs_i
    return output


@register_jitted
def apply_and_concat_one_nb(
    ntimes: int,
    apply_func_nb: tp.R1ApplyFunc,
    *args,
) -> tp.Array2d:
    """Execute the JIT-compiled function multiple times, each returning a single array,
    and horizontally concatenate the results.

    Args:
        ntimes (int): Number of times to execute `apply_func_nb`.
        apply_func_nb (R1ApplyFunc): Callback function that accepts an index and additional arguments,
            and returns an array.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        Array2d: Concatenated 2D array of the outputs from each function call.

    See:
        `custom_apply_and_concat_one_nb`
    """
    return custom_apply_and_concat_one_nb(np.arange(ntimes), apply_func_nb, *args)


@register_jitted
def to_2d_multiple_nb(a: tp.Iterable[tp.Array]) -> tp.List[tp.Array2d]:
    """Expand the dimensions of each array in the iterable along axis 1.

    Args:
        a (Iterable[Array]): Iterable of arrays.

    Returns:
        List[Array2d]: List of arrays, each expanded along axis 1.
    """
    lst = list()
    for _a in a:
        lst.append(to_2d_one_nb(_a))
    return lst


@register_jitted
def custom_apply_and_concat_multiple_nb(
    indices: tp.Array1d,
    apply_func_nb: tp.RMApplyFunc,
    *args,
) -> tp.List[tp.Array2d]:
    """Execute a JIT-compiled function that returns multiple arrays per index and
    horizontally concatenate each corresponding output.

    Args:
        indices (Array1d): 1D array of indices to iterate over.
        apply_func_nb (RMApplyFunc): Callback function that accepts an index and additional arguments,
            and returns multiple arrays.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        List[Array2d]: List of 2D arrays, each resulting from horizontally
            concatenating the corresponding outputs across all calls.
    """
    outputs = list()
    outputs_0 = to_2d_multiple_nb(apply_func_nb(indices[0], *args))
    for j in range(len(outputs_0)):
        outputs.append(
            np.empty(
                (outputs_0[j].shape[0], len(indices) * outputs_0[j].shape[1]),
                dtype=outputs_0[j].dtype,
            )
        )
    for i in range(len(indices)):
        if i == 0:
            outputs_i = outputs_0
        else:
            outputs_i = to_2d_multiple_nb(apply_func_nb(indices[i], *args))
        for j in range(len(outputs_i)):
            outputs[j][:, i * outputs_i[j].shape[1] : (i + 1) * outputs_i[j].shape[1]] = outputs_i[
                j
            ]
    return outputs


@register_jitted
def apply_and_concat_multiple_nb(
    ntimes: int,
    apply_func_nb: tp.RMApplyFunc,
    *args,
) -> tp.List[tp.Array2d]:
    """Execute the JIT-compiled function multiple times, each returning multiple arrays,
    and horizontally concatenate the corresponding outputs.

    Args:
        ntimes (int): Number of times to execute `apply_func_nb`.
        apply_func_nb (RMApplyFunc): Callback function that accepts an index and additional arguments,
            and returns multiple arrays.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        List[Array2d]: List of 2D arrays, each representing the concatenated outputs for
            one of the multiple return arrays.

    See:
        `custom_apply_and_concat_multiple_nb`
    """
    return custom_apply_and_concat_multiple_nb(np.arange(ntimes), apply_func_nb, *args)


def apply_and_concat_each(
    tasks: tp.TasksLike,
    n_outputs: tp.Optional[int] = None,
    execute_kwargs: tp.KwargsLike = None,
) -> tp.Union[None, tp.Array2d, tp.List[tp.Array2d]]:
    """Execute a set of tasks and concatenate their outputs.

    Each task is executed using `vectorbtpro.utils.execution.execute`. The function determines the number
    of outputs produced and concatenates them accordingly:

    * If no output is produced, returns None.
    * If a single output is produced, returns a 2D array.
    * If multiple outputs are produced, returns a list of 2D arrays.

    Args:
        tasks (TasksLike): Tasks (i.e., functions with their arguments) to execute.
        n_outputs (Optional[int]): Number of arrays returned by each function call.
        execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

            See `vectorbtpro.utils.execution.execute`.

    Returns:
        Union[None, Array2d, List[Array2d]]:
            * None if no outputs are produced.
            * 2D array if a single output is produced.
            * List of 2D arrays if multiple outputs are produced.
    """
    from vectorbtpro.base.merging import column_stack_arrays

    if execute_kwargs is None:
        execute_kwargs = {}

    out = execute(tasks, **execute_kwargs)
    if n_outputs is None:
        if out[0] is None:
            n_outputs = 0
        elif isinstance(out[0], (tuple, list, List)):
            n_outputs = len(out[0])
        else:
            n_outputs = 1
    if n_outputs == 0:
        return None
    if n_outputs == 1:
        if isinstance(out[0], (tuple, list, List)) and len(out[0]) == 1:
            out = list(map(lambda x: x[0], out))
        return column_stack_arrays(out)
    return list(map(column_stack_arrays, zip(*out)))


def apply_and_concat(
    ntimes: int,
    apply_func: tp.CApplyFunc,
    *args,
    n_outputs: tp.Optional[int] = None,
    jitted_loop: bool = False,
    jitted_warmup: bool = False,
    execute_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Union[None, tp.Array2d, tp.List[tp.Array2d]]:
    """Run a function multiple times and concatenate the results based on the number
    of output arrays produced.

    The function `apply_func` must accept an index as its first parameter, followed by
    positional and keyword arguments. Depending on the `jitted_loop` flag, a JIT-compiled
    version of the iteration may be used.

    Args:
        ntimes (int): Number of times to execute `apply_func`.
        apply_func (CApplyFunc): Callback function for applying to each index.
        *args: Positional arguments for `apply_func`.
        n_outputs (Optional[int]): Number of arrays returned by each function call.
        jitted_loop (bool): Flag indicating whether to use a JIT-compiled loop.
        jitted_warmup (bool): If True, perform a warm-up call for the JIT-compiled function.
        execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

            See `vectorbtpro.utils.execution.execute`.
        **kwargs: Keyword arguments for `apply_func` when not using a JIT-compiled loop.

    Returns:
        Union[None, Array2d, List[Array2d]]:
            * None if no outputs are produced.
            * 2D array if a single output is produced.
            * List of 2D arrays if multiple outputs are produced.

    See:
        * `custom_apply_and_concat_none_nb` if `jitted_loop` is True and `n_outputs` is 0
        * `custom_apply_and_concat_one_nb` if `jitted_loop` is True and `n_outputs` is 1
        * `custom_apply_and_concat_multiple_nb` if `jitted_loop` is True and `n_outputs` > 1

    !!! note
        When `jitted_loop` is True, `n_outputs` must be provided as Numba does not support
        variable keyword arguments.
    """
    if jitted_loop:
        if n_outputs is None:
            raise ValueError("Jitted iteration requires n_outputs")
        if n_outputs == 0:
            func = jit_reg.resolve(custom_apply_and_concat_none_nb)
        elif n_outputs == 1:
            func = jit_reg.resolve(custom_apply_and_concat_one_nb)
        else:
            func = jit_reg.resolve(custom_apply_and_concat_multiple_nb)
        if jitted_warmup:
            func(np.array([0]), apply_func, *args, **kwargs)

        def _tasks_template(chunk_meta):
            tasks = []
            for _chunk_meta in chunk_meta:
                if _chunk_meta.indices is not None:
                    chunk_indices = np.asarray(_chunk_meta.indices)
                else:
                    if _chunk_meta.start is None or _chunk_meta.end is None:
                        raise ValueError("Each chunk must have a start and an end index")
                    chunk_indices = np.arange(_chunk_meta.start, _chunk_meta.end)
                tasks.append(Task(func, chunk_indices, apply_func, *args, **kwargs))
            return tasks

        tasks = RepFunc(_tasks_template)
    else:
        tasks = [(apply_func, (i, *args), kwargs) for i in range(ntimes)]
    if execute_kwargs is None:
        execute_kwargs = {}
    execute_kwargs["size"] = ntimes
    return apply_and_concat_each(
        tasks,
        n_outputs=n_outputs,
        execute_kwargs=execute_kwargs,
    )


@register_jitted
def select_and_combine_nb(
    i: int,
    obj: tp.Any,
    others: tp.Sequence,
    combine_func_nb: tp.CombineFunc,
    *args,
) -> tp.AnyArray:
    """Select an element from a sequence and combine it with an object using a JIT-compiled function.

    This function selects an element from `others` using the provided index and then combines `obj`
    with the selected element via `combine_func_nb`.

    Args:
        i (int): Index used to select an element from `others`.
        obj (Any): Primary object to combine.
        others (Sequence): Sequence of objects available for combination.
        combine_func_nb (CombineFunc): Callback function that accepts two objects and additional arguments,
            and returns a combined object.
        *args: Positional arguments for `combine_func_nb`.

    Returns:
        AnyArray: Result of combining `obj` with the selected element.
    """
    return combine_func_nb(obj, others[i], *args)


@register_jitted
def combine_and_concat_nb(
    obj: tp.Any,
    others: tp.Sequence,
    combine_func_nb: tp.CombineFunc,
    *args,
) -> tp.Array2d:
    """Combine and concatenate the given object with a sequence of objects using a
    Numba-compiled combination function.

    Args:
        obj (Any): Primary object to combine.
        others (Sequence): Sequence of objects to combine with.
        combine_func_nb (CombineFunc): Callback function that accepts two objects and additional arguments,
            and returns a combined object.
        *args: Positional arguments for `combine_func_nb`.

    Returns:
        Array2d: Concatenated result obtained by combining the objects.

    See:
        `apply_and_concat_one_nb`
    """
    return apply_and_concat_one_nb(
        len(others), select_and_combine_nb, obj, others, combine_func_nb, *args
    )


def select_and_combine(
    i: int,
    obj: tp.Any,
    others: tp.Sequence,
    combine_func: tp.PyCombineFunc,
    *args,
    **kwargs,
) -> tp.AnyArray:
    """Combine the primary object with an element from a sequence at a specified index
    using a combination function.

    Args:
        i (int): Index of the element in `others` to combine.
        obj (Any): Primary object to combine.
        others (Sequence): Sequence of objects.
        combine_func (PyCombineFunc): Callback function that accepts two objects and additional arguments,
            and returns a combined object.
        *args: Positional arguments for `combine_func`.
        **kwargs: Keyword arguments for `combine_func`.

    Returns:
        AnyArray: Result of combining `obj` with the element at the specified index.
    """
    return combine_func(obj, others[i], *args, **kwargs)


def combine_and_concat(
    obj: tp.Any,
    others: tp.Sequence,
    combine_func: tp.AnyCombineFunc,
    *args,
    jitted_loop: bool = False,
    **kwargs,
) -> tp.Array2d:
    """Combine the primary object with each element in a sequence using a specified
    combination function and concatenate the results.

    Args:
        obj (Any): Primary object to combine.
        others (Sequence): Sequence of objects to combine with.
        combine_func (AnyCombineFunc): Callback function that accepts two objects and additional arguments,
            and returns a combined object.
        *args: Positional arguments for `combine_func`.
        jitted_loop (bool): Flag indicating whether to use a JIT-compiled loop.
        **kwargs: Keyword arguments for `combine_func`.

    Returns:
        Array2d: Concatenated result obtained after combining the objects.

    See:
        * `select_and_combine_nb` if `jitted_loop` is True
    """
    if jitted_loop:
        apply_func = jit_reg.resolve(select_and_combine_nb)
    else:
        apply_func = select_and_combine
    return apply_and_concat(
        len(others),
        apply_func,
        obj,
        others,
        combine_func,
        *args,
        n_outputs=1,
        jitted_loop=jitted_loop,
        **kwargs,
    )


@register_jitted
def combine_multiple_nb(
    objs: tp.Sequence,
    combine_func_nb: tp.CombineFunc,
    *args,
) -> tp.Any:
    """Combine a sequence of objects pairwise using a Numba-compiled combination function.

    Args:
        objs (Sequence): Sequence of objects to combine.
        combine_func_nb (CombineFunc): Callback function that accepts two objects and additional arguments,
            and returns a combined object.
        *args: Positional arguments for `combine_func_nb`.

    Returns:
        Any: Result obtained by pairwise combining all objects in the sequence.
    """
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func_nb(result, objs[i], *args)
    return result


def combine_multiple(
    objs: tp.Sequence,
    combine_func: tp.PyCombineFunc,
    *args,
    jitted_loop: bool = False,
    **kwargs,
) -> tp.Any:
    """Combine a sequence of objects pairwise into a single object using a provided combination function.

    Args:
        objs (Sequence): Sequence of objects to combine.
        combine_func (PyCombineFunc): Callback function that accepts two objects and additional arguments,
            and returns a combined object.
        *args: Positional arguments for `combine_func`.
        jitted_loop (bool): Flag indicating whether to use a JIT-compiled loop.
        **kwargs: Keyword arguments for `combine_func`.

    Returns:
        Any: Combined result after pairwise merging of the objects.

    See:
        * `combine_multiple_nb` if `jitted_loop` is True

    !!! note
        Numba doesn't support variable keyword arguments.
    """
    if jitted_loop:
        func = jit_reg.resolve(combine_multiple_nb)
        return func(objs, combine_func, *args)
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func(result, objs[i], *args, **kwargs)
    return result
