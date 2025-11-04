# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for hashing."""

from functools import cached_property as cachedproperty

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

__all__ = []


class Hashable(Base):
    """Class for representing hashable objects using a custom hash key."""

    @staticmethod
    def get_hash(*args, **kwargs) -> int:
        """Compute a hash value based on the provided arguments.

        Args:
            *args: Positional arguments for hash computation.
            **kwargs: Keyword arguments for hash computation.

        Returns:
            int: Computed hash value.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def hash_key(self) -> tuple:
        """Unique key used for computing the instance's hash.

        Returns:
            tuple: Unique key for hash computation.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @cachedproperty
    def hash(self) -> int:
        """Computed hash value of the instance based on its hash key.

        Returns:
            int: Computed hash value.
        """
        return hash(self.hash_key)

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, type(self)):
            return self.hash_key == other.hash_key
        raise NotImplementedError


class UnhashableArgsError(Exception):
    """Exception raised for unhashable arguments."""

    pass


def hash_args(
    func: tp.Callable,
    args: tp.Args,
    kwargs: tp.Kwargs,
    ignore_args: tp.Optional[tp.Iterable[tp.AnnArgQuery]] = None,
) -> int:
    """Compute a hash based on the annotated arguments of `func`.

    Args:
        func (Callable): Function whose arguments are processed.
        args (Args): Positional arguments for `func`.
        kwargs (Kwargs): Keyword arguments for `func`.
        ignore_args (Optional[Iterable[AnnArgQuery]]): Sequence of queries for arguments to ignore.

    Returns:
        int: Computed hash value.
    """
    from vectorbtpro.utils.parsing import annotate_args, flatten_ann_args, ignore_flat_ann_args

    if ignore_args is None:
        ignore_args = []
    ann_args = annotate_args(func, args, kwargs, only_passed=True)
    flat_ann_args = flatten_ann_args(ann_args)
    if len(ignore_args) > 0:
        flat_ann_args = ignore_flat_ann_args(flat_ann_args, ignore_args)
    try:
        return hash(tuple(map(lambda x: (x[0], x[1]["value"]), flat_ann_args.items())))
    except TypeError:
        raise UnhashableArgsError
