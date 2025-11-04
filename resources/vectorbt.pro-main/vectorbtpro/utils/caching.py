# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for caching operations, including clearing Python cache directories
and bytecode files, and managing cacheable properties and methods.

!!! info
    For default settings, see `vectorbtpro._settings.caching`.
"""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.path_ import remove_dir

__all__ = [
    "clear_pycache",
    "Cacheable",
]


def clear_pycache() -> None:
    """Clear `__pycache__` directories and Python bytecode files (.pyc) from the project directory.

    Returns:
        None
    """
    import pathlib

    for p in pathlib.Path(__file__).parent.parent.rglob("__pycache__"):
        remove_dir(p, with_contents=True)
    for p in pathlib.Path(__file__).parent.parent.rglob("*.py[co]"):
        p.unlink()


class Cacheable(Base):
    """Class for managing cacheable properties and methods.

    Required for registering `vectorbtpro.utils.decorators.cacheable_property` and
    `vectorbtpro.utils.decorators.cacheable_method`.

    See `vectorbtpro.registries.ca_registry` for details on the caching procedure.

    !!! info
        For default settings, see `vectorbtpro._settings.caching`.
    """

    def __init__(self) -> None:
        from vectorbtpro._settings import settings

        caching_cfg = settings["caching"]

        if not caching_cfg["register_lazily"]:
            instance_setup = self.get_ca_setup()
            if instance_setup is not None:
                for unbound_setup in instance_setup.unbound_setups:
                    unbound_setup.cacheable.get_ca_setup(self)

    @hybrid_method
    def get_ca_setup(cls_or_self) -> tp.Union["CAClassSetup", "CAInstanceSetup"]:
        """Retrieve the caching setup for the class or instance.

        If called on an instance, returns a `vectorbtpro.registries.ca_registry.CAInstanceSetup`.
        If called on a class, returns a `vectorbtpro.registries.ca_registry.CAClassSetup`.

        Returns:
            Union[CAClassSetup, CAInstanceSetup]: Caching setup corresponding to the caller.
        """
        from vectorbtpro.registries.ca_registry import CAClassSetup, CAInstanceSetup

        if isinstance(cls_or_self, type):
            return CAClassSetup.get(cls_or_self)
        return CAInstanceSetup.get(cls_or_self)
