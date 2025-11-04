# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing version and release information for vectorbtpro.

This module attempts to read the version from the project's pyproject.toml file,
and if unsuccessful, it falls back on using `importlib.metadata` for package version detection.
"""

try:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    from pathlib import Path

    with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    __version__ = pyproject["project"]["version"]
except Exception:
    import importlib.metadata

    __version__ = importlib.metadata.version(__package__ or __name__)

__release__ = "v" + __version__

__all__ = [
    "__version__",
    "__release__",
]
