"""
Pytest configuration for validation tests.

These tests require optional dependencies (VectorBT Pro, Backtrader, Zipline).
If dependencies aren't installed, tests are automatically skipped.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_vectorbt: Test requires VectorBT Pro (commercial, optional dependency)"
    )
    config.addinivalue_line(
        "markers",
        "requires_backtrader: Test requires Backtrader (optional dependency)"
    )
    config.addinivalue_line(
        "markers",
        "requires_zipline: Test requires Zipline (optional dependency)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests requiring optional dependencies when not installed.

    This allows validation tests to be run without failures when dependencies
    are missing. Tests will show as "skipped" instead of "failed".
    """
    # Check which optional dependencies are available
    try:
        import vectorbtpro  # noqa: F401
        has_vectorbt = True
    except ImportError:
        has_vectorbt = False

    try:
        import backtrader  # noqa: F401
        has_backtrader = True
    except ImportError:
        has_backtrader = False

    try:
        import zipline  # noqa: F401
        has_zipline = True
    except ImportError:
        has_zipline = False

    # Auto-skip tests based on imports in test files
    skip_vectorbt = pytest.mark.skip(reason="VectorBT Pro not installed (optional dependency)")
    skip_backtrader = pytest.mark.skip(reason="Backtrader not installed (optional dependency)")
    skip_zipline = pytest.mark.skip(reason="Zipline not installed (optional dependency)")

    for item in items:
        # Read test file to check for imports
        try:
            with open(item.fspath, 'r') as f:
                content = f.read()

            # Skip if test imports vectorbt/vectorbtpro but it's not installed
            if not has_vectorbt:
                if ('import vectorbt' in content or 'import vectorbtpro' in content or
                    'from vectorbt' in content or 'from common' in content):
                    item.add_marker(skip_vectorbt)

            # Skip if test imports backtrader but it's not installed
            if not has_backtrader:
                if 'import backtrader' in content or 'from backtrader' in content or 'from common' in content:
                    item.add_marker(skip_backtrader)

            # Skip if test imports zipline but it's not installed
            if not has_zipline:
                if 'import zipline' in content or 'from zipline' in content or 'from common' in content:
                    item.add_marker(skip_zipline)

        except Exception:
            # If we can't read the file, continue without skipping
            pass
