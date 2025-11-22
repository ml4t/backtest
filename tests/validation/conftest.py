"""
Pytest configuration for validation tests.

These tests require optional dependencies (VectorBT, Backtrader, Zipline).
Tests are only skipped if they EXPLICITLY require a missing dependency.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_vectorbt_pro: Test requires VectorBT Pro (commercial, optional dependency)"
    )
    config.addinivalue_line(
        "markers",
        "requires_vectorbt: Test requires VectorBT OSS (optional dependency)"
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
    Skip tests based on EXPLICIT markers, not based on file content.

    Tests should use explicit markers like @pytest.mark.requires_vectorbt_pro
    if they require specific dependencies. This avoids over-aggressive skipping.
    """
    # Check which optional dependencies are available
    # NOTE: VectorBT OSS and Pro CANNOT coexist - they both register .vbt accessor.
    # If Pro is available, we skip importing OSS to avoid accessor collision.
    try:
        import vectorbtpro  # noqa: F401
        has_vectorbt_pro = True
        has_vectorbt_oss = False  # Don't import OSS if Pro is available
    except ImportError:
        has_vectorbt_pro = False
        try:
            import vectorbt  # noqa: F401
            has_vectorbt_oss = True
        except ImportError:
            has_vectorbt_oss = False

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

    # Skip tests with explicit markers when dependencies missing
    for item in items:
        # Skip if marked requires_vectorbt_pro but Pro not installed
        if item.get_closest_marker("requires_vectorbt_pro") and not has_vectorbt_pro:
            item.add_marker(pytest.mark.skip(reason="VectorBT Pro not installed"))

        # Skip if marked requires_vectorbt but OSS not installed
        if item.get_closest_marker("requires_vectorbt") and not has_vectorbt_oss:
            item.add_marker(pytest.mark.skip(reason="VectorBT OSS not installed"))

        # Skip if marked requires_backtrader but not installed
        if item.get_closest_marker("requires_backtrader") and not has_backtrader:
            item.add_marker(pytest.mark.skip(reason="Backtrader not installed"))

        # Skip if marked requires_zipline but not installed
        if item.get_closest_marker("requires_zipline") and not has_zipline:
            item.add_marker(pytest.mark.skip(reason="Zipline not installed"))
