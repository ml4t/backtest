"""Configuration constants for QEngine.

This module contains all magic numbers and configuration defaults used throughout
the backtesting engine. Centralizing these values improves maintainability and
makes it easier to adjust behavior without modifying core logic.
"""

# =============================================================================
# Capital and Portfolio Defaults
# =============================================================================

DEFAULT_INITIAL_CAPITAL = 100_000.0
"""Default starting capital for backtests (USD)."""

DEFAULT_CURRENCY = "USD"
"""Default currency for portfolio accounting."""

# =============================================================================
# Event Processing
# =============================================================================

PROGRESS_LOG_INTERVAL = 10_000
"""Log progress message every N events during backtesting."""

# =============================================================================
# Order Execution
# =============================================================================

MIN_FILL_SIZE = 0.01
"""Minimum meaningful fill quantity (shares/contracts)."""

MAX_COMMISSION_CALC_ITERATIONS = 10
"""Maximum iterations for binary search in commission calculations."""

# =============================================================================
# Commission Defaults (basis points, where 1 bp = 0.01%)
# =============================================================================

DEFAULT_COMMISSION_RATE_BPS = 10
"""Default commission rate: 10 basis points = 0.1% = $0.001 per dollar."""

DEFAULT_TAKER_FEE_BPS = 10
"""Default taker fee for maker-taker exchanges: 10 bps = 0.1%."""

# =============================================================================
# Slippage Defaults (basis points)
# =============================================================================

SLIPPAGE_EQUITY_BPS = 1
"""Default slippage for equities: 1 basis point = 0.01%."""

SLIPPAGE_FUTURES_BPS = 2
"""Default slippage for futures: 2 basis points = 0.02%."""

SLIPPAGE_FX_BPS = 0.5
"""Default slippage for FX: 0.5 basis points = 0.005%."""

SLIPPAGE_CRYPTO_BPS = 10
"""Default slippage for cryptocurrencies: 10 basis points = 0.1%."""

# =============================================================================
# Helper Functions
# =============================================================================


def bps_to_decimal(bps: float) -> float:
    """Convert basis points to decimal representation.

    Args:
        bps: Value in basis points (e.g., 10 for 0.1%)

    Returns:
        Decimal representation (e.g., 0.001 for 10 bps)

    Examples:
        >>> bps_to_decimal(1)
        0.0001
        >>> bps_to_decimal(10)
        0.001
        >>> bps_to_decimal(100)
        0.01
    """
    return bps / 10_000


def decimal_to_bps(decimal: float) -> float:
    """Convert decimal to basis points.

    Args:
        decimal: Decimal value (e.g., 0.001 for 10 bps)

    Returns:
        Basis points (e.g., 10 for 0.001)

    Examples:
        >>> decimal_to_bps(0.0001)
        1.0
        >>> decimal_to_bps(0.001)
        10.0
        >>> decimal_to_bps(0.01)
        100.0
    """
    return decimal * 10_000
