"""Test data generators for validation scenarios.

All generators return (prices_df, entries, exits) or (prices_df, entries) tuples,
using pandas DataFrames with DatetimeIndex and numpy boolean arrays.

Only depends on stdlib + numpy + pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_random_walk(
    n_bars: int = 100,
    seed: int = 42,
    hold_bars: int = 5,
    trade_spacing: int = 10,
    base_price: float = 100.0,
    daily_vol: float = 0.02,
    use_nyse_calendar: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Standard random walk with alternating entry/exit signals.

    Used by: scenarios 01, 02, 05, 06, 07, 08.
    """
    np.random.seed(seed)

    returns = np.random.randn(n_bars) * daily_vol
    prices = base_price * np.exp(np.cumsum(returns))

    if use_nyse_calendar:
        import exchange_calendars as xcals

        nyse = xcals.get_calendar("XNYS")
        start = pd.Timestamp("2020-01-02")
        all_sessions = nyse.sessions_in_range(start, start + pd.Timedelta(days=n_bars * 2))
        dates = pd.DatetimeIndex(all_sessions[:n_bars]).tz_localize("UTC")
    else:
        dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)

    i = 0
    while i < n_bars - (hold_bars + 1):
        entries[i] = True
        exits[i + hold_bars] = True
        i += trade_spacing

    return df, entries, exits


def generate_short_signals(
    n_bars: int = 200,
    seed: int = 42,
    hold_bars: int = 10,
    trade_spacing: int = 20,
    start_bar: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Random walk with SHORT entry/exit signals.

    Used by: scenario 11 (short only).
    """
    np.random.seed(seed)

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02
    closes = base_price * np.exp(np.cumsum(returns))

    opens = closes * (1 + np.random.randn(n_bars) * 0.003)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_bars)) * 0.005)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )

    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)

    idx = start_bar
    while idx < n_bars - (hold_bars + 1):
        entries[idx] = True
        exits[idx + hold_bars] = True
        idx += trade_spacing

    return df, entries, exits


def generate_stop_loss_data(seed: int = 42) -> pd.DataFrame:
    """Deterministic declining price path to trigger stop-loss.

    Used by: scenario 03. Entry at bar 0 ($100), stop triggers at bar 5 ($94.50).
    """
    np.random.seed(seed)

    n_bars = 20
    closes = np.array([
        100.0, 99.0, 98.0, 97.0, 96.0,
        94.5, 93.0, 92.0, 91.0, 90.0,
        89.0, 88.0, 87.0, 86.0, 85.0,
        84.0, 83.0, 82.0, 81.0, 80.0,
    ])

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")
    opens = closes + 0.5
    highs = opens + 0.5
    lows = closes - 0.5

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n_bars, 100000),
        },
        index=dates,
    )


def generate_take_profit_data(seed: int = 42) -> pd.DataFrame:
    """Deterministic rising price path to trigger take-profit.

    Used by: scenario 04. Entry at bar 0 ($100), TP triggers around bar 6 ($111).
    """
    np.random.seed(seed)

    n_bars = 20
    closes = np.array([
        100.0, 101.0, 102.5, 104.0, 105.5,
        107.0, 111.0, 112.0, 113.0, 114.0,
        115.0, 116.0, 117.0, 118.0, 119.0,
        120.0, 121.0, 122.0, 123.0, 124.0,
    ])

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")
    opens = closes - 0.5
    highs = closes + 0.5
    lows = opens - 0.5

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n_bars, 100000),
        },
        index=dates,
    )


def generate_trending_data(
    n_bars: int = 100,
    seed: int = 42,
    entry_bars: list[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Trending data with controlled pullbacks for trailing stop scenarios.

    Used by: scenario 09 (trailing stop long).
    Returns (prices_df, entries) - no explicit exit signals.
    """
    np.random.seed(seed)
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 30:
            change = np.random.randn() * 0.01 + 0.005
        elif i < 35:
            change = -0.02 + np.random.randn() * 0.005
        elif i < 60:
            change = np.random.randn() * 0.01 + 0.003
        elif i < 65:
            change = -0.015 + np.random.randn() * 0.005
        else:
            change = np.random.randn() * 0.01 + 0.001
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    if entry_bars is None:
        entry_bars = [0, 40]

    entries = np.zeros(n_bars, dtype=bool)
    for b in entry_bars:
        if b < n_bars:
            entries[b] = True

    return df, entries


def generate_bracket_data(
    n_bars: int = 100,
    seed: int = 42,
    entry_bars: list[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Volatile data designed to trigger both SL and TP in bracket orders.

    Used by: scenario 10 (bracket order).
    """
    np.random.seed(seed)
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 8:
            change = np.random.randn() * 0.005 + 0.003
        elif i < 15:
            change = -0.015 + np.random.randn() * 0.005
        elif i < 30:
            change = np.random.randn() * 0.005 + 0.002
        elif i < 50:
            change = np.random.randn() * 0.005 + 0.008
        elif i < 70:
            change = -0.005 + np.random.randn() * 0.005
        else:
            change = np.random.randn() * 0.005 + 0.001
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    if entry_bars is None:
        entry_bars = [0, 20, 55]

    entries = np.zeros(n_bars, dtype=bool)
    for b in entry_bars:
        if b < n_bars:
            entries[b] = True

    return df, entries


def generate_short_trending_data(
    n_bars: int = 100,
    seed: int = 42,
    entry_bars: list[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Downtrending data for SHORT position trailing stop.

    Used by: scenario 12 (short trailing stop).
    """
    np.random.seed(seed)
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 30:
            change = np.random.randn() * 0.01 - 0.005  # Down for short profits
        elif i < 35:
            change = 0.02 + np.random.randn() * 0.005  # Up reversal triggers TSL
        elif i < 60:
            change = np.random.randn() * 0.01 - 0.003
        elif i < 65:
            change = 0.015 + np.random.randn() * 0.005
        else:
            change = np.random.randn() * 0.01 - 0.001
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    if entry_bars is None:
        entry_bars = [0, 40]

    entries = np.zeros(n_bars, dtype=bool)
    for b in entry_bars:
        if b < n_bars:
            entries[b] = True

    return df, entries


def generate_rule_combo_data(
    scenario: str = "tp_first",
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Data for rule combination scenarios (TSL+TP, TSL+SL, triple).

    Used by: scenarios 13, 14, 15.

    Args:
        scenario: "tp_first" (steady rise), "tsl_first" (rise then drop),
                  "sl_first" (immediate drop)
    """
    np.random.seed(seed)
    n_bars = 50

    if scenario == "tp_first":
        # Steady rise hits TP before TSL engages
        base = 100.0
        closes = [base]
        for _ in range(1, n_bars):
            closes.append(closes[-1] * 1.007)
        closes = np.array(closes)

    elif scenario == "tsl_first":
        # Rise to 6% then sharp drop triggers TSL at 5%
        base = 100.0
        closes = [base]
        for i in range(1, n_bars):
            if i < 10:
                closes.append(closes[-1] * 1.006)
            elif i == 10:
                closes.append(closes[-1] * 0.94)  # -6% drop
            else:
                closes.append(closes[-1] * 1.002)
        closes = np.array(closes)

    elif scenario == "sl_first":
        # Immediate drop triggers SL
        base = 100.0
        closes = [base]
        for i in range(1, n_bars):
            if i < 3:
                closes.append(closes[-1] * 0.97)  # -3% per bar
            else:
                closes.append(closes[-1] * 1.001)
        closes = np.array(closes)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")
    opens = closes - 0.3
    highs = closes + 0.5
    lows = closes - 0.5

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n_bars, 100000.0),
        },
        index=dates,
    )

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def generate_stress_data(
    n_bars: int = 1500,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extended market with multiple regimes for stress testing.

    Used by: scenario 16.
    9 regime transitions, 9 entry signals.
    """
    np.random.seed(seed)

    prices = [100.0]
    for i in range(1, n_bars):
        if i < 100:
            change = 0.002 + np.random.randn() * 0.01
        elif i < 200:
            change = -0.005 + np.random.randn() * 0.01
        elif i < 400:
            change = 0.003 + np.random.randn() * 0.01
        elif i < 500:
            change = np.random.randn() * 0.02
            if np.random.random() < 0.05:
                change += np.random.choice([-0.1, 0.1])
        elif i < 700:
            change = 0.004 + np.random.randn() * 0.01
        elif i < 750:
            change = -0.03 + np.random.randn() * 0.01
        elif i < 800:
            change = 0.02 + np.random.randn() * 0.01
        elif i < 1000:
            change = -0.002 + np.random.randn() * 0.01
        elif i < 1200:
            change = np.random.randn() * 0.04
        else:
            change = 0.002 + np.random.randn() * 0.01
        prices.append(max(prices[-1] * (1 + change), 1.0))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.003),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.008),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.008),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    entry_bars = [0, 100, 200, 400, 500, 700, 800, 1000, 1200]
    entries = np.zeros(n_bars, dtype=bool)
    for b in entry_bars:
        if b < n_bars:
            entries[b] = True

    return df, entries
