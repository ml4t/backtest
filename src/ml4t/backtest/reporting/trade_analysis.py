"""Trade analysis utilities for post-backtest performance attribution.

This module provides utility functions for analyzing trade data from backtests,
including:
- Win rate and profitability metrics by exit rule
- Average hold time and duration analysis
- P&L attribution by rule and feature
- Rule effectiveness scoring
- Feature correlation with trade outcomes

All functions are Polars-native for performance with large trade datasets.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl


def win_rate_by_rule(trades: pl.DataFrame) -> dict[str, float]:
    """Calculate win rate (% profitable trades) for each exit rule.

    Args:
        trades: Polars DataFrame with 'exit_reason' and 'pnl' columns

    Returns:
        Dictionary mapping exit_reason to win_rate (0.0 to 1.0)

    Example:
        >>> trades = pl.DataFrame({
        ...     "exit_reason": ["stop_loss", "take_profit", "stop_loss", "time_stop"],
        ...     "pnl": [-100, 200, -50, 150]
        ... })
        >>> win_rate_by_rule(trades)
        {'stop_loss': 0.0, 'take_profit': 1.0, 'time_stop': 1.0}
    """
    if trades.is_empty():
        return {}

    # Filter completed trades (exit_reason not null)
    completed = trades.filter(pl.col("exit_reason").is_not_null())

    if completed.is_empty():
        return {}

    # Calculate win rate per rule
    result = (
        completed.group_by("exit_reason")
        .agg(
            [
                pl.col("pnl").is_not_null().sum().alias("count"),
                (pl.col("pnl") > 0).sum().alias("wins"),
            ]
        )
        .with_columns((pl.col("wins") / pl.col("count")).alias("win_rate"))
        .select(["exit_reason", "win_rate"])
    )

    return dict(zip(result["exit_reason"].to_list(), result["win_rate"].to_list()))


def avg_hold_time_by_rule(trades: pl.DataFrame) -> dict[str, timedelta]:
    """Calculate average hold time for each exit rule.

    Args:
        trades: Polars DataFrame with 'exit_reason' and 'duration_seconds' columns

    Returns:
        Dictionary mapping exit_reason to average timedelta

    Example:
        >>> trades = pl.DataFrame({
        ...     "exit_reason": ["stop_loss", "take_profit", "stop_loss"],
        ...     "duration_seconds": [3600, 7200, 5400]
        ... })
        >>> result = avg_hold_time_by_rule(trades)
        >>> result["stop_loss"].total_seconds()
        4500.0
    """
    if trades.is_empty():
        return {}

    # Check if required columns exist
    if "duration_seconds" not in trades.columns or "exit_reason" not in trades.columns:
        return {}

    # Filter completed trades with duration
    completed = trades.filter(
        pl.col("exit_reason").is_not_null() & pl.col("duration_seconds").is_not_null()
    )

    if completed.is_empty():
        return {}

    # Calculate average duration per rule
    result = (
        completed.group_by("exit_reason")
        .agg(pl.col("duration_seconds").mean().alias("avg_duration"))
        .select(["exit_reason", "avg_duration"])
    )

    return {
        rule: timedelta(seconds=avg_secs)
        for rule, avg_secs in zip(
            result["exit_reason"].to_list(), result["avg_duration"].to_list()
        )
    }


def pnl_attribution(trades: pl.DataFrame) -> dict[str, float]:
    """Calculate P&L contribution of each exit rule to total P&L.

    Args:
        trades: Polars DataFrame with 'exit_reason' and 'pnl' columns

    Returns:
        Dictionary mapping exit_reason to total P&L

    Example:
        >>> trades = pl.DataFrame({
        ...     "exit_reason": ["stop_loss", "take_profit", "stop_loss", "time_stop"],
        ...     "pnl": [-100, 200, -50, 150]
        ... })
        >>> pnl_attribution(trades)
        {'stop_loss': -150, 'take_profit': 200, 'time_stop': 150}
    """
    if trades.is_empty():
        return {}

    # Filter completed trades
    completed = trades.filter(
        pl.col("exit_reason").is_not_null() & pl.col("pnl").is_not_null()
    )

    if completed.is_empty():
        return {}

    # Sum P&L by rule
    result = (
        completed.group_by("exit_reason")
        .agg(pl.col("pnl").sum().alias("total_pnl"))
        .select(["exit_reason", "total_pnl"])
    )

    return dict(zip(result["exit_reason"].to_list(), result["total_pnl"].to_list()))


def rule_effectiveness(trades: pl.DataFrame) -> pl.DataFrame:
    """Calculate comprehensive effectiveness metrics for each exit rule.

    Combines multiple metrics into a single DataFrame for easy comparison:
    - trigger_count: Number of times rule triggered
    - win_count: Number of winning trades
    - win_rate: Win rate (0.0 to 1.0)
    - total_pnl: Total P&L from rule
    - avg_pnl: Average P&L per trade
    - avg_return_pct: Average return percentage
    - avg_duration_bars: Average hold time in bars

    Args:
        trades: Polars DataFrame with trade data

    Returns:
        Polars DataFrame with rule effectiveness metrics

    Example:
        >>> trades = pl.DataFrame({
        ...     "exit_reason": ["stop_loss", "take_profit", "stop_loss", "time_stop"],
        ...     "pnl": [-100, 200, -50, 150],
        ...     "return_pct": [-0.05, 0.10, -0.025, 0.075],
        ...     "duration_bars": [10, 20, 15, 25]
        ... })
        >>> result = rule_effectiveness(trades)
        >>> result.columns
        ['exit_reason', 'trigger_count', 'win_count', 'win_rate', 'total_pnl', 'avg_pnl', 'avg_return_pct', 'avg_duration_bars']
    """
    if trades.is_empty():
        return pl.DataFrame()

    # Filter completed trades
    completed = trades.filter(
        pl.col("exit_reason").is_not_null() & pl.col("pnl").is_not_null()
    )

    if completed.is_empty():
        return pl.DataFrame()

    # Calculate all metrics grouped by exit_reason
    result = completed.group_by("exit_reason").agg(
        [
            pl.col("pnl").count().alias("trigger_count"),
            (pl.col("pnl") > 0).sum().alias("win_count"),
            pl.col("pnl").sum().alias("total_pnl"),
            pl.col("pnl").mean().alias("avg_pnl"),
            pl.col("return_pct").mean().alias("avg_return_pct"),
            pl.col("duration_bars").mean().alias("avg_duration_bars"),
        ]
    )

    # Add win_rate calculation
    result = result.with_columns(
        (pl.col("win_count") / pl.col("trigger_count")).alias("win_rate")
    )

    # Sort by total_pnl descending
    result = result.sort("total_pnl", descending=True)

    return result.select(
        [
            "exit_reason",
            "trigger_count",
            "win_count",
            "win_rate",
            "total_pnl",
            "avg_pnl",
            "avg_return_pct",
            "avg_duration_bars",
        ]
    )


def feature_correlation(
    trades: pl.DataFrame, features: list[str] | None = None
) -> pl.DataFrame:
    """Calculate correlation between features and trade outcomes.

    Computes Pearson correlation between feature values at entry and:
    - Trade P&L
    - Return percentage
    - Trade duration

    Args:
        trades: Polars DataFrame with trade data including feature columns
        features: List of feature column names to analyze. If None, auto-detects
                 columns ending with '_entry' or '_exit'.

    Returns:
        Polars DataFrame with feature correlations

    Example:
        >>> trades = pl.DataFrame({
        ...     "pnl": [100, -50, 200, -75],
        ...     "return_pct": [0.05, -0.025, 0.10, -0.0375],
        ...     "atr_entry": [2.5, 3.0, 2.0, 3.5],
        ...     "volatility_entry": [0.02, 0.03, 0.015, 0.035]
        ... })
        >>> result = feature_correlation(trades)
        >>> result.columns
        ['feature', 'corr_pnl', 'corr_return_pct', 'corr_duration_bars']
    """
    if trades.is_empty():
        return pl.DataFrame()

    # Auto-detect features if not specified
    if features is None:
        features = [
            col
            for col in trades.columns
            if col.endswith("_entry") or col.endswith("_exit")
        ]

    if not features:
        return pl.DataFrame()

    # Filter out rows with null values in outcome columns
    completed = trades.filter(pl.col("pnl").is_not_null())

    if completed.is_empty():
        return pl.DataFrame()

    correlations = []

    for feature in features:
        if feature not in completed.columns:
            continue

        # Check if feature column has non-null values
        if completed[feature].null_count() == len(completed):
            continue

        # Calculate correlations
        corr_pnl = completed.select(pl.corr(feature, "pnl")).item()
        corr_return = (
            completed.select(pl.corr(feature, "return_pct")).item()
            if "return_pct" in completed.columns
            else None
        )
        corr_duration = (
            completed.select(pl.corr(feature, "duration_bars")).item()
            if "duration_bars" in completed.columns
            else None
        )

        correlations.append(
            {
                "feature": feature,
                "corr_pnl": corr_pnl,
                "corr_return_pct": corr_return,
                "corr_duration_bars": corr_duration,
            }
        )

    if not correlations:
        return pl.DataFrame()

    result = pl.DataFrame(correlations)

    # Sort by absolute correlation with P&L
    result = result.with_columns(pl.col("corr_pnl").abs().alias("abs_corr_pnl"))
    result = result.sort("abs_corr_pnl", descending=True).drop("abs_corr_pnl")

    return result


def analyze_trades(trades: pl.DataFrame) -> dict[str, Any]:
    """Comprehensive trade analysis returning all metrics in a single call.

    Convenience function that computes all analysis metrics and returns
    them in a structured dictionary.

    Args:
        trades: Polars DataFrame with trade data

    Returns:
        Dictionary containing:
        - summary: Overall statistics (total trades, win rate, total P&L, etc.)
        - by_rule: Rule effectiveness DataFrame
        - win_rates: Win rate by rule dict
        - hold_times: Average hold time by rule dict
        - pnl_attribution: P&L by rule dict
        - feature_correlations: Feature correlation DataFrame

    Example:
        >>> trades = pl.DataFrame({...})  # Trade data
        >>> results = analyze_trades(trades)
        >>> results["summary"]["total_pnl"]
        12500.0
        >>> results["by_rule"]  # DataFrame with rule effectiveness
    """
    if trades.is_empty():
        return {
            "summary": {},
            "by_rule": pl.DataFrame(),
            "win_rates": {},
            "hold_times": {},
            "pnl_attribution": {},
            "feature_correlations": pl.DataFrame(),
        }

    # Overall summary statistics
    completed = trades.filter(pl.col("pnl").is_not_null())

    if not completed.is_empty():
        summary = {
            "total_trades": len(completed),
            "winning_trades": int((completed["pnl"] > 0).sum()),
            "losing_trades": int((completed["pnl"] < 0).sum()),
            "overall_win_rate": float((completed["pnl"] > 0).mean()),
            "total_pnl": float(completed["pnl"].sum()),
            "avg_pnl": float(completed["pnl"].mean()),
            "median_pnl": float(completed["pnl"].median()),
            "max_win": float(completed["pnl"].max()),
            "max_loss": float(completed["pnl"].min()),
        }

        if "return_pct" in completed.columns:
            summary["avg_return_pct"] = float(completed["return_pct"].mean())
            summary["median_return_pct"] = float(completed["return_pct"].median())

        if "duration_bars" in completed.columns:
            summary["avg_duration_bars"] = float(completed["duration_bars"].mean())
            summary["median_duration_bars"] = float(completed["duration_bars"].median())
    else:
        summary = {}

    return {
        "summary": summary,
        "by_rule": rule_effectiveness(trades),
        "win_rates": win_rate_by_rule(trades),
        "hold_times": avg_hold_time_by_rule(trades),
        "pnl_attribution": pnl_attribution(trades),
        "feature_correlations": feature_correlation(trades),
    }
