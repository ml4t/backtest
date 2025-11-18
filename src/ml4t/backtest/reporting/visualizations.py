"""Visualization utilities for trade analysis and performance attribution.

This module provides plotting functions for visualizing backtest results:
- Rule performance (win rate, average P&L)
- Hold time distributions
- Feature importance and correlations
- Exit reason breakdowns
- MAE/MFE scatter plots for exit efficiency

All functions use matplotlib with publication-ready styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from ml4t.backtest.reporting.trade_analysis import (
    feature_correlation,
    pnl_attribution,
    rule_effectiveness,
    win_rate_by_rule,
)


def _apply_publication_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent styling to plots for publication-ready output.

    Args:
        ax: Matplotlib axes object
        title: Plot title
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
    """
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_rule_performance(
    trades: pl.DataFrame,
    figsize: tuple[float, float] = (12, 6),
    save_path: Path | str | None = None,
) -> Figure:
    """Plot rule performance showing win rate and average P&L by exit rule.

    Creates a dual-axis bar chart with:
    - Left axis: Win rate (0-100%)
    - Right axis: Average P&L ($)

    Args:
        trades: Polars DataFrame with trade data (requires 'exit_reason', 'pnl' columns)
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> trades = pl.DataFrame({
        ...     "exit_reason": ["stop_loss", "take_profit", "stop_loss", "time_stop"],
        ...     "pnl": [-100, 200, -50, 150]
        ... })
        >>> fig = plot_rule_performance(trades)
        >>> plt.show()
    """
    if trades.is_empty():
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No trade data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Check required columns
    required_cols = ["exit_reason", "pnl"]
    missing_cols = [col for col in required_cols if col not in trades.columns]
    if missing_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            f"Missing required columns: {', '.join(missing_cols)}",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Get data using trade_analysis functions
    win_rates = win_rate_by_rule(trades)
    pnl_data = pnl_attribution(trades)

    if not win_rates or not pnl_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "Insufficient data for plotting",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Ensure matching rules between win_rates and pnl_data
    rules = sorted(set(win_rates.keys()) & set(pnl_data.keys()))
    if not rules:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No matching rules found",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    win_rate_values = [win_rates[rule] * 100 for rule in rules]  # Convert to percentage
    avg_pnl_values = [pnl_data[rule] for rule in rules]

    # Count trades per rule for context
    trade_counts = trades.group_by("exit_reason").agg(pl.count().alias("count"))
    count_dict = dict(zip(trade_counts["exit_reason"].to_list(), trade_counts["count"].to_list()))
    rule_labels = [f"{rule}\n(n={count_dict.get(rule, 0)})" for rule in rules]

    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(rules))
    width = 0.35

    # Plot win rate on left axis
    bars1 = ax1.bar(
        x_pos - width / 2,
        win_rate_values,
        width,
        label="Win Rate (%)",
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Win Rate (%)", fontsize=11, color="#2ecc71")
    ax1.tick_params(axis="y", labelcolor="#2ecc71", labelsize=9)
    ax1.set_ylim(0, 110)  # Give some headroom

    # Add percentage labels on bars
    for bar, value in zip(bars1, win_rate_values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#27ae60",
        )

    # Plot average P&L on right axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x_pos + width / 2,
        avg_pnl_values,
        width,
        label="Avg P&L ($)",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_ylabel("Average P&L ($)", fontsize=11, color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db", labelsize=9)

    # Add P&L labels on bars
    for bar, value in zip(bars2, avg_pnl_values):
        height = bar.get_height()
        y_offset = 5 if value >= 0 else -10
        va = "bottom" if value >= 0 else "top"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            f"${value:.0f}",
            ha="center",
            va=va,
            fontsize=8,
            color="#2980b9",
        )

    # Add zero line for P&L axis
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # Styling
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(rule_labels, fontsize=9)
    ax1.set_title("Rule Performance: Win Rate vs Average P&L", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="y")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_hold_time_distribution(
    trades: pl.DataFrame,
    bins: int = 30,
    figsize: tuple[float, float] = (10, 6),
    save_path: Path | str | None = None,
) -> Figure:
    """Plot histogram of trade hold times (duration in bars).

    Args:
        trades: Polars DataFrame with trade data (requires 'duration_bars' column)
        bins: Number of histogram bins
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> trades = pl.DataFrame({
        ...     "duration_bars": [10, 20, 15, 25, 30, 12, 18, 22]
        ... })
        >>> fig = plot_hold_time_distribution(trades)
        >>> plt.show()
    """
    if trades.is_empty() or "duration_bars" not in trades.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No duration data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Filter valid durations
    durations = trades.filter(pl.col("duration_bars").is_not_null())["duration_bars"].to_numpy()

    if len(durations) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No valid duration data",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)

    n, bins_edges, patches = ax.hist(
        durations,
        bins=bins,
        color="#9b59b6",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add statistics
    mean_duration = float(np.mean(durations))
    median_duration = float(np.median(durations))
    std_duration = float(np.std(durations))

    # Add vertical lines for mean and median
    ax.axvline(
        mean_duration,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_duration:.1f} bars",
    )
    ax.axvline(
        median_duration,
        color="#f39c12",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_duration:.1f} bars",
    )

    # Add text box with statistics
    stats_text = (
        f"Total Trades: {len(durations)}\n"
        f"Mean: {mean_duration:.1f} bars\n"
        f"Median: {median_duration:.1f} bars\n"
        f"Std Dev: {std_duration:.1f} bars\n"
        f"Min: {durations.min():.0f} bars\n"
        f"Max: {durations.max():.0f} bars"
    )
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    _apply_publication_style(ax, "Hold Time Distribution", "Hold Time (bars)", "Number of Trades")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_importance(
    trades: pl.DataFrame,
    top_n: int = 15,
    figsize: tuple[float, float] = (10, 8),
    save_path: Path | str | None = None,
) -> Figure:
    """Plot feature correlation with trade outcomes (P&L).

    Shows top N features ranked by absolute correlation with P&L.

    Args:
        trades: Polars DataFrame with trade data and feature columns
        top_n: Number of top features to display
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> trades = pl.DataFrame({
        ...     "pnl": [100, -50, 200, -75],
        ...     "atr_entry": [2.5, 3.0, 2.0, 3.5],
        ...     "volatility_entry": [0.02, 0.03, 0.015, 0.035]
        ... })
        >>> fig = plot_feature_importance(trades)
        >>> plt.show()
    """
    # Get feature correlations
    correlations_df = feature_correlation(trades)

    if correlations_df.is_empty():
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No feature data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Get top N features
    top_features = correlations_df.head(min(top_n, len(correlations_df)))

    features = top_features["feature"].to_list()
    correlations = top_features["corr_pnl"].to_numpy()  # Use corr_pnl column

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=figsize)

    # Color bars based on correlation sign
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in correlations]

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, correlations, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, correlations)):
        x_offset = 0.01 if value >= 0 else -0.01
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=ha,
            fontsize=8,
        )

    # Add zero line
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()  # Highest correlation at top

    _apply_publication_style(
        ax,
        f"Feature Importance: Correlation with P&L (Top {len(features)})",
        "Correlation Coefficient",
        "",
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.7, label="Positive Correlation"),
        Patch(facecolor="#e74c3c", alpha=0.7, label="Negative Correlation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_exit_reasons(
    trades: pl.DataFrame,
    figsize: tuple[float, float] = (10, 8),
    save_path: Path | str | None = None,
) -> Figure:
    """Plot pie chart showing distribution of exit reasons.

    Args:
        trades: Polars DataFrame with trade data (requires 'exit_reason' column)
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> trades = pl.DataFrame({
        ...     "exit_reason": ["stop_loss", "take_profit", "stop_loss", "time_stop", "take_profit"]
        ... })
        >>> fig = plot_exit_reasons(trades)
        >>> plt.show()
    """
    if trades.is_empty() or "exit_reason" not in trades.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No exit reason data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Count exit reasons
    exit_counts = trades.filter(pl.col("exit_reason").is_not_null()).group_by("exit_reason").agg(
        pl.count().alias("count")
    )

    if exit_counts.is_empty():
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No valid exit reasons",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Sort by count descending
    exit_counts = exit_counts.sort("count", descending=True)

    reasons = exit_counts["exit_reason"].to_list()
    counts = exit_counts["count"].to_numpy()

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(reasons)))  # type: ignore

    # Create pie chart
    fig, ax = plt.subplots(figsize=figsize)

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=reasons,
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct * sum(counts) / 100)})",
        colors=colors,
        startangle=90,
        explode=[0.05 if i == 0 else 0 for i in range(len(reasons))],  # Explode largest slice
        shadow=True,
    )

    # Enhance text
    for text in texts:
        text.set_fontsize(10)
        text.set_weight("bold")

    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(9)
        autotext.set_weight("bold")

    ax.set_title(
        f"Exit Reason Distribution (Total: {sum(counts)} trades)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_mfe_mae_scatter(
    trades: pl.DataFrame,
    figsize: tuple[float, float] = (10, 8),
    save_path: Path | str | None = None,
) -> Figure:
    """Plot scatter of Maximum Favorable Excursion vs Maximum Adverse Excursion.

    This visualization helps identify exit efficiency:
    - Points above diagonal: Exited too early (left profits on table)
    - Points near diagonal: Good exit timing
    - Points with large MAE: Wide stops or poor risk management

    Args:
        trades: Polars DataFrame with trade data (requires 'mfe', 'mae', 'pnl' columns)
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> trades = pl.DataFrame({
        ...     "mfe": [150, 80, 220, 50],
        ...     "mae": [-30, -60, -40, -80],
        ...     "pnl": [100, -50, 200, -75]
        ... })
        >>> fig = plot_mfe_mae_scatter(trades)
        >>> plt.show()
    """
    required_cols = ["mfe", "mae", "pnl"]
    if trades.is_empty() or not all(col in trades.columns for col in required_cols):
        fig, ax = plt.subplots(figsize=figsize)
        missing = [col for col in required_cols if col not in trades.columns]
        msg = f"Missing columns: {', '.join(missing)}" if missing else "No trade data available"
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Filter valid data
    valid_trades = trades.filter(
        pl.col("mfe").is_not_null() & pl.col("mae").is_not_null() & pl.col("pnl").is_not_null()
    )

    if valid_trades.is_empty():
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No valid MFE/MAE data",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    mfe = valid_trades["mfe"].to_numpy()
    mae = valid_trades["mae"].to_numpy()
    pnl = valid_trades["pnl"].to_numpy()

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize)

    # Color by P&L (winners green, losers red)
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnl]

    scatter = ax.scatter(
        mae,
        mfe,
        c=colors,
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add diagonal line (MAE = MFE would mean perfect exit timing)
    # But since MAE is negative, we draw from (min_mae, -min_mae) to (0, 0)
    min_val = min(mae.min(), -mfe.max())
    max_val = max(mae.max(), -mfe.min())
    ax.plot([min_val, 0], [0, -min_val], "k--", linewidth=1, alpha=0.5, label="Perfect Exit")

    # Add quadrant lines
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    _apply_publication_style(
        ax,
        "Exit Efficiency: MFE vs MAE",
        "Maximum Adverse Excursion (MAE) $",
        "Maximum Favorable Excursion (MFE) $",
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.6, label=f"Winners (n={sum(1 for p in pnl if p > 0)})"),
        Patch(facecolor="#e74c3c", alpha=0.6, label=f"Losers (n={sum(1 for p in pnl if p <= 0)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Add interpretation text
    interpretation = (
        "Interpretation:\n"
        "• Upper left: Large unrealized profit, exited early\n"
        "• Lower right: Large drawdown, poor risk mgmt\n"
        "• Near diagonal: Efficient exits"
    )
    ax.text(
        0.97,
        0.03,
        interpretation,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
