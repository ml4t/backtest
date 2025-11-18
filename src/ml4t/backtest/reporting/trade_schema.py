"""Comprehensive trade recording schema with ML, risk, and context fields.

This module defines the schema for detailed trade records that capture:
- Entry/exit execution details (timestamps, prices, quantities)
- ML signals and predictions at entry and exit
- Technical indicators and features at entry and exit
- Risk management decisions (stop-loss, take-profit, exit reasons)
- Market context (VIX, regime, sector performance)

The schema is designed for:
- Post-backtest analysis and debugging
- ML model evaluation and feature importance analysis
- Risk management review and optimization
- Strategy performance attribution
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl


class ExitReason(str, Enum):
    """Enumeration of trade exit reasons for analysis."""

    SIGNAL = "signal"  # Normal signal-based exit
    STOP_LOSS = "stop_loss"  # Stop-loss triggered
    TAKE_PROFIT = "take_profit"  # Take-profit triggered
    TIME_STOP = "time_stop"  # Maximum hold time exceeded
    RISK_RULE = "risk_rule"  # Risk rule triggered (VIX, volatility, etc)
    POSITION_SIZE = "position_size"  # Position sizing constraint
    END_OF_DATA = "end_of_data"  # Backtest ended with open position
    MANUAL = "manual"  # Manual exit (e.g., user intervention)
    UNKNOWN = "unknown"  # Unknown or unspecified


@dataclass
class MLTradeRecord:
    """Enhanced trade record with ML signals, risk management, and context.

    This schema captures comprehensive information about each trade for
    post-backtest analysis and ML model evaluation.

    Core Trade Details:
        trade_id: Unique trade identifier
        asset_id: Asset symbol/identifier
        direction: "long" or "short"

    Entry Details:
        entry_dt: Entry timestamp
        entry_price: Entry fill price
        entry_quantity: Entry quantity (always positive)
        entry_commission: Entry commission cost
        entry_slippage: Entry slippage cost
        entry_order_id: Entry order identifier

    Exit Details:
        exit_dt: Exit timestamp
        exit_price: Exit fill price
        exit_quantity: Exit quantity (should match entry_quantity)
        exit_commission: Exit commission cost
        exit_slippage: Exit slippage cost
        exit_order_id: Exit order identifier
        exit_reason: Reason for exit (see ExitReason enum)

    Trade Metrics:
        pnl: Net profit/loss (after all costs)
        return_pct: Return percentage on capital at risk
        duration_bars: Number of bars held
        duration_seconds: Hold time in seconds

    ML Signals (Entry):
        ml_score_entry: ML model score/prediction at entry
        predicted_return_entry: Predicted return at entry
        confidence_entry: Model confidence at entry (0-1)

    ML Signals (Exit):
        ml_score_exit: ML model score/prediction at exit
        predicted_return_exit: Predicted return at exit
        confidence_exit: Model confidence at exit (0-1)

    Technical Indicators (Entry):
        atr_entry: Average True Range at entry
        volatility_entry: Realized volatility at entry
        momentum_entry: Momentum indicator at entry
        rsi_entry: RSI indicator at entry

    Technical Indicators (Exit):
        atr_exit: Average True Range at exit
        volatility_exit: Realized volatility at exit
        momentum_exit: Momentum indicator at exit
        rsi_exit: RSI indicator at exit

    Risk Management:
        stop_loss_price: Stop-loss price (if set)
        take_profit_price: Take-profit price (if set)
        risk_reward_ratio: Risk/reward ratio at entry
        position_size_pct: Position size as % of portfolio

    Market Context (Entry):
        vix_entry: VIX level at entry
        market_regime_entry: Market regime at entry (e.g., "bull", "bear", "sideways")
        sector_performance_entry: Sector performance at entry

    Market Context (Exit):
        vix_exit: VIX level at exit
        market_regime_exit: Market regime at exit
        sector_performance_exit: Sector performance at exit

    Additional Metadata:
        metadata: Dictionary for any additional custom fields
    """

    # Core trade details
    trade_id: int
    asset_id: str
    direction: str  # "long" or "short"

    # Entry details
    entry_dt: datetime
    entry_price: float
    entry_quantity: float
    entry_commission: float = 0.0
    entry_slippage: float = 0.0
    entry_order_id: str = ""

    # Exit details
    exit_dt: datetime | None = None
    exit_price: float | None = None
    exit_quantity: float | None = None
    exit_commission: float = 0.0
    exit_slippage: float = 0.0
    exit_order_id: str = ""
    exit_reason: ExitReason = ExitReason.UNKNOWN

    # Trade metrics
    pnl: float | None = None
    return_pct: float | None = None
    duration_bars: int | None = None
    duration_seconds: float | None = None

    # ML signals at entry
    ml_score_entry: float | None = None
    predicted_return_entry: float | None = None
    confidence_entry: float | None = None

    # ML signals at exit
    ml_score_exit: float | None = None
    predicted_return_exit: float | None = None
    confidence_exit: float | None = None

    # Technical indicators at entry
    atr_entry: float | None = None
    volatility_entry: float | None = None
    momentum_entry: float | None = None
    rsi_entry: float | None = None

    # Technical indicators at exit
    atr_exit: float | None = None
    volatility_exit: float | None = None
    momentum_exit: float | None = None
    rsi_exit: float | None = None

    # Risk management
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    risk_reward_ratio: float | None = None
    position_size_pct: float | None = None

    # Market context at entry
    vix_entry: float | None = None
    market_regime_entry: str | None = None
    sector_performance_entry: float | None = None

    # Market context at exit
    vix_exit: float | None = None
    market_regime_exit: str | None = None
    sector_performance_exit: float | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction.

        Returns:
            Dictionary with all trade fields, excluding metadata.
        """
        return {
            # Core trade details
            "trade_id": self.trade_id,
            "asset_id": self.asset_id,
            "direction": self.direction,
            # Entry details
            "entry_dt": self.entry_dt,
            "entry_price": self.entry_price,
            "entry_quantity": self.entry_quantity,
            "entry_commission": self.entry_commission,
            "entry_slippage": self.entry_slippage,
            "entry_order_id": self.entry_order_id,
            # Exit details
            "exit_dt": self.exit_dt,
            "exit_price": self.exit_price,
            "exit_quantity": self.exit_quantity,
            "exit_commission": self.exit_commission,
            "exit_slippage": self.exit_slippage,
            "exit_order_id": self.exit_order_id,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            # Trade metrics
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "duration_bars": self.duration_bars,
            "duration_seconds": self.duration_seconds,
            # ML signals at entry
            "ml_score_entry": self.ml_score_entry,
            "predicted_return_entry": self.predicted_return_entry,
            "confidence_entry": self.confidence_entry,
            # ML signals at exit
            "ml_score_exit": self.ml_score_exit,
            "predicted_return_exit": self.predicted_return_exit,
            "confidence_exit": self.confidence_exit,
            # Technical indicators at entry
            "atr_entry": self.atr_entry,
            "volatility_entry": self.volatility_entry,
            "momentum_entry": self.momentum_entry,
            "rsi_entry": self.rsi_entry,
            # Technical indicators at exit
            "atr_exit": self.atr_exit,
            "volatility_exit": self.volatility_exit,
            "momentum_exit": self.momentum_exit,
            "rsi_exit": self.rsi_exit,
            # Risk management
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "risk_reward_ratio": self.risk_reward_ratio,
            "position_size_pct": self.position_size_pct,
            # Market context at entry
            "vix_entry": self.vix_entry,
            "market_regime_entry": self.market_regime_entry,
            "sector_performance_entry": self.sector_performance_entry,
            # Market context at exit
            "vix_exit": self.vix_exit,
            "market_regime_exit": self.market_regime_exit,
            "sector_performance_exit": self.sector_performance_exit,
        }


def get_schema() -> dict[str, pl.DataType]:
    """Get Polars schema for MLTradeRecord.

    Returns:
        Dictionary mapping column names to Polars data types.
    """
    return {
        # Core trade details
        "trade_id": pl.Int64,
        "asset_id": pl.Utf8,
        "direction": pl.Utf8,
        # Entry details
        "entry_dt": pl.Datetime,
        "entry_price": pl.Float64,
        "entry_quantity": pl.Float64,
        "entry_commission": pl.Float64,
        "entry_slippage": pl.Float64,
        "entry_order_id": pl.Utf8,
        # Exit details
        "exit_dt": pl.Datetime,
        "exit_price": pl.Float64,
        "exit_quantity": pl.Float64,
        "exit_commission": pl.Float64,
        "exit_slippage": pl.Float64,
        "exit_order_id": pl.Utf8,
        "exit_reason": pl.Utf8,
        # Trade metrics
        "pnl": pl.Float64,
        "return_pct": pl.Float64,
        "duration_bars": pl.Int64,
        "duration_seconds": pl.Float64,
        # ML signals at entry
        "ml_score_entry": pl.Float64,
        "predicted_return_entry": pl.Float64,
        "confidence_entry": pl.Float64,
        # ML signals at exit
        "ml_score_exit": pl.Float64,
        "predicted_return_exit": pl.Float64,
        "confidence_exit": pl.Float64,
        # Technical indicators at entry
        "atr_entry": pl.Float64,
        "volatility_entry": pl.Float64,
        "momentum_entry": pl.Float64,
        "rsi_entry": pl.Float64,
        # Technical indicators at exit
        "atr_exit": pl.Float64,
        "volatility_exit": pl.Float64,
        "momentum_exit": pl.Float64,
        "rsi_exit": pl.Float64,
        # Risk management
        "stop_loss_price": pl.Float64,
        "take_profit_price": pl.Float64,
        "risk_reward_ratio": pl.Float64,
        "position_size_pct": pl.Float64,
        # Market context at entry
        "vix_entry": pl.Float64,
        "market_regime_entry": pl.Utf8,
        "sector_performance_entry": pl.Float64,
        # Market context at exit
        "vix_exit": pl.Float64,
        "market_regime_exit": pl.Utf8,
        "sector_performance_exit": pl.Float64,
    }


def trades_to_polars(trades: list[MLTradeRecord]) -> pl.DataFrame:
    """Convert list of MLTradeRecord to Polars DataFrame.

    Args:
        trades: List of trade records

    Returns:
        Polars DataFrame with trade data
    """
    if not trades:
        return pl.DataFrame(schema=get_schema())

    # Convert to dictionaries
    trade_dicts = [t.to_dict() for t in trades]

    # Create DataFrame (Polars infers types efficiently)
    return pl.DataFrame(trade_dicts)


def polars_to_trades(df: pl.DataFrame) -> list[MLTradeRecord]:
    """Convert Polars DataFrame to list of MLTradeRecord.

    Args:
        df: Polars DataFrame with trade data

    Returns:
        List of trade records
    """
    if len(df) == 0:
        return []

    trades = []
    for row in df.iter_rows(named=True):
        # Convert exit_reason string back to enum
        exit_reason = ExitReason(row["exit_reason"]) if row["exit_reason"] else ExitReason.UNKNOWN

        trade = MLTradeRecord(
            # Core trade details
            trade_id=row["trade_id"],
            asset_id=row["asset_id"],
            direction=row["direction"],
            # Entry details
            entry_dt=row["entry_dt"],
            entry_price=row["entry_price"],
            entry_quantity=row["entry_quantity"],
            entry_commission=row["entry_commission"],
            entry_slippage=row["entry_slippage"],
            entry_order_id=row["entry_order_id"],
            # Exit details
            exit_dt=row["exit_dt"],
            exit_price=row["exit_price"],
            exit_quantity=row["exit_quantity"],
            exit_commission=row["exit_commission"],
            exit_slippage=row["exit_slippage"],
            exit_order_id=row["exit_order_id"],
            exit_reason=exit_reason,
            # Trade metrics
            pnl=row["pnl"],
            return_pct=row["return_pct"],
            duration_bars=row["duration_bars"],
            duration_seconds=row["duration_seconds"],
            # ML signals at entry
            ml_score_entry=row["ml_score_entry"],
            predicted_return_entry=row["predicted_return_entry"],
            confidence_entry=row["confidence_entry"],
            # ML signals at exit
            ml_score_exit=row["ml_score_exit"],
            predicted_return_exit=row["predicted_return_exit"],
            confidence_exit=row["confidence_exit"],
            # Technical indicators at entry
            atr_entry=row["atr_entry"],
            volatility_entry=row["volatility_entry"],
            momentum_entry=row["momentum_entry"],
            rsi_entry=row["rsi_entry"],
            # Technical indicators at exit
            atr_exit=row["atr_exit"],
            volatility_exit=row["volatility_exit"],
            momentum_exit=row["momentum_exit"],
            rsi_exit=row["rsi_exit"],
            # Risk management
            stop_loss_price=row["stop_loss_price"],
            take_profit_price=row["take_profit_price"],
            risk_reward_ratio=row["risk_reward_ratio"],
            position_size_pct=row["position_size_pct"],
            # Market context at entry
            vix_entry=row["vix_entry"],
            market_regime_entry=row["market_regime_entry"],
            sector_performance_entry=row["sector_performance_entry"],
            # Market context at exit
            vix_exit=row["vix_exit"],
            market_regime_exit=row["market_regime_exit"],
            sector_performance_exit=row["sector_performance_exit"],
        )
        trades.append(trade)

    return trades


def export_parquet(
    trades: list[MLTradeRecord] | pl.DataFrame,
    path: Path | str,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    """Export trades to Parquet file with compression.

    Args:
        trades: List of trade records or Polars DataFrame
        path: Output file path
        compression: Compression algorithm ("zstd", "snappy", "gzip", "lz4", "uncompressed")
        compression_level: Compression level (1-22 for zstd, higher = more compression)

    Raises:
        ValueError: If trades is empty or invalid type
    """
    # Convert to DataFrame if needed
    if isinstance(trades, list):
        df = trades_to_polars(trades)
    elif isinstance(trades, pl.DataFrame):
        df = trades
    else:
        raise ValueError(f"trades must be list[MLTradeRecord] or pl.DataFrame, got {type(trades)}")

    if len(df) == 0:
        raise ValueError("Cannot export empty trades DataFrame")

    # Ensure path is Path object
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to Parquet
    df.write_parquet(
        output_path,
        compression=compression,
        compression_level=compression_level,
    )


def import_parquet(path: Path | str) -> pl.DataFrame:
    """Import trades from Parquet file.

    Args:
        path: Input file path

    Returns:
        Polars DataFrame with trade data

    Raises:
        FileNotFoundError: If file does not exist
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {input_path}")

    return pl.read_parquet(input_path)


def append_trades(
    new_trades: list[MLTradeRecord] | pl.DataFrame,
    path: Path | str,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    """Append trades to existing Parquet file (incremental writes).

    This function reads the existing file, appends new trades, and writes back.
    For large files, this can be memory-intensive. Consider using a different
    approach (e.g., write to separate files and concatenate later) for very
    large datasets.

    Args:
        new_trades: List of new trade records or Polars DataFrame
        path: Parquet file path
        compression: Compression algorithm
        compression_level: Compression level

    Raises:
        ValueError: If new_trades is empty or invalid type
    """
    # Convert to DataFrame if needed
    if isinstance(new_trades, list):
        new_df = trades_to_polars(new_trades)
    elif isinstance(new_trades, pl.DataFrame):
        new_df = new_trades
    else:
        raise ValueError(f"new_trades must be list[MLTradeRecord] or pl.DataFrame, got {type(new_trades)}")

    if len(new_df) == 0:
        raise ValueError("Cannot append empty trades DataFrame")

    # Read existing trades if file exists
    output_path = Path(path)
    if output_path.exists():
        existing_df = pl.read_parquet(output_path)
        combined_df = pl.concat([existing_df, new_df])
    else:
        combined_df = new_df

    # Write back
    export_parquet(combined_df, output_path, compression, compression_level)
