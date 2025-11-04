"""Data schemas for QEngine."""

from dataclasses import dataclass

import polars as pl


@dataclass
class MarketDataSchema:
    """Schema definition for market data."""

    timestamp_col: str = "timestamp"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"

    def get_dtypes(self) -> dict[str, pl.DataType]:
        """Get Polars data types for the schema."""
        return {
            self.timestamp_col: pl.Datetime("ns"),
            self.open_col: pl.Float64,
            self.high_col: pl.Float64,
            self.low_col: pl.Float64,
            self.close_col: pl.Float64,
            self.volume_col: pl.Int64,
        }

    def validate(self, df: pl.DataFrame) -> None:
        """Validate a DataFrame against this schema."""
        required_cols = [
            self.timestamp_col,
            self.open_col,
            self.high_col,
            self.low_col,
            self.close_col,
            self.volume_col,
        ]

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate data types
        for col, expected_dtype in self.get_dtypes().items():
            if col in df.columns:
                actual_dtype = df[col].dtype
                if not self._compatible_dtypes(actual_dtype, expected_dtype):
                    raise TypeError(
                        f"Column {col} has type {actual_dtype}, expected {expected_dtype}",
                    )

    def _compatible_dtypes(self, actual: pl.DataType, expected: pl.DataType) -> bool:
        """Check if data types are compatible."""
        # Allow int to float conversion
        if expected == pl.Float64 and actual in [pl.Int32, pl.Int64]:
            return True
        # Allow different datetime precisions
        if isinstance(expected, pl.Datetime) and isinstance(actual, pl.Datetime):
            return True
        return actual == expected


@dataclass
class SignalSchema:
    """Schema definition for ML signals."""

    timestamp_col: str = "timestamp"
    asset_id_col: str = "asset_id"
    signal_col: str = "signal"
    confidence_col: str | None = "confidence"
    model_id_col: str | None = "model_id"

    def get_dtypes(self) -> dict[str, pl.DataType]:
        """Get Polars data types for the schema."""
        dtypes = {
            self.timestamp_col: pl.Datetime("ns"),
            self.asset_id_col: pl.Utf8,
            self.signal_col: pl.Float64,
        }

        if self.confidence_col:
            dtypes[self.confidence_col] = pl.Float64
        if self.model_id_col:
            dtypes[self.model_id_col] = pl.Utf8

        return dtypes

    def validate(self, df: pl.DataFrame) -> None:
        """Validate a DataFrame against this schema."""
        required_cols = [
            self.timestamp_col,
            self.asset_id_col,
            self.signal_col,
        ]

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate data types
        for col, expected_dtype in self.get_dtypes().items():
            if col in df.columns:
                actual_dtype = df[col].dtype
                if not self._compatible_dtypes(actual_dtype, expected_dtype):
                    raise TypeError(
                        f"Column {col} has type {actual_dtype}, expected {expected_dtype}",
                    )

    def _compatible_dtypes(self, actual: pl.DataType, expected: pl.DataType) -> bool:
        """Check if data types are compatible."""
        # Allow int to float conversion
        if expected == pl.Float64 and actual in [pl.Int32, pl.Int64]:
            return True
        # Allow different datetime precisions
        if isinstance(expected, pl.Datetime) and isinstance(actual, pl.Datetime):
            return True
        return actual == expected
