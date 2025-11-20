"""Configuration schema and loaders for declarative backtesting.

This module provides a comprehensive configuration system for specifying backtesting
setups using YAML or JSON. It enables declarative specification of data sources,
feature providers, risk rules, and execution parameters.

Design Principles:
    - Type-safe configuration with Pydantic validation
    - Clear error messages for invalid configs
    - Environment variable substitution for sensitive/path data
    - Immutable after loading (frozen models)
    - Support for common patterns (single-asset, multi-asset, ML strategies)

Example YAML Config:
    ```yaml
    data_sources:
      prices:
        path: ${DATA_PATH}/prices.parquet
        format: parquet
      signals:
        path: ${DATA_PATH}/ml_signals.parquet
        columns: [signal_long, signal_short, confidence]

    features:
      type: precomputed
      path: ${DATA_PATH}/features.parquet
      columns: [atr, rsi, volatility]

    execution:
      initial_capital: 100000
      commission:
        type: per_share
        rate: 0.005
      slippage:
        type: percentage
        rate: 0.001

    risk_rules:
      max_position_size: 0.1
      stop_loss: 0.02
    ```

Usage:
    >>> from pathlib import Path
    >>> from ml4t.backtest.config import BacktestConfig
    >>>
    >>> # Load from YAML
    >>> config = BacktestConfig.from_yaml(Path("config.yaml"))
    >>>
    >>> # Load from JSON
    >>> config = BacktestConfig.from_json(Path("config.json"))
    >>>
    >>> # Access configuration
    >>> print(config.execution.initial_capital)
    >>> print(config.data_sources.prices.path)
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


# ============================================================================
# Enums for Configuration Options
# ============================================================================


class DataFormat(str, Enum):
    """Supported data file formats."""

    PARQUET = "parquet"
    CSV = "csv"
    HDF5 = "hdf5"


class FeatureProviderType(str, Enum):
    """Types of feature providers."""

    PRECOMPUTED = "precomputed"
    CALLABLE = "callable"


class CommissionType(str, Enum):
    """Commission model types."""

    PER_SHARE = "per_share"
    PERCENTAGE = "percentage"
    FIXED = "fixed"
    TIERED = "tiered"


class SlippageType(str, Enum):
    """Slippage model types."""

    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLUME_SHARE = "volume_share"


# ============================================================================
# Data Source Configuration
# ============================================================================


class DataSourceConfig(BaseModel):
    """Configuration for a single data source (prices, signals, features).

    Attributes:
        path: Path to data file (supports environment variable expansion)
        format: File format (parquet, csv, hdf5)
        columns: Optional list of columns to load (None = load all)
        timestamp_column: Name of timestamp column (default: 'timestamp')
        asset_column: Name of asset ID column (default: 'asset_id')
        filters: Optional list of filter expressions
    """

    path: str = Field(..., description="Path to data file")
    format: DataFormat = Field(
        DataFormat.PARQUET, description="File format (parquet, csv, hdf5)"
    )
    columns: list[str] | None = Field(
        None, description="Columns to load (None = all)"
    )
    timestamp_column: str = Field(
        "timestamp", description="Name of timestamp column"
    )
    asset_column: str = Field("asset_id", description="Name of asset ID column")
    filters: list[str] | None = Field(None, description="Filter expressions")

    @field_validator("path")
    @classmethod
    def expand_env_vars(cls, v: str) -> str:
        """Expand environment variables in path."""
        expanded = os.path.expandvars(v)
        if "${" in expanded:
            # Find undefined variables
            undefined = [
                var
                for var in expanded.split("${")[1:]
                if "}" in var and not os.getenv(var.split("}")[0])
            ]
            if undefined:
                raise ValueError(
                    f"Undefined environment variable(s): {', '.join(undefined)}"
                )
        return expanded

    @model_validator(mode="after")
    def validate_path_exists(self) -> "DataSourceConfig":
        """Validate that the data file exists."""
        path = Path(self.path)
        if not path.exists():
            raise ValueError(
                f"Data file not found: {self.path}\n"
                f"Hint: Check that the path is correct and environment "
                f"variables are set."
            )
        if not path.is_file():
            raise ValueError(f"Path is not a file: {self.path}")
        return self


class DataSourcesConfig(BaseModel):
    """Configuration for all data sources.

    Attributes:
        prices: Price data (OHLCV) source
        signals: Optional ML signals source
        features: Optional features source (alternative to feature_provider)
        context: Optional market-wide context data (VIX, SPY, etc.)
    """

    prices: DataSourceConfig = Field(..., description="Price data (OHLCV)")
    signals: DataSourceConfig | None = Field(None, description="ML signals")
    features: DataSourceConfig | None = Field(None, description="Features")
    context: DataSourceConfig | None = Field(
        None, description="Market context (VIX, SPY)"
    )

    @model_validator(mode="after")
    def validate_at_least_prices(self) -> "DataSourcesConfig":
        """Ensure at least price data is provided."""
        if not self.prices:
            raise ValueError(
                "Price data is required. Specify 'prices' in data_sources."
            )
        return self


# ============================================================================
# Feature Provider Configuration
# ============================================================================


class PrecomputedFeaturesConfig(BaseModel):
    """Configuration for precomputed feature provider.

    Attributes:
        type: Must be 'precomputed'
        path: Path to features file
        columns: Optional list of feature columns to use
        timestamp_column: Name of timestamp column
        asset_column: Name of asset ID column
    """

    type: Literal[FeatureProviderType.PRECOMPUTED] = Field(
        FeatureProviderType.PRECOMPUTED
    )
    path: str = Field(..., description="Path to precomputed features file")
    columns: list[str] | None = Field(
        None, description="Feature columns (None = all)"
    )
    timestamp_column: str = Field("timestamp")
    asset_column: str = Field("asset_id")

    @field_validator("path")
    @classmethod
    def expand_env_vars(cls, v: str) -> str:
        """Expand environment variables in path."""
        return os.path.expandvars(v)


class CallableFeaturesConfig(BaseModel):
    """Configuration for callable feature provider.

    Attributes:
        type: Must be 'callable'
        module: Python module containing the callable
        function: Function name to use
        kwargs: Optional keyword arguments to pass to function
    """

    type: Literal[FeatureProviderType.CALLABLE] = Field(
        FeatureProviderType.CALLABLE
    )
    module: str = Field(..., description="Python module with feature function")
    function: str = Field(..., description="Function name")
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Function kwargs"
    )


FeaturesConfig = PrecomputedFeaturesConfig | CallableFeaturesConfig | None


# ============================================================================
# Risk Rules Configuration
# ============================================================================


class RiskRulesConfig(BaseModel):
    """Configuration for risk management rules.

    Note: This is a basic structure for Phase 1. Full risk rule implementation
    will be expanded in Phase 2.

    Attributes:
        max_position_size: Maximum position size as fraction of portfolio (0.0-1.0)
        stop_loss: Stop loss as fraction of entry price (0.0-1.0)
        take_profit: Take profit as fraction of entry price (0.0+)
        max_portfolio_heat: Maximum portfolio risk (fraction of NAV)
        min_vix: Minimum VIX level to allow trading
        max_vix: Maximum VIX level to allow trading
    """

    max_position_size: float | None = Field(
        None, ge=0.0, le=1.0, description="Max position size (fraction)"
    )
    stop_loss: float | None = Field(
        None, ge=0.0, le=1.0, description="Stop loss (fraction)"
    )
    take_profit: float | None = Field(
        None, ge=0.0, description="Take profit (fraction)"
    )
    max_portfolio_heat: float | None = Field(
        None, ge=0.0, le=1.0, description="Max portfolio risk (fraction)"
    )
    min_vix: float | None = Field(None, ge=0.0, description="Min VIX to trade")
    max_vix: float | None = Field(None, ge=0.0, description="Max VIX to trade")

    @model_validator(mode="after")
    def validate_vix_range(self) -> "RiskRulesConfig":
        """Validate VIX range makes sense."""
        if (
            self.min_vix is not None
            and self.max_vix is not None
            and self.min_vix > self.max_vix
        ):
            raise ValueError(
                f"min_vix ({self.min_vix}) must be <= max_vix ({self.max_vix})"
            )
        return self


# ============================================================================
# Execution Configuration
# ============================================================================


class CommissionConfig(BaseModel):
    """Configuration for commission model.

    Attributes:
        type: Commission model type
        rate: Commission rate (interpretation depends on type)
        minimum: Optional minimum commission per trade
    """

    type: CommissionType = Field(
        CommissionType.PER_SHARE, description="Commission model type"
    )
    rate: float = Field(..., ge=0.0, description="Commission rate")
    minimum: float | None = Field(
        None, ge=0.0, description="Minimum commission per trade"
    )


class SlippageConfig(BaseModel):
    """Configuration for slippage model.

    Attributes:
        type: Slippage model type
        rate: Slippage rate (interpretation depends on type)
    """

    type: SlippageType = Field(
        SlippageType.PERCENTAGE, description="Slippage model type"
    )
    rate: float = Field(..., ge=0.0, description="Slippage rate")


class ExecutionConfig(BaseModel):
    """Configuration for execution parameters.

    Attributes:
        initial_capital: Starting cash amount
        commission: Commission model configuration
        slippage: Slippage model configuration
        enable_margin: Whether to enable margin trading
        max_leverage: Maximum leverage allowed (1.0 = no leverage)
        execution_delay: Whether to delay fills to next bar (prevents lookahead)
        allow_immediate_reentry: Allow re-entry on same bar as exit
    """

    initial_capital: float = Field(
        100000.0, gt=0.0, description="Starting cash amount"
    )
    commission: CommissionConfig | None = Field(
        None, description="Commission model"
    )
    slippage: SlippageConfig | None = Field(None, description="Slippage model")
    enable_margin: bool = Field(False, description="Enable margin trading")
    max_leverage: float = Field(
        1.0, ge=1.0, description="Max leverage (1.0 = no leverage)"
    )
    execution_delay: bool = Field(
        True, description="Delay fills to prevent lookahead bias"
    )
    allow_immediate_reentry: bool = Field(
        True, description="Allow same-bar re-entry"
    )


# ============================================================================
# Main Configuration
# ============================================================================


class BacktestConfig(BaseModel):
    """Main configuration for backtesting.

    This is the top-level configuration object that combines all aspects of
    a backtest: data sources, features, risk rules, and execution parameters.

    Attributes:
        name: Optional configuration name for identification
        description: Optional description of the configuration
        data_sources: Data source configurations
        features: Optional feature provider configuration
        risk_rules: Optional risk management rules
        execution: Execution parameters
    """

    model_config = ConfigDict(
        frozen=True,  # Make config immutable after loading
        extra="forbid",  # Raise error on unknown fields
    )

    name: str | None = Field(None, description="Configuration name")
    description: str | None = Field(None, description="Configuration description")
    data_sources: DataSourcesConfig = Field(..., description="Data sources")
    features: FeaturesConfig = Field(None, description="Feature provider config")
    risk_rules: RiskRulesConfig | None = Field(
        None, description="Risk management rules"
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig, description="Execution parameters"
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "BacktestConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated BacktestConfig instance

        Raises:
            ConfigError: If file not found, invalid YAML, or validation fails

        Example:
            >>> config = BacktestConfig.from_yaml(Path("config.yaml"))
        """
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(
                f"Configuration file not found: {path}\n"
                f"Hint: Check that the path is correct."
            )
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Invalid YAML in {path}:\n{e}\n"
                f"Hint: Check YAML syntax (indentation, colons, quotes)."
            )

        try:
            return cls(**data)
        except Exception as e:
            raise ConfigError(
                f"Configuration validation failed for {path}:\n{e}\n"
                f"Hint: Check that all required fields are present and types are correct."
            )

    @classmethod
    def from_json(cls, path: Path) -> "BacktestConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            Validated BacktestConfig instance

        Raises:
            ConfigError: If file not found, invalid JSON, or validation fails

        Example:
            >>> config = BacktestConfig.from_json(Path("config.json"))
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ConfigError(
                f"Configuration file not found: {path}\n"
                f"Hint: Check that the path is correct."
            )
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"Invalid JSON in {path}:\n{e}\n"
                f"Hint: Check JSON syntax (commas, brackets, quotes)."
            )

        try:
            return cls(**data)
        except Exception as e:
            raise ConfigError(
                f"Configuration validation failed for {path}:\n{e}\n"
                f"Hint: Check that all required fields are present and types are correct."
            )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration

        Example:
            >>> config.to_yaml(Path("config.yaml"))
        """
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)

    def to_json(self, path: Path, indent: int = 2) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save JSON configuration
            indent: JSON indentation level (default: 2)

        Example:
            >>> config.to_json(Path("config.json"))
        """
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=indent)
