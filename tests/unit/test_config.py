"""Tests for configuration schema and loaders.

This test suite validates:
- Pydantic model validation
- YAML/JSON loading and parsing
- Environment variable substitution
- Error handling and messages
- Default values
- Frozen (immutable) configuration
"""

import json
import os
import tempfile
from pathlib import Path

import polars as pl
import pytest
import yaml

from ml4t.backtest.config import (
    BacktestConfig,
    CallableFeaturesConfig,
    CommissionConfig,
    CommissionType,
    ConfigError,
    DataFormat,
    DataSourceConfig,
    DataSourcesConfig,
    ExecutionConfig,
    FeatureProviderType,
    PrecomputedFeaturesConfig,
    RiskRulesConfig,
    SlippageConfig,
    SlippageType,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_price_data(temp_dir):
    """Create sample price data file."""
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "asset_id": ["AAPL"] * 10,
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000000] * 10,
        }
    )
    path = temp_dir / "prices.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def sample_signals_data(temp_dir):
    """Create sample signals data file."""
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "asset_id": ["AAPL"] * 10,
            "signal": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0],
            "confidence": [0.8, 0.9, 0.5, 0.6, 0.85, 0.9, 0.4, 0.95, 0.88, 0.5],
        }
    )
    path = temp_dir / "signals.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def sample_features_data(temp_dir):
    """Create sample features data file."""
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "asset_id": ["AAPL"] * 10,
            "atr": [2.5, 2.6, 2.4, 2.7, 2.8, 2.6, 2.5, 2.4, 2.6, 2.7],
            "rsi": [65, 70, 60, 55, 45, 40, 50, 60, 65, 70],
            "volatility": [0.02, 0.025, 0.022, 0.028, 0.03, 0.027, 0.024, 0.022, 0.025, 0.026],
        }
    )
    path = temp_dir / "features.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def minimal_yaml_config(temp_dir, sample_price_data):
    """Create minimal valid YAML configuration."""
    config = {
        "data_sources": {
            "prices": {
                "path": str(sample_price_data),
                "format": "parquet",
            }
        }
    }
    path = temp_dir / "minimal.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def full_yaml_config(temp_dir, sample_price_data, sample_signals_data, sample_features_data):
    """Create full YAML configuration with all sections."""
    config = {
        "name": "test_strategy",
        "description": "Test configuration",
        "data_sources": {
            "prices": {
                "path": str(sample_price_data),
                "format": "parquet",
                "timestamp_column": "timestamp",
                "asset_column": "asset_id",
            },
            "signals": {
                "path": str(sample_signals_data),
                "format": "parquet",
                "columns": ["timestamp", "asset_id", "signal", "confidence"],
            },
        },
        "features": {
            "type": "precomputed",
            "path": str(sample_features_data),
            "columns": ["atr", "rsi", "volatility"],
        },
        "execution": {
            "initial_capital": 100000,
            "commission": {
                "type": "per_share",
                "rate": 0.005,
                "minimum": 1.0,
            },
            "slippage": {
                "type": "percentage",
                "rate": 0.001,
            },
            "enable_margin": False,
            "max_leverage": 1.0,
            "execution_delay": True,
            "allow_immediate_reentry": True,
        },
        "risk_rules": {
            "max_position_size": 0.5,
            "stop_loss": 0.05,
            "take_profit": 0.15,
        },
    }
    path = temp_dir / "full.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ============================================================================
# Test DataSourceConfig
# ============================================================================


def test_data_source_config_valid(sample_price_data):
    """Test valid data source configuration."""
    config = DataSourceConfig(path=str(sample_price_data), format=DataFormat.PARQUET)
    assert config.path == str(sample_price_data)
    assert config.format == DataFormat.PARQUET
    assert config.timestamp_column == "timestamp"
    assert config.asset_column == "asset_id"


def test_data_source_config_env_vars(sample_price_data, temp_dir):
    """Test environment variable expansion in path."""
    os.environ["TEST_DATA_PATH"] = str(temp_dir)
    config = DataSourceConfig(
        path="${TEST_DATA_PATH}/prices.parquet", format=DataFormat.PARQUET
    )
    assert config.path == str(sample_price_data)
    del os.environ["TEST_DATA_PATH"]


def test_data_source_config_undefined_env_var():
    """Test error on undefined environment variable."""
    with pytest.raises(ValueError, match="Undefined environment variable"):
        DataSourceConfig(path="${UNDEFINED_VAR}/data.parquet", format=DataFormat.PARQUET)


def test_data_source_config_file_not_found(temp_dir):
    """Test error when data file doesn't exist."""
    with pytest.raises(ValueError, match="Data file not found"):
        DataSourceConfig(path=str(temp_dir / "nonexistent.parquet"), format=DataFormat.PARQUET)


def test_data_source_config_path_is_directory(temp_dir):
    """Test error when path is a directory."""
    with pytest.raises(ValueError, match="Path is not a file"):
        DataSourceConfig(path=str(temp_dir), format=DataFormat.PARQUET)


# ============================================================================
# Test DataSourcesConfig
# ============================================================================


def test_data_sources_config_minimal(sample_price_data):
    """Test minimal data sources configuration with only prices."""
    prices = DataSourceConfig(path=str(sample_price_data), format=DataFormat.PARQUET)
    config = DataSourcesConfig(prices=prices)
    assert config.prices.path == str(sample_price_data)
    assert config.signals is None
    assert config.features is None
    assert config.context is None


def test_data_sources_config_full(sample_price_data, sample_signals_data, sample_features_data):
    """Test full data sources configuration."""
    prices = DataSourceConfig(path=str(sample_price_data), format=DataFormat.PARQUET)
    signals = DataSourceConfig(path=str(sample_signals_data), format=DataFormat.PARQUET)
    features = DataSourceConfig(path=str(sample_features_data), format=DataFormat.PARQUET)

    config = DataSourcesConfig(prices=prices, signals=signals, features=features)
    assert config.prices.path == str(sample_price_data)
    assert config.signals.path == str(sample_signals_data)
    assert config.features.path == str(sample_features_data)


# ============================================================================
# Test FeaturesConfig
# ============================================================================


def test_precomputed_features_config(sample_features_data):
    """Test precomputed features configuration."""
    config = PrecomputedFeaturesConfig(
        type=FeatureProviderType.PRECOMPUTED,
        path=str(sample_features_data),
        columns=["atr", "rsi"],
    )
    assert config.type == FeatureProviderType.PRECOMPUTED
    assert config.path == str(sample_features_data)
    assert config.columns == ["atr", "rsi"]


def test_callable_features_config():
    """Test callable features configuration."""
    config = CallableFeaturesConfig(
        type=FeatureProviderType.CALLABLE,
        module="my_module.features",
        function="compute_features",
        kwargs={"lookback": 14},
    )
    assert config.type == FeatureProviderType.CALLABLE
    assert config.module == "my_module.features"
    assert config.function == "compute_features"
    assert config.kwargs == {"lookback": 14}


# ============================================================================
# Test RiskRulesConfig
# ============================================================================


def test_risk_rules_config_valid():
    """Test valid risk rules configuration."""
    config = RiskRulesConfig(
        max_position_size=0.5,
        stop_loss=0.05,
        take_profit=0.15,
        min_vix=10.0,
        max_vix=40.0,
    )
    assert config.max_position_size == 0.5
    assert config.stop_loss == 0.05
    assert config.take_profit == 0.15
    assert config.min_vix == 10.0
    assert config.max_vix == 40.0


def test_risk_rules_config_invalid_vix_range():
    """Test error when min_vix > max_vix."""
    with pytest.raises(ValueError, match="min_vix.*must be.*max_vix"):
        RiskRulesConfig(min_vix=50.0, max_vix=20.0)


def test_risk_rules_config_invalid_position_size():
    """Test error when position size out of range."""
    with pytest.raises(ValueError):
        RiskRulesConfig(max_position_size=1.5)  # > 1.0


def test_risk_rules_config_optional_fields():
    """Test risk rules with optional fields."""
    config = RiskRulesConfig()
    assert config.max_position_size is None
    assert config.stop_loss is None


# ============================================================================
# Test ExecutionConfig
# ============================================================================


def test_execution_config_defaults():
    """Test execution configuration defaults."""
    config = ExecutionConfig()
    assert config.initial_capital == 100000.0
    assert config.commission is None
    assert config.slippage is None
    assert config.enable_margin is False
    assert config.max_leverage == 1.0
    assert config.execution_delay is True
    assert config.allow_immediate_reentry is True


def test_execution_config_with_models():
    """Test execution configuration with commission and slippage models."""
    commission = CommissionConfig(type=CommissionType.PER_SHARE, rate=0.005, minimum=1.0)
    slippage = SlippageConfig(type=SlippageType.PERCENTAGE, rate=0.001)

    config = ExecutionConfig(
        initial_capital=1000000,
        commission=commission,
        slippage=slippage,
        enable_margin=True,
        max_leverage=2.0,
    )

    assert config.initial_capital == 1000000
    assert config.commission.type == CommissionType.PER_SHARE
    assert config.slippage.type == SlippageType.PERCENTAGE
    assert config.enable_margin is True
    assert config.max_leverage == 2.0


def test_execution_config_invalid_capital():
    """Test error when initial capital is negative or zero."""
    with pytest.raises(ValueError):
        ExecutionConfig(initial_capital=0.0)

    with pytest.raises(ValueError):
        ExecutionConfig(initial_capital=-1000.0)


def test_execution_config_invalid_leverage():
    """Test error when leverage is less than 1.0."""
    with pytest.raises(ValueError):
        ExecutionConfig(max_leverage=0.5)


# ============================================================================
# Test BacktestConfig - YAML Loading
# ============================================================================


def test_backtest_config_from_yaml_minimal(minimal_yaml_config):
    """Test loading minimal YAML configuration."""
    config = BacktestConfig.from_yaml(minimal_yaml_config)
    assert config.data_sources.prices.path is not None
    assert config.execution.initial_capital == 100000.0


def test_backtest_config_from_yaml_full(full_yaml_config):
    """Test loading full YAML configuration."""
    config = BacktestConfig.from_yaml(full_yaml_config)

    # Check top-level fields
    assert config.name == "test_strategy"
    assert config.description == "Test configuration"

    # Check data sources
    assert config.data_sources.prices.path is not None
    assert config.data_sources.signals.path is not None

    # Check features
    assert config.features is not None
    assert config.features.type == FeatureProviderType.PRECOMPUTED
    assert "atr" in config.features.columns

    # Check execution
    assert config.execution.initial_capital == 100000
    assert config.execution.commission.type == CommissionType.PER_SHARE
    assert config.execution.slippage.type == SlippageType.PERCENTAGE

    # Check risk rules
    assert config.risk_rules is not None
    assert config.risk_rules.max_position_size == 0.5


def test_backtest_config_from_yaml_file_not_found(temp_dir):
    """Test error when YAML file doesn't exist."""
    with pytest.raises(ConfigError, match="Configuration file not found"):
        BacktestConfig.from_yaml(temp_dir / "nonexistent.yaml")


def test_backtest_config_from_yaml_invalid_yaml(temp_dir):
    """Test error when YAML is malformed."""
    invalid_yaml = temp_dir / "invalid.yaml"
    with open(invalid_yaml, "w") as f:
        f.write("{ invalid yaml content: [")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        BacktestConfig.from_yaml(invalid_yaml)


def test_backtest_config_from_yaml_validation_error(temp_dir, sample_price_data):
    """Test error when YAML content doesn't pass validation."""
    invalid_config = {
        "data_sources": {
            "prices": {
                "path": str(sample_price_data),
                "format": "parquet",
            }
        },
        "execution": {
            "initial_capital": -1000,  # Invalid negative capital
        },
    }
    path = temp_dir / "invalid_config.yaml"
    with open(path, "w") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ConfigError, match="Configuration validation failed"):
        BacktestConfig.from_yaml(path)


# ============================================================================
# Test BacktestConfig - JSON Loading
# ============================================================================


def test_backtest_config_from_json(temp_dir, sample_price_data):
    """Test loading JSON configuration."""
    config_dict = {
        "data_sources": {
            "prices": {
                "path": str(sample_price_data),
                "format": "parquet",
            }
        },
        "execution": {
            "initial_capital": 50000,
        },
    }
    json_path = temp_dir / "config.json"
    with open(json_path, "w") as f:
        json.dump(config_dict, f)

    config = BacktestConfig.from_json(json_path)
    assert config.execution.initial_capital == 50000


def test_backtest_config_from_json_file_not_found(temp_dir):
    """Test error when JSON file doesn't exist."""
    with pytest.raises(ConfigError, match="Configuration file not found"):
        BacktestConfig.from_json(temp_dir / "nonexistent.json")


def test_backtest_config_from_json_invalid_json(temp_dir):
    """Test error when JSON is malformed."""
    invalid_json = temp_dir / "invalid.json"
    with open(invalid_json, "w") as f:
        f.write("{ invalid json content")

    with pytest.raises(ConfigError, match="Invalid JSON"):
        BacktestConfig.from_json(invalid_json)


# ============================================================================
# Test BacktestConfig - Saving
# ============================================================================


def test_backtest_config_to_yaml(temp_dir, sample_price_data):
    """Test saving configuration to YAML."""
    prices = DataSourceConfig(path=str(sample_price_data), format=DataFormat.PARQUET)
    data_sources = DataSourcesConfig(prices=prices)
    config = BacktestConfig(data_sources=data_sources, name="test")

    output_path = temp_dir / "output.yaml"
    config.to_yaml(output_path)

    # Reload and verify
    reloaded = BacktestConfig.from_yaml(output_path)
    assert reloaded.name == "test"
    assert reloaded.data_sources.prices.path == str(sample_price_data)


def test_backtest_config_to_json(temp_dir, sample_price_data):
    """Test saving configuration to JSON."""
    prices = DataSourceConfig(path=str(sample_price_data), format=DataFormat.PARQUET)
    data_sources = DataSourcesConfig(prices=prices)
    config = BacktestConfig(data_sources=data_sources, name="test")

    output_path = temp_dir / "output.json"
    config.to_json(output_path)

    # Reload and verify
    reloaded = BacktestConfig.from_json(output_path)
    assert reloaded.name == "test"


# ============================================================================
# Test BacktestConfig - Immutability
# ============================================================================


def test_backtest_config_frozen(minimal_yaml_config):
    """Test that configuration is immutable after loading."""
    config = BacktestConfig.from_yaml(minimal_yaml_config)

    # Attempt to modify should raise error
    with pytest.raises(Exception):  # Pydantic raises ValidationError or AttributeError
        config.name = "modified"


def test_backtest_config_extra_fields_forbidden(temp_dir, sample_price_data):
    """Test that extra fields raise an error."""
    config_dict = {
        "data_sources": {
            "prices": {
                "path": str(sample_price_data),
                "format": "parquet",
            }
        },
        "unknown_field": "value",  # Extra field not in schema
    }
    yaml_path = temp_dir / "extra_fields.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f)

    with pytest.raises(ConfigError, match="Configuration validation failed"):
        BacktestConfig.from_yaml(yaml_path)


# ============================================================================
# Test Environment Variable Substitution
# ============================================================================


def test_env_var_substitution_simple(temp_dir, sample_price_data):
    """Test simple environment variable substitution."""
    os.environ["MY_DATA_DIR"] = str(temp_dir)

    config_dict = {
        "data_sources": {
            "prices": {
                "path": "${MY_DATA_DIR}/prices.parquet",
                "format": "parquet",
            }
        }
    }
    yaml_path = temp_dir / "env_var.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f)

    config = BacktestConfig.from_yaml(yaml_path)
    assert config.data_sources.prices.path == str(sample_price_data)

    del os.environ["MY_DATA_DIR"]


def test_env_var_substitution_multiple(temp_dir, sample_price_data):
    """Test multiple environment variables in config."""
    os.environ["DATA_ROOT"] = str(temp_dir)
    os.environ["PRICE_FILE"] = "prices.parquet"

    config_dict = {
        "data_sources": {
            "prices": {
                "path": "${DATA_ROOT}/${PRICE_FILE}",
                "format": "parquet",
            }
        }
    }
    yaml_path = temp_dir / "multi_env.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f)

    config = BacktestConfig.from_yaml(yaml_path)
    assert config.data_sources.prices.path == str(sample_price_data)

    del os.environ["DATA_ROOT"]
    del os.environ["PRICE_FILE"]


# ============================================================================
# Test Error Messages
# ============================================================================


def test_error_message_missing_required_field(temp_dir):
    """Test clear error message when required field is missing."""
    config_dict = {
        "execution": {
            "initial_capital": 100000,
        }
        # Missing required 'data_sources'
    }
    yaml_path = temp_dir / "missing_field.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f)

    with pytest.raises(ConfigError, match="Configuration validation failed"):
        BacktestConfig.from_yaml(yaml_path)


def test_error_message_wrong_type(temp_dir, sample_price_data):
    """Test clear error message when field has wrong type."""
    config_dict = {
        "data_sources": {
            "prices": {
                "path": str(sample_price_data),
                "format": "parquet",
            }
        },
        "execution": {
            "initial_capital": "not a number",  # Wrong type
        },
    }
    yaml_path = temp_dir / "wrong_type.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f)

    with pytest.raises(ConfigError, match="Configuration validation failed"):
        BacktestConfig.from_yaml(yaml_path)


# ============================================================================
# Test Example Configs
# ============================================================================


@pytest.mark.integration
def test_example_configs_syntax():
    """Test that example configs have valid syntax (not full validation).

    This test only checks YAML/JSON syntax, not file paths which may not exist.
    """
    examples_dir = Path(__file__).parent.parent.parent / "examples" / "configs"

    if not examples_dir.exists():
        pytest.skip("Examples directory not found")

    for config_file in examples_dir.glob("*.yaml"):
        with open(config_file, "r") as f:
            try:
                data = yaml.safe_load(f)
                assert isinstance(data, dict), f"{config_file} should contain a dictionary"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax in {config_file}: {e}")
