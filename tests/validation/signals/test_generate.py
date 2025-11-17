"""Unit tests for signal generation module."""

import pandas as pd
import pytest

from tests.validation.signals.generate import (
    generate_sma_crossover,
    load_signal_set,
    save_signal_set,
)


def test_generate_sma_crossover():
    """Test SMA crossover signal generation."""
    # Create simple test data
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    # Create price series that will cross: low -> high -> low
    prices = pd.Series(
        [100] * 15 + [110] * 20 + [100] * 15,  # Steady  # Rising  # Falling
        index=dates,
    )

    # Generate signals with fast=5, slow=10
    signals = generate_sma_crossover(prices, fast=5, slow=10)

    # Verify structure
    assert isinstance(signals, pd.DataFrame)
    assert list(signals.columns) == ["entry", "exit"]
    assert len(signals) == len(prices)
    assert signals.index.equals(prices.index)

    # Verify signals are boolean
    assert signals["entry"].dtype == bool
    assert signals["exit"].dtype == bool

    # Verify we get entry signal when fast crosses above slow
    assert signals["entry"].sum() > 0, "Should have at least one entry signal"

    # Verify we get exit signal when fast crosses below slow
    assert signals["exit"].sum() > 0, "Should have at least one exit signal"


def test_save_and_load_signal_set(tmp_path):
    """Test saving and loading signal sets."""
    # Create test data
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {
            "open": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [100 + i for i in range(10)],
            "volume": [1000] * 10,
        },
        index=dates,
    )

    signals = pd.DataFrame(
        {
            "entry": [True, False, False, True, False, False, False, True, False, False],
            "exit": [False, True, False, False, True, False, False, False, True, False],
        },
        index=dates,
    )

    metadata = {"signal_type": "test", "parameters": {"foo": "bar"}}

    # Save signal set
    # Note: This will save to tests/validation/signals/ not tmp_path
    # For now, we'll just test the validation logic
    try:
        output_path = save_signal_set("test_signal_set", data, signals, metadata)

        # Load it back
        loaded = load_signal_set("test_signal_set")

        # Verify structure
        assert "data" in loaded
        assert "signals" in loaded
        assert "metadata" in loaded

        # Verify data matches
        pd.testing.assert_frame_equal(loaded["data"], data)
        pd.testing.assert_frame_equal(loaded["signals"], signals)

        # Verify metadata
        assert loaded["metadata"]["signal_type"] == "test"
        assert loaded["metadata"]["parameters"] == {"foo": "bar"}
        assert "generated_at" in loaded["metadata"]

    finally:
        # Cleanup test file
        import os

        test_file = output_path
        if os.path.exists(test_file):
            os.remove(test_file)


def test_signal_validation():
    """Test signal format validation."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {
            "open": range(10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
            "volume": range(10),
        },
        index=dates,
    )

    # Missing columns should raise error
    bad_signals = pd.DataFrame({"entry": [True] * 10}, index=dates)

    with pytest.raises(ValueError, match="must have 'entry' and 'exit' columns"):
        save_signal_set("bad_test", data, bad_signals)

    # Non-datetime index should raise error
    bad_signals2 = pd.DataFrame({"entry": [True] * 10, "exit": [False] * 10}, index=range(10))

    with pytest.raises(ValueError, match="must have DatetimeIndex"):
        save_signal_set("bad_test2", data, bad_signals2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
