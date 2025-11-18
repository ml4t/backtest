#!/bin/bash
# Test runner for data layer unit tests (TASK-INT-012)

set -e

cd /home/stefan/ml4t/software/backtest

echo "=== Activating virtual environment ==="
source .venv/bin/activate

echo ""
echo "=== Running Data Layer Unit Tests ==="
pytest tests/unit/test_polars_feed.py \
       tests/unit/test_feature_provider.py \
       tests/unit/test_validation.py \
       -v \
       --cov=src/ml4t/backtest/data \
       --cov-report=term-missing \
       --cov-report=html:htmlcov_data_layer \
       --tb=short

echo ""
echo "=== Test Summary ==="
echo "Test files:"
echo "  - test_polars_feed.py"
echo "  - test_feature_provider.py"
echo "  - test_validation.py"
echo ""
echo "Coverage report saved to: htmlcov_data_layer/index.html"
