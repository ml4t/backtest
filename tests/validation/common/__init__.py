"""Common validation infrastructure for backtesting engine comparison."""
from .data_generator import generate_ohlcv, load_real_crypto_data
from .signal_generator import generate_fixed_entries, generate_entry_exit_pairs
from .engine_wrappers import (
    BacktestConfig,
    BacktestResult,
    EngineWrapper,
    VectorBTWrapper,
    ZiplineWrapper,
    BacktraderWrapper,
    QEngineWrapper,
)
from .comparison import compare_trades, assert_identical, print_validation_report

__all__ = [
    'generate_ohlcv',
    'load_real_crypto_data',
    'generate_fixed_entries',
    'generate_entry_exit_pairs',
    'BacktestConfig',
    'BacktestResult',
    'EngineWrapper',
    'VectorBTWrapper',
    'ZiplineWrapper',
    'BacktraderWrapper',
    'QEngineWrapper',
    'compare_trades',
    'assert_identical',
    'print_validation_report',
]
