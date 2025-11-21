"""Common validation infrastructure for backtesting engine comparison."""
from .data_generator import generate_ohlcv, load_real_crypto_data
from .signal_generator import (
    generate_fixed_entries,
    generate_entry_exit_pairs,
    generate_exit_on_next_entry,
)
from .engine_wrappers import (
    BacktestConfig,
    BacktestResult,
    EngineWrapper,
    VectorBTWrapper,
    ZiplineWrapper,
    BacktraderWrapper,
    BacktestWrapper,
)
from .comparison import compare_trades, assert_identical, print_validation_report

__all__ = [
    'generate_ohlcv',
    'load_real_crypto_data',
    'generate_fixed_entries',
    'generate_entry_exit_pairs',
    'generate_exit_on_next_entry',
    'BacktestConfig',
    'BacktestResult',
    'EngineWrapper',
    'VectorBTWrapper',
    'ZiplineWrapper',
    'BacktraderWrapper',
    'BacktestWrapper',
    'compare_trades',
    'assert_identical',
    'print_validation_report',
]
