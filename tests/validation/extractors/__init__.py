"""Platform-specific trade extractors.

Each extractor converts a platform's native trade format
to the StandardTrade representation for comparison.
"""

import sys
from pathlib import Path

# Handle imports when run as script vs module
try:
    from .ml4t.backtest import extract_ml4t.backtest_trades
    from .vectorbt import extract_vectorbt_trades
    from .backtrader import extract_backtrader_trades
    from .zipline import extract_zipline_trades
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from ml4t.backtest import extract_ml4t.backtest_trades
    from vectorbt import extract_vectorbt_trades
    from backtrader import extract_backtrader_trades
    from zipline import extract_zipline_trades

__all__ = [
    'extract_ml4t.backtest_trades',
    'extract_vectorbt_trades',
    'extract_backtrader_trades',
    'extract_zipline_trades',
]
