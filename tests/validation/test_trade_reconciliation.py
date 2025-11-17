"""
Trade-by-trade reconciliation across all frameworks.

Purpose:
1. Validate ml4t.backtest correctness by achieving 100% trade matching
2. Identify configuration options needed for transparency

Philosophy:
- NO unexplained variance is acceptable
- Every trade difference must be reconciled with source code citations
- Divergences reveal what MUST be configurable (not just "different")
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.qengine_adapter import BacktestAdapter  # ml4t.backtest adapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.base import FrameworkConfig, TradeRecord


SIGNAL_DIR = Path(__file__).parent / "signals"


@dataclass
class TradeComparison:
    """Comparison of a single trade across frameworks."""
    signal_timestamp: datetime
    signal_type: str  # 'entry' or 'exit'
    signal_index: int  # Which signal number (1st entry, 2nd entry, etc.)

    # What each framework did
    qengine_trade: TradeRecord | None
    backtrader_trade: TradeRecord | None
    vectorbt_trade: TradeRecord | None

    # Analysis
    all_match: bool
    divergence_type: str | None  # 'price', 'quantity', 'missing', 'extra'
    explanation: str | None
    source_code_citation: str | None


@dataclass
class ReconciliationReport:
    """Complete reconciliation report for a signal dataset."""
    signal_file: str
    signal_type: str  # 'sma_crossover', 'random', 'rebalancing'
    total_signals: int

    # Trade counts
    qengine_trades: int
    backtrader_trades: int
    vectorbt_trades: int

    # Matching analysis
    perfect_matches: int
    divergences: int
    first_divergence_index: int | None

    # Detailed comparisons
    trade_comparisons: List[TradeComparison]

    # Configuration insights
    required_config_options: List[str]

    def print_summary(self):
        """Print reconciliation summary."""
        print(f"\n{'='*80}")
        print(f"Trade Reconciliation Report: {self.signal_file}")
        print(f"{'='*80}")
        print(f"Signal Type: {self.signal_type}")
        print(f"Total Signals: {self.total_signals}")
        print()
        print(f"Trade Counts:")
        print(f"  ml4t.backtest:  {self.qengine_trades} trades")
        print(f"  Backtrader:     {self.backtrader_trades} trades")
        print(f"  VectorBT:       {self.vectorbt_trades} trades")
        print()

        if self.perfect_matches == len(self.trade_comparisons):
            print("✓ PERFECT MATCH - All frameworks agree 100%")
        else:
            print(f"✗ DIVERGENCE DETECTED")
            print(f"  Perfect matches: {self.perfect_matches}/{len(self.trade_comparisons)}")
            print(f"  Divergences: {self.divergences}")
            print(f"  First divergence at signal #{self.first_divergence_index}")

        print()

    def print_detailed_comparison(self, max_trades: int = 20):
        """Print detailed trade-by-trade comparison."""
        print(f"\n{'='*80}")
        print("Detailed Trade-by-Trade Comparison")
        print(f"{'='*80}")
        print(f"(Showing first {max_trades} trades)")
        print()

        for i, comparison in enumerate(self.trade_comparisons[:max_trades]):
            signal_num = comparison.signal_index + 1
            print(f"Signal #{signal_num} @ {comparison.signal_timestamp.strftime('%Y-%m-%d')} ({comparison.signal_type.upper()})")

            # ml4t.backtest
            if comparison.qengine_trade:
                t = comparison.qengine_trade
                print(f"  ml4t.backtest:  {t.action} {t.quantity:.2f} @ ${t.price:.2f} = ${t.value:,.2f}")
            else:
                print(f"  ml4t.backtest:  [NO TRADE]")

            # Backtrader
            if comparison.backtrader_trade:
                t = comparison.backtrader_trade
                print(f"  Backtrader:     {t.action} {t.quantity:.2f} @ ${t.price:.2f} = ${t.value:,.2f}")
            else:
                print(f"  Backtrader:     [NO TRADE]")

            # VectorBT
            if comparison.vectorbt_trade:
                t = comparison.vectorbt_trade
                print(f"  VectorBT:       {t.action} {t.quantity:.2f} @ ${t.price:.2f} = ${t.value:,.2f}")
            else:
                print(f"  VectorBT:       [NO TRADE]")

            # Analysis
            if comparison.all_match:
                print(f"  ✓ MATCH")
            else:
                print(f"  ✗ DIVERGENCE: {comparison.divergence_type}")
                if comparison.explanation:
                    print(f"    Explanation: {comparison.explanation}")
                if comparison.source_code_citation:
                    print(f"    Source: {comparison.source_code_citation}")

            print()

    def print_configuration_insights(self):
        """Print required configuration options discovered."""
        print(f"\n{'='*80}")
        print("Configuration Requirements")
        print(f"{'='*80}")
        print()

        if not self.required_config_options:
            print("✓ No additional configuration needed - frameworks match perfectly")
        else:
            print("The following behaviors MUST be configurable:")
            print()
            for option in self.required_config_options:
                print(f"  • {option}")

        print()


def extract_trades_from_framework_result(
    result,
    framework_name: str,
) -> List[TradeRecord]:
    """Extract trade records from framework result."""
    if not result or not hasattr(result, 'trades'):
        return []

    return result.trades


def align_trades_with_signals(
    signals: pd.DataFrame,
    trades: List[TradeRecord],
    framework_name: str,
) -> Dict[int, TradeRecord]:
    """
    Align trades with signal indices.

    Returns dictionary mapping signal_index -> TradeRecord
    """
    # Build signal timestamp index
    entry_signals = [(i, ts) for i, (ts, entry) in enumerate(zip(signals.index, signals['entry'])) if entry]
    exit_signals = [(i, ts) for i, (ts, exit_sig) in enumerate(zip(signals.index, signals['exit'])) if exit_sig]

    # Map trades to signals
    trade_map = {}

    for trade in trades:
        # Find closest signal timestamp
        # For BUY trades, match to entry signals
        # For SELL trades, match to exit signals
        if trade.action == 'BUY':
            # Find nearest entry signal
            for sig_idx, sig_ts in entry_signals:
                # Allow some tolerance for timestamp matching (next-bar execution)
                time_diff = abs((trade.timestamp - sig_ts).total_seconds())
                if time_diff <= 86400:  # Within 1 day
                    trade_map[sig_idx] = trade
                    break
        else:  # SELL
            for sig_idx, sig_ts in exit_signals:
                time_diff = abs((trade.timestamp - sig_ts).total_seconds())
                if time_diff <= 86400:
                    trade_map[sig_idx] = trade
                    break

    return trade_map


def compare_trades(
    qengine_trade: TradeRecord | None,
    backtrader_trade: TradeRecord | None,
    vectorbt_trade: TradeRecord | None,
) -> tuple[bool, str | None, str | None]:
    """
    Compare trades across frameworks.

    Returns:
        (all_match, divergence_type, explanation)
    """
    # Collect non-None trades
    trades = []
    if qengine_trade:
        trades.append(('ml4t.backtest', qengine_trade))
    if backtrader_trade:
        trades.append(('Backtrader', backtrader_trade))
    if vectorbt_trade:
        trades.append(('VectorBT', vectorbt_trade))

    if len(trades) == 0:
        return True, None, None  # No trades = match

    if len(trades) < 3:
        # Missing trade(s)
        missing = []
        if not qengine_trade:
            missing.append('ml4t.backtest')
        if not backtrader_trade:
            missing.append('Backtrader')
        if not vectorbt_trade:
            missing.append('VectorBT')

        explanation = f"Missing trades from: {', '.join(missing)}"
        return False, 'missing', explanation

    # All frameworks have trades - compare values
    reference_trade = qengine_trade  # Use ml4t.backtest as reference

    # Check action
    if not all(t.action == reference_trade.action for _, t in trades):
        actions = {name: t.action for name, t in trades}
        return False, 'action', f"Action mismatch: {actions}"

    # Check quantity (allow small tolerance for rounding)
    quantities = [t.quantity for _, t in trades]
    max_qty = max(quantities)
    min_qty = min(quantities)
    qty_variance = ((max_qty - min_qty) / min_qty) if min_qty > 0 else 0

    if qty_variance > 0.001:  # 0.1% tolerance
        qtys = {name: t.quantity for name, t in trades}
        return False, 'quantity', f"Quantity mismatch: {qtys}"

    # Check price (allow 0.5% tolerance for slippage differences)
    prices = [t.price for _, t in trades]
    max_price = max(prices)
    min_price = min(prices)
    price_variance = ((max_price - min_price) / min_price) if min_price > 0 else 0

    if price_variance > 0.005:  # 0.5% tolerance
        prices_dict = {name: t.price for name, t in trades}
        return False, 'price', f"Price mismatch: {prices_dict}"

    # All match!
    return True, None, None


def reconcile_trades(
    signal_file: Path,
    config: FrameworkConfig | None = None,
) -> ReconciliationReport:
    """
    Perform complete trade reconciliation for a signal dataset.

    Returns:
        ReconciliationReport with detailed analysis
    """
    print(f"\n{'='*80}")
    print(f"Reconciling Trades: {signal_file.name}")
    print(f"{'='*80}")

    # Load signals
    with open(signal_file, 'rb') as f:
        signal_set = pickle.load(f)

    metadata = signal_set.get('metadata', {})
    signal_type = metadata.get('signal_type', 'unknown')

    # For multi-asset, use first asset (single-asset reconciliation for now)
    first_asset = list(signal_set['assets'].keys())[0]
    asset_data = signal_set['assets'][first_asset]
    data = asset_data['data']
    signals = asset_data['signals']

    print(f"Signal type: {signal_type}")
    print(f"Asset: {first_asset}")
    print(f"Total entry signals: {signals['entry'].sum()}")
    print(f"Total exit signals: {signals['exit'].sum()}")

    # Use default config if not provided
    if config is None:
        config = FrameworkConfig.realistic()

    # Run all frameworks
    print(f"\nRunning frameworks...")

    adapters = {
        'ml4t.backtest': BacktestAdapter(),
        'Backtrader': BacktraderAdapter(),
        'VectorBT': VectorBTAdapter(),
    }

    results = {}
    for name, adapter in adapters.items():
        print(f"  Running {name}...")
        result = adapter.run_with_signals(data, signals, config)
        results[name] = result
        print(f"    {result.num_trades} trades")

    # Extract trades
    qengine_trades = extract_trades_from_framework_result(results['ml4t.backtest'], 'ml4t.backtest')
    backtrader_trades = extract_trades_from_framework_result(results['Backtrader'], 'Backtrader')
    vectorbt_trades = extract_trades_from_framework_result(results['VectorBT'], 'VectorBT')

    print(f"\nTrade extraction:")
    print(f"  ml4t.backtest: {len(qengine_trades)} trades")
    print(f"  Backtrader: {len(backtrader_trades)} trades")
    print(f"  VectorBT: {len(vectorbt_trades)} trades")

    # Align trades with signals
    qengine_map = align_trades_with_signals(signals, qengine_trades, 'ml4t.backtest')
    backtrader_map = align_trades_with_signals(signals, backtrader_trades, 'Backtrader')
    vectorbt_map = align_trades_with_signals(signals, vectorbt_trades, 'VectorBT')

    # Create trade comparisons
    all_signal_indices = set(qengine_map.keys()) | set(backtrader_map.keys()) | set(vectorbt_map.keys())

    trade_comparisons = []
    perfect_matches = 0
    divergences = 0
    first_divergence_index = None

    for sig_idx in sorted(all_signal_indices):
        # Determine signal type
        if sig_idx < len(signals) and signals['entry'].iloc[sig_idx]:
            sig_type = 'entry'
        elif sig_idx < len(signals) and signals['exit'].iloc[sig_idx]:
            sig_type = 'exit'
        else:
            sig_type = 'unknown'

        sig_timestamp = signals.index[sig_idx] if sig_idx < len(signals) else None

        qengine_trade = qengine_map.get(sig_idx)
        backtrader_trade = backtrader_map.get(sig_idx)
        vectorbt_trade = vectorbt_map.get(sig_idx)

        # Compare
        all_match, divergence_type, explanation = compare_trades(
            qengine_trade,
            backtrader_trade,
            vectorbt_trade,
        )

        comparison = TradeComparison(
            signal_timestamp=sig_timestamp,
            signal_type=sig_type,
            signal_index=sig_idx,
            qengine_trade=qengine_trade,
            backtrader_trade=backtrader_trade,
            vectorbt_trade=vectorbt_trade,
            all_match=all_match,
            divergence_type=divergence_type,
            explanation=explanation,
            source_code_citation=None,  # To be filled in during investigation
        )

        trade_comparisons.append(comparison)

        if all_match:
            perfect_matches += 1
        else:
            divergences += 1
            if first_divergence_index is None:
                first_divergence_index = sig_idx

    # Identify required configuration options
    required_config_options = []

    # Check if we need partial fill configuration
    if any(c.divergence_type == 'missing' for c in trade_comparisons):
        required_config_options.append(
            "allow_partial_fills: bool - Some frameworks reject orders entirely when "
            "insufficient cash, others reduce order size to affordable amount"
        )

    # Check if we need fill timing configuration
    if any(c.divergence_type == 'price' for c in trade_comparisons):
        required_config_options.append(
            "fill_timing: 'same_close' | 'next_open' | 'next_close' - Frameworks differ "
            "on when orders execute relative to signal bar"
        )

    # Create report
    report = ReconciliationReport(
        signal_file=signal_file.name,
        signal_type=signal_type,
        total_signals=signals['entry'].sum() + signals['exit'].sum(),
        qengine_trades=len(qengine_trades),
        backtrader_trades=len(backtrader_trades),
        vectorbt_trades=len(vectorbt_trades),
        perfect_matches=perfect_matches,
        divergences=divergences,
        first_divergence_index=first_divergence_index,
        trade_comparisons=trade_comparisons,
        required_config_options=required_config_options,
    )

    return report


def test_reconcile_sma_crossover():
    """Reconcile SMA crossover signals."""
    signal_file = SIGNAL_DIR / "sp500_top10_sma_crossover.pkl"
    if not signal_file.exists():
        print(f"⚠️ Signal file not found: {signal_file}")
        return None

    report = reconcile_trades(signal_file)
    report.print_summary()
    report.print_detailed_comparison(max_trades=10)
    report.print_configuration_insights()

    return report


def test_reconcile_random():
    """Reconcile random signals."""
    signal_file = SIGNAL_DIR / "sp500_top10_random_5pct.pkl"
    if not signal_file.exists():
        print(f"⚠️ Signal file not found: {signal_file}")
        return None

    report = reconcile_trades(signal_file)
    report.print_summary()
    report.print_detailed_comparison(max_trades=10)
    report.print_configuration_insights()

    return report


def test_reconcile_rebalancing():
    """Reconcile rebalancing signals."""
    signal_file = SIGNAL_DIR / "sp500_top10_rebal_momentum_top5_weekly.pkl"
    if not signal_file.exists():
        print(f"⚠️ Signal file not found: {signal_file}")
        return None

    report = reconcile_trades(signal_file)
    report.print_summary()
    report.print_detailed_comparison(max_trades=10)
    report.print_configuration_insights()

    return report


def run_all_reconciliations():
    """Run all trade reconciliations."""
    print("="*80)
    print("TRADE RECONCILIATION SUITE")
    print("="*80)
    print()
    print("Purpose:")
    print("  1. Validate ml4t.backtest correctness (100% trade matching)")
    print("  2. Identify required configuration options (transparency)")
    print()

    reports = []

    # Test 1: SMA Crossover
    print("\n" + "="*80)
    print("TEST 1: SMA Crossover Reconciliation")
    print("="*80)
    report = test_reconcile_sma_crossover()
    if report:
        reports.append(('SMA Crossover', report))

    # Test 2: Random
    print("\n" + "="*80)
    print("TEST 2: Random Signals Reconciliation")
    print("="*80)
    report = test_reconcile_random()
    if report:
        reports.append(('Random', report))

    # Test 3: Rebalancing
    print("\n" + "="*80)
    print("TEST 3: Rebalancing Reconciliation")
    print("="*80)
    report = test_reconcile_rebalancing()
    if report:
        reports.append(('Rebalancing', report))

    # Final summary
    print("\n" + "="*80)
    print("RECONCILIATION SUMMARY")
    print("="*80)

    all_config_options = set()
    for test_name, report in reports:
        print(f"\n{test_name}:")
        if report.perfect_matches == len(report.trade_comparisons):
            print(f"  ✓ PERFECT MATCH - All {len(report.trade_comparisons)} trades reconciled")
        else:
            print(f"  ✗ {report.divergences} divergences found")
            print(f"    First divergence at signal #{report.first_divergence_index}")
            all_config_options.update(report.required_config_options)

    print("\n" + "="*80)
    print("CONFIGURATION API REQUIREMENTS")
    print("="*80)
    print("\nBased on reconciliation, ml4t.backtest must expose:")
    print()
    for option in sorted(all_config_options):
        print(f"  • {option}")

    print("\n✅ Trade reconciliation complete!")


if __name__ == "__main__":
    run_all_reconciliations()
