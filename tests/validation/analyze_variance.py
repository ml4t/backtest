"""
Exact variance analysis - Compare trades across frameworks to find source of $1,143.72 difference.

Compares ml4t.backtest vs VectorBT vs Backtrader trade-by-trade.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from frameworks import BacktestAdapter, VectorBTAdapter, BacktraderAdapter
from frameworks.base import FrameworkConfig
from test_integrated_framework_alignment import generate_momentum_signals, generate_synthetic_multi_asset_data


def analyze_variance():
    """Run all 3 frameworks and compare trades exactly."""

    # Generate test data (25 stocks, 1 year)
    print("Generating test data...")
    test_data = generate_synthetic_multi_asset_data(
        num_symbols=25,
        num_days=252,
        start_date='2020-01-02'
    )

    # Generate signals
    signals = generate_momentum_signals(
        test_data,
        lookback=20,
        rotation_days=20,
        num_positions=5
    )

    config = FrameworkConfig(
        initial_capital=100000.0,
        commission_pct=0.001,
        slippage_pct=0.001,
        fill_timing="next_open",
        fractional_shares=True,
    )

    print(f"\n{'='*80}")
    print("RUNNING 3-WAY FRAMEWORK COMPARISON")
    print(f"{'='*80}")
    print(f"Signals: {len(signals)} total")
    print(f"Config: {config.commission_pct:.1%} commission, {config.slippage_pct:.1%} slippage\n")

    # Run all frameworks
    adapters = {
        "ml4t.backtest": BacktestAdapter(),
        "VectorBT": VectorBTAdapter(),
        "Backtrader": BacktraderAdapter(),
    }

    results = {}
    for name, adapter in adapters.items():
        print(f"Running {name}...")
        result = adapter.run_with_signals(test_data, signals, config)
        results[name] = result
        print(f"  Final Value: ${result.final_value:,.2f}")
        print(f"  Trades: {result.num_trades}")
        print(f"  Return: {result.total_return:.2f}%\n")

    # Analyze trade differences
    print(f"\n{'='*80}")
    print("TRADE-BY-TRADE ANALYSIS")
    print(f"{'='*80}\n")

    # Get trades from each framework
    qengine_trades = results["ml4t.backtest"].trades
    vectorbt_trades = results["VectorBT"].trades
    backtrader_trades = results["Backtrader"].trades

    print(f"Trade Counts:")
    print(f"  ml4t.backtest: {len(qengine_trades)} trades")
    print(f"  VectorBT: {len(vectorbt_trades)} trades")
    print(f"  Backtrader: {len(backtrader_trades)} trades")
    print(f"  Difference: ml4t.backtest has {len(qengine_trades) - len(vectorbt_trades)} more trades than VectorBT")
    print(f"  Difference: ml4t.backtest has {len(qengine_trades) - len(backtrader_trades)} more trades than Backtrader\n")

    # Compare first 20 trades from each framework
    print("First 20 Trades Comparison:")
    print(f"{'='*80}")

    for i in range(min(20, len(qengine_trades), len(vectorbt_trades), len(backtrader_trades))):
        q_trade = qengine_trades[i]
        v_trade = vectorbt_trades[i] if i < len(vectorbt_trades) else None
        b_trade = backtrader_trades[i] if i < len(backtrader_trades) else None

        print(f"\nTrade #{i+1}:")
        print(f"  ml4t.backtest:  {q_trade.timestamp.date()} {q_trade.action:4s} {q_trade.quantity:8.2f} @ ${q_trade.price:7.2f} = ${q_trade.value:10,.2f} (comm: ${q_trade.commission:.2f})")
        if v_trade:
            print(f"  VectorBT:       {v_trade.timestamp.date()} {v_trade.action:4s} {v_trade.quantity:8.2f} @ ${v_trade.price:7.2f} = ${v_trade.value:10,.2f} (comm: ${v_trade.commission:.2f})")
        if b_trade:
            print(f"  Backtrader:     {b_trade.timestamp.date()} {b_trade.action:4s} {b_trade.quantity:8.2f} @ ${b_trade.price:7.2f} = ${b_trade.value:10,.2f} (comm: ${b_trade.commission:.2f})")

    # Calculate total commission and slippage
    print(f"\n{'='*80}")
    print("COMMISSION & SLIPPAGE ANALYSIS")
    print(f"{'='*80}\n")

    qengine_total_commission = sum(t.commission for t in qengine_trades)
    vectorbt_total_commission = sum(t.commission for t in vectorbt_trades)
    backtrader_total_commission = sum(t.commission for t in backtrader_trades)

    print(f"Total Commissions Paid:")
    print(f"  ml4t.backtest: ${qengine_total_commission:,.2f}")
    print(f"  VectorBT: ${vectorbt_total_commission:,.2f}")
    print(f"  Backtrader: ${backtrader_total_commission:,.2f}\n")

    # Variance breakdown
    print(f"\n{'='*80}")
    print("VARIANCE BREAKDOWN")
    print(f"{'='*80}\n")

    qengine_value = results["ml4t.backtest"].final_value
    vectorbt_value = results["VectorBT"].final_value
    backtrader_value = results["Backtrader"].final_value

    variance_qe_vb = vectorbt_value - qengine_value
    variance_qe_bt = backtrader_value - qengine_value

    print(f"Final Values:")
    print(f"  ml4t.backtest: ${qengine_value:,.2f}")
    print(f"  VectorBT: ${vectorbt_value:,.2f}")
    print(f"  Backtrader: ${backtrader_value:,.2f}\n")

    print(f"Variance (vs ml4t.backtest):")
    print(f"  VectorBT: ${variance_qe_vb:+,.2f} ({variance_qe_vb/qengine_value*100:+.4f}%)")
    print(f"  Backtrader: ${variance_qe_bt:+,.2f} ({variance_qe_bt/qengine_value*100:+.4f}%)\n")

    print(f"Extra trades in ml4t.backtest: {len(qengine_trades) - len(vectorbt_trades)} more than VectorBT")
    print(f"Commission difference: ${qengine_total_commission - vectorbt_total_commission:,.2f}")
    print(f"\nHypothesis: ml4t.backtest rebalances positions more frequently,")
    print(f"            accumulating {len(qengine_trades) - len(vectorbt_trades)} extra trades = ${qengine_total_commission - vectorbt_total_commission:,.2f} in extra costs.")

    return results


if __name__ == "__main__":
    results = analyze_variance()
