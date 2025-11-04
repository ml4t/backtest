#!/usr/bin/env python3
"""
Main validation runner for cross-platform backtest validation.

Usage:
    python run_validation.py --strategy ma_cross --platforms qengine,vectorbt_pro
    python run_validation.py --strategy all --platforms all
"""
import argparse
from datetime import datetime
from pathlib import Path

from adapters import (
    BacktraderAdapter,
    QEngineAdapter,
    VectorBTFreeAdapter,
    VectorBTProAdapter,
    ZiplineAdapter,
)
from data import load_test_data
from signals import MACrossoverSignals, MeanReversionSignals, RandomSignals
from validators import TradeValidator


def get_available_platforms():
    """Get list of available platform adapters."""
    return {
        'qengine': QEngineAdapter,
        'vectorbt_pro': VectorBTProAdapter,
        'vectorbt_free': VectorBTFreeAdapter,
        'zipline': ZiplineAdapter,
        'backtrader': BacktraderAdapter,
    }


def get_available_strategies():
    """Get list of available signal generators."""
    return {
        'ma_cross': lambda: MACrossoverSignals(fast_period=10, slow_period=30, quantity=100),
        'ma_cross_sl': lambda: MACrossoverSignals(
            fast_period=10, slow_period=30, quantity=100,
            stop_loss_pct=0.05, take_profit_pct=0.10
        ),
        'mean_reversion': lambda: MeanReversionSignals(
            rsi_period=14, oversold=30, overbought=70,
            stop_loss_pct=0.05, take_profit_pct=0.10
        ),
        'random': lambda: RandomSignals(
            signal_frequency=0.05, seed=42,
            allow_stop_loss=True, allow_take_profit=True
        ),
    }


def run_validation(
    strategy_name: str,
    platforms: list[str],
    start_date: str = '2020-01-01',
    end_date: str = '2020-12-31',
    symbols: list[str] = None,
    initial_capital: float = 100_000,
    commission: float = 0.001,
):
    """Run cross-platform validation.

    Args:
        strategy_name: Signal strategy to test
        platforms: List of platform names to compare
        start_date: Backtest start date
        end_date: Backtest end date
        symbols: List of symbols (default: ['AAPL'])
        initial_capital: Starting capital
        commission: Commission rate (0.001 = 0.1%)
    """
    print("=" * 80)
    print("CROSS-PLATFORM BACKTEST VALIDATION")
    print("=" * 80)
    print(f"Strategy: {strategy_name}")
    print(f"Platforms: {', '.join(platforms)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {symbols or ['AAPL']}")
    print("=" * 80)
    print("")

    # Load data
    print("📊 Loading data...")
    symbols = symbols or ['AAPL']
    data = load_test_data(
        'daily_equities',
        start_date=start_date,
        end_date=end_date,
        symbols=symbols
    )
    print(f"   Loaded {len(data)} bars")

    # Generate signals
    print(f"📈 Generating {strategy_name} signals...")
    strategies_map = get_available_strategies()
    signal_generator = strategies_map[strategy_name]()
    signals = signal_generator.generate_signals(data)
    print(f"   Generated {len(signals)} signals")

    # Run backtests on each platform
    results = {}
    available_platforms = get_available_platforms()

    for platform_name in platforms:
        print(f"\n🔧 Running on {platform_name}...")

        try:
            # Create adapter
            adapter_class = available_platforms[platform_name]
            adapter = adapter_class()

            # Prepare data in platform-specific format
            if platform_name == 'qengine':
                platform_data = data
            else:
                # Most platforms need pandas
                platform_data = data.to_pandas()

                # Some need specific index
                if 'timestamp' in platform_data.columns:
                    platform_data = platform_data.set_index('timestamp')

            # Run backtest
            result = adapter.run_backtest(
                signals=signals,
                data=platform_data,
                initial_capital=initial_capital,
                commission=commission,
            )

            results[platform_name] = result

            print(f"   ✓ Completed in {result.execution_time:.2f}s")
            print(f"   Trades: {len(result.get_closed_trades())}")
            print(f"   P&L: ${result.total_pnl():,.2f}")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Validate results
    if len(results) >= 2:
        print("\n🔍 Validating results...")
        validator = TradeValidator(tolerance_pct=0.1)
        report = validator.compare_results(results)

        print("\n" + report.summary)

        # Save report
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = Path('results') / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save HTML report
        html_path = output_dir / 'validation_report.html'
        validator.generate_html_report(report, str(html_path))
        print(f"\n📄 Report saved to: {html_path}")

        # Save raw results
        import json
        for platform_name, result in results.items():
            result_path = output_dir / f'{platform_name}_results.json'
            with open(result_path, 'w') as f:
                json.dump({
                    'platform': result.platform,
                    'trade_count': len(result.trades),
                    'total_pnl': result.total_pnl(),
                    'win_rate': result.win_rate(),
                    'metrics': result.metrics,
                    'config': result.config,
                    'execution_time': result.execution_time,
                }, f, indent=2, default=str)

        print(f"💾 Raw results saved to: {output_dir}/")

    else:
        print("\n⚠️  Need at least 2 platforms to compare")

    print("\n✅ Validation complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Cross-platform backtest validation')

    parser.add_argument(
        '--strategy',
        default='ma_cross',
        choices=['ma_cross', 'ma_cross_sl', 'mean_reversion', 'random', 'all'],
        help='Signal strategy to test'
    )

    parser.add_argument(
        '--platforms',
        default='qengine,vectorbt_pro',
        help='Comma-separated list of platforms (or "all")'
    )

    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Backtest start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        default='2020-12-31',
        help='Backtest end date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--symbols',
        default='AAPL',
        help='Comma-separated list of symbols'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100_000,
        help='Initial capital'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (0.001 = 0.1%%)'
    )

    args = parser.parse_args()

    # Parse platforms
    if args.platforms == 'all':
        platforms = list(get_available_platforms().keys())
    else:
        platforms = [p.strip() for p in args.platforms.split(',')]

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]

    # Run validation
    if args.strategy == 'all':
        for strategy_name in ['ma_cross', 'ma_cross_sl', 'mean_reversion', 'random']:
            print(f"\n\n{'='*80}")
            print(f"TESTING STRATEGY: {strategy_name}")
            print(f"{'='*80}\n")

            run_validation(
                strategy_name=strategy_name,
                platforms=platforms,
                start_date=args.start_date,
                end_date=args.end_date,
                symbols=symbols,
                initial_capital=args.capital,
                commission=args.commission,
            )
    else:
        run_validation(
            strategy_name=args.strategy,
            platforms=platforms,
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=symbols,
            initial_capital=args.capital,
            commission=args.commission,
        )


if __name__ == '__main__':
    main()
